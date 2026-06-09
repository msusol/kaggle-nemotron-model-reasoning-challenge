# ADR-0005 — Filter Adapter Keys to attn+shared_experts for Competition Submission

**Status:** Accepted

## Context

NemotronH is a hybrid Mamba-2 / MoE model. Its MLP layers contain two expert types:

1. **Routed (main) experts** — `mixer.experts.*`. Each token is dispatched to one of N experts via a router. Unsloth internally fuses all expert weight matrices into a single tensor for efficiency, saving LoRA weights as `mixer.experts.lora_A.weight` and `mixer.experts.lora_B.weight` (shape: `[n_experts × rank, hidden]`).

2. **Shared experts** — `mixer.shared_experts.*`. Every token always passes through these, unconditionally. They are standard `nn.Linear` layers with `up_proj` and `down_proj`, and PEFT applies LoRA to them in the normal way.

3. **Attention projections** — `q_proj`, `k_proj`, `v_proj`, `o_proj`. Standard linear layers handled identically by both Unsloth and standard PEFT.

The competition evaluator loads submitted adapters using **standard HuggingFace PEFT** (`PeftModel.from_pretrained`), not Unsloth. It has no knowledge of Unsloth's fused expert representation.

### Failure mode discovered

Early run1 submissions (v61) included Unsloth-format fused MoE keys (`mixer.experts.lora_A/B.weight`). The evaluator attempted to attach these LoRA matrices to a `mixer.experts` module — which does not exist as a standalone `nn.Linear` in the standard model — and raised a `TypeError`, causing **ERROR** status on all three initial submission attempts.

The error was doubly obscured: PEFT's internal `_get_peft_type()` catches **any** exception from `LoraConfig(**config_dict)` and re-raises it as `ValueError: "Can't find adapter_config.json"`, masking the real `TypeError` from Unsloth-specific fields (`use_qalora`, `target_parameters`, `qalora_group_size`) present in the saved config.

### Key counts observed

| Adapter | Keys | Breakdown | Evaluator result |
|---|---|---|---|
| run1 (initial) | 186 | 48 attn + 46 fused MoE + 92 shared_experts | ERROR |
| run1 (fix) | 48 | attn only | 0.54 (works, but loses shared_experts) |
| run2 onwards | 140 | 48 attn + 92 shared_experts | 0.54–0.56 (optimal) |

The 92 shared_experts keys arise because `up_proj` and `down_proj` are in `target_modules` and the shared expert layers are standard `nn.Linear` — PEFT attaches LoRA to them directly, producing valid evaluator-compatible weights.

## Decision

Before building `submission.zip`, verify the saved `adapter_model.safetensors` contains **no** fused MoE expert keys (keys matching `mixer.experts.lora_A` or `mixer.experts.lora_B` without the `shared_` prefix). Submit only:

- 48 attention keys: `*.{q,k,v,o}_proj.lora_{A,B}.weight`
- 92 shared expert keys: `*.mixer.shared_experts.{up,down}_proj.lora_{A,B}.weight`

Additionally, always rebuild `adapter_config.json` from scratch with only standard PEFT fields before submission, discarding Unsloth-specific fields and correcting `base_model_name_or_path` from the Kaggle-internal path to the canonical HuggingFace ID (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).

When warmstarting from a saved adapter, pass a clean `LoraConfig` explicitly to `PeftModel.from_pretrained(..., config=_warmstart_cfg)` to bypass PEFT's auto-loading path (which re-triggers the masking error on Unsloth-saved configs).

## Consequences

- All submissions from run2 onwards score correctly (no ERROR status).
- Routed expert LoRA weights are **not submitted** — the evaluator cannot use them. This caps the adapter's expressiveness to attention and shared expert layers only (~140 keys vs. the full ~418 Unsloth trains).
- Future runs with Unsloth training still benefit from the fused MoE expert training internally (better convergence), but that signal is discarded at submission time.
- A future option is to train without Unsloth (`FastLanguageModel.get_peft_model` disabled) so standard PEFT applies LoRA only to named modules — this naturally produces 140-key adapters with no post-processing needed.

## Architecture Diagram

```
NemotronH-30B — LoRA Layer Topology
─────────────────────────────────────────────────────────────────────

  Token Embeddings
        │
  ┌─────▼───────────────────────────────────────────────────────────┐
  │  Hybrid Layer Stack                                             │
  │                                                                 │
  │   ┌─────────────────────┐        ┌────────────────────────────┐ │
  │   │   Mamba-2 Layer     │        │   Attention + MoE Layer    │ │
  │   │                     │        │                            │ │
  │   │   SSM state-space   │        │  ┌── Attention ──────────┐ │ │
  │   │   (no LoRA targets) │        │  │  q_proj  ●── LoRA ✓  │ │ │
  │   │                     │        │  │  k_proj  ●── LoRA ✓  │ │ │
  │   └─────────────────────┘        │  │  v_proj  ●── LoRA ✓  │ │ │
  │         (alternating)            │  │  o_proj  ●── LoRA ✓  │ │ │
  │                                  │  └──────────────────────┘ │ │
  │                                  │                            │ │
  │                                  │  ┌── Shared Experts ─────┐ │ │
  │                                  │  │  (every token)        │ │ │
  │                                  │  │  up_proj   ●── LoRA ✓ │ │ │
  │                                  │  │  down_proj ●── LoRA ✓ │ │ │
  │                                  │  └──────────────────────┘ │ │
  │                                  │                            │ │
  │                                  │  ┌── Routed Experts ─────┐ │ │
  │                                  │  │  (1-of-N per token)   │ │ │
  │                                  │  │  [fused weight tensor] │ │ │
  │                                  │  │  experts.lora_A  ✗    │ │ │
  │                                  │  │  experts.lora_B  ✗    │ │ │
  │                                  │  │  (Unsloth internal,   │ │ │
  │                                  │  │   no std PEFT module) │ │ │
  │                                  │  └──────────────────────┘ │ │
  │                                  └────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────┘
        │
  LM Head (no LoRA)


  LoRA Key Count Breakdown (140 keys submitted to competition)
  ─────────────────────────────────────────────────────────────────

  Attention layers:
    q_proj   lora_A  +  lora_B  ┐
    k_proj   lora_A  +  lora_B  │  ×6 attention layers  =  48 keys  ✓
    v_proj   lora_A  +  lora_B  │
    o_proj   lora_A  +  lora_B  ┘

  MoE layers — shared experts:
    up_proj   lora_A  +  lora_B  ┐
                                  │  ×23 MoE layers       =  92 keys  ✓
    down_proj lora_A  +  lora_B  ┘

  MoE layers — routed experts (Unsloth fused, NOT submitted):
    mixer.experts.lora_A  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  46 keys  ✗
    mixer.experts.lora_B  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌         ✗

  Submitted:  48 + 92  =  140 keys
  Discarded:  46 fused MoE keys  (crash standard PEFT evaluator)
```

## Are These the Right Keys?

The 140 keys are known to be **evaluator-compatible** but not necessarily **optimal**.

### What is confirmed

**Empirically:** Run2 and run3 both submitted 140-key adapters and received scores (0.54, 0.56) with no ERROR. This confirms the evaluator loads them without crashing.

**Mechanically:** The evaluator uses standard `PeftModel.from_pretrained`. It attaches LoRA matrices by looking up each key's module path in `model.named_modules()`. All 140 keys map to real `nn.Linear` modules that exist in the standard HF NemotronH model — they load cleanly.

### What is uncertain

**Routed expert value:** The 46 discarded fused MoE keys represent real trained weights. Unsloth trains the routed expert layers for a reason — they likely contribute meaningfully to the model's capacity. We lose that signal at submission time purely due to format incompatibility. If the competition evaluator were Unsloth-aware, we'd submit ~186 keys and potentially score higher.

**`gate_proj` coverage:** The `target_modules` list in training includes `gate_proj`, but the saved adapter contains no `gate_proj` keys. This suggests `shared_experts` in NemotronH does not have a gate projection (or it is absorbed into the fused routed-expert tensor). Not yet verified against the base model's module tree.

### Definitive verification

To enumerate every eligible linear layer in the model and compare against submitted keys:

```python
for name, mod in model.named_modules():
    if any(t in name for t in ["shared_experts", "experts", "q_proj", "gate_proj"]):
        print(name, type(mod).__name__)
```

Running this against the base model locally would confirm whether any standard `nn.Linear` modules exist that are absent from the 140-key set. This has not yet been done — the empirical evidence (scores without ERROR) has been treated as sufficient for the current run strategy.

## Related

- ADR-0004 — gradient checkpointing bypass (separate NemotronH constraint)
- `notebook/v09_train_kaggle.ipynb` — cell-lora (warmstart path), cell-save, submission zip cell
