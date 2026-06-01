# NemotronH Architecture Notes and Known Issues

## Architecture overview

Nemotron-3 Nano 30B (A3B) is a hybrid model — not a standard dense Transformer:

- **23 Mamba-2 SSM layers** — state-space sequence model, no attention matrix
- **6 full attention layers** — standard multi-head attention
- **128-expert MoE FFN** per layer, routing 5–6 experts per token
- Total: ~30B parameters, ~3B active per token (the "A3B" designation)

## What works (as of transformers 5.5.3)

`transformers>=5.5.3` ships native NemotronH support. No `trust_remote_code` needed.
PEFT LoRA training at BF16, seq_len=8192 on a single GB10 (130 GB unified memory)
**is proven working** — see v0.4b training run in `docs/plans/leaderboard.md`.

| Approach | Status | Notes |
|---|---|---|
| BF16 load + PEFT LoRA | ✅ Works | v0.4b confirmed, ~46 s/step at seq=8192 |
| `transformers>=5.5.3` native | ✅ Works | No `trust_remote_code` needed |
| NeMo framework SFT | ✅ Expected to work | Uses pre-tokenized JSONL, no runtime tokenization |
| 4-bit / 8-bit quantization | ❌ Broken | MoE expert tensors cause shape mismatches with BitsAndBytes |
| `trust_remote_code=True` | ❌ Avoid | Overrides the native KV-cache fix in transformers 5.5.3 |
| `target_modules="all-linear"` | ❌ Avoid | Wraps all 128 expert copies → billions of LoRA params, OOM |
| HF gradient_checkpointing_enable() | ❌ Not supported | NemotronHForCausalLM raises ValueError; use NeMo's selective GC instead |

## LoRA target modules

Confirmed layer names (via meta-device dump, 2026-05-31):

```
conv1d, down_proj, embeddings, gate, in_proj, k_proj, lm_head,
norm, norm_f, o_proj, out_proj, q_proj, up_proj, v_proj
```

**Use this regex in PEFT:**
```
.*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj)$
```

- `up_proj`/`down_proj` — MoE expert FFN layers (NOT `fc1`/`fc2` as some sources claim)
- `in_proj`/`out_proj` — Mamba SSM input/output projections
- `lm_head` — excluded: PEFT classifies as embedding layer, bloats adapter by ~4 MB
- `conv1d` — Mamba convolution, excluded (frozen in reference notebook)
- `gate` — MoE router, excluded (frozen in reference notebook)
- `embeddings` — input embeddings, excluded

## Memory budget (GB10, 130 GB unified)

| Phase | Memory |
|---|---|
| Base model (BF16) | ~60 GB |
| LoRA adapters | ~100 MB |
| Activations at seq=8192 (no GC) | ~20–40 GB |
| Peak during loading (mmap + CUDA) | ~110 GB |
| **Total peak** | **~100–110 GB** |

Loading OOM prevention: run with `vfs_cache_pressure=500` and keep `min_free_kbytes`
at default (~44 MB) — not the aggressive 40 GB value that caused loading kills.
See `docs/investigate/v0.4-kaggle-regression.md` and `docs/adr/` for full history.

## Known inference issue: `\boxed{–}` placeholder

The huikang corpus responses contain an intermediate `\boxed{–}` placeholder inside
`<think>` before the real answer. Always extract the **last** `\boxed{}` from model
output, not the first. See `scripts/validate_metric.py` (`boxed[-1]` pattern).
