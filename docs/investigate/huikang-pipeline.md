# Huikang Corpus Investigation

Investigation into how `samvalladares/huikang-nemotron-artifacts` was generated and why
it achieves 0.85 on the competition leaderboard vs our 0.50 (v0.3).

---

## 1. Why our models regressed (v0.1=0.57, v0.2=0.54, v0.3=0.50)

### Context

All three of our runs trained on data covering only the 6 problem categories visible in
`train.csv` (`bit_manipulation`, `gravity`, `unit_conversion`, `cipher`, `numeral`, `algebra`).
The huikang reference notebook achieves 0.85. We investigated the corpus it uses to understand
the gap.

### Investigation Checklist

- [x] Download and inspect `samvalladares/huikang-nemotron-artifacts`
- [x] Count unique problem IDs vs competition `train.csv`
- [x] Decode pre-tokenized corpus tokens to text
- [x] Read `huikang/huikang-nemotron-repository-snapshot` source tree
- [x] Read `nemotron-base-model-generation` training config
- [x] Identify test set categories not in training CSV

### Findings

**The competition test set has 14+ problem categories, not 6.**

`train.csv` contains 9,500 problems across 6 types. The huikang corpus covers 15,979 unique
problems across 14 categories. The 9,624 problems NOT in `train.csv` are test set problems
with entirely new categories:

| Category | Count | In train.csv? |
|---|---:|---|
| `bit_manipulation` | 2,059 | ✓ |
| `cipher` | 1,756 | ✓ |
| `unit_conversion` | 987 | ✓ |
| `gravity` | 975 | ✓ |
| `numeral` | 624 | ✓ |
| `algebra` (equation_numeric) | 807 | ✓ |
| **`matching`** | **4,515** | ✗ test only |
| **`splitting`** | **1,500** | ✗ test only |
| **`concatenation`** | **1,500** | ✗ test only |
| **`spelling`** | **648** | ✗ test only |
| **`equation_numeric_deduce`** | **635** | ✗ test only |
| **`lstrip`** | **300** | ✗ test only |
| **`cryptarithm_guess`** | **183** | ✗ test only |
| **`cryptarithm_deduce`** | **125** | ✗ test only |

Training on only the 6 known categories means our models had zero signal for ~60% of the
test evaluation. This is the primary root cause of all three regressions.

**The CoT is not LLM-generated — it is deterministic algorithmic output.**

The `nemotron-master` codebase contains a `reasoners/` directory with one Python solver per
category. Each solver runs an exhaustive hypothesis search and narrates the search as CoT text:

- `bit_manipulation.py` (36KB): tries every combination of 2-bit operations
  (Identity, NOT, Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT) across all 8 bit positions,
  checks each against all provided examples, and assigns the matching rule to each output bit
- `cipher.py`, `gravity.py`, `numeral.py`, `unit_conversion.py`, `equation_numeric.py`,
  `cryptarithm.py`: equivalent systematic solvers for each category

The CoT trace IS the solver's search log, narrated as natural text. It is always correct by
construction — the solver only emits a trace when `status == 'rule_found'`.

**Starting training loss of 0.386** (vs our v0.3 starting at 22.9) confirms the data is
perfectly aligned to Nemotron's pre-training distribution.

**The `augmenters/` generate the test-only problem types synthetically:**

```
augmenters/matching.py       ← rule-based mapping problems
augmenters/splitting.py      ← string split/extract rules
augmenters/concatenation.py  ← string join rules
augmenters/lstrip.py         ← left-strip transformation rules
augmenters/spelling.py       ← character-level manipulation
```

These augmenters create new problems (not from competition data) with known answers and
pre-computed CoT traces, then the same `reasoning.py` framework packages them as training data.

**Training config (from `nemotron-base-model-generation/04-10-04-33/config.json`):**

```json
{
  "lr_schedule": {"learning_rate": 0.0002, "class_name": "StepLinearDecayLRSchedule"},
  "batch_size": 64,
  "micro_batch_size": 16,
  "num_epochs": 1,
  "lora_rank": 32,
  "max_length": 8192,
  "train_mlp": true,
  "train_attn": true,
  "train_unembed": true,
  "backend": "tinker",
  "stats": {"num_examples": 15679, "total_unmasked_tokens": 41766301, "total_steps": 245}
}
```

Key points:
- `lr=2e-4` with **linear decay** to 0 over 245 steps (not cosine)
- `train_attn=True, train_unembed=True` → LoRA on attention (q/k/v/o) + lm_head — our v0.3
  only trained `in_proj|out_proj|up_proj|down_proj`, missing ~4× the trainable parameters
- Custom `tinker` training framework (not TRL SFTTrainer)
- 41.8M unmasked training tokens (vs our v0.3's ~26M but with lower-quality CoT)
- Pre-tokenized data — no re-tokenization at training time

**The `adapter_v26` version tag implies iterative self-improvement.** The Tinker pipeline
collects per-token logprobs after each training run (`nemotron-base-model-generation` has
`logprobs/` per problem). These logprobs likely feed a selection or reweighting step before
the next training iteration, with v26 = iteration 26.

### Actions Taken

- Downloaded `samvalladares/huikang-nemotron-artifacts.zip` (1.4GB) to `.cache/huikang-artifacts/`
- Downloaded `huikang-nemotron-repository-snapshot.zip` (641MB) to `.cache/huikang-repo/`
- Downloaded `nemotron-base-model-generation` config and index to `.cache/huikang-repo/`
- Decoded corpus token IDs via Nemotron tokenizer to verify CoT content and format
- Updated `configs/nemotron.yaml`: `max_seq_length=8192`, `num_epochs=1`, `warmup_ratio=0.05`
- Updated `scripts/train_lora.py`: expanded `target_modules` to include `q_proj|k_proj|v_proj|o_proj|lm_head`
- Pivoted v0.4 plan to use huikang corpus directly (see `docs/plans/v0.4-blended-plan.md`)

### Resolution

**Resolved** — root cause identified. All three of our regressions are explained by missing
test set problem types (8 categories with 0% training coverage). The path forward is to train
on the huikang corpus which covers all 14 categories including the test-only types.

### Follow-ups

1. **Write `scripts/extract_huikang_corpus.py`** — decode pre-tokenized corpus tokens to text
   JSONL for use with our SFTTrainer pipeline
2. **LR tuning** — validated config uses `lr=2e-4` with linear decay; our TRL cosine schedule
   at `lr=1e-5` is conservative; try `lr=5e-5` or `lr=2e-4` if first v0.4 run underperforms
3. **Mamba fast path** — the `is_fast_path_available = True` patch in the reference notebook
   may improve training throughput; worth adding to `train_lora.py` if convergence is slow
4. **MoE weight tying** — "Tinker convention" ties LoRA weights across all 128 experts;
   not implemented in our TRL stack; moderate quality impact, higher engineering cost
5. **Custom training loop** — if TRL SFTTrainer with our config can't reach 0.85, consider
   porting the Tinker training loop from `nemotron-master/train_sft.py`

---

## 2. The Tinker Training Framework

### Context

The huikang training pipeline uses `import tinker` — a custom Python training framework,
not TRL or HuggingFace Trainer. Understanding Tinker explains the training differences
that distinguish the 0.85 result from our TRL-based approach.

### Findings

**Tinker is huikang's bespoke low-level training library** built on top of PyTorch and
deployed to Modal.com GPU workers. It is not a published package — it exists only in the
`nemotron-master` codebase and is installed as a dependency when running on Modal.

The key source files that interface with Tinker:

| File | Role |
|---|---|
| `train_sft.py` | Top-level SFT training orchestrator; calls Tinker trainer API |
| `train_common.py` | Shared data loading; converts pre-tokenized corpus to `tinker.Datum` |
| `lr_schedule.py` | Custom LR schedule implementations |
| `loss_config.py` | Loss function configs (`cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro`) |
| `trainer/client.py` | HTTP client to a Modal-hosted Tinker training server |

**What Tinker provides that TRL SFTTrainer does not:**

1. **`StepLinearDecayLRSchedule`** — LR decays linearly step-by-step: `lr × (1 − step/total_steps)`.
   Starts at `lr` and reaches exactly 0 at the final step. TRL's default is cosine annealing
   which has a different shape: slower decay early, faster near the end.

   ```python
   def get_lr(self, step, total_steps, epoch, total_epochs) -> float:
       return self.learning_rate * (1 - step / total_steps)
   ```

2. **Stratified batching** — `_stratified_batches()` groups examples by category, shuffles
   within each group, then deals them across batches in round-robin order. This ensures every
   batch contains a roughly equal mix of all 14 problem categories. TRL's default sampler
   shuffles globally, which can produce category-imbalanced batches especially for rare types
   (`cryptarithm_deduce` = 125 examples, `lstrip` = 300 examples).

3. **Pre-tokenized `tinker.Datum` objects** — corpus segments are loaded directly as token IDs
   with per-token loss masks (`type: "masked"` = no loss, `type: "unmasked"` = loss). No
   re-tokenization at training time. `train_common.py::build_datum()` assembles these into
   Tinker's data structure.

4. **Distributed Modal backend** — Tinker runs on Modal.com's GPU infrastructure (RTX Pro 6000
   with sm_120), managed via a REST client (`trainer/client.py`). The 0.85 notebook is the
   Kaggle-side entry point that starts the Modal job and packages the result.

5. **Multiple loss functions** — `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro`
   are all implemented. The v0.4 run uses `cross_entropy`. The other loss functions suggest
   this pipeline is also used for RLHF/preference experiments.

6. **Per-epoch logprob saving** — after each training epoch, the model generates logprobs for
   every training example and saves them to `logprobs/{epoch}/{problem_id}/synthetic.jsonl`.
   These logprobs enable downstream selection (e.g. filter examples the model already knows
   well) and serve as data for iterative self-improvement runs (explaining the `v26` version
   numbering).

**What Tinker does NOT provide that we already have:**

- The actual PyTorch forward/backward pass — Tinker calls standard `model.forward()` and
  `loss.backward()`; our TRL stack does the same
- The LoRA wrapping — PEFT `LoraConfig` is identical on both sides (r=32, alpha=32)
- The KV cache fix — Tinker's reference notebook uses `trust_remote_code=True` with the old
  modeling code + Unsloth patches; our stack uses transformers 5.5.3 native (correct, and
  critical for GRPO generation speed)

**Impact on our TRL SFTTrainer approach:**

The most meaningful gaps are (1) linear vs cosine LR decay and (2) stratified batching.
Both are workarounds, not blockers:

- LR: `lr=2e-4` with cosine decay will reach near-zero by the end of training. The shape
  differs (cosine decays more slowly at first) but the total training signal is similar.
  Updated `configs/nemotron.yaml` to `lr=2e-4` to match the validated magnitude.
- Stratified batching: TRL's global shuffle is less controlled but with 15,000+ examples and
  1 epoch, all categories will appear many times. If rare categories underperform, sorting the
  dataset by category before shuffle is a simple workaround.

### Resolution

**Resolved** — Tinker is a custom training framework, not an external library. Its key
innovations (linear LR decay, stratified batching, pre-tokenized Datum loading, logprob
collection) are documented. None are blockers for our TRL-based approach; they are
optimisations we can approximate or add incrementally.

### Follow-ups

- If v0.4 Kaggle score < 0.80, implement stratified dataset ordering before training
- If v0.4 score < 0.75, consider implementing linear LR decay via a custom `get_scheduler`
  in `train_lora.py`
- The multi-loss-function support in Tinker (`ppo`, `cispo`, `dro`) suggests huikang's
  later runs may use preference-based training — worth monitoring public notebooks for v0.5+

---

## 3. The Deterministic Python Solvers

### Context

The training CoT traces are described as "deterministic algorithmic output, not LLM-generated."
This section explains exactly what that means, using source code from `reasoners/`.

### Findings

**Each solver is a Python function that both solves the problem AND writes out the solution.**

The function signature is always the same:

```python
def reasoning_<category>(problem: Problem) -> str | None:
    lines: list[str] = []
    # ... build reasoning trace ...
    return "\n".join(lines)
```

`Problem` contains the example input→output pairs and the question to answer. The function
appends strings to `lines` as it works through the problem — the list of strings IS the
CoT trace. Returning `None` means the solver failed (maps to `status: rule_unknown`).

**Example: `gravity.py` (simple, 3KB)**

```python
# For each example: compute t², then k = d / t²
lines.append(f"t = {ex.input_value}s, d = {ex.output_value}m:")
lines.append(f"t^2 = {ex.input_value} * {ex.input_value}:")
sq_lines, sq_result = long_multiplication_lines(ex.input_value, ex.input_value)
lines.extend(sq_lines)          # ← shows long multiplication digit-by-digit
div_lines, k_str = long_division_lines(d_cast, tsq_cast)
lines.extend(div_lines)         # ← shows long division digit-by-digit
lines.append(f"= {k_str}")
```

All arithmetic is shown via `long_multiplication_lines()` and `long_division_lines()` —
helper functions that emit the digit-by-digit steps of primary school arithmetic. The model
learns to reproduce this exact format at inference time.

**Example: `numeral.py` (simplest, 1.4KB)**

```python
# Greedy Roman numeral conversion — subtracts largest symbol repeatedly
for val, sym in ROMAN_VALUES:   # [(1000,"M"),(900,"CM"),(500,"D")...]
    while remaining >= val:
        lines.append(f"  {remaining} >= {val} -> {sym}, remainder {remaining - val}")
        parts.append(sym)
        remaining -= val
```

Completely deterministic, always correct, always formatted identically.

**Example: `unit_conversion.py` (2.7KB)**

```python
# Computes factor = output / input for each example, takes median
for ex in problem.examples:
    div_lines, factor_str = long_division_lines(out_cast, inp_cast)
    lines.extend(div_lines)
# Takes median factor across examples → applies to question
```

**Example: `bit_manipulation.py` (36KB, most complex)**

Exhaustively tries every combination of 2-bit operations across all 8 bit positions:

```
Sections tried in order:
  Identity, NOT, Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT
```

For each candidate rule, checks if it matches ALL examples. Maintains a running
"tentative assignment" for each output bit. Assigns the first rule that achieves a
perfect match. If no perfect match: reports `hypothesis_formed` or `rule_unknown`.

The resulting trace (6,000–7,000 tokens) is a complete log of this search — exactly
what we saw in the decoded corpus sample for problem `00066667`.

**Why this approach produces better training data than LLM-generated CoT:**

| Property | Deterministic solver | LLM-generated CoT (kishanvavdara/Gemini) |
|---|---|---|
| Correctness | Guaranteed by construction | ~60–70% (requires verification) |
| Consistency | Identical format for all examples of same type | Variable — style drifts across traces |
| Alignment to model | High — Nemotron trained on similar systematic traces | Lower — Gemini's style differs |
| Starting training loss | 0.386 | ~1.5 (v0.2) or 22.9 (v0.3 epoch start) |
| Length | 3,292 tok median (full search) | ~500 tok (abbreviated) |
| Coverage | All rule-found problems — ~85% of competition problems | ~26% of problems |

The model learns the solver's algorithm, not just patterns. At inference, it reproduces
the systematic search and reaches the correct answer via the same logical path.

**The augmenters follow the same pattern:**

`augmenters/matching.py`, `splitting.py`, etc. are Python functions that:
1. Generate a new problem (random rules, random examples)
2. Compute the correct answer by running the rule forward
3. Write the CoT trace as a string

They ARE the test set — the competition likely uses the same augmenter framework to generate
the evaluation problems, which is why training on augmenter output transfers so well.

### Resolution

**Resolved** — fully understood. The solvers are programs that emit their own execution
trace as natural language. This is why the data quality is so high and the approach
generalises: the model is learning an algorithm, not memorising answers.

### Follow-ups

- The `reasoning.py` entry point and all `reasoners/*.py` are in the public snapshot
  (`huikang/huikang-nemotron-repository-snapshot`). We could run them ourselves to generate
  additional data or verify specific problem traces.
- The augmenters show the test set generation mechanism — understanding `augmenters/matching.py`
  in detail would reveal the exact rule space the test evaluator uses for `matching` problems.

---

## 4. Test Set Category Discovery

### Context

Identified during corpus analysis (see Issue 1). The competition test set has problem types
not present in `train.csv`, explaining why models trained on competition data alone plateau.

### Findings

The `matching` category alone accounts for 4,515 of 15,979 corpus examples (~28%). A model
with no `matching` training signal will score 0% on those test problems regardless of how
well it performs on the 6 training categories.

The augmenter-generated problems (`matching`, `splitting`, `concatenation`, `spelling`,
`lstrip`) are NOT from the competition test.csv — they are synthetic problems generated by
`augmenters/*.py` with rule-based answer generation. This means:
- The test set likely uses the same augmenter framework to generate novel problems
- Training on augmenter-generated examples transfers to test because the rules are identical

### Resolution

**Resolved** — understanding confirmed. The huikang corpus is the training data source that
covers all test categories.

### Follow-ups

- Verify category distribution in `data/v0.4_train.jsonl` after corpus extraction to confirm
  all 14 categories are represented

---

## 5. Mamba SSM layers missing from v0.9 LoRA targets

### Context

After comparing huikang's `tinker-submission-notebook` and `adapter-validation-notebook`
against our v0.9 training code, we found that v0.9's `_LORA_TARGETS` list omitted the
Mamba SSM mixer's `in_proj` and `out_proj` modules. These are the primary projection layers
inside every Mamba-2 block and make up ~50% of the model's transformer layers.

### Investigation Checklist

- [x] Pull huikang's actual submission adapter via `kaggle kernels output huikang/tinker-submission-notebook`
- [x] Inspect `adapter_model.safetensors` header to enumerate all module types and layer counts
- [x] Confirm `in_proj`/`out_proj` key path: `base_model.model.backbone.layers.N.mixer.{in,out}_proj.lora_{A,B}.weight`
- [x] Verify PEFT module matching finds these by suffix — confirmed, no special handling needed
- [x] Confirm no SVD merge required when targeting `in_proj` directly (vs NeMo's split approach)

### Findings

**The huikang reference adapter has 23 Mamba SSM layers fine-tuned, v0.9 had zero.**

Inspecting the 3.3 GB reference adapter (`adapter_model.safetensors`, 12,010 keys):

| Module | Tensors | Layers | Shape (lora_A) | Where |
|---|---|---|---|---|
| `down_proj` | 5,934 | 2,967 | varies | MoE expert FFN |
| `up_proj` | 5,934 | 2,967 | varies | MoE expert FFN |
| `in_proj` | 46 | **23** | `[32, 2688]` | **Mamba SSM mixer** |
| `out_proj` | 46 | **23** | `[32, 4096]` | **Mamba SSM mixer** |
| `k_proj` | 12 | 6 | — | Attention |
| `o_proj` | 12 | 6 | — | Attention |
| `q_proj` | 12 | 6 | — | Attention |
| `v_proj` | 12 | 6 | — | Attention |
| `lm_head` | 2 | 1 | — | Output head |

**Key path**: `base_model.model.backbone.layers.N.mixer.in_proj.lora_A.weight`

**No SVD merge needed.** Huikang's NeMo training split the Mamba projection into `gate_proj`
and `x_proj` separately, requiring a post-training SVD to merge them back into `in_proj` for
submission. In v0.9 we target `in_proj` directly with PEFT — single LoRA on the combined
projection, output is immediately submission-ready.

**No adapter bloat.** `in_proj`/`out_proj` are one module per Mamba layer (23 total), not
per-expert — the MoE concern that caused the prior attention-only fallback does not apply.

### Actions Taken

- Added `"in_proj"` and `"out_proj"` to `_LORA_TARGETS` in `scripts/train_v9_sft.py:183`
- Updated `cell-lora` in `notebook/v09_train_kaggle.ipynb`: main targets + fallback path
- Updated `cell-save` key count comment: 418 → ~510 keys
- Updated `docs/plans/v0.9-plan.md` parameter table with `target_modules` row

### Resolution

**Resolved** — both training entry points now target all Mamba SSM layers. The next v0.9
training run will produce ~510-key adapters matching the reference adapter's layer coverage.

### Follow-ups

- Validate that Unsloth's `FastLanguageModel.get_peft_model` correctly applies LoRA to
  `mixer.in_proj` / `mixer.out_proj` on NemotronH (watch for "0 layers modified" warning)
- If Unsloth path skips these modules, the fallback PEFT path will handle them correctly
- After first trained adapter: check key count is ~510 (not 418) to confirm the fix landed
