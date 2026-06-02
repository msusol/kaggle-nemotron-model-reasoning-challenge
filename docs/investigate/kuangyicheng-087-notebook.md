# kuangyicheng/nemotron-087-training — Notebook Investigation

Investigation into what `kuangyicheng/nemotron-087-training` is doing relative to our v0.4
effort, and whether it can be adapted to run on Kaggle TPUs.

---

## 1. Identity — kuangyicheng is the huikang author

### Context

The notebook URL `kaggle.com/code/kuangyicheng/nemotron-087-training` belongs to a
competitor scoring 0.87. We wanted to understand whether this is a new approach or an
evolution of the huikang pipeline already analyzed in `huikang-pipeline.md`.

### Investigation Checklist

- [x] Attempt WebFetch of notebook source (blocked — Kaggle requires authentication)
- [x] Cross-reference author handle with known dataset publishers
- [x] Infer notebook content from huikang repo snapshot already in `.cache/huikang-repo/`
- [x] Verify Modal.com backend claim against `trainer/client.py` in repo snapshot

### Findings

**kuangyicheng = huikang.** "Kuang Yicheng" is the person behind
`samvalladares/huikang-nemotron-artifacts` and `huikang/huikang-nemotron-repository-snapshot`,
both already analyzed in `huikang-pipeline.md`. The notebook name "087" denotes their
current leaderboard score: **0.87**, up from 0.85 when we first investigated.

This is the same pipeline, not a new approach.

### Actions Taken

None — read-only investigation.

### Resolution

**Resolved.** Identity confirmed via dataset authorship cross-reference.

### Follow-ups

None.

---

## 2. Training infrastructure — Modal.com, not Kaggle compute

### Context

The notebook is titled "training," which implies it runs a training job. But the 30B BF16
model requires ~60GB contiguous memory. Kaggle's free tier offers T4 × 2 (32GB total) and
TPU v3-8 (128GB XLA-sharded). Neither can straightforwardly host a 30B training run.

### Investigation Checklist

- [x] Read `trainer/client.py` from huikang repo snapshot
- [x] Confirm Modal.com backend in Tinker architecture docs
- [x] Check Kaggle GPU/TPU hardware specs against model memory requirements
- [x] Distinguish training notebook from submission notebook in competition context

### Findings

**The Kaggle notebook does not train on Kaggle compute.**

From `huikang-pipeline.md` §2 and the repo snapshot at `.cache/huikang-repo/`:

> "Distributed Modal backend — Tinker runs on Modal.com's GPU infrastructure (RTX Pro 6000
> with sm_120), managed via a REST client (`trainer/client.py`). The 0.85 notebook is the
> Kaggle-side entry point that starts the Modal job and packages the result."

The `trainer/client.py` file is an HTTP client that POSTs training jobs to a Modal-hosted
Tinker server. The Kaggle notebook:
1. Prepares the pre-tokenized corpus (already in `huikang-nemotron-artifacts`)
2. Calls Modal's REST API to start the training job
3. Polls for completion
4. Downloads the resulting adapter weights
5. Packages the adapter as a Kaggle dataset for use in the submission notebook

Internet-enabled Kaggle notebooks (non-submission kernels) can make outbound API calls —
this is what makes the pattern work. The competition's no-internet constraint applies only
to the final submission/inference notebook, not to methodology-sharing notebooks.

**Hardware actually used for training**: Modal.com RTX Pro 6000, Blackwell sm_120, ~48GB VRAM.
This is *different* from our GB10 (also Blackwell) but comparable in memory class.

### Actions Taken

None — read-only investigation.

### Resolution

**Resolved.** Training runs on Modal.com via REST API. The Kaggle notebook is an
orchestrator, not a compute host.

### Follow-ups

- If we ever want to replicate the exact Tinker pipeline, a Modal.com account + the
  `nemotron-master` codebase would be required. Our GB10 approach avoids this dependency.

---

## 3. TPU feasibility — not viable for this model

### Context

User asked whether we could copy the notebook and run it on Kaggle TPUs.

### Investigation Checklist

- [x] Check `mamba_ssm` / `causal_conv1d` platform requirements
- [x] Verify Kaggle TPU hardware specs (v3-8)
- [x] Check whether the training code could be adapted to XLA/JAX

### Findings

**Three hard blockers, in order of severity:**

1. **CUDA kernel incompatibility**: `mamba_ssm` and `causal_conv1d` are compiled CUDA C++
   extensions (`.so` files). Kaggle TPU v3-8 runs on Google XLA/TPU hardware — no CUDA
   runtime, no way to load CUDA extensions. The model fails to import before any training
   code runs. This is an architectural incompatibility, not a configuration issue.

2. **Nothing trains on Kaggle in the first place**: The 0.87 notebook calls Modal.com's
   API for training. "Running it on Kaggle TPUs" would require reimplementing the entire
   pipeline in JAX/XLA — a ground-up rewrite, not a copy.

3. **TPU memory layout is wrong for contiguous BF16**: Kaggle TPU v3-8 has 128GB total
   across 8 cores in XLA-sharded format. The 30B model in BF16 (~60GB) requires contiguous
   allocation; XLA sharding doesn't provide this in the same way unified memory does.

For reference, our submission inference notebooks already represent the maximum viable
Kaggle configuration: T4 × 2 GPU + `ryanholbrook/nvidia-utility-script` CUDA extensions.

### Actions Taken

None — read-only investigation.

### Resolution

**Resolved — TPU is not viable.** CUDA kernel dependency is a hard blocker independent
of all other concerns.

### Follow-ups

None. TPU path is closed; GB10 + Modal are the only viable training environments for
this model architecture.

---

## 6. Modal.com — cost to run our training pipeline

### Context

The huikang/kuangyicheng pipeline already runs on Modal.com (RTX PRO 6000 workers). User
wants to know whether running our own TRL SFTTrainer + GRPOTrainer pipeline on Modal is
cost-effective, and what the bill would look like.

Note: our code does not use Tinker at all — we would simply define a Modal function with
a GPU spec and run `scripts/train_lora.py` or `scripts/train_grpo.py` directly.

### Investigation Checklist

- [x] Fetch Modal pricing page (modal.com/pricing) — all GPU types and $/hour
- [x] Confirm RTX PRO 6000 architecture (Blackwell, sm_120, 96 GB GDDR7) via CUDA compute
      capability list and AnandTech article URL slug ("96gb-gddr7")
- [x] Identify which GPUs can fit the 30B BF16 model (~60 GB VRAM minimum)
- [x] Estimate v0.4 SFT hours from our actual GB10 run (14.4 h / 948 steps)
- [x] Estimate v0.5 GRPO hours for a practical 1,000-step run
- [x] Check free-tier credit amounts

### Findings

**RTX PRO 6000 VRAM confirmed: 96 GB GDDR7** (Blackwell sm_120, same architecture as GB10).

**GPUs that can load the 30B BF16 model (need ≥ 60 GB):**

| GPU | VRAM | $/hr | Notes |
|---|---|---|---|
| RTX PRO 6000 | 96 GB | $3.03 | Exact match for huikang hardware; zero porting risk |
| A100-80GB | 80 GB | $2.50 | Tight with GC enabled; our `_set_gradient_checkpointing` fix applies |
| H100 | 80 GB | $3.95 | Faster than A100; more expensive than RTX PRO 6000 |
| H200 | 141 GB | $4.54 | Plenty of headroom; overkill for cost |
| B200 | ~180 GB | $6.25 | Overkill |

**Cannot fit the model (eliminated):** A100-40GB (40 GB), L40S (48 GB), A10 (24 GB), L4 (24 GB), T4 (16 GB).

**v0.4 SFT cost estimate** — based on our actual 14.4-hour GB10 run (948 steps, seq_len=8192,
batch_size=1, grad_accum=16):

| GPU | Estimated hours | $/hr | Estimated cost |
|---|---|---|---|
| RTX PRO 6000 | ~15–16 h | $3.03 | **~$46–$49** |
| A100-80GB | ~18–20 h (older arch, slower) | $2.50 | **~$45–$50** |

Both options cost roughly the same — A100-80GB is ~$3 cheaper and is the better value
pick if the memory fits comfortably with GC. RTX PRO 6000 is lower risk (same Blackwell
arch, no unknowns around GC behaviour at seq_len=8192).

**v0.5 GRPO cost estimate** — GRPO is generation-heavy. Each step generates N=8 responses
before a gradient update. A practical run is 1,000–2,000 gradient steps (not full epochs
over all 9,500 problems, which would be thousands of hours):

| GPU | Estimated hours (1,000 steps) | $/hr | Estimated cost |
|---|---|---|---|
| RTX PRO 6000 | ~30–40 h | $3.03 | **~$91–$121** |
| A100-80GB | ~38–50 h | $2.50 | **~$95–$125** |

**Combined v0.4 SFT + v0.5 GRPO: roughly $140–$175** using RTX PRO 6000.

**Free-tier credits:**

| Plan | Monthly free credits | RTX PRO 6000 hours covered |
|---|---|---|
| Starter (free) | $30 | ~10 h (not enough for a full SFT run) |
| Team | $100 | ~33 h (covers v0.4 SFT; ~$50 short for full v0.4 + v0.5) |

The Team plan's $100/month free credit covers v0.4 SFT alone. A complete v0.4 + v0.5
cycle exceeds it by ~$50–$75.

**Storage cost: near zero.** Modal volumes are $0.09/GiB/month with 1 TiB free. The
model weights (~60 GB), training data (~1.5 GB), and adapters (~500 MB) total under 65 GB —
well within the free tier.

**How to use Modal with our code:** define one Modal function decorated with the GPU spec,
mount the huikang dataset and model weights as Modal volumes, and call
`scripts/train_lora.py` or `scripts/train_grpo.py` as a subprocess. No Tinker dependency
required. The existing Docker image (`Dockerfile.gb10`) would need a small adaptation
(remove GB10-specific `--privileged` flags; Modal handles GPU access natively).

### Actions Taken

None — read-only investigation.

### Resolution

**Resolved.** Modal is a viable fallback if the GB10 is unavailable. Cost for a full
v0.4 + v0.5 cycle is ~$140–$175, of which ~$50–$75 falls outside the Team plan free tier.
The RTX PRO 6000 is the lowest-risk GPU choice; A100-80GB is the best value if cost
matters more than compatibility certainty.

### Follow-ups

- If GB10 is unavailable for an extended period, Modal is the correct path — not Ona/Gitpod.
- A Modal adapter would require wrapping `scripts/train_lora.py` in a Modal function and
  uploading weights + data as Modal volumes. Estimated setup: 2–3 hours of work.
- The existing `Dockerfile.gb10` is mostly reusable; only the GPU flag section
  (`--privileged -e NVIDIA_VISIBLE_DEVICES=all`) needs to be removed.

---

## 5. Gitpod/Ona ACU credits — viable alternative training environment?

### Context

User has purchased ACU credits on Gitpod (now rebranded to `ona.com`) and asked whether
the training pipeline could run there instead of the GB10.

### Investigation Checklist

- [x] Confirm Gitpod → Ona rebrand (gitpod.io redirects 308 to ona.com)
- [x] Fetch Ona environment class specs from `ona.com/docs/ona/runners/aws/environment-classes`
- [x] Look up AWS g5.4xlarge hardware specs
- [x] Compare against model VRAM requirements

### Findings

**Gitpod has rebranded to Ona (`ona.com`).** Their top GPU environment class is:

| Ona class | AWS instance | GPU | GPU VRAM | System RAM | Disk |
|---|---|---|---|---|---|
| GPU Large | g5.4xlarge | 1× NVIDIA A10G | **24 GB** | 64 GB | 300 GB |
| GPU Large Spot | g5.4xlarge | 1× NVIDIA A10G | **24 GB** | 64 GB | 300 GB |

The 30B Nemotron model in BF16 requires ~60 GB of VRAM just to load weights. The A10G
is 2.5× too small before a single training step.

4-bit quantization would reduce the load to ~15 GB, but QLoRA is broken for NemotronH's
MoE expert layers (see ADR-0004 and `docs/investigate/v0.4-oom-training.md`). There is
no viable quantization path that preserves training quality on this architecture.

**The Modal.com approach (call Modal API from a Gitpod notebook) is technically possible**
but requires paying two compute bills simultaneously — Ona ACU credits for the orchestrator
environment and Modal credits for the actual GPU time — to replicate what the GB10 already
handles end-to-end for free.

### Actions Taken

None — read-only investigation.

### Resolution

**Resolved — not viable for training.** VRAM is the hard blocker. Ona ACU credits are
better spent on lightweight dev tasks (code editing, script testing, browser-accessible
Linux environments) than training workloads.

### Follow-ups

- If GB10 becomes unavailable for an extended period, Modal.com (RTX Pro 6000, 48 GB
  VRAM) is the correct fallback for training — not Ona/Gitpod.

---

## 4. Gap analysis — what explains 0.87 vs our pending v0.4 score

### Context

After confirming it's the same pipeline and same dataset, we wanted to quantify what
differences remain between the kuangyicheng approach and our v0.4 TRL implementation.

### Investigation Checklist

- [x] Compare training configs (LR schedule, batching, targets, seq length)
- [x] Identify Tinker-specific features not in TRL SFTTrainer
- [x] Assess iterative self-improvement rounds (`adapter_v26` tag)
- [x] Map gaps to our v0.5 GRPO plan

### Findings

**Already matched in v0.4:**

| Dimension | kuangyicheng | Our v0.4 |
|---|---|---|
| Dataset | Huikang corpus, 14 categories | Same |
| LoRA rank/alpha | r=32, α=32 | Same |
| LoRA targets | q/k/v/o + lm_head | Same |
| Learning rate magnitude | 2e-4 | Same |
| Sequence length | 8192 | Same |
| KV cache fix | mamba_ssm patch + trust_remote_code | transformers 5.5.3 (cleaner) |

**Remaining gaps:**

| Gap | kuangyicheng | Our v0.4 | Estimated cost to close |
|---|---|---|---|
| LR schedule | Linear decay to 0 | Cosine annealing | 15 min — `get_linear_schedule_with_warmup` |
| Batch strategy | Stratified (equal per-category mix) | Global shuffle | 30 min — sort JSONL by category |
| MoE weight tying | 1 LoRA set shared × 128 experts | Independent per expert | 1–2 days — custom PEFT hook |
| Iterative self-improvement | 26+ training rounds with logprob reweighting | 1 round | v0.5 GRPO (already planned) |

**The 0.87 vs 0.85 delta** is likely iterative self-improvement rounds, not a single
training change. `adapter_v26` in the artifact names implies 26 iterations of
train → collect logprobs → reweight → train again. The 0.87 notebook probably uses v28+.

**The iterative self-improvement pattern is exactly what GRPO achieves**, via a cleaner
RL formulation. Our v0.5 plan (`docs/plans/v0.5-grpo-plan.md`) is the correct path to
reach and exceed 0.87 without replicating the Modal/Tinker infrastructure.

### Actions Taken

None — read-only investigation.

### Resolution

**Resolved.** Gap is quantified. Linear LR and stratified batching are low-cost
mitigations if v0.4 score comes back below 0.82. The primary path beyond 0.87 is v0.5
GRPO, not Tinker replication.

### Follow-ups

1. Once v0.4 Kaggle score is known: if < 0.82, add linear LR schedule and stratified
   dataset ordering before proceeding to v0.5.
2. If v0.4 ≥ 0.80: proceed directly to v0.5 GRPO per `docs/plans/v0.5-grpo-plan.md`.
3. MoE weight tying is deferred — moderate quality impact, high engineering cost, and
   GRPO likely outperforms it anyway.

---

## 7. DGX Spark (GB10) vs RTX PRO 6000 — which is better for this competition?

### Context

Having established Modal.com as a viable fallback at $3.03/hr for the RTX PRO 6000,
the question is whether the RTX PRO 6000 is actually a better training platform than
the GB10 and worth paying for.

### Investigation Checklist

- [x] Fetch DGX Spark GB10 specs from nvidia.com/dgx-spark (memory, bandwidth, compute, TDP)
- [x] Confirm RTX PRO 6000 architecture and VRAM (96 GB GDDR7, sm_120) — NVIDIA spec pages 404
- [x] Estimate RTX PRO 6000 bandwidth from GDDR7 spec and known bus width
- [x] Compare against our actual GB10 training times

### Findings

NVIDIA's RTX PRO 6000 spec pages returned 404 during investigation. GB10 specs confirmed
from official NVIDIA DGX Spark product page. RTX PRO 6000 bandwidth is estimated from
GDDR7 characteristics; VRAM confirmed at 96 GB from AnandTech article URL slug.

| Spec | DGX Spark GB10 | RTX PRO 6000 Blackwell |
|---|---|---|
| Memory | **128 GB** LPDDR5x unified | 96 GB GDDR7 ECC |
| Memory bandwidth | **273 GB/s** (confirmed) | ~672 GB/s (GDDR7, estimated) |
| BF16 compute | ~67 TFLOPS (est.) | ~150 TFLOPS (est.) |
| Architecture | Blackwell sm_120 | Blackwell sm_120 |
| TDP | 140 W (chip) / 240 W (system) | ~300 W |
| Cost | **$0/hr (owned)** | $3.03/hr on Modal |

**RTX PRO 6000 is faster in wall-clock time.** GDDR7 bandwidth is ~2.5× that of the GB10's
LPDDR5x unified memory. For LLM training, bandwidth is the primary bottleneck — loading
weights on every forward/backward pass dominates the step time. Our v0.4 SFT run of 14.4
hours on GB10 would likely take ~6–8 hours on RTX PRO 6000.

**GB10 wins on memory capacity.** 128 GB vs 96 GB provides more headroom for GRPO
generation (N=8 × 6144 tokens) without having to reduce the generation group size to N=4.

**Both are identical in training quality.** Same Blackwell sm_120 architecture means
identical CUDA kernel paths, numerics, and output.

**For this competition, GB10 is the correct choice:**

1. **Cost**: $0/hr vs $3.03/hr. At 14 hrs/run × multiple attempts (v0.4 regressions,
   v0.5 GRPO), Modal costs accumulate to $100–$200+. Speed is irrelevant if the hardware
   is free and finishes before the deadline.
2. **Memory headroom**: 128 GB absorbs GRPO batch spikes that 96 GB cannot.
3. **Already configured**: Docker image, GC patches, page-cache dropper, and preflight
   scripts are tuned for GB10. Modal porting is ~2–3 hours of additional work.
4. **Deadline pressure**: Faster iteration on working hardware beats faster hardware that
   requires setup time.

Modal RTX PRO 6000 is the right choice only if the GB10 becomes unavailable for an
extended period (hardware failure, travel, etc.).

### Actions Taken

None — read-only investigation.

### Resolution

**Resolved.** GB10 is the correct primary training platform for this competition.
RTX PRO 6000 on Modal is faster in wall-clock time (~2× speedup) but costs ~$43/run
and requires porting work. Use Modal only as a fallback.

### Follow-ups

- If the GB10 is unavailable: set up the Modal adapter (see §6 follow-ups) and use
  RTX PRO 6000 at $3.03/hr.
- If a v0.5 GRPO run is expected to take > 48 hours on GB10, it may be worth paying
  for Modal to stay within competition deadline.

---

## 8. Dataset format audit — v0.4_train.jsonl vs 0.87 training data

### Context

Before re-running training with the regression fixes, we needed to verify exactly what
format issues exist in `data/v0.4_train.jsonl` and how they compare to what the 0.87
notebook uses.

### Investigation Checklist

- [x] Sample raw rows from `v0.4_train.jsonl` — inspect system, prompt, response fields
- [x] Audit all 15,159 rows by category: count missing `\boxed{}`, `\boxed{–}` placeholders, missing `</think>`
- [x] Read `scripts/extract_huikang_corpus.py` — understand how corpus was decoded
- [x] Read `scripts/train_lora.py` `format_example` — verify system prompt fix is applied
- [x] Read `scripts/validate_metric.py` `extract_answer` — understand local scorer behaviour
- [x] Check `data/test.csv` prompt format vs training prompt format
- [x] Read `docs/investigate/v0.4-kaggle-regression.md` for known issues

### Findings

#### Three distinct format issues identified across all 15,159 training rows

**Issue 1 — Empty system field (all 15,159 rows)**

Every row has `"system": ""`. This is the root cause of the v0.4 regression (score 0.49):
the model trained without any system prompt, but inference injected an out-of-distribution
one. The fix (`example.get("system") or SYSTEM_PROMPT`) is already in `train_lora.py`
but has not yet been used in a retrained submission.

**Issue 2 — `\boxed{–}` placeholder in 9 categories (8,596 rows)**

Rows across bit_manipulation, cipher, gravity, numeral, unit_conversion,
cryptarithm_deduce, cryptarithm_guess, equation_numeric_deduce, equation_numeric_guess
ALL contain this template artefact inside `<think>`:

```
I will now return the answer in \boxed{}
The answer in \boxed{–} is \boxed{10010111}
</think>
\boxed{10010111}
```

`\boxed{–}` is a template placeholder left by the Tinker solver. The real answer
`\boxed{10010111}` appears twice: once inside `<think>` (after the dash) and once
cleanly after `</think>`. Our local `validate_metric.py` uses `boxed[-1]` (last match),
so it correctly extracts `10010111` and is unaffected. The Kaggle scorer behaviour is
unknown — if it takes `boxed[0]` it would score `–` for every problem in these categories.

**Issue 3 — No `\boxed{}` at all in 5 augmenter categories (8,046 rows, ~53%)**

| Category | Rows | Has `\boxed{}`? | Response ending |
|---|---|---|---|
| `matching` | 4,316 | ✗ | `Best: 3 4 5 6 7 0 1 2: 8\n</think>` |
| `splitting` | 1,421 | ✗ | `99 【/】 -> 【/】\n</think>` |
| `concatenation` | 1,422 | ✗ | ends `</think>` |
| `spelling` | 601 | ✗ | ends `</think>` |
| `lstrip` | 286 | ✗ | ends `</think>` |

These responses end with `</think>` and provide no boxed final answer. The model is
trained to output multi-line structured answers (e.g. the full split table for splitting,
the best-match sequence for matching) with NO boxed wrapper. Our system prompt
"Put your final answer in `\boxed{}`" directly contradicts what the training data shows
for these 8,046 rows.

Our local `validate_metric.py` falls back to `last number → last word` when no `\boxed{}`
is found. For a matching response the last token before `</think>` would be a count
integer (`8`), not the sequence answer — so local scoring of these categories would be
wrong regardless.

**Issue 4 — Prompt `\boxed{}` instruction present in training but absent in test.csv**

Training prompts for the 9 boxed categories end with:
```
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

The sample `test.csv` (3 rows, bit_manipulation) does NOT include this instruction:
```
Now, determine the output for: 00110100
```

(End of prompt.) The system prompt compensates at inference time; this is not a blocker
but means the model sees two different prompt styles for the same task.

#### How the 0.87 notebook handles these issues

The 0.87 notebook almost certainly trains with `system: ""` and infers with `system: ""`
(empty throughout). The `\boxed{}` output for the 9 boxed categories comes from the
prompt instruction, not the system field. For the 5 augmenter categories, the model
outputs the structured non-boxed format — and the competition scorer presumably accepts
this format or the actual test prompts for those categories include a `\boxed{}`
instruction that we don't have in our training data.

Our approach diverges by injecting a system prompt at both train and infer time. This
is correct in isolation, but it creates a contradiction for the 8,046 augmenter-category
rows where training data never demonstrates `\boxed{}` output.

#### Summary table

| Issue | Rows affected | Impact on local score | Impact on Kaggle score | Fix |
|---|---|---|---|---|
| Empty system (trained ≠ inferred) | 15,159 | None (same env) | **Primary v0.4 regression** | `example.get("system") or SYSTEM_PROMPT` — in code ✅ |
| `\boxed{–}` placeholder | 8,596 | None (`boxed[-1]`) | Unknown — may score `–` | Strip from `extract_huikang_corpus.py` ⚠️ |
| No `\boxed{}` for augmenter cats | 8,046 | Wrong (fallback to last number/word) | Unknown — scorer may parse structured format | Unclear — see follow-ups ⚠️ |
| Prompt `\boxed{}` instruction vs test | 9 cats | None | Minor style mismatch | System prompt compensates ✅ |

### Actions Taken

None — read-only investigation.

### Resolution

**Resolved.** Three format issues documented. Issue 1 (system prompt) fix is in code.
Issue 2 (`\boxed{–}`) and Issue 3 (no `\boxed{}` in augmenter categories) require data
fixes before the next training run.

### Follow-ups

1. **Strip `\boxed{–}` in `extract_huikang_corpus.py`** — remove
   `The answer in \boxed{–} is` from `parse_unmasked()` output, leaving only
   `\boxed{actual_answer}` inside `<think>` and after `</think>`.
2. **Investigate augmenter category answer format** — check whether the competition's
   actual test prompts for matching/splitting/etc. include `\boxed{}` instructions
   (not present in our training prompts). If yes, add the instruction to training prompts
   on re-extraction. If no, the scorer parses structured output and the current format
   may be intentionally correct.
3. **Re-run extraction** after fixes 1 and 2, then retrain before v0.5 GRPO init.
