# Competition Overview

## Objective

The Kaggle NVIDIA Nemotron Model Reasoning Challenge asks participants to improve reasoning accuracy while keeping the base model fixed to Nemotron-3-Nano-30B and submitting only a compatible LoRA adapter.[cite:1] The challenge is centered on open reasoning methods such as prompting, data curation, synthetic data, RL, and lightweight fine-tuning.[cite:1]

## Evaluation behavior

Submissions are evaluated by loading the Nemotron base model together with the submitted LoRA adapter using vLLM.[cite:1] The evaluation prompt asks the model to put its final answer inside `\\boxed{}`, and the metric first tries to extract that boxed value before falling back to other heuristics.[cite:1][cite:54]

## Timeline

| Date | Event |
|---|---|
| March 16, 2026 | Start date |
| April 9, 2026 | Midpoint cut-off date |
| **June 8, 2026** | Entry deadline + Team merger deadline — must have accepted rules |
| **June 15, 2026 23:59 UTC** | **Final submission deadline** |

## Hard constraints

- LoRA rank must be 32 or less.[cite:1]
- The adapter must include `adapter_config.json`.[cite:1][cite:54]
- vLLM generation uses temperature 0.0 and top_p 1.0.[cite:1]
- The scoring stack is designed around the submitted adapter, not a custom runtime like DSPy.[cite:1][cite:35]

## Development implication

DSPy can still be used heavily offline for data generation, prompt search, or evaluation workflows.[cite:35] The practical rule is that anything discovered with DSPy must be distilled into the LoRA adapter and prompt behavior that Kaggle's single-call evaluation will observe.[cite:1]
