# Mainstream LLM Zero-Shot Baseline Benchmark

Source thread: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684283
Notebook: `notebook/mainstream-llm-performance-comparison.ipynb` (pulled from `jiazhuang/nemotron-mainstream-llm-performance-comparison`)

---

## 1. Zero-shot frontier LLM ceiling on the competition dataset

### Context

A community member (jiazhuang) pre-collected responses from 10 frontier LLMs on the competition's training sample and published the results. The notebook analyses those cached responses to establish how well off-the-shelf models perform on the exact task before any fine-tuning. This serves as a practical ceiling reference when evaluating our own trained adapter.

### Investigation Checklist

- [x] Understand the methodology and scoring pipeline
- [x] Identify reusable answer-extraction and scoring logic
- [x] Identify any known failure modes in the extractor
- [x] Map results to our own training/evaluation strategy

### Findings

**Dataset structure.** The pre-collected JSONL (`train_sample_llm_result.jsonl`) has three columns per row: `prompt`, `answer` (ground truth), and `llm_output` (dict of model-name → raw response string). No live API calls are made in the notebook.

**Prompting method.** Identical to the official evaluation:
```
<problem prompt> + '\nPlease put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`'
```
All models received `max_tokens=32768`.

**Results (final, after LaTeX postprocessing fix):**

| Model | Think | Score |
|---|---|---|
| Gemini-3.1-Pro | ✅ | 0.81 |
| Claude-Opus-4.6 | ✅ | 0.78 |
| DeepSeek-V3.2 | ✅ | 0.74 |
| Kimi-K2.5 | ✅ | 0.72 |
| Qwen3-Max | ✅ | 0.72 |
| MiniMax-M2.5 | ✅ | 0.66 |
| Qwen3.5-Plus | ✅ | 0.64 |
| GLM-5 | ✅ | 0.52 |
| Claude-Sonnet-4.5 | ❌ | 0.51 |
| GPT-5.4 | ❌ | 0.36 |

**Thinking mode gap is decisive.** Every model with extended thinking enabled scores ≥0.52; non-thinking models cap at 0.51. The gap between the best thinking model (0.81) and best non-thinking model (0.51) is 0.30.

**Answer extraction pipeline.**
1. `extract_final_answer()` — tries `\boxed{\text{...}}` first, then `\boxed{...}`, then "Final answer:" patterns, then last number, then last line.
2. `latex_postprocess()` — strips `\\ ` escaped spaces and drops `\text{unit}` suffixes.
3. `verify()` — numeric answers use `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`; text answers use case-insensitive string match. API failures (None response) are scored as incorrect.

**Claude Opus LaTeX verbosity bug.** Without `latex_postprocess`, Opus scored substantially lower because it emits rigorous LaTeX inside `\boxed{}`:
- `\boxed{alice\ follows\ above\ garden}` — escaped spaces
- `\boxed{\text{student draws the curious key}}` — `\text{}` wrapper
- `\boxed{49.37 \text{ m}}` — appended unit in `\text{}`

These all fail naive string comparison against plain-text ground truth. The postprocessing step recovers the correct answer in all three cases.

### Actions Taken

- Pulled notebook to `notebook/mainstream-llm-performance-comparison.ipynb`.

### Resolution

**Status:** Resolved (information gathering complete).

The notebook establishes that the best achievable zero-shot score on this dataset is ~0.81, with thinking-enabled models occupying the top 8 positions. Our fine-tuned adapter competes in a different category (small model, specialised training) but this table is the relevant quality bar for data-generation oracle selection.

### Follow-ups

- **Use `extract_final_answer` + `latex_postprocess` in our eval pipeline.** The extractor in this notebook is more robust than a plain `\boxed{}` regex; the LaTeX postprocessing step is essential if we evaluate against Claude Opus-generated responses or generate training data with Opus. Copy the final versions from cells 24–25 of the notebook.
- **Data generator choice.** The scores confirm that Claude-Opus-4.6 (0.78) and DeepSeek-V3.2 (0.74) are strong oracle candidates for our NVIDIA API data generation plan, only 3–7 points behind the top model. Gemini-3.1-Pro at 0.81 is the best available oracle if API access is practical.
- **Non-thinking models are poor oracles.** GPT-5.4 at 0.36 and Sonnet-4.5 at 0.51 would produce lower-quality chain-of-thought training traces; avoid using non-thinking models for data generation.
- **Postprocess training targets.** If we generate SFT training data using Opus, run `latex_postprocess` on the extracted answers before storing them as training labels, or the model will learn to produce `\text{}`-wrapped answers that fail the official evaluator.
