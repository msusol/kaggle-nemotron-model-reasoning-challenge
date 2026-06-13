# Investigation: Nemotron-H Chat Template — Source of Truth

**Status**: resolved  
**Source file**: `.cache/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848/chat_template.jinja`

The same template is embedded in `tokenizer_config.json` (`chat_template` field). Both are authoritative — they are identical.

---

## Key template behaviors

### 1. `<think>` is only prepended during inference, not training

The critical lines (at end of template):

```jinja
{%- if add_generation_prompt %}
    {%- if enable_thinking %}
        {{- '<|im_start|>assistant\n<think>\n' }}
    {%- else %}
        {{- '<|im_start|>assistant\n<think></think>' }}
    {%- endif %}
{%- endif %}
```

`<think>\n` is emitted **only** when `add_generation_prompt=True` AND `enable_thinking=True`. Our training script calls `apply_chat_template(..., add_generation_prompt=False, enable_thinking=True)` — so this branch is **never reached during training**.

The v0.9 plan states *"The Nemotron-H chat template auto-prepends `<think>\n` at the assistant turn."* This is **imprecise**: the prepend happens at the generation prompt (inference), not when rendering historical assistant turns in the training sequence.

### 2. How the template handles assistant content during training

For the non-last-turn assistant message path (lines 109–119):

```jinja
{%- set content = message.content | default('', true) %}
{%- if content is string -%}
    {%- if '<think>' not in content and '</think>' not in content -%}
        {%- set content = "<think></think>" ~ content -%}
    {%- endif -%}
```

Three cases for our Format 4 content `{trace}\n</think>\n\boxed{answer}`:

| Content contains | Template action | Result |
|---|---|---|
| Neither `<think>` nor `</think>` | Prepends `<think></think>` | `<think></think>{content}` |
| `</think>` only (Format 4 traces) | **Left as-is** | `{trace}\n</think>\n\boxed{answer}` |
| Both `<think>` and `</think>` | Left as-is | `<think>{trace}</think>\n\boxed{answer}` |

**Format 4 traces** (from huikang/kishanvavdara, no `<think>` opener) hit the second case. The rendered training token sequence is:

```
<|im_start|>assistant
{reasoning trace}
</think>
\boxed{answer}<|im_end|>
```

### 3. Train/inference context mismatch

| Context | Token sequence before model generates |
|---|---|
| **Training** | `<\|im_start\|>assistant\n{reasoning trace}` (model predicts next tokens) |
| **Inference** | `<\|im_start\|>assistant\n<think>\n` (generation prompt; model continues) |

During training the model predicts the **full trace starting without `<think>`**. At inference it continues from `<think>\n`. The model must bridge this — it works in practice because the BF16 base model was pre-trained with thinking and understands `<think>` as the reasoning opener, so the fine-tuning signal on `</think>\n\boxed{}` is sufficient.

### 4. `enable_thinking=False` path

When `enable_thinking=False` with `add_generation_prompt=True`:

```jinja
{{- '<|im_start|>assistant\n<think></think>' }}
```

The evaluator patch in `v0.11-lora-grpo-spark-plan.md` toggled `enable_thinking=false` in the tokenizer to suppress thinking for NeMo RL rollouts. This emits `<think></think>` (empty, immediate close) instead of opening the block — relevant if switching to non-thinking inference.

---

## Implication for generate_equation_symbolic.py

The synthetic CoT in `generate_equation_symbolic.py` builds:

```python
lines = ["<think>", "We need to infer...", ...]
cot = "\n".join(lines)
assistant = f"{cot}\n</think>\n\\boxed{{{test_out}}}"
```

This produces content that starts with `<think>`, so the template's case-3 path applies: content is left as-is, producing:

```
<|im_start|>assistant
<think>
We need to infer...
</think>
\boxed{answer}<|im_end|>
```

This is **different** from the other 15,681 training examples (which have no `<think>` opener). The 500 equation_symbolic examples are more aligned with the inference context (which provides `<think>\n`), but they are inconsistent with the rest of the corpus.

**Fix**: remove `"<think>"` from the `lines` list in `build_cot()` so all training examples have the same format:

```python
lines = [
    "We need to infer a hidden transformation rule from the given examples.",
    "",
    ...
]
```

This is a minor data quality issue (500/16,181 = 3% of training examples). Low priority given the June 15 deadline.

---

## Format 4: origin and taxonomy

"Format 4" is **not from any external citation** — it was coined within this project in `scripts/show_data_formats.py`, a diagnostic script written during the v0.4r3 regression investigation (`docs/investigate/v0.4r3-training-data-alignment.md`) to compare four training data format variants side-by-side.

| Label | Description | Source |
|---|---|---|
| **Format 1** | v0.1 raw — bare answer, no `\boxed{}`, no `<think>` | early project data |
| **Format 2** | v0.5 SFT — template-reasoning trace, missing `</think>` wrapper | v0.5 pipeline bug |
| **Format 3** | kishanvavdara / huikang long CoT — has `<think>...</think>` but 3,000+ token traces | huikang corpus |
| **Format 4** | **CORRECT target** — `{trace}\n</think>\n\boxed{}`, aligned to Nemotron-H chat template | this investigation |

"Format 4" is simply the 4th item enumerated in that script. The term became canonical at `9376afc feat(v0.9): add SFT pipeline — Format 4, 14 categories, base model` — the commit that introduced the v0.9 training pipeline using the correct format.

### Why Formats 1–3 were wrong: the v0.4r3 regression

Score history that triggered the investigation: **v0.1=0.57 → v0.4-r1=0.49 → v0.4-r2=0.50 → v0.4-r3=0.48** — three successive patches made the score *worse*.

Full root-cause analysis is in `docs/investigate/v0.4r3-training-data-alignment.md`. The three problems found:

**Problem 1 — Long CoT hits Kaggle's token limit (Format 3)**

The huikang corpus (which kishanvavdara traces derive from) uses 3,000+ token systematic hypothesis-search reasoning chains. Kaggle's evaluation runner has a fixed `max_new_tokens` budget. A model trained on Format 3 attempts to generate the full chain before reaching `\boxed{}`, hits the limit mid-chain, and never outputs an answer. This is the primary reason Format 3 fails in competition.

**Problem 2 — Training prompt has `\boxed{}` instruction; eval prompt does not**

For the 9 "boxed" categories (bit, cipher, numeral, unit, equation…), the huikang corpus appended:
```
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```
The competition eval runner sends the raw problem without this instruction. The model learned to rely on this trigger and would not emit `\boxed{}` without it. Augmenter categories never had the instruction at all, so they scored 0 across all v0.4 runs.

**Problem 3 — Format inconsistency across the corpus**

Some huikang/kishanvavdara entries had both `<think>` and `</think>` (template case 3, left as-is), others had only `</think>` (template case 2), and v0.1 had neither (template case 1, `<think></think>` prepended). The model saw three different assistant-turn structures and learned none of them reliably.

### Resolution: Format 4

The v0.5 plan adopted kuangyicheng's method (the 0.87 notebook): short responses, competition-matched prompt format, warmstart from huikang's v27 adapter. The correct assistant content structure — `{concise trace}\n</think>\n\boxed{answer}` with no `<think>` opener — is what this doc calls Format 4. It:
- Keeps traces short enough to complete within `max_new_tokens`
- Aligns with the Nemotron-H chat template's case-2 path (see §2 above)
- Does not depend on a `\boxed{}` instruction in the user prompt

---

## Summary

| Claim | Accurate? |
|---|---|
| "Chat template auto-prepends `<think>\n`" | **Partially** — only at inference (`add_generation_prompt=True`), not during training |
| Format 4 training target: `{trace}\n</think>\n\boxed{}` | **Correct** — template leaves this as-is |
| Model sees `</think>` during training | **Yes** — it's in the loss-bearing tokens |
| Model sees `<think>` at start of assistant turn during training | **No** — not with our `add_generation_prompt=False` setup |
| Train/inference mismatch is harmful | **Not in practice** — base model's pre-training handles the `<think>` token |
