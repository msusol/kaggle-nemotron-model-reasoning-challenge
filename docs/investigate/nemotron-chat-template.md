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

## Summary

| Claim | Accurate? |
|---|---|
| "Chat template auto-prepends `<think>\n`" | **Partially** — only at inference (`add_generation_prompt=True`), not during training |
| Format 4 training target: `{trace}\n</think>\n\boxed{}` | **Correct** — template leaves this as-is |
| Model sees `</think>` during training | **Yes** — it's in the loss-bearing tokens |
| Model sees `<think>` at start of assistant turn during training | **No** — not with our `add_generation_prompt=False` setup |
| Train/inference mismatch is harmful | **Not in practice** — base model's pre-training handles the `<think>` token |
