# Dataset Comparison: Raw Competition Data vs Peer CoT Dataset

Examines the difference between the two training data sources used across runs:

| Run | Dataset | Kaggle Score |
|---|---|---|
| v0.1-baseline | Raw competition data — bare labels only | 0.57 |
| v0.2-cot | Peer CoT dataset — Gemini-2.0-flash reasoning traces | TBD |

---

## Source and Schema

### Raw competition data (`download_data.py`)

Downloaded via `kagglehub.competition_download("nvidia-nemotron-model-reasoning-challenge")`.

| Column | Content |
|---|---|
| `id` | Example ID |
| `prompt` | Competition problem text |
| `answer` | Ground-truth final answer |

Converted to JSONL with `response = "Final answer: \boxed{answer}"` — **no reasoning, just the answer**.

### Peer CoT dataset (`download_peer_cot.py`)

Downloaded via `kagglehub.dataset_download("kienngx/nemotron-30b-competition-trainingdata-cot-labels")`.

| Column | Content |
|---|---|
| `id` | Example ID |
| `prompt` | Same competition problem text |
| `answer` | Ground-truth final answer |
| `generated_cot` | Gemini-2.0-flash reasoning trace |
| `label` | Problem category (not used in training) |

Converted to JSONL with `response = "{generated_cot}\nFinal answer: \boxed{answer}"` — **full reasoning chain before the answer**.

---

## JSONL Format Comparison

Both datasets produce the same three output files (`train.jsonl`, `valid.jsonl`, `valid_labels.jsonl`)
with the same schema, so `train_lora.py` is unchanged between runs. The only difference is the
`response` field.

**v0.1-baseline row:**
```json
{
  "id": "ec897854",
  "prompt": "In Alice's Wonderland, a secret unit conversion is applied...\nNow, convert the following measurement: 9.22 m",
  "response": "Final answer: \\boxed{13.55}",
  "system": "You are a careful reasoning model. Solve the problem step by step and end with Final answer: \\boxed{...}."
}
```

**v0.2-cot row (same example):**
```json
{
  "id": "ec897854",
  "prompt": "In Alice's Wonderland, a secret unit conversion is applied...\nNow, convert the following measurement: 9.22 m",
  "response": "Let's analyze the given examples to identify the conversion rule:\n\n18.06 m -> 26.55 ; Ratio 26.55/18.06 = 1.4701\n48.97 m -> 71.99 ; Ratio 71.99/48.97 = 1.4701\n...\n\nIt appears the consistent ratio is around 1.47. Applying this to our target value:\n9.22 m -> X\n    X = 9.22 * 1.47 = 13.5534\n\nRounded to two decimal places, the converted measurement is 13.55.\nFinal answer: \\boxed{13.55}",
  "system": "You are a careful reasoning model. Solve the problem step by step and end with Final answer: \\boxed{...}."
}
```

---

## Side-by-Side Examples

### Example 1 — Unit conversion (`id=ec897854`) ✓ Strong CoT

**Prompt:**
```
In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:
18.06 m becomes 26.55
48.97 m becomes 71.99
40.49 m becomes 59.52
32.64 m becomes 47.98
30.42 m becomes 44.72
Now, convert the following measurement: 9.22 m
```

**v0.1-baseline response:**
```
Final answer: \boxed{13.55}
```

**v0.2-cot response:**
```
Let's analyze the given examples to identify the conversion rule:

18.06 m -> 26.55 ; Ratio 26.55/18.06 = 1.4701
48.97 m -> 71.99 ; Ratio 71.99/48.97 = 1.4701
40.49 m -> 59.52 ; Ratio 59.52/40.49 = 1.4700
32.64 m -> 47.98 ; Ratio 47.98/32.64 = 1.4700
30.42 m -> 44.72 ; Ratio 44.72/30.42 = 1.4700

It appears the consistent ratio is around 1.47. Applying this to our target value:

9.22 m -> X
    X = 9.22 * 1.47 = 13.5534

Rounded to two decimal places, the converted measurement is 13.55.
Final answer: \boxed{13.55}
```

**Assessment:** Gemini correctly identified the ~1.47× ratio, showed the arithmetic, and reached the right answer. This is the best-case CoT — systematic, correct, and teachable.

---

### Example 2 — Text cipher (`id=73cb93e8`) ~ Weak but coherent CoT

**Prompt:**
```
In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
kqootf bqfsyhv fyh vfkqglh xqe -> rabbit watches the strange map
fyh btvh sqf nmpgd -> the wise cat found
vfpdhgf bktfhv fyh dqki lqkdhg -> student writes the dark garden
kqootf bqfsyhv fyh vtruhk xtkkmk -> rabbit watches the silver mirror
Now, decrypt the following text: yqffhk dtvsmuhkv qkmpgd bmgdhkrqgd
```

**v0.1-baseline response:**
```
Final answer: \boxed{hatter discovers around wonderland}
```

**v0.2-cot response:**
```
Given the encryptions, with that it is possible to derive that all of the rules and phrases
has a one to one mapping, so by extrapolating the result that needs to generate a specific
conclusion to determine it. We need to solve all the translations.

This makes "hatter discovers around wonderland" the most related to what those
transformation is. And this would then apply based on how the phrases translate.
Final answer: \boxed{hatter discovers around wonderland}
```

**Assessment:** The CoT is brief and vague — it asserts a pattern without demonstrating the actual letter mapping. Still, the answer is correct and the reasoning isn't wrong, just thin. The model will learn that cipher problems involve pattern-matching word-by-word.

---

### Example 3 — Text cipher (`id=514fb8df`) ✗ Degenerate CoT

**Prompt:**
```
In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
pda zsha pxqpta gqayrh -> the wise turtle dreams
payldaq srywsuah rsqqeq -> teacher imagines mirror
oxaau zypldah yqexug pezaq -> queen watches around tower
fuswdp bettezh pqayhxqa -> knight follows treasure
dyppaq zqspah uayq elayu -> hatter writes near ocean
Now, decrypt the following text: payldaq zypldah s
```

**v0.1-baseline response:**
```
Final answer: \boxed{teacher watches inside mountain}
```

**v0.2-cot response:**
```
Let's analyze the provided examples:

1. pda zsha pxqpta gqayrh -> the wise turtle dreams
2. payldaq srywsuah rsqqeq -> teacher imagines mirror
...

Find some similar

For what

At the base the pattern shows is
The " What. We need to" The. I get all this: From it. It that a all has!

All to from: 4! The! What to

Let focus. On what : It can for 857! We then from: Now" know what each. Can : "DO with it" And there will!
...
Final answer: \boxed{teacher watches inside mountain}
```

**Assessment:** Gemini lost coherence mid-reasoning — the CoT collapses into noise before landing on the answer. This is the downside of the peer dataset: for harder problems, some Gemini traces degenerate. The answer itself (`teacher watches inside mountain`) appears to be a plausible guess but the reasoning provides no signal.

---

## Dataset Statistics

| Metric | Raw competition | Peer CoT |
|---|---|---|
| Source rows | 9,500 | 9,500 |
| Filtered (bad CoT) | n/a | 213 (< 20 chars) |
| Train examples | 8,550 | 8,358 |
| Valid examples | 950 | 929 |
| Median response chars | ~25 | ~641 |
| p90 response chars | ~35 | ~2,094 |
| p99 response chars | ~40 | ~7,266 |
| Estimated p50 tokens | ~10 | ~248 |
| Estimated p90 tokens | ~15 | ~627 |

---

## Hypothesis: Why CoT Should Help

The v0.1-baseline trained on `Final answer: \boxed{42}` responses. The model saw:
- The problem → the answer
- No reasoning path between them

The model learned **answer formatting** (outputting `\boxed{}`) but not **how to reason** about the problem types.

With CoT traces, the model sees:
- The problem → reasoning steps → the answer
- Explicit ratio-finding for unit conversions
- Word-by-word pattern matching for ciphers
- Bit manipulation analysis for binary problems

The hypothesis is that the model learns to **produce intermediate reasoning** that makes it more likely to reach the correct answer, especially on unseen problem variations. Even imperfect CoT (Example 2) provides weak signal about *what kind of reasoning* a problem category calls for.

---

## Caveats

- **CoT quality is uneven.** Approximately 2–5% of traces are degenerate (Example 3). These may
  introduce noise rather than signal for those specific examples.
- **Same prompts and answers.** Both datasets cover the identical 9,500 competition problems.
  Any accuracy improvement comes purely from the richer `response` field, not new problems.
- **Gemini's reasoning may be incorrect.** Gemini could reach the right answer via wrong reasoning.
  The model trains to imitate the trace regardless of whether the logic is valid.
- **Longer sequences = slower training.** CoT responses are ~25× longer on average, increasing
  compute per step even though `max_seq_length=2048` truncates the longest 0.9% of examples.
