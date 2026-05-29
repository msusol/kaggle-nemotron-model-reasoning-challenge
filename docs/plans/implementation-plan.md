# Implementation Plan for Adapting the Existing DSPy + PEFT Workflow

## Phase 1: Smoke test the model stack

The first task is to verify that the current container can load the Nemotron-3-Nano-30B Hugging Face checkpoint and generate text successfully.[cite:34][cite:49] This is more important than starting training immediately because the current Transformers pin may be too old for the model definition.[cite:49]

### Phase 1 tasks

1. Replace the prior model ID with `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` or the exact competition-compatible checkpoint used in the workflow.[cite:34]
2. Test tokenizer load, model load, one-token generation, and one batch forward pass.[cite:49]
3. Only after that, test PEFT wrapping and confirm a tiny adapter can be saved with `adapter_config.json`.[cite:62]

## Phase 2: Adapt the fine-tuning objective

The competition metric prefers boxed extraction, so the supervised targets should end with one final `\\boxed{}` answer line.[cite:1][cite:54] Existing instruction-tuning or DSPy-generated outputs should be reformatted so they emphasize deterministic final-answer behavior rather than open-ended helpfulness.[cite:1]

### Phase 2 tasks

- Reformat datasets into reasoning tasks with a single final answer.[cite:1]
- Update prompt templates so the assistant always ends with `Final answer: \\boxed{...}`.[cite:1]
- Add validation code that parses the boxed answer first and checks correctness with the same style of comparison used by Kaggle.[cite:1][cite:54]

## Phase 3: Keep DSPy offline

DSPy remains useful for generating better prompts, decompositions, or synthetic reasoning traces.[cite:35] The important implementation rule is that Kaggle will not run the DSPy graph, so DSPy should be treated as an offline optimizer whose outputs become training examples or prompt templates for the final LoRA model.[cite:1][cite:35]

## Phase 4: Package the submission

The final output should be the adapter directory only, saved via Hugging Face PEFT and zipped into `submission.zip`.[cite:1][cite:62] A packaging script should verify that the archive contains `adapter_config.json` and the adapter weight files, and nothing unnecessary.[cite:1]
