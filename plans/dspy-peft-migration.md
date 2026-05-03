# DSPy + PEFT Migration Plan

## Summary

The existing Hugging Face + DSPy + PEFT workflow should be treated as a research and training environment, while the Kaggle submission should be treated as a distilled artifact containing only the Nemotron LoRA adapter.[cite:1][cite:35]

## DSPy role in this competition

DSPy is still useful for offline optimization, especially for prompt search, decomposition strategies, synthetic data generation, and scoring experiments.[cite:35] The key limit is that Kaggle scoring will not execute custom DSPy control flow, so any gains found through DSPy must be transferred into the final LoRA's behavior.[cite:1][cite:35]

## PEFT changes

- Use LoRA only, not a different adapter family, unless the output still resolves to a compatible LoRA submission format.[cite:1]
- Enforce rank 32 or below.[cite:1]
- Verify Nemotron target modules before launching long runs because target names may differ from the previous model family.[cite:34]

## Data changes

The old data pipeline should be changed so targets are reasoning-centric rather than just instruct-style answers.[cite:1] Each target answer should end with one final boxed answer line because that directly supports the metric extraction path used by Kaggle.[cite:1]

## Validation changes

Validation should move from generic response quality checks to metric-aware answer extraction and answer correctness scoring.[cite:1] That means local scripts should check whether the model gives a parseable final answer, not just whether its reasoning sounds plausible.[cite:1]

## Export changes

After fine-tuning, the adapter should be written with the normal Hugging Face PEFT export path so that adapter weights and `adapter_config.json` are saved together.[cite:30] That export directory becomes the source material for `submission.zip`.[cite:1]
