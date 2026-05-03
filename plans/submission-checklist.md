# Submission Checklist

## Adapter checks

- Base model used for training is Nemotron-3-Nano-30B.[cite:1][cite:34]
- LoRA rank is 32 or lower.[cite:1]
- Adapter directory contains `adapter_config.json`.[cite:1]
- Adapter loads successfully against the Nemotron base model in local testing.[cite:30]

## Behavior checks

- Model produces a final answer in `\\boxed{}` format consistently.[cite:1]
- Local validation script can extract the final answer from representative outputs.[cite:1]
- Zero-temperature generation does not break answer formatting.[cite:1]

## Packaging checks

- `submission.zip` contains only the assets required by the competition runner.[cite:1]
- Archive opens cleanly and preserves the adapter file structure.[cite:1][cite:30]

## Competition checks

- Submission deadline and team-merger deadline have been reviewed.[cite:1]
- Public notebook and write-up plan exist for prize eligibility.[cite:1]
