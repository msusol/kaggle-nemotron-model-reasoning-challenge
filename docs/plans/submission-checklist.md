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

## Kaggle notebook environment

- transformers 5.5.3 wheel uploaded as a Kaggle dataset input.[cite:147]
- Wheel installed at notebook start with:
  ```python
  !pip install -q --no-deps --force-reinstall \
    "/kaggle/input/<dataset>/transformers-5.5.3-py3-none-any.whl"
  ```
- No `trust_remote_code=True` in any `from_pretrained` call — omitting it ensures
  transformers 5.5.3's native NemotronH code is used, not the cached buggy version.[cite:147]

## Competition checks

- Submission deadline and team-merger deadline have been reviewed.[cite:1]
- Public notebook and write-up plan exist for prize eligibility.[cite:1]
