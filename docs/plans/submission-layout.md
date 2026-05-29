# Submission Layout

## Expected contents

The Kaggle competition requires a LoRA adapter for Nemotron-3-Nano-30B, and the adapter must include `adapter_config.json`.[cite:1][cite:54] NVIDIA documentation for LoRA checkpoints also shows the common saved form as adapter weights plus `adapter_config.json` in the adapter directory.[cite:62]

## Practical adapter directory

A practical exported adapter directory will usually look like this:

```text
my_adapter/
├── adapter_config.json
├── adapter_model.safetensors
└── README_or_metadata_optional.txt
```

The exact weight filename may vary, but the important requirement is that the adapter config and saved LoRA weights are both present and loadable by the runtime.[cite:1][cite:62]

## Packaging rule

The `submission.zip` archive should contain the adapter payload in the structure expected by the competition's loader and should avoid unrelated training artifacts such as optimizer checkpoints, TensorBoard logs, or cached datasets.[cite:1] The safest workflow is to export the adapter to a clean directory and zip that directory as the submission artifact.[cite:30][cite:62]
