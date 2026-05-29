# Review of the Current `Dockerfile.gb10`

## What is already good

The current image starts from NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:25.12-py3`, which is appropriate for CUDA 13-era GB10 workflows and gives a strong base for large-model training.[cite:57] The file already installs the usual Hugging Face fine-tuning stack: `transformers`, `datasets`, `accelerate`, `peft`, `trl`, and `dspy-ai`.[cite:35]

The source build of `bitsandbytes` is also useful because Blackwell and CUDA 13 support have required newer builds and source-based workarounds in some environments.[cite:57][cite:60] Keeping this path makes sense if QLoRA is still part of the experimentation plan.[cite:57]

## Highest-risk issue

The biggest likely issue is the pinned `transformers==4.50.0` version.[cite:49] NVIDIA's Nemotron 3 Nano model card indicates Hugging Face usage tested on Transformers 4.57.3, so the first assumption should be that model loading or architecture support may require an upgrade before training begins.[cite:49][cite:53]

## Version recommendations

### Likely keep

- `datasets==3.2.0`, `accelerate==1.3.0`, `peft==0.14.0`, and the general Python tooling are reasonable as a first pass if they work with the upgraded model stack.[cite:49]
- The general environment variables for Hugging Face cache and tokenizer parallelism are fine for this workflow.[cite:34]

### Re-evaluate immediately

- `transformers==4.50.0` should be tested against Nemotron-3-Nano-30B before any real work.[cite:49]
- `trl<0.16.0` should be checked only after Transformers compatibility is confirmed because training-stack interactions often break at the version boundary.[cite:49]
- The bitsandbytes compute capability list currently ends at `90`; GB10 / Blackwell discussions refer to `sm_121`, so the current build flags may need updating if 4-bit loading is unstable on the target hardware.[cite:55][cite:60]

## Minimal change strategy

A safe strategy is to change as little as possible at first.[cite:49] Start by upgrading Transformers to a Nemotron-tested version, run a smoke test for tokenizer and model load, and only then decide whether other libraries must move with it.[cite:49][cite:53]
