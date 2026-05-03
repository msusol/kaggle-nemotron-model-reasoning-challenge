# Proposed Changes to `Dockerfile.gb10`

## Summary

The existing Dockerfile is already a strong base, so the best approach is a narrow update rather than a rebuild from scratch.[cite:57] The most important changes are a Nemotron-compatible Transformers version check, a smoke-test script, and packaging scripts for the Kaggle adapter output.[cite:49][cite:54]

## Suggested edits

### 1. Revisit the Transformers pin

Replace the current fixed pin on `transformers==4.50.0` with either a newer tested version or a controlled range that includes Nemotron-tested releases.[cite:49][cite:53]

Example direction:

```dockerfile
RUN pip install \
    "transformers>=4.57.3,<4.58" \
    "datasets==3.2.0" \
    "accelerate==1.3.0" \
    "peft==0.14.0" \
    "torchao==0.16.0" \
    "trl<0.16.0" \
    "sentencepiece" \
    "scipy" \
    "evaluate" \
    "scikit-learn" \
    "pydantic" \
    "pandas" \
    "tqdm" \
    "jsonschema"
```

This exact combination still needs a smoke test because TRL and PEFT compatibility can shift when Transformers changes.[cite:49]

### 2. Add model defaults

Add environment variables for the Nemotron base model and output paths so training and packaging scripts are easier to standardize.[cite:34]

```dockerfile
ENV BASE_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    ADAPTER_OUTPUT_DIR=/workspace/output/adapter \
    SUBMISSION_DIR=/workspace/output/submission
```

### 3. Add helper scripts

Copy in or create three helper scripts inside the image:

- `scripts/smoke_test_nemotron.py` for model-load validation,[cite:49]
- `scripts/validate_metric.py` for boxed-answer validation,[cite:1][cite:54]
- `scripts/package_submission.sh` for building `submission.zip`.[cite:1]

### 4. Re-check bitsandbytes build flags

If QLoRA is needed, revisit the hard-coded compute capability list because Blackwell GB10 discussions reference `sm_121`, while the current flags stop at `90`.[cite:55][cite:60] If standard bf16 LoRA works within available memory, that may be the simpler early path.[cite:49]

## Recommended order of operations

1. Update Transformers and run smoke test.[cite:49]
2. Verify Nemotron load and generation.[cite:49]
3. Verify PEFT save and adapter export.[cite:62]
4. Add metric-aware validation.[cite:1]
5. Add submission packaging.[cite:1]
