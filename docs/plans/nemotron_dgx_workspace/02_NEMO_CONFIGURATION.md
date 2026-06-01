# Step 2: NeMo Configuration

## nemotron_sft_config.yaml

```yaml
model:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  num_moe_experts: 128
  moe_router_topk: 6
  pre_tokenized_dataset: True
  data:
    train_ds:
      file_names: ["/data/nemo_train.jsonl"]
      max_seq_length: 8192
    validation_ds:
      file_names: ["/data/nemo_valid.jsonl"]
      max_seq_length: 8192
  micro_batch_size: 1
  global_batch_size: 4
  seq_length: 8192
  bf16: True
  # Selective activation checkpointing frees intermediate activations after each layer's
  # forward pass, keeping peak activation memory manageable at seq_len=8192.
  activations_checkpoint_method: "uniform"
  activations_checkpoint_granularity: "selective"
  peft:
    peft_scheme: "lora"
    lora_tuning:
      r: 32                  # competition maximum
      adapter_alpha: 32      # match lora_alpha=32 used in PEFT reference training
      lora_dropout: 0.05
      # Attention projections + Mamba in_proj/out_proj + MoE FFN up_proj/down_proj.
      # lm_head excluded: PEFT classifies it as an embedding layer and bloats the adapter.
      target_modules:
        - "q_proj"
        - "k_proj"
        - "v_proj"
        - "o_proj"
        - "in_proj"
        - "out_proj"
        - "up_proj"
        - "down_proj"

exp_manager:
  exp_dir: "/data/nemotron_runs"
  name: "nemotron3-nano-reasoning-lora-v0.4b"
  create_wandb_logger: True
  wandb_logger_kwargs:
    project: "nemotron3-nano-tuning"
    entity: "YOUR-WANDB-USERNAME"
```

## Key parameter notes

| Parameter | Value | Rationale |
|---|---|---|
| `r` | 32 | Competition maximum — do not increase |
| `adapter_alpha` | 32 | Matches `lora_alpha` in PEFT training; `alpha/r = 1.0` scaling |
| `micro_batch_size` | 1 | Required at seq_len=8192 with 128 GB memory |
| `global_batch_size` | 4 | Effective gradient accumulation over 4 micro-batches |
| `activations_checkpoint_granularity` | selective | Frees activation memory layer-by-layer during backward |

## System prompt

The dataset tokens were produced with:
```
"Solve the problem step by step. Put your final answer in \boxed{}."
```
Use this exact prompt at inference time to stay in-distribution.
