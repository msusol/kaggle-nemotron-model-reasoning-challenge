# Step 3: Docker Execution

## Start the NeMo training container

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
  -v /home/your_user/data:/data \
  -v /home/your_user/configs:/workspace/configs \
  -e WANDB_API_KEY="your_wandb_key" \
  nvcr.io/nvidia/nemo:25.02 /bin/bash
```

Use NeMo 25.02 or later — earlier releases (24.07) have known issues with the
NemotronH hybrid Mamba/MoE architecture.

## Download the dataset inside the container

```bash
pip install kaggle
kaggle datasets download gdataranger/huikang-nemotron-nemo-sft-r32 -p /data --unzip
```

## Kick off training

```bash
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
    --config-path=/workspace/configs \
    --config-name=nemotron_sft_config.yaml
```

## Monitor training

```bash
# Loss and accuracy logged to wandb and to /data/nemotron_runs/
tail -f /data/nemotron_runs/nemotron3-nano-reasoning-lora-v0.4b/logs/training.log
```

## Expected runtime

At `micro_batch_size=1`, `global_batch_size=4`, `seq_length=8192` on a single DGX H100:
- ~40–60 s/step (estimate; depends on DGX generation)
- ~948 steps for one epoch over 15,159 examples
- ~12–16 hours total
