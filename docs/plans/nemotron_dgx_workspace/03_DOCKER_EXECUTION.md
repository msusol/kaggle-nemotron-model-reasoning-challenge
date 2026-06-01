# Step 3: Docker Execution
Run this to start your isolated training container:
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
  -v /home/your_user/data:/data \
  -v /home/your_user/configs:/workspace/configs \
  -e WANDB_API_KEY="your_wandb_key" \
  nvcr.io/nvidia/nemo:24.07 /bin/bash
```
Kick off training from inside the container bash shell:
```bash
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
    --config-path=/workspace/configs \
    --config-name=nemotron_sft_config.yaml
```