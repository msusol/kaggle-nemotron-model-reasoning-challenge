# LoRA GRPO on DGX Spark — Implementation Plan

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`  
**Hardware:** DGX Spark (Blackwell GB10, aarch64, 130.7 GB HBM + 121 GB LPDDR5x)  
**Framework:** NeMo RL (`nano-v3` branch) + NeMo Gym  
**Algorithm:** GRPO with DAPO extensions, LoRA adapters, Megatron inference backend

---

## Why LoRA GRPO + Megatron backend

The BF16 model weights occupy ~60 GB of HBM on load. Full-weight GRPO requires
optimizer states (~120 GB for Adam) and gradients (~60 GB) on top — far exceeding
the 130 GB HBM budget. vLLM also cannot be loaded alongside the training state for
the same reason.

The solution is two-fold:

1. **LoRA** — freeze the 60 GB base weights; only train small adapter matrices
   (~1–2 GB optimizer state total). The frozen weights double as the rollout model.
2. **Megatron inference backend** — replaces vLLM for rollout generation, reusing
   the in-memory Megatron training weights with no separate process, no weight copy,
   no refit buffer.

---

## Directory structure

```
docs/plans/lora-grpo-spark/
├── README.md                          ← this file
├── 01-docker-build.md                 ← Dockerfile + run flags for GB10/arm64
├── 02-environment-setup.md            ← NeMo RL clone, NeMo Gym, model download
├── 03-data-preparation.md             ← dataset format, NeMo Gym environments
├── 04-lora-grpo-config.yaml           ← single-node Megatron LoRA GRPO config
├── 05-training-launch.md              ← launch commands, validation steps
├── 06-checkpoint-export.md            ← LoRA merge → HF format
├── 07-troubleshooting.md              ← OOM tuning, sm_121a issues, HBM cleanup
└── Dockerfile.spark                   ← complete Dockerfile for DGX Spark
```

---

## Quick-start sequence

```bash
# 1. Build the Docker image (one-time, ~30–60 min)
docker build -f Dockerfile.spark -t nemo-rl-spark:latest .

# 2. Pre-flight HBM cleanup (before every training run)
systemctl --user stop gnome-remote-desktop.service
docker rm -f nemo-rl-run 2>/dev/null || true

# 3. Launch container
docker run --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name nemo-rl-run \
  -v $(pwd):/workspace \
  -it nemo-rl-spark:latest bash

# 4. Inside container — validate HBM
python3 -c "import torch; f,t=torch.cuda.mem_get_info(); \
  print(f'HBM free: {f/1e9:.1f} GB / {t/1e9:.1f} GB')"

# 5. Run 3-step validation
cd /opt/nemo-rl
TORCH_CUDA_ARCH_LIST="12.1" HF_HOME=$PWD/.cache/ \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config /workspace/docs/plans/lora-grpo-spark/04-lora-grpo-config.yaml \
  grpo.max_num_steps=3

# 6. Full training run (remove max_num_steps override)
TORCH_CUDA_ARCH_LIST="12.1" HF_HOME=$PWD/.cache/ \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config /workspace/docs/plans/lora-grpo-spark/04-lora-grpo-config.yaml
```

---

## Memory budget at a glance

| Component                  | Estimate     | Notes                              |
|----------------------------|--------------|------------------------------------|
| Base weights (BF16)        | ~60 GB       | frozen, shared with Megatron gen   |
| LoRA adapters (r=32)       | ~0.5 GB      | trainable parameters               |
| Adam optimizer (LoRA only) | ~1–2 GB      | only covers adapter params         |
| Gradients (LoRA only)      | ~0.5 GB      |                                    |
| Activations + act. ckpt    | ~10–15 GB    | with activation_checkpointing=true |
| KV cache (Megatron gen)    | ~12 GB       | buffer_size_gb in config           |
| **Total estimated**        | **~85 GB**   | ~45 GB headroom on 130 GB HBM      |

---

## Key references

- NeMo RL repo (nano-v3 branch): https://github.com/NVIDIA-NeMo/RL/tree/nano-v3
- NeMo Gym docs: https://docs.nvidia.com/nemo/gym/0.2.1/
- Nemotron model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- DGX Spark loading thread: https://forums.developer.nvidia.com/t/loading-nvidia-nvidia-nemotron-3-nano-30b-a3b-bf16-on-dgx-spark/372168
- NeMo RL v0.5.0 release notes: https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.5.0
