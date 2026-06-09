# 02 — Environment setup

## Inside the container

All commands below run inside the `nemo-rl-spark:latest` container.

---

## NeMo RL — nano-v3 branch

The Dockerfile clones NeMo RL to `/opt/nemo-rl` on the `nano-v3` branch.
This branch contains the Mamba-H / NemotronH Megatron model support that
`main` does not yet have.

```bash
# Verify you are on the correct branch
cd /opt/nemo-rl
git branch
# Expected output: * nano-v3

# Verify submodules (Megatron-Core, NeMo-Gym, etc.) are checked out
git submodule status | head -5
```

If you need to update:

```bash
git pull origin nano-v3
git submodule update --init --recursive
uv pip install -e ".[mcore]" --no-build-isolation
```

---

## NeMo Gym

NeMo Gym provides the RL training environments (math, code, tool-use, etc.)
that feed rollout prompts and verify rewards.

```bash
# Install (included in nemo-rl nano-v3 as a submodule, or install separately)
uv pip install nemo-gym

# Verify
python3 -c "import nemo_gym; print(nemo_gym.__version__)"
```

---

## Model download

```bash
# Set workspace
export WORKSPACE=/workspace
export HF_HOME=$WORKSPACE/.cache/huggingface
mkdir -p $HF_HOME

# Download BF16 model (~60 GB safetensors)
HF_TOKEN=<your_huggingface_token> \
huggingface-cli download \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --local-dir $WORKSPACE/model \
  --local-dir-use-symlinks False

# Verify download
ls -lh $WORKSPACE/model/*.safetensors | wc -l
# Expected: 13 safetensors files
du -sh $WORKSPACE/model/
# Expected: ~60 GB
```

### Chat template patch

The Nemotron-3-Nano tokenizer ships with `enable_thinking=true` in the chat
template, which generates reasoning traces during rollouts. Disable it so the
model outputs plain answers during RL (enables controlled reasoning-on/off):

```bash
TCFG=$(find $WORKSPACE/model -name tokenizer_config.json | head -1)
echo "Patching: $TCFG"

# Disable thinking mode for rollout generation
sed -i 's/enable_thinking=true/enable_thinking=false/g' $TCFG

# Verify patch applied
grep -c "enable_thinking=false" $TCFG
# Expected: 1 or more
```

> You can re-enable thinking for inference after training by reverting this patch
> or loading the original tokenizer from HuggingFace directly.

---

## Python package verification

```bash
python3 - <<'EOF'
import torch
import transformers
import nemo_rl

# Check transformers version — must be >= 5.5.3 for native NemotronHForCausalLM
print(f"transformers: {transformers.__version__}")
assert transformers.__version__ >= "5.5.3", \
  "Need transformers >= 5.5.3 for native NemotronH support"

# Verify GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
f, t = torch.cuda.mem_get_info()
print(f"HBM: {f/1e9:.1f} GB free / {t/1e9:.1f} GB total")

# Verify causal-conv1d compiled for sm_121
import causal_conv1d
print(f"causal_conv1d: OK")

# Verify mamba-ssm patch
from mamba_ssm.ops import selective_scan_interface
print("mamba-ssm: OK (selective_scan_cuda patched)")
EOF
```

---

## Storage requirements

| Item                         | Size       |
|------------------------------|------------|
| Model weights (BF16)         | ~60 GB     |
| HuggingFace cache            | ~5 GB      |
| Training data (JSONL)        | ~1–5 GB    |
| Checkpoints (per save)       | ~1–2 GB    |
| Logs                         | ~100 MB    |
| **Minimum recommended**      | **~110 GB** |

Use a mounted volume (`-v /fast/storage:/workspace`) rather than the container
overlay filesystem — safetensors reads and checkpoint writes are I/O heavy.
