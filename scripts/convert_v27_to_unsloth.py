#!/usr/bin/env python3
"""Convert huikang v27 adapter keys to current Unsloth format.

v27 was saved by an older Unsloth that stored MoE experts as batched 3-D tensors
under w1/w2/w3 keys and used model.model.layers paths. Current Unsloth exposes
128 individual expert nn.Linear modules under model.backbone.layers paths.

This conversion enables PeftModel.from_pretrained(model, converted_v27) to load
ALL 12,008 LoRA modules (including MoE experts) instead of only the ~186 standard
attention-layer keys, giving a proper warmstart for joint MoE+attention training.

Key mappings applied
--------------------
Path:        model.model.layers.X  →  model.backbone.layers.X
w1 (up_proj):
  w1.lora_A  (1, r, in_dim)    →  experts.N.up_proj.lora_A  (r, in_dim)   [broadcast]
  w1.lora_B  (128, out_dim, r) →  experts.N.up_proj.lora_B  (out_dim, r)  [slice N]
w2 (down_proj):
  w2.lora_A  (128, r, in_dim)  →  experts.N.down_proj.lora_A (r, in_dim)  [slice N]
  w2.lora_B  (1, out_dim, r)   →  experts.N.down_proj.lora_B (out_dim, r) [broadcast]
w3:          shape (0,) — empty, skipped
gate_proj / x_proj: not in current Unsloth model, skipped
shared_experts, attention layers: path fix only

Usage
-----
    python scripts/convert_v27_to_unsloth.py \\
        --input  output/adapter_huikang_v27 \\
        --output output/adapter_huikang_v27_unsloth
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


NUM_EXPERTS = 128
OLD_PATH = "base_model.model.model.layers."
NEW_PATH = "base_model.model.backbone.layers."

# Keys in v27 that have no Unsloth equivalent or hurt inference — skip entirely
# lm_head: kuangyicheng notebook notes "NO lm_head LoRA (drops score with SFTTrainer)"
SKIP_PATTERNS = ("gate_proj", "x_proj", ".w3.", "lm_head")


def convert(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    # ── read all v27 tensors ────────────────────────────────────────────────
    v27: dict[str, torch.Tensor] = {}
    with safe_open(str(src / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            v27[k] = f.get_tensor(k)

    print(f"v27 keys loaded: {len(v27)}")

    out: dict[str, torch.Tensor] = {}
    skipped = 0
    converted_moe = 0
    path_fixed = 0

    for key, tensor in v27.items():
        # Skip Mamba-specific layers not present in current Unsloth model
        if any(p in key for p in SKIP_PATTERNS):
            skipped += 1
            continue

        # Skip empty tensors (w3 entries with shape (0,))
        if tensor.numel() == 0:
            skipped += 1
            continue

        # Fix path prefix first (applies to all remaining keys)
        new_key = key.replace(OLD_PATH, NEW_PATH)

        # ── MoE expert batched → individual conversion ──────────────────────
        # Pattern: ...experts.w1.lora_{A|B}.weight  or  ...experts.w2.lora_{A|B}.weight
        moe_match = re.search(r"\.mixer\.experts\.(w1|w2)\.(lora_[AB])\.weight$", new_key)
        if moe_match:
            w_name  = moe_match.group(1)  # "w1" or "w2"
            lora_ab = moe_match.group(2)  # "lora_A" or "lora_B"
            prefix  = new_key[: moe_match.start()]  # everything before .mixer.experts.wX...

            # Determine projection name and slice/broadcast axis
            # w1 = up_proj:   lora_A (1, r, d_in) → broadcast; lora_B (128, d_out, r) → slice
            # w2 = down_proj: lora_A (128, r, d_in) → slice;   lora_B (1, d_out, r)   → broadcast
            proj = "up_proj" if w_name == "w1" else "down_proj"
            broadcast = (
                (w_name == "w1" and lora_ab == "lora_A") or
                (w_name == "w2" and lora_ab == "lora_B")
            )

            if broadcast:
                # Shape (1, ...) — squeeze and share across all experts
                base = tensor.squeeze(0)  # (r, d_in) or (d_out, r)
                for n in range(NUM_EXPERTS):
                    expert_key = f"{prefix}.mixer.experts.{n}.{proj}.{lora_ab}.weight"
                    out[expert_key] = base.clone().to(torch.bfloat16)
            else:
                # Shape (128, ...) — one slice per expert
                assert tensor.shape[0] == NUM_EXPERTS, \
                    f"Expected {NUM_EXPERTS} experts, got {tensor.shape[0]} in {key}"
                for n in range(NUM_EXPERTS):
                    expert_key = f"{prefix}.mixer.experts.{n}.{proj}.{lora_ab}.weight"
                    out[expert_key] = tensor[n].clone().to(torch.bfloat16)

            converted_moe += 1
            continue

        # ── All other keys: path already fixed, just cast dtype ─────────────
        out[new_key] = tensor.to(torch.bfloat16)
        path_fixed += 1

    print(f"MoE batched tensors expanded: {converted_moe} → {converted_moe * NUM_EXPERTS} keys")
    print(f"Standard keys (path fixed):   {path_fixed}")
    print(f"Skipped (gate_proj/x_proj/w3/empty): {skipped}")
    print(f"Total output keys: {len(out)}")

    # ── save converted adapter ──────────────────────────────────────────────
    save_file(out, str(dst / "adapter_model.safetensors"))
    size_gb = (dst / "adapter_model.safetensors").stat().st_size / 1e9
    print(f"Saved: {dst / 'adapter_model.safetensors'}  ({size_gb:.2f} GB)")

    # Copy and patch adapter_config.json
    cfg_src = src / "adapter_config.json"
    with open(cfg_src) as f:
        cfg = json.load(f)
    # target_modules stays all-linear; Unsloth will resolve to backbone.layers structure
    with open(dst / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("adapter_config.json copied")

    # Copy tokenizer files if present
    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "special_tokens_map.json"):
        s = src / fname
        if s.exists():
            shutil.copy2(s, dst / fname)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  required=True, help="Path to v27 adapter dir")
    ap.add_argument("--output", required=True, help="Output dir for converted adapter")
    args = ap.parse_args()

    convert(Path(args.input), Path(args.output))
    print("\nConversion complete. Use --warmstart-dir pointing to the output dir.")


if __name__ == "__main__":
    main()
