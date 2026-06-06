#!/usr/bin/env python3
"""
remap_sft_adapter_for_grpo.py — Remap v0.5 SFT adapter keys to GRPO-compatible format.

The v0.5 SFT adapter (12,008 keys) was trained with Unsloth's per-expert MoE expansion
and its internal `backbone` attribute rename. GRPO uses the standard `model` attribute
name and fused `shared_experts` only (232 keys). The two key differences:

  SFT  key: base_model.model.backbone.layers.N.mixer.in_proj.lora_A.weight
  GRPO key: base_model.model.model.layers.N.mixer.in_proj.lora_A.default.weight

Transformation:
  1. Drop the 11,776 per-expert keys (experts.0-127.up_proj/down_proj) — no GRPO counterpart
  2. Rename `backbone` → `model`
  3. Add `.default` infix: `lora_A.weight` → `lora_A.default.weight`

Output: output/adapter_v5_sft_grpo_warmstart/ with remapped adapter + config

Usage:
  python scripts/remap_sft_adapter_for_grpo.py
  python scripts/remap_sft_adapter_for_grpo.py \
      --src output/adapter_v5_sft_unsloth \
      --dst output/adapter_v5_sft_grpo_warmstart
"""

import argparse
import json
import re
import shutil
import struct
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent


def remap_key(k: str) -> str | None:
    """Return the GRPO-format key, or None to drop this key."""
    # Drop per-expert keys (experts.0 .. experts.127)
    if re.search(r"\.mixer\.experts\.\d+\.", k):
        return None
    # backbone → model
    k = k.replace(".backbone.", ".model.")
    # lora_A.weight / lora_B.weight → lora_A.default.weight / lora_B.default.weight
    k = re.sub(r"\.(lora_[AB])\.weight$", r".\1.default.weight", k)
    return k


def remap_safetensors(src_path: Path, dst_path: Path) -> tuple[int, int]:
    """Read src, remap keys, write dst. Returns (total_src_keys, written_keys)."""
    with open(src_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len)
        data_blob = f.read()

    src_header = json.loads(header_json)
    metadata = src_header.get("__metadata__", {})

    # Build remapped header and collect tensor data offsets
    remapped = {}
    data_offset = 0  # current write position in new data blob
    new_data_parts = []

    for src_key, tensor_info in src_header.items():
        if src_key == "__metadata__":
            continue
        dst_key = remap_key(src_key)
        if dst_key is None:
            continue
        # Slice the tensor bytes from the original blob
        src_start, src_end = tensor_info["data_offsets"]
        tensor_bytes = data_blob[src_start:src_end]
        byte_count = len(tensor_bytes)
        remapped[dst_key] = {
            "dtype": tensor_info["dtype"],
            "shape": tensor_info["shape"],
            "data_offsets": [data_offset, data_offset + byte_count],
        }
        new_data_parts.append(tensor_bytes)
        data_offset += byte_count

    if metadata:
        remapped["__metadata__"] = metadata

    new_header_bytes = json.dumps(remapped, separators=(",", ":")).encode("utf-8")
    # safetensors requires header size to be a multiple of 8
    pad = (8 - len(new_header_bytes) % 8) % 8
    new_header_bytes += b" " * pad

    with open(dst_path, "wb") as f:
        f.write(struct.pack("<Q", len(new_header_bytes)))
        f.write(new_header_bytes)
        for part in new_data_parts:
            f.write(part)

    total_src = sum(1 for k in src_header if k != "__metadata__")
    written = len(remapped) - (1 if "__metadata__" in remapped else 0)
    return total_src, written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(WORKSPACE / "output" / "adapter_v5_sft_unsloth"),
                    help="Source SFT adapter directory")
    ap.add_argument("--dst", default=str(WORKSPACE / "output" / "adapter_v5_sft_grpo_warmstart"),
                    help="Destination remapped adapter directory")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Source: {src}")
    print(f"Dest:   {dst}")

    # Remap adapter weights
    src_st = src / "adapter_model.safetensors"
    dst_st = dst / "adapter_model.safetensors"
    print(f"Remapping {src_st.name} ...")
    total, written = remap_safetensors(src_st, dst_st)
    print(f"  Source keys: {total}")
    print(f"  Written keys: {written}  (dropped {total - written} per-expert keys)")
    size_mb = dst_st.stat().st_size / 1e6
    print(f"  Output size: {size_mb:.1f} MB")

    # Verify a sample of remapped keys
    with open(dst_st, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    sample_keys = [k for k in hdr if k != "__metadata__"][:6]
    print("  Sample keys:")
    for k in sample_keys:
        print(f"    {k}")

    # Copy adapter_config.json
    src_cfg = src / "adapter_config.json"
    dst_cfg = dst / "adapter_config.json"
    shutil.copy2(src_cfg, dst_cfg)
    print(f"Copied adapter_config.json")

    # Copy tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "special_tokens_map.json"]:
        src_f = src / fname
        if src_f.exists():
            shutil.copy2(src_f, dst / fname)
            print(f"Copied {fname}")

    print()
    print("Done. Warm-start adapter ready for GRPO:")
    print(f"  --adapter-dir {dst}")
    print()
    print("The remapped adapter has 232 non-expert LoRA keys in GRPO format.")
    print("MoE routed-expert layers (experts.0-127) start from random init in GRPO")
    print("(there is no corresponding fused module in the GRPO model architecture).")


if __name__ == "__main__":
    main()
