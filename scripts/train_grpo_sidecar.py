#!/usr/bin/env python3
"""v0.10 GRPO training — self-improvement with external vLLM sidecar for generation.

New architecture vs v0.6 (TRL in-process GRPO):
  - vLLM sidecar (FP8, separate Docker container) handles all rollout generation
  - Training container (BF16) handles only forward+backward passes
  - Standalone GRPO loop (no TRL GRPOTrainer dependency)
  - LoRA weights synced to vLLM every N steps via load_lora_adapter API

Memory layout on GB10 (130.7 GB):
  vLLM FP8 sidecar:  nvidia/...-FP8  ~37 GB (launched by run_grpo_sidecar.sh)
  Training BF16 base + LoRA optimizer + activations: ~83 GB
  Total: ~120 GB (10 GB headroom)

GRPO loss (DeepSeek-R1 formulation):
  Advantage: A_{b,g} = (R_{b,g} - mean_G) / (std_G + eps)
  Loss = -mean(A * sum_t log_pi(t)) + beta * KL(pi || pi_ref)
  where pi_ref = base model (LoRA disabled)

Usage (via run_grpo_sidecar.sh — always inside tmux):
    tmux new -s grpo_v10
    RUN_NAME=grpo_v10 bash scripts/run_grpo_sidecar.sh
"""

import argparse
import contextlib
import csv
import functools
import importlib.machinery
import math
import re
import sys
import types
import threading
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# ── Import Unsloth FIRST (same as train_grpo.py) ───────────────────────────
try:
    from unsloth import FastLanguageModel as _FLM  # noqa: F401
except Exception:
    pass


def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__path__ = []
    return m


for _stub_name in (
    "llm_blender", "llm_blender.blender",
    "llm_blender.blender.blender", "llm_blender.blender.blender_utils",
):
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = _make_stub(_stub_name)

warnings.filterwarnings("ignore", message=r".*save_embedding_layers.*")
warnings.filterwarnings("ignore", message=r".*Could not find a config file.*")
warnings.filterwarnings("ignore", message=r".*Unable to fetch remote file.*")
warnings.filterwarnings("ignore", message=r".*use_return_dict.*deprecated.*")
warnings.filterwarnings("ignore", message=r".*torchvision.*")

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_answer(text: str) -> str:
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""


def answers_match(pred: str, truth: str, rel_tol: float = 1e-4) -> bool:
    p, t = pred.strip(), truth.strip()
    if p == t:
        return True
    try:
        return math.isclose(float(p), float(t), rel_tol=rel_tol, abs_tol=0.0)
    except Exception:
        return False


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir",        required=True,
                    help="SFT adapter dir (Unsloth format); passed to FastLanguageModel.from_pretrained")
    ap.add_argument("--grpo-warmstart",     default=None,
                    help="Pre-remapped 232-key adapter (model.layers, .default. keys). "
                         "Build with: python scripts/remap_sft_adapter_for_grpo.py")
    ap.add_argument("--train-file",         required=True,  help="data/train.csv")
    ap.add_argument("--output-dir",         default=None)
    ap.add_argument("--vllm-server-url",    default="http://localhost:8000",
                    help="vLLM sidecar base URL (--network=host so localhost works)")
    ap.add_argument("--rollout-adapter-path", default=None,
                    help="Path for periodic LoRA sync to vLLM; default: output_dir/rollout_adapter")
    ap.add_argument("--rollout-sync-steps", type=int,   default=50,
                    help="Sync LoRA weights to vLLM every N steps")
    ap.add_argument("--num-steps",          type=int,   default=500)
    ap.add_argument("--num-generations",    type=int,   default=4)
    ap.add_argument("--learning-rate",      type=float, default=1e-6)
    ap.add_argument("--max-new-tokens",     type=int,   default=512)
    ap.add_argument("--max-prompt-length",  type=int,   default=1024)
    ap.add_argument("--kl-coeff",           type=float, default=0.04)
    ap.add_argument("--batch-size",         type=int,   default=1)
    ap.add_argument("--warmup-steps",       type=int,   default=20)
    ap.add_argument("--seed",               type=int,   default=3407)
    return ap.parse_args()


def _make_cache_dropper(interval: float = 30.0):
    import torch as _torch
    stop = threading.Event()
    def _loop():
        while not stop.wait(interval):
            try:
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("3\n")
            except OSError:
                pass
            try:
                _torch.cuda.empty_cache()
            except Exception:
                pass
    t = threading.Thread(target=_loop, daemon=True, name="cache-dropper")
    t.start()
    return stop


import torch
from datasets import Dataset
from safetensors import safe_open
from safetensors.torch import save_file as st_save_file


# ── LoRA weight sync to vLLM sidecar ───────────────────────────────────────

def sync_lora_to_vllm(model, tokenizer, vllm_url: str, rollout_path: Path) -> bool:
    """Save BF16 adapter to shared volume and trigger vLLM LoRA hot-swap.

    FP8 base + BF16 LoRA: adapters must be explicitly cast to BF16 before
    upload to avoid dtype mismatch (learned from mineral-hr-llm project).
    Non-fatal: returns False on error so training continues.
    """
    import requests
    try:
        model.save_pretrained(str(rollout_path))
        tokenizer.save_pretrained(str(rollout_path))
        # Ensure adapters are BF16 on disk (not mixed from optimizer states)
        st_path = rollout_path / "adapter_model.safetensors"
        if st_path.exists():
            tensors = {}
            with safe_open(str(st_path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k).to(torch.bfloat16)
            st_save_file(tensors, str(st_path))
        r = requests.post(
            f"{vllm_url}/v1/load_lora_adapter",
            json={"lora_name": "nemotron-policy", "lora_path": str(rollout_path)},
            timeout=120,
        )
        print(f"  vLLM LoRA sync → HTTP {r.status_code}", flush=True)
        return r.status_code == 200
    except Exception as e:
        print(f"  vLLM LoRA sync failed (non-fatal): {e}", flush=True)
        return False


# ── GRPO loss (standalone, no TRL dependency) ───────────────────────────────

@contextmanager
def _maybe_disable_adapter(model):
    """Disable LoRA adapter if available; no-op otherwise."""
    try:
        with model.disable_adapter():
            yield
    except (AttributeError, TypeError):
        # Fallback: manually disable LoRA adapter layers via PEFT API
        try:
            model.base_model.disable_adapter_layers()
            yield
        finally:
            try:
                model.base_model.enable_adapter_layers()
            except Exception:
                pass


def _get_token_log_probs(model, input_ids, attention_mask):
    """Shifted per-token log probs: position i predicts token i+1."""
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1, :]          # (B, L-1, V)
    shift_labels = input_ids[:, 1:].contiguous()       # (B, L-1)
    log_probs = shift_logits.float().log_softmax(-1)   # fp32 for stability
    return log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)


def compute_grpo_loss(
    model,
    input_ids,
    attention_mask,
    completion_mask,   # (B, L-1) — 1 for completion tokens
    advantages,        # (B,) group-relative
    kl_coeff: float,
):
    """GRPO loss = policy gradient + KL penalty.

    Policy gradient: -mean(A * sum_comp(log_pi - log_ref))
    KL penalty:       beta * mean( sum_comp( pi/ref * log(pi/ref) ) )
    Both averaged over completion tokens per sequence.
    """
    # Policy log probs (gradient flows through this)
    policy_lp = _get_token_log_probs(model, input_ids, attention_mask)  # (B, L-1)

    # Reference log probs (frozen: base model, LoRA disabled)
    with torch.no_grad(), _maybe_disable_adapter(model):
        ref_lp = _get_token_log_probs(model, input_ids, attention_mask)  # (B, L-1)

    n_comp = completion_mask.sum(-1).clamp(min=1)          # (B,)
    log_ratio = (policy_lp - ref_lp) * completion_mask     # (B, L-1)

    # Sequence-level log ratio
    seq_log_ratio = log_ratio.sum(-1) / n_comp              # (B,)

    # Policy gradient loss
    pg_loss = -(advantages * seq_log_ratio).mean()

    # KL: E_pi[log pi/ref] ≈ E_pi[log_ratio] (first-order approx; clipped for stability)
    kl_per_seq = log_ratio.sum(-1) / n_comp                 # (B,)
    kl_loss = kl_per_seq.clamp(min=0).mean()

    return pg_loss + kl_coeff * kl_loss, pg_loss.item(), kl_loss.item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path(args.adapter_dir).parent / f"adapter_grpo_sidecar_{ts}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output:         {args.output_dir}", flush=True)
    print(f"vLLM server:    {args.vllm_server_url}", flush=True)
    print(f"Rollout sync:   every {args.rollout_sync_steps} steps", flush=True)

    rollout_path = Path(args.rollout_adapter_path or Path(args.output_dir) / "rollout_adapter")
    rollout_path.mkdir(parents=True, exist_ok=True)

    # Flag file path: written after model is loaded so the orchestration script
    # knows it's safe to start the vLLM sidecar without OOM risk.
    trainer_ready_flag = Path(args.output_dir) / ".trainer_model_ready"
    vllm_ready_flag    = Path(args.output_dir) / ".vllm_sidecar_ready"
    # Clean stale flags from previous runs
    trainer_ready_flag.unlink(missing_ok=True)
    vllm_ready_flag.unlink(missing_ok=True)

    # ── Model loading (aligned with train_grpo.py) ──────────────────────────
    # Unsloth's FastLanguageModel.from_pretrained always creates fused-MoE LoRA
    # (232 modules, model.layers naming) for NemotronH in GRPO mode — even when
    # loading a per-expert SFT adapter (12,008 keys, backbone.layers naming).
    # The 11,776 per-expert SFT keys are simply missing from the GRPO structure
    # and never load. Fix: accept the 232 Unsloth modules, then inject the 232
    # warm-start weights from the pre-remapped adapter (remap_sft_adapter_for_grpo.py).
    # Do NOT fall back to AutoModelForCausalLM — loading a second 30B copy causes OOM.
    _stop = _make_cache_dropper()
    from unsloth import FastLanguageModel

    print(f"Loading base + SFT adapter via FastLanguageModel: {args.adapter_dir}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
        trust_remote_code=False,
        unsloth_force_compile=False,
        attn_implementation="eager",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for _n, _p in model.named_parameters():
        if ".lora_" in _n:
            _p.requires_grad_(True)

    n_lora = sum(1 for n, _ in model.named_parameters() if ".lora_" in n)
    print(f"Unsloth created {n_lora} LoRA modules (GRPO fused-MoE mode)", flush=True)

    if args.grpo_warmstart:
        from safetensors.torch import load_file as _st_load
        _ws_path = Path(args.grpo_warmstart) / "adapter_model.safetensors"
        if _ws_path.exists():
            print(f"Injecting warm-start weights from {_ws_path}", flush=True)
            _ws = _st_load(str(_ws_path), device="cpu")
            _params = dict(model.named_parameters())
            _loaded, _missing_ws = 0, []
            for _k, _v in _ws.items():
                if _k in _params:
                    _params[_k].data.copy_(_v.to(_params[_k].dtype).to(_params[_k].device))
                    _loaded += 1
                else:
                    _missing_ws.append(_k)
            print(f"Warm-start: injected {_loaded}/{len(_ws)} keys", flush=True)
            if _missing_ws:
                print(f"Warm-start: {len(_missing_ws)} keys not found: {_missing_ws[:3]}",
                      flush=True)
            del _ws, _params
            torch.cuda.empty_cache()
        else:
            print(f"WARNING: remapped adapter not found at {_ws_path} — LoRA starts cold",
                  flush=True)
    else:
        print("WARNING: --grpo-warmstart not provided — LoRA starts from random init. "
              "Run scripts/remap_sft_adapter_for_grpo.py first.", flush=True)

    _stop.set()
    torch.cuda.empty_cache()
    gpu_free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"Model ready. GPU free={gpu_free:.1f}GB", flush=True)

    # ── Signal orchestrator that trainer model is loaded ─────────────────────
    # run_grpo_sidecar.sh polls for this file before starting the vLLM sidecar.
    # This prevents the ~60 GB training load peak from colliding with vLLM's
    # ~32 GB FP8 load peak in the same unified memory pool.
    trainer_ready_flag.write_text(f"gpu_free_gb={gpu_free:.1f}\n")
    print(f"TRAINER_MODEL_READY: flag written → {trainer_ready_flag}", flush=True)
    print(f"TRAINER_MODEL_READY: waiting for vLLM sidecar at {args.vllm_server_url}...",
          flush=True)

    # ── Mamba fast path ──────────────────────────────────────────────────────
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = True
            print("Mamba fast path enabled", flush=True)
            break

    # ── Wait for vLLM sidecar (orchestrator starts it after this process
    #    writes .trainer_model_ready, then writes .vllm_sidecar_ready) ────────
    import time, openai, requests as _req
    _vllm_wait_timeout = 2700  # 45 min — FP8 eager≈10 min; compiled (VLLM_ENFORCE_EAGER=0) takes 23+ min
    _vllm_wait_start   = time.monotonic()
    _vllm_health_url   = f"{args.vllm_server_url}/health"

    # Prefer flag file (written by run_grpo_sidecar.sh after LoRA is loaded);
    # fall back to polling the health endpoint directly.
    while True:
        if vllm_ready_flag.exists():
            print("VLLM_READY: flag file detected", flush=True)
            break
        try:
            r = _req.get(_vllm_health_url, timeout=5)
            if r.status_code == 200:
                print("VLLM_READY: health endpoint 200", flush=True)
                break
        except Exception:
            pass
        elapsed = time.monotonic() - _vllm_wait_start
        if elapsed > _vllm_wait_timeout:
            print("ERROR: vLLM sidecar did not become ready within "
                  f"{_vllm_wait_timeout}s — aborting", flush=True)
            sys.exit(1)
        if int(elapsed) % 30 == 0:
            print(f"  Still waiting for vLLM... ({elapsed:.0f}s)", flush=True)
        time.sleep(5)

    vllm_client = openai.OpenAI(
        base_url=f"{args.vllm_server_url}/v1",
        api_key="none",
    )
    print("vLLM client ready", flush=True)

    # ── LoRA params: trainable fp32 ──────────────────────────────────────────
    lora_count = 0
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.requires_grad_(True)
            param.data = param.data.to(torch.float32)
            lora_count += 1
    print(f"LoRA params trainable (fp32): {lora_count}", flush=True)

    # ── Gradient checkpointing (NemotronH native path) ───────────────────────
    _gc_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    try:
        _gc_target = getattr(model, "base_model", model)
        _gc_target = getattr(_gc_target, "model", _gc_target)
        _gc_target._set_gradient_checkpointing(
            enable=True, gradient_checkpointing_func=_gc_func
        )
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled", flush=True)
    except Exception as _e:
        print(f"Gradient checkpointing unavailable: {_e}", flush=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}", flush=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    records = []
    with open(args.train_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", row.get("question", ""))
            answer = str(row.get("answer", row.get("label", ""))).strip()
            if prompt and answer:
                records.append({"prompt": prompt, "answer": answer})
    import random
    random.seed(args.seed)
    random.shuffle(records)
    print(f"Dataset: {len(records)} problems", flush=True)

    # ── Reward function ──────────────────────────────────────────────────────
    def reward_fn(completions, answers):
        rewards = []
        for completion, truth in zip(completions, answers):
            pred = extract_answer(completion)
            rewards.append(1.0 if (pred and answers_match(pred, str(truth))) else 0.0)
        return rewards

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=args.learning_rate * 0.1)

    # ── Tokenizer helpers ────────────────────────────────────────────────────
    pad_id = tokenizer.pad_token_id
    device = next(p for p in model.parameters() if p.requires_grad).device

    def _tokenize_batch(prompts, completions):
        """Tokenize prompt+completion pairs; return input_ids, attn_mask, completion_mask."""
        full_texts = [p + c for p, c in zip(prompts, completions)]
        encodings = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length + args.max_new_tokens,
            add_special_tokens=False,
        )
        input_ids = encodings.input_ids.to(device)
        attn_mask = encodings.attention_mask.to(device)

        # Build completion mask: 1 for completion tokens in the shifted (L-1) space
        B, L = input_ids.shape
        comp_mask = torch.zeros(B, L - 1, device=device)
        for i, (p, c) in enumerate(zip(prompts, completions)):
            p_ids = tokenizer.encode(p, add_special_tokens=False)
            p_len = len(p_ids)
            # In shifted space position j predicts token j+1 (0-indexed).
            # Completion tokens start at index p_len in the full sequence,
            # so in shifted space they start at p_len-1.
            start = max(0, p_len - 1)
            # End = last non-padding position in shifted space
            end = int(attn_mask[i, 1:].sum().item())
            if start < end:
                comp_mask[i, start:end] = 1.0
        return input_ids, attn_mask, comp_mask

    # ── GRPO training loop ───────────────────────────────────────────────────
    print("Starting GRPO training...", flush=True)
    model.train()
    step = 0
    data_iter = iter(records)
    steps_since_sync = 0
    log_every = 5

    while step < args.num_steps:
        # Sample a problem (cycle through dataset)
        try:
            sample = next(data_iter)
        except StopIteration:
            random.shuffle(records)
            data_iter = iter(records)
            sample = next(data_iter)

        prompt = sample["prompt"]
        truth = sample["answer"]

        # ── Generate G completions from vLLM ─────────────────────────────────
        try:
            resp = vllm_client.completions.create(
                model="nemotron-policy",
                prompt=prompt,
                max_tokens=args.max_new_tokens,
                temperature=0.7,
                n=args.num_generations,
            )
            completions = [c.text for c in resp.choices]
        except Exception as e:
            print(f"Step {step}: vLLM generation failed ({e}), skipping", flush=True)
            continue

        # ── Rewards and group-relative advantages ─────────────────────────────
        rewards = reward_fn(completions, [truth] * args.num_generations)
        r = torch.tensor(rewards, dtype=torch.float32)  # (G,)
        adv = (r - r.mean()) / (r.std() + 1e-8)         # (G,)
        adv = adv.to(device)

        # ── Skip step if all rewards identical (no gradient signal) ───────────
        if r.std().item() < 1e-6:
            if step % log_every == 0:
                print(f"Step {step}: uniform reward ({r.mean():.2f}), skipping", flush=True)
            step += 1
            scheduler.step()
            continue

        # ── Tokenize prompt+completions ───────────────────────────────────────
        prompts_rep = [prompt] * args.num_generations
        input_ids, attn_mask, comp_mask = _tokenize_batch(prompts_rep, completions)

        if comp_mask.sum() == 0:
            print(f"Step {step}: no completion tokens after tokenization, skipping", flush=True)
            step += 1
            continue

        # ── Compute GRPO loss ─────────────────────────────────────────────────
        optimizer.zero_grad()
        loss, pg, kl = compute_grpo_loss(
            model, input_ids, attn_mask, comp_mask, adv, args.kl_coeff
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()

        step += 1
        steps_since_sync += 1

        # ── Logging ──────────────────────────────────────────────────────────
        if step % log_every == 0:
            gpu_free = torch.cuda.mem_get_info()[0] / 1e9
            print(
                f"Step {step:4d}  loss={loss.item():.4f}  pg={pg:.4f}  kl={kl:.4f}"
                f"  reward={r.mean().item():.3f}±{r.std().item():.3f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
                f"  gpu_free={gpu_free:.1f}GB",
                flush=True,
            )

        # ── Checkpoint every 100 steps ────────────────────────────────────────
        if step % 100 == 0:
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"Checkpoint saved: {ckpt_dir}", flush=True)

        # ── Sync LoRA to vLLM every N steps ──────────────────────────────────
        if steps_since_sync >= args.rollout_sync_steps:
            print(f"Step {step}: syncing LoRA to vLLM...", flush=True)
            sync_lora_to_vllm(model, tokenizer, args.vllm_server_url, rollout_path)
            steps_since_sync = 0

    print("GRPO training complete", flush=True)

    # ── Save final adapter in BF16 ────────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    st_path = Path(args.output_dir) / "adapter_model.safetensors"
    if st_path.exists():
        tensors = {}
        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k).to(torch.bfloat16)
        st_save_file(tensors, str(st_path))
        size_gb = st_path.stat().st_size / 1e9
        print(f"Saved {len(tensors)} keys ({size_gb:.2f} GB bf16) → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
