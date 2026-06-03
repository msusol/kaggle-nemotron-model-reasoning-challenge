# DGX Long Training Rules — nemotron-gb10

This project trains Nemotron-3-Nano-30B on a DGX Spark (GB10, 128 GB unified memory)
using the `nemotron-gb10:latest` Docker image. Training runs take **12–16 hours**.

## Always use tmux — never run_in_background

Do NOT invoke `bash scripts/run_train.sh` with `run_in_background: true` in the Bash
tool. The background task runner holds the stdout pipe; when it dies, the log stops
updating even though Docker keeps training for hours with no visibility.

**Always tell the user to run training themselves:**

```bash
tmux new -s train
RUN_NAME=<name> bash scripts/run_train.sh
# Ctrl+B then D to detach — session survives SSH drops
# tmux attach -t train to reattach
```

The log is written to `output/train_<RUN_NAME>_<timestamp>.log` via `tee` inside
`run_train.sh`. As long as tmux keeps the shell alive, the log stays live.

To follow it from another terminal:
```bash
tail -f output/train_<RUN_NAME>_<timestamp>.log
```

## Never kill a running nemotron-gb10 container without asking

`nemotron-gb10:latest` training runs take 12–16 hours. Killing one wastes the full run.
Always confirm with the user before issuing `docker stop` or `docker kill`.

Check what is running first:
```bash
docker ps --filter "ancestor=nemotron-gb10:latest"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

## How to start a training run correctly

```bash
# From the project root, inside a tmux session:
tmux new -s train
RUN_NAME=huikang_v4r3 bash scripts/run_train.sh

# The script handles everything:
#   - pauses nginx-proxy and rnaseq-server containers
#   - GPU pre-flight check (aborts if > 1 GB stale allocations)
#   - sets VM tuning (vfs_cache_pressure, min_free_kbytes)
#   - launches nemotron-gb10:latest with --privileged -e NVIDIA_VISIBLE_DEVICES=all
#   - writes log to output/train_<RUN_NAME>_<timestamp>.log
#   - resumes paused containers on exit
```

## Diagnosing a silent run (log stopped updating)

Do NOT assume training crashed. Check in this order:

1. **Container still running?**
   ```bash
   docker ps --filter "ancestor=nemotron-gb10:latest"
   ```

2. **GPU actively computing?** (>80% = training; 0% = done or stalled)
   ```bash
   nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
   ```

3. **Process alive and accumulating CPU time?**
   ```bash
   docker exec <container_id> ps aux | grep train_lora
   ```
   Run twice 30s apart — the CPU TIME column must increase.

4. **Reattach for live output** — detach with `Ctrl+P Ctrl+Q`, never `Ctrl+C`:
   ```bash
   docker attach <container_id>
   ```

If GPU >80% and CPU time is growing: training is running fine, only the log pipe broke.

Estimate current step:
```
elapsed_seconds ÷ ~50s/step  (at seq_len=8192, batch_size=1, grad_accum=16)
```

Do not restart. Wait for the container to exit naturally; the adapter saves to
`output/adapter_<RUN_NAME>_<timestamp>/` on completion.

## If tmux is not available inside the container

Fall back to `nohup` on the host:
```bash
nohup RUN_NAME=<name> bash scripts/run_train.sh \
  > output/train_<name>_nohup.log 2>&1 &
echo $! > output/train_<name>.pid
```

Monitor: `tail -f output/train_<name>_nohup.log`
Stop: `kill $(cat output/train_<name>.pid)`