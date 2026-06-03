# Remote DGX training rules

## When starting long-running training jobs

- You are working on a DGX server over SSH. SSH connections and VS Code / Claude sessions may drop unexpectedly.
- Never start long-running training jobs directly in a plain interactive shell tied to the SSH connection.
- Instead, ALWAYS start training runs inside a tmux session running inside the Docker container.

### tmux pattern inside container

- When the user asks you to start a training run (for example, running `scripts/run_train.sh`),
  do NOT run the script directly.
- Instead, run this pattern, substituting the actual container name and RUN_NAME:

  ```bash
  docker exec -it <container_name> tmux new -d -s train \
    "RUN_NAME=<run_name> bash scripts/run_train.sh"
  ```

- After starting the training session, print clear instructions for how to reattach later, e.g.:

  ```text
  To reattach after an SSH drop:
    ssh <dgx-host>
    docker exec -it <container_name> tmux attach -t train
  ```

### If tmux is not available

- If `tmux` is not installed or the command fails, fall back to using `nohup` plus log files:

  ```bash
  nohup RUN_NAME=<run_name> bash scripts/run_train.sh \
    > logs/<run_name>.out 2>&1 &
  echo $! > logs/<run_name>.pid
  ```

- After starting the process with `nohup`, tell the user exactly how to monitor and manage it:

  ```text
  To follow logs:
    tail -f logs/<run_name>.out

  To stop the run:
    kill $(cat logs/<run_name>.pid)
  ```

### General constraints

- Assume SSH connections may drop at any time; prefer patterns that keep processes alive independently of the SSH session.
- After starting a job, always confirm:
  - The exact command you ran.
  - Where logs will go (tmux session name or logfile path).
  - How the user can resume monitoring after reconnecting.

## Never use run_in_background for training scripts

Do NOT invoke `bash scripts/run_train.sh` (or any long Docker training command) with
`run_in_background: true` in the Bash tool. The background task runner holds the stdout
pipe; when killed (timeout, session end, user interrupt), the pipe breaks and the log
file stops updating — even though training continues inside Docker for hours unmonitored.

Always instruct the user to run training themselves in tmux (see pattern above).

## Never kill a running training container without asking

Before issuing `docker stop` or `docker kill` on a nemotron-gb10 container, confirm
with the user. Training runs are 12–16 hours; killing one wastes the full run.

Check what's running first:
```bash
docker ps --filter "ancestor=nemotron-gb10:latest"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

## Diagnosing a silent/stalled run

If the log file stops updating, do NOT assume training crashed. Check in order:

1. **Container still running?**
   ```bash
   docker ps --filter "ancestor=nemotron-gb10:latest"
   ```
2. **GPU actively computing?** (>80% = training; 0% = stalled or done)
   ```bash
   nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
   ```
3. **Process still alive and accumulating CPU time?**
   ```bash
   docker exec <id> ps aux | grep train_lora
   ```
   Run twice 30s apart — CPU time column should increase.

4. **Attach for live output** (detach with `Ctrl+P Ctrl+Q` — never `Ctrl+C`):
   ```bash
   docker attach <container_id>
   ```

If GPU >80% and CPU time growing: training is running, log pipe just broke. Estimate
current step: `(elapsed_seconds) / ~50s_per_step`. Do not restart.