#!/usr/bin/env bash
# Start JupyterLab in a detached tmux session on the DGX Spark.
# Access at http://192.168.68.54:8888  (password: jupyter)

SESSION="jupyter"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "source /home/msusol/miniconda3/etc/profile.d/conda.sh && \
   conda activate base && \
   jupyter lab --config /home/msusol/.jupyter/jupyter_server_config.py 2>&1 | tee /tmp/jupyter.log"

echo "JupyterLab started in tmux session '$SESSION'."
echo "  URL:      http://192.168.68.54:8888"
echo "  Password: jupyter"
echo "  Logs:     /tmp/jupyter.log"
echo "  Attach:   tmux attach -t $SESSION"
echo "  Stop:     tmux kill-session -t $SESSION"
