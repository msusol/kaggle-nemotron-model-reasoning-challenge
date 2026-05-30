#!/usr/bin/env zsh
# Pause and resume non-training Docker containers around a training run.
#
# Usage:
#   bash scripts/services.sh pause    # stop all running containers except the training image;
#                                     # save names to .paused_containers
#   bash scripts/services.sh resume   # restart every container listed in .paused_containers
#
# The script is agnostic to which containers are running — it snapshots whatever
# is up at pause time and restores exactly that set on resume.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
STATE_FILE="${WORKSPACE}/.paused_containers"
TRAINING_IMAGE="nemotron-gb10"

pause() {
  local running
  running=$(docker ps --format '{{.Names}}\t{{.Image}}' \
    | awk -v img="$TRAINING_IMAGE" '$2 !~ img {print $1}')

  if [[ -z "$running" ]]; then
    echo "No non-training containers currently running."
    return
  fi

  echo "$running" > "$STATE_FILE"
  echo "Stopping:"
  while IFS= read -r name; do
    echo "  $name"
    docker stop "$name" > /dev/null
  done <<< "$running"
  echo "Saved container list to ${STATE_FILE}"
}

resume() {
  if [[ ! -f "$STATE_FILE" ]]; then
    echo "No saved container list found at ${STATE_FILE} — nothing to resume."
    return
  fi

  echo "Restarting:"
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    echo "  $name"
    docker start "$name" > /dev/null
  done < "$STATE_FILE"

  rm "$STATE_FILE"
  echo "Done. Removed ${STATE_FILE}"
}

case "${1:-}" in
  pause)  pause  ;;
  resume) resume ;;
  *)
    echo "Usage: $0 {pause|resume}"
    exit 1
    ;;
esac
