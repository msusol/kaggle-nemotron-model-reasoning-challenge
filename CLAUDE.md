# CLAUDE.md

Documentation, commit-message, shell, and most Kaggle-workflow conventions (citations,
leaderboard tracking, DGX tmux discipline, notebook-push workflow) are inherited from the
shared workspace-root rules — see `../CLAUDE.md` and `../.cline/rules/`. Only rules not
covered there are listed below.

The `.clinerules/` files referenced here live on the DGX Spark host this project trains
on, not on this machine — that's expected, not broken.

### Project-specific rules
- [.clinerules/13-docker-stop-failed.md](.clinerules/13-docker-stop-failed.md) — Force-stop containers when `docker stop` is denied
- [.clinerules/14-docker-gpu-gb10.md](.clinerules/14-docker-gpu-gb10.md) — Use `--privileged -e NVIDIA_VISIBLE_DEVICES=all` for GPU on GB10 (not `--gpus all --runtime=nvidia`)
- [.clinerules/16-readme-sync.md](.clinerules/16-readme-sync.md) — Update README.md layout and commands when scripts, data files, or Dockerfiles change
- [.clinerules/20-copy-paste-content.md](.clinerules/20-copy-paste-content.md) — Write copy/paste content (forum replies, comments, messages) to `/tmp/<slug>.md`
- [.clinerules/21-submission-packaging.md](.clinerules/21-submission-packaging.md) — Run `bash scripts/package_submission.sh` from host; script handles Docker internally; never wrap in extra docker run/exec
