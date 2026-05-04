# CLAUDE.md

Project rules are in [.clinerules/](.clinerules/):

- [01-global.md](.clinerules/01-global.md) — Check `plans/*.md`, `TODO.md`, and `README.md` before significant work
- [02-plan-and-todo-sync.md](.clinerules/02-plan-and-todo-sync.md) — Keep plans and TODO.md in sync
- [03-desync-cleanup.md](.clinerules/03-desync-cleanup.md) — Handle plan/TODO desync cleanup
- [10-commit-description.md](.clinerules/10-commit-description.md) — Commit message guidelines
- [11-markdown-codeblocks.md](.clinerules/11-markdown-codeblocks.md) — Format bash flags on separate lines in markdown
- [12-docker-stop-failed.md](.clinerules/12-docker-stop-failed.md) — Force-stop containers when `docker stop` is denied
- [13-docker-gpu-gb10.md](.clinerules/13-docker-gpu-gb10.md) — Use `--privileged -e NVIDIA_VISIBLE_DEVICES=all` for GPU on GB10 (not `--gpus all --runtime=nvidia`)
- [14-citations.md](.clinerules/14-citations.md) — Use `[cite:N]` inline; register in `plans/CITATIONS.md`; number = max existing + 1
- [15-readme-sync.md](.clinerules/15-readme-sync.md) — Update README.md layout and commands when scripts, data files, or Dockerfiles change
