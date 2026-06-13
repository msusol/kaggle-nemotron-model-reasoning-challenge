# CLAUDE.md

Project rules are in [.clinerules/](.clinerules/):

### Documentation framework
- [01-global.md](.clinerules/01-global.md) — `docs/` directory structure and canonical folder roles
- [02-plan-and-todo-sync.md](.clinerules/02-plan-and-todo-sync.md) — Keep `docs/plans/` and `docs/plans/TODO.md` in sync
- [03-desync-cleanup.md](.clinerules/03-desync-cleanup.md) — Handle plan/TODO desync cleanup
- [04-docs-canonical.md](.clinerules/04-docs-canonical.md) — Canonical documentation model (ADR, specs, plans, roadmap, process, investigate)
- [05-docs-investigate.md](.clinerules/05-docs-investigate.md) — Investigation log format for `docs/investigate/`
- [06-docs-plans.md](.clinerules/06-docs-plans.md) — Planning document rules for `docs/plans/`
- [07-docs-adr.md](.clinerules/07-docs-adr.md) — Architecture Decision Record rules
- [08-docs-specs.md](.clinerules/08-docs-specs.md) — Feature/subsystem specification rules
- [09-docs-process.md](.clinerules/09-docs-process.md) — Process document rules for `docs/process/`

### Coding and commit standards
- [10-commit-description.md](.clinerules/10-commit-description.md) — Commit message guidelines (Conventional Commits)
- [11-markdown-codeblocks.md](.clinerules/11-markdown-codeblocks.md) — Format bash flags on separate lines in markdown
- [12-shell.md](.clinerules/12-shell.md) — Use `zsh` shebangs and invocations

### Project-specific rules
- [13-docker-stop-failed.md](.clinerules/13-docker-stop-failed.md) — Force-stop containers when `docker stop` is denied
- [14-docker-gpu-gb10.md](.clinerules/14-docker-gpu-gb10.md) — Use `--privileged -e NVIDIA_VISIBLE_DEVICES=all` for GPU on GB10 (not `--gpus all --runtime=nvidia`)
- [15-citations.md](.clinerules/15-citations.md) — Use `[cite:N]` inline; register in `docs/plans/CITATIONS.md`; number = max existing + 1
- [16-readme-sync.md](.clinerules/16-readme-sync.md) — Update README.md layout and commands when scripts, data files, or Dockerfiles change
- [17-leaderboard.md](.clinerules/17-leaderboard.md) — Update `docs/plans/leaderboard.md` after each completed training run and validation pass
- [18-dgx-long-training-rules.md](.clinerules/18-dgx-long-training-rules.md) — Always use tmux for training; never run_in_background; broken pipe diagnosis
- [19-kaggle-notebook-workflow.md](.clinerules/19-kaggle-notebook-workflow.md) — All Kaggle changes via `kaggle kernels push`; never instruct UI edits; committed runs preferred
- [20-copy-paste-content.md](.clinerules/20-copy-paste-content.md) — Write copy/paste content (forum replies, comments, messages) to `/tmp/<slug>.md`
