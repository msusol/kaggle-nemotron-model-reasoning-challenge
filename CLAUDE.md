# CLAUDE.md

Global docs/planning rules and workspace-level Kaggle rules (competition-agnostic
`kaggle-*` conventions) are inherited automatically — the former from the globally
deployed `docs` plugin, the latter from `../CLAUDE.md` at the Kaggle workspace root
(Claude Code walks up the directory tree loading every `CLAUDE.md` it finds). No
need to duplicate either here.

This project's own rules — specific to this competition (Nemotron-3-Nano-30B on the
DGX Spark GB10) — live in `.claude/rules/` and load automatically via the
`@`-imports below.

| File | Covers |
|------|--------|
| `13-docker-stop-failed.md` | Force-stop containers when `docker stop` is denied |
| `14-docker-gpu-gb10.md` | GB10 GPU flags and aarch64 CUDA extension build issues |
| `15-citations.md` | `[cite:N]` inline citation format, registered in `docs/plans/CITATIONS.md` |
| `16-readme-sync.md` | Keep `README.md` in sync with scripts/data/Dockerfiles |
| `17-leaderboard.md` | Update `docs/plans/leaderboard.md` after each training/validation run |
| `18-dgx-long-training-rules.md` | tmux discipline, never `run_in_background` for training, diagnosing silent runs |
| `19-kaggle-notebook-workflow.md` | This project's kernel push/GPU-selection workflow (RTX Pro 6000 specifics) |
| `20-copy-paste-content.md` | Write copy/paste content (forum replies, comments) to `/tmp/<slug>.md` |
| `21-submission-packaging.md` | This project's `package_submission.sh` / `docker exec` packaging workflow |

@.claude/rules/13-docker-stop-failed.md
@.claude/rules/14-docker-gpu-gb10.md
@.claude/rules/15-citations.md
@.claude/rules/16-readme-sync.md
@.claude/rules/17-leaderboard.md
@.claude/rules/18-dgx-long-training-rules.md
@.claude/rules/19-kaggle-notebook-workflow.md
@.claude/rules/20-copy-paste-content.md
@.claude/rules/21-submission-packaging.md
