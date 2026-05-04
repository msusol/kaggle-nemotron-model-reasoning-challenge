---
paths:
  - "README.md"
  - "scripts/**"
  - "data/**"
  - "Dockerfile*"
---

# README.md synchronization

Update `README.md` whenever any of the following change:

| Change | README section to update |
|---|---|
| New or removed script in `scripts/` | Repository layout + Commands (add/remove step) |
| New or removed runner script (`run_*.sh`) | Commands section — add a numbered step with usage |
| New or removed data file in `data/` | Repository layout `data/` block |
| New `Dockerfile.*` variant | Repository layout + Build section |
| `max_memory`, `max_new_tokens`, or other tunable in a script | Relevant Commands note |

## What to keep accurate

- **Repository layout** — every file or directory listed must exist; remove entries for deleted files.
- **Commands** — step numbers must be sequential; instructions must match the actual script behavior.
- **Notes** (memory, timing, OOM warnings) — must reflect current script values, not historical ones.

## What not to add

- Do not add step-by-step implementation detail that belongs in `plans/*.md`.
- Do not document internal functions or class APIs — README covers usage, not internals.
- Do not create a new README section for every minor script change; batch related changes into one update.
