---
paths:
  - "plans/**/*.md"
  - "TODO.md"
---

# Plan implementation and TODO.md synchronization

- Always locate the root-level `TODO.md` before making implementation changes.
- If `TODO.md` does not exist, create it at the project root with the heading `# TODO`.
- All tasks in `TODO.md` must use GitHub-style checkboxes:
  - Open task: `- [ ] Task description`
  - Completed task: `- [x] Task description`

## Multiple plans

All plan files live in `plans/`. Each plan maps to its own `TODO.md` section:

| Plan file              | TODO.md section  |
|------------------------|------------------|
| `plans/dspy.md`        | `## DSPy Plan`   |
| `plans/raft_plan.md`   | `## RAFT Plan`   |

New plans go in `plans/` and get a new `## <Plan Name>` section and `### <Plan Name>` group in Next steps.

- Each plan has its own top-level section in `TODO.md`. Do not merge tasks across sections.
- When a task is completed, mark it `[x]` only in the section that owns it.
- When adding tasks derived from a plan file, place them under the correct section.

## Deriving tasks from plans

- Scan each plan file for bullet lists and concrete section headings.
- For each actionable item, ensure there is a corresponding checkbox entry in `TODO.md`
  under the matching plan section.
- Keep TODO text concise but traceable to the plan.

## Next steps section

- `TODO.md` must end with a `## Next steps` section.
- Next steps must be grouped by plan using `### DSPy Plan` and `### RAFT Plan` subheadings.
- Each item is numbered sequentially within its group.
- An item belongs to the plan whose work it advances — do not mix plans within a group.
- When a next step is completed or deferred, remove or demote it and renumber.
- Deferred items stay in Next steps with a short note: `— deferred, <reason>`.
