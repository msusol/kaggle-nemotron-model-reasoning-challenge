---
paths:
  - "plan.md"
  - "plans/**/*.md"
  - "TODO.md"
---

# Handling heavy desynchronization between plans and TODO.md

When `plan.md` / `plans/**/*.md` and `plans/TODO.md` appear heavily out of sync, do not guess or silently paper over inconsistencies. Instead, follow this cleanup workflow.

## 1. Detecting heavy desync

Treat the situation as a potential heavy desync if you notice any of the following:

- Many tasks present in plan files but missing from `plans/TODO.md`.
- Many tasks present in `plans/TODO.md` but no longer present in any plan.
- Widespread mismatches between checkbox states in `TODO.md` vs the current reality of the implementation.
- Plan sections that appear substantially rewritten or restructured compared to their corresponding sections in `TODO.md`.

## 2. Pause and confirm with the user

Before performing large-scale edits:

- Inform the user that plan files and `TODO.md` appear out of sync.
- Propose running a one-time “sync cleanup” to reconcile them.
- Ask the user to confirm:
  - Whether the plan files should be treated as the source of truth, or
  - Whether `TODO.md` should be treated as the source of truth, or
  - Whether they want a conservative merge that preserves both and requires manual review.

## 3. Source-of-truth strategies

If the user confirms that **plans are the source of truth**:

- For each relevant plan file:
  - Rebuild or update its corresponding section in `TODO.md` to mirror the current plan tasks.
  - Remove obviously stale `TODO.md` entries that refer to tasks no longer present in any plan.

If the user confirms that **TODO.md is the source of truth**:

- For each plan file:
  - Update or recreate its task sections so they match the tasks and status currently listed in `TODO.md` for that plan.
  - Optionally add a short note in each affected plan indicating that it has been synchronized from `TODO.md`.

If the user chooses a **conservative merge**:

- Do not delete tasks outright.
- Instead:
  - Create a `## Review needed` or `## Unmapped tasks (manual review)` section in `plans/TODO.md`.
  - Move ambiguous or stale items there with a note that they require user decisions.

## 4. Mechanical cleanup steps

When performing a sync cleanup (after user confirmation):

- Enumerate all plan files (`plan.md` and `plans/**/*.md`) and collect their task lists.
- Enumerate all sections and checkbox items in `plans/TODO.md`.
- For each plan file:
  - Ensure there is a corresponding section in `TODO.md` (for example `## <plan-file-name> – <summary>`).
  - Align tasks and their checkbox states between the plan and TODO section according to the chosen source-of-truth strategy.
- For `plans/TODO.md` items that do not map cleanly to any current plan:
  - Either remove them (if plans are the source of truth), or
  - Move them into the review section for later manual decisions.

## 5. Documentation and follow-up

After completing a heavy desync cleanup:

- Add a short note near the top of `plans/TODO.md` describing:
  - When the cleanup occurred.
  - Which source-of-truth strategy was used.
- Optionally add a short note in each modified plan file indicating that it has been synchronized with `TODO.md`.
- For subsequent sessions:
  - Resume the normal synchronization behavior defined in the other `.clinerules/` files.
  - Avoid repeating heavy cleanup unless a similar large desync is detected again.

## 6. Safety constraints

- Do not delete large blocks of tasks from `plans/TODO.md` or plan files without:
  - Explaining the change to the user, and
  - Getting explicit confirmation for the chosen strategy.
- When in doubt about whether an item is obsolete or duplicated:
  - Prefer preserving it under a “Review needed” or “Unmapped tasks” section instead of deleting it.
