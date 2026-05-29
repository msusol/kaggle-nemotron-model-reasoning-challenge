# Plan Implementation and TODO.md Synchronization Rules

This document establishes guidelines for maintaining consistency between implementation plans in `docs/plans/*.md` and the central task checklist at `docs/plans/TODO.md`.

## Key Requirements

**Checklist Creation:** "Always locate `docs/plans/TODO.md` before making implementation changes." If plan files exist but the TODO file doesn't, create it immediately before starting work.

**Task Format:** All entries use GitHub-style checkboxes: `- [ ] Task description` for open items and `- [x] Task description` for completed ones.

**Organization:** Each plan file gets its own dedicated section in TODO.md. The document provides a mapping table showing how files correspond to TODO sections.

## Workflow Guidance

When deriving tasks from plans, scan for bullet lists and actionable headings, then add matching checkbox entries under the appropriate section. Keep descriptions concise but traceable to their source plan.

The TODO.md file should conclude with a "Next steps" section organized by plan. Items should be numbered when sequence matters, and deferred items noted with brief explanations before removal.
