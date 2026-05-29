# Global Project Workflow Overview

This document establishes conventions for managing documentation in projects using a hierarchical `docs/` directory structure.

## Key Resolution Process

The workflow prioritizes finding the **nearest** `docs/` root by checking the current directory first, then walking up toward the repository root. "Stop at the first directory that contains `docs/` — that is the **docs root** for this context." If no docs folder exists anywhere, one should be created at the repository root level.

## Canonical Directory Roles

The framework designates specific purposes for subdirectories within the resolved docs root:

- **docs/specs/** - Houses feature and subsystem design documentation
- **docs/plans/** - Contains implementation plans and task sequencing guidance
- **docs/roadmap/** - Manages time-based planning and initiative priorities
- **docs/investigate/** - Stores investigation logs and issue analysis
- **docs/process/** - Holds reusable workflow and operational procedures

## Primary Purpose

This structure enables independent projects or sub-projects to maintain separate documentation hierarchies without parent-level overrides, treating the resolved `docs/` location as "the canonical home for long-lived project documentation."
