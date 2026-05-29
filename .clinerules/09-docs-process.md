# Process Document Rules

This file establishes guidelines for creating and maintaining operational documentation in the `docs/process/` directory.

## Key Provisions

**Purpose**: Files in `docs/process/` serve as "reusable operational guides — stable enough to live in the repo, likely to be followed repeatedly, and owned by the team rather than a single task."

**Creation Triggers**: Process documents should be created when new runnable artifacts (scripts, Docker services, CLI tools, Makefile targets, scheduled jobs) are added, recurring workflows requiring multiple steps emerge, setup procedures from investigations warrant reuse, or existing docs need updates.

**What to Exclude**: One-off commands covered by code comments or single-task setup steps should not become process documents.

**Naming**: Use descriptive kebab-case filenames directly under `docs/process/`, with one document per distinct workflow.

**Required Sections**:
- Title and workflow description
- Prerequisites listing dependencies
- Numbered, executable steps with exact commands
- Optional: expected output, troubleshooting, and related documentation links

**Maintenance**: Process documents should be updated whenever the underlying process changes, with updates committed alongside any artifact modifications to prevent documentation drift.
