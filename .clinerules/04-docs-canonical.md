# Canonical Documentation Model

This document establishes a structured approach to organizing long-lived project documentation within a `docs/` directory. It designates `docs/index.md` as the primary navigation entry point and defines specific folder roles:

- **ADR folder**: Captures architectural decisions, tradeoffs, and superseded choices
- **Specs folder**: Contains feature specifications and subsystem design documentation
- **Plans folder**: Houses feature-level implementation breakdowns and execution guidance
- **Roadmap folder**: Manages time-based planning and initiative sequencing
- **Process folder**: Documents reusable operational guidance and team workflows
- **Investigate folder**: Stores investigation logs and debugging trails

The document includes a practical classification principle: "If a document is expected to matter in 6 months, it should probably live in `docs/`."

It further clarifies distinctions between document types—ADRs capture immutable decisions, specs describe what to build and why, and plans outline how and when to execute. For substantial decisions affecting multiple documents, the guidance suggests creating an ADR that other documents reference.

The file also addresses legacy locations, recommending migration of durable content from older plan files into the new structured folders rather than maintaining parallel documentation.
