# Architecture Decision Record (ADR) Rules

This document establishes guidelines for managing Architecture Decision Records in the `docs/adr/` directory.

## Key Principles

ADRs serve as append-only historical records capturing individual architectural decisions along with their context, alternatives, and consequences—not as evolving design documents.

## When to Document

Create an ADR when a decision significantly shapes the architecture, involves meaningful tradeoffs, will likely be revisited, or when a spec cannot adequately contain the rationale. Skip ADRs for routine implementation choices, local refactors, style preferences, or temporary experiments.

## File Management

Files follow the pattern `NNNN-kebab-case-title.md` using sequential zero-padded numbers. Numbers are never reused, even when an ADR is superseded or rejected.

## Required Sections

Every ADR must include:
- Title with number and decision name
- Status (Proposed, Accepted, Superseded, Deprecated, or Rejected)
- Context describing forces and constraints
- Decision stating what was chosen
- Consequences noting outcomes and tradeoffs

Optional sections include alternatives considered, related decisions, and references.

## Immutability Principle

Once an ADR reaches "Accepted" status, only typographical corrections are permitted. Material changes require a new superseding ADR. Proposed ADRs may be edited freely until finalization.

## Documentation Integration

ADRs should be indexed in related documentation, and findings from investigation or planning documents should be extracted into ADRs rather than remaining scattered across multiple files.
