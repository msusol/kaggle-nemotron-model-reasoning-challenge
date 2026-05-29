# Feature and Subsystem Specification Rules

This document outlines authoring guidelines for specifications in `docs/specs/`, distinguishing them from architectural decision records (ADRs) and implementation plans.

## Key Distinctions

Specs capture design intent—what to build and why—and evolve as designs mature. Unlike ADRs, which are immutable decisions, specs are living documents. Unlike plans, specs focus on design rather than execution sequencing.

## When to Write a Spec

Create a spec when a feature requires coordinated design before work begins, when an existing feature undergoes redesign affecting its public interface or behavior, when planning documents accumulate significant design rationale, or when multiple teams must align on integration boundaries.

Do not create specs for routine refactoring, individual architectural choices (use ADRs instead), or task sequencing (use plans instead).

## Minimal Format

New specs should start lean with these core sections: title, summary, goals, design, and optional open questions. The guidance emphasizes "tight and decision-oriented" prose, using diagrams only when they clarify better than text.

## Expansion Criteria

Expand specs only when features touch multiple systems, meaningful tradeoffs exist, risks require tracking, interfaces need precise documentation, or implementation reveals unknowns.

## Evolution Rules

Edit specs in place rather than creating versioned copies. Extract structural decisions into ADRs when their rationale outlives the feature. Move execution details to corresponding plan documents. Once features stabilize, update specs to reflect actual implementation rather than freezing them at the design stage.
