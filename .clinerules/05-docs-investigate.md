# Investigation Document Rules

This page establishes formatting standards for analyst-mode investigation logs in `docs/investigate/`. Here are the key guidelines:

**Document Structure:**
Files in this directory should treat each level-2 heading (`##`) as a separate issue entry, which may contain one or multiple issues.

**Required Subsections:**
When actively investigating, add these sections if missing: Context, Investigation Checklist, Findings, Actions Taken, Resolution, and Follow-ups.

**Section Definitions:**

The "Findings" section captures "observations, evidence, confirmed causes, explicitly marked hypotheses, conclusions drawn from investigation" — but excludes changes, edits, mitigations, or commands.

"Actions Taken" documents code changes, config modifications, commands executed, mitigations, rollbacks, and verification steps — without restating findings.

"Resolution" requires a status (resolved, partially resolved, unresolved, deferred, or not reproducible) plus a brief outcome statement with verification details when relevant.

"Follow-ups" addresses remaining risks, open questions, deferred work, needed validation, and next steps.

**Editing Standards:**
Only modify actively-worked issues, avoid unnecessary rewrites of unchanged entries, and preserve existing numbered issue headings like "## 1. Some issue found."
