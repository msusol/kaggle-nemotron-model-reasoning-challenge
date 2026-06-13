# Rule 20 — Copy/Paste Content Goes to /tmp

When generating content intended for the user to copy/paste into an external system
(forum replies, Kaggle discussion posts, GitHub comments, Slack messages, emails, etc.),
**always write the content to a `/tmp` file** in addition to (or instead of) displaying
it inline.

## Requirements

- Format: Markdown (`.md` extension)
- Path: `/tmp/<descriptive-slug>.md` — name should reflect the destination or topic
- After writing, tell the user the file path on one line so they can open it directly
- Content in the file should be ready to paste as-is: no wrapper text, no meta-commentary,
  no "here is the reply:" preamble — just the content itself

## Examples of content that triggers this rule

- Kaggle discussion replies
- GitHub issue or PR comments
- README badges or snippets for external repos
- Slack / email drafts
- Any message prefaced with "here's a draft reply" or "write a thread reply"

## Example

```zsh
# Bad — content only shown inline, hard to copy
echo "Reply text here..."

# Good — written to /tmp for easy copy/paste
Write /tmp/kaggle-reply-684251.md  ← use the Write tool, then tell the user:
# "Written to /tmp/kaggle-reply-684251.md"
```
