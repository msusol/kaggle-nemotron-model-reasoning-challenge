# Markdown Code Block Formatting

When writing bash commands inside markdown code blocks, format each flag or
argument on its own line using `\` line continuations. This ensures commands
are readable in the IDE and can be copied without terminal line-wrap issues.

## Rules

- One flag/argument per line
- `\` continuation at the end of each line except the last
- Indent continuation lines by 2 spaces
- Apply to all bash commands with more than 2 arguments
- For commands the user will run repeatedly (docker run, training launches, curl tests),
  create a script in `scripts/` and reference it instead of inlining the full command.
  Multi-line commands with `\` continuations break when copied from the Claude Code CLI.
- Long `curl` commands with JSON bodies must always go in a script — never inline in chat
  or docs. JSON payloads with embedded quotes are especially prone to shell quoting
  breakage when copied from markdown. Reference the script path instead:
  `bash scripts/test_fastapi.sh`

## Example

Bad:
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v "$PWD":/workspace -w /workspace mineral-hr-llm-gb10 python train.py --model foo --epochs 3
```

Good:
```bash
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  -v "$PWD":/workspace \
  -w /workspace \
  mineral-hr-llm-gb10 \
  python train.py \
    --model foo \
    --epochs 3
```
