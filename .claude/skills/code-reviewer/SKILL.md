---
name: code-reviewer
description: Dispatch the code-reviewer agent in the background to review code for bugs, logic errors, security vulnerabilities, and quality issues. Pass optional args to specify scope (file path, description). Defaults to current git diff.
---

Immediately dispatch the `code-reviewer` subagent using the Agent tool with `run_in_background: true`.

- `subagent_type`: "code-reviewer"
- `prompt`: "$ARGS — if no scope provided, review the current git diff (run git diff HEAD to gather context)"
- `run_in_background`: true

Do nothing else. Do not pre-process, read files, or build context yourself — the agent has full tool access and will gather what it needs. Respond only with a single line: "Code reviewer dispatched — you'll be notified when done."
