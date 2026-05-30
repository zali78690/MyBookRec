---
name: code-simplifier
description: Dispatch the code-simplifier agent in the background to simplify and refine recently modified code while preserving functionality. Pass optional args to specify files or scope.
---

Immediately dispatch the `code-simplifier` subagent using the Agent tool with `run_in_background: true`.

- `subagent_type`: "code-simplifier"
- `prompt`: "$ARGS — if no scope provided, simplify recently modified files (run git diff HEAD --name-only to identify them)"
- `run_in_background`: true

Do nothing else. Do not pre-process, read files, or build context yourself — the agent has full tool access and will gather what it needs. Respond only with a single line: "Code simplifier dispatched — you'll be notified when done."
