---
name: code-reviewer
description: Spawn the code-reviewer agent to review code for bugs, logic errors, security vulnerabilities, and quality issues. Pass optional args to specify scope (file path, git range, or description). Defaults to reviewing the current git diff.
---

# Code Review

Spawn the `code-reviewer` subagent using the Agent tool with `subagent_type: "code-reviewer"`.

## Prompt to pass

Build a prompt from this template — fill in what you know, omit what you don't:

```
Review scope: {args if provided, otherwise "the current git diff (git diff HEAD)"}

Context: {brief description of what changed or what to focus on, if known}

Return a structured report grouped by severity (Critical / Important). Include file path, line number, confidence score, and a concrete fix suggestion for each finding. Skip issues below confidence 80.
```

## Steps

1. If no args were passed, run `git diff HEAD` to understand what changed and use that as context.
2. Spawn the agent: `Agent({ subagent_type: "code-reviewer", prompt: <filled template> })`
3. Present the agent's findings to the user, preserving its structure.
