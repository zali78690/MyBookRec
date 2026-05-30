---
name: code-simplifier
description: Spawn the code-simplifier agent to simplify and refine recently modified code for clarity, consistency, and maintainability while preserving all functionality. Pass optional args to specify files or scope.
---

# Code Simplification

Spawn the `code-simplifier` subagent using the Agent tool with `subagent_type: "code-simplifier"`.

## Prompt to pass

Build a prompt from this template:

```
Simplify and refine the following scope: {args if provided, otherwise "recently modified files in the current git diff"}

Focus on: removing unused imports/variables, improving clarity, applying Pythonic patterns (comprehensions, f-strings, pathlib, enumerate/zip), consolidating duplicate logic, and reducing unnecessary nesting.

Preserve all functionality exactly. Do not add features or change behavior.
```

## Steps

1. If no args were passed, run `git diff HEAD --name-only` to identify recently modified files and include them as context.
2. Spawn the agent: `Agent({ subagent_type: "code-simplifier", prompt: <filled template> })`
3. Present the agent's changes or recommendations to the user.
