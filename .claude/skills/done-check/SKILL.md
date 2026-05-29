---
name: done-check
description: Run the four definition-of-done gates for retina-screen (ruff check, ruff format check, mypy strict, pytest) and report pass/fail per gate. Use this before claiming a change is complete, after finishing a module, or when verifying that the codebase is still green.
---

Run the four gates in order. Capture stdout/stderr for each. Continue on failure (don't stop at the first red gate — the user wants the full picture).

```
uv run ruff check src
uv run ruff format --check src
uv run mypy
uv run pytest
```

Report back in this exact shape:

```
done-check
  ruff check       : PASS | FAIL (N issues)
  ruff format      : PASS | FAIL (N files would reformat)
  mypy (strict)    : PASS | FAIL (N errors in M files)
  pytest           : PASS | FAIL (N failed, M passed)
```

If any gate fails, include the first 20 lines of its output under the summary so the user can see what broke without scrolling.

If a gate cannot run because the project isn't scaffolded yet (no `pyproject.toml`, no `src/`), report `SKIP (not scaffolded)` for that gate rather than treating it as a failure.

Do not attempt to fix failures unless explicitly asked. Reporting is the job; fixing is a separate request.
