# Contributing to Ember

Thanks for your interest! Ember is small and pragmatic — contributions that keep it that
way are very welcome.

## Dev setup

```bash
git clone https://github.com/guames/ember && cd ember
python -m venv .venv && source .venv/bin/activate
pip install -e ".[vision,dev]"
```

## Before opening a PR

```bash
ruff check .          # lint
ruff format --check . # formatting
pytest                # unit tests (these run without loading any model)
```

The unit tests in [`tests/`](tests/) are intentionally **model-free** so they run in CI on
non-Apple machines. Anything that actually loads MLX weights belongs in a manual/integration
script, not the CI test suite.

## Guidelines

- Keep the server a single dependency-light process. New heavy deps should be optional
  extras (like `[vision]`).
- Match the existing style: small functions, clear names, comments only where the *why*
  isn't obvious.
- New runtime knobs go through env vars (documented in the README) and `/status`.
- Open an issue first for anything large or architectural.

## Reporting bugs

Include: macOS + chip, Python version, `pip show ember-mlx` version, your `ember.yaml`
(models), and the relevant lines from the server log.
