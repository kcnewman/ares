# Contributing to ARES

Thank you for contributing. This document defines the engineering standards for changes in this repository.

## Development Setup

1. Install Python `3.12.x`.
2. Install `uv`.
3. Sync dependencies:

```bash
uv sync --dev
```

## Project Conventions

- Keep changes scoped to a single concern when possible.
- Prefer explicit, typed, and testable logic in pipeline stages.
- Do not hardcode runtime paths or secrets; use config and environment variables.
- Keep artifact contracts stable unless schema changes are intentional and documented.

## Branch and Commit Guidelines

- Create feature branches from `main`.
- Use clear commit messages that describe behavior changes.
- Reference impacted stage(s) in commit descriptions for pipeline-related work.

## Testing Requirements

Run tests before opening a PR:

```bash
uv run pytest -q
```

When relevant, also run:

- `uv run python main.py` for full pipeline verification
- API smoke test on `/health` and `/predict`
- Batch inference smoke test with `src/ares/pipeline/inference.py`

## Documentation Requirements

Any behavior, interface, config, or artifact contract change must update docs in the same PR:

- Root `README.md` for high-level usage
- `ARCHITECTURE.md` for structural changes
- Relevant file(s) in `docs/`

## Pull Request Checklist

- [ ] Tests pass locally.
- [ ] Docs updated for behavior/config/API changes.
- [ ] No secrets or credentials committed.
- [ ] New config keys documented in `docs/CONFIGURATION.md`.
- [ ] API contract updates documented in `docs/API.md`.

## Reporting Issues

Include enough context to reproduce:

- command run
- expected result
- actual result
- relevant logs (`logs/running_logs.log`)
- related artifact paths
