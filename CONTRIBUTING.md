# Contributing

## No CLA

Contributions are accepted under the Apache-2.0 licence the project already carries. There is no
contributor licence agreement, no copyright assignment, and no separate signature step — opening a
pull request is enough. Apache-2.0 already includes a patent grant from contributors (section 3),
which is the protection a CLA is usually reached for, and asking people to sign paperwork to fix a
typo costs more than it returns.

If that ever has to change, it will change for future contributions only and be announced in a
release note, never applied retroactively to work already merged.

## Getting set up

```
uv sync --extra dev
uv run python scripts/seed_demo.py     # small SQLite demo database under demo/
uv run pytest -q
```

Tests run without any LLM credentials or network access: anything needing a model is behind a fake
or an `importorskip`. If a test of yours needs a live endpoint, it belongs behind the same seam.

The warehouse benchmark scripts need optional drivers — `uv sync --extra warehouse` — and real
Snowflake or Databricks credentials. They are not part of the default test run.

## What makes a good pull request

- **A test that fails before the change and passes after.** For a bug, the test should reproduce
  the defect rather than assert the fixed behaviour in the abstract.
- **A comment that says why, where the why isn't obvious.** This codebase is unusually heavy on
  rationale comments, deliberately: several of them exist because a plausible-looking change
  silently broke a guarantee, and the comment is what stops it being reintroduced. Match that
  standard rather than stripping it.
- **Fail-closed, always.** No policy, no grants, no snapshot means no data. A change that makes
  the engine answer where it previously deferred needs to argue for itself explicitly.

## Reporting a security issue

Do not open a public issue for a vulnerability. Email **security@agenticfabriq.com** with enough
detail to reproduce. We will confirm receipt and tell you when a fix ships.

For anything that is a limitation rather than a vulnerability — the engine answering something it
should refuse, an access check with a gap, a grader disagreeing with itself — a public issue is the
right place, and there are already several open. We would rather those be discussed in the open
than discovered quietly.

## Reporting a wrong answer

The useful report includes the question, the SQL the engine produced, the SQL you expected, and the
schema (or enough of it to reproduce). `mnemiq ask` prints the SQL and the tables it read; the trace
on `/v1/ask` carries the enrichment version too, which matters because the same question can resolve
differently against a different snapshot.
