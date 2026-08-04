# mnemiq

An open-source engine that answers natural-language questions over your database — and is
built to be *trusted*: it defers rather than guess, and every answer carries an auditable trace.

## The idea

A trustworthy data agent is a systems problem, not a model problem. mnemiq pairs a **stochastic
proposer** (an LLM writes candidate SQL) with a **deterministic decider** (shape, authorization,
transpile, and `EXPLAIN` against the real source). If the data can't answer the question, the
engine says so instead of inventing a number.

## Why trust it

- **It never invents an answer.** Across every evaluation run, zero questions the data could not
  answer were answered anyway — they were deferred, with an explanation.
- **Two independent locks on access.** The model is never *shown* a table the caller may not see
  (access-scoped retrieval), and the decider re-checks every referenced object against grants
  before anything runs. The database's own permission error is never the control.
- **Fail-closed everywhere.** No policy, no grants, no snapshot → no data.
- **Every answer is auditable.** The engine emits a stable trace — the tables used, the
  enrichment version, timing — alongside the SQL it ran.

## What the numbers say

Measured, not asserted. Grading is result-based: a different query that returns the right facts
passes; the harness reports both exact-match and got-the-facts accuracy.

| corpus | accuracy |
|---|---|
| ACME (in-domain, 30 cases) | 100% |
| BIRD mini-dev (500 questions, 11 unseen schemas) | 40.5% exact-match · 52.2% got-the-facts |

The single-shot engine is competitive with strong frontier one-shot baselines on BIRD
(GPT-4o 34%, Claude 3.7 41%). The distance to leaderboard pipelines is added machinery —
candidate selection, verification — which is on the roadmap, not a difference in the core.

## Quickstart

```
uv sync
export MNEMIQ_LLM_BASE_URL=... MNEMIQ_LLM_API_KEY=... MNEMIQ_LLM_MODEL=...
export MNEMIQ_PG_DSN=postgresql://user:pass@host:5432/db
uv run mnemiq enrich      # profile + describe the schema, save a snapshot
uv run mnemiq build       # index it for retrieval
uv run mnemiq ask "how many claims are there?"
```

Access is fail-closed: set `MNEMIQ_AUTHZ_PATH` to a policy file granting objects to roles, and
pass `--roles analyst`. Without a policy, the engine grants nothing and defers.

## Use it from a browser (workbench)

```
cd workbench && pnpm install && pnpm build
uv run mnemiq serve --http     # http://127.0.0.1:8080
```

One process serves both the workbench and the HTTP API — `POST /v1/ask` (JSON),
`POST /v1/chat` (SSE, [AG-UI](https://github.com/ag-ui-protocol/ag-ui) event
vocabulary), `GET /v1/schema`. Every answer shows the SQL that produced it and the
tables it read; a question the data cannot support comes back as a stated reason,
not a guess. See [`workbench/README.md`](workbench/README.md).

## Use it from an AI agent (MCP)

`uv run mnemiq serve` exposes two read-only, access-scoped tools over stdio — `db_read(question)`
(answer + SQL + trace) and `get_schema()`. Point any MCP client at it:

```json
{ "mcpServers": { "mnemiq": { "command": "mnemiq", "args": ["serve"] } } }
```

## Architecture

```
source → enrichment (profiling + LLM descriptions + foreign keys + glossary)
       → access-scoped retrieval → generation
       → deterministic decider (shape / access / transpile / EXPLAIN)
       → execution → synthesis → trace
```

## Grounding code columns

Short codes like `E11` or `NC-17` mean nothing on their own. mnemiq grounds them from the data
itself, from a standard code system (an ontology in TTL/SKOS/OWL), or from a hand-written operator
dictionary — always **grounded-or-bare**, never guessed. See [docs/grounding.md](docs/grounding.md).

## Open core

The engine is Apache-2.0 and stands alone. Two commercial planes build on it and are **not**
open source: **Verity** (trust, grading, drift) and **Agentic Fabriq** (governance, per-group
authorization, audit). They align to the open contract (`mnemiq-contract`) by reference; no paid
capability lives in the open core.

## Status

v0.1: the full read path, evaluated on ACME and BIRD. Roadmap: cross-source federation, tiered
modes, the governed write plane, multi-replica deployment, local models, and the self-maintaining
loops.
