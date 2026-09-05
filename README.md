# mnemiq

An open-source engine that answers natural-language questions over your database — and is
built to be *trusted*: it defers rather than guess, and every answer carries an auditable trace.

## The idea

A trustworthy data agent is a systems problem, not a model problem. mnemiq pairs a **stochastic
proposer** (an LLM writes candidate SQL) with a **deterministic decider** (shape, authorization,
transpile, and `EXPLAIN` against the real source). If the data can't answer the question, the
engine says so instead of inventing a number.

## Why trust it

- **It refuses rather than guesses.** When the data cannot answer a question, the engine states the
  reason instead of inventing a number — every unanswerable ACME case, and 26 of 30 results on an
  adversarial refusal set built to tempt it. The residual is instructive: the case it answers is
  *"the lifetime value of our average customer"*, where a plausible-looking derivation exists over
  the columns and the business definition does not. Refusal is measured here, not asserted.
- **Two independent locks on access.** The model is never *shown* a table the caller may not see
  (access-scoped retrieval), and the decider re-checks every referenced object against grants
  before anything runs. The database's own permission error is never the control.
- **Fail-closed everywhere.** No policy, no grants, no snapshot → no data.
- **Every answer is auditable.** The engine emits a stable trace — the tables used, the
  enrichment version, timing — alongside the SQL it ran.

## What the numbers say

Measured, not asserted. Grading is result-based: a different query that returns the right facts
passes; the harness reports both exact-match and got-the-facts accuracy.

| corpus | exact-match | got-the-facts | run |
|---|---|---|---|
| ACME (in-domain, 25 answerable of 30) | 88.0% | 100.0% | `plan15/enrichment-on.json` |
| BIRD mini-dev (487 answerable, 11 unseen schemas) | 48.9% | 63.2% | `gpt55-duckdb-pg.jsonl` |
| Spider 2.0-lite (135 local of 547, 30 schemas) | 37.0% | 58.5% | `spider2-full-k24.jsonl` |

Each row names ONE run and both of its numbers come from that run. Those artifacts are not
distributed — `eval-reports/` is gitignored — so the names identify a run in our records rather
than a file you can open.

**The denominators differ by row, and for different reasons.** Both rates are taken over the
*answerable* cases (`correct + correct_facts + wrong + deferred_wrongly + error`), but what falls
outside that is not one thing:

- **ACME** — 25 of 30. The five missing are cases the engine correctly declined; a correct
  deferral is not scored as an attempt.
- **BIRD** — 487 of mini-dev's 500. The thirteen missing were never measured: the harness drops a
  case whose *gold* result set exceeds its row cap, before the engine sees it. They are not
  refusals.
- **Spider** — all 135 attempted, but 135 is not the benchmark. Spider 2.0-lite ships 547
  instances; the engine runs the 135 **local (SQLite)** ones and filters out 412 that need
  BigQuery or Snowflake adapters and credentials it does not have. Those are not failures, and
  they are not attempts either — the row covers a quarter of the suite.

**Two caveats on exact-match, one per benchmark.** The harness subtracts results it cannot
verify against the gold engine (`exact = correct - unportable_exact`). Both BIRD runs cited on this
page — the table row and the local-model figure below — predate that accounting and carry no
portability data, so their exact-match is the raw `correct` rate and should be read as the ceiling
of a range: the gap is about 2.4 points on a 487-case run. The Spider figures are unaffected, though not by a
check that could have found otherwise: on the local slice the flag is set from the adapter
(`dialect == "sqlite"`), so every executed case records portable and the subtraction is
structurally zero. Gold and engine are the same engine there, which is why the code calls the
claim free — the published Spider rates are what the harness prints, and nothing was verified to
make that so.

The Spider row is the retrieval `k=24` configuration. Across the three hosted Spider runs
exact-match spans 34.8–37.8% and got-the-facts 51.9–58.5%, so read it as one point in that spread
rather than as a stable rate.

Those are three different questions, not three attempts at one. ACME is in-domain — one enriched
schema with a golden set, the regime a real deployment is in. BIRD and Spider are **cold start**:
unseen schemas, no glossary, no examples. Spider 2.0 is the hard one by design — real
data-application schemas, often more than a thousand columns.

On BIRD the single-shot engine sits in the range of BIRD's own reported single-shot baselines
(GPT-4o 34.4, Claude 3.7 41.1, o3-mini 42.6). The distance to leaderboard pipelines is added
machinery — candidate selection, verification — and task-specific fine-tuning, not a difference
in the core.

**On local models, the honest result** — and these are two different runs, not one configuration
measured twice. On BIRD, a 24 GB Qwen2.5-Coder-14B with constrained decoding and 5-sample
self-consistency reaches **50.7% exact-match** (54.4% got-the-facts,
`minidev-pg-14b-guided-sc5.jsonl`), close to frontier. On Spider 2.0-lite the same model
single-shot reaches **5.9%** (6.7% got-the-facts, `spider2-qwen2.5-coder-14b.jsonl`), where the
frontier configuration holds at 37.0% and 58.5%.

**What constrained decoding does on Spider, we cannot say from these runs**, and the arithmetic is
the reason. The 32B pair is controlled — same engine revision — and moves 3 correct cases to 7 of
135 (`spider2-32b-baseline.jsonl` → `spider2-32b-guided.jsonl`). The 14B pair moves 8 to 6, and is
not controlled at all: its two runs are 311 engine commits apart and the baseline's revision is
recorded dirty. Four cases either way is smaller than the 3.0-point spread this page already
reports across three runs of ONE hosted configuration, so neither delta is separable from
run-to-run variance on a 135-case slice.

What the local Spider runs do support is a ceiling: every one of them lands under 7% exact-match
where the frontier configuration holds at 37.0%. That gap is an order of magnitude, and it is the
only claim here that does not rest on four cases.

Local capability is far more schema-dependent than the BIRD number alone suggests — the same 14B
reaching 50.7% on BIRD is the contrast. Measure on your own schema before committing an
architecture to it.

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

## Deploying against Oracle

The Oracle read plane refuses writes, but that refusal is partly a property of your **deployment**
rather than of the engine: a `SELECT` can reach an `AUTONOMOUS_TRANSACTION` function through a
view, and restricting the caller does not close it, because a view resolves its references with the
view owner's rights. Pointing the read plane at a database that is open read-only does close it,
measured, and mnemiq reports at boot whether you are in that deployment or resting on the engine's gate alone. See
[docs/oracle-deployment.md](docs/oracle-deployment.md) before connecting a production source.

## Open core

The engine is Apache-2.0 and stands alone. Two commercial planes build on it and are **not**
open source: **Verity** (trust, grading, drift) and **Agentic Fabriq** (governance, per-group
authorization, audit). They align to the open contract (`mnemiq-contract`) by reference; no paid
capability lives in the open core.

## Status

v0.1: the full read path — enrichment, retrieval, the decider, execution, trace — evaluated on
ACME, BIRD mini-dev and Spider 2.0-lite, with a local-model program alongside. Tiered modes
(`instant` / `thinking` / `deep`), row- and column-level security, the governed write path,
cross-source federation and multi-replica deployment are built and wired behind the same
interfaces. Next: additional source adapters, the self-maintaining loops, and hardening the write
plane against a production source.
