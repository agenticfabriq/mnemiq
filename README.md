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
| ACME (in-domain, 25 answerable of 30) | 88.0% | 100.0% | `plan15/enrichment-on.json` ([one case flips](#a-note-on-the-acme-row)) |
| BIRD mini-dev (487 answerable, 11 unseen schemas) | 42.3% | 63.2% | `gpt55-duckdb-pg.jsonl` |
| Spider 2.0-lite (135 local of 547, 30 schemas) | 37.0% | 58.5% | `spider2-full-k24.jsonl` |

<a id="a-note-on-the-acme-row"></a>**A note on the ACME row.** That run is a favourable sample of a corpus with one unstable case. Across eleven CI runs of the nightly gate, `fire-count` is answered wrongly in eight of them, and the level the project actually gates on is 96.0% got-the-facts / 84.0% exact (`evals/trend.json`). The row above is not false — it names its run, and that run really scored it — but the reproducible number is one case lower, and a reader comparing the headline against the gate deserves to be told which is which.

Each row names ONE run, and every number comes from that run alone — but not all of them are
READ from it: BIRD's exact-match is a probe replayed against the gold engine, described below,
because the run predates that accounting. Those artifacts are not distributed — `eval-reports/`
is gitignored — so the names identify a run in our records rather than a file you can open, and
reproducing the BIRD replay needs the artifact plus a Postgres holding mini-dev.

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

**Exact-match on BIRD excludes answers that are right but not portable.** BIRD grades one engine
against itself; mnemiq answers over DuckDB and the gold runs on Postgres, so an answer can be
correct and still use SQL the gold engine will not parse — `DOUBLE` for `DOUBLE PRECISION`,
`YEAR(d)`, `QUALIFY`, or a quoted `"Match"` where Postgres holds `match`. Those are excluded from
exact-match and kept in got-the-facts, which is why the two columns differ by more on BIRD than
elsewhere.

Neither BIRD run stored that probe, so it was **replayed** for this page: every case each run
graded CORRECT was re-executed against the Postgres gold engine. The frontier run loses 32 of 238
(48.9% raw → **42.3%**), the local run 10 of 247 (50.7% raw → **48.7%**). An earlier version of
this section put the gap at "about 2.4 points" from a fleet-wide average. In cases rather than points, which is
how the counts read without rounding: it predicts about 12 per run against 487 answerable; the local
run lost 10 and the frontier run 32. Right to within two cases for one, short by twenty for the
other -- a fleet average is not a per-run estimate. What drives that is
not isolated here — the local run also used constrained decoding, which is a plausible direct
cause of fewer `QUALIFY`s reaching the gold engine — so treat it as a property of the RUN, not of
the model. The Spider and ACME rates are unaffected, and for
two DIFFERENT reasons, neither of which is a check that passed. Spider records the flag on every
case, but from the adapter — `dialect == "sqlite"` — so on this SQLite slice it is `True` whatever
SQL was emitted; run the same slice through a DuckDB attachment and it flips `False` for every
case and the printed exact-match becomes 0.0%. ACME never records it at all: portability is probed
only when gold runs on a different adapter than the answer, and ACME passes one. Neither is a ceiling the way the BIRD
figures are, though: on a single-engine run there is genuinely nothing to subtract, so the zero is
right rather than missing. What is absent is the check, not the correction.

The Spider row is the retrieval `k=24` configuration. Across the three hosted Spider runs
exact-match spans 34.8–37.8% and got-the-facts 51.9–58.5%, so read it as one point in that spread
rather than as a stable rate.

Those are three different questions, not three attempts at one. ACME is in-domain — one enriched
schema with a golden set, the regime a real deployment is in. BIRD and Spider are **cold start**:
unseen schemas, no glossary, no examples. Spider 2.0 is the hard one by design — real
data-application schemas, often more than a thousand columns.

On BIRD, **the table's row** sits in the range of BIRD's own reported single-shot baselines
(GPT-4o 34.4, Claude 3.7 41.1, o3-mini 42.6): at 42.3% it is level with the top of that range, not
past it.

The 48.7% reported further down is not a counter-example to that, and the difference is
configuration rather than a contradiction. That run executes up to five candidates per question
and five on 339 of 487, so it is not a single-shot number and does not belong beside a single-shot
baseline. The table's run reads as single-shot: its candidate count and
inter-candidate agreement score are unset on all 487 cases, where the five-candidate run populates
both on most of its own. Read that as strong evidence rather than proof — several code paths write
a blank pair, so the signature is not unique to a single-shot run, and neither artifact ships here
for anyone to re-check.

The distance to leaderboard pipelines is added machinery — candidate selection, verification — and
task-specific fine-tuning, not a difference in the core.

**On local models, the honest result** — and these are two different runs, not one configuration
measured twice. On BIRD, a 24 GB Qwen2.5-Coder-14B with constrained decoding and 5-sample
self-consistency reaches **48.7% exact-match** (54.4% got-the-facts,
`minidev-pg-14b-guided-sc5.jsonl`).

On that metric it is **above** the frontier run in the table — 48.7% against 42.3%, and note that
those two are not like for like: five candidates against one, which is why the baselines paragraph
compares only the table's row. What separates them here is portability rather than answers: on raw CORRECT the local run is
already slightly ahead (247 against 238, nine cases), and the frontier run then loses three times
as many to SQL Postgres will not parse (32 against 10). Read it as one run each, and as a statement about which dialect these two RUNS emitted -- not
about which model reasons better, and not about the models either: the runs differ in candidate
count (five against one) and in decoding, so the model is one of at least three variables. On got-the-facts, where portability is not excluded, the order is the
usual one: 63.2% against 54.4%. That does not settle the question either -- the same run
differences sit under both metrics -- it just shows the reversal is specific to what exact-match
excludes. On Spider 2.0-lite the same model
single-shot reaches **5.9%** (6.7% got-the-facts, `spider2-qwen2.5-coder-14b.jsonl`), where the
frontier configuration holds at 37.0% and 58.5%.

**These local runs do not support comparisons between them, and the counts are why.** On a
135-case slice, the 32B moves 3 correct cases to 7 with constrained decoding; the 14B moves 8 to
6, across runs 311 engine commits apart with a `-dirty` baseline. Two hosted runs of the frontier
configuration differ by 4 cases from each other. Every difference among the LOCAL arms is the same
handful of cases, so no ordering among those is claimed here, and the effect of
constrained decoding on Spider is not something these runs can settle.

The one durable observation is the size of the remaining gap: no local arm exceeds 8 of 135
exact-match, against 50 of 135 for the frontier configuration. That is not a comparison between
local arms, and it does not isolate a cause — the BIRD and Spider local figures use different
configurations as well as different schemas, so the 48.7% / 5.9% contrast is not a schema effect
on its own.

Measure on your own schema before committing an architecture to it.

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
