# mnemiq

**Text-to-SQL you can tune to your database.** */NEM-ik/ — the "m" is silent, as in mnemonic.*

An open-source engine that answers natural-language questions over your database, built so that
every stage between the question and the SQL is a setting you can read, change, and measure.

![The mnemiq workbench answering two questions and declining a third. Each answer shows the SQL
that produced it, the tables it read, and how many candidate queries agreed. The declined question
asks for ticket revenue; the engine states that the tables hold concert years and stadium capacity
but no ticket price, tickets sold, or revenue column.](docs/workbench.png)

Two answers and a refusal. The third question asks for revenue the database does not hold, and the
engine says so — naming the columns it would have needed — instead of returning a number that looks
right. That distinction is the whole design.

**Read more** · [Launch article](https://agenticfabriq.com/blog/mnemiq/launch) ·
[Technical report (PDF)](docs/mnemiq-technical-report.pdf) ·
[Paper (PDF)](docs/mnemiq-paper.pdf) ·
[beacon](https://github.com/agenticfabriq/beacon) — the grader and results tracker

## Why mnemiq exists

Every text-to-SQL product has an accuracy number. Almost none of them were measured on a database
that looks like yours.

A system can do well on a benchmark and then struggle on your warehouse because the schema is
larger, the naming is different, the business definitions live in people's heads, or the
configuration that suited the benchmark simply doesn't suit your data. When it underperforms, a
closed system gives you no way to find out why, and nothing to change.

So the question worth asking isn't *how accurate is it*. It's:

> How well will this work on my database — and what can I change if it doesn't?

mnemiq is built to make both halves answerable. It is Apache-2.0, runs inside your own environment
on models you choose, and exposes the major parts of the pipeline as settings rather than
internals. No accuracy number applies to your database until you have run it on your database;
mnemiq is the engine and the evaluation harness for doing that.

## How it works

mnemiq separates writing SQL from deciding to run it. A model proposes a query. A deterministic
layer then rules on it before anything touches the database. The query has to be read-only, may
only reference objects the caller is allowed to see, has to compile in the source's own SQL
dialect, and has to survive an `EXPLAIN`. Fail any of those and you get a refusal with a stated
reason rather than a plausible number.

Two properties fall out of that ordering. Permissions apply *before* schema retrieval, so the model
is never shown a table the caller may not see — naming it is useless rather than refused. And every
answer carries a trace: the SQL that ran, the tables it touched, the enrichment version behind it.

![The nine stages of a mnemiq answer: knowledge sources, enrichment, the semantic contract,
retrieval, generation, the decider, execution, verification, and the answer with its trace.
Deterministic stages run first and last; the model proposes only in the middle.](docs/pipeline.png)

Read the diagram left to right, top to bottom. The stages in red are the ones that can stop an
answer: the decider refuses or repairs, execution runs under policy, and verification can defer.
The model appears once, at stage 05, and everything around it is deterministic.

### The pipeline is settings, not internals

The parts people usually can't reach are the parts mnemiq puts in your hands:

- **Which model writes the SQL** — hosted or local, one candidate or several.
- **How much schema context is retrieved**, and how it is ranked.
- **How much semantic enrichment is built**, and whether a human certifies it.
- **How aggressively the system refuses** — the verifier and its threshold.
- **What the decider enforces**, including row and column policy applied to the query tree rather
  than requested of the model.

Each of those is a dial with a cost on the other side, which is why they are dials and not
defaults. More context is not free. More compute is not automatically better. The right setting
depends on your data, and the point of the harness is that you can find out rather than guess.

### The semantic layer has tiers

Before any question is asked, mnemiq can inspect the database and build context around the schema:

- **Tier 0** — tables and columns only.
- **Tier 1** — adds structural information: primary and foreign keys, profiling, value
  distributions.
- **Tier 2** — adds meaning an LLM proposes: table and column descriptions, grain, glossary terms,
  coded-value meanings.

Enrichment is a multiplier on meaning that isn't already in the schema. Where column names already
say what they hold, richer cards add length without adding signal. Where three columns are all
called *revenue* by three different teams, the meaning is in a person, not the schema — and that is
exactly what a certified definition captures. For production use, definitions can be reviewed and
certified by a named owner, and the operator's dictionary overrides everything the model proposed.

Codes are grounded or left bare, never guessed. `E11` or `NC-17` take their meaning from the data
itself, from a standard code system (TTL/SKOS/OWL), or from a hand-written dictionary, with the
source recorded. No evidence, no meaning. See [docs/grounding.md](docs/grounding.md).

## Measure it on your own database

The evaluation harness is part of the engine, not a separate research project. It runs a question
set against a configuration, grades results by the data returned rather than by string-matching the
SQL, and reports right, refused, and wrong as three separate numbers — because a system can buy
accuracy by answering less often, and a single figure hides that.

A useful first pass, on your data:

- Take one meaningful slice of your schema, not the whole warehouse.
- Write 20–30 questions people actually ask, and tag each one: *answerable from column names*,
  *needs a definition*, *should be refused*.
- Run it with enrichment on and off, a local model and a hosted one, the verifier at two
  thresholds.
- Read the result by tag. The tags are the diagnosis: if the definition-band questions fail while
  the schema-band questions pass, you have a glossary problem and documentation will pay for
  itself. If both already pass, you were about to spend a quarter on something worth very little.

The same harness runs the public benchmarks (BIRD mini-dev, Spider 1.0, Spider 2.0-lite) and the
warehouse comparison scripts under `scripts/`, so the setup you use on your data is the setup the
published numbers came from.

Grading itself lives in **[beacon](https://github.com/agenticfabriq/beacon)**, a separate
Apache-2.0 repository: the grader that decides what counts as correct, and the tracker that holds
every run behind the published figures. Keeping it out of the engine is deliberate — a system
should not mark its own homework, and the same grader scores mnemiq, Snowflake Cortex Analyst and
Databricks Genie in the comparison. Per-question results are published there, so a number in the
launch article can be traced to the SQL and the rows that produced it.

## Quickstart

Runs on a clean clone with no database of your own and no Docker. The seed step writes a small
SQLite database plus its source manifest and access policy under `demo/`.

```
uv sync
uv run python scripts/seed_demo.py

export MNEMIQ_LLM_BASE_URL=...  MNEMIQ_LLM_API_KEY=...  MNEMIQ_LLM_MODEL=...
export MNEMIQ_SOURCES_PATH=demo/sources.json
export MNEMIQ_AUTHZ_PATH=demo/authz.json
export MNEMIQ_STORE_PATH=demo/store.duckdb

uv run mnemiq enrich      # profile + describe the schema  (~30 s on the demo's 4 tables)
uv run mnemiq build       # index it for retrieval          (~2 s)
uv run mnemiq ask "how many customers are there by country?" --roles analyst
```

`mnemiq enrich` is the only slow step: it profiles every column and makes one LLM pass over the
schema, so expect **roughly 30 seconds for the demo's four tables** and longer in proportion to
your own. It prints nothing until each table completes — it is working, not hung. The result is
cached, so you pay it once per schema rather than per question.

Two more worth trying, because they show the parts that aren't the model:

```
uv run mnemiq ask "how many enterprise customers are there?" --roles analyst
uv run mnemiq ask "what was our total revenue last quarter?"   --roles analyst
```

The first joins through a lookup table to resolve a coded column — `segment_cd` holds `A`/`B`/`C`
and nothing in the name says "enterprise". The second is refused: the demo schema has no price or
revenue column, and the engine says so instead of returning a number.

Access is fail-closed. `--roles analyst` is required — without a role the engine grants nothing
and defers, which is the correct behaviour and the first thing people mistake for a bug. No policy,
no grants, no snapshot — no data.

### Which LLM endpoints work

**Any OpenAI-compatible `/v1` endpoint.** mnemiq talks to `MNEMIQ_LLM_BASE_URL` through the
standard OpenAI client, so vLLM, Ollama, llama.cpp's server, LM Studio, vendor gateways and the
hosted APIs all work — set the base URL, a key (any non-empty string for local servers that
ignore it) and a model name. Nothing about the engine assumes a hosted provider, which is what
"runs inside your perimeter" means in practice: point it at a local server and no schema, no
question and no row ever leaves your network. Embeddings follow the same setting, or their own
via `MNEMIQ_EMBED_*`.

## Use it from a browser (workbench)

```
cd workbench && pnpm install && pnpm build
uv run mnemiq serve --http     # http://127.0.0.1:8080
```

One process serves both the workbench and the HTTP API — `POST /v1/ask` (JSON), `POST /v1/chat`
(SSE, [AG-UI](https://github.com/ag-ui-protocol/ag-ui) event vocabulary), `GET /v1/schema`. Every
answer shows the SQL that produced it and the tables it read; a question the data cannot support
comes back as a stated reason, not a guess — that is the interface pictured at the top of this
file. See [`workbench/README.md`](workbench/README.md).

## Use it from an AI agent (MCP)

`uv run mnemiq serve` exposes two read-only, access-scoped tools over stdio — `db_read(question)`
(answer + SQL + trace) and `get_schema()`. Point any MCP client at it:

```json
{ "mcpServers": { "mnemiq": { "command": "mnemiq", "args": ["serve"] } } }
```

## Sources

Postgres, SQLite, DuckDB, Oracle, Snowflake and Databricks, with DuckDB as the universal executor.
The semantic model (`mnemiq-contract`) is open, and dbt-semantic-interfaces import/export ships
with it.

### Deploying against Oracle

The Oracle read plane refuses writes, but that refusal is partly a property of your **deployment**
rather than of the engine: a `SELECT` can reach an `AUTONOMOUS_TRANSACTION` function through a
view, and restricting the caller does not close it, because a view resolves its references with the
view owner's rights. Pointing the read plane at a database that is open read-only does close it,
measured, and mnemiq reports at boot whether you are in that deployment or resting on the engine's
gate alone. See [docs/oracle-deployment.md](docs/oracle-deployment.md) before connecting a
production source.

## What's commercial

**The engine is Apache-2.0 and always will be — enrichment included.** Nothing here is a
time-limited or feature-gated build, and no capability is stubbed out pending a licence key.

Specifically open, because these are the parts people assume are held back: the enrichment
pipeline including the LLM pass and coded-value grounding (`src/mnemiq/enrichment/`), the verifier
and its judge (`src/mnemiq/verify/`), the access checks (`src/mnemiq/authz/`, `src/mnemiq/sql/`),
the result grader (`src/mnemiq/eval/grade.py`), and the benchmark harness that produced the
published numbers (`scripts/`).

Commercial are two things that sit *around* the engine rather than inside it: **Verity**, a managed
grading and drift service, and the **Agentic Fabriq control plane** — identity, vaulted
credentials, per-group grants and audit across many sources. Both talk to the engine through the
open contract (`mnemiq-contract`), so a self-hosted deployment is not a degraded one; it is the
same read path without a managed service in front of it.

Read the code rather than taking this on trust — that is the point of shipping it.

## Known weaknesses

Filed as open issues rather than left to be discovered, because they are readable in the source
either way: the [verifier fails open when its judge is unreachable](https://github.com/agenticfabriq/mnemiq/issues/2),
[verification is off by default](https://github.com/agenticfabriq/mnemiq/issues/3) despite being the
only lever measured to reduce the wrong-rate, [`MNEMIQ_ROLES` is ignored by the CLI](https://github.com/agenticfabriq/mnemiq/issues/4),
and [lineage reports `unconfirmed-function-identity`](https://github.com/agenticfabriq/mnemiq/issues/5)
on ordinary queries. Contributions and arguments welcome on all four.

## Status

v0.1: the full read path — enrichment, retrieval, the decider, execution, trace — evaluated on
ACME, BIRD mini-dev, Spider 1.0 and Spider 2.0-lite, with a local-model program alongside. Tiered
modes (`instant` / `thinking` / `deep`), row- and column-level security, the governed write path,
cross-source federation and multi-replica deployment are built and wired behind the same
interfaces. Next: additional source adapters, the self-maintaining loops, and hardening the write
plane against a production source.
