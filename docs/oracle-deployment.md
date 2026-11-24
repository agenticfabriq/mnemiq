# Deploying mnemiq against Oracle Database

This is the operator's half of the read plane. mnemiq's Oracle adapter refuses writes, but
**that refusal is a property of your deployment, not of this process**, and the difference is
measurable. This page says exactly what the engine enforces, what only the database can enforce,
and how to tell which one you have — at boot, and while the process runs.

Everything asserted here was measured against Oracle Database 23ai Free. Where a claim was *not*
measured, it says so.

## 1. What the adapter enforces on its own

With `read_only=True` the adapter applies a **default-deny leading-keyword gate**: a statement is
refused unless it begins with `SELECT` or `WITH`.

That gate is not belt-and-braces, it is the only control of its kind, because Oracle has no
session-level read-only switch:

| attempt | result |
|---|---|
| `ALTER SESSION SET READ ONLY = TRUE` | ORA-02248 |
| `ALTER SESSION ENABLE READ ONLY` | ORA-00922 |
| `SET TRANSACTION READ ONLY` | works, **but scoped to the transaction and blocks DML only** |

DDL implicitly commits, so it runs straight through `SET TRANSACTION READ ONLY`. Measured: before
the keyword gate existed, a `read_only=True` adapter successfully `TRUNCATE`d and `DROP`ped a
table. The gate is what stops that, and the read-only transaction is what stops direct DML.

## 2. What the adapter cannot enforce, and no statement check can

**A `SELECT` can write.** A PL/SQL function declared `PRAGMA AUTONOMOUS_TRANSACTION` runs in its
own transaction, so the enclosing read-only transaction never applies to it. Measured:
`SELECT f_auto FROM dual` through a `read_only=True` adapter inserted a row and committed it.

Two things make this unfixable in the engine rather than merely unfixed:

- **A view hides the call.** `SELECT n FROM v_sneaky` writes a row while the statement text
  contains no function name at all. Parsing for callables cannot find it.
- **Restricting the caller does not help.** A view resolves its references with the **view
  owner's** rights. Measured: a principal holding `SELECT` on one view and nothing else — no
  `EXECUTE`, no DML, owning no objects — read that view and a row was inserted.

A function *without* the autonomous pragma is stopped by Oracle itself (ORA-14551, *cannot perform
a DML operation inside a query*), so the pragma is the whole of the gap.

**The same applies while merely validating a statement.** Oracle runs a SQL macro's body at
*compile* time, and validation compiles. Measured: `validate()` on
`SELECT m_macro(1) FROM dual`, where `m_macro` is a `SQL_MACRO(SCALAR)` function declared
`PRAGMA AUTONOMOUS_TRANSACTION`, inserted and committed a row — through the read-only adapter, on
the step whose purpose is to decide whether the statement should run. So a statement the engine is
about to **refuse** can still cost a write. Compiling is how Oracle checks syntax, and
`EXPLAIN PLAN FOR` is not an alternative (it writes to `PLAN_TABLE`, which the read-only adapter
cannot do).

## 3. The control that closes it

**Point the read plane at a database that is open read-only.** An Active Data Guard standby, a
read-only PDB, or a refreshable clone.

Measured, on the exact shape above:

```
database READ WRITE   SELECT through the definer-rights view   -> row inserted
database READ ONLY    SELECT through the definer-rights view   -> ORA-16000, no row
database READ ONLY    validate() on the SQL-macro statement    -> ORA-62565, no row
database READ ONLY    SELECT count(*) FROM t                   -> works normally
```

This is the only control that does not depend on what the governed schema happens to contain, and
it closes both the execution path and the parse-time path. Reads are unaffected.

Configure it the way your topology provides it. On Autonomous Database that is an Autonomous Data
Guard standby or a refreshable clone; **that equivalence is by open mode and was not measured
here**, unlike the rows above.

### Necessary, but not sufficient on its own

Connect as a principal holding `SELECT` and nothing else — no `EXECUTE`, no DML, owning no objects
in the governed schema. This removes every *direct* write path and is worth doing. It does **not**
make the connection unable to cause a write, for the definer's-rights reason in §2. Do not treat it
as the control.

## 4. Telling which deployment you have — at boot, and after

`assert_read_only()` runs at startup and reports one of three verdicts. **Only `constrained` is
re-probed afterwards** — it is the only verdict with an assurance to lose.

Read that as a limit on *reporting*, not on protection. If you reopen a `gate_only` or
`unverifiable`
database as READ ONLY while the process runs, **the database begins refusing writes immediately** —
`ORA-16000` does not wait for mnemiq to notice. What does not update is the verdict: it was
emitted once at
startup and is not re-emitted, so nothing further appears in the log for that deployment. So
reopening read-only is the right move during an
incident and takes effect at once; restart afterwards to make the engine's own report agree.

| verdict | meaning | what to do |
|---|---|---|
| `constrained` | The database is open READ ONLY and refuses every write from every principal. | Nothing. This is the deployment §3 describes. Logged at INFO. |
| `gate_only` | This connection holds write-shaped privilege — it owns tables, or holds INSERT/UPDATE/DELETE/ALTER/EXECUTE directly, through a role, or via PUBLIC. `read_only` rests on the engine's gate alone. | Narrow the principal (§3), and prefer a read-only database. Logged at WARNING. |
| `unverifiable` | No write-shaped privilege found, which is **not** the same as being unable to write. | This is the ceiling for a minimal read principal on a writable database — no further check exists, so the action is §3 or an accepted risk. Logged at WARNING, and it is the verdict `MNEMIQ_ACK_ADVISORIES` exists for: it was once demoted to INFO and that suppressed the gap entirely at the default level. |

`assert_enforcing()` separately reports whether VPD is attached to what this connection can see:
`bypassing` (the principal holds `EXEMPT ACCESS POLICY`), `partial`, `attached`, or `unverifiable`.

What holds after boot, and what does not:

- **`constrained` is re-probed; the others are boot samples.** The database's open mode is
  re-checked on a bounded TTL, on the connection already leased for the query, so a database
  reopened `READ WRITE` mid-process reports the **lapse** rather than carrying the boot assurance
  silently to the end of the run. The TTL is the exposure window and is stated rather than argued
  away: one extra round trip per interval, not per query. It is `MNEMIQ_ORACLE_READ_ONLY_TTL_S`,
  default **300 seconds**; `0` disables re-probing and returns the old boot-sample behaviour. Only
the lapse is reported — a deployment
  that was never `constrained` was already warned at boot. The VPD verdicts from
  `assert_enforcing()` are still boot samples and do not re-probe.
- **They warn; they do not refuse.** A `bypassing` verdict does not stop startup today, because
  under the current design the engine still applies its own row and column filters, so refusing to
  boot would take down a working deployment over a control that is not yet load-bearing. When
  enforcement is delegated to the database, this must become fail-closed.

`MNEMIQ_ACK_ADVISORIES` silences a **boot** verdict you have assessed and accepted, keyed
`<advisory>:<verdict>` — for example `read-only-basis:gate_only`. It is keyed by verdict on
purpose: the match is on the exact key, so a changed verdict falls outside the
acknowledgement. Whether that is audible depends on the new verdict, not on the acknowledgement:
acknowledge `read-only-basis:gate_only` and narrow the principal as §3 advises, and the resulting
`unverifiable` **warns** at boot — you acknowledged a state you assessed and this is a different
one.
Reopen the database read-only instead and the resulting `constrained` is **quiet**, because it is a
verdict that needs no action; at the default WARNING level the read-only line goes quiet.

That silence covers one advisory of the two. The VPD advisory is scored and acknowledged
separately, under the `source-enforcement` key, so a `constrained` deployment whose enforcement
verdict is `partial` or `unverifiable` still warns on that line until you acknowledge that verdict
too. Use the verdict you actually got: `source-enforcement:partial` or
`source-enforcement:unverifiable`. Either one silences only itself — copying the wrong one is a
silent no-op that looks exactly like no acknowledgement at all. `bypassing` warns whether or not
you acknowledge it, for the
reason below.

A THIRD advisory shares the setting: `functions`, which reports whether this source can confirm
what a call is bound to. Its verdicts are `functions:defines` (the schema defines functions of its
own, the normal state of a healthy schema), `functions:unavailable` (the source was asked and could
not answer) and `functions:never-asked` (this adapter does not implement `user_functions`).

Acknowledge `functions:defines` once you have read which functions it names. That verdict is not a
fault, and it is the one worth silencing: while it holds, every answer carrying a call reports
lineage `unknown`, because the engine cannot tell a builtin from a same-named function. How many
answers that is depends on how many of your queries call a function at all, not on how many
functions your schema defines: one is enough to turn it on. Measured at 92% of the BIRD and
Spider corpora and 88% of the ACME demo set, which is why the fact is said once here and NOT on
each answer. The other two verdicts are faults and are better fixed than silenced.

Silencing is per verdict here too, so acknowledging `functions:defines` leaves an outage audible.

Two verdicts cannot be silenced at all, and setting a key for either is worse than not setting one.

**`source-enforcement:bypassing` is refused.** Acknowledging it produces the original warning
*and* a
second warning saying the acknowledgement was refused. A principal holding `EXEMPT ACCESS POLICY`
bypasses every row policy in the database; that is not a deployment shape to accept quietly, and the
engine declines to let you.

**A lapse is not acknowledgeable either.** The re-probe reports through the log directly and
consults no
acknowledgement set, so there is no `read-only-basis:lapsed` key and setting one silences nothing.
That is deliberate in effect if not by design — an acknowledgement records a judgement about a
deployment you inspected, and a database that has *changed open mode underneath you* is not that
deployment any more.

## 5. Connecting

| setting | purpose |
|---|---|
| `MNEMIQ_ORACLE_USER` / `MNEMIQ_ORACLE_PASSWORD` | credentials for the read principal |
| `MNEMIQ_ORACLE_CONFIG_DIR` | directory holding `tnsnames.ora`, and for mTLS the wallet |
| `MNEMIQ_ORACLE_WALLET_PASSWORD` | password for an encrypted wallet |
| `MNEMIQ_ORACLE_POOL_MAX` | max pooled connections (default 4). Concurrency above this queues for one |
| `MNEMIQ_ORACLE_ACQUIRE_TIMEOUT_S` | how long a request waits for a connection before failing (default 10) |
| `MNEMIQ_ORACLE_PROBE_TIMEOUT_S` | call timeout for statements the engine issues about itself (default 30); `0` disables |

Connections are **pooled, one leased per operation**. Size the pool to your worker
concurrency: requests beyond `MNEMIQ_ORACLE_POOL_MAX` wait for a connection and then fail with
`DPY-4005` rather than waiting indefinitely, which is deliberate — an unbounded wait is how one
slow statement becomes a whole-server stall.

`MNEMIQ_ORACLE_PROBE_TIMEOUT_S` bounds only the statements mnemiq issues *about itself*:
introspection, view text, the boot advisories, and the validation parse. **Your queries are not
bounded by it.** Profiling a large table can legitimately take minutes, and there is no measured
ceiling to default to — so if you want a ceiling on data queries, set one at the caller.

The driver runs in thin mode, which needs the encrypted `ewallet.pem` from the wallet archive;
`cwallet.sso` is not used. Keep the wallet outside every git working tree.

If the read principal is not the schema owner, the adapter issues
`ALTER SESSION SET CURRENT_SCHEMA` at connect time so unqualified names resolve to the governed
schema. Note that this changes name resolution only — it grants nothing.

## 6. Enriching a production database

`mnemiq enrich` profiles each table before describing it: one count over every column, a
`GROUP BY` per code-like column, partition counts, the grounding and join checks, the value
index. Each of those reads the table in full, and on large tables one read can take hours. So
on Oracle, enrich decides how to profile a source from its manifest entry before reading
anything.

**The size check.** When a source's entry names no `"profile"`, enrich first reads each
table's size from the optimizer statistics (`NUM_ROWS × AVG_ROW_LEN` in `ALL_TAB_STATISTICS`),
which reads no table data. If any table holds more than 1 GiB by that estimate, enrich stops
before reading any table, names the large tables, and says how to go on. A table with no
statistics, or with statistics Oracle marks stale, may be large without the check knowing it, so
enrich names those tables and goes on.

**The two keys.** `"profile"` says how the source is profiled:

```json
[{"id": "erp", "kind": "oracle", "target": "db.example.com:1521/ERP", "catalog": "erp",
  "schema": "APP", "tables": ["ORDERS", "ORDER_LINES", "CUSTOMERS"],
  "profile": "statistics", "exact_tables": ["CUSTOMERS"]}]
```

- `"profile": "exact"` profiles every table in full, as on any other source, with no size check.
- `"profile": "statistics"` reads each table's description from the data dictionary and none of
  its rows: row and null counts and an approximate distinct count from the optimizer statistics,
  keys from the constraints and indexes the table declares, and code lists from its histograms.
- `"exact_tables"` (with `"statistics"` only) names tables to profile in full anyway, as names or
  `*`/`?` patterns matched without regard to case. Use it for the small tables whose exact facts
  matter most, such as the dimensions your questions filter on. A name that matches no table is
  refused. The size check applies to these tables: one above 1 GiB stops enrich before any table
  is read, and enrich says to take it out of `"exact_tables"`.

**The statement time limit.** Every statement enrich sends to the source, in either mode, ends
after `MNEMIQ_ENRICH_STATEMENT_TIMEOUT_S` seconds (600 by default; 0 for no limit), bounded by the
driver's call timeout, which cancels the statement on the server. What a cut statement costs:

- A table's profile ends at the first cut statement; it is never retried column by column.
  Profiled exactly, the table is left out of the model, and enrich names it. Named in
  `"exact_tables"`, it keeps its estimate from the dictionary instead, and enrich says so.
- A grounding or value-index statement that is cut skips that one item.
- Enrich ends with a warning naming every table a cut statement was reading, and the snapshot
  records the ones cut before it was saved (the `enrich:time-limit` job).

The limit is per statement, not per table: a table under the size line with many columns can
still take hours of statements that each finish in time. On Oracle it is the driver's call
timeout, which bounds each round trip to the database: every statement enrich sends is a count, a
`GROUP BY` or a `DISTINCT`, whose work Oracle finishes before it returns the first row, so the
limit bounds that work. A statement that streams rows could outlast it in many short round trips;
of enrich's phases only the experimental examples phase sends one. A read-only statement that meets
Oracle's ORA-01466 (a table redefined while it is read) is retried on a fresh connection up to
three times, each with the full limit, so one statement can run at most four limits plus 2.5 s.
The examples phase's own bound (`MNEMIQ_ENRICH_EXAMPLES`) is capped by the limit too. On a
Postgres source the limit is also the connection's statement timeout, the bound that stops the
server's scan; a `-c statement_timeout` written in the source's DSN or `PGOPTIONS` comes after it
and takes precedence, as it does for answering, so an explicit setting there is the operator's.

An unknown key in a source's entry is refused rather than ignored, so a misspelled `"profile"`
cannot fall back to reading every table.

**Reading the dictionary at scale.** Statistics mode and the size check read the dictionary in
batches: each view once for up to 500 tables, so a schema of 600 tables costs a handful of
statements rather than several per table. Each statement is bounded by the probe timeout
(`MNEMIQ_ORACLE_PROBE_TIMEOUT_S`, 30 s by default); a batch that fails is read table by table
instead, and enrich logs why.

**Comments.** In either mode, enrich reads each table's and column's comment from the dictionary
(`ALL_TAB_COMMENTS`, `ALL_COL_COMMENTS`), reading no table data, and keeps it apart from the
model's description: the describing model is given it as the database's own text, and the card
prints it, so table search reads it too. When a prompt must be trimmed to fit the model's window,
every model description goes before any comment. Instant mode's bare cards still carry names and
types only, without comments.

A comment is free text a person wrote, and can name or describe any table or column in the schema.
So comments, a table's and a column's alike, are shown only to someone who may see every table
enriched and is denied or masked no column; anyone else gets cards without them (their model
descriptions, which are scoped column by column, stay). A role granted the whole schema with no
column policy sees every comment.

A comment is database text, like a column name or a glossary definition, and is treated the same
way: its whitespace is flattened (so it cannot start a line of its own on a card), it is cut to
300 characters, and phrases that address the model are screened out. That screen is defence in
depth, not a guarantee: a comment written as an instruction still reaches the model. What bounds
it is downstream. The describing model's reply may annotate only the columns it was asked about
and explain only codes that were observed, and every statement the model writes for an answer is
refused unless it is a single read within the person's grants, so a comment can steer an answer
only within what that person may already read. Whoever can comment on
a table (its owner, or a holder of `COMMENT ANY TABLE`) can write into the prompt.

**Enriching a subset at a time.** A run enriches what the source's `"tables"` lists, and its
snapshot replaces the last: a table no longer listed leaves. To build a large schema up in batches
instead, list the next batch and run `mnemiq enrich --add`. The run's tables replace their earlier
entries, every other table in the latest snapshot keeps its own (descriptions included, which are
most of a run's time), and the joins are read again over all of them: declared keys from the
dictionary for every table, and inference among the tables profiled exactly, each table in the
mode its own run used, so no estimated table's rows are read. Views are discovered again too, and
if an earlier run's view discovery did not complete, the view inventory stays unavailable (row-
filtered views are refused) until a run without `--add` discovers them. The certified set and the
dictionary are laid again over every table, so a certification made since an earlier batch reaches its
columns. `--add` refuses before reading
anything when the store holds no snapshot of the source, or when the source no longer reports a
table the snapshot holds, and before any model call when the ontology changed since. A table whose
profile fails in an `--add` run keeps its earlier
enrichment, and enrich says so. Worked examples written for an earlier batch predate the later
tables, so none spans batches until those tables are enriched again. To serve the result, list
every table enriched under `"tables"`, or none.

**Checking table search.** On a few tables every table reaches the prompt; on hundreds, each
question's retrieval picks `MNEMIQ_RETRIEVAL_K` of them (24 by default), and a question whose
tables it misses cannot be answered. `mnemiq check-search --questions questions.json` asks each
question as answering would, up to the model call (no model is asked), and says where each table
its gold SQL reads was found, and whether the prompt fits the model's window. The file is a JSON
list of `{"id", "question", "gold_sql"}`, and the identity is the one `ask` would use (`--principal`,
`--roles`). The report holds counts and question ids only, unless `--names` is given: write it with
`--out report.txt` and share that file, not the terminal, where the deployment's own startup
warnings can name tables and columns. A gold query that does not parse, or reads a table the
identity cannot see (absent from the deployment or not granted, which read alike), is reported
apart, never as a search miss.

**What statistics mode gives up.**

- **Counts are estimates.** They are kept apart from exact counts. The model that describes the
  columns sees them marked as estimates, and nothing that needs an exact count reads them: an
  estimated column is never reported as entirely NULL, and gets no value index (so values in a
  question are not checked against it before the query runs).
- **Keys come only from declarations.** A column counts as unique only where an enabled,
  validated single-column primary key or unique constraint, or a valid unique index on the plain
  column, declares it. An approximate distinct count can never prove a key. It can show that a
  column repeats, though, and the fan-out guard uses that. No join is inferred to or from an
  estimated table, because checking a candidate reads its rows. Declared foreign keys are kept.
- **The fan-out guard runs.** It refuses a sum through a join that repeats rows, reading declared
  keys and estimates. With no partition facts, it also refuses a join to the current rows of a
  type-2 dimension (`is_current = 1`), which an exact profile would show is safe: name that
  dimension in `"exact_tables"` to have its partitions profiled.
- **Code lists come only from complete frequency histograms.** A column gets a list only where
  the histogram is `FREQUENCY` with one endpoint per distinct value, was built from every non-null
  row in the same gather as the table's statistics, and Oracle says the table's statistics are not
  stale; and only where every value reads back exactly. "The same gather" is read from
  `LAST_ANALYZED`, which Oracle keeps to the second, and no dictionary view a reader can see says
  which gather built a histogram: so of two gathers of one table within the same second, the first
  one's histogram passes for the second's, and if the column's values changed in between without
  the row count changing, its list can be out of date.
  Measured on 23ai Free: Oracle often stores no text for short `VARCHAR2` values, only a numeric
  encoding of their first bytes. mnemiq reads a value back from that encoding only on a database
  whose character set is UTF-8 (`AL32UTF8` or `UTF8`), and only when the value is at most 5
  printable ASCII bytes and agrees with the column's lowest and highest values. In a single-byte
  character set a longer value can share a short one's encoding, so there such a column gets no
  list. A value holding a NUL character can share one in any character set; no statistic shows
  it. Oracle
  keeps 64 bytes of a value's text, so values of 64 bytes or more give no list. Otherwise the column gets no
  list, never a partial one. A dictionary's code meanings attach only to codes in a column's
  list, so for a column without one only the dictionary's description is kept: name the codes
  there, or in your definitions.
- **Everything is as old as the statistics.** Gather them before enriching
  (`DBMS_STATS.GATHER_TABLE_STATS`, or let Oracle's maintenance job do it), and re-enrich when
  codes change. A code added since the statistics were gathered is missing from the column's list,
  and the dictionary's meaning for it is dropped. Enrich reports how many tables it profiled each
  way, and names the estimated tables that have no statistics, columns without them, or stale ones,
  and the dictionary's code meanings it cannot apply.
- **Nothing else reads an estimated table either.** Grounding skips joins to it, it has no value
  index, and the examples phase (`MNEMIQ_ENRICH_EXAMPLES`), which runs the queries it writes, is
  refused with `"statistics"`.

### Enriching on a DuckDB copy

The other way to profile a large schema exactly without loading production: copy its tables into
one DuckDB file, enrich and build there, and serve the store on Oracle. Every count, code and join
check then reads the copy, and Oracle is read once per table, by the copy.
`scripts/copy_oracle_to_duckdb.py` makes the copy from the Oracle source in your manifest:

```bash
MNEMIQ_ORACLE_USER=... MNEMIQ_ORACLE_PASSWORD=... \
  python scripts/copy_oracle_to_duckdb.py --sources sources.json --out copy.duckdb
```

It reads the source's `"tables"` (every table the schema owner holds when there are none), and then:

1. **Copies the tables into the file's `main` schema,** under the names Oracle stores (upper case,
   or as quoted). Before reading any row it prints a disk estimate (`NUM_ROWS × AVG_ROW_LEN`
   summed, plus the LOB segments when the reader owns the tables, and the tables with no statistics
   named), and stops past `--max-gb` (10 by default). DuckDB compresses by column, so the file is
   usually smaller than the estimate. It writes the copy's source entry beside the file, as
   `copy.duckdb.source.json`: the Oracle source's `id` and `catalog`, `"kind": "duckdb"`,
   `"schema": "main"` and the same tables.
2. **In a separate folder, point a deployment at the copy:** that entry as its manifest
   (`MNEMIQ_SOURCES_PATH`), with the same definitions and access policy files you use on Oracle.
   Run `mnemiq enrich`, then `mnemiq build`.
3. **Copy the `store.duckdb` it writes into your Oracle deployment.** The Oracle source entry must
   keep the same `id` and `catalog`, and list the same tables under `"tables"`, so the store's
   tables name the source it serves.

**What the copy keeps.** Table and column comments (written as DuckDB comments, which enrich reads
from the copy as it reads them from Oracle, under the same rules), and the foreign keys the schema
declares between copied tables (written, in each key's declared column order, to a
`mnemiq.declared_keys` table in the file, outside
`main`, which enrich reads as the copy's declared keys). Schema `mnemiq` holds only these records of
the copy, so a DuckDB source naming it as its `"schema"`, in any case, is refused. A key into another schema, or to a table
not copied, is left out. Nothing in DuckDB enforces a key, so an orphan row cannot stop the copy.

**How it reads Oracle.** `SELECT` only, each table in its own read-only transaction (a single
transaction held for the whole copy would need Oracle's undo to reach back to its start, and a long
copy would fail with ORA-01555). Rows arrive in batches (`--batch-rows`, 100,000 by default), each
checked with Arrow's own validation before it is stored. Each table is copied in one DuckDB
transaction, committed after its last row, so a copy that stops part way, even by the process dying,
leaves no trace of the table it was copying (and, with `--replace`, keeps the copy it was replacing):
run it again and it copies what is missing, skipping the tables it recorded as copied whole
(`--replace` copies those again too). The file records the one source it is a copy of
(`mnemiq.copy_source`: the target, the database it reached by that database's own names, so a TNS
alias that now resolves elsewhere is told apart, the owner and the user that read it, never a
password) and each
table copied whole (`mnemiq.copied_tables`), so a file holding another source's copy, or tables with
no record of their source, is refused before Oracle is reached: copy into a new `--out` instead. The
copy holds the file from that check to its end, so a second copy into it, or anything else opening
it, is refused until the first is done. Until a run has published its tables, comments and keys,
the file records the copy as unfinished (`mnemiq.copy_state`), and `mnemiq enrich` and `mnemiq
refresh` refuse it, saying to rerun the copy: a `--replace` stopped part way would otherwise be
enriched with new tables beside old keys.

**What it costs.**

- **No access rules on the copy.** It holds every row of every table it copied, outside the
  database's grants, auditing and row policies. Keep it where you would keep an export.
- **The store is as current as the copy,** and covers only the tables in it. Refreshing means
  copying again (`--replace`) and enriching again.
- **Types are DuckDB's.** A `NUMBER` whose declaration bounds its values arrives exactly: with no
  scale and at most 18 digits as `BIGINT`, otherwise as `DECIMAL(p, s)`, up to 38 digits. A
  `NUMBER` declared with no precision or scale, a `FLOAT`, or one past 38 digits arrives as
  `DOUBLE`, so digits past about 15 significant ones are lost on the copy: no fixed decimal type
  holds every value such a column can. `DATE` and `TIMESTAMP` arrive as `TIMESTAMP`, `CLOB` and
  `BLOB` as `VARCHAR` and `BLOB`, each value under 1 GB. The cards show these types, and the profile
  reads values as the copy holds them, but answering on Oracle runs against Oracle's own.
- **Names DuckDB cannot keep apart are refused.** DuckDB matches names without regard to case, so
  two tables, or two columns of one table, whose names differ only in case stop the copy before any
  row is read.
