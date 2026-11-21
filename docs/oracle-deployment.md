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
