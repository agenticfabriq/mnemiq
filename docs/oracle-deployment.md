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

Read that as a limit on *reporting*, not on protection. If you reopen a `gate_only` or `unverifiable`
database as READ ONLY while the process runs, **the database begins refusing writes immediately** —
`ORA-16000` does not wait for mnemiq to notice. What does not update is the verdict: it was emitted once at
startup and is not re-emitted, so nothing further appears in the log for that deployment. So reopening read-only is the right move during an
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
  default **300 seconds**; `0` disables re-probing and returns the old boot-sample behaviour. Only the lapse is reported — a deployment
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
`unverifiable` **warns** at boot — you acknowledged a state you assessed and this is a different one.
Reopen the database read-only instead and the resulting `constrained` is **quiet**, because it is a
verdict that needs no action; at the default WARNING level the read-only line goes quiet.

That silence covers one advisory of the two. The VPD advisory is scored and acknowledged
separately, under the `source-enforcement` key, so a `constrained` deployment whose enforcement
verdict is `partial` or `unverifiable` still warns on that line until you acknowledge that verdict
too — `source-enforcement:partial`. `bypassing` warns whether or not you acknowledge it, for the
reason below.

Two verdicts cannot be silenced at all, and setting a key for either is worse than not setting one.

**`source-enforcement:bypassing` is refused.** Acknowledging it produces the original warning *and* a
second warning saying the acknowledgement was refused. A principal holding `EXEMPT ACCESS POLICY`
bypasses every row policy in the database; that is not a deployment shape to accept quietly, and the
engine declines to let you.

**A lapse is not acknowledgeable either.** The re-probe reports through the log directly and consults no
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
