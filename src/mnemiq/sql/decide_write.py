from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import ViewDefinition
from mnemiq.sql.authz_guard import (
    check_access,
    check_opaque_columns,
    check_unmodelled_calls,
    check_unmodelled_source,
)
from mnemiq.sql.functions import (
    UnreadableCalls,
    called_names,
    inventory_from,
    opaque_columns,
)
from mnemiq.sql.cls import check_cls
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.prove import prove
from mnemiq.sql.qualify import object_key
from mnemiq.sql.scope import base_tables, check_policy_target, fields_as_members, known_columns
from mnemiq.sql.rls import apply_row_filters_to_write, check_filter_targets, unplaceable_policy
from mnemiq.sql.verdict import ApprovedWrite, Refusal, RefusalCode
from mnemiq.sql.views import check_views

_WRITE_ROOTS = (exp.Insert, exp.Update, exp.Delete)


def _has_unscoped_with(ast: exp.Expression) -> bool:
    """Is a WITH attached directly to the write root, rather than inside the query it wraps?

    Asked STRUCTURALLY -- "is any direct child a `With` node" -- and deliberately not by looking
    up an arg by name. The first version read `ast.args["with_"]`, which is this sqlglot's
    spelling; `pyproject.toml` declares `sqlglot>=25` and 25.34.1 spells the same arg `with`, so
    a name-keyed lookup silently returns None across most of the supported range and the floor
    never fires. A guard that finds nothing approves everything.

    This repo has already paid for that lesson once: `views.py` records a whitelist that read
    `args["from"]` where that sqlglot said `from_`, enumerated no sources, and therefore approved
    every shape. Node TYPES are stable across versions in a way arg NAMES are not.

    The CTE spelled inside the statement (`INSERT INTO t (cols) WITH x AS (...) SELECT ...`)
    lives under the projection, not under the root, so it is correctly not matched here: it is
    inside the scope `build_scope` roots at, and every guard already sees it.
    """
    for value in ast.args.values():
        if isinstance(value, exp.With):
            return True
        if isinstance(value, list) and any(isinstance(v, exp.With) for v in value):
            return True
    return False


# A query a leading WITH can move into. A Subquery is only the parentheses around one, so the
# Select or set operation inside it is the node whose names the CTEs bind.
_QUERIES = (exp.Select, exp.SetOperation)


def _own_with(node: exp.Expression) -> tuple[str, exp.With] | None:
    """The WITH attached directly to `node` and the arg holding it, found by node TYPE (see
    `_has_unscoped_with` for why never by arg name). The key comes back so the WITH can be
    re-parked under the same spelling, which every root in one sqlglot version shares."""
    for key, value in node.args.items():
        if isinstance(value, exp.With):
            return key, value
    return None


def _inside(node: exp.Expression, ancestor: exp.Expression) -> bool:
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False


def _statement_queries(ast: exp.Expression, skip: exp.Expression) -> list[exp.Expression]:
    """The queries directly under a write: each Select or set operation with no other query
    between it and the root -- an INSERT's source, a WHERE or SET subquery, a derived table.
    Nothing inside `skip` (the leading WITH itself) counts."""
    out: list[exp.Expression] = []

    def visit(node: exp.Expression) -> None:
        for child in node.iter_expressions():
            if child is skip:
                continue
            if isinstance(child, _QUERIES):
                out.append(child)
            else:
                visit(child)

    visit(ast)
    return out


def _lower_leading_with(ast: exp.Expression) -> exp.Expression | None:
    """Move an INSERT's leading WITH into the INSERT's source query, or None to refuse.

    M30's remainder. `build_scope` roots at the query a write runs, so CTEs parked on the write
    root are read by no guard (see the UNSCOPED_CTE comment in `check_write_shape`). The same CTEs
    spelled inside that query are read by every guard. So this moves the WITH there, and the
    statement every guard checks is then the statement that runs -- one implementation of the
    scope rules, not a second one taught to look in the root's WITH (M7's lesson).

    Moved only where the move cannot change what the statement means, which is an INSERT's
    source query and nowhere else. That query has no outer scope, so a CTE body moved into it
    binds exactly the names it bound before. A subquery of an UPDATE or DELETE does have one:
    the target row. Measured on DuckDB 1.5.4, a body reading `scratch.amount`, or an unqualified
    `amount` its own tables lack, fails with the WITH leading ("Referenced table/column not
    found") and RUNS once moved, correlated to the row being updated -- and a body correlated
    that way is evaluated per row, not once. Telling those apart needs the schema and the
    engine's binding rules, so a leading WITH on an UPDATE or DELETE stays refused. Also refused:

      * a CTE body that is not a query (`WITH d AS (DELETE ... RETURNING ...)`): no read to
        govern it as.
      * a CTE name used outside the source query -- including the INSERT target itself, which
        DuckDB binds to the table but which this does not assume of every engine.
      * CTEs used by any query besides the source (an ON CONFLICT subquery, a VALUES scalar):
        the move would leave that use bound to a table, or need a copy evaluated once per copy.
      * a source query with its own WITH: merging two lists changes which name a body binds,
        and that is the forward-reference ground M150 found engines disagree on.

    A name is matched case-insensitively and with no schema qualifier. That over-matches a
    quoted identifier, and over-matching only refuses more or moves the WITH somewhere it is
    shadowed and inert. Under-matching is the unsafe direction: a reference the move missed would
    read the table instead of the CTE.
    """
    if not isinstance(ast, exp.Insert):
        return None
    found = _own_with(ast)
    if found is None:
        return None  # a WITH in a list arg, which no supported sqlglot builds
    key, leading = found
    ctes = leading.expressions
    if not ctes or not all(isinstance(cte.this, exp.Query) for cte in ctes):
        return None
    names = {cte.alias.casefold() for cte in ctes}

    def names_a_cte(table: exp.Table) -> bool:
        return (not table.args.get("db") and not table.args.get("catalog")
                and table.name.casefold() in names)

    queries = _statement_queries(ast, leading)
    users = [q for q in queries if any(names_a_cte(t) for t in q.find_all(exp.Table))]
    outside = [t for t in ast.find_all(exp.Table)
               if names_a_cte(t) and not _inside(t, leading)
               and not any(_inside(t, q) for q in queries)]
    source = ast.expression
    while isinstance(source, (exp.Subquery, exp.Paren)):  # INSERT INTO t (SELECT ...)
        source = source.this
    if outside or len(users) != 1 or users[0] is not source:
        return None
    query = source
    if _own_with(query) is not None or key not in type(query).arg_types:
        return None
    query.set(key, leading.pop())
    return ast


def _target_node(ast: exp.Expression) -> exp.Table | None:
    """The table node being written, as a NODE, or None if this engine cannot name exactly one.

    RLS needs the identity, not the name: the target is the one table a write must not wrap in a
    derived table, and the same name can appear elsewhere in the statement as an ordinary read
    that must be.

    None means REFUSE, never "guess". This used to prefer `ast.this`, which for a multi-target DELETE is the whole join
    expression, so `DELETE s FROM t JOIN s` resolved to `t` -- a table the statement only READS,
    and the grant was checked against it. Re-measured 2026-09-03 on sqlglot 30.12.0: `ast.this`
    prints `t JOIN s ON t.id = s.id`, while a bare `find(exp.Table)` returns `s`, the right
    table. So the defect was the `this`-first ordering, not the `find` fallback, which this
    comment previously blamed and which never ran on that shape. The fix is to refuse rather
    than guess: None
    the statement only reads while `s`, the one it deletes from, was never authorized at all.
    sqlglot puts the deleted tables in `args["tables"]` for that shape. DuckDB happens to reject
    the syntax, so the EXPLAIN proof caught it downstream -- an authorization decision rescued by
    a parser error is not a control, and the next dialect need not oblige.
    """
    if ast.args.get("tables"):  # multi-target DELETE: the deleted table is not `this`
        return None
    this = ast.this
    if isinstance(this, exp.Schema):  # INSERT INTO t (cols)
        this = this.this
    return this if isinstance(this, exp.Table) else None


def _target_table(ast: exp.Expression) -> str | None:
    """The target's object-id, spelled the way the snapshot spells one.

    `object_key`, not `.name`. Reading `.name` dropped every qualifier before the write grant was
    checked, so `UPDATE pg.claim` was authorized against a grant on bare `claim` and then executed
    against `pg.claim` -- and the RLS half of this same change already used `object_key`, so one
    change asked the same question two ways. An INSERT is the shape with no backstop: an
    UPDATE/DELETE target is also a base table and `check_access` refuses it when it is not
    visible, but an INSERT target is not a base table and no guard before this one sees it.
    """
    node = _target_node(ast)
    if node is not None and isinstance(ast, exp.Update):
        # A MySQL-style multi-table UPDATE -- `UPDATE a, b SET a.x = 1, b.y = 2` -- parses with
        # `b` hanging off `a`'s OWN `joins`, not in a `tables` arg (that arg is DELETE-only), so
        # the guard above never sees it. Measured before this: APPROVED with target='a' while
        # `b.y` was written and `b` was never checked for a write grant.
        #
        # The presence of `joins` CANNOT be the test, and the review gate blocked the version
        # that used it: sqlglot parses `UPDATE a JOIN b ON ... SET a.x = b.y` identically, and
        # that statement assigns only `a.x` and is perfectly legitimate. What separates them is
        # what the SET clause ASSIGNS to, so that is what this reads.
        assigned, unattributable = set(), False
        for assignment in ast.args.get("expressions") or []:
            lhs = assignment.this if isinstance(assignment, exp.EQ) else assignment
            if isinstance(lhs, exp.Column) and lhs.table:
                assigned.add(lhs.table)
            else:
                # An UNQUALIFIED column, or a row-value `SET (a.x, b.y) = (...)` tuple. Against a
                # single-table UPDATE that is unambiguous. Against a joined one it is not: MySQL
                # resolves a bare `SET x = 1` to whichever joined table owns `x`, so `UPDATE a
                # JOIN b ... SET x = 1` can write `b.x` while this resolves the target as `a`.
                # We cannot attribute it without a schema, so it counts as unattributable rather
                # than as an assignment to the target.
                unattributable = True
        # Refuse when the statement joins and anything is unattributable, when more than one
        # table is assigned, or when the one assigned table is not the resolved target. Each is
        # the same condition: this engine cannot name the single object being written, and an
        # authorization decision must not be made against a guess.
        if len(assigned) > 1 or (assigned and node.alias_or_name not in assigned):
            return None
        if unattributable and node.args.get("joins"):
            return None
    return object_key(node) if node is not None else None


def check_write_shape(sql: str, dialect: str = "duckdb") -> exp.Expression | Refusal:
    """One bounded DML statement, or a Refusal. UPDATE/DELETE must be bounded by a WHERE."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except Exception:
        return Refusal(code=RefusalCode.PARSE_ERROR, message="That is not valid SQL.")
    if len(statements) != 1:
        return Refusal(
            code=RefusalCode.NOT_A_SINGLE_STATEMENT,
            message="Return exactly one statement. Multiple statements are never executed.",
        )
    ast = statements[0]
    if isinstance(ast, _WRITE_ROOTS) and _has_unscoped_with(ast):
        # A LEADING `WITH` parks its CTEs in the write root's `with_` arg, outside where
        # `build_scope` roots itself -- so `base_tables` returns [] and every guard sees an empty
        # statement. Not "the filter is missing": no guard runs. Measured, each against the plain
        # spelling as a control: a read of a table with NO grant was approved and executed, and so
        # was a reference to a column that does not exist.
        #
        # So an INSERT's WITH is moved into its source query, where every guard reads it and from
        # where it runs (M30). Anywhere a move could change the statement's meaning -- every
        # UPDATE and DELETE among them -- it is declined rather than approved blind;
        # `_lower_leading_with` says why for each shape.
        lowered = _lower_leading_with(ast)
        if lowered is None:
            return Refusal(
                code=RefusalCode.UNSCOPED_CTE,
                message=("Put the WITH inside the query that uses its CTEs, merging it with any "
                         "WITH that query already has, or inline it. A leading WITH is checked "
                         "only on an INSERT whose SELECT is the one query using its CTEs, where "
                         "each CTE is a SELECT and none is named like the target."),
            )
        ast = lowered
    if not isinstance(ast, _WRITE_ROOTS):
        return Refusal(
            code=RefusalCode.NOT_A_WRITE,
            message="Only INSERT, UPDATE, or DELETE is allowed here. DDL is never executed.",
            subject=type(ast).__name__.upper(),
        )
    if isinstance(ast, (exp.Update, exp.Delete)) and ast.args.get("where") is None:
        return Refusal(
            code=RefusalCode.UNBOUNDED_WRITE,
            message="An UPDATE or DELETE must have a WHERE clause; an unbounded mutation is refused.",
        )
    return ast


def decide_write(
    sql: str,
    visible: dict[str, set[str]],
    grants: GrantSet,
    *,
    policy: AccessPolicy,
    adapter=None,
    dialect: str = "duckdb",
    target: str | None = None,
    views: dict[str, ViewDefinition] | None = None,
    writes_enabled: bool = False,
) -> ApprovedWrite | Refusal:
    """The deterministic write decider: deployment switch -> shape -> table/column authz ->
    target-writable -> proof.

    `writes_enabled` is the deployment-level switch (`settings.write_enabled`). **It defaults to
    False on purpose.** A security parameter whose default is permissive means any caller that
    forgets it fails open; defaulting to disabled means forgetting fails closed. That is surprising
    for a library API and correct for a decider.

    M3: this switch reached only the adapter (`read_only=not write_enabled`), never a decider. So a
    deployment with writes off still APPROVED the write, executed it, and reported a refusal built
    from whatever the read-only attachment raised -- the database as the control, which is the
    principle this project defines itself against.

    `policy` is **required**, by the same argument one paragraph up. It used to default to `None`
    and become an empty `AccessPolicy` -- a security parameter with a permissive default, in the
    signature whose sibling parameter is documented as fail-closed for exactly that reason. Empty
    is a legitimate value (a deployment with no policy, and `Runtime.write` passes it when there
    is no snapshot), which is precisely why it must be *stated*: "no policy supplied" and "a
    policy that restricts nothing" are the same denial and different facts, and collapsing them is
    M2's finding and the peer lane's M45.
    """
    target = target or dialect
    if not writes_enabled:
        # First, before shape or authz: if the deployment does not do writes, nothing about this
        # particular statement matters, and an EXPLAIN against a read-only source is work done to
        # reach a foregone conclusion.
        return Refusal(
            code=RefusalCode.WRITES_DISABLED,
            message=(
                "This deployment has writes disabled, so no write can be approved. This is a "
                "deployment setting, not a limit on your grants."
            ),
            subject=None,
        )
    shaped = check_write_shape(sql, dialect=dialect)
    if isinstance(shaped, Refusal):
        return shaped
    ambiguous = fields_as_members(shaped, target, known_columns(visible, policy))  # M165, M167
    if ambiguous is not None:
        return ambiguous
    # M150, as on the read side: no policy is applied for an engine whose CTE binding it misreads.
    unsupported = check_policy_target(shaped, policy, target)
    if unsupported is not None:
        return unsupported

    refusal = check_access(shaped, visible, target)  # every referenced table readable, cols exist
    if refusal is not None:
        return refusal
    # M65: what `check_access` reads -- `base_tables` -- is empty for a source it cannot model,
    # and an empty list passes. After it, so a source it already refuses keeps its refusal.
    unmodelled = check_unmodelled_source(shaped)
    if unmodelled is not None:
        return unmodelled

    # The read path's M43 fix, on the write path, because the premise is identical and a write
    # is the worse place to lose it: measured before this line existed, `UPDATE claim SET
    # amount = pg_read_file('/etc/passwd') WHERE id = 1` and `INSERT INTO claim (id, amount)
    # SELECT customer_rows(), 1` were both APPROVED with tables=['claim'] -- the same audit lie,
    # and here it persists what it read into a table.
    # The inventory too, or the two paths diverge on the same source: without it this is the
    # allowlist alone, so a macro named `length` or `count_star` could feed an approved UPDATE
    # that stores what it read into a granted table. Both dialects, as on the read path: the
    # guard renders to find the names the source will be asked for, and `target` has already
    # been defaulted to `dialect` at the top of this function.
    # Asked about the names this statement calls, where per-name is sound (M101): reading a
    # whole Oracle dictionary cost 4.8 s a decision at 107,886 synonyms. A statement whose calls
    # cannot be read gets the whole-source answer, so the guard decides it as it always did.
    try:
        called = called_names(shaped, dialect, target)
    except UnreadableCalls:
        called = None
    functions = inventory_from(adapter, called=called)
    opaque = check_unmodelled_calls(shaped, functions, dialect, target)
    if opaque is not None:
        return opaque

    # And the column form, for the reason above stated once more: a write is the worse place to
    # lose it. `INSERT INTO notes SELECT id, leaked FROM vc_t` stores what the virtual column's
    # expression returned into a table the caller may read forever (M100, the M30 shape).
    computed = check_opaque_columns(shaped, opaque_columns(adapter, functions), target)
    if computed is not None:
        return computed

    # A write needs RAW access: a masked column is treated as denied for writes.
    write_cls = check_cls(shaped, AccessPolicy(denied=policy.denied | policy.masked), target)
    if write_cls is not None:
        return write_cls


    tgt = _target_table(shaped)
    if tgt is None:
        return Refusal(
            code=RefusalCode.AMBIGUOUS_WRITE_TARGET,
            message="This engine cannot tell which single table this statement writes to.",
        )
    if not grants.allows_write(tgt):
        return Refusal(
            code=RefusalCode.UNAUTHORIZED_WRITE,
            message=f"You may not write to {tgt!r}.",
            subject=tgt,
        )

    # Ordered AFTER the grant check, and that ordering is the finding. Mirroring `decide`
    # (access -> CLS -> views) is wrong on this path: an INSERT's target is not a base table, so
    # `check_access` never covers it, and `check_views` walked it anyway. `INSERT INTO
    # hidden_view ...` then refused UNGOVERNED_VIEW naming the view AND its row-filtered base --
    # three facts about objects the caller holds no grant on, and distinguishable from the
    # plain-target refusal, so a probe rather than one leaked bit.
    #
    # A row filter cannot be applied through a view, so a write that READS one is declined
    # exactly as a read of it is. The rewrite below cannot see through a view either, so without
    # this a filter on the view's base table reached nothing: measured, an INSERT selecting from
    # a governed view was approved and copied every row into an ungoverned table, where a later
    # plain SELECT returns them forever. The read decider has had this floor since M27; the write
    # decider had no `views` parameter at all, so it could not have run it even in principle.
    ungoverned = check_views(shaped, {} if views is None else views, set(policy.row_filters),
                             known=set(policy.policy_schema),
                             granted=set(visible))
    if ungoverned is not None:
        return ungoverned

    # Provenance is read from the statement as ASKED, before the rewrite touches it -- the same
    # ordering, and the same reason, as the read path. The RLS predicate reads whatever the POLICY
    # names, so reporting the rewritten tree tells the caller which tables their own policy
    # consults; an entitlements table is the standard shape and one no caller is granted (M28).
    #
    # `base_tables`, not `find_all` minus a flat set of CTE aliases (M49). That construct is the
    # one M31 filed and this field kept, because it reads like a log line and is not: it is the
    # ACL `_retrieve_examples` filters on, so a CTE named after a governed table deletes that
    # table from the list and an example whose SQL names it is then shown to a caller who may not
    # read it. Same resolver as the three guards, so the premise cannot drift apart again.
    #
    # The target is unioned in because a PLAIN INSERT does not read it, so `base_tables`
    # correctly omits it while the audit record must still name it. Every other write shape now
    # arrives with the target already in the read set: `_target_reads` models an UPDATE/DELETE
    # target, and an upsert's, as the reads they are. Earlier drafts of this comment claimed the
    # asymmetry ran the other way, then that the target was absent from every resolving shape --
    # both measured false, and the second was describing a defect rather than a design.
    tables = sorted({object_key(t) for t in base_tables(shaped, target)} | {tgt})

    # RLS: the read path's implementation, not a second copy of it. This block used to filter
    # only the table being WRITTEN -- so every table a write READ was ungoverned (M30), and the
    # filter was spliced in without validation (M7). Both are one defect: there were two
    # implementations and the second was wrong. `apply_row_filters_to_write` wraps the reads and
    # conjoins the target, using the same `_validate_filter` the read decider uses.
    # Codex's review of #111: not where the read list fell back, where the rewrite would filter a
    # CTE named like a filtered table as if it were the table.
    # M50, at the same point as on the read path and for the same reason.
    misplaced = check_filter_targets(policy)
    if misplaced is not None:
        return misplaced
    unplaceable = unplaceable_policy(shaped, policy, target)
    if unplaceable is not None:
        return unplaceable
    asked = shaped.copy()  # M51: the write as asked, before the row-filter rewrite
    governed, narrowed = apply_row_filters_to_write(
        shaped, policy, visible, _target_node(shaped), dialect, executes_as=target)
    if isinstance(governed, Refusal):
        return governed
    shaped = governed

    plan_sql = shaped.sql(dialect=dialect)
    target_sql = sqlglot.transpile(plan_sql, read=dialect, write=target)[0]
    asked_plan = asked.sql(dialect=dialect)
    asked_sql = (target_sql if asked_plan == plan_sql
                 else sqlglot.transpile(asked_plan, read=dialect, write=target)[0])
    if adapter is not None:
        # the snapshot can be stale, and only the source knows the truth. `prove` picks the proof
        # the source understands -- `EXPLAIN` is not Oracle syntax (see mnemiq.sql.prove).
        refused = prove(adapter, target_sql)
        if refused is not None:
            return refused

    return ApprovedWrite(plan_sql=plan_sql, target_sql=target_sql, target=tgt, tables=tables,
                         narrowed=narrowed, asked_sql=asked_sql)
