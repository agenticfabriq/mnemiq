from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import ViewDefinition
from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.cls import check_cls
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.qualify import object_key
from mnemiq.sql.rls import apply_row_filters_to_write
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


def _target_node(ast: exp.Expression) -> exp.Table | None:
    """The table node being written, as a NODE, or None if this engine cannot name exactly one.

    RLS needs the identity, not the name: the target is the one table a write must not wrap in a
    derived table, and the same name can appear elsewhere in the statement as an ordinary read
    that must be.

    None means REFUSE, never "guess". This used to fall back to `ast.find(exp.Table)` -- the
    first table in the tree -- which is a source for any shape whose target is not `this`.
    Measured: `DELETE s FROM t JOIN s` resolved to `t`, so the grant was checked against a table
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
        # Refused by shape, which is a question with no unmodelled cases -- either the arg is set
        # or it is not. Modelling the scope instead is M48's job in the v1 lane; until it lands,
        # a shape this engine cannot see into is declined rather than approved blind. The same
        # CTE spelled INSIDE the INSERT parses into the projection, where `build_scope` sees it,
        # and stays approved and governed.
        return Refusal(
            code=RefusalCode.UNSCOPED_CTE,
            message=("Put the WITH clause inside the statement rather than before it, or inline "
                     "it: a leading WITH is not visible to this engine's policy checks."),
        )
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

    refusal = check_access(shaped, visible)  # every referenced table readable, columns exist
    if refusal is not None:
        return refusal

    # A write needs RAW access: a masked column is treated as denied for writes.
    write_cls = check_cls(shaped, AccessPolicy(denied=policy.denied | policy.masked))
    if write_cls is not None:
        return write_cls

    # Provenance is read from the statement as ASKED, before the rewrite touches it -- the same
    # ordering, and the same reason, as the read path. The RLS predicate reads whatever the POLICY
    # names, so reporting the rewritten tree tells the caller which tables their own policy
    # consults; an entitlements table is the standard shape and one no caller is granted (M28).
    # Computed here, above the rewrite, rather than at the return: measured below the rewrite it
    # answered `['claim', 'entitlement', 'scratch']` for a caller granted neither `entitlement`
    # nor sight of it. What the engine reads on the policy's behalf is not the caller's lineage.
    cte_names = {c.alias_or_name for c in shaped.find_all(exp.CTE)}
    tables = sorted({object_key(t) for t in shaped.find_all(exp.Table)} - cte_names)

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
    # (access -> CLS -> views) is wrong on this path for the reason that keeps recurring here: an
    # INSERT's target is not a base table, so `check_access` never covers it, and `check_views`
    # walked it anyway. (An UPDATE/DELETE target IS a base table and `check_access` refuses it
    # already -- the asymmetry is INSERT's alone, as `_target_table` says above.) `INSERT INTO hidden_view ...` then refused UNGOVERNED_VIEW naming the
    # view AND its row-filtered base -- three facts about objects the caller holds no grant on,
    # and distinguishable from the plain-target refusal, so a probe rather than one leaked bit.
    #
    # A row filter cannot be applied through a view, so a write that READS one is declined
    # exactly as a read of it is. The rewrite below cannot see through a view either, so without
    # this a filter on the view's base table reached nothing: measured, an INSERT selecting from
    # a governed view was approved and copied every row into an ungoverned table, where a later
    # plain SELECT returns them forever. The read decider has had this floor since M27; the write
    # decider had no `views` parameter at all, so it could not have run it even in principle.
    ungoverned = check_views(shaped, views or {}, set(policy.row_filters),
                             known=set(policy.policy_schema))
    if ungoverned is not None:
        return ungoverned

    # RLS: the read path's implementation, not a second copy of it. This block used to filter
    # only the table being WRITTEN -- so every table a write READ was ungoverned (M30), and the
    # filter was spliced in without validation (M7). Both are one defect: there were two
    # implementations and the second was wrong. `apply_row_filters_to_write` wraps the reads and
    # conjoins the target, using the same `_validate_filter` the read decider uses.
    governed = apply_row_filters_to_write(shaped, policy, visible, _target_node(shaped), dialect)
    if isinstance(governed, Refusal):
        return governed
    shaped = governed

    plan_sql = shaped.sql(dialect=dialect)
    target_sql = sqlglot.transpile(plan_sql, read=dialect, write=target)[0]
    if adapter is not None:
        try:
            adapter.execute(f"EXPLAIN {target_sql}")  # plans without executing; proves the SQL
        except Exception as exc:
            return Refusal(
                code=RefusalCode.EXPLAIN_FAILED,
                message=f"The source rejected this query: {exc}",
            )

    return ApprovedWrite(plan_sql=plan_sql, target_sql=target_sql, target=tgt, tables=tables)
