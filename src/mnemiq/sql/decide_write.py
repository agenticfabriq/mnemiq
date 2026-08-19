from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.cls import check_cls
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import apply_row_filters_to_write
from mnemiq.sql.verdict import ApprovedWrite, Refusal, RefusalCode

_WRITE_ROOTS = (exp.Insert, exp.Update, exp.Delete)


def _target_node(ast: exp.Expression) -> exp.Table | None:
    """The table node being written, as a NODE. RLS needs the identity, not the name: the target
    is the one table a write must not wrap in a derived table, and the same name can appear
    elsewhere in the statement as an ordinary read that must be."""
    this = ast.this
    if isinstance(this, exp.Schema):  # INSERT INTO t (cols)
        this = this.this
    if isinstance(this, exp.Table):
        return this
    return ast.find(exp.Table)


def _target_table(ast: exp.Expression) -> str | None:
    node = _target_node(ast)
    return node.name if node is not None else None


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

    tgt = _target_table(shaped)
    if tgt is None or not grants.allows_write(tgt):
        return Refusal(
            code=RefusalCode.UNAUTHORIZED_WRITE,
            message=f"You may not write to {tgt!r}.",
            subject=tgt,
        )

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

    cte_names = {c.alias_or_name for c in shaped.find_all(exp.CTE)}
    tables = sorted({t.name for t in shaped.find_all(exp.Table)} - cte_names)
    return ApprovedWrite(plan_sql=plan_sql, target_sql=target_sql, target=tgt, tables=tables)
