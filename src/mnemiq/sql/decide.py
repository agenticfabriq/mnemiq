from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.contract import ViewDefinition
from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.cls import check_cls
from mnemiq.sql.guard import MAX_ROWS, check_shape
from mnemiq.sql.lint import lint
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.prove import prove
from mnemiq.sql.qualify import expand_tables, object_key
from mnemiq.sql.scope import base_tables
from mnemiq.sql.rls import apply_row_and_mask
from mnemiq.sql.values_check import check_values
from mnemiq.sql.views import check_views
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode, Verdict


def decide(
    sql: str,
    visible: dict[str, set[str]],
    adapter=None,
    dialect: str = "duckdb",
    target: str = "postgres",
    max_rows: int = MAX_ROWS,
    values=None,
    policy: AccessPolicy | None = None,
    registry: dict[str, str] | None = None,
    views: dict[str, ViewDefinition] | None = None,
) -> Verdict:
    """The deterministic decider: shape, then access, then proof against the real source.

    Order matters. Shape first, because a DROP is a DROP regardless of what it names. Access
    second, because an unauthorized query must never reach the database -- the DB's own
    permission error is not the control: by the time it fires, we have already confirmed to
    the model that the table exists.
    """
    policy = policy or AccessPolicy()
    shaped = check_shape(sql, dialect=dialect, max_rows=max_rows)
    if isinstance(shaped, Refusal):
        return shaped

    refusal = check_access(shaped, visible)
    if refusal is not None:
        return refusal

    cls = check_cls(shaped, policy)  # column deny / mask-in-predicate
    if cls is not None:
        return cls

    # A row filter cannot be applied through a view, so a view that reads a filtered table is
    # declined rather than answered past. The floor, deliberately -- see `check_views` (M27).
    ungoverned = check_views(shaped, {} if views is None else views, set(policy.row_filters),
                             known=set(policy.policy_schema))
    if ungoverned is not None:
        return ungoverned

    violation = lint(shaped)
    if violation is not None:
        # silently-wrong SQL: runs fine, wrong answer. Repairable -- the corrector fixes it.
        return Refusal(code=RefusalCode.LOGIC_LINT, message=violation.message)

    if values is not None:
        # a filter literal that does not exist in its column runs fine and answers wrong --
        # another silently-wrong class the corrector fixes, given the real values to pick from.
        # M5: the index is unfiltered and this runs before the RLS rewrite below, so the
        # refusal must know which tables this identity only sees a slice of.
        grounding = check_values(shaped, visible, values, row_filtered=set(policy.row_filters))
        if grounding is not None:
            return grounding

    # Provenance is read from the query as ASKED, before either rewrite touches it.
    #
    # It must precede `expand_tables`, which rewrites the identifiers. It must also precede the
    # RLS rewrite, for a reason that only appeared once filters could reach through another
    # table (M28): the injected predicate reads whatever the POLICY names, so reporting the
    # rewritten tree would tell the caller which tables their own policy consults -- an
    # entitlements table being the standard shape, and one no caller is granted. What the
    # engine reads on the policy's behalf is not the caller's lineage. The audit trace is the
    # channel for that, and it is not this one.
    #
    # The same move corrects `columns`: the derived table projects every visible column of a
    # filtered table, so reading it afterwards reported columns the query never mentioned.
    #
    # `base_tables`, not `find_all` minus a flat set of CTE aliases (M49). That construct is the
    # one M31 filed and this field kept, because it reads like a log line and is not: it is the
    # ACL `_retrieve_examples` filters on, so a CTE named after a governed table deletes that
    # table from the list and an example whose SQL names it is then shown to a caller who may not
    # read it. Same resolver as the three guards, so the premise cannot drift apart again.
    tables = sorted({object_key(t) for t in base_tables(shaped)})
    columns = sorted({c.name for c in shaped.find_all(exp.Column)})

    if not policy.empty:
        shaped = apply_row_and_mask(shaped, policy, visible, dialect=dialect)
        if isinstance(shaped, Refusal):
            return shaped

    # Federation: 'catalog.table' -> 'catalog.schema.table' so DuckDB resolves it. No-op single-source.
    expand_tables(shaped, registry or {})

    plan_sql = shaped.sql(dialect=dialect)
    target_sql = sqlglot.transpile(plan_sql, read=dialect, write=target)[0]

    if adapter is not None:
        # the snapshot can be stale, and only the source knows the truth. `prove` picks the proof
        # the source understands -- `EXPLAIN` is not Oracle syntax (see mnemiq.sql.prove).
        refused = prove(adapter, target_sql)
        if refused is not None:
            return refused

    return Approved(plan_sql=plan_sql, target_sql=target_sql, tables=tables, columns=columns)
