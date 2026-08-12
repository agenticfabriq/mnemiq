from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.contract import ViewDefinition
from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.cls import check_cls
from mnemiq.sql.guard import MAX_ROWS, check_shape
from mnemiq.sql.lint import lint
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.qualify import expand_tables, object_key
from mnemiq.sql.rls import apply_row_and_mask
from mnemiq.sql.values_check import check_values
from mnemiq.sql.views import inline_views
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
    cte_names = {cte.alias_or_name for cte in shaped.find_all(exp.CTE)}
    tables = sorted({object_key(t) for t in shaped.find_all(exp.Table)} - cte_names)
    columns = sorted({c.name for c in shaped.find_all(exp.Column)})

    if not policy.empty:
        # Views resolve to their base tables FIRST, so the rewrite below lands the filter at
        # the leaves rather than on a view's output -- which for an aggregating view is not a
        # weaker fix but an impossible one, the tenancy column having been grouped away (M27).
        #
        # Gated on an active policy for the same reason the rest of this is: an ungoverned
        # deployment gains nothing from the rewrite and should not pay its risk.
        expanded = inline_views(shaped, views or {}, policy.policy_schema)
        if isinstance(expanded, Refusal):
            return expanded
        shaped = expanded.ast

        # RLS + source-side mask rewrite (filtered/masked tables -> derived). `visible` is
        # widened by exactly the base tables inlining introduced: they are absent from the
        # caller's map, and without them the rewriter skips them and the view returns
        # unfiltered rows -- M27 surviving its own fix. A row filter's subquery is also absent
        # from `visible` and must NOT be filtered; it is excluded because it is injected after
        # this scan, and now also because it is not in `introduced`.
        shaped = apply_row_and_mask(
            shaped, policy, {**visible, **expanded.introduced}, dialect=dialect
        )
        if isinstance(shaped, Refusal):
            return shaped

    # Federation: 'catalog.table' -> 'catalog.schema.table' so DuckDB resolves it. No-op single-source.
    expand_tables(shaped, registry or {})

    plan_sql = shaped.sql(dialect=dialect)
    target_sql = sqlglot.transpile(plan_sql, read=dialect, write=target)[0]

    if adapter is not None:
        try:
            adapter.execute(f"EXPLAIN {target_sql}")
        except Exception as exc:
            # the snapshot can be stale, and only the source knows the truth
            return Refusal(
                code=RefusalCode.EXPLAIN_FAILED,
                message=f"The source rejected this query: {exc}",
            )

    return Approved(plan_sql=plan_sql, target_sql=target_sql, tables=tables, columns=columns)
