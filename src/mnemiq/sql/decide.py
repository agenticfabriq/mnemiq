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
        # Two passes, because inlining REMOVES the object a view-keyed policy is attached to.
        #
        # First the objects the caller actually named. A view granted on its own carries its
        # own filters and masks -- `customer_list.email` is masked, `customer_list: sid = 1`
        # filters -- and both are expressible on the view's output. Inlining first threw the
        # node away and the policy with it, which is how a masked column came back in the
        # clear (Codex review, 2026-08-12).
        injected: set[int] = set()
        shaped = apply_row_and_mask(
            shaped, policy, visible, dialect=dialect, injected=injected
        )
        if isinstance(shaped, Refusal):
            return shaped

        # Then resolve views to their bases, so a filter on a base lands at the leaves rather
        # than on a view's output -- which for an aggregating view is not a weaker fix but an
        # impossible one, the tenancy column having been grouped away (M27).
        expanded = inline_views(shaped, views or {}, policy.policy_schema,
                                protect=injected)
        if isinstance(expanded, Refusal):
            return expanded
        shaped = expanded.ast

        # A DENIED column is not readable through a view either. `check_cls` ran before
        # inlining, when the body was still one opaque node, and the rewrite below only knows
        # how to NULL a masked column -- so a denied base column came back raw through a
        # granted view. Masks were fixed here and denials were not, because nothing asked what
        # ELSE keys on the object inlining removes.
        #
        # Scoped to columns that resolve to an INTRODUCED table: re-running check_cls over the
        # whole tree would fire on the rewriter's own projections, which list every visible
        # column of a filtered table including denied ones.
        # Only what the CALLER reads. `columns` was taken from the query as asked, so a body
        # that merely mentions a restricted column in a projection nobody selected is not a
        # disclosure -- refusing on that would make any view naming one unusable.
        exposed = {
            pair
            for published in columns
            for pair in expanded.exposes.get(published, ())
        }
        denied_read = exposed & policy.denied
        if denied_read:
            name = sorted(denied_read)[0][1]
            return Refusal(
                code=RefusalCode.UNAUTHORIZED_COLUMN,
                message=f"You may not read the column {name!r}.",
                subject=name,
            )

        # Finally the bases inlining introduced. Restricted to the NODES it added: a caller may
        # also name one of those tables directly, and that reference was already rewritten in
        # the first pass. A row filter's own subquery (M28) is excluded for free -- it is
        # injected during a pass, never scanned by one.
        if expanded.nodes:
            shaped = apply_row_and_mask(
                shaped, policy, expanded.introduced, dialect=dialect, only=expanded.nodes
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
