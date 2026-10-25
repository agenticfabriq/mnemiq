from __future__ import annotations

from dataclasses import replace

import sqlglot
from sqlglot import exp

from mnemiq.contract import ViewDefinition
from mnemiq.redact import policy_safe_detail
from mnemiq.sql.functions import (
    UnreadableCalls,
    called_names,
    inventory_from,
    opaque_columns,
)
from mnemiq.sql.authz_guard import (
    check_access,
    check_opaque_columns,
    check_unmodelled_calls,
    check_unmodelled_source,
)
from mnemiq.sql.cls import check_cls
from mnemiq.sql.fanout_check import KeyFacts, check_fanout
from mnemiq.sql.guard import MAX_ROWS, check_shape, niladic_names
from mnemiq.sql.lint import lint
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.prove import prove_governed
from mnemiq.sql.qualify import expand_tables, object_key
from mnemiq.sql.lineage import lineage_for
from mnemiq.sql.scope import (base_tables, check_policy_target, fields_as_members, known_columns,
                              scope_resolved)
from mnemiq.sql.reading import UNSUPPORTED_RAISES, as_duckdb_reads, kept, rendered, unprintable
from mnemiq.sql.rls import apply_row_and_mask, check_filter_targets, unplaceable_policy
from mnemiq.sql.values_check import check_values
from mnemiq.sql.views import check_views
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode, Verdict


def _render_target(tree: exp.Expression, plan_sql: str, dialect: str, target: str,
                   *, policys: bool = False) -> str | Refusal:
    """`plan_sql` in the target's dialect, through the row-cap fallback where sqlglot needs it.
    One path for the executed statement and the one a caller is shown (M51), so the two agree
    whenever the policy changed nothing. A plan the target's generator cannot print is a refusal
    (`unprintable`), the policy's where `policys`, since the statement as asked printed."""
    try:
        try:
            return rendered(plan_sql, dialect, target)
        except sqlglot.errors.ParseError:
            return _transpile_under_the_cap(tree, dialect, target)
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        return unprintable(target, exc, policys=policys, tree=tree)


def _transpile_under_the_cap(shaped: exp.Expression, dialect: str, target: str) -> str:
    """The plan's round trip to the target with the row cap taken off, and the cap put back after.

    sqlglot 30.12 cannot parse a LIMIT straight after `GROUP BY ROLLUP`, `CUBE`, `GROUPING SETS` or `WITH
    ROLLUP`, and the cap is what puts one there, so re-reading the capped plan raised out of `decide` (M137).
    The round trip itself is still wanted: it normalizes dialect types (BigQuery's FLOAT is 64-bit and re-reads
    as DuckDB's DOUBLE, where the tree rendered straight to DuckDB says REAL and loses precision). So the plan is
    re-read without its LIMIT, which sqlglot can parse, and the LIMIT goes back onto the re-read tree.
    """
    limit = shaped.args.get("limit")
    uncapped = shaped.copy()
    uncapped.set("limit", None)
    reread = as_duckdb_reads(sqlglot.parse_one(uncapped.sql(dialect=dialect), read=dialect))
    # The lost-TRY_CAST check re-reads the rendered text, so it runs on the uncapped render: the capped
    # one is the text sqlglot cannot parse, which is why this fallback exists (the review gate on round 68).
    kept(reread, reread.sql(dialect=target, **UNSUPPORTED_RAISES), target)
    if limit is not None:
        reread.set("limit", limit.copy())
    return reread.sql(dialect=target, **UNSUPPORTED_RAISES)


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
    keys: KeyFacts | None = None,
) -> Verdict:
    """The deterministic decider: shape, then access, then proof against the real source.

    Order matters. Shape first, because a DROP is a DROP regardless of what it names. Access
    second, because an unauthorized query must never reach the database -- the DB's own
    permission error is not the control: by the time it fires, we have already confirmed to
    the model that the table exists.
    """
    policy = policy or AccessPolicy()
    # M88: `visible` is the schema, and the shape check was guessing about the same names at
    # the same moment -- is a bare `claim_amount` the column or the whole row? Handing it the
    # map two lines before `check_access` gets it turns three findings' worth of fail-closed
    # false refusals into a question with an answer.
    shaped = check_shape(sql, dialect=dialect, max_rows=max_rows, executes_as=target,
                         columns=visible)
    if isinstance(shaped, Refusal):
        return shaped
    # M165: `c.secret.status` reads a field of `secret`, and every check below reads columns, so
    # the reference is made a field access over `secret` before any of them runs.
    ambiguous = fields_as_members(shaped, target, known_columns(visible, policy))
    if ambiguous is not None:
        return ambiguous
    # M194: a bare `current_date` where a known table has a column so named reads the column
    # through the function's node, which no check below sees.
    named = niladic_names(shaped, known_columns(visible, policy), dialect, target)  # the advice in the dialect the model writes
    if named is not None:
        return named
    # M150: before any policy is applied, refuse an engine whose CTE binding the rewrite misreads.
    unsupported = check_policy_target(shaped, policy, target)
    if unsupported is not None:
        return unsupported
    row_cap = shaped.meta.get("row_cap")  # read now: the rewrites below build new trees

    refusal = check_access(shaped, visible, target, columns=known_columns(visible, policy))
    if refusal is not None:
        return refusal
    # M65: what `check_access` reads -- `base_tables` -- is empty for a source it cannot model,
    # and an empty list passes. After it, so a source it already refuses keeps its refusal.
    unmodelled = check_unmodelled_source(shaped)
    if unmodelled is not None:
        return unmodelled

    # Beside `check_access` because it defends the same premise: that the decider sees every
    # table the query reads. `check_access` walks `exp.Table` nodes, and a function in
    # projection position is not one -- so an opaque call slipped past both it and the RLS
    # rewrite (M43). This does not decide whether such a call is dangerous; it says the query
    # cannot be decided, which is the true thing.
    # One inventory for the whole decision. `lineage_for` reads it too, and asking twice would
    # let the gate and the report disagree about the same source in the same statement.
    # Asked about the names this statement calls, where per-name is sound (M101): reading a
    # whole Oracle dictionary cost 4.8 s a decision at 107,886 synonyms. A statement whose calls
    # cannot be read gets the whole-source answer, so the guard decides it as it always did.
    try:
        called = called_names(shaped, dialect, target)
    except UnreadableCalls:
        called = None
    functions = inventory_from(adapter, called=called)
    # Both dialects, because `called_names` renders to find what the tree does not carry, and
    # more spellings can only over-refuse. `dialect` is what this was parsed as, `target` what
    # will run it.
    opaque = check_unmodelled_calls(shaped, functions, dialect, target)
    if opaque is not None:
        return opaque

    # A COLUMN can be the call. A virtual column runs its stored expression on read, so
    # `SELECT id, leaked FROM t` executes a UDF that the statement never names -- measured on
    # Oracle, returning an SSN from an ungranted table while every check above passed it (M100).
    # Third shape of the same thing, after `count(*)` reaching `count_star` and `id + 1`
    # reaching a macro named `+`, and answered the same way: ask the source, do not enumerate.
    computed = check_opaque_columns(shaped, opaque_columns(adapter, functions), target,
                                    columns=known_columns(visible, policy))
    if computed is not None:
        return computed

    cls = check_cls(shaped, policy, target,  # column deny / mask-in-predicate
                    columns=known_columns(visible, policy))
    if cls is not None:
        return cls

    # A row filter cannot be applied through a view, so a view that reads a filtered table is
    # declined rather than answered past. The floor, deliberately -- see `check_views` (M27).
    ungoverned = check_views(shaped, {} if views is None else views, set(policy.row_filters),
                             known=set(policy.policy_schema),
                             granted=set(visible))
    if ungoverned is not None:
        return ungoverned

    violation = lint(shaped)
    if violation is not None:
        # silently-wrong SQL: runs fine, wrong answer. Repairable -- the corrector fixes it.
        return Refusal(code=RefusalCode.LOGIC_LINT, message=violation.message)

    # M50: a filter key that names nothing may be the misspelled key of any table this reads, so
    # nothing is approved while one is held. Not at the top: a statement an earlier check refuses
    # keeps that check's more precise reason (a view inventory that failed, a body that will not
    # parse), and those checks name objects, never row values. But BEFORE value grounding, whose
    # refusal lists values from the unfiltered index and withholds only for the filters it is
    # told of -- none, for a dropped key. Placed after it first, and Codex's review of #118
    # measured the leak: a mistyped key listed `east` from rows a correct key withholds.
    misplaced = check_filter_targets(policy)
    if misplaced is not None:
        return misplaced

    if values is not None:
        # a filter literal that does not exist in its column runs fine and answers wrong --
        # another silently-wrong class the corrector fixes, given the real values to pick from.
        # M5: the index is unfiltered and this runs before the RLS rewrite below, so the
        # refusal must know which tables this identity only sees a slice of.
        grounding = check_values(shaped, visible, values, row_filtered=set(policy.row_filters),
                                 masks=policy.masks)
        if grounding is not None:
            return grounding

    if keys is not None:
        # An aggregate over rows a join has multiplied runs fine and answers inflated (M109) --
        # the third silently-wrong class, and the one a literal fix cannot reach, so it goes last.
        # `keys` comes from the snapshot's column profiles; None means the guard is off.
        inflated = check_fanout(shaped, visible, keys, dialect=dialect)
        if inflated is not None:
            return inflated

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
    tables = sorted({object_key(t) for t in base_tables(shaped, target)})
    # Beside the list, never apart from it (M56): a bare list is worse than no list, because
    # absent reads as "not recorded" and `[]` reads as "nothing was read". Computed where the list
    # is computed, so the two cannot come from different paths and disagree -- M7's shape.
    #
    # `views if views is not None`, NOT `views or {}`. `ViewInventory` subclasses dict, so a source
    # with genuinely no views is FALSY and `or` would swap an inventory that knows it was asked for
    # a bare `{}` that knows nothing -- M52's defect in one operator, inside the artifact built to
    # prevent it.
    # LIVE from the adapter, not the snapshot -- see `inventory_from` for why the two sources
    # differ.
    lineage = lineage_for(shaped, tables, {} if views is None else views,
                          functions=functions, called=called,
                          scope_resolved=scope_resolved(shaped, target))
    columns = sorted({c.name for c in shaped.find_all(exp.Column)})

    narrowed: list = []
    # M51: the statement as asked, kept before any policy rewrite -- what a caller is shown.
    asked = shaped.copy()
    unplaceable = unplaceable_policy(shaped, policy, target)
    if unplaceable is not None:
        return unplaceable
    if not policy.empty:
        shaped, narrowed = apply_row_and_mask(shaped, policy, visible, dialect=dialect,
                                              executes_as=target)
        if isinstance(shaped, Refusal):
            return shaped

    # Federation: 'catalog.table' -> 'catalog.schema.table' so DuckDB resolves it. No-op single-source.
    expand_tables(shaped, registry or {})
    expand_tables(asked, registry or {})

    # The statement as asked first, then the governed plan: a rewrite the target cannot print where
    # the statement prints is the policy's refusal, not the model's.
    asked_plan = asked.sql(dialect=dialect)
    asked_sql = _render_target(asked, asked_plan, dialect, target)
    if isinstance(asked_sql, Refusal):
        return asked_sql
    plan_sql = shaped.sql(dialect=dialect)
    target_sql = (asked_sql if asked_plan == plan_sql
                  else _render_target(shaped, plan_sql, dialect, target, policys=True))
    if isinstance(target_sql, Refusal):
        return target_sql

    if adapter is not None:
        # the snapshot can be stale, and only the source knows the truth. `prove` picks the proof
        # the source understands -- `EXPLAIN` is not Oracle syntax (see mnemiq.sql.prove).
        # A refusal of what the policy added is the operator's, not the model's (M158).
        refused = prove_governed(adapter, target_sql, asked_sql)
        if refused is not None:
            # EXPLAIN_FAILED: what the source refused is the policy's rewrite, so its words can
            # quote the policy and the rows it removes, and the model repairs against them. `prove`
            # has logged them whole; under a policy the refusal keeps the error's class only
            # (`policy_safe_detail`). POLICY_REWRITE_REJECTED carries no source words at all;
            # SOURCE_UNAVAILABLE carries the driver's, which name no row, kept to their class too.
            return replace(refused, source_detail=policy_safe_detail(
                refused.source_detail, asked=asked_sql, executed=target_sql, dialect=target))

    return Approved(plan_sql=plan_sql, target_sql=target_sql, tables=tables,
                    lineage=lineage, columns=columns, narrowed=narrowed, row_cap=row_cap,
                    asked_sql=asked_sql, asked_plan=asked_plan)
