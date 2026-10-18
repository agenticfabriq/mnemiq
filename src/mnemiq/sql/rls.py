from __future__ import annotations

from dataclasses import dataclass
import logging

import sqlglot
from sqlglot import exp

from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.qualify import names_one_object, object_key
from mnemiq.sql.scope import base_tables, column_tables, cte_references, scope_resolved
from mnemiq.sql.verdict import Refusal, RefusalCode

logger = logging.getLogger(__name__)


# The stand-in for the table a filter is attached to, so the predicate can be validated as the
# query it becomes. Chosen to be unspellable in SQL a policy author would write.
_SUBJECT = "__mnemiq_filtered__"

# Engines that may evaluate a statement's own predicates on rows a row filter removes, and the
# derived tables for which therefore carry a fence (M160). Oracle merges a filtered derived table
# into the outer query, or pushes the outer predicates into it, so the caller's condition can run
# before the policy's: measured on Oracle Free 23, where a `NO_MERGE` hint did not stop it. A view
# reading ROWNUM is neither merged nor has predicates pushed into it, and `ROWNUM > 0` holds for
# every row, so the fence changes which rows the outer query sees first and never which rows.
_FENCED = frozenset({"oracle"})


def _validate_filter(
    filt: str,
    cols: set[str],
    dialect: str,
    policy_schema: dict[str, set[str]] | None = None,
    *,
    executes_as: str | None = None,
) -> exp.Expression | None:
    """Parse a row filter and confirm it can only speak about what it is entitled to; else None.

    Two scopes, two rules. **Outside** a subquery the predicate is a WHERE on one table and may
    name only that table's own columns -- nothing else is in scope there. **Inside** a subquery
    it may name any table in `policy_schema`, the policy author's visibility.

    M28: the single rule was the outer one, applied everywhere, which made child-table tenancy
    structurally inexpressible -- `payment` has no `store_id`, so the only way to say "payments
    belonging to this store's customers" is to reach through `customer`, and that was refused.
    The shipped pagila policy did exactly that, so the two tables the coverage warning told the
    author to filter became unqueryable instead.

    The subquery is resolved against the POLICY's visibility and never the caller's, because
    the policy is what defines the caller's boundary and resolving it through that boundary is
    circular -- and because an entitlements table, the standard shape for this, is one no
    caller is ever granted. Without a `policy_schema` a subquery cannot be checked at all, so
    it is refused: an unvalidatable filter must not become a filter that silently does nothing.
    """
    try:
        expr = sqlglot.parse_one(filt, read=dialect)
    except Exception:
        return None
    if not isinstance(expr, exp.Condition):
        return None  # a statement, not a predicate -- `DELETE FROM t` is not a row filter

    # Validate the predicate as the query it will become, so the ONE scope-aware resolver
    # answers "which table does this column belong to" here as well.
    #
    # This used to build its own alias map with a descendant-wide `find_all`, so a nested
    # `FROM other e` overwrote an outer `FROM entitlement e` and an invalid column was checked
    # against the wrong table. That is the third place a flat alias map has been a bypass --
    # `check_access`, `check_cls`, and here. There is now one resolver and no local maps.
    if _SUBJECT in filt:
        # The stand-in is a real identifier at execution time and is exempt from the table
        # check below, so a filter naming it would read whatever table happens to carry that
        # name. Unspellable by convention is not unspellable.
        return None
    schema = dict(policy_schema or {})
    schema[_SUBJECT] = set(cols)
    try:
        wrapped = sqlglot.parse_one(f"SELECT 1 FROM {_SUBJECT} WHERE {filt}", read=dialect)
    except Exception:
        return None
    resolved = column_tables(wrapped, executes_as or dialect)
    if resolved is None:
        return None  # scopes unreadable -> the filter cannot be validated, so it is refused

    for table in base_tables(wrapped, executes_as or dialect):
        if any("." in part for part in (table.text("catalog"), table.text("db"), table.name)):
            return None  # `"pg.entitlement"` is not `pg.entitlement`, and a dotted key cannot
        key = object_key(table)
        if key != _SUBJECT and key not in schema:
            return None  # a table the policy author has not got

    for column in wrapped.find_all(exp.Column):
        owner = resolved.get(id(column))
        if owner is None:
            if column.table:
                return None  # qualified by something no scope here defines
            # Unqualified: the subquery's own tables, or a correlated reference to the row
            # being filtered. Fail-closed across both rather than resolving ambiguity.
            reachable = {c for t in base_tables(wrapped, executes_as or dialect)
                         for c in schema.get(object_key(t), ())}
            if column.name not in reachable:
                return None
        elif column.name not in schema.get(owner, set()):
            return None
    return expr


# Conversions that raise on a value they cannot read (M159). A row filter runs over every row of
# its table, the ones it removes included, so one bad value in a hidden row fails every question on
# the table, and the error can quote that value. sqlglot's typed parses, and Oracle's conversions it
# leaves untyped. TRY_CAST and a DEFAULT ... ON CONVERSION ERROR clause do not raise.
_PARSES = (exp.ToNumber, exp.StrToDate, exp.StrToTime)
_UNTYPED_CONVERSIONS = frozenset({"to_binary_double", "to_binary_float", "to_timestamp_tz",
                                  "to_dsinterval", "to_yminterval"})
# Text read as JSON, which DuckDB refuses on malformed text and quotes ("Input: ..."): every
# scalar JSON reader in `duckdb_functions()` that does so in 1.5.4, measured, by how sqlglot parses
# it (a test re-measures the list, at the arguments it knows for each reader). Oracle's JSON functions default to NULL ON ERROR, so this is
# DuckDB's alone. Safe over a column the snapshot types JSON, or over `TRY_CAST(value AS JSON)`.
_JSON_PARSES = (exp.JSONExtract, exp.JSONExtractScalar, exp.ParseJSON, exp.JSONKeys, exp.JSONType)
_UNTYPED_JSON_PARSES = frozenset({
    "from_json", "from_json_strict", "json_array_length", "json_contains", "json_deserialize_sql",
    "json_exists", "json_merge_patch", "json_pretty", "json_structure", "json_transform",
    "json_transform_strict", "json_value"})
# The strict transforms raise where a WELL-FORMED value does not fit the structure asked for, and
# quote it (measured: `json_transform_strict(attrs, '{"a":"INTEGER"}')` over a JSON column whose
# `a` holds text; `json_transform` gives NULL there), so a typed column does not make them safe
# (Codex's review of #160). Only a constant cannot fail on a row.
_STRICT_JSON_TRANSFORMS = frozenset({"from_json_strict", "json_transform_strict"})
# A comparison converts a text column to the other side's type, row by row, in both engines:
# DuckDB on `=`, `<>` and IN (its ordering comparisons refuse at bind, on no row's value), Oracle
# on every comparison (an implicit TO_NUMBER or TO_DATE). Seen only where the snapshot types both
# sides; an untyped column compared to a number is not.
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NullSafeEQ,
                exp.NullSafeNEQ, exp.In, exp.Between)
_ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod, exp.IntDiv)
_INTEGER_RANK = {exp.DataType.Type.TINYINT: 1, exp.DataType.Type.SMALLINT: 2,
                 exp.DataType.Type.INT: 3, exp.DataType.Type.BIGINT: 4,
                 exp.DataType.Type.INT128: 5}
# Every value of these renders as text, so a cast of one to unbounded text cannot fail (DuckDB).
_RENDERS_AS_TEXT = (set(_INTEGER_RANK) | exp.DataType.NUMERIC_TYPES | exp.DataType.TEMPORAL_TYPES
                    | exp.DataType.TEXT_TYPES | {exp.DataType.Type.BOOLEAN})
# Each engine's non-raising spelling, as the refusal names it. Oracle's DEFAULT clause is kept by
# sqlglot on CAST and TO_NUMBER only: given a format argument, `CAST(d AS DATE DEFAULT NULL ON
# CONVERSION ERROR, 'YYYY-MM-DD')` is read as TO_DATE and rendered without the clause, and
# TO_DATE's own DEFAULT does not parse.
_SAFE_FORM = {
    "duckdb": ("TRY_CAST(value AS type), which gives NULL where the value does not convert "
               "(for JSON, read from TRY_CAST(value AS JSON), and json_transform rather than "
               "json_transform_strict)"),
    "oracle": ("CAST(value AS type DEFAULT NULL ON CONVERSION ERROR) with no format argument, or "
               "TO_NUMBER(value DEFAULT NULL ON CONVERSION ERROR)"),
}


def _cast_cannot_fail(source: str | None, to: exp.DataType, engine: str) -> bool:
    """Whether every value of a column typed `source` converts to `to` without error."""
    if source is None:
        return False  # the snapshot did not say: unknown, and refused
    try:
        known = exp.DataType.build(source, dialect=engine)
    except Exception:
        return False
    if known == to:
        return True
    if known.this in _INTEGER_RANK and to.this in _INTEGER_RANK:
        return _INTEGER_RANK[known.this] <= _INTEGER_RANK[to.this]
    return (engine == "duckdb" and to.this in (exp.DataType.Type.TEXT, exp.DataType.Type.VARCHAR)
            and not to.expressions and known.this in _RENDERS_AS_TEXT)


def _json_cannot_fail(source, type_of, engine: str) -> bool:
    """Whether the text a JSON function reads is already JSON, or NULL where it is not."""
    if source is None or source.find(exp.Column) is None:
        return True  # a constant cannot fail on a row
    if isinstance(source, exp.TryCast):
        # NULL where the text is not JSON. To any other type it is text again, which the reader
        # parses: `TRY_CAST(region AS VARCHAR)` raised as `region` does (Codex's review of #160).
        return source.to.this == exp.DataType.Type.JSON
    if not isinstance(source, exp.Column):
        return False
    known = type_of(source)
    try:
        return known is not None and exp.DataType.build(known, dialect=engine).this == (
            exp.DataType.Type.JSON)
    except Exception:
        return False


def _known_type(value, type_of, engine: str) -> exp.DataType | None:
    """`value`'s type where it is plain to see: a typed column, a literal, a cast, arithmetic."""
    while isinstance(value, exp.Paren):
        value = value.this
    if isinstance(value, exp.Boolean):
        return exp.DataType.build("BOOLEAN")
    if isinstance(value, exp.DateStrToDate):  # Oracle's DATE '...' literal
        return exp.DataType.build("DATE")
    if isinstance(value, exp.Column):
        known = type_of(value)
        try:
            return None if known is None else exp.DataType.build(known, dialect=engine)
        except Exception:
            return None
    if isinstance(value, exp.Cast):
        return value.to
    if isinstance(value, exp.Neg):
        value = value.this
    if isinstance(value, exp.Literal) and value.is_number or isinstance(value, _ARITHMETIC):
        return exp.DataType.build("DOUBLE")
    return None


def _converts_a_text_column(node: exp.Expression, type_of, engine: str) -> bool:
    """Whether a comparison sets a text column against a value of another type."""
    if isinstance(node, exp.In):
        sides = [node.this, *node.expressions]
        query = node.args.get("query")
        inner = query.this if isinstance(query, exp.Subquery) else query
        if isinstance(inner, exp.Select) and len(inner.selects) == 1:
            sides.append(inner.selects[0])
    elif isinstance(node, exp.Between):
        sides = [node.this, node.args.get("low"), node.args.get("high")]
    else:
        sides = [node.this, node.expression]
    text = other = False
    for side in sides:
        while isinstance(side, exp.Paren):
            side = side.this  # `(region) = 7` converts the column as `region = 7` does (Codex, #160)
        known = _known_type(side, type_of, engine)
        if known is None:
            continue
        if known.this in exp.DataType.TEXT_TYPES:
            text = text or isinstance(side, exp.Column)
        else:
            other = True
    return text and other


def raising_conversion(filt: str, table: str, dialect: str, policy: AccessPolicy, *,
                       executes_as: str | None = None) -> exp.Expression | None:
    """The first conversion in `table`'s row filter that can raise on a row's value, else None.

    For a filter `_validate_filter` accepted. Type-aware where the snapshot knows the operand's
    type, which takes a bare column: `CAST(id AS BIGINT)` over an INTEGER cannot fail, and is
    kept. An operand reading no column is a constant, which cannot fail on a row. Anything else
    is refused: an expression's type, or a column's the snapshot does not record, is unknown.
    """
    engine = (executes_as or dialect).lower()
    wrapped = sqlglot.parse_one(f"SELECT 1 FROM {_SUBJECT} WHERE {filt}", read=dialect)
    owners = column_tables(wrapped, engine) or {}

    def types_of(key: str) -> dict[str, str]:
        key = table if key == _SUBJECT else key
        return next((cols for obj, cols in policy.policy_types.items()
                     if names_one_object(key, [obj])), {})

    def type_of(column: exp.Column) -> str | None:
        owner = owners.get(id(column))
        if owner is None and not column.table:
            # Unqualified, which the resolver leaves unowned: the one table of the column's own
            # SELECT that has a column of that name, as `_validate_filter` reads it. None where no
            # table or more than one does -- a correlated reference among them -- so it is unknown.
            select = column.find_ancestor(exp.Select)
            having = [key for key in {object_key(t) for t in base_tables(select, engine)}
                      if column.name.lower() in types_of(key)] if select is not None else []
            owner = having[0] if len(having) == 1 else None
        return None if owner is None else types_of(owner).get(column.name.lower())

    for node in wrapped.args["where"].walk():
        if engine == "duckdb" and (isinstance(node, _JSON_PARSES) or (
                isinstance(node, exp.Anonymous) and node.name.lower() in _UNTYPED_JSON_PARSES)):
            # The typed readers parse their first argument; an untyped one may parse any (both of
            # `json_contains` and `json_merge_patch` do), so each argument must be safe.
            sources = [node.this] if isinstance(node, _JSON_PARSES) else node.expressions
            if isinstance(node, exp.Anonymous) and node.name.lower() in _STRICT_JSON_TRANSFORMS:
                if any(source.find(exp.Column) for source in sources):
                    return node  # raises on a well-formed value too, see _STRICT_JSON_TRANSFORMS
                continue
            if all(_json_cannot_fail(source, type_of, engine) for source in sources):
                continue
            return node
        if isinstance(node, _COMPARISONS):
            if _converts_a_text_column(node, type_of, engine):
                return node
            continue
        untyped = isinstance(node, exp.Anonymous) and node.name.lower() in _UNTYPED_CONVERSIONS
        if not (untyped or isinstance(node, (exp.Cast, *_PARSES))):
            continue
        if (isinstance(node, exp.TryCast) or node.args.get("safe")
                or node.args.get("default") is not None):
            continue  # TRY_CAST, or Oracle's DEFAULT ... ON CONVERSION ERROR
        operands = node.expressions if untyped else [node.this]
        if not any(arg.find(exp.Column) for arg in operands):
            continue  # a constant cannot fail on a row
        if (isinstance(node, exp.Cast) and isinstance(node.this, exp.Column)
                and _cast_cannot_fail(type_of(node.this), node.to, engine)):
            continue
        return node
    return None


def _raising_refusal(node: exp.Expression, table: str, engine: str) -> Refusal:
    safe = _SAFE_FORM.get(engine, "a form that cannot fail, or a column of the right type")
    # The conversion is the policy's text, so it goes to the operator's log only.
    logger.warning("row filter for %r: %s can raise on a row's value, which fails every read of "
                   "the table and can quote that value in the error; write %s (M159)",
                   table, node.sql(dialect=engine), safe)
    return Refusal(
        code=RefusalCode.INVALID_ROW_FILTER,
        message=(f"The row-access policy for {table!r} converts a value in a way that can fail on "
                 "the table's data, so it cannot be applied. The operator must rewrite the "
                 f"conversion as {safe}; the server log names it."),
        subject=table,
    )


def _derived_table(
    table: str, alias: str, cols: set[str], masked_cols: set[str], filt: exp.Expression | None,
    fence: bool = False,
) -> exp.Subquery:
    """(SELECT <cols, masked->NULL AS c> FROM table [WHERE filt [AND ROWNUM > 0]]) AS alias.

    `fence` only where there is a filter: a mask removes no row, and a merged mask still reads
    NULL, so only a filter has rows for the outer predicates to reach first (`_FENCED`)."""
    projections: list[exp.Expression] = []
    # Folded on both sides: `cols` carries the snapshot's column spelling and `masked_cols` the
    # policy's, and a mask that matches the table but not the column name nulls nothing.
    folded_masked = {m.lower() for m in masked_cols}
    for c in sorted(cols):
        if c.lower() in folded_masked:
            projections.append(exp.alias_(exp.null(), c))
        else:
            projections.append(exp.column(c))
    inner = exp.select(*projections).from_(exp.to_table(table))
    if filt is not None:
        inner = inner.where(filt)
        if fence:
            inner = inner.where(exp.GT(this=exp.column("ROWNUM"), expression=exp.Literal.number(0)))
    return exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(alias)))


@dataclass(frozen=True)
class Narrowing:
    """What the access policy did to ONE object, so an answer can say it was narrowed.

    TABLE-GRANULAR on purpose. Column granularity would require the loop to know which projected
    column each masked source column reaches, and it does not -- a masked column can be consumed by
    an aggregate and never appear in the output at all. Claiming per-column detail the loop cannot
    support is how a disclosure becomes a lie, so this names the OBJECT and stops there.

    Each flag has exactly one meaning and it is written on the field. Do not paraphrase them here:
    an earlier draft of this docstring glossed `columns` as "columns were masked on this table",
    which is the rewrite-based reading the field explicitly disclaims -- two contracts for one
    field, in the type whose purpose is to not over-claim. The field comment is the contract.

    It carries NO predicate and NO policy identity. The caller is entitled to know its answer was
    narrowed; it is not entitled to the rule that narrowed it, which would leak the shape of other
    principals' access.
    """

    object: str          # the table, in the query's own spelling
    rows: bool           # a row filter was applied to it
    # TRUE when the query references a column name this table masks AND either ownership resolved
    # to THIS table, or ownership could not be resolved at all. That is the whole rule; the two
    # halves are why neither "your answer lost a column" nor "the rewrite masked something" is a
    # safe reading of it.
    #
    # Where ownership RESOLVES, the flag is exact. `SELECT p.ssn FROM person p JOIN claim c` with
    # `claim.ssn` masked no longer reports `claim` -- it did until ownership resolution landed,
    # which was tolerable for an internal flag and a lie once a caller-facing sentence was built
    # on it.
    #
    # Where ownership does NOT resolve -- resolver failure, a CTE qualifier, a genuinely ambiguous
    # unqualified name -- it still over-reports, deliberately: the same fallback makes the masking
    # itself apply, and that direction only ever masks MORE. So a disclosure on those queries can
    # still name a table whose column the caller never read. That is the residual, it is bounded to
    # the unresolved cases, and `tests/test_mask_attribution.py` pins each one.
    #
    # And it UNDER-reports independently of any of that: a table wrapped for its ROW FILTER has its
    # masked columns NULLed regardless, so `SELECT id FROM claim` under a filter plus a mask on
    # `ssn` rewrites to `... NULL AS ssn ...` while this stays False. Kept, because a caller reading
    # `id` was not narrowed by a mask on `ssn`.
    columns: bool


def _record(into: list[Narrowing], one: Narrowing) -> None:
    """One OBJECT, one record -- the loop walks table NODES, and an object can appear many times.

    A self-join, or a DELETE whose WHERE reads its own target, reaches the same table twice and
    would otherwise emit "claim, claim" to anyone listing or counting narrowed objects. Merging
    ORs the flags, because two references can be narrowed differently: one filtered, one masked.
    """
    for i, existing in enumerate(into):
        if existing.object == one.object:
            into[i] = Narrowing(object=one.object,
                                rows=existing.rows or one.rows,
                                columns=existing.columns or one.columns)
            return
    into.append(one)


def check_filter_targets(policy: AccessPolicy) -> Refusal | None:
    """Refuse everything while a row filter names no object the snapshot holds (M50).

    Every read, not only reads of a table the key might have meant: which table a misspelled key
    meant is the one thing this cannot know. The caller is not told the key -- it names a table
    the policy governs, which the caller may not be granted -- only that the operator must act;
    `build_access_policy` logs the key for them.
    """
    if not policy.unknown_filter_targets:
        return None
    return Refusal(
        code=RefusalCode.INVALID_ROW_FILTER,
        message=("This identity's access policy has a row filter that names no table this "
                 "deployment knows, so no query can be governed for it. The operator must "
                 "correct the policy; the server log names the filter."),
    )


def unplaceable_policy(ast: exp.Expression, policy: AccessPolicy, dialect: str | None) -> Refusal | None:
    """Refuse to rewrite a statement whose read list fell back when a CTE in it is named after a
    table the policy governs (Codex's review of #111).

    When the walk misses a read, `base_tables` falls back to every table node -- CTE references
    included -- which is right for authorization, where over-reporting refuses. The rewrite reads
    the same list, so a CTE named like a filtered table was filtered as if it were the table:
    `WITH claim(amount) AS (VALUES ((SELECT max(amount) FROM claim)) UNION ALL SELECT 1) SELECT
    amount FROM claim` under `amount > 10` was APPROVED and answered `[30]` where the truth is
    `[30, 1]` -- measured on DuckDB, read and write paths both.

    Only that collision is refused. A missed read with no CTE named after a governed table is
    placed correctly -- M121's write through a scope that missed its FROM table is approved and
    row-filtered -- and refusing it would decline a policy this engine can apply. Any fallback, not
    only a missed read: a scope that does not resolve at all gives `base_tables` the same list
    (review gate), so `scope_resolved` decides, which answers for every cause.
    """
    if policy.empty or scope_resolved(ast, dialect):
        return None
    governed = list(policy.row_filters) + [t for t, _ in policy.masked | policy.denied]
    for cte in cte_references(ast, dialect):
        if names_one_object(cte.name, governed):
            return Refusal(
                code=RefusalCode.UNMODELLED_SOURCE,
                message=(
                    f"The query names a CTE {cte.name!r} after a table with a row or column "
                    "policy, in a statement this engine cannot fully resolve -- a table read "
                    "inside a VALUES or a set-operation branch is one cause -- so it cannot tell "
                    "the two apart to apply the policy. Rename the CTE, or write it with joins "
                    "and subqueries."
                ),
                subject=cte.name,
            )
    return None


def apply_row_and_mask(
    ast: exp.Expression,
    policy: AccessPolicy,
    visible: dict[str, set[str]],
    dialect: str = "duckdb",
    exclude: exp.Expression | None = None,
    executes_as: str | None = None,
) -> tuple[exp.Expression | Refusal, list[Narrowing]]:
    """Wrap each base table that has a row filter or a referenced masked column in a derived
    table that applies the filter and NULLs masked columns AT THE SOURCE. Returns the rewritten
    AST, or a Refusal for a policy-invalid filter.

    `exclude` skips ONE node, by identity. It exists for the target of an UPDATE/DELETE, which is
    the single position where a derived table is not SQL -- there is nothing to mutate through
    `UPDATE (SELECT ...) AS claim`. That target is filtered by conjunction instead, in
    `apply_row_filters_to_write`, which selects the same rows because the target's rows ARE the
    rows being mutated -- except on a `_FENCED` engine, where a conjunction does not order the
    filter first and the write is refused there instead (M160). By identity and never by name: `UPDATE claim ... WHERE id IN (SELECT id FROM
    claim ...)` is two nodes with one name that need opposite treatment, and matching on the name
    would skip both -- the flat-name-set bypass M31 and the view rounds each paid for once.
    """
    # The list rides with the AST rather than beside it, and the return type is a tuple so a
    # caller CANNOT quietly drop it: an ignored narrowing is exactly the silence this exists to
    # end, and it would be invisible at the call site if the signature still returned one value.
    narrowed: list[Narrowing] = []
    if not policy.row_filters and not policy.masked:
        return ast, narrowed

    # Folded on BOTH sides: the table keys come from the snapshot and `name` below comes from the
    # query, and comparing them exactly is what let `CLAIM` and `claim` name different objects.
    masked_by_table: dict[str, set[str]] = {}
    for tbl, col in policy.masked:
        masked_by_table.setdefault(tbl.lower(), set()).add(col.lower())

    # WHICH table a masked column name belongs to. Matching the bare name against every table
    # marked a table as masked whenever any table masked that name -- so `SELECT p.ssn FROM person
    # p JOIN claim c` with only `claim.ssn` masked reported `claim`, and the caller-facing sentence
    # built on that flag told them a column was masked when nothing they read was.
    #
    # SAFETY IS THE CONSTRAINT, not accuracy: the old behaviour masked MORE than necessary, never
    # less. So attribution only ever NARROWS this set where the owner is known, and every
    # unresolved case keeps the old answer:
    #   * `column_tables` returns None -- scopes could not be resolved at all -> mask, as before.
    #   * a column has no entry -- its qualifier names a CTE or derived table -> mask, as before.
    #   * an unqualified `ssn` across two tables that both mask it is genuinely ambiguous, and
    #     resolution declines to guess -> mask, as before.
    # Only a column resolved to a DIFFERENT base table is dropped, which is the reported defect
    # and the one case where the old answer was certainly wrong.
    owners = column_tables(ast, executes_as or dialect)
    referenced_masked: set[str] = set()
    for column in ast.find_all(exp.Column):
        owner = None if owners is None else owners.get(id(column))
        for tbl, cols in masked_by_table.items():
            if column.name.lower() not in cols:
                continue
            if owner is None or owner.lower() == tbl:
                referenced_masked.add(tbl)

    # Resolved before the loop mutates the tree: `replace` invalidates the scope it was read from.
    for table_node in base_tables(ast, executes_as or dialect):
        if exclude is not None and table_node is exclude:
            continue
        name = object_key(table_node)
        if name not in visible:
            continue
        needs_filter = policy.row_filter_for(name) is not None
        needs_mask = name.lower() in referenced_masked
        if not (needs_filter or needs_mask):
            continue
        filt: exp.Expression | None = None
        if needs_filter:
            filt = _validate_filter(
                policy.row_filter_for(name), visible[name], dialect, policy.policy_schema,
                executes_as=executes_as,
            )
            if filt is None:
                return Refusal(
                    code=RefusalCode.INVALID_ROW_FILTER,
                    message=f"The row-access policy for {name!r} is not a valid predicate.",
                    subject=name,
                ), []
            raising = raising_conversion(policy.row_filter_for(name), name, dialect, policy,
                                         executes_as=executes_as)
            if raising is not None:
                return _raising_refusal(raising, name, (executes_as or dialect).lower()), []
        derived = _derived_table(
            name, table_node.alias_or_name, visible[name],
            # `.lower()`: this dict is keyed folded and `name` is the QUERY's spelling.
            # Missing it left `needs_mask` correctly true while the column set handed to
            # the projection was EMPTY, so the masked column was emitted as its real
            # value instead of NULL -- the mask silently doing nothing.
            masked_by_table.get(name.lower(), set()), filt,
            # The engine that RUNS it, as everywhere else here: the fence is about its optimizer.
            fence=(executes_as or dialect).lower() in _FENCED,
        )
        table_node.replace(derived)
        _record(narrowed, Narrowing(object=name, rows=needs_filter, columns=needs_mask))
    return ast, narrowed


def apply_row_filters_to_write(
    ast: exp.Expression,
    policy: AccessPolicy,
    visible: dict[str, set[str]],
    target: exp.Expression | None,
    dialect: str = "duckdb",
    executes_as: str | None = None,
) -> tuple[exp.Expression | Refusal, list[Narrowing]]:
    """Govern a write with the read path's RLS. Returns the rewritten AST or a Refusal, and what
    the policy narrowed.

    M7 filed "two implementations with different semantics" and M30 measured what the second one
    permitted. The measurements, on `2af9443`: the write path bound a filter only to the table
    being *written*, so every table a write *read* was unfiltered -- `INSERT INTO scratch SELECT
    id, amount FROM claim` copied governed rows into an ungoverned table, where a later plain
    SELECT returns them forever. One approved write turned a filtered table into an unfiltered
    copy. And it parsed the filter with a bare `parse_one` rather than validating it, so a policy
    naming an arbitrary table was spliced in unchecked, a non-predicate became `WHERE id = 1 AND
    DELETE FROM claim`, and a typo raised `ParseError` out of the decider instead of refusing.

    So this is not a new implementation; it is the split that lets there be only one. Reads are
    wrapped by `apply_row_and_mask` unchanged. The UPDATE/DELETE target is conjoined, because it
    cannot be wrapped -- and it is conjoined with a predicate that went through the SAME
    `_validate_filter`, which is what closes the validation half rather than patching its five
    symptoms one at a time. On a `_FENCED` engine (Oracle) a filtered target is refused instead:
    the conjunction cannot be fenced, and the statement's own WHERE can run on removed rows (M160).

    An INSERT target is exempt and stays exempt: an INSERT does not read its target, and filtering
    rows on the way IN is not what a row filter means.
    """
    # `executes_as` forwarded, or the write path governs its reads with the PARSE dialect
    # while the read path uses the executing one -- two resolvers, one statement, and the
    # write half is the one that copies unfiltered rows somewhere durable.
    rewritten, narrowed = apply_row_and_mask(ast, policy, visible, dialect=dialect,
                                            exclude=target, executes_as=executes_as)
    if isinstance(rewritten, Refusal):
        return rewritten, []
    ast = rewritten

    if not isinstance(ast, (exp.Update, exp.Delete)) or not isinstance(target, exp.Table):
        return ast, narrowed
    name = object_key(target)
    filt = policy.row_filter_for(name)
    if filt is None:
        return ast, narrowed
    predicate = _validate_filter(filt, visible.get(name, set()), dialect, policy.policy_schema,
                                 executes_as=executes_as)
    if predicate is None:
        return Refusal(
            code=RefusalCode.INVALID_ROW_FILTER,
            message=f"The row-access policy for {name!r} is not a valid predicate.",
            subject=name,
        ), []
    raising = raising_conversion(filt, name, dialect, policy, executes_as=executes_as)
    if raising is not None:
        return _raising_refusal(raising, name, (executes_as or dialect).lower()), []
    if (executes_as or dialect).lower() in _FENCED:
        # The target is filtered by conjunction, and a conjunction is not a fence: the engine may
        # still evaluate the statement's own WHERE on rows the filter removes (M160, measured on
        # Oracle for an UPDATE). Refused until a write shape is found that fences it. The owner's
        # decision, 2026-10-15: reads fenced, these writes refused.
        return Refusal(
            code=RefusalCode.POLICY_TARGET_UNSUPPORTED,
            message=(f"An UPDATE or DELETE of {name!r}, which has a row filter, is refused on "
                     "this source: the engine cannot be made to apply the filter before the "
                     "statement's own conditions. This is a limit of this deployment, not of "
                     "your grants."),
            subject=name,
        ), []
    existing = ast.args.get("where")
    combined = exp.and_(existing.this, predicate) if existing is not None else predicate
    ast.set("where", exp.Where(this=combined))
    # The target is EXCLUDED from the loop above -- it cannot be wrapped in a derived table -- so
    # its narrowing is recorded here or nowhere. An implementer who wires up the read path and
    # stops ships a governed DELETE narrowed from 47 rows to 3 that reports nothing: this feature's
    # own silence, one decider over.
    _record(narrowed, Narrowing(object=name, rows=True, columns=False))
    return ast, narrowed
