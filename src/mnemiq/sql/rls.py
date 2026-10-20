from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import threading

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


# A value that is one of several branches: CASE, COALESCE (IFNULL and NVL parse to it), IF.
_BRANCHED = (exp.Case, exp.Coalesce, exp.If)


def _branches(value: exp.Expression) -> list[exp.Expression]:
    """The values a branched expression can be, its conditions left out."""
    if isinstance(value, exp.Case):
        default = value.args.get("default")
        return ([branch.args["true"] for branch in value.args.get("ifs", [])]
                + ([default] if default is not None else []))
    if isinstance(value, exp.If):
        return [branch for branch in (value.args.get("true"), value.args.get("false"))
                if branch is not None]
    return [value.this, *value.expressions]


_READER_LOCK = threading.Lock()
_READER = None


def _duckdb(sql: str, params=()):
    """One value from DuckDB's own reading of a constant expression, through mnemiq's local-only
    door (M107): what a literal or a cast means is the engine's to say, not Python's."""
    global _READER
    from mnemiq.duckdb_extensions import connect_duckdb

    with _READER_LOCK:
        if _READER is None:
            _READER = connect_duckdb(local_only=True)
        return _READER.execute(sql, list(params)).fetchone()[0]


def _duckdb_rows(sql: str, params=()):
    """Every row of a catalogue question put to the same local-only reader."""
    global _READER
    from mnemiq.duckdb_extensions import connect_duckdb

    with _READER_LOCK:
        if _READER is None:
            _READER = connect_duckdb(local_only=True)
        return _READER.execute(sql, list(params)).fetchall()


def _json_text(text: str) -> bool:
    """Whether a literal is JSON as DuckDB reads it: asked of DuckDB's own `json_valid`, since
    it is DuckDB's reader that fails on the row. Python's grammar differs both ways (it takes a
    lone surrogate escape DuckDB refuses; it refuses `[1,]` and `-NaN`, which DuckDB reads: the
    review gate and Codex's fourth review of #160), and `json_valid` agreed with
    `json_extract_string` on every literal measured. Through mnemiq's own door, local-only (the
    JSON functions are built in, so nothing is fetched; M107). Without the driver, Python's
    reader with every object pair kept and the result encoded, the stricter side."""
    try:
        return bool(_duckdb("SELECT json_valid($1)", [text]))
    except Exception:  # noqa: BLE001 -- no DuckDB here: Python's reader stands in
        try:
            json.dumps(json.loads(text, object_pairs_hook=lambda pairs: pairs),
                       ensure_ascii=False).encode("utf-8")
        except (ValueError, UnicodeEncodeError):
            return False
        return True


def _json_cannot_fail(source, type_of, engine: str, *, chosen_by_the_row: bool = False) -> bool:
    """Whether the text a JSON function reads is already JSON, or NULL where it is not.

    `chosen_by_the_row`: the source is one branch of a CASE, COALESCE or IF, so the row's data
    decides whether it is read. A constant at the top is read on every row and fails every read
    at once, which the operator sees; on a branch it fails on the rows that take the branch,
    which can be the hidden ones (the review gate on #160), so there a constant is safe only
    where it is JSON.
    """
    while isinstance(source, exp.Paren):
        source = source.this  # `(attrs)` is `attrs` (Codex's second review of #160)
    if source is None or isinstance(source, exp.Null):
        return True
    if source.find(exp.Column) is None and not chosen_by_the_row:
        return True  # a constant cannot fail on a row
    if isinstance(source, exp.Literal):
        return source.is_string and _json_text(source.this)
    if isinstance(source, exp.Cast):  # TryCast is a Cast
        if source.to.this != exp.DataType.Type.JSON:
            # To any other type it is text again, which the reader parses: `TRY_CAST(region AS
            # VARCHAR)` raised as `region` does (Codex's review of #160).
            return False
        # TRY_CAST: NULL where the text is not JSON. CAST: JSON where what is cast is.
        return isinstance(source, exp.TryCast) or _json_cannot_fail(
            source.this, type_of, engine, chosen_by_the_row=True)
    if isinstance(source, _BRANCHED):
        # `COALESCE(attrs, '{}'::JSON)`: JSON on every branch, so JSON whichever is taken.
        return all(_json_cannot_fail(branch, type_of, engine, chosen_by_the_row=True)
                   for branch in _branches(source))
    known = _known_type(source, type_of, engine)
    if known is not None and known.this not in exp.DataType.TEXT_TYPES:
        # A value of a known non-text type, JSON itself or a number, a boolean, a date, a list, a
        # struct, a subquery of one of those: serialised to JSON, never parsed (measured,
        # `json_contains(COALESCE(attrs, []), '1')` runs; Codex's fifth review of #160).
        return True
    if not isinstance(source, exp.Column):
        return False
    known = type_of(source)
    try:
        return known is not None and exp.DataType.build(known, dialect=engine).this == (
            exp.DataType.Type.JSON)
    except Exception:
        return False


def _branch_type(value, type_of, engine: str) -> exp.DataType | None:
    """The type of a branched value, from its branches: measured on DuckDB, NULL and a text
    literal take the other branches' type (`CASE WHEN c THEN 7 ELSE 'b' END` is INTEGER, and
    converts a text column it is compared with), a text column beside a number is a bind fault
    on no row's value, and the rest share a type. Unknown where any branch's type is. Measured
    on Oracle too: a mixed CASE or COALESCE is ORA-00932 at parse. Oracle's NVL is not read
    here as Oracle types it (its leading argument's type, the other converted row by row):
    `_nvl_converts_a_text_column` refuses that conversion before this typing is consulted, so a
    mixed NVL reads as text here only on a filter already refused, and a text-led NVL compared
    with a number (`NVL(region, id) = 7`, ORA-01722 on the hidden row) is caught through
    `_converts_a_text_column`. A value whose every branch is a text literal or NULL is text, not
    a literal: DuckDB types it VARCHAR (measured: beside a number it is the bind fault, and an
    outer CASE holding it beside a text column stays text, so the outer value is refused where
    a comparison converts it; Codex's third review of #160)."""
    known, literal = [], False
    for branch in _branches(value):
        while isinstance(branch, exp.Paren):
            branch = branch.this  # `(NULL)` is NULL (Codex's fourth review of #160)
        if isinstance(branch, exp.Null):
            continue
        if isinstance(branch, exp.Literal) and branch.is_string:
            literal = True
            continue
        typed = _known_type(branch, type_of, engine)
        if typed is None:
            return None
        known.append(typed)
    # On DuckDB a JSON branch makes the whole value JSON, the other branches converted to it row
    # by row (measured: `COALESCE(attrs, region)` is JSON, and malformed text on the row that
    # takes region; Codex's fourth review of #160). The walk refuses that mix at the branched
    # node itself, on DuckDB, through `_coerces_to_json`; the typing below reads a JSON branch
    # beside text as text, which only a comparison visited earlier in the walk consults, and a
    # refusal there or at the node is a refusal either way.
    text = [typed for typed in known if typed.this in exp.DataType.TEXT_TYPES]
    if text:
        return text[0]
    if known:
        return known[0]
    return exp.DataType.build("VARCHAR") if literal else None


# DuckDB's calls that return JSON, typed by name so nothing runs (Codex's nineteenth review of
# #160: `list_contains([region], to_json(id))`, `region = to_json(id)`, a CASE between them and
# `[to_json(id)] = [region]` were of unknown type on the call's side and approved, where DuckDB
# unifies the text to JSON and parses it). By sqlglot's node where it has one, by name where
# it does not: the names the local reader's own catalogue lists with a JSON or JSON[] return
# (`_json_returners`; twelve on 1.5.4, the typed list below where the catalogue cannot be
# asked). `json_transform`, `from_json` and `json_value` are not among them: the first two
# return the structure they are asked for, the third VARCHAR, and typed JSON they would pass
# beside a JSON value where they return text (the review gate on the twenty-first round, twice).
# A name the catalogue does not list, a source-defined function among them, is unknown here,
# and unknown beside text is not refused, since that would refuse `lower(region) = 'x'` (the
# residual; the review gate on the twentieth round).
_JSON_CALL_NODES = (exp.JSONExtract, exp.JSONObject, exp.JSONArray, exp.JSONArrayAgg,
                    exp.JSONObjectAgg, exp.ParseJSON)
_JSON_RETURNERS_KNOWN = frozenset({"to_json", "row_to_json", "array_to_json", "json_quote",
                                   "json_object", "json_array", "json_merge_patch",
                                   "json_extract", "json_extract_path", "json_structure",
                                   "json_serialize_sql", "json_serialize_plan"})
_JSON_RETURNERS: frozenset[str] | None = None
_JSON_EXTRACTS = frozenset({"json_extract", "json_extract_path"})
# The json extension's aggregate macros, which the catalogue lists with no return type (Codex's
# twenty-second review of #160: `region = (SELECT json_group_array(id) FROM claim)` approved,
# region parsed as JSON on the row): typed JSON by name.
_JSON_MACROS = frozenset({"json_group_array", "json_group_object", "json_group_structure"})
# The extracts that return text, VARCHAR or VARCHAR[] by the path's shape (the same review:
# `(attrs ->> '$.a') = 7` approved, the text converted to a number on the row).
_TEXT_EXTRACT_NODES = (exp.JSONExtractScalar,)
_TEXT_EXTRACT_NAMES = frozenset({"json_extract_string", "json_extract_path_text"})


def _json_returners() -> frozenset[str]:
    """The calls the local reader's catalogue answers with JSON or JSON[] (any overload), the
    typed list where the catalogue cannot be asked (logged: a missing name is an approval)."""
    global _JSON_RETURNERS
    if _JSON_RETURNERS is None:
        try:
            rows = _duckdb_rows("SELECT DISTINCT lower(function_name) FROM duckdb_functions() "
                                "WHERE return_type IN ('JSON', 'JSON[]')")
            _JSON_RETURNERS = (frozenset(name for (name,) in rows) or _JSON_RETURNERS_KNOWN) | _JSON_MACROS
        except Exception:  # noqa: BLE001 -- the catalogue could not be asked: the typed list
            logger.warning("DuckDB's catalogue could not be asked which calls return JSON; "
                           "the typed list stands in (M159)")
            _JSON_RETURNERS = _JSON_RETURNERS_KNOWN | _JSON_MACROS
    return _JSON_RETURNERS


def _is_json_call(node: exp.Expression) -> bool:
    return isinstance(node, _JSON_CALL_NODES) or (
        isinstance(node, exp.Anonymous) and node.name.lower() in _json_returners())


def _is_text_extract(node: exp.Expression) -> bool:
    return isinstance(node, _TEXT_EXTRACT_NODES) or (
        isinstance(node, exp.Anonymous) and node.name.lower() in _TEXT_EXTRACT_NAMES)


def _scalar_projections(node: exp.Expression) -> list[exp.Expression] | None:
    """A scalar subquery's one projection per branch, unaliased, else None: one for a SELECT,
    one per branch of a set operation (Codex's twenty-third review of #160: `(SELECT attrs ...
    UNION ALL SELECT attrs ... LIMIT 1)` was unknown, and JSON beside text approved)."""
    if not isinstance(node, exp.Subquery):
        return None
    branches, pending = [], [node.this]
    while pending:
        query = pending.pop()
        if isinstance(query, exp.SetOperation):
            pending.extend((query.right, query.left))
        elif isinstance(query, exp.Select) and len(query.selects) == 1:
            branches.append(query.selects[0].unalias())
        else:
            return None
    return branches


def _scalar_projection(node: exp.Expression) -> exp.Expression | None:
    """The first branch's projection (the walk's handle on the subquery), else None."""
    branches = _scalar_projections(node)
    return branches[0] if branches else None


def _subquery_type(node: exp.Expression, type_of) -> str | None:
    """A scalar subquery's type: its one projection's, each branch of a set operation typed and
    the branches unified as DuckDB unifies them (`typeof((SELECT id ... UNION ALL SELECT region
    ... LIMIT 1))` is VARCHAR, not the first branch's INTEGER: the review gate on the
    twenty-sixth round), None where a branch cannot be typed."""
    branches = _scalar_projections(node)
    if not branches:
        return None
    kinds = [_pure_type(branch, type_of) for branch in branches]
    if any(kind is None for kind in kinds):
        return None
    if len(set(kinds)) == 1:
        return kinds[0]
    try:
        # Unified as a set operation unifies, not as COALESCE does: DuckDB refuses
        # `COALESCE(NULL::INTEGER, NULL::VARCHAR)` and unions the two to VARCHAR.
        branches_sql = " UNION ALL ".join(f"SELECT NULL::{k} AS x" for k in kinds)
        return _duckdb(f"SELECT typeof(x) FROM ({branches_sql}) LIMIT 1")
    except Exception:  # noqa: BLE001 -- the branches cannot be unified by the probe: unknown
        return None


def _is_typed_call(node: exp.Expression) -> bool:
    """A call whose type is known here without running it: a JSON returner or a text extract."""
    return _is_json_call(node) or _is_text_extract(node)


def _json_call_type(node: exp.Expression) -> str:
    """JSON, or JSON[] for an extract whose path is anything but one string literal
    (`json_extract(x, ['$.a'])` is JSON[] to DuckDB, and so is a cast or a column holding a
    list of paths; typed as a scalar it unified with `[region]` as a list of text would not:
    the review gate on the twentieth and twenty-first rounds). A path of unknown shape is read
    as a list, fail-closed: a scalar extract beside text is then refused as a list would be."""
    leaf = "VARCHAR" if _is_text_extract(node) else "JSON"
    path = None
    if isinstance(node, (exp.JSONExtract, exp.JSONExtractScalar)):
        path = node.expression
    elif isinstance(node, exp.Anonymous) and node.name.lower() in _JSON_EXTRACTS | _TEXT_EXTRACT_NAMES:
        path = node.expressions[1] if len(node.expressions) > 1 else None
    if path is None or (isinstance(path, exp.Literal) and path.is_string) or isinstance(path, exp.JSONPath):
        return leaf
    return leaf + "[]"


# The calls DuckDB hands a lambda to. Anywhere else `a -> b` is DuckDB's JSON extract, which
# sqlglot reads as a lambda inside a call's arguments or beside an operator (`list_contains(
# [region], attrs -> '$.a')`, `region = attrs -> '$.a'`: Codex's nineteenth review of #160), so the
# reader here sees a statement the engine does not run; refused, with the plain spelling in the
# advice (`json_extract(attrs, '$.a')`, which sqlglot reads as the extract everywhere).
_LAMBDA_TAKERS_KNOWN = frozenset({"list_transform", "array_transform", "apply", "list_apply",
                                  "array_apply", "list_filter", "array_filter", "filter",
                                  "list_reduce", "array_reduce", "reduce"})
_LAMBDA_TAKER_NODES = (exp.Transform, exp.ArrayFilter, exp.Reduce)
_LAMBDA_TAKERS: frozenset[str] | None = None


def _lambda_takers() -> frozenset[str]:
    """The calls DuckDB hands a lambda to, asked of the local reader's own catalogue
    (`duckdb_functions()`, a LAMBDA among the parameter types: eleven names on 1.5.4), the
    list above where the catalogue cannot be asked. A typed list missed `array_apply` and
    listed `list_sort`, `list_any_value` and `list_grade_up`, which take none (the review gate
    on the twentieth and twenty-first rounds)."""
    global _LAMBDA_TAKERS
    if _LAMBDA_TAKERS is None:
        try:
            rows = _duckdb_rows("SELECT DISTINCT lower(function_name) FROM duckdb_functions() "
                                "WHERE list_contains(parameter_types, 'LAMBDA')")
            _LAMBDA_TAKERS = frozenset(name for (name,) in rows) or _LAMBDA_TAKERS_KNOWN
        except Exception:  # noqa: BLE001 -- the catalogue could not be asked: the typed list
            logger.warning("DuckDB's catalogue could not be asked which calls take a lambda; "
                           "the typed list stands in (M159)")
            _LAMBDA_TAKERS = _LAMBDA_TAKERS_KNOWN
    return _LAMBDA_TAKERS


def _arrow_read_as_a_lambda(node: exp.Expression) -> bool:
    if not isinstance(node, exp.Lambda):
        return False
    parent = node.parent
    return not (isinstance(parent, _LAMBDA_TAKER_NODES) or (
        isinstance(parent, exp.Anonymous) and parent.name.lower() in _lambda_takers()))


def _known_type(value, type_of, engine: str) -> exp.DataType | None:
    """`value`'s type where it is plain to see: a typed column, a literal, a cast, arithmetic, an
    interval, a list, a struct or a row, a scalar subquery's one projection, a branched value
    whose branches are typed, NULLIF's first argument (Codex's second review of #160). A call
    DuckDB answers with JSON is typed through `_pure_type`, as a `NULL::JSON`."""
    while isinstance(value, exp.Paren):
        value = value.this
    if isinstance(value, exp.Boolean):
        return exp.DataType.build("BOOLEAN")
    if isinstance(value, exp.DateStrToDate):  # Oracle's DATE '...' literal
        return exp.DataType.build("DATE")
    if isinstance(value, exp.Interval):
        return exp.DataType.build("INTERVAL")
    if isinstance(value, exp.Array):
        return exp.DataType(this=exp.DataType.Type.ARRAY)
    if isinstance(value, (exp.Struct, exp.Tuple)) or (
            isinstance(value, exp.Anonymous) and value.name.lower() == "row"):
        return exp.DataType(this=exp.DataType.Type.STRUCT)
    if isinstance(value, exp.Column):
        known = type_of(value)
        try:
            return None if known is None else exp.DataType.build(known, dialect=engine)
        except Exception:
            return None
    if isinstance(value, exp.Cast):
        return value.to
    if isinstance(value, exp.Subquery):
        inner = value.this
        if isinstance(inner, exp.Select) and len(inner.selects) == 1:
            return _known_type(inner.selects[0].unalias(), type_of, engine)
        return None  # a set operation's is typed through `_type_of_value`, by `_subquery_type`
    if isinstance(value, (exp.Nullif, exp.Max, exp.Min)):
        return _known_type(value.this, type_of, engine)
    if isinstance(value, exp.Nvl2):
        return _known_type(value.args["true"], type_of, engine)  # Oracle: its second argument's
    if isinstance(value, (exp.Count, exp.Sum, exp.Avg)):
        # A number of some width, which is all this reader's callers ask (text or not); the
        # width is `_duckdb_type`'s to tell, and it never reads this answer.
        return exp.DataType.build("DOUBLE")
    if isinstance(value, _BRANCHED):
        return _branch_type(value, type_of, engine)
    if isinstance(value, exp.Neg):
        return _known_type(value.this, type_of, engine)  # `-id` is id's type
    if isinstance(value, exp.Literal) and value.is_number or isinstance(value, _ARITHMETIC):
        return exp.DataType.build("DOUBLE")
    return None


def _coerces_to_json(node: exp.Expression, type_of, engine: str) -> bool:
    """DuckDB types a CASE, COALESCE or IF with a JSON branch as JSON and converts every other
    branch to JSON on the rows that take it: `COALESCE(attrs, region)`, `COALESCE(attrs, 'x')`
    and `IF(c, 'x', attrs)` are Malformed JSON on those rows (measured; Codex's fourth review
    of #160), with no JSON reader around them. So beside a JSON branch every branch must be JSON
    already, NULL, a literal that is JSON, or a value of a known type that is not text: a
    number, a boolean, a date, a list, a struct or a JSON-typed subquery is serialised, not
    parsed (measured, `COALESCE(attrs, 0)`, `COALESCE(attrs, [])`, `COALESCE(attrs, d)` run;
    Codex's fifth review). Text, and a value of unknown type, is parsed on the rows that take
    it."""
    return _values_coerce_to_json(_branches(node), type_of, engine)


def _map_key(item: exp.PropertyEQ) -> exp.Expression:
    """A MAP literal's key as the engine evaluates it: sqlglot parses `MAP {attrs: 1}`'s key as a
    struct-field identifier, which stands for the column, read in the key's own SELECT."""
    key = item.this
    if isinstance(key, exp.Identifier):
        as_column = exp.Column(this=key.copy())
        as_column.parent = item
        return as_column
    return key


def _untypable(value: exp.Expression, type_of, engine: str) -> bool:
    """Whether the value's type cannot be told (a call, a subquery with an aggregate) or cannot
    be read: asked of DuckDB's own grammar where DuckDB executes, of the reader here elsewhere."""
    if engine == "duckdb":
        kind = _duckdb_type(value, type_of)
        return kind is None or kind is _UNREADABLE
    return _type_of_value(value, type_of, engine) is None


def _type_of_value(value: exp.Expression, type_of, engine: str) -> exp.DataType | None:
    """A value's type as the reader here knows it, else as DuckDB types a pure expression (a
    list element or a struct field read out of a container, `[attrs][1]`, `({'a': attrs}).a`,
    which the reader does not follow; Codex's tenth review of #160)."""
    known = _known_type(value, type_of, engine)
    if known is not None or engine != "duckdb":
        return known
    if _scalar_projection(value) is not None:
        # A scalar subquery is its projection's type, the projection alone handed to the probe,
        # never the subquery, whose FROM would run (as `_duckdb_type` reads it): `region =
        # (SELECT json_group_array(id) FROM claim)` (Codex's twenty-second review of #160).
        kind = _subquery_type(value, type_of)
    else:
        kind = _pure_type(value, type_of)
    if kind is None:
        return None
    try:
        return exp.DataType.build(kind, dialect="duckdb")
    except Exception:  # noqa: BLE001
        return None


def _values_coerce_to_json(values, type_of, engine: str) -> bool:
    """`_coerces_to_json`'s reading over any values DuckDB unifies to one type: a branched
    value's branches, a list's elements, a map's keys or its values (Codex's ninth review of
    #160: `[attrs, region]` is JSON[], region parsed on every row), a list search's element and
    argument."""
    typed = [_type_of_value(value, type_of, engine) for value in values]
    if not any(known is not None and known.this == exp.DataType.Type.JSON for known in typed):
        return False
    return not all(
        (known is not None and known.this not in exp.DataType.TEXT_TYPES)
        or _json_cannot_fail(value, type_of, engine, chosen_by_the_row=True)
        for value, known in zip(values, typed))


# What may be handed to DuckDB to type: a value expression of columns, literals, casts,
# arithmetic, lists, structs, maps, branched values and the comparisons and boolean operators their
# conditions are made of, which binds without running anything once each column is a typed
# NULL. No function call, no subquery: typing those would evaluate policy text on the
# validator's connection (a table function reading a file, a subquery over a billion rows),
# which a validator must not do (the review gate on #160); their type is unknown here.
_PURE = (exp.Column, exp.Identifier, exp.Literal, exp.Boolean, exp.Null, exp.Paren, exp.Neg,
         exp.Cast, exp.DataType, exp.DataTypeParam, exp.Interval, exp.Var, exp.DateStrToDate,
         exp.Case, exp.If, exp.Coalesce, exp.Nullif, exp.Array, exp.Struct, exp.PropertyEQ,
         exp.ToMap, exp.Map,  # `MAP {'a': attrs}`, `MAP(['a'], [attrs])` (the review gate)
         exp.Bracket, exp.Dot,  # `[attrs][1]`, `({'a': attrs}).a` (Codex's tenth review)
         exp.Tuple,  # `('a', 1)`, a row value, as `map_from_entries` takes them
         exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.And, exp.Or, exp.Not, exp.Is,
         exp.In, exp.Between, exp.Like, exp.ILike, *_ARITHMETIC)


def _pure_type(value: exp.Expression, type_of) -> str | None:
    """DuckDB's type for a pure value expression (`_PURE`: columns, literals, casts, arithmetic),
    each column replaced by a NULL of its snapshot type, so the expression binds and nothing
    runs; None for anything else, or where a column's type is not known."""
    while isinstance(value, exp.Paren):
        value = value.this
    if isinstance(value, exp.Column):
        # The value itself: its declared type, since replacing the root of the copy rebinds
        # nothing and the probe would ask DuckDB for a column with no FROM (the review gate on
        # the twenty-fourth round: `[region] = [(SELECT attrs FROM claim LIMIT 1)]` approved).
        return type_of(value)
    # A call that returns JSON is a typed NULL to the probe, as a column is: nothing of it runs,
    # and what it holds (`to_json(id)` beside `[region]`) is JSON to DuckDB's unification.
    # A scalar subquery with one pure projection is that projection to the probe: `[region] =
    # [(SELECT json_group_array(id) FROM claim)]` (the review gate on the twenty-third round),
    # the projection alone handed over, never the subquery, whose FROM would run.
    json_calls = [node for node in value.walk() if _is_typed_call(node) or _scalar_projection(node) is not None]
    inside_a_json_call = {id(inner) for call in json_calls for inner in call.walk() if inner is not call}
    if not all(id(node) in inside_a_json_call or _is_typed_call(node) or _scalar_projection(node) is not None
               or isinstance(node, (*_PURE, *_PURE_CALL_NODES)) or (
               isinstance(node, exp.Anonymous)
               and node.name.lower() in _LIST_CONSTRUCTORS | _PURE_CALLS)
               for node in value.walk()):
        return None
    # A MAP literal's keys are expressions DuckDB evaluates (`MAP {id: 1}` is keyed by the
    # column, MAP(INTEGER, INTEGER)), which sqlglot parses as struct-field identifiers, so they
    # are typed here as columns of the value's own SELECT (the review gate on #160: left as
    # names, the map was typed MAP(VARCHAR, INTEGER), and `MAP {attrs: 1}` beside `MAP {region:
    # 1}` read as one type where DuckDB parses region to JSON). A key of any other shape is not
    # pure.
    keys = []
    for to_map in value.find_all(exp.ToMap):
        if not isinstance(to_map.this, exp.Struct):
            return None
        for item in to_map.this.expressions:
            if not isinstance(item, exp.PropertyEQ):
                return None
            if isinstance(item.this, exp.Identifier):
                as_column = exp.Column(this=item.this.copy())
                as_column.parent = item  # read in the value's own SELECT, as the engine reads it
                keys.append(as_column)
            # any other key (a literal, a cast such as `'{}'::JSON`) is a pure value of its own
    # The types are read off the original nodes, which the resolver knows by identity; the
    # copy's are then replaced in the same traversal order.
    declared = [type_of(column) for column in value.find_all(exp.Column)
                if id(column) not in inside_a_json_call]
    key_types = [type_of(column) for column in keys]
    if None in declared or None in key_types:
        return None
    typed_nulls = value.copy()
    if _is_typed_call(typed_nulls):
        return _json_call_type(typed_nulls)
    if _scalar_projection(value) is not None:
        return _subquery_type(value, type_of)
    # Typed calls first, so a scalar subquery inside one goes with it (the review gate on the
    # twenty-fourth round: typing it from the copy, whose nodes `type_of` does not know, made
    # the whole value unknown and disarmed the typed-call reading); then each scalar subquery
    # outside a typed call and outside another, typed from the ORIGINAL node's projection and
    # replaced in the copy in the same traversal order.
    for call in list(typed_nulls.find_all(*_JSON_CALL_NODES, *_TEXT_EXTRACT_NODES, exp.Anonymous)):
        if _is_typed_call(call):
            call.replace(exp.Cast(this=exp.Null(), to=exp.DataType.build(_json_call_type(call))))
    originals = [sq for sq in value.find_all(exp.Subquery)
                 if id(sq) not in inside_a_json_call and _scalar_projection(sq) is not None]
    inside_another = {id(inner) for sq in originals for inner in sq.walk() if inner is not sq}
    originals = [sq for sq in originals if id(sq) not in inside_another]
    projected = [_subquery_type(sq, type_of) for sq in originals]
    if None in projected:
        return None
    copies = [sq for sq in typed_nulls.find_all(exp.Subquery) if _scalar_projection(sq) is not None]
    copies = [sq for sq in copies if not any(sq is not other and sq.find_ancestor(exp.Subquery) is other for other in copies)]
    if len(copies) != len(originals):
        return None  # the copy and the original disagree on the subqueries: not shown safe
    for subquery, kind_of in zip(copies, projected):
        subquery.replace(exp.Cast(this=exp.Null(), to=exp.DataType.build(kind_of, dialect="duckdb")))
    for column, kind_of in zip(list(typed_nulls.find_all(exp.Column)), declared):
        column.replace(exp.Cast(this=exp.Null(), to=exp.DataType.build(kind_of, dialect="duckdb")))
    copied_keys = [item.this for to_map in typed_nulls.find_all(exp.ToMap)
                   for item in to_map.this.expressions if isinstance(item.this, exp.Identifier)]
    for key, kind_of in zip(copied_keys, key_types):
        key.replace(exp.Cast(this=exp.Null(), to=exp.DataType.build(kind_of, dialect="duckdb")))
    try:
        return _duckdb(f"SELECT typeof({typed_nulls.sql(dialect='duckdb')})")
    except Exception:  # noqa: BLE001 -- unreadable by DuckDB: the type cannot be told
        return None


# `_duckdb_type`'s answer for a text or JSON branch: a text literal beside it is taken as it is
# (or the mix is a bind fault on no row's value), or parsed by `_coerces_to_json`'s rule.
_TEXT = "text"
_UNREADABLE = "unreadable"  # a declared or engine-reported type the reader here cannot parse
_LIST_CONSTRUCTORS = frozenset({"array_value", "list_pack"})  # `list_value` parses as a list
# Container accessors that build a value from their arguments and run nothing, typed by DuckDB
# like the containers they read (Codex's twelfth review of #160: `map_values(MAP(['a'],
# [attrs])) = map_values(MAP(['a'], [region]))`, `list_extract([attrs], 1) = region` and the
# struct and key twins were of unknown type on both sides, and parsed region as JSON on the
# hidden row); by sqlglot's node where it has one, by name where it does not.
_PURE_CALL_NODES = (exp.MapKeys, exp.StructExtract, exp.ArrayDistinct, exp.MapFromEntries,
                    exp.ArraySlice, exp.SortArray, exp.Flatten, exp.ArrayReverse)
_PURE_CALLS = frozenset({"map_values", "map_entries", "list_extract", "array_extract", "list_element",
                         "array_element", "list_slice", "list_reverse", "list_unique",
                         "list_distinct", "array_distinct", "map_keys", "struct_extract",
                         "flatten", "list_sort", "map_from_entries", "row"})
# A list searched for a value (the element type unified with the value's), or two lists searched
# against each other; by sqlglot's node where it has one, by name where it does not.
_LIST_SEARCHES = (exp.ArrayContains, exp.ArrayPosition, exp.ArrayOverlaps, exp.ArrayContainsAll,
                  exp.Contains, exp.ArrayContainedBy)
_LIST_PAIRS = (exp.ArrayOverlaps, exp.ArrayContainsAll, exp.ArrayContainedBy)
_LIST_SEARCH_NAMES = frozenset({"list_has", "array_has", "list_position", "list_indexof",
                                "array_indexof", "list_has_all", "array_has_all", "list_has_any",
                                "array_has_any", "array_contains", "list_contains"})
_LIST_PAIR_NAMES = frozenset({"list_has_all", "array_has_all", "list_has_any", "array_has_any"})
# A map looked up by a key, which DuckDB converts to the map's key type (Codex's eleventh
# review of #160: `element_at(MAP([attrs], [1]), region)` parses region as JSON on every row).
_MAP_LOOKUPS = frozenset({"element_at", "map_extract", "map_extract_value", "map_contains"})
_MAP_ONLY_LOOKUPS = frozenset({"map_extract", "map_extract_value", "map_contains"})
_TEXT_NAMES = frozenset({"VARCHAR", "TEXT", "STRING", "CHAR", "BPCHAR", "NVARCHAR", "NCHAR", "NAME"})


def _split_top(text: str) -> list[str]:
    """`text` split on the commas outside any parentheses, brackets or quotes."""
    parts, depth, quote, current = [], 0, None, []
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def _parse_type(text: str):
    """A DuckDB type, in DuckDB's words, as a tree: `("leaf", NAME)`, `("array", element)`,
    `("struct", [(name or None, field), ...])`, `("map", key, value)`, `("union", [(name,
    member), ...])`. DuckDB's own grammar, read directly: sqlglot's DataType reads a nested
    unnamed struct (`STRUCT(STRUCT(JSON))`, which `row(row(attrs))` gives) as a struct literal
    (the review gate on #160). A leaf keeps its parameters beside its name (`DECIMAL(2,1)`,
    `ENUM('x')`: without its labels an ENUM is no cast target, and the unification's failure
    read as a bind fault let a struct holding one through; Codex's fourteenth review)."""
    text = text.strip()
    bracket = re.search(r"\[\d*\]$", text)
    if bracket is not None:
        return ("array", _parse_type(text[:bracket.start()]))
    quoted = re.match(r'("(?:[^"]|"")*")\s*(?:\((.*)\))?$', text, re.S)
    if quoted is not None:
        # DuckDB quotes a type name that is a keyword (`typeof([])` is `"NULL"[]`): kept as
        # spelled, so a cast target rebuilds, and read bare by `_bare` (Codex's sixteenth
        # review of #160, whose fix refused every type the reader could not parse).
        return ("leaf", quoted.group(1), f"({quoted.group(2)})" if quoted.group(2) is not None else "")
    head = re.match(r"([A-Za-z_][A-Za-z_0-9 ]*?)\s*(?:\((.*)\))?$", text, re.S)
    if head is None:
        raise ValueError(text)
    name, inner = head.group(1).strip().upper(), head.group(2)
    if name not in ("STRUCT", "MAP", "UNION"):
        return ("leaf", name, f"({inner})" if inner is not None else "")
    if name == "STRUCT":
        fields = []
        for item in _split_top(inner or ""):
            label, rest = _split_label(item)
            fields.append((label, _parse_type(rest)))
        return ("struct", fields)
    if name == "MAP":
        key, value = _split_top(inner or "")
        return ("map", _parse_type(key), _parse_type(value))
    if name == "UNION":
        members = []
        for item in _split_top(inner or ""):
            label, rest = _split_label(item)
            members.append((label, _parse_type(rest)))
        return ("union", members)
    raise ValueError(text)


def _bare(name: str) -> str:
    """A leaf's name without DuckDB's quoting: `"NULL"` is NULL."""
    if name.startswith('"') and name.endswith('"'):
        return name[1:-1].replace('""', '"').upper()
    return name


def _split_label(item: str) -> tuple[str | None, str]:
    """A struct field or union member: `name TYPE` or, unnamed, `TYPE` (a space outside any
    parentheses separates the name, a quoted name kept whole)."""
    if item.startswith('"'):
        # A doubled quote inside the name is one quote (`"a""b" JSON` names a""b: Codex's
        # sixteenth review of #160); the first lone quote closes the name. Nothing after it
        # means the quoted token is an unnamed field's TYPE, a keyword DuckDB quotes as a type
        # name (`typeof(row(NULL, 1))` is `STRUCT("NULL", INTEGER)`: the review gate).
        name, position = [], 1
        while position < len(item):
            if item[position] == '"':
                if item[position + 1:position + 2] == '"':
                    name.append('"')
                    position += 2
                    continue
                rest = item[position + 1:].strip()
                if not rest or rest.startswith(("[", "(")):
                    return None, item  # the type itself, with its array suffix or parameters
                return "".join(name), rest
            name.append(item[position])
            position += 1
        raise ValueError(f"unterminated quoted name in {item!r}")
    depth = 0
    for position, ch in enumerate(item):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == " " and depth == 0:
            rest = item[position + 1:].strip()
            if rest.upper().startswith(("WITH ", "WITHOUT ")):
                return None, item  # an unnamed `TIMESTAMP WITH TIME ZONE` (the review gate)
            return item[:position], rest
    return None, item


def _castable(node) -> str:
    """The type back in DuckDB's words as a cast target, an unnamed struct's fields named by
    position (`STRUCT(f1 JSON)`), which `row(attrs)` gives and no cast accepts (Codex's
    thirteenth review of #160)."""
    kind = node[0]
    if kind == "leaf":
        return node[1] + node[2]
    if kind == "array":
        return _castable(node[1]) + "[]"
    if kind == "map":
        return f"MAP({_castable(node[1])}, {_castable(node[2])})"
    fields = [f'"{(name or f"f{position}").replace(chr(34), chr(34) * 2)}" {_castable(field)}'
              for position, (name, field) in enumerate(node[1], 1)]
    return f"{'STRUCT' if kind == 'struct' else 'UNION'}({', '.join(fields)})"


def _holds_union(node) -> bool:
    kind = node[0]
    if kind == "leaf":
        return False
    if kind == "array":
        return _holds_union(node[1])
    if kind == "map":
        return _holds_union(node[1]) or _holds_union(node[2])
    return kind == "union" or any(_holds_union(field) for _, field in node[1])


def _holds_json(node) -> bool:
    kind = node[0]
    if kind == "leaf":
        return _bare(node[1]) == "JSON"
    if kind == "array":
        return _holds_json(node[1])
    if kind == "map":
        return _holds_json(node[1]) or _holds_json(node[2])
    return any(_holds_json(field) for _, field in node[1])


def _text_becomes_json(kind, unified) -> bool:
    """Whether converting a value of type `kind` to `unified` parses text as JSON anywhere: the
    two types walked side by side, a list's element with the element, a struct's fields paired
    by name where both name them (DuckDB matches them so, and `{'a': attrs, 'b': region}`
    beside `{'b': region, 'a': attrs}` runs) and by position otherwise, a map's key with the
    key and value with the value. Shapes that do not line up cannot be told, and are read as
    converting."""
    if unified[0] == "leaf":
        return _bare(unified[1]) == "JSON" and kind[0] == "leaf" and _bare(kind[1]) in _TEXT_NAMES
    if kind[0] != unified[0]:
        return True
    if kind[0] == "array":
        return _text_becomes_json(kind[1], unified[1])
    if kind[0] == "map":
        return _text_becomes_json(kind[1], unified[1]) or _text_becomes_json(kind[2], unified[2])
    if len(kind[1]) != len(unified[1]):
        return True
    if all(name is not None for name, _ in kind[1]) and all(name is not None for name, _ in unified[1]):
        by_name = {name: field for name, field in unified[1]}
        if len(by_name) != len(unified[1]) or any(name not in by_name for name, _ in kind[1]):
            return True
        pairs = [(field, by_name[name]) for name, field in kind[1]]
    else:
        pairs = [(inner, outer) for (_, inner), (_, outer) in zip(kind[1], unified[1])]
    return any(_text_becomes_json(inner, outer) for inner, outer in pairs)


def _json_inside_a_container(values, type_of) -> bool:
    """Whether DuckDB's unification of `values` (a branched value's branches, a comparison's
    sides) converts text to JSON inside a list or a struct, on the rows that take the branch
    or on every row: `CASE WHEN id = 1 THEN [attrs] ELSE [region] END` is JSON[], and Malformed
    JSON on the row that takes region, as are the struct twin and `[attrs] = [region]` (measured
    on DuckDB 1.5.4; Codex's eighth review of #160), where `[id]` beside `[attrs]` is serialised
    and runs. Each value's type is DuckDB's own (`_duckdb_type`, nothing runs), and so is the
    unified type (`typeof(COALESCE(NULL::T1, ...))`): refused where it holds JSON and a value's
    type differs from it with text where the unified type has JSON (`_text_becomes_json`), or
    cannot be told. A mix DuckDB cannot unify is a bind fault on no row's value, not this
    class."""
    kinds = []
    for value in values:
        while isinstance(value, exp.Paren):
            value = value.this
        if value is None or isinstance(value, exp.Null):
            continue
        kinds.append(_duckdb_type(value, type_of))
    if any(kind is _UNREADABLE for kind in kinds):
        return True  # a declared type the reader cannot parse: not shown safe
    try:
        known = [_parse_type(kind) for kind in kinds if kind is not None and kind is not _TEXT]
    except ValueError:
        return True  # a type the reader here cannot read: not shown safe
    if not any(_holds_json(node) for node in known):
        return False
    if any(kind is None for kind in kinds):
        return True  # a value of unknown type beside JSON: may hold text, parsed on its rows
    if any(_holds_union(node) for node in known) and any(node != known[0] for node in known):
        # Two UNION types unify by the operator and the operand order, not by their members
        # (measured, DuckDB 1.5.4: COALESCE and a list keep the first operand's type, CASE and
        # IF the last's, at the top level and inside a struct alike), so the COALESCE asked
        # below answers for one operator only. Refused wherever the UNION sits: the operator
        # that happens to keep the text member runs, a false refusal (Codex's fifteenth review
        # of #160, and the review gate on the sixteenth round's draft, which exempted a branch
        # and let the struct-nested pair and a list search through).
        return True
    try:
        unified = _parse_type(_duckdb("SELECT typeof(COALESCE("
                                      + ", ".join(f"NULL::{_castable(node)}" for node in known) + "))"))
    except Exception:  # noqa: BLE001 -- not shown safe: the unification could not be asked
        # A mix DuckDB cannot unify is a bind fault on no row's value, and was left alone; so was
        # a cast target the reader here had rebuilt wrongly (an ENUM without its labels, Codex's
        # fourteenth review), which is the fail-open side. Refused, fail-closed: the statement
        # the engine cannot bind is refused either way.
        return True
    return any(node != unified and _text_becomes_json(node, unified) for node in known)


def _duckdb_type(branch: exp.Expression, type_of) -> str | None:
    """The type DuckDB gives a branch, in DuckDB's words, or `_TEXT`. A column's is the
    snapshot's; a cast's its target; a pure value expression's (`7`, `1.5`, `id + 1`, `INTERVAL
    '1 day'`, `[1, 2]`, a nested CASE) is asked of DuckDB with each column replaced by a NULL of
    its snapshot type, so nothing runs and the answer is the engine's (`typeof(7)` is INTEGER,
    `typeof(1.5)` DECIMAL(2,1), `typeof(id + 1)` over an INTEGER id is INTEGER, where the type
    reader here says DOUBLE for each: the review gate on #160). None where it cannot be told
    without running something: a call, a subquery whose projection is not pure. The type
    reader's guess is not taken instead (Codex's seventh review of #160: it typed a nested CASE
    of integers and a counted subquery DOUBLE, and 'nan' passed the cast to DOUBLE where DuckDB
    converts it to INTEGER and BIGINT on the row that takes it, and fails)."""
    while isinstance(branch, exp.Paren):
        branch = branch.this
    kind = None
    if isinstance(branch, exp.Column):
        kind = type_of(branch)
    elif isinstance(branch, exp.Cast):
        kind = branch.to.sql(dialect="duckdb")
    elif isinstance(branch, exp.Subquery):
        # A scalar subquery is its one projection's type (the gate's `(SELECT id + 1 FROM
        # entitlement LIMIT 1)` beside 'nan'); the projection alone is handed to DuckDB, never
        # the subquery, whose FROM would run.
        inner = branch.this
        if isinstance(inner, exp.Select) and len(inner.selects) == 1:
            kind = _pure_type(inner.selects[0].unalias(), type_of)
    else:
        kind = _pure_type(branch, type_of)
    if kind is None:
        return None
    try:
        parsed = _parse_type(kind)  # DuckDB's own grammar: a nested unnamed struct reads too
    except ValueError:
        # A declared type (a column's, from the snapshot; a cast's target) the reader cannot
        # parse is not an unknown value: it may hold JSON, and reading it as unknown let a
        # UNION("a""b" JSON) member pass (Codex's sixteenth review of #160). Refused by every
        # reader of this answer.
        return _UNREADABLE
    if parsed[0] == "leaf" and (_bare(parsed[1]) in _TEXT_NAMES or _bare(parsed[1]) == "JSON"):
        return _TEXT
    return kind


def _literal_cannot_convert(node: exp.Expression, type_of, engine: str) -> bool:
    """A text literal on a branch beside a branch of a known non-text type takes the branches'
    type, and is converted on the rows that take it: `CASE WHEN id = 1 THEN 7 ELSE 'north' END =
    7` runs on the first row and fails on the second (measured on DuckDB; Codex's fifth review of
    #160), as do a DATE, a BOOLEAN and an INTERVAL beside `'x'`. Whether a literal converts is
    DuckDB's to say, asked through TRY_CAST to the type DuckDB unifies the typed branches to
    (Python's parsers take an out-of-range integer, Unicode digits, a basic ISO date and `nan`,
    which DuckDB refuses: the review gate); refused where DuckDB cannot be asked. A text branch
    (a column, a cast) beside the literal means the value is text, or the mix is a bind fault on
    no row's value, so no literal is converted. A branch whose type cannot be told without
    running it, a call or a subquery with an aggregate, leaves the literal's target unknown,
    and the literal is refused (Codex's seventh review of #160); a cast in its place, `CAST('7'
    AS DOUBLE)`, is a typed branch and not a literal."""
    literals, kinds, unknown = [], [], False
    for branch in _branches(node):
        while isinstance(branch, exp.Paren):
            branch = branch.this
        if isinstance(branch, exp.Literal) and branch.is_string:
            literals.append(branch.this)
        elif not isinstance(branch, exp.Null):
            kind = _duckdb_type(branch, type_of)
            if kind is None or kind is _UNREADABLE:
                unknown = True
            elif kind is _TEXT:
                return False
            else:
                kinds.append(kind)
    if not literals:
        return False
    if unknown:
        return True
    if not kinds:
        return False
    try:
        unified = _duckdb("SELECT typeof(COALESCE(" + ", ".join(f"NULL::{k}" for k in kinds) + "))")
        if _parse_type(unified)[0] != "leaf":
            # To a list, a struct or a map, TRY_CAST gives a container of NULLs where an
            # element does not convert (`TRY_CAST('[north]' AS INTEGER[])` is `[NULL]`), not
            # NULL, so it cannot say whether the literal converts; the literal is refused, and
            # `CAST('[1]' AS INTEGER[])` is a typed branch (the review gate on #160).
            return True
        return any(not _duckdb(f"SELECT TRY_CAST($1 AS {unified}) IS NOT NULL", [text])
                   for text in literals)
    except Exception:  # noqa: BLE001 -- unreadable: the literal may fail on a row, refused
        return True


def _nvl_converts_a_text_column(node: exp.Expression, type_of, engine: str) -> bool:
    """Oracle's NVL and NVL2 take their leading argument's type and convert the other argument
    to it row by row: `NVL(id, region)` is ORA-01722 on a row whose id is NULL and whose region
    is not a number (measured; the review gate on #160), where COALESCE and CASE refuse the mix
    at parse, on no row's value."""
    if isinstance(node, exp.Coalesce) and node.args.get("is_nvl"):
        lead, others = node.this, node.expressions
    elif isinstance(node, exp.Nvl2):
        lead, others = node.args["true"], [node.args["false"]]
    else:
        return False
    typed = _known_type(lead, type_of, engine)
    if typed is None or typed.this in exp.DataType.TEXT_TYPES:
        return False
    return any((known := _known_type(other, type_of, engine)) is not None
               and known.this in exp.DataType.TEXT_TYPES and other.find(exp.Column) is not None
               for other in others)


def _sides(node: exp.Expression) -> list:
    """The values a comparison, NULLIF, IN or BETWEEN sets against each other."""
    if isinstance(node, exp.In):
        sides = [node.this, *node.expressions]
        query = node.args.get("query")
        inner = query.this if isinstance(query, exp.Subquery) else query
        if isinstance(inner, exp.Select) and len(inner.selects) == 1:
            sides.append(inner.selects[0])
        return sides
    if isinstance(node, exp.Between):
        return [node.this, node.args.get("low"), node.args.get("high")]
    return [node.this, node.expression]


def _converts_a_text_column(node: exp.Expression, type_of, engine: str) -> bool:
    """Whether a comparison sets a text column against a value of another type."""
    text = other = False
    for side in _sides(node):
        while isinstance(side, exp.Paren):
            side = side.this  # `(region) = 7` converts the column as `region = 7` does (Codex, #160)
        known = _type_of_value(side, type_of, engine)  # a container accessor's, as DuckDB's
        if known is None:
            continue
        if known.this in exp.DataType.TEXT_TYPES:
            # A text side holding a column is converted row by row: the column itself, or a
            # text-typed value over one (`NVL(region, id) = 7` on Oracle, `COALESCE(region, 'x')
            # = 7` on DuckDB, measured). A text literal holds no column and is never counted.
            # The column may sit in a condition: a literal the row chooses by a column is read on
            # the rows that choose it, and whether it converts is not evaluated here, so
            # `CASE WHEN id > 0 THEN 'x' ELSE '1' END = 7` is refused (it fails on the rows with
            # id > 0, the review gate on #160) and so is the same with '0', which would run; the
            # operator writes such literals in the other side's type.
            text = text or side.find(exp.Column) is not None
        else:
            other = True
    return text and other


def raising_conversion(filt: str, table: str, dialect: str, policy: AccessPolicy, *,
                       executes_as: str | None = None) -> exp.Expression | None:
    """The first conversion in `table`'s row filter that can raise on a row's value, else None.

    For a filter `_validate_filter` accepted. Two rules, by the shape. An explicit conversion
    (a cast, a parse) is type-aware where the snapshot knows the operand's type, which takes a
    bare column: `CAST(id AS BIGINT)` over an INTEGER cannot fail, and is kept; an operand
    reading no column is a constant, which cannot fail on a row; anything else is refused, an
    expression's type, or a column's the snapshot does not record, being unknown. An implicit
    conversion (a comparison, NULLIF, Oracle's NVL and NVL2) is refused only where both sides'
    types are known and one is a text column or a text-typed value over one: a side of unknown
    type (`UPPER(region) > 0`, `NVL(id, TRIM(region))`) is not counted, since refusing every
    comparison with an untyped expression would refuse most filters, and the guide says so. The
    check matches known spellings; it is not a proof.
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
            # The SELECT's own sources, not a subquery's inside it: `COALESCE(attrs, (SELECT attrs
            # FROM claim WHERE id = 1))` reads the outer `attrs` from the filtered table alone,
            # and counting the subquery's `claim` beside it read the name as ambiguous and the
            # JSON column as unknown (Codex's fifth review of #160).
            own = [t for t in base_tables(select, engine)
                   if t.find_ancestor(exp.Select) is select] if select is not None else []
            while select is not None and select.args.get("from_") is None:
                # A SELECT with no FROM at all reads the enclosing one's sources, as the engine
                # resolves a correlated name (`(SELECT region) = 7`: Codex's twenty-third
                # review of #160, the column unknown and the conversion unread). One with a
                # FROM, a derived table among its sources, keeps its own, so a derived column
                # is not read as the outer base column of the same name (the review gate).
                select = select.find_ancestor(exp.Select)
                own = [t for t in base_tables(select, engine)
                       if t.find_ancestor(exp.Select) is select] if select is not None else []
            having = [key for key in {object_key(t) for t in own}
                      if column.name.lower() in types_of(key)]
            owner = having[0] if len(having) == 1 else None
        return None if owner is None else types_of(owner).get(column.name.lower())

    for node in wrapped.args["where"].walk():
        if engine == "duckdb" and _arrow_read_as_a_lambda(node):
            return node  # `attrs -> '$.a'` where DuckDB extracts JSON and sqlglot sees a lambda
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
        if isinstance(node, (*_COMPARISONS, exp.Nullif)):
            # NULLIF compares its two arguments as `=` does (measured: `NULLIF(region, 7)`
            # converts the column, Codex's second review of #160).
            if _converts_a_text_column(node, type_of, engine):
                return node
            if engine == "duckdb" and _json_inside_a_container(_sides(node), type_of):
                return node
            continue
        if engine == "oracle" and _nvl_converts_a_text_column(node, type_of, engine):
            return node
        if engine == "duckdb" and isinstance(node, _BRANCHED) and (
                _coerces_to_json(node, type_of, engine)
                or _json_inside_a_container(_branches(node), type_of)
                or _literal_cannot_convert(node, type_of, engine)):
            return node
        if engine == "duckdb" and (isinstance(node, exp.Bracket)
                                   or (isinstance(node, exp.ToMap) and isinstance(node.this, exp.Bracket))
                                   or (isinstance(node, exp.Anonymous) and node.name.lower() in _MAP_LOOKUPS)):
            # A map looked up by a key: the key is converted to the map's key type, so a text
            # key beside a JSON-keyed map is parsed on every row (Codex's eleventh review of
            # #160). The map is typed by DuckDB; its key type stands in for the map beside
            # the key under both readings. A list subscripted by a number converts nothing.
            # A subscripted map literal is parsed with the bracket inside the map (the review
            # gate on cd17747: `MAP {...}[region]`), so the map is rebuilt over its struct.
            map_only = isinstance(node, exp.Anonymous) and node.name.lower() in _MAP_ONLY_LOOKUPS
            if isinstance(node, exp.ToMap):
                bracket = node.this
                container = exp.ToMap(this=bracket.this.copy())
                container.parent = node
                key = bracket.expressions[0] if bracket.expressions else None
            elif isinstance(node, exp.Bracket):
                container, key = node.this, (node.expressions[0] if node.expressions else None)
            else:
                operands = list(node.expressions)
                container, key = (operands + [None, None])[:2]
            kind = _type_of_value(container, type_of, engine) if container is not None else None
            if (key is not None and kind is not None and kind.this == exp.DataType.Type.MAP
                    and kind.expressions and isinstance(kind.expressions[0], exp.DataType)):
                key_type = exp.Cast(this=exp.Null(), to=kind.expressions[0].copy())
                key_type.parent = node
                pair = [key_type, key]
                if _values_coerce_to_json(pair, type_of, engine) or _json_inside_a_container(pair, type_of):
                    return node
            elif key is not None and kind is None and map_only:
                return node  # a map the reader cannot type may be keyed by JSON (the gate)
        if engine == "duckdb" and (isinstance(node, _LIST_SEARCHES) or (
                isinstance(node, exp.Anonymous) and node.name.lower() in _LIST_SEARCH_NAMES)):
            # A list search unifies the list's element type with what is searched for, or with
            # the other list's elements: `list_contains([attrs], region)` and `array_position`
            # parse region as JSON on every row (Codex's tenth review of #160). Read as the
            # element (the list subscripted) beside the argument, or the two lists together.
            if isinstance(node, exp.Anonymous):
                operands = list(node.expressions)
                haystack, needle = (operands + [None, None])[:2]
            else:
                haystack, needle = node.this, node.args.get("expression")
            if isinstance(node, exp.Contains):
                # DuckDB's `contains` searches a string as well as a list: a list search only
                # where the reader types the first argument as a list (the review gate: over a
                # JSON column it is the string search, and over a list a call returns it is
                # left to the other rules).
                kind = _type_of_value(haystack, type_of, engine) if haystack is not None else None
                if kind is None or kind.this not in (exp.DataType.Type.ARRAY, exp.DataType.Type.LIST):
                    haystack = None
            if haystack is not None and needle is not None:
                pairs = isinstance(node, _LIST_PAIRS) or (
                    isinstance(node, exp.Anonymous) and node.name.lower() in _LIST_PAIR_NAMES)
                element = exp.Bracket(this=haystack.copy(), expressions=[exp.Literal.number(1)])
                element.parent = node
                pair = [haystack, needle] if pairs else [element, needle]
                if _values_coerce_to_json(pair, type_of, engine) or _json_inside_a_container(pair, type_of):
                    return node
                lists = pair if pairs else [element]
                if any(_untypable(value, type_of, engine) for value in lists):
                    # A list the reader cannot type (`list_distinct([attrs])`, a call) may hold
                    # JSON, and the search would parse what it looks for (the review gate on
                    # e12b82d): refused, fail-closed. The value searched for, where untyped
                    # beside a JSON element, is refused above, and beside a text element it
                    # parses nothing. Typed in DuckDB's own grammar: sqlglot's DataType does
                    # not read `STRUCT(INTEGER, "NULL"[])`, which `row(id, [])` is, and the
                    # valid search over it was refused (Codex's eighteenth review of #160).
                    return node
        if engine == "duckdb" and (isinstance(node, (exp.Array, exp.ToMap)) or (
                isinstance(node, exp.Anonymous) and node.name.lower() in _LIST_CONSTRUCTORS)):
            # A list's elements, and a map's keys and its values, are unified to one type as a
            # branched value's branches are: `[attrs, region]` is JSON[], region parsed on every
            # row (Codex's ninth review of #160: indexed, wrapped in a struct, or as a map's
            # values, approved and Malformed JSON on the hidden row). The list spelled as a
            # call (`array_value`, `list_value`, `list_pack`) and a map literal subscripted at
            # once (`MAP {...}['b']`, whose struct sqlglot wraps in the bracket) are the same
            # shapes (the review gate on 3d52790).
            if isinstance(node, exp.ToMap):
                struct = node.this
                while isinstance(struct, (exp.Bracket, exp.Paren)):
                    struct = struct.this
                items = struct.expressions if isinstance(struct, exp.Struct) else []
                groups = [[_map_key(item) for item in items if isinstance(item, exp.PropertyEQ)],
                          [item.expression for item in items if isinstance(item, exp.PropertyEQ)]]
            else:
                groups = [node.expressions]
            if any(_values_coerce_to_json(group, type_of, engine) or _json_inside_a_container(group, type_of)
                   for group in groups if group):
                return node
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
    if engine == "duckdb" and isinstance(node, exp.Lambda):
        safe = ("json_extract(value, path) in place of `value -> path` here: inside a call's "
                "arguments or beside an operator the parser reads the arrow as a lambda where "
                "DuckDB reads a JSON extract, so the filter cannot be checked as DuckDB runs it")
    if engine == "duckdb" and isinstance(node, _BRANCHED):
        safe += (", and a text literal on a branch as CAST('...' AS type): beside a branch of "
                 "another type it is converted to that type on the rows that take it, and beside "
                 "a call or an aggregate the type it would be converted to cannot be told")
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
