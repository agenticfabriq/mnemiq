"""Saved metrics: the ones a dataset already carries, read as the builder, and reused where a planned measure says the
same thing.

`list_metrics` (Superset's own tool) gives each dataset's saved metrics, paged. It carries no currency and no
certification, so the extension's `dataset_metrics` adds them; a Superset without the extension leaves both `UNKNOWN`,
which is not "none": nothing is then preferred on certification.

Reuse is deterministic: a saved metric on the measure's own dataset whose normalized expression equals the planned
one. Certified metrics come first, then name order.

A joined table's saved metric is carried onto a join as its table defines it (`carry`): its expression rendered for
mnemiq's own statement over the join and for the joined dataset, from one parse.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mnemiq.sql.identifiers import resolve_identifier
from mnemiq.superset.adapter import JINJA_DELIMITERS
from mnemiq.superset.client import SupersetResponseTooLarge, SupersetToolError
from mnemiq.superset.extension import ExtensionMissing, dataset_metrics
from mnemiq.superset.measures import Measure
from mnemiq.superset.settings import SUPERSET_DIALECT

__all__ = ["Carried", "SavedMetric", "UNKNOWN", "carry", "normalized", "read_saved_metrics", "reusable"]

UNKNOWN: object = object()

PAGE_SIZE = 25
# The page size a retry asks for after Superset's response size guard refused a page: a fifth of the default.
# `EMBEDDED_DIMENSIONS_MAX_PAGE_SIZE` (4) in `superset/mcp_service/semantic_layer/schemas.py` is the cap only when
# `include_compatible_dimensions` is on, which mnemiq never sends, so it does not bind this request.
SMALL_PAGE_SIZE = 5


@dataclass(frozen=True)
class SavedMetric:
    dataset_id: int
    name: str
    expression: str
    verbose_name: str | None
    d3format: str | None
    currency: dict | None | object
    certified_by: str | None | object
    warning_text: str | None


def normalized(expression: str) -> str:
    """The expression parsed and written back as Postgres, with identifiers normalized. Raises on SQL that does not
    parse."""
    return sqlglot.parse_one(expression, dialect="postgres").sql(dialect="postgres", normalize=True)


def _list_dataset(client, dataset_id: int, page_size: int) -> list[dict]:
    rows: list[dict] = []
    page = 1
    while True:
        answer = client.call("list_metrics", {"dataset_id": dataset_id, "page_size": page_size, "page": page})
        if answer.get("success") is False or "metrics" not in answer:
            raise SupersetToolError("Superset's list_metrics tool answered with an error", detail=str(answer))
        rows.extend(answer["metrics"])
        if page >= int(answer.get("total_pages") or 1):
            return rows
        page += 1


def _read_dataset(client, dataset_id: int) -> list[dict]:
    """Every saved metric of one dataset, or `SupersetResponseTooLarge`: never a partial list. A page Superset's size
    guard refused is asked for again, from the first page, at a smaller size, once."""
    try:
        return _list_dataset(client, dataset_id, PAGE_SIZE)
    except SupersetResponseTooLarge:
        return _list_dataset(client, dataset_id, SMALL_PAGE_SIZE)


def read_saved_metrics(client, *, dataset_ids: tuple[int, ...]) -> list[SavedMetric]:
    listed = {dataset_id: _read_dataset(client, dataset_id) for dataset_id in dataset_ids}
    try:
        extra = {(d["dataset_id"], m["metric_name"]): m for d in dataset_metrics(client, dataset_ids)
                 for m in d.get("metrics") or []}
        missing = False
    except ExtensionMissing:
        extra, missing = {}, True
    saved: list[SavedMetric] = []
    for dataset_id, rows in listed.items():
        for row in rows:
            if not row.get("expression") or not row.get("name"):
                continue
            # A metric the extension does not answer for is unknown, not "without currency or certification": the
            # builder was just shown it by `list_metrics`, so the two views disagree or it changed between the calls.
            more = extra.get((dataset_id, row["name"]))
            known = not missing and more is not None
            saved.append(SavedMetric(
                dataset_id=dataset_id, name=row["name"], expression=row["expression"],
                verbose_name=row.get("verbose_name"), d3format=row.get("d3format"),
                currency=more.get("currency") if known else UNKNOWN,
                certified_by=more.get("certified_by") if known else UNKNOWN,
                warning_text=row.get("warning_text")))
    return saved


def same_definition(kept: Measure, planned: Measure) -> bool:
    """Whether `planned` is the measure `kept`: its kind and dataset the same, its expression the same once normalized
    (as written, when it does not parse). A dataset is part of a measure: the same formula on another dataset is
    another measure, which its dashboard keeps apart."""
    if kept.kind != planned.kind or kept.dataset_id != planned.dataset_id:
        return False
    try:
        return normalized(kept.expression) == normalized(planned.expression)
    except SqlglotError:
        return kept.expression.strip() == planned.expression.strip()


def reusable(measure: Measure, saved: list[SavedMetric]) -> SavedMetric | None:
    """The saved metric on the measure's dataset that says what the measure says, certified ones first, then by name."""
    try:
        wanted = normalized(measure.expression)
    except SqlglotError:
        return None
    matches = []
    for metric in saved:
        if metric.dataset_id != measure.dataset_id:
            continue
        try:
            if normalized(metric.expression) == wanted:
                matches.append(metric)
        except SqlglotError:
            continue
    certified = lambda m: isinstance(m.certified_by, str) and bool(m.certified_by)  # noqa: E731
    matches.sort(key=lambda m: (not certified(m), m.name))
    return matches[0] if matches else None


@dataclass(frozen=True)
class Carried:
    """A saved metric's expression rendered for a join, twice from one parse, differing only in column references:
    `decided` names each column by its table's join qualifier (`c."amount"`), for mnemiq's own statement over the
    join; `superset` names it as the joined dataset does (`"amount"`, or `"claim_created_at"` for a column two joined
    tables share), for the chart."""

    decided: str
    superset: str


_NOT_ONE = "mnemiq cannot read its expression as one SQL expression"
# Names sqlglot reads as a column where Postgres, unquoted and unqualified, reads a function of the session: carried
# as `c."user"`, a metric of `MAX(user)` would read a column named user instead.
_POSTGRES_FUNCTIONS = frozenset({"user", "current_role", "system_user"})


def carry(metric: SavedMetric, *, table: str, stored: str, columns: Collection[str], qualifier: exp.Identifier,
          shown: Mapping[str, str], rows_are_base: bool = False) -> Carried | str:
    """`metric`'s expression carried onto a join as `table` defines it, or why it cannot be.

    `table` is the table as the snapshot names it and `stored` as Superset stores it: a column the expression names
    with a table names one of these. `columns` are the table's columns the builder can read, `qualifier` the table's
    qualifier in the join (its alias, else its name, with its quoting), and `shown` each of `columns`' name on the
    joined dataset. The expression reads only its own table's rows, by naming no other table's columns, and at least
    one of its own unless `rows_are_base`; what it computes is the deciders' to check (`check_measure`). Printed by
    sqlglot, so the text may differ from the expression's own (`amount::numeric` is `CAST(amount AS DECIMAL)`), both
    renderings alike.

    `rows_are_base` says the join's rows are `table`'s, each once: the caller says so only of a join mnemiq made from
    `table`, `table` LEFT JOIN tables whose far sides the plan check showed hold each value once. There an expression
    that names no column (`COUNT(*)`, `SUM(1)`) counts `table`'s rows, as on `table` alone, and is carried as written;
    on any other join it would count the join's rows, and is refused."""
    expression = metric.expression
    # The joined dataset's measure is a Superset SQL metric, which Superset renders as a template.
    if (marker := next((d for d in JINJA_DELIMITERS if d in expression), None)) is not None:
        return f"it holds {marker}, which Superset reads as a template marker"
    try:
        tree = sqlglot.parse_one(f"SELECT {expression}", read=SUPERSET_DIALECT)
    except SqlglotError:
        return _NOT_ONE
    # One expression and nothing else: no FROM, DISTINCT or second expression, and no alias (`AS total`, `AS (a, b)`),
    # which names a result column where a metric has none.
    if not isinstance(tree, exp.Select) or len(tree.expressions) != 1 \
            or {key for key, value in tree.args.items() if value} != {"expressions"} \
            or isinstance(tree.expressions[0], (exp.Alias, exp.Aliases)):
        return _NOT_ONE
    # An alias inside it is how sqlglot reads `(TABLE t)`: a column named TABLE, aliased t.
    if len(list(tree.find_all(exp.Select))) != 1 or tree.find(exp.Subquery, exp.Exists, exp.Table, exp.Alias):
        return "it reads more than its own table's rows: a subquery, EXISTS or another table"
    named = list(tree.find_all(exp.Column))
    for column in named:
        if isinstance(column.this, exp.Star):
            return "it names *"
        if column.args.get("db") or column.args.get("catalog"):
            return "it names a column with a schema"
        if column.args.get("table") is not None \
                and resolve_identifier(column.args["table"], SUPERSET_DIALECT) not in (table, stored):
            return f"it names {column.sql(dialect=SUPERSET_DIALECT)}, a column of another table"
        if column.args.get("table") is None and not column.this.args.get("quoted") \
                and resolve_identifier(column.this, SUPERSET_DIALECT) in _POSTGRES_FUNCTIONS:
            return (f"it names {column.this.sql(dialect=SUPERSET_DIALECT)}, which Postgres reads as a function, not a "
                    "column")
        if resolve_identifier(column.this, SUPERSET_DIALECT) not in columns:
            return (f"it names {column.this.sql(dialect=SUPERSET_DIALECT)}, which {table} does not have or the "
                    "builder cannot read")
    if not named and not rows_are_base:
        return f"it names no column of {table}, so on a join it would count the join's rows"

    def render(column_for) -> str:
        return tree.expressions[0].transform(
            lambda node: column_for(resolve_identifier(node.this, SUPERSET_DIALECT))
            if isinstance(node, exp.Column) else node).sql(dialect=SUPERSET_DIALECT)

    return Carried(
        decided=render(lambda name: exp.Column(this=exp.to_identifier(name, quoted=True), table=qualifier.copy())),
        superset=render(lambda name: exp.Column(this=exp.to_identifier(shown[name], quoted=True))))
