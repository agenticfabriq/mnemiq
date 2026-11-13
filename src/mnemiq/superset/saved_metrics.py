"""Saved metrics: the ones a dataset already carries, read as the builder, and reused where a planned measure says the
same thing.

`list_metrics` (Superset's own tool) gives each dataset's saved metrics, paged. It carries no currency and no
certification, so the extension's `dataset_metrics` adds them; a Superset without the extension leaves both `UNKNOWN`,
which is not "none": nothing is then preferred on certification.

Reuse is deterministic: a saved metric on the measure's own dataset whose normalized expression equals the planned
one. Certified metrics come first, then name order.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot.errors import SqlglotError

from mnemiq.superset.client import SupersetResponseTooLarge, SupersetToolError
from mnemiq.superset.extension import ExtensionMissing, dataset_metrics
from mnemiq.superset.measures import Measure

__all__ = ["SavedMetric", "UNKNOWN", "normalized", "read_saved_metrics", "reusable"]

UNKNOWN: object = object()

PAGE_SIZE = 25
# The page size a retry asks for after Superset's response size guard refused a page: the plan's value, a fifth of the
# default. `EMBEDDED_DIMENSIONS_MAX_PAGE_SIZE` (4) in `superset/mcp_service/semantic_layer/schemas.py` is the cap only
# when `include_compatible_dimensions` is on, which mnemiq never sends, so it does not bind this request.
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
