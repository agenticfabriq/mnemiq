from __future__ import annotations

from dataclasses import dataclass, field

from mnemiq.adapters.base import SourceAdapter
from mnemiq.catalog import TableInfo, is_key_like, is_sensitive_name


@dataclass
class ColumnStats:
    table: str
    column: str
    row_count: int
    distinct_count: int
    null_count: int
    top_k: list[tuple] = field(default_factory=list)


def _top_k(adapter: SourceAdapter, table: str, column: str, k: int) -> list[tuple]:
    if k <= 0:  # callers that only need counts (e.g. key detection) pass k=0
        return []
    # NULL is an absence, not an observed value -- it must never become a code
    rows = adapter.execute(
        f'SELECT "{column}", count(*) AS c FROM "{table}" WHERE "{column}" IS NOT NULL '
        f"GROUP BY 1 ORDER BY c DESC, 1 LIMIT {k}"
    )
    return [(r[0], r[1]) for r in rows]


def profile_column(adapter: SourceAdapter, table: str, column: str, k: int = 10) -> ColumnStats:
    row_count, distinct_count, non_null = adapter.execute(
        f'SELECT count(*), count(DISTINCT "{column}"), count("{column}") FROM "{table}"'
    )[0]
    return ColumnStats(
        table=table,
        column=column,
        row_count=row_count,
        distinct_count=distinct_count,
        null_count=row_count - non_null,
        top_k=_top_k(adapter, table, column, k),
    )


def profile_table(
    adapter: SourceAdapter,
    table: TableInfo,
    k: int = 10,
    code_max_distinct: int = 25,
    key_columns: set[str] | None = None,
) -> list[ColumnStats]:
    """One counts query for the whole table; top-k only where it is cheap and useful."""
    cols = [c.name for c in table.columns]
    if not cols:
        return []
    keys = key_columns or set()

    selects = ["count(*)"]
    for c in cols:
        selects += [f'count(DISTINCT "{c}")', f'count("{c}")']
    row = adapter.execute(f'SELECT {", ".join(selects)} FROM "{table.name}"')[0]

    row_count = row[0]
    stats: list[ColumnStats] = []
    for i, c in enumerate(cols):
        distinct_count, non_null = row[1 + 2 * i], row[2 + 2 * i]
        s = ColumnStats(
            table=table.name,
            column=c,
            row_count=row_count,
            distinct_count=distinct_count,
            null_count=row_count - non_null,
        )
        # A small observed value set is a candidate coded vocabulary; the semantic pass gives
        # the codes meaning. Two kinds of column are excluded before we ever read a value:
        #   - keys: low distinct-count on a foreign key is an artifact of sample size
        #     (by name suffix, or declared in the catalog -- key_columns)
        #   - sensitive: harvesting a name column would copy real people into the snapshot
        if (
            not is_key_like(c)
            and c not in keys
            and not is_sensitive_name(c)
            and 0 < distinct_count <= code_max_distinct
            and distinct_count < row_count
        ):
            s.top_k = _top_k(adapter, table.name, c, k)
        stats.append(s)
    return stats
