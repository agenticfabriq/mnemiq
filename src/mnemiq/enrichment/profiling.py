from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot

from mnemiq.adapters.base import SourceAdapter
from mnemiq.catalog import TableInfo, is_key_like, is_sensitive_name

# Column types no aggregate may touch. Oracle raises ORA-22849, "Type CLOB is not supported for
# this function or operator", for **`count(col)` as well as `count(DISTINCT col)`** -- measured
# against a live 23ai/26ai instance, not inferred from the DISTINCT restriction alone. The list is
# Oracle's documented LOB family plus VECTOR, and it is matched on the BASE type name because
# `all_tab_columns` reports `VECTOR` for a column declared `VECTOR(1024, FLOAT32)`.
#
# Keyed by dialect because it is a property of the SOURCE, not of this module: Postgres aggregates
# `text` and `bytea` without complaint, and a shared exclusion list would silently stop profiling
# columns there that profile fine.
_UNAGGREGATABLE: dict[str, frozenset[str]] = {
    "oracle": frozenset({"CLOB", "NCLOB", "BLOB", "BFILE", "LONG", "LONG RAW", "VECTOR",
                         "XMLTYPE", "ROWID", "UROWID"}),
}


def _unaggregatable(adapter: SourceAdapter) -> frozenset[str]:
    return _UNAGGREGATABLE.get(getattr(adapter, "dialect", ""), frozenset())


def _base_type(data_type: str) -> str:
    """`VECTOR(1024, FLOAT32)` -> `VECTOR`, `TIMESTAMP(6) WITH TIME ZONE` -> `TIMESTAMP`."""
    return data_type.split("(", 1)[0].strip().upper()


@dataclass
class ColumnStats:
    table: str
    column: str
    row_count: int
    distinct_count: int | None
    null_count: int | None
    top_k: list[tuple] = field(default_factory=list)


def _top_k(adapter: SourceAdapter, table: str, column: str, k: int) -> list[tuple]:
    if k <= 0:  # callers that only need counts (e.g. key detection) pass k=0
        return []
    # NULL is an absence, not an observed value -- it must never become a code
    # Transpiled to the source's dialect rather than emitted raw. `LIMIT` is Postgres/DuckDB
    # syntax: Oracle answers ORA-03049, "SQL keyword 'LIMIT' is not syntactically valid", and
    # wants `FETCH FIRST k ROWS ONLY`. Measured against a real Oracle schema, that one clause
    # excluded 20 of 51 tables from the model. sqlglot is already this project's transpiler and
    # knows the rewrite; a hand-rolled dialect switch here would be a second, drifting one.
    # The column is NAMED in GROUP BY and ORDER BY rather than referenced by ordinal. `GROUP BY 1`
    # is a positional reference Oracle rejects by default -- ORA-03162, "must appear in the GROUP
    # BY clause ... as 'group_by_position_enabled' is FALSE' -- and sqlglot does not rewrite an
    # ordinal into a name, so transpiling alone left this broken: it fixed `LIMIT` and 26 tables
    # still failed on the ordinal. Naming the column needs no dialect knowledge at all.
    query = (
        f'SELECT "{column}", count(*) AS c FROM "{table}" WHERE "{column}" IS NOT NULL '
        f'GROUP BY "{column}" ORDER BY c DESC, "{column}" LIMIT {k}'
    )
    dialect = getattr(adapter, "dialect", "duckdb")
    if dialect != "duckdb":
        query = sqlglot.transpile(query, read="duckdb", write=dialect)[0]
    rows = adapter.execute(query)
    return [(r[0], r[1]) for r in rows]


def profile_column(adapter: SourceAdapter, table: str, column: str, k: int = 10,
                   data_type: str | None = None) -> ColumnStats:
    """`data_type` lets this refuse the same column types `profile_table` skips.

    Without it this was the sibling hole to the one `profile_table` closes. `infer_relationships`
    calls this for every key-LIKE column -- anything named `*_id` or `*_identifier` -- and on Oracle
    a CLOB or VECTOR with such a name raises ORA-22849 here. That exception leaves
    `build_relationships` entirely, and the pipeline marks the whole `infer:relationships` job
    failed, so ONE column costs every relationship in the source. The same "one bad column takes
    down the operation" shape, one entry point over, found by review of the commit that fixed it in
    the other.

    Callers that cannot supply a type get today's behaviour exactly: the column is queried.
    """
    if data_type is not None and _base_type(data_type) in _unaggregatable(adapter):
        # No query at all. Unknown counts make `is_key_of` false, which is the conservative and
        # correct answer -- a column whose values cannot be counted cannot be shown to be a key.
        return ColumnStats(table=table, column=column,
                           row_count=adapter.execute(f'SELECT count(*) FROM "{table}"')[0][0],
                           distinct_count=None, null_count=None)
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
    if not table.columns:
        return []
    keys = key_columns or set()

    # Columns the source cannot aggregate are left OUT of the counts query and carried through
    # with unknown stats. They used to be included, and because this is ONE query for the whole
    # table, a single CLOB made the statement fail and excluded EVERY column of that table --
    # measured on a real Oracle schema, 27 of 51 tables lost entirely to that, on top of the 20
    # lost to `LIMIT`. A column whose values cannot be counted is still a column the model should
    # know exists; its absence is what makes the engine answer as though the field is not there.
    blocked = _unaggregatable(adapter)
    cols = [c.name for c in table.columns if _base_type(c.data_type) not in blocked]
    skipped = [c.name for c in table.columns if _base_type(c.data_type) in blocked]

    if cols:
        selects = ["count(*)"]
        for c in cols:
            selects += [f'count(DISTINCT "{c}")', f'count("{c}")']
        row = adapter.execute(f'SELECT {", ".join(selects)} FROM "{table.name}"')[0]
        row_count = row[0]
    else:
        # Every column is unaggregatable, so there is nothing to count them WITH; the row count
        # still is, and it is what tells a reader the table is not empty.
        row_count = adapter.execute(f'SELECT count(*) FROM "{table.name}"')[0][0]
        row = [row_count]

    stats: list[ColumnStats] = []
    for name in skipped:
        # distinct_count None, not 0: nothing was measured. A zero here would read as "no values"
        # and could make the column look like an empty coded vocabulary.
        stats.append(ColumnStats(table=table.name, column=name, row_count=row_count,
                                 distinct_count=None, null_count=None))
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
            # Harvest the WHOLE vocabulary, not a top-k slice: the column already qualifies as a
            # code set (<= code_max_distinct distinct), so a rare code must not be dropped -- else a
            # dictionary can never ground it and the model never sees it. Bounded by the cap.
            s.top_k = _top_k(adapter, table.name, c, code_max_distinct)
        stats.append(s)
    return stats
