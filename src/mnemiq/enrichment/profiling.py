from __future__ import annotations

import logging
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
logger = logging.getLogger(__name__)

_UNAGGREGATABLE: dict[str, frozenset[str]] = {
    "oracle": frozenset({"CLOB", "NCLOB", "BLOB", "BFILE", "LONG", "LONG RAW", "VECTOR",
                         "XMLTYPE", "ROWID", "UROWID"}),
}


# Why a column's count failed, from the DATABASE rather than from a proxy.
#
# The previous two versions inferred it: run a DISTINCT that names no user column and see whether
# it works. That was unsound in both directions and each fix moved the unsoundness rather than
# removing it -- unbounded, the probe could fail the table it was diagnosing; bounded, it could
# succeed on 100 dense integers where every real column's DISTINCT fails on memory, and a dedup
# over ROWNUM is nothing like one over a wide high-cardinality column however many rows it reads.
# A proxy workload cannot answer a question about a different workload.
#
# Oracle already says why. Measured: CLOB gives ORA-22849, a user-defined object type gives
# ORA-22950, a missing table gives ORA-00942. So a failure whose code is a TYPE error is the
# column's own problem and the table is kept with that column unmeasured; anything else -- a
# capacity error, a permission, an unrecognised code -- is treated as systemic and re-raised,
# which restores the pre-fallback behaviour and is the loud direction.
#
# Only measured codes are listed. An unrecognised type error therefore costs its table, which is
# the conservative way to be wrong: reported FAILED and visible, rather than `done` over a model
# that measured nothing.
_TYPE_ERROR_CODES: dict[str, frozenset[int]] = {"oracle": frozenset({22849, 22950})}


def _is_type_error(adapter: SourceAdapter, exc: Exception) -> bool:
    codes = _TYPE_ERROR_CODES.get(getattr(adapter, "dialect", ""), frozenset())
    err = exc.args[0] if exc.args else None
    return getattr(err, "code", None) in codes


def _unaggregatable(adapter: SourceAdapter) -> frozenset[str]:
    return _UNAGGREGATABLE.get(getattr(adapter, "dialect", ""), frozenset())


def _base_type(data_type: str) -> str:
    """`VECTOR(1024, FLOAT32)` -> `VECTOR`, `TIMESTAMP(6) WITH TIME ZONE` -> `TIMESTAMP`."""
    return data_type.split("(", 1)[0].strip().upper()


# How a column's statistics came to be what they are. Three states, because `None` counts are
# reached two ways that call for opposite responses: an unsupported type is nothing to fix, and a
# failed measurement is.
MEASURED = "measured"
UNSUPPORTED = "unsupported"  # the source cannot aggregate this type at all
FAILED = "failed"            # the measurement was attempted and did not complete


@dataclass
class ColumnStats:
    table: str
    column: str
    row_count: int
    distinct_count: int | None
    null_count: int | None
    top_k: list[tuple] = field(default_factory=list)
    # WHY the counts are None, because `None` alone cannot say. Two very different columns reach
    # it: one whose TYPE the source cannot aggregate -- a LOB, an object type -- which is a normal
    # and permanent property of the schema, and one whose measurement FAILED for a cause that has
    # nothing to do with the column, like temp space or a privilege. Both persisted as
    # `distinct_count=None, null_count=None` and were indistinguishable in the snapshot; the
    # failure existed only in a log line (**M73**). This is the absence/failure collapse inside
    # the per-column fallback added for **M67**, which was itself a fix for that collapse.
    measurement: str = MEASURED
    # Why `measurement` is not MEASURED, for BOTH of the states that are not:
    # the driver's error text for FAILED, and the standing reason for
    # UNSUPPORTED. `None` only on a column that measured.
    failure: str | None = None


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

    measured: dict[str, tuple] = {}
    # column -> (measurement, cause). Absent means measured.
    why: dict[str, tuple[str, str]] = {}
    if cols:
        selects = ["count(*)"]
        for c in cols:
            selects += [f'count(DISTINCT "{c}")', f'count("{c}")']
        try:
            row = adapter.execute(f'SELECT {", ".join(selects)} FROM "{table.name}"')[0]
            row_count = row[0]
            for i, c in enumerate(cols):
                measured[c] = (row[1 + 2 * i], row[2 + 2 * i])
        except Exception as exc:
            # LOGGED, because otherwise this is the absence/failure collapse inside the fix for
            # one: a column skipped by `_UNAGGREGATABLE` and a column that BROKE both end up as
            # `distinct_count=None`, and nothing distinguishes "this type is known-uncountable"
            # from "a permission, a driver fault, or a bug in this file". A known type is skipped
            # before any query and logs NOTHING, so a line here always means something unplanned.
            logger.warning("batched profile of %r failed (%s); measuring column by column",
                           table.name, exc)
            # ONE query per table is the fast path, and it makes any single unaggregatable column
            # fatal to every column beside it. `_UNAGGREGATABLE` removes the ones we can NAME, and
            # it can never be complete: a user-defined object type reports its OWN name as its
            # data type, so `ADDR_T` is unlistable, and measured on a live instance
            # `count(DISTINCT)` on one raises ORA-22950 -- a different code from the ORA-22849 the
            # LOB types give. A denylist is therefore an optimisation and cannot be the guarantee.
            #
            # So on failure, measure column by column. A column that will not count costs ITSELF
            # and nothing else, whatever the reason and whatever Oracle calls it.
            row_count = adapter.execute(f'SELECT count(*) FROM "{table.name}"')[0][0]
            failures: list[Exception] = []
            for c in cols:
                try:
                    measured[c] = adapter.execute(
                        f'SELECT count(DISTINCT "{c}"), count("{c}") FROM "{table.name}"')[0]
                except Exception as col_exc:
                    logger.warning("column %r.%r could not be counted and is carried with unknown "
                                   "stats: %s", table.name, c, col_exc)
                    measured[c] = (None, None)
                    # Recorded per column, not just logged. A log line is not a record: the
                    # snapshot outlives the run that produced it, and a reader of the snapshot
                    # could not tell this column from one whose type cannot be counted at all.
                    why[c] = (UNSUPPORTED if _is_type_error(adapter, col_exc) else FAILED,
                              str(col_exc).strip().splitlines()[0])
                    failures.append(col_exc)
            nothing_measured = all(v == (None, None) for v in measured.values())
            systemic = next((e for e in failures if not _is_type_error(adapter, e)), None)
            if nothing_measured and systemic is not None:
                # Not the column's type, so not the column's fault: capacity, permission, or a
                # cause this does not recognise. Carrying unknowns here would report a table that
                # measured nothing as `done`, which is M59's class. Raise the exception that
                # DECIDED it, not the batched one -- reporting ORA-22950 for an ORA-01652 sends
                # the reader to the wrong problem.
                #
                # Gated on nothing having measured. A PARTIAL failure keeps its table even when one
                # column died of capacity: real statistics for the other columns are worth more
                # than the tidiness of refusing them, and the failure is in the log either way.
                raise systemic
            if all(v == (None, None) for v in measured.values()):
                # Every failure was a recognised TYPE error, so the table is kept with all of its
                # columns unmeasured. That is a real state -- a table whose every column is a LOB
                # or an object type -- and it is why the earlier "all of them failed, therefore
                # systemic" rule was wrong: for a table of ONE column those are the same fact, so
                # a single CLOB was carried while a single `ADDR_T` was excluded, the denylist
                # deciding survival again.
                logger.warning("no column of %r could be counted; all %d carried with unknown "
                               "stats, every failure a known column-type error", table.name,
                               len(measured))
    else:
        # Every column is unaggregatable, so there is nothing to count them WITH; the row count
        # still is, and it is what tells a reader the table is not empty.
        row_count = adapter.execute(f'SELECT count(*) FROM "{table.name}"')[0][0]

    stats: list[ColumnStats] = []
    for name in skipped:
        # distinct_count None, not 0: nothing was measured. A zero here would read as "no values"
        # and could make the column look like an empty coded vocabulary.
        #
        # `UNSUPPORTED`, and it is the whole point of the field: this column is EXPECTED to carry
        # no counts and always will, because the source cannot aggregate its type. Nothing to fix,
        # nothing to retry, and it must not look like the failure below.
        stats.append(ColumnStats(table=table.name, column=name, row_count=row_count,
                                 distinct_count=None, null_count=None,
                                 measurement=UNSUPPORTED,
                                 failure="the source cannot aggregate this column's type"))
    for c in cols:
        distinct_count, non_null = measured[c]
        measurement, failure = why.get(c, (MEASURED, None))
        s = ColumnStats(
            table=table.name,
            column=c,
            row_count=row_count,
            distinct_count=distinct_count,
            null_count=None if non_null is None else row_count - non_null,
            measurement=measurement,
            failure=failure,
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
            and distinct_count is not None  # unmeasured: not a vocabulary candidate
            and 0 < distinct_count <= code_max_distinct
            and distinct_count < row_count
        ):
            # Harvest the WHOLE vocabulary, not a top-k slice: the column already qualifies as a
            # code set (<= code_max_distinct distinct), so a rare code must not be dropped -- else a
            # dictionary can never ground it and the model never sees it. Bounded by the cap.
            s.top_k = _top_k(adapter, table.name, c, code_max_distinct)
        stats.append(s)
    return stats
