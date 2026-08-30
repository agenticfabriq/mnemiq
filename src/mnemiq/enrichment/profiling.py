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


# Asks whether this session can perform a DISTINCT at all, without naming any user column.
#
# BOUNDED, and that is the point rather than an optimisation. The first version ran
# `count(DISTINCT ROWNUM)` over the WHOLE table -- a full access plus a dedup of every row, which
# is precisely the operation most likely to fail under the resource pressure the probe exists to
# detect. On a large table it could turn a benign per-column failure into an excluded table by
# failing itself. A diagnostic must not be more expensive than the thing it diagnoses. Measured on
# 200k rows: 0.0185s unbounded, 0.0011s bounded, and the bounded plan still carries a HASH GROUP BY
# (with a COUNT STOPKEY above it), so it still exercises the machinery.
#
# ROWNUM rather than a constant, and that is measured too: `count(DISTINCT 1)` runs in 0.0020s
# against a table where a real column's DISTINCT takes 0.0081s -- it is folded away, so it would
# answer "yes, this session can sort" for a session that cannot. ROWNUM varies per row and costs
# more than the real column, so it is doing the work.
#
# A dialect with NO entry gets no probe and the conservative answer, never a weak one. The first
# version defaulted to that foldable constant, so on DuckDB -- this project's primary adapter -- a
# session that could not sort still answered yes and a systemic failure landed as `done`.
# Each value is a SQL TEMPLATE carrying a `{table}` placeholder, not a bare expression -- the
# bounding syntax is dialect-specific, so the whole statement belongs to the dialect. An entry
# without the placeholder is refused at import rather than probing the wrong table silently:
# `str.format` ignores an unused keyword, so a malformed entry would run and answer
# confidently about a table nobody asked about. A comment cannot stop that; the check below can.
_DISTINCT_PROBE: dict[str, str] = {
    "oracle": 'SELECT count(DISTINCT ROWNUM) FROM (SELECT 1 FROM "{table}" FETCH FIRST 100 ROWS ONLY)',
}

# The same question without the bound, asked ONLY when the bounded answer is "yes" and every
# column still failed. Bounding fixed a probe that could fail the table it diagnosed; it also
# shrank the probe's footprint far below the batched query's, so a genuine temp-space exhaustion
# could pass on 100 rows while the real full-table sort cannot -- and the table would then be
# carried as `done` with everything unknown, which is the M59 class this branch exists to stop.
#
# So the cheap probe runs first and settles the common cases, and this one settles only the
# ambiguous one. It can still fail on a huge all-unmeasurable table, and that is the deliberate
# trade: a table reported FAILED is visible and an operator sees it, while a systemically failed
# table reported `done` is a wrong model nobody is told about.
_DISTINCT_PROBE_FULL: dict[str, str] = {
    "oracle": 'SELECT count(DISTINCT ROWNUM) FROM "{table}"',
}


if not all("{table}" in q
           for d in (_DISTINCT_PROBE, _DISTINCT_PROBE_FULL) for q in d.values()):
    # Raised, not asserted: `python -O` strips asserts, and a guard that a standard invocation
    # mode removes is not a guard. Pointed out by review of the version that used one.
    raise ValueError("every probe template must contain {table}")


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

    measured: dict[str, tuple] = {}
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
            for c in cols:
                try:
                    measured[c] = adapter.execute(
                        f'SELECT count(DISTINCT "{c}"), count("{c}") FROM "{table.name}"')[0]
                except Exception as col_exc:
                    logger.warning("column %r.%r could not be counted and is carried with unknown "
                                   "stats: %s", table.name, c, col_exc)
                    measured[c] = (None, None)
            probe = _DISTINCT_PROBE.get(getattr(adapter, "dialect", ""))
            sortable = False
            deciding: Exception | None = None
            if probe is not None and all(v == (None, None) for v in measured.values()):
                try:
                    adapter.execute(probe.format(table=table.name))
                    sortable = True
                except Exception:
                    sortable = False
                full = _DISTINCT_PROBE_FULL.get(getattr(adapter, "dialect", ""))
                if sortable and full is not None:
                    # Ambiguous: nothing measured, yet a 100-row dedup succeeded. Ask at full size
                    # before concluding the columns were at fault.
                    try:
                        adapter.execute(full.format(table=table.name))
                    except Exception as full_exc:
                        logger.warning("%r sorts at 100 rows but not at full size; treating as "
                                       "systemic rather than per-column", table.name)
                        sortable = False
                        # The exception that DECIDED this, not the one that started the fallback.
                        # A bare `raise` below would re-raise the batched failure -- so a table
                        # whose batch died on ORA-22950 (a column type) and whose full sort died on
                        # ORA-01652 (capacity) would report the column error, misattributing a
                        # systemic failure to a type problem for whoever reads it.
                        deciding = full_exc
            if not sortable and all(v == (None, None) for v in measured.values()):
                if deciding is not None:
                    raise deciding
                # The session cannot DISTINCT at all, so this is systemic -- a permission, a
                # driver fault, temp space exhausted by the sort. Carrying unknowns for every
                # column would report a table that measured NOTHING as `done`, and
                # `enrich_structural` marks a table done whenever this returns, so
                # `profile_outcome` would say `complete` over a model that measured nothing. That
                # is M59's bug class, and the codebase already paid for it once as the Pagila
                # timestamptz failure that dropped 14 of 15 tables while reporting success.
                # Re-raising restores the pre-fallback behaviour exactly.
                raise
            # The session CAN sort, so every failure here was the column's own. This is asked
            # rather than inferred from "all of them failed", which was wrong for a table of ONE
            # column: measured, a single CLOB was carried with unknown stats while a single
            # user-defined `ADDR_T` -- the identical situation -- was re-raised and the table
            # excluded, purely because the first type is on `_UNAGGREGATABLE` and the second
            # cannot be. That made the list load-bearing again, which is what the fallback exists
            # to stop.
            if all(v == (None, None) for v in measured.values()):
                # Gated, because it was not. It fired whenever the probe succeeded and claimed
                # "no column could be counted ... all N carried" on a table where one column had
                # measured perfectly well -- false on both counts, and contradicting the comment
                # directly above it, which reasons about the partial case.
                logger.warning("no column of %r could be counted, though the session can sort; "
                               "all %d carried with unknown stats", table.name, len(measured))
    else:
        # Every column is unaggregatable, so there is nothing to count them WITH; the row count
        # still is, and it is what tells a reader the table is not empty.
        row_count = adapter.execute(f'SELECT count(*) FROM "{table.name}"')[0][0]

    stats: list[ColumnStats] = []
    for name in skipped:
        # distinct_count None, not 0: nothing was measured. A zero here would read as "no values"
        # and could make the column look like an empty coded vocabulary.
        stats.append(ColumnStats(table=table.name, column=name, row_count=row_count,
                                 distinct_count=None, null_count=None))
    for c in cols:
        distinct_count, non_null = measured[c]
        s = ColumnStats(
            table=table.name,
            column=c,
            row_count=row_count,
            distinct_count=distinct_count,
            null_count=None if non_null is None else row_count - non_null,
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
