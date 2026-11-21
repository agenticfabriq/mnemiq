"""Before an exact profile of an Oracle source, how large its tables are by Oracle's own statistics.

Exact profiling reads each table in full many times: one count over every column, a GROUP BY per code-like column, up
to eight partition counts, the grounding and join checks, the value index. On a production database's large tables
one such read was measured at hours, so a whole profile is days of load on the database. When the manifest has not
said how to profile an Oracle source, enrich therefore reads each table's size from the dictionary first, which reads
no table data, and stops before any table is read if one is large. `"profile": "statistics"` on the source profiles
it from those statistics instead (`enrichment.statistics`), and `"profile": "exact"` is how an operator says to read
the tables in full anyway. Under statistics mode the same line applies to the tables `"exact_tables"` names, the ones
it still reads in full.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# A table is large above this much estimated data, NUM_ROWS x AVG_ROW_LEN from ALL_TAB_STATISTICS.
LARGE_TABLE_BYTES = 1 << 30


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.1f}"


def exact_profile_refusal(spec, adapter) -> tuple[str | None, str | None]:
    """(refusal, warning) for exactly profiling `spec` through `adapter`.

    Only an Oracle source whose manifest names no "profile" is checked, and only then is anything read: the table list
    and each table's size from the dictionary. The refusal names every table above `LARGE_TABLE_BYTES` and how to go
    on. Two kinds of table can be large without the statistics saying so, and the warning names both while enrich goes
    on as before: one with no table-level statistics, whose size is unknown, and one whose statistics Oracle marks
    stale, which may have grown since they were gathered."""
    exact_tables = spec.profile == "statistics" and bool(spec.exact_tables)
    if spec.kind != "oracle" or (spec.profile and not exact_tables):
        return None, None
    tables = list(adapter.introspect())
    if exact_tables:
        from mnemiq.enrichment.pipeline import exact_table_names

        # Resolved as enrich resolves them, so the check and the profile agree on which tables are read in full.
        names = exact_table_names(tables, spec.exact_tables, spec.id)
        tables = [t for t in tables if t in names]
    large: list[tuple[str, int]] = []
    unknown: list[str] = []
    stale: list[str] = []
    # One batched read where the adapter has it; one per table otherwise, and also when the batch fails (one statement
    # over 500 tables can outlast the probe timeout where each table's own read would not).
    sizes = None
    if callable(getattr(adapter, "table_sizes", None)):
        try:
            sizes = adapter.table_sizes(tables)
        except Exception as exc:  # noqa: BLE001 -- the per-table reads remain, and the check still runs
            logger.warning("the batched size read failed (%s); sizes are read per table", exc)
    for table in tables:
        size = sizes[table] if sizes is not None else adapter.table_size(table)
        if size.rows is None or size.avg_row_len is None:
            unknown.append(table)
            continue
        estimate = size.rows * size.avg_row_len
        if estimate > LARGE_TABLE_BYTES:
            large.append((table, estimate))
        elif size.stale is True:
            stale.append(f"{table} (~{_gib(estimate)} GiB as of {size.analyzed})")
    parts = []
    if unknown:
        parts.append(f"{len(unknown)} table(s) of source {spec.id!r} have no table-level optimizer statistics, so "
                     f"their size is unknown: {', '.join(unknown)}")
    if stale:
        parts.append(f"{len(stale)} table(s) of source {spec.id!r} have statistics Oracle marks stale, so they may "
                     f"have grown since: {', '.join(stale)}")
    warning = None
    if parts:
        warning = ("; ".join(parts) + ". Exact profiling reads each of them in full; to check their size first, "
                   "gather their statistics (DBMS_STATS.GATHER_TABLE_STATS) and run enrich again, or leave them out "
                   'of the source\'s "tables".')
    if not large:
        return None, warning
    listed = ", ".join(f"{table} (~{_gib(size)} GiB)" for table, size in large)
    if exact_tables:
        return (f"enrich stopped before reading any table: {len(large)} table(s) that \"exact_tables\" of source "
                f"{spec.id!r} names hold about {_gib(sum(size for _, size in large))} GiB by Oracle's statistics, "
                f"and an exact profile reads each of them in full many times: {listed}. To profile them from the "
                "dictionary instead, take them out of \"exact_tables\"."), warning
    refusal = (f"enrich stopped before reading any table: {len(large)} table(s) of source {spec.id!r} hold about "
               f"{_gib(sum(size for _, size in large))} GiB by Oracle's statistics, and exact profiling reads each of "
               f"them in full many times: {listed}. On this source in the manifest, set \"profile\": \"statistics\" to "
               "read Oracle's optimizer statistics and no table data (\"exact_tables\" names tables to count exactly), "
               "or \"profile\": \"exact\" to profile them in full anyway; or list the tables to enrich in its "
               "\"tables\".")
    return refusal, warning
