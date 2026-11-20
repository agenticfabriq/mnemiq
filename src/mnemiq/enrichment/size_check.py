"""Before an exact profile of an Oracle source, how large its tables are by Oracle's own statistics.

Exact profiling reads each table in full many times: one count over every column, a GROUP BY per code-like column, up
to eight partition counts, the grounding and join checks, the value index. On a production database's large tables
one such read was measured at hours, so a whole profile is days of load on the database. When the manifest has not
said how to profile an Oracle source, enrich therefore reads each table's size from the dictionary first, which reads
no table data, and stops before any table is read if one is large. `"profile": "exact"` on the source is how an
operator says to read them anyway.
"""
from __future__ import annotations

# A table is large above this much estimated data, NUM_ROWS x AVG_ROW_LEN from ALL_TAB_STATISTICS.
LARGE_TABLE_BYTES = 1 << 30


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.1f}"


def statistics_refusal(spec) -> str | None:
    """Why `spec` cannot be enriched as it stands, when its manifest asks for statistics profiling, which this version
    does not have. Taking it as "skip the size check" would profile every table in full, the harm the check exists
    for."""
    if spec.profile != "statistics":
        return None
    return (f'source {spec.id!r}: "profile": "statistics" is not available in this version of mnemiq; leave '
            '"profile" out, or set it to "exact"')


def exact_profile_refusal(spec, adapter) -> tuple[str | None, str | None]:
    """(refusal, warning) for exactly profiling `spec` through `adapter`.

    Only an Oracle source whose manifest names no "profile" is checked, and only then is anything read: the table list
    and each table's size from the dictionary. The refusal names every table above `LARGE_TABLE_BYTES` and how to go
    on. A table with no statistics has an unknown size: it is named in the warning, and enrich goes on as before."""
    if spec.kind != "oracle" or spec.profile:
        return None, None
    large: list[tuple[str, int]] = []
    unknown: list[str] = []
    for table in adapter.introspect():
        size = adapter.table_size(table)
        if size.rows is None or size.avg_row_len is None:
            unknown.append(table)
        elif size.rows * size.avg_row_len > LARGE_TABLE_BYTES:
            large.append((table, size.rows * size.avg_row_len))
    warning = None
    if unknown:
        warning = (f"{len(unknown)} table(s) of source {spec.id!r} have no optimizer statistics, so their size is "
                   f"unknown and exact profiling reads them in full: {', '.join(unknown)}")
    if not large:
        return None, warning
    listed = ", ".join(f"{table} (~{_gib(size)} GiB)" for table, size in large)
    refusal = (f"enrich stopped before reading any table: {len(large)} table(s) of source {spec.id!r} hold about "
               f"{_gib(sum(size for _, size in large))} GiB by Oracle's statistics, and exact profiling reads each of "
               f"them in full many times: {listed}. To profile them in full anyway, set \"profile\": \"exact\" on this "
               f"source in the manifest; to leave them out, list the tables to enrich in its \"tables\".")
    return refusal, warning
