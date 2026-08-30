"""Profiling emitted one dialect's SQL for every source, and real Oracle data is what showed it.

Against a 51-table Oracle schema, 47 tables failed to profile and the model came back with 4
tables and 10 columns. Three defects, each hidden behind the one before it -- fixing the first
revealed the second, and fixing that revealed the third. None was reachable from the container
schema used until then, which is small and typed entirely in NUMBER and VARCHAR2.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mnemiq.catalog import ColumnInfo, TableInfo
from mnemiq.enrichment.profiling import _base_type, _unaggregatable, profile_table


@dataclass
class _Recorder:
    """An adapter that records SQL and answers with plausible counts."""

    dialect: str = "oracle"

    def __post_init__(self):
        self.sql: list[str] = []

    def execute(self, sql):
        self.sql.append(sql)
        if sql.lstrip().upper().startswith("SELECT COUNT(*) FROM"):
            return [(7,)]
        if "count(*)" in sql and "GROUP BY" not in sql:
            n = sql.count("count(DISTINCT")
            return [tuple([7] + [3, 7] * n)]
        return [("a", 4), ("b", 3)]


def _table(*cols: tuple[str, str]) -> TableInfo:
    return TableInfo(name="T", columns=[ColumnInfo(name=n, data_type=t) for n, t in cols])


def test_the_top_k_query_uses_no_limit_keyword_on_oracle():
    """ORA-03049: `LIMIT` is Postgres/DuckDB syntax. One clause cost 20 of 51 tables."""
    a = _Recorder(dialect="oracle")
    profile_table(a, _table(("STATUS", "VARCHAR2")))
    topk = [s for s in a.sql if "GROUP BY" in s]
    assert topk, "no top-k query was issued"
    assert "LIMIT" not in topk[0].upper()
    assert "FETCH FIRST" in topk[0].upper()


def test_the_top_k_query_never_groups_by_ordinal():
    """ORA-03162: Oracle rejects positional GROUP BY by default, and sqlglot does not rewrite an
    ordinal into a name -- so transpiling alone left 26 tables still failing. Naming the column
    needs no dialect knowledge, so it is asserted for EVERY dialect."""
    for dialect in ("oracle", "postgres", "duckdb", "sqlite"):
        a = _Recorder(dialect=dialect)
        profile_table(a, _table(("STATUS", "VARCHAR2")))
        topk = [s for s in a.sql if "GROUP BY" in s][0]
        assert "GROUP BY 1" not in topk, f"{dialect}: positional group-by"
        assert '"STATUS"' in topk.split("GROUP BY", 1)[1]


def test_an_unaggregatable_column_does_not_take_its_table_down_with_it():
    """The amplifier. `profile_table` issues ONE counts query for the whole table, so a single
    CLOB made the statement fail and excluded every column of that table -- 27 of 51.

    Oracle refuses `count(col)` as well as `count(DISTINCT col)` on a LOB, measured; it is not
    only the DISTINCT that is unsupported.
    """
    a = _Recorder(dialect="oracle")
    stats = profile_table(a, _table(("ID", "VARCHAR2"), ("BODY", "CLOB"), ("EMB", "VECTOR(1024)")))
    counts = [s for s in a.sql if "count(DISTINCT" in s]
    assert counts and '"BODY"' not in counts[0] and '"EMB"' not in counts[0]
    assert {s.column for s in stats} == {"ID", "BODY", "EMB"}, "every column is still described"


def test_a_skipped_column_reports_unknown_not_zero():
    """`distinct_count=0` would read as "no values" and could make the column look like an empty
    coded vocabulary. Nothing was measured, so nothing is claimed."""
    a = _Recorder(dialect="oracle")
    body = next(s for s in profile_table(a, _table(("ID", "VARCHAR2"), ("BODY", "CLOB")))
                if s.column == "BODY")
    assert body.distinct_count is None and body.null_count is None
    assert body.row_count == 7, "the row count is still known and still worth reporting"


def test_a_table_of_only_unaggregatable_columns_still_yields_its_row_count():
    a = _Recorder(dialect="oracle")
    stats = profile_table(a, _table(("BODY", "CLOB")))
    assert [s.row_count for s in stats] == [7]
    assert not any("count(DISTINCT" in s for s in a.sql), "nothing countable to count"


def test_the_exclusion_list_is_per_dialect():
    """Postgres aggregates `text` and `bytea` without complaint. A shared list would silently stop
    profiling columns on sources that profile them fine."""
    assert "CLOB" in _unaggregatable(_Recorder(dialect="oracle"))
    assert _unaggregatable(_Recorder(dialect="postgres")) == frozenset()
    assert _unaggregatable(_Recorder(dialect="duckdb")) == frozenset()


@pytest.mark.parametrize("declared, base", [
    ("VECTOR(1024, FLOAT32)", "VECTOR"),
    ("CLOB", "CLOB"),
    ("TIMESTAMP(6) WITH TIME ZONE", "TIMESTAMP"),
    ("  clob  ", "CLOB"),
])
def test_the_base_type_is_matched_not_the_declaration(declared, base):
    """`all_tab_columns` reports `VECTOR` for `VECTOR(1024, FLOAT32)` in some versions and the
    parameterised form in others; matching the declaration verbatim would miss one of them."""
    assert _base_type(declared) == base


def test_profile_column_skips_an_unaggregatable_type_when_told_one():
    """The sibling entry point. `infer_relationships` calls this for every `*_id` column, so on
    Oracle a CLOB with such a name raised ORA-22849 out of `build_relationships` and cost EVERY
    relationship in the source -- one column, the whole job."""
    from mnemiq.enrichment.profiling import profile_column

    a = _Recorder(dialect="oracle")
    s = profile_column(a, "T", "DOC_ID", k=0, data_type="CLOB")
    assert s.distinct_count is None and s.row_count == 7
    assert not any("count(DISTINCT" in q for q in a.sql), "it must not query what it cannot count"


def test_profile_column_without_a_declared_type_behaves_exactly_as_before():
    """Callers that cannot supply a type must not silently change behaviour."""
    from mnemiq.enrichment.profiling import profile_column

    a = _Recorder(dialect="oracle")
    profile_column(a, "T", "C", k=0)
    assert any("count(DISTINCT" in q for q in a.sql)


def test_relationship_inference_passes_the_declared_type_to_the_profiler(monkeypatch):
    """`infer_relationships` must hand the column's type down, or the skip cannot happen there.

    Asserted on the CALL rather than end to end, and that is a correction: the first version built
    a one-table catalog and called `infer_relationships`, which never profiles anything because the
    parent is the table the column NAMES and no such table existed. It passed with the fix reverted
    -- proving only that the function returns. This asserts the thing the fix actually changes.
    """
    from mnemiq.enrichment import joins

    seen: list = []

    def _spy(adapter, table, col, k=10, data_type=None):
        seen.append((table, col, data_type))
        from mnemiq.enrichment.profiling import ColumnStats
        return ColumnStats(table=table, column=col, row_count=0, distinct_count=None,
                           null_count=None)

    monkeypatch.setattr(joins, "profile_column", _spy)
    catalog = [
        TableInfo(name="DOC", columns=[ColumnInfo(name="DOC_ID", data_type="CLOB")]),
        TableInfo(name="ITEM", columns=[ColumnInfo(name="DOC_ID", data_type="CLOB")]),
    ]
    joins.infer_relationships(_Recorder(dialect="oracle"), catalog)
    assert seen, "inference never profiled anything -- the catalog does not reach the path"
    assert all(t == "CLOB" for _, _, t in seen), f"declared type not passed: {seen}"


@pytest.mark.parametrize("column", ["PARTY_ID", "party_id", "Party_Id", "CLAIM_IDENTIFIER"])
def test_a_key_column_is_recognised_whatever_case_the_source_folds_to(column):
    """Oracle folds unquoted identifiers to UPPER, so a case-sensitive check returned False for
    every column of an Oracle source. Measured consequence on a real schema: 11 ID-named columns
    had their values harvested as candidate coded vocabularies -- UUIDs and owner identifiers --
    where a lowercase source would have excluded all of them. `is_sensitive_name` was already
    case-insensitive; the asymmetry is what made this survivable for so long."""
    from mnemiq.catalog import is_key_like

    assert is_key_like(column)


def test_an_id_named_column_is_not_harvested_as_a_vocabulary_on_an_uppercase_source():
    """The consequence, not just the predicate: `profile_table` must not collect a value set for
    a key-like column even when the source names it in upper case."""
    a = _Recorder(dialect="oracle")
    stats = profile_table(a, _table(("PARTY_ID", "VARCHAR2")))
    assert not any("GROUP BY" in s for s in a.sql), "no value set may be harvested for a key"
    assert stats[0].top_k == []


@pytest.mark.parametrize("column, stem", [
    ("party_id", "party"), ("PARTY_ID", "PARTY"), ("CLAIM_IDENTIFIER", "CLAIM"),
    ("description", "description"),
])
def test_the_stem_matches_case_insensitively_and_keeps_the_sources_case(column, stem):
    """The stem is looked up against table names from the SAME source, so it must keep that
    source's folding while matching the suffix regardless of it. Case-sensitive, `_stem("DOC_ID")`
    returned `"DOC_ID"`, no table matched, and inferred relationships were impossible on Oracle --
    silently, because declared foreign keys still arrive from the catalog and the model looks
    populated."""
    from mnemiq.enrichment.joins import _stem

    assert _stem(column) == stem


class _BatchFails(_Recorder):
    """A source that refuses one column, whatever the batched query asks for.

    Modelled on a real Oracle user-defined object type: `count(DISTINCT addr)` raises ORA-22950 --
    a DIFFERENT code from the ORA-22849 the LOB types give -- while `count(addr)` succeeds.
    """

    def execute(self, sql):
        self.sql.append(sql)
        if '"ADDR"' in sql and "count(DISTINCT" in sql:
            raise RuntimeError("ORA-22950: cannot order objects without MAP or ORDER method")
        if sql.lstrip().upper().startswith("SELECT COUNT(*) FROM"):
            return [(3,)]
        if "count(*)" in sql and "GROUP BY" not in sql:
            return [tuple([3] + [2, 3] * sql.count("count(DISTINCT"))]
        if "count(DISTINCT" in sql:
            return [(2, 3)]
        return [("open", 2), ("shut", 1)]


def test_one_unlistable_column_does_not_erase_the_columns_beside_it():
    """The denylist can never be complete, so it cannot be the guarantee.

    A user-defined object type reports its OWN name as its data type -- `ADDR_T` -- so no list of
    type names can enumerate it. Measured on a live instance: the batched counts query fails and,
    before this fallback, took every column of the table with it, including a plain NUMBER.
    """
    a = _BatchFails(dialect="oracle")
    stats = {s.column: s for s in profile_table(
        a, _table(("ID", "NUMBER"), ("STATUS", "VARCHAR2"), ("ADDR", "ADDR_T")))}

    assert stats["ID"].distinct_count == 2, "a countable column must still be measured"
    assert stats["STATUS"].distinct_count == 2
    assert stats["ADDR"].distinct_count is None, "the one that cannot count reports unknown"
    assert stats["ADDR"].null_count is None
    assert all(s.row_count == 3 for s in stats.values())


def test_the_fallback_only_runs_when_the_batch_fails():
    """It costs one query per column, so it must not become the normal path."""
    a = _Recorder(dialect="oracle")
    profile_table(a, _table(("ID", "NUMBER"), ("STATUS", "VARCHAR2")))
    batched = [q for q in a.sql if q.count("count(DISTINCT") > 1]
    per_column = [q for q in a.sql if q.count("count(DISTINCT") == 1 and "GROUP BY" not in q]
    assert len(batched) == 1 and per_column == [], "the happy path is still one query"


def test_an_unmeasured_column_is_never_treated_as_a_vocabulary():
    """`0 < distinct_count` raises on None, and a column with unknown counts is not a candidate
    coded vocabulary in any case -- nothing was observed to harvest."""
    a = _BatchFails(dialect="oracle")
    addr = {s.column: s for s in profile_table(
        a, _table(("ID", "NUMBER"), ("ADDR", "ADDR_T")))}["ADDR"]
    assert addr.top_k == []


def test_a_broken_column_is_logged_and_a_known_unsupported_one_is_not(caplog):
    """Both produce `distinct_count=None`, so without a log they are the same signal.

    A type named in `_UNAGGREGATABLE` is skipped before any query is issued -- nothing went wrong,
    and a warning there would fire on every LOB column of every enrich. A column that reached the
    database and FAILED is unplanned by definition: a permission, a driver fault, or a bug here.
    A line always means the second. That distinction is the whole point of the fix this sits in,
    and I had reproduced the collapse inside it.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        profile_table(_Recorder(dialect="oracle"), _table(("ID", "NUMBER"), ("BODY", "CLOB")))
    assert caplog.text == "", "a known-unsupported type is skipped, not failed"

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        profile_table(_BatchFails(dialect="oracle"), _table(("ID", "NUMBER"), ("ADDR", "ADDR_T")))
    assert "ADDR" in caplog.text and "ORA-22950" in caplog.text
    assert "batched profile" in caplog.text, "and the fallback itself is announced"


def test_a_failure_that_takes_every_column_is_still_a_failed_table():
    """The fallback must not convert a systemic failure into a successful-looking empty table.

    `enrich_structural` marks a table `done` whenever `profile_table` returns, so swallowing every
    column's failure would make `profile_outcome` report `complete` over a model that measured
    nothing -- M59's bug class, one call frame below where it was closed. A cause that takes out
    every column is systemic, not typed: a permission, a driver fault, temp space exhausted by the
    sort a DISTINCT needs. Before the fallback existed such an exception propagated and the table
    was marked failed; it still must.
    """
    class _AllDistinctFail(_Recorder):
        def execute(self, sql):
            self.sql.append(sql)
            if "count(DISTINCT" in sql:
                raise RuntimeError("ORA-01652: unable to extend temp segment")
            if sql.lstrip().upper().startswith("SELECT COUNT(*) FROM"):
                return [(3,)]
            return [("x", 1)]

    with pytest.raises(RuntimeError, match="ORA-01652"):
        profile_table(_AllDistinctFail(dialect="oracle"), _table(("A", "NUMBER"), ("B", "NUMBER")))


def test_a_partial_failure_is_still_carried_rather_than_raised():
    """The distinction the re-raise turns on: SOME columns measured means the cause was the column,
    not the session, and the table is worth keeping."""
    stats = {s.column: s for s in profile_table(
        _BatchFails(dialect="oracle"), _table(("ID", "NUMBER"), ("ADDR", "ADDR_T")))}
    assert stats["ID"].distinct_count == 2 and stats["ADDR"].distinct_count is None


@pytest.mark.parametrize("data_type", ["CLOB", "ADDR_T"])
def test_a_single_uncountable_column_is_treated_the_same_whether_or_not_it_is_listed(data_type):
    """The denylist must not decide whether a table survives.

    Measured before this: a single CLOB was carried with unknown stats while a single
    user-defined `ADDR_T` -- the identical situation -- was re-raised and the table excluded, for
    no reason but that one type is nameable and the other is not. "All columns failed" is a bad
    proxy for "systemic" when the table has ONE column, because there the two are the same fact.
    """
    class _OneBadColumn(_Recorder):
        def execute(self, sql):
            self.sql.append(sql)
            if "count(DISTINCT" in sql and '"ADDR"' in sql:
                raise RuntimeError("ORA-22950: cannot order objects")
            if "count(DISTINCT ROWNUM)" in sql:
                return [(3,)]
            if sql.lstrip().upper().startswith("SELECT COUNT(*) FROM"):
                return [(3,)]
            return [(3, 2, 3)]

    stats = profile_table(_OneBadColumn(dialect="oracle"), _table(("ADDR", data_type)))
    assert [(s.column, s.distinct_count, s.row_count) for s in stats] == [("ADDR", None, 3)]


def test_the_probe_is_what_separates_systemic_from_column_specific():
    """Asked, not inferred. The probe uses ROWNUM on Oracle because it varies per row and so
    exercises the sort a temp-space failure would break; `count(DISTINCT 1)` is a constant the
    optimiser may fold, which would answer yes on a session that cannot actually sort."""
    class _CannotSort(_Recorder):
        def execute(self, sql):
            self.sql.append(sql)
            if "count(DISTINCT" in sql:
                raise RuntimeError("ORA-01652: unable to extend temp segment")
            return [(3,)]

    a = _CannotSort(dialect="oracle")
    with pytest.raises(RuntimeError, match="ORA-01652"):
        profile_table(a, _table(("ADDR", "ADDR_T")))
    assert any("count(DISTINCT ROWNUM)" in q for q in a.sql), "the probe must actually be asked"


def test_a_dialect_without_a_probe_keeps_the_conservative_answer():
    """No probe means no refinement, not a weak probe.

    The first version defaulted to `count(DISTINCT 1)`, and a constant is foldable -- so on DuckDB,
    this project's primary adapter, a session that could not sort at all still answered "yes" and
    the table came back `done` with every column unknown. That is the systemic-failure-looks-like-
    success bug reintroduced for every dialect except Oracle, by the very commit meant to fix it.
    """
    class _CannotSort(_Recorder):
        def execute(self, sql):
            self.sql.append(sql)
            if "count(DISTINCT" in sql:
                raise RuntimeError("out of temporary space")
            return [(3,)]

    for dialect in ("duckdb", "postgres", "sqlite"):
        a = _CannotSort(dialect=dialect)
        with pytest.raises(RuntimeError, match="temporary space"):
            profile_table(a, _table(("A", "INTEGER"), ("B", "INTEGER")))
        assert not any("count(DISTINCT 1)" in q for q in a.sql), (
            f"{dialect}: a foldable constant must not be used as the probe"
        )


def test_the_all_unknown_warning_does_not_fire_on_a_partial_failure(caplog):
    """It claimed "no column could be counted ... all N carried" on a table where one column had
    measured perfectly well -- false on both counts, and contradicting the comment above it."""
    import logging

    with caplog.at_level(logging.WARNING):
        stats = {s.column: s for s in profile_table(
            _BatchFails(dialect="oracle"), _table(("ID", "NUMBER"), ("ADDR", "ADDR_T")))}
    assert stats["ID"].distinct_count == 2, "precondition: this is a PARTIAL failure"
    assert "no column of" not in caplog.text
    assert "ADDR" in caplog.text, "the column that did fail is still named"


def test_the_probe_is_bounded_so_it_cannot_fail_the_table_it_is_diagnosing():
    """A diagnostic must not be more expensive than the thing it diagnoses.

    The first version ran `count(DISTINCT ROWNUM)` over the WHOLE table -- a full access plus a
    dedup of every row, which is exactly the operation most likely to fail under the resource
    pressure the probe exists to detect. On a large table it could turn a benign per-column failure
    into an excluded table by failing itself. Measured on 200k rows: 0.0185s unbounded, 0.0011s
    bounded, and the bounded plan still carries a HASH GROUP BY, so it still does the work.
    """
    class _OneBad(_Recorder):
        def execute(self, sql):
            self.sql.append(sql)
            if "count(DISTINCT" in sql and '"ADDR"' in sql:
                raise RuntimeError("ORA-22950: cannot order objects")
            if "ROWNUM" in sql:
                return [(3,)]
            if sql.lstrip().upper().startswith("SELECT COUNT(*) FROM"):
                return [(3,)]
            return [(3, 2, 3)]

    a = _OneBad(dialect="oracle")
    profile_table(a, _table(("ADDR", "ADDR_T")))
    probe = next(q for q in a.sql if "ROWNUM" in q)
    assert "FETCH FIRST" in probe, "the probe must be bounded, not a full-table dedup"
    assert "HASH" not in probe  # sanity: this is the SQL, not a plan


def test_a_probe_that_passes_small_and_fails_large_is_treated_as_systemic():
    """Bounding the probe fixed one bug and opened another, and this is the seam between them.

    A bounded probe's temp footprint is far below the batched query's, so a genuine ORA-01652
    could pass on 100 rows while the real full-table sort cannot -- and the table would be carried
    as `done` with every column unknown, which is the M59 class. The cheap probe settles the common
    cases; the full-size one is asked ONLY when nothing measured and the cheap one said yes.
    """
    class _SmallOkLargeFails(_Recorder):
        def execute(self, sql):
            self.sql.append(sql)
            if "FETCH FIRST 100" in sql:
                return [(100,)]                       # the bounded probe succeeds
            if "count(DISTINCT ROWNUM)" in sql:
                raise RuntimeError("ORA-01652: unable to extend temp segment")   # capacity
            if "count(DISTINCT" in sql:
                raise RuntimeError("ORA-22950: cannot order objects")            # column type
            return [(3,)]

    a = _SmallOkLargeFails(dialect="oracle")
    # DISTINCT messages on purpose: the batched failure and the deciding failure must be
    # distinguishable, or the test cannot tell which exception propagated. An earlier version used
    # the same string at both sites and so could not have caught a bare `raise` re-raising the
    # wrong one -- which is exactly what it was doing.
    with pytest.raises(RuntimeError, match="ORA-01652"):
        profile_table(a, _table(("A", "NUMBER"), ("B", "NUMBER")))
    assert any("FETCH FIRST 100" in q for q in a.sql), "the cheap probe runs first"
    assert any(q.endswith('FROM "T"') and "ROWNUM" in q for q in a.sql), "and escalates"


def test_the_full_probe_is_not_asked_when_something_measured():
    """It is the expensive one, so it runs only where the answer is genuinely ambiguous."""
    a = _BatchFails(dialect="oracle")
    profile_table(a, _table(("ID", "NUMBER"), ("ADDR", "ADDR_T")))
    assert not any(q.endswith('FROM "T"') and "ROWNUM" in q for q in a.sql)


def test_a_malformed_probe_template_is_refused_under_python_O():
    """A bare `assert` is stripped by `python -O`, so the guard has to raise."""
    import subprocess
    import sys

    code = (
        "import mnemiq.enrichment.profiling as p, importlib, sys\n"
        "src = open(p.__file__).read().replace(\n"
        "    '_DISTINCT_PROBE: dict[str, str] = {',\n"
        "    '_DISTINCT_PROBE: dict[str, str] = {\"bad\": \"SELECT 1\",')\n"
        "try:\n"
        "    exec(compile(src, 'p.py', 'exec'), {'__name__': 'x'})\n"
        "    print('ACCEPTED')\n"
        "except ValueError:\n"
        "    print('REFUSED')\n"
    )
    out = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert "REFUSED" in out.stdout, f"guard stripped under -O: {out.stdout}{out.stderr}"
