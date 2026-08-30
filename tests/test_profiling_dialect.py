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
