"""A source's manifest may name the tables enrichment sees (M110).

Without it, enrichment took every table the source reported -- for Oracle, every table the schema
owner holds -- and a design partner had to copy their five tables out of production to evaluate on
a different engine. These pin the manifest field, the scoped view of the catalogue, and the path a
deployment actually takes: `adapter_for` from a manifest, then structural enrichment.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from mnemiq.adapters.scoped import TableScopedAdapter, TablesNotFound
from mnemiq.config import Settings


def _manifest(tmp_path, **entry) -> Settings:
    source = tmp_path / "src.duckdb"
    if not source.exists():
        con = duckdb.connect(str(source))
        con.execute("CREATE TABLE a_keep (id INTEGER, label TEXT)")
        con.execute("CREATE TABLE b_keep (id INTEGER, a_id INTEGER)")
        con.execute("CREATE TABLE c_drop (id INTEGER, secret TEXT)")
        con.execute("INSERT INTO a_keep VALUES (1, 'x'), (2, 'y')")
        con.execute("INSERT INTO b_keep VALUES (1, 1)")
        con.execute("INSERT INTO c_drop VALUES (1, 'hidden')")
        con.close()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "duckdb", "target": str(source),
                                     "catalog": "src", "schema": "main", **entry}]))
    return Settings(sources_path=str(manifest), store_path=str(tmp_path / "store.duckdb"),
                    llm_base_url=None, llm_api_key=None, llm_model=None, acme_data_dir=None,
                    pg_dsn=None)


# --- the manifest field -------------------------------------------------------------------------

def test_tables_is_optional_and_read_as_given(tmp_path):
    assert _manifest(tmp_path).source_specs()[0].tables == ()
    assert _manifest(tmp_path, tables=["A_KEEP", " b_* "]).source_specs()[0].tables == (
        "A_KEEP", "b_*")


@pytest.mark.parametrize("bad", ["a_keep", [], ["a_keep", ""], ["a_keep", 3]])
def test_a_tables_value_that_is_not_a_list_of_names_is_refused(tmp_path, bad):
    """A bare string would iterate as one-letter patterns; an empty list would read as none."""
    with pytest.raises(ValueError, match='"tables" must be a non-empty list'):
        _manifest(tmp_path, tables=bad).source_specs()


# --- the scoped view ----------------------------------------------------------------------------

class _Source:
    dialect = "duckdb"

    def list_columns(self):
        return [("A_KEEP", "id", "int"), ("B_KEEP", "id", "int"), ("C_DROP", "secret", "text")]

    def introspect(self):
        return ["A_KEEP", "B_KEEP", "C_DROP"]

    def foreign_keys(self):
        return [("B_KEEP", "a_id", "A_KEEP", "id", "fk1"), ("B_KEEP", "c_id", "C_DROP", "id", "fk2")]

    def view_definitions(self):
        return [("A_VIEW", "select 1", "duckdb"), ("C_VIEW", "select 2", "duckdb")]

    def execute(self, sql):
        return f"ran {sql}"


def test_the_catalogue_shows_only_listed_tables_matched_without_case():
    scoped = TableScopedAdapter(_Source(), ("a_keep", "b_*"), "s")
    assert {t for t, _c, _d in scoped.list_columns()} == {"A_KEEP", "B_KEEP"}
    assert scoped.introspect() == ["A_KEEP", "B_KEEP"]


def test_a_foreign_key_needs_both_ends_in_scope():
    scoped = TableScopedAdapter(_Source(), ("a_keep", "b_keep"), "s")
    assert scoped.foreign_keys() == [("B_KEEP", "a_id", "A_KEEP", "id", "fk1")]


def test_views_are_scoped_by_name():
    scoped = TableScopedAdapter(_Source(), ("a_*",), "s")
    assert [v[0] for v in scoped.view_definitions()] == ["A_VIEW"]


def test_everything_else_is_the_sources():
    scoped = TableScopedAdapter(_Source(), ("a_keep",), "s")
    assert scoped.dialect == "duckdb" and scoped.execute("x") == "ran x"
    assert not hasattr(scoped, "user_functions"), "an optional method the source lacks stays absent"


@pytest.mark.parametrize("listed, missing", [(("a_keep", "a_kep"), "a_kep"),
                                             (("a_keep", "z_*"), "z_*")])
def test_a_name_or_pattern_matching_nothing_fails_and_says_which(listed, missing):
    with pytest.raises(TablesNotFound, match=rf"lists table\(s\) it does not report: {missing}"):
        TableScopedAdapter(_Source(), listed, "s").list_columns()


# --- the path a deployment takes ----------------------------------------------------------------

def test_enrichment_from_the_manifest_sees_only_the_listed_tables(tmp_path):
    from mnemiq.adapters.resolve import adapter_for
    from mnemiq.enrichment.pipeline import enrich_structural

    settings = _manifest(tmp_path, tables=["A_KEEP", "b_*"])
    spec = settings.source_specs()[0]
    snapshot = enrich_structural(adapter_for(spec, settings), spec.id)

    assert {b.object_id for b in snapshot.source_bindings} == {"a_keep", "b_keep"}
    assert {c.object_id for c in snapshot.columns} == {"a_keep", "b_keep"}
    assert not any("secret" in c.name for c in snapshot.columns)
    assert not any("c_drop" in j.id for j in snapshot.jobs), "the unlisted table was not profiled"


def test_without_a_list_every_table_is_enriched_as_before(tmp_path):
    from mnemiq.adapters.resolve import adapter_for
    from mnemiq.enrichment.pipeline import enrich_structural

    settings = _manifest(tmp_path)
    spec = settings.source_specs()[0]
    snapshot = enrich_structural(adapter_for(spec, settings), spec.id)
    assert {b.object_id for b in snapshot.source_bindings} == {"a_keep", "b_keep", "c_drop"}


def test_enrich_with_a_misspelt_table_stops_with_one_line(tmp_path, capsys):
    from mnemiq.cli import _cmd_enrich

    settings = _manifest(tmp_path, tables=["a_keep", "b_kep"])
    assert _cmd_enrich(settings) == 1
    err = capsys.readouterr().err
    assert "lists table(s) it does not report: b_kep" in err
    assert "Traceback" not in err
