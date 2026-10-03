"""The binder's value probe on a non-DuckDB source (M126).

Found by running enrichment against a local Oracle: the probe was raw DuckDB SQL, `LIMIT` and all,
so Oracle refused it (ORA-03049) on every column it reached, and a large code system could never
bind there. The same run showed the probe ran with a
glossary that held no code systems at all, where it can match nothing; fixed only for dialect, that
would have become a DISTINCT scan of the source per column.
"""
from __future__ import annotations

import sqlite3

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.ontology.binder import bind_schemes, suggest_bindings
from mnemiq.ontology.records import Concept, ConceptScheme, Definition, OntologyRecords

CODES = ["R", "G", "B", "Y", "P"]


class _OracleLike:
    """Answers the probe as Oracle would: refuses `LIMIT`, otherwise returns the column's values."""

    dialect = "oracle"

    def __init__(self, values):
        self.values, self.queries = values, []

    def execute(self, sql):
        self.queries.append(sql)
        if "LIMIT" in sql.upper():
            raise RuntimeError("ORA-03049: SQL keyword 'LIMIT' is not syntactically valid")
        return [(v,) for v in self.values]


def _probed_snapshot(tmp_path):
    """A coded column with its harvested values removed, so the binder has to probe for them --
    what profiling leaves for a code system larger than its harvest limit."""
    path = tmp_path / "s.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE sample (sample_pk INTEGER PRIMARY KEY, colour_code TEXT);"
                      "INSERT INTO sample VALUES " + ",".join(
                          f"({i + 1},'{v}')" for i, v in enumerate(CODES)) + ";")
    con.commit()
    con.close()
    snap = enrich_structural(SQLiteAdapter(str(path)), "p")
    return snap.model_copy(update={"columns": [
        c.model_copy(update={"coded_values": []}) for c in snap.columns]})


def _scheme():
    return ConceptScheme(id="urn:colour", label="Colour Codes", concepts=[
        Concept(id=f"urn:colour:{n}", notation=n, pref_label=n) for n in CODES])


def test_the_probe_is_written_in_the_sources_dialect_and_the_column_binds(tmp_path):
    adapter = _OracleLike(CODES)
    out = bind_schemes(adapter, _probed_snapshot(tmp_path), OntologyRecords(schemes=[_scheme()]))

    probes = [q for q in adapter.queries if "colour_code" in q]
    assert probes and all("FETCH FIRST 1000 ROWS ONLY" in q for q in probes), adapter.queries
    assert next(c for c in out.columns if c.name == "colour_code").code_scheme is not None


def test_a_glossary_with_no_code_systems_probes_nothing(tmp_path):
    adapter = _OracleLike(CODES)
    records = OntologyRecords(definitions=[Definition(id="d1", term="t", domain="plant", definition="x")])
    snap = _probed_snapshot(tmp_path)

    out = bind_schemes(adapter, snap, records)
    assert suggest_bindings(adapter, out, records) == []
    assert adapter.queries == [], "no scheme to match, so no scan of the source"
    assert [d.id for d in out.definitions][-1] == "d1", "its definitions still land"
