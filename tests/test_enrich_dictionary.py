import json
import logging
import sqlite3

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.enrichment.pipeline import enrich_structural


def _origin_snapshot(tmp_path):
    path = tmp_path / "p.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE sampledata (sample_pk INTEGER PRIMARY KEY, origin TEXT, notes TEXT);
        INSERT INTO sampledata VALUES (1,'3','a'),(2,'2','b'),(3,'3','c'),(4,'1','d');
    """)
    con.commit()
    con.close()
    return enrich_structural(SQLiteAdapter(str(path)), "p")


def test_load_dictionary_parses_and_rejects_malformed(tmp_path):
    from mnemiq.enrichment.dictionary import load_dictionary

    good = tmp_path / "d.json"
    good.write_text(json.dumps({"columns": {
        "sampledata.origin": {"description": "origin class", "codes": {"1": "US", "3": "unknown"}}}}))
    d = load_dictionary(str(good))
    assert d.columns["sampledata.origin"].codes["3"] == "unknown"

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    try:
        load_dictionary(str(bad))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_apply_dictionary_grounds_observed_codes_and_sets_description(tmp_path, caplog):
    from mnemiq.enrichment.dictionary import ColumnEntry, DataDictionary
    from mnemiq.enrichment.grounding import apply_dictionary

    snap = _origin_snapshot(tmp_path)
    d = DataDictionary(columns={
        "sampledata.origin": ColumnEntry(description="origin class",
                                         codes={"1": "US", "3": "unknown", "9": "not-observed"}),
        "sampledata.ghost": ColumnEntry(codes={"x": "y"}),
    })
    with caplog.at_level(logging.WARNING):
        out = apply_dictionary(snap, d)

    origin = next(c for c in out.columns if c.id == "sampledata.origin")
    meanings = {cv.code: (cv.meaning, cv.source) for cv in origin.coded_values}
    assert meanings["1"] == ("US", "dictionary")
    assert meanings["3"] == ("unknown", "dictionary")
    assert "9" not in meanings                      # code not observed -> ignored
    assert origin.description == "origin class"
    assert any("ghost" in r.message for r in caplog.records)  # unknown column warned
