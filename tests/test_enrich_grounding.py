import sqlite3

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.enrichment.pipeline import enrich_structural


def _sqlite(tmp_path, name, script):
    path = tmp_path / f"{name}.sqlite"
    con = sqlite3.connect(path)
    con.executescript(script)
    con.commit()
    con.close()
    return SQLiteAdapter(str(path))


def test_correlated_grounds_from_a_functional_label_sibling(tmp_path):
    from mnemiq.enrichment.grounding import ground_from_correlated

    adapter = _sqlite(tmp_path, "t", """
        CREATE TABLE t (id INTEGER PRIMARY KEY, status TEXT, status_name TEXT, memo TEXT);
        INSERT INTO t VALUES (1,'A','Active','x'),(2,'A','Active','y'),(3,'B','Blocked','z');
    """)
    snap = enrich_structural(adapter, "t")
    grounded = ground_from_correlated(adapter, snap)
    assert grounded["t.status"] == {"A": "Active", "B": "Blocked"}


def test_correlated_skips_when_not_functional(tmp_path):
    from mnemiq.enrichment.grounding import ground_from_correlated

    # 'A' maps to two labels -> not a function -> no grounding
    adapter = _sqlite(tmp_path, "t", """
        CREATE TABLE t (id INTEGER PRIMARY KEY, status TEXT, status_name TEXT);
        INSERT INTO t VALUES (1,'A','Active'),(2,'A','Enabled'),(3,'B','Blocked');
    """)
    snap = enrich_structural(adapter, "t")
    assert ground_from_correlated(adapter, snap).get("t.status") in (None, {})


def test_correlated_skips_when_ambiguous(tmp_path):
    from mnemiq.enrichment.grounding import ground_from_correlated

    # two label-like siblings both functional -> ambiguous -> skip
    adapter = _sqlite(tmp_path, "t", """
        CREATE TABLE t (id INTEGER PRIMARY KEY, status TEXT, status_name TEXT, status_label TEXT);
        INSERT INTO t VALUES (1,'A','Active','Act'),(2,'B','Blocked','Blk');
    """)
    snap = enrich_structural(adapter, "t")
    assert ground_from_correlated(adapter, snap).get("t.status") in (None, {})


def test_lookup_grounds_fk_child_from_dimension_label(tmp_path):
    from mnemiq.enrichment.grounding import ground_from_lookup

    adapter = _sqlite(tmp_path, "film", """
        CREATE TABLE language (language_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE film (film_id INTEGER PRIMARY KEY, title TEXT,
                           language_id INTEGER REFERENCES language(language_id));
        INSERT INTO language VALUES (1,'English'),(2,'Italian');
        INSERT INTO film VALUES (1,'A',1),(2,'B',2),(3,'C',1);
    """)
    snap = enrich_structural(adapter, "film")
    grounded = ground_from_lookup(adapter, snap)
    # language_id is a key (no harvested coded_values) yet gets grounded from the dim label
    assert grounded["film.language_id"] == {"1": "English", "2": "Italian"}


def test_lookup_skips_dimension_without_a_label_column(tmp_path):
    from mnemiq.enrichment.grounding import ground_from_lookup

    adapter = _sqlite(tmp_path, "shop", """
        CREATE TABLE dim (dim_id INTEGER PRIMARY KEY, qty INTEGER);
        CREATE TABLE fact (id INTEGER PRIMARY KEY, dim_id INTEGER REFERENCES dim(dim_id));
        INSERT INTO dim VALUES (1,10),(2,20);
        INSERT INTO fact VALUES (1,1),(2,2);
    """)
    snap = enrich_structural(adapter, "shop")
    assert ground_from_lookup(adapter, snap).get("fact.dim_id") in (None, {})


def test_ground_codes_precedence_dictionary_over_lookup(tmp_path):
    from mnemiq.enrichment.dictionary import ColumnEntry, DataDictionary
    from mnemiq.enrichment.grounding import ground_codes

    adapter = _sqlite(tmp_path, "film", """
        CREATE TABLE language (language_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE film (film_id INTEGER PRIMARY KEY, title TEXT,
                           language_id INTEGER REFERENCES language(language_id));
        INSERT INTO language VALUES (1,'English'),(2,'Italian');
        INSERT INTO film VALUES (1,'A',1),(2,'B',2);
    """)
    snap = enrich_structural(adapter, "film")
    d = DataDictionary(columns={"film.language_id": ColumnEntry(codes={"1": "Anglais"})})
    out = ground_codes(adapter, snap, d)

    lang = next(c for c in out.columns if c.id == "film.language_id")
    meanings = {cv.code: (cv.meaning, cv.source) for cv in lang.coded_values}
    assert meanings["1"] == ("Anglais", "dictionary")   # dictionary wins over lookup
    assert meanings["2"] == ("Italian", "lookup")        # lookup fills the rest
    assert out.version != snap.version                    # re-versioned


def test_ground_codes_survives_a_failing_source(tmp_path, monkeypatch):
    from mnemiq.enrichment import grounding

    adapter = _sqlite(tmp_path, "t", """
        CREATE TABLE t (id INTEGER PRIMARY KEY, status TEXT, status_name TEXT);
        INSERT INTO t VALUES (1,'A','Active'),(2,'A','Active'),(3,'B','Blocked');
    """)
    snap = enrich_structural(adapter, "t")
    monkeypatch.setattr(grounding, "ground_from_lookup",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = grounding.ground_codes(adapter, snap)  # must not raise
    status = next(c for c in out.columns if c.id == "t.status")
    assert {cv.code: cv.meaning for cv in status.coded_values} == {"A": "Active", "B": "Blocked"}


def test_end_to_end_card_shows_grounded_and_dictionary_meanings(tmp_path):
    import json

    from mnemiq.enrichment.dictionary import load_dictionary
    from mnemiq.enrichment.enricher import FakeEnricher
    from mnemiq.enrichment.grounding import ground_codes
    from mnemiq.enrichment.semantic import enrich_semantic
    from mnemiq.semantic.cards import build_cards

    # 3 rows so `status` has distinct(2) < row_count(3) and is harvested as a code vocabulary.
    adapter = _sqlite(tmp_path, "shop", """
        CREATE TABLE language (language_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE film (film_id INTEGER PRIMARY KEY, title TEXT, status TEXT,
                           status_name TEXT, language_id INTEGER REFERENCES language(language_id));
        INSERT INTO language VALUES (1,'English'),(2,'Italian');
        INSERT INTO film VALUES (1,'A','L','Live',1),(2,'B','A','Archived',2),(3,'C','L','Live',1);
    """)
    dpath = tmp_path / "d.json"
    dpath.write_text(json.dumps({"columns": {"film.language_id": {"codes": {"2": "Italiano"}}}}))

    snap = enrich_structural(adapter, "shop")
    snap = ground_codes(adapter, snap, load_dictionary(str(dpath)))
    snap = enrich_semantic(snap, FakeEnricher({}))  # no LLM meanings
    cards = {c.object_id: c.text for c in build_cards(snap)}

    assert "L = Live" in cards["film"]        # correlated (status -> status_name)
    assert "1 = English" in cards["film"]     # lookup (language_id -> language.name)
    assert "2 = Italiano" in cards["film"]    # dictionary overrides lookup
