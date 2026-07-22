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
