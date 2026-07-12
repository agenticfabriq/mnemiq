from mnemiq.store.bootstrap import init_store


def test_init_store_loads_extensions_and_is_idempotent(tmp_path):
    p = str(tmp_path / "semantic.duckdb")
    con = init_store(p)
    exts = {
        r[0]
        for r in con.execute(
            "SELECT extension_name FROM duckdb_extensions() WHERE loaded"
        ).fetchall()
    }
    assert {"vss", "fts"} <= exts
    con.close()
    con2 = init_store(p)
    ver = con2.execute("SELECT count(*) FROM enrichment_version").fetchone()[0]
    assert ver == 0
    con2.close()
