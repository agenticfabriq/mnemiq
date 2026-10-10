"""M139: a value too long to be a code keeps a column's value set out of the snapshot.

SQLite, so these run without the ACME Postgres that gates the rest of the profiling tests.
"""
import sqlite3

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.catalog import introspect
from mnemiq.enrichment.profiling import profile_table


def _stats(tmp_path, rows, name="t.sqlite"):
    path = tmp_path / name
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, kind TEXT, doc TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()
    adapter = SQLiteAdapter(str(path))
    table = next(x for x in introspect(adapter) if x.name == "t")
    return {s.column: s for s in profile_table(adapter, table)}


def test_a_column_holding_a_document_is_not_a_code_list(tmp_path):
    # BEAVER's nova: a text column with 18 distinct JSON documents of up to 2,975 characters became an
    # 18-entry "code list" and a 24,678-character card line that no prompt fit can shrink.
    blob = '{"cells": [' + ", ".join('{"id": %d}' % i for i in range(40)) + "]}"
    assert len(blob) > 200
    stats = _stats(tmp_path, [(1, "a", blob), (2, "b", "{}"), (3, "a", "{}")])
    assert stats["doc"].top_k == []
    assert {v for v, _ in stats["kind"].top_k} == {"a", "b"}  # the short codes beside it are untouched


def test_the_length_bound_is_200_characters_inclusive(tmp_path):
    at, over = "x" * 200, "y" * 201
    kept = _stats(tmp_path, [(1, "a", at), (2, "a", "z"), (3, "b", "z")], name="kept.sqlite")
    assert at in {v for v, _ in kept["doc"].top_k}
    dropped = _stats(tmp_path, [(1, "a", over), (2, "a", "z"), (3, "b", "z")], name="dropped.sqlite")
    assert dropped["doc"].top_k == []
