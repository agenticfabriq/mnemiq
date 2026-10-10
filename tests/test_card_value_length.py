"""M139: a value too long to be a code stays whole in the snapshot and is cut in the card.

SQLite, so these run without the ACME Postgres that gates the rest of the profiling tests.
"""
import sqlite3

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.semantic.cards import build_cards

# BEAVER's nova keeps JSON documents of up to 2,975 characters in text columns with 18 distinct values.
BLOB = '{"cells": [' + ", ".join('{"id": %d}' % i for i in range(300)) + "]}"


def _snapshot(tmp_path, docs):
    path = tmp_path / "t.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, doc TEXT)")
    # One repeat, so the column has fewer distinct values than rows and qualifies as a value set.
    con.executemany("INSERT INTO t VALUES (?, ?)", enumerate([*docs, docs[-1]], start=1))
    con.commit()
    con.close()
    return enrich_structural(SQLiteAdapter(str(path)), "src")


def _card_line(snapshot):
    card = next(c for c in build_cards(snapshot) if c.object_id == "t")
    return next(line for line in card.text.splitlines() if line.startswith("- doc ("))


def test_a_document_among_short_codes_stays_whole_in_the_snapshot(tmp_path):
    # Grounding and the binder read the snapshot's values, so a short code beside a document must
    # still be there, and so must the document.
    snapshot = _snapshot(tmp_path, [BLOB, "short-a", "short-b"])
    doc = next(c for c in snapshot.columns if c.id == "t.doc")
    assert {cv.code for cv in doc.coded_values} == {BLOB, "short-a", "short-b"}


def test_the_card_cuts_the_document_and_shows_the_short_codes_whole(tmp_path):
    line = _card_line(_snapshot(tmp_path, [BLOB, "short-a", "short-b"]))
    assert BLOB not in line
    assert BLOB[:200] in line
    assert f"{len(BLOB):,} characters" in line  # says it was cut, and from how long
    assert "never compare with it" in line  # the model is told to copy codes exactly; a cut one matches nothing
    assert "short-a" in line and "short-b" in line


def test_a_value_of_200_characters_is_shown_whole(tmp_path):
    at = "x" * 200
    line = _card_line(_snapshot(tmp_path, [at, "z"]))
    assert at in line and "characters" not in line


def test_a_value_of_201_characters_is_cut(tmp_path):
    over = "y" * 201
    line = _card_line(_snapshot(tmp_path, [over, "z"]))
    assert over not in line and "201 characters" in line
