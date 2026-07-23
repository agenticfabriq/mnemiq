from __future__ import annotations

import duckdb

from mnemiq.contract import Snapshot
from mnemiq.semantic.textmatch import similarity

_DDL = """
CREATE TABLE IF NOT EXISTS ontology_concept (
  source_id  TEXT,
  scheme_id  TEXT,
  notation   TEXT,
  label      TEXT,
  definition TEXT
)
"""


def build_ontology_index(records, snapshot: Snapshot, con: duckdb.DuckDBPyConnection) -> int:
    """Index the concepts of schemes actually bound to a column in this snapshot.

    One row per label VARIANT (prefLabel + each altLabel) so a synonym matches as well as the
    preferred term. Unbound schemes are skipped: a 70k-concept vocabulary nobody uses must not
    enter the store. Delete-then-insert per source, like build_value_index -- a stale index is
    worse than a missing one, because at read time the two are indistinguishable.
    """
    con.execute(_DDL)
    con.execute("DELETE FROM ontology_concept WHERE source_id = ?", [snapshot.source_id])

    bound = {c.code_scheme.id for c in snapshot.columns if c.code_scheme is not None}
    rows: list[list[str]] = []
    for scheme in records.schemes:
        if scheme.id not in bound:
            continue
        for concept in scheme.concepts:
            for label in [concept.pref_label, *concept.alt_labels]:
                if label:
                    rows.append([snapshot.source_id, scheme.id, concept.notation, label,
                                 concept.definition or ""])

    if rows:
        con.executemany(
            "INSERT INTO ontology_concept "
            "(source_id, scheme_id, notation, label, definition) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


class OntologyIndex:
    """Read side. Ensures its table on construction, so a store built before ontology
    grounding answers 'nothing indexed' rather than raising."""

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con
        con.execute(_DDL)

    def nearest(self, scheme_id: str, text: str, k: int = 5,
                floor: float = 0.25) -> list[tuple[str, str]]:
        """Top-k (notation, label) for the question text, best label variant per code.

        The floor sits BELOW the binder's affinity gate on purpose: a miss here costs one
        missing hint, whereas a wrong bind corrupts every answer for that column.
        """
        rows = self._con.execute(
            "SELECT notation, label FROM ontology_concept WHERE scheme_id = ?", [scheme_id]
        ).fetchall()

        scored = []
        for notation, label in rows:
            score = similarity(text, label)
            if score >= floor:
                scored.append((score, notation, label))
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))

        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for _score, notation, label in scored:
            if notation in seen:
                continue  # one row per code: alt-label variants must not crowd out other codes
            seen.add(notation)
            out.append((notation, label))
            if len(out) >= k:
                break
        return out
