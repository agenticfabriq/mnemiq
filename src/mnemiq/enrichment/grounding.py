from __future__ import annotations

import logging

from mnemiq.catalog import is_key_like
from mnemiq.contract import CodedValue, Snapshot
from mnemiq.enrichment.pipeline import content_version

logger = logging.getLogger(__name__)

# Positive allowlist for "label" columns. Deliberately NOT is_sensitive_name, which excludes
# the very `*_name`/`name` columns a lookup/correlated label lives in (language.name,
# status_name). This admits those while never reading arbitrary (possibly-PII) columns.
_LABEL_TOKENS = ("name", "title", "label", "desc")


def _label_like(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in _LABEL_TOKENS)


def _columns_by_table(snapshot: Snapshot) -> dict[str, list]:
    by_table: dict[str, list] = {}
    for col in snapshot.columns:
        by_table.setdefault(col.object_id, []).append(col)
    return by_table


def _is_functional(adapter, table: str, c: str, d: str) -> bool:
    """Each non-null C value maps to exactly one D value."""
    rows = adapter.execute(
        f'SELECT "{c}", count(DISTINCT "{d}") FROM "{table}" '
        f'WHERE "{c}" IS NOT NULL GROUP BY "{c}"'
    )
    return bool(rows) and all(r[1] == 1 for r in rows)


def _safe_functional(adapter, table: str, c: str, d: str) -> bool:
    try:
        return _is_functional(adapter, table, c, d)
    except Exception as exc:
        logger.warning("FD check failed for %s.%s~%s: %s", table, c, d, exc)
        return False


def ground_from_correlated(adapter, snapshot: Snapshot) -> dict[str, dict[str, str]]:
    """Ground a coded column from a sibling label column it functionally determines."""
    out: dict[str, dict[str, str]] = {}
    for table, cols in _columns_by_table(snapshot).items():
        for col in cols:
            if not col.coded_values:
                continue
            candidates = [
                d.name for d in cols
                if d.name != col.name and _label_like(d.name) and not is_key_like(d.name)
            ]
            functional = [d for d in candidates if _safe_functional(adapter, table, col.name, d)]
            if len(functional) != 1:
                continue  # zero or ambiguous -> ground nothing
            d = functional[0]
            try:
                pairs = adapter.execute(
                    f'SELECT DISTINCT "{col.name}", "{d}" FROM "{table}" '
                    f'WHERE "{col.name}" IS NOT NULL AND "{d}" IS NOT NULL'
                )
            except Exception as exc:
                logger.warning("correlated grounding failed for %s.%s: %s", table, col.name, exc)
                continue
            mapping = {str(code): str(label) for code, label in pairs if str(label) != ""}
            if mapping:
                out[col.id] = mapping
    return out


_CODE_MAX_DISTINCT = 25  # keep parity with profiling.profile_table


def ground_from_lookup(adapter, snapshot: Snapshot) -> dict[str, dict[str, str]]:
    """Ground a low-cardinality FK child column from its dimension table's label column.

    Independent of the structural code-harvest: FK children are excluded from harvesting
    (they are keys), so this is where dimension codes (film.language_id -> language.name)
    actually get grounded.
    """
    by_id = {(c.object_id, c.name): c for c in snapshot.columns}
    by_table = _columns_by_table(snapshot)
    out: dict[str, dict[str, str]] = {}

    for rel in snapshot.relationships:
        for jk in rel.join_keys:
            child = by_id.get((rel.from_, jk.left))
            if child is None or not child.distinct_count:
                continue
            if not (0 < child.distinct_count <= _CODE_MAX_DISTINCT):
                continue
            labels = [
                c.name for c in by_table.get(rel.to, [])
                if _label_like(c.name) and c.name != jk.right and not is_key_like(c.name)
            ]
            if len(labels) != 1:
                continue  # no clear label, or ambiguous -> skip
            label = labels[0]
            try:
                pairs = adapter.execute(
                    f'SELECT DISTINCT u."{jk.right}", u."{label}" '
                    f'FROM "{rel.from_}" t JOIN "{rel.to}" u ON t."{jk.left}" = u."{jk.right}" '
                    f'WHERE t."{jk.left}" IS NOT NULL'
                )
            except Exception as exc:
                logger.warning("lookup grounding failed for %s.%s: %s", rel.from_, jk.left, exc)
                continue
            mapping = {str(code): str(lbl) for code, lbl in pairs if str(lbl) != ""}
            if mapping:
                out[child.id] = mapping
    return out


def ontology_meanings(snapshot: Snapshot, records) -> dict[str, dict[str, str]]:
    """notation -> pref_label for bound columns' harvested codes.

    Takes no adapter: binding already did the database work, so this is a pure lookup. Only
    columns that actually harvested a vocabulary can be filled -- a large code system arrives
    with no coded_values at all, and is served by the concept index at question time instead.
    """
    by_scheme = {
        s.id: {c.notation.strip().casefold(): c.pref_label for c in s.concepts}
        for s in records.schemes
    }
    out: dict[str, dict[str, str]] = {}
    for col in snapshot.columns:
        if col.code_scheme is None or not col.coded_values:
            continue
        table = by_scheme.get(col.code_scheme.id, {})
        mapping = {
            cv.code: table[cv.code.strip().casefold()]
            for cv in col.coded_values
            if cv.code.strip().casefold() in table
        }
        if mapping:
            out[col.id] = mapping
    return out


def apply_dictionary(snapshot: Snapshot, dictionary) -> Snapshot:
    """Overlay operator meanings + column descriptions. Highest precedence. Not re-versioned."""
    entries = dictionary.columns
    known = {c.id for c in snapshot.columns}
    for col_id in entries:
        if col_id not in known:
            logger.warning("dictionary column %r not in schema; skipped", col_id)

    new_cols = []
    for col in snapshot.columns:
        entry = entries.get(col.id)
        if entry is None:
            new_cols.append(col)
            continue
        coded = [
            CodedValue(code=cv.code, meaning=entry.codes[cv.code], source="dictionary")
            if cv.code in entry.codes else cv
            for cv in col.coded_values
        ]
        new_cols.append(col.model_copy(update={
            "coded_values": coded,
            "description": entry.description or col.description,
        }))
    return snapshot.model_copy(update={"columns": new_cols}, deep=True)


def _safe(fn, adapter, snapshot) -> dict[str, dict[str, str]]:
    try:
        return fn(adapter, snapshot)
    except Exception as exc:
        logger.warning("%s failed; skipped: %s", getattr(fn, "__name__", fn), exc)
        return {}


def ground_codes(adapter, snapshot: Snapshot, dictionary=None, ontology=None) -> Snapshot:
    """Fill CodedValue.meaning: ontology < correlated < lookup < operator dictionary.

    Ontology is the floor, not a competitor: an external standard describes the world, while a
    correlated/lookup label describes THIS database, so the database wins wherever it speaks.
    Seeding the merge map first and letting the in-DB sources overwrite IS that precedence.
    """
    onto: dict[str, dict[str, str]] = {}
    if ontology is not None:
        from mnemiq.ontology.binder import bind_schemes

        snapshot = bind_schemes(adapter, snapshot, ontology)
        try:
            onto = ontology_meanings(snapshot, ontology)
        except Exception as exc:
            logger.warning("ontology grounding failed; skipped: %s", exc)

    corr = _safe(ground_from_correlated, adapter, snapshot)
    look = _safe(ground_from_lookup, adapter, snapshot)

    # column_id -> {code: (meaning, source)}; later sources overwrite earlier ones
    merged: dict[str, dict[str, tuple[str, str]]] = {}
    for col_id, m in onto.items():
        merged.setdefault(col_id, {}).update({k: (v, "ontology") for k, v in m.items()})
    for col_id, m in corr.items():
        merged.setdefault(col_id, {}).update({k: (v, "correlated") for k, v in m.items()})
    for col_id, m in look.items():
        merged.setdefault(col_id, {}).update({k: (v, "lookup") for k, v in m.items()})

    new_cols = []
    for col in snapshot.columns:
        grounded = merged.get(col.id)
        if not grounded:
            new_cols.append(col)
            continue
        existing = {cv.code: cv for cv in col.coded_values}
        codes = list(existing) + [c for c in grounded if c not in existing]  # preserve order
        coded = []
        for code in codes:
            if code in grounded:
                meaning, source = grounded[code]
                coded.append(CodedValue(code=code, meaning=meaning, source=source))
            else:
                coded.append(existing[code])
        new_cols.append(col.model_copy(update={"coded_values": coded}))

    grounded_snap = snapshot.model_copy(update={"columns": new_cols}, deep=True)
    if dictionary is not None:
        grounded_snap = apply_dictionary(grounded_snap, dictionary)
    grounded_snap.version = content_version(grounded_snap)
    return grounded_snap
