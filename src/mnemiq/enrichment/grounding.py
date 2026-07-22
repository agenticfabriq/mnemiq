from __future__ import annotations

import logging

from mnemiq.catalog import is_key_like
from mnemiq.contract import Snapshot

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
