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
