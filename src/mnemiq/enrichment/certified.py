from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from mnemiq.contract import CertifiedRecord, CodedValue, Snapshot

logger = logging.getLogger(__name__)

# object_type -> Snapshot list attribute, for standalone (non-column) records
_STANDALONE = {
    "definition": "definitions",
    "metric": "metrics",
    "relationship": "relationships",
    "table_facts": "table_facts",
    "example": "examples",
}


def fetch_certified_records(settings) -> list[CertifiedRecord]:
    """Pull certified records from Verity. Fail-soft: any network/decode error degrades to
    local-only enrichment (mnemiq is never bricked by a Verity outage)."""
    url = getattr(settings, "verity_records_url", None)
    if not url:
        return []
    token = getattr(settings, "verity_token", None) or ""
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("verity records unreachable; enriching local-only: %s", exc)
        return []

    out: list[CertifiedRecord] = []
    for item in payload.get("records", []):
        try:
            out.append(CertifiedRecord.model_validate(item))
        except Exception as exc:  # one malformed record must not sink the batch
            logger.warning("skipping malformed certified record: %s", exc)
    return out


def apply_certified(snapshot: Snapshot, records: list[CertifiedRecord]) -> Snapshot:
    """Overlay certified MEANING onto locally-profiled STRUCTURE.

    A column's meaning (description, semantic_type, pii_level, coded_values, code_scheme) comes
    from the certified record; its structure (data_type, row/distinct/null counts) stays local.
    Standalone objects (definition/metric/...) are appended to their lists. A certified column
    with no local counterpart is dropped -- mnemiq grounds what is physically in the DB.
    """
    by_type: dict[str, list] = {}
    for rec in records:
        by_type.setdefault(rec.envelope.object_type, []).append(rec.payload)

    certified_cols = {c.id: c for c in by_type.get("column", [])}
    known = {c.id for c in snapshot.columns}
    for col_id in certified_cols:
        if col_id not in known:
            logger.warning("certified column %r not in local schema; skipped (stale/drift)", col_id)

    new_cols = []
    for col in snapshot.columns:
        cert = certified_cols.get(col.id)
        if cert is None:
            new_cols.append(col)
            continue
        coded = [CodedValue(code=cv.code, meaning=cv.meaning, source="certified")
                 for cv in cert.coded_values]
        new_cols.append(col.model_copy(update={
            "description": cert.description,
            "semantic_type": cert.semantic_type,
            "pii_level": cert.pii_level,
            "coded_values": coded,
            "code_scheme": cert.code_scheme,
        }))

    updates: dict = {"columns": new_cols}
    for object_type, attr in _STANDALONE.items():
        payloads = by_type.get(object_type, [])
        if payloads:
            updates[attr] = [*getattr(snapshot, attr), *payloads]

    return snapshot.model_copy(update=updates, deep=True)
