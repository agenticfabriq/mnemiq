from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from mnemiq.contract import CertifiedRecord, CodedValue, Job, Snapshot
from mnemiq.enrichment.verity_auth import access_token
from mnemiq.ontology.records import ConceptScheme
from mnemiq.semantic.values import SENSITIVE_PII

logger = logging.getLogger(__name__)

# object_type -> Snapshot list attribute, for standalone (non-column) records
_STANDALONE = {
    "definition": "definitions",
    "metric": "metrics",
    "dimension": "dimensions",
    "relationship": "relationships",
    "table_facts": "table_facts",
    "example": "examples",
}


_MAX_PAGES = 10000  # a guard against a misbehaving server; real corpora are far smaller


def fetch_certified_records(settings) -> list[CertifiedRecord]:
    """Pull certified records from Verity, paginating (`next_cursor`) and syncing incrementally
    (`since` watermark). Fail-soft: any auth/network/decode error degrades to local-only enrichment
    (mnemiq is never bricked by a Verity outage or a revoked credential)."""
    url = getattr(settings, "verity_records_url", None)
    if not url:
        return []

    watermark_path = _watermark_path(settings)
    since = _read_watermark(watermark_path, url)
    page_size = getattr(settings, "verity_page_size", 500)

    out: list[CertifiedRecord] = []
    cursor: str | None = None
    latest_watermark: str | None = None
    fully_drained = False
    for _ in range(_MAX_PAGES):
        payload = _get_records(settings, _page_url(url, since, cursor, page_size))
        if payload is None:
            break  # a page failed -> keep what we have; do NOT advance the watermark
        for item in payload.get("records", []):
            try:
                out.append(CertifiedRecord.model_validate(item))
            except Exception as exc:  # one malformed record must not sink the batch
                logger.warning("skipping malformed certified record: %s", exc)
        if payload.get("watermark"):
            latest_watermark = payload["watermark"]
        cursor = payload.get("next_cursor")
        if not cursor:
            fully_drained = True
            break

    if fully_drained and latest_watermark and watermark_path:
        _write_watermark(watermark_path, url, latest_watermark)
    return out


def _page_url(url: str, since: str | None, cursor: str | None, limit: int | None = None) -> str:
    from urllib.parse import urlencode

    params: dict[str, object] = {}
    if since:
        params["since"] = since
    if cursor:
        params["cursor"] = cursor
    if limit and limit > 0:
        params["limit"] = limit
    if not params:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode(params)}"


def _watermark_path(settings) -> str | None:
    path = getattr(settings, "verity_watermark_path", None)
    if path:
        return path
    store = getattr(settings, "store_path", None)
    if store:
        return os.path.join(os.path.dirname(store) or ".", "verity-watermark.json")
    return None


def _read_watermark(path: str | None, url: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle).get(url)
    except Exception as exc:  # a corrupt/unreadable sidecar just means a full pull
        logger.warning("verity watermark unreadable; full pull: %s", exc)
        return None


def _write_watermark(path: str, url: str, watermark: str) -> None:
    try:
        data: dict = {}
        if os.path.exists(path):
            with open(path) as handle:
                data = json.load(handle)
        data[url] = watermark
        with open(path, "w") as handle:
            json.dump(data, handle)
    except Exception as exc:  # unwritable -> next sync is non-incremental, never fatal
        logger.warning("verity watermark unwritable; next sync non-incremental: %s", exc)


def _get_records(settings, url: str) -> dict | None:
    """GET the records with a Bearer token, refreshing once on 401.

    A 401 is the expected answer when the cached token expired server-side or the credential
    was revoked, so it earns exactly one forced refresh + retry -- never a loop. Returns None
    when the pull could not be completed.
    """
    for force_refresh in (False, True):
        token = access_token(settings, force_refresh=force_refresh)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 401 and token is not None and not force_refresh:
                logger.info("verity rejected the token (401); refreshing once")
                continue
            logger.warning("verity records unreachable; enriching local-only: %s", exc)
            return None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("verity records unreachable; enriching local-only: %s", exc)
            return None
    return None


def certified_concept_schemes(records: list[CertifiedRecord]) -> list[ConceptScheme]:
    """The certified `concept_scheme` records as ConceptScheme objects. The records pull already
    resolved each payload to its typed form, so this is a typed filter; non-scheme records are
    ignored (definitions flow via apply_certified). Never raises -- fail-soft like the rest of the pull."""
    return [
        record.payload
        for record in records
        if record.envelope.object_type == "concept_scheme"
        and isinstance(record.payload, ConceptScheme)
    ]


def _is_attested(record: CertifiedRecord) -> bool:
    """Is this record's certification claim backed by someone?

    `Provenance.status` is a producer-supplied string and `certifier` is optional, so a record can
    assert certification with nobody attesting it -- which is exactly what Verity published for every
    draft-certified record before its D56 fix. That is fine for meaning; it is not fine for a field
    that gates whether we harvest a column's values (register M1).
    """
    provenance = record.envelope.provenance
    return (
        provenance is not None
        and provenance.status == "certified"
        and bool(provenance.certifier)
    )


def apply_certified(snapshot: Snapshot, records: list[CertifiedRecord]) -> Snapshot:
    """Overlay certified MEANING onto locally-profiled STRUCTURE.

    A column's meaning (description, semantic_type, pii_level, coded_values, code_scheme) comes
    from the certified record; its structure (data_type, row/distinct/null counts) stays local.
    Standalone objects (definition/metric/...) are appended to their lists. A certified column
    with no local counterpart is dropped -- mnemiq grounds what is physically in the DB.

    **`pii_level` is trusted only from an attested record (M1).** An unattested one may still supply
    description, semantic_type, coded_values and code_scheme, and may still *raise* sensitivity --
    it simply cannot move a column OUT of `SENSITIVE_PII`, because that is the set
    `semantic.values._qualifies` gates value harvesting on. The rule is set membership against that
    same set rather than an ordering over the level strings, so the check and the gate cannot drift.
    """
    by_type: dict[str, list] = {}
    attested: dict[str, bool] = {}
    for rec in records:
        by_type.setdefault(rec.envelope.object_type, []).append(rec.payload)
        if rec.envelope.object_type == "column":
            # Last writer wins, matching the payload dict below.
            attested[rec.envelope.object_id] = _is_attested(rec)

    refused_downgrades: list[str] = []
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
        pii_level = cert.pii_level
        if (
            not attested.get(col.id, False)
            and col.pii_level in SENSITIVE_PII
            and pii_level not in SENSITIVE_PII
        ):
            logger.warning(
                "refusing to lower pii_level on %r from %r to %r: the certified record names no "
                "certifier, and this field gates value harvesting",
                col.id, col.pii_level, pii_level,
            )
            refused_downgrades.append(col.id)
            pii_level = col.pii_level
        new_cols.append(col.model_copy(update={
            "description": cert.description,
            "semantic_type": cert.semantic_type,
            "pii_level": pii_level,
            "coded_values": coded,
            "code_scheme": cert.code_scheme,
        }))

    updates: dict = {"columns": new_cols}
    if refused_downgrades:
        # `snapshot.jobs` is this codebase's structured run record, so a security-relevant refusal
        # goes there rather than living only in a log line nobody reads (register M2's lesson).
        updates["jobs"] = [
            *snapshot.jobs,
            Job(
                id="certified:pii_downgrade_refused",
                source_id=snapshot.source_id,
                kind="certified_pii_downgrade_refused",
                status="refused",
                checkpoints=sorted(refused_downgrades),
            ),
        ]
    for object_type, attr in _STANDALONE.items():
        payloads = by_type.get(object_type, [])
        if payloads:
            updates[attr] = [*getattr(snapshot, attr), *payloads]

    return snapshot.model_copy(update=updates, deep=True)
