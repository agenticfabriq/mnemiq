from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

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
_CACHE_VERSION = 1
# How stale a merged set may get before it is reconciled in full. A withdrawal is invisible to
# a delta, so this window is the only thing that ever removes one locally.
_DEFAULT_RESYNC_SECS = 86400


def fetch_certified_records(settings) -> list[CertifiedRecord]:
    """The certified set: whatever was cached locally, brought up to date from Verity.

    The pull is incremental (`since` watermark) and `enrich` is a from-scratch rebuild, so before
    M18 a second run overlaid an empty delta onto a fresh profile and silently lost every certified
    record -- and with it `_protected`, which is what stops the LLM re-guessing the columns a human
    certified. Measured live: pull 1 returned 38 records, pull 2 returned 0.

    So a delta is now merged into a locally persisted set rather than being mistaken for one. Two
    things follow, and both are the point rather than side effects:

    * **an outage degrades to last-known-good.** Fail-soft used to mean "lose every certified
      record"; returning the cached set is what makes it mean what it says.
    * **the set is periodically re-pulled in full.** A withdrawal is invisible to a delta --
      `latest_certified_versions` drops a deprecated record rather than tombstoning it -- so merging
      forever would ground on meaning Verity has withdrawn. Verity's own comment
      (`routes.rs:6452`) calls the periodic full pull the safety net; this is mnemiq performing it.
    """
    url = getattr(settings, "verity_records_url", None)
    if not url:
        return []

    cache_path = _watermark_path(settings)
    cached, watermark, synced_at = _read_cache(cache_path, url)

    full = _needs_full_sync(cached, watermark, synced_at, settings)
    since = None if full else watermark
    page_size = getattr(settings, "verity_page_size", 500)

    pulled: list[dict] = []
    cursor: str | None = None
    latest_watermark: str | None = None
    fully_drained = False
    for _ in range(_MAX_PAGES):
        payload = _get_records(settings, _page_url(url, since, cursor, page_size))
        if payload is None:
            break  # a page failed -> keep what we have; do NOT advance the watermark
        pulled.extend(payload.get("records", []))
        if payload.get("watermark"):
            latest_watermark = payload["watermark"]
        cursor = payload.get("next_cursor")
        if not cursor:
            fully_drained = True
            break

    if not fully_drained:
        # A partial pull cannot be merged: an incremental one has an unknown tail, and a full one
        # would silently truncate the set. Serve what we last knew and leave the sidecar alone.
        logger.warning(
            "verity pull incomplete; serving %d cached certified records", len(cached))
        return _parse(cached)

    records = pulled if full else _merge(cached, pulled)
    if cache_path:
        _write_cache(
            cache_path, url, records,
            watermark=latest_watermark or watermark,
            # Only a FULL pull renews the window. If a delta renewed it the window would never
            # expire, the full re-sync would never run, and the withdrawal gap would be permanent
            # while looking fixed.
            synced_at=_now_iso() if full else synced_at,
        )
    return _parse(records)


def _record_identity(item: dict) -> tuple[str, str]:
    envelope = item.get("envelope") or {}
    return (envelope.get("object_type") or "", envelope.get("object_id") or "")


def _merge(cached: list[dict], delta: list[dict]) -> list[dict]:
    """Cached set updated by a delta, keyed on `(object_type, object_id)`.

    That pair is the identity Verity itself uses -- its own uniqueness is
    `(tenant, object_type, object_id, version_hash)`, and `object_id` alone does not identify a
    record (`loss_ratio` is legitimately both a metric and a glossary definition; Verity's D124 was
    exactly this mistake on its side of the wire).
    """
    by_identity = {_record_identity(item): item for item in cached}
    for item in delta:
        by_identity[_record_identity(item)] = item
    # Sorted so the sidecar's bytes are stable across runs that changed nothing.
    return [by_identity[key] for key in sorted(by_identity)]


def _parse(items: list[dict]) -> list[CertifiedRecord]:
    out: list[CertifiedRecord] = []
    for item in items:
        try:
            out.append(CertifiedRecord.model_validate(item))
        except Exception as exc:  # one malformed record must not sink the batch
            logger.warning("skipping malformed certified record: %s", exc)
    return out


def _needs_full_sync(cached, watermark, synced_at, settings) -> bool:
    if not cached or not watermark or not synced_at:
        # Includes the upgrade path: a legacy sidecar holds a watermark and no records, and asking
        # for a delta with nothing to merge it into is the defect this function exists to prevent.
        return True
    window = getattr(settings, "verity_full_resync_after_secs", _DEFAULT_RESYNC_SECS)
    if window <= 0:
        return True
    age = _now_epoch() - _epoch_of(synced_at)
    return age >= window


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


def _read_cache(path: str | None, url: str) -> tuple[list[dict], str | None, str | None]:
    """The cached records, watermark and last-full-sync time for this URL.

    A **legacy** sidecar -- `{"<url>": "<watermark>"}`, written before M18 -- is deliberately read
    as *watermark present, nothing to merge into*: the watermark is dropped and only the emptiness
    is returned, so `_needs_full_sync` calls for one full pull and the deployment heals itself on
    first run. Honouring that watermark would ask for a delta against no base, which is the bug.
    """
    if not path or not os.path.exists(path):
        return [], None, None
    try:
        with open(path) as handle:
            data = json.load(handle)
    except Exception as exc:  # a corrupt/unreadable sidecar just means a full pull
        logger.warning("verity record cache unreadable; full pull: %s", exc)
        return [], None, None
    source = (data.get("sources") or {}).get(url)
    if not isinstance(source, dict):
        return [], None, None
    return (
        source.get("records") or [],
        source.get("watermark"),
        source.get("synced_at"),
    )


def _write_cache(path: str, url: str, records: list[dict], *,
                 watermark: str | None, synced_at: str | None) -> None:
    """The records and the watermark describing them, in one write.

    One file rather than two because a watermark that advanced while its records did not is exactly
    the split-brain M18 is about: the next run would ask for a delta against a set that was never
    saved.
    """
    try:
        data: dict = {}
        if os.path.exists(path):
            try:
                with open(path) as handle:
                    data = json.load(handle)
            except Exception:
                data = {}          # a legacy or corrupt sidecar is replaced, not merged into
        if not isinstance(data.get("sources"), dict):
            data = {"version": _CACHE_VERSION, "sources": {}}
        data["version"] = _CACHE_VERSION
        data["sources"][url] = {
            "watermark": watermark,
            "synced_at": synced_at or _now_iso(),
            "records": records,
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(data, handle)
        os.replace(tmp, path)      # atomic: a crash mid-write must not leave a half-set
    except Exception as exc:  # unwritable -> next sync is a full pull, never fatal
        logger.warning("verity record cache unwritable; next sync is a full pull: %s", exc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


def _epoch_of(stamp: str) -> float:
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except Exception:
        return 0.0  # an unparseable stamp reads as infinitely old -> full pull


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
