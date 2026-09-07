from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from mnemiq.config import CERTIFIED_RECORDS_PATH
from mnemiq.contract import PII_LEVELS, CertifiedRecord, CodedValue, Job, Snapshot
from mnemiq.contract.semantic import CertifiedRef
from mnemiq.enrichment.verity_auth import access_token
from mnemiq.ontology.records import ConceptScheme
from mnemiq.semantic.values import SENSITIVE_PII

logger = logging.getLogger(__name__)

# What an unrecognised `pii_level` is treated as (M41). `pii` over `phi` is a deliberate choice,
# not an ordering: both keep the column out of the value index, and claiming health data we were
# never told about would be a second wrong assertion on top of the one being corrected.
#
# Written as a literal and CHECKED, rather than derived. The first version read
# `sorted(SENSITIVE_PII)[0]`, which looks defensive and is not: it selects by spelling, so it
# happened to give "pii" only because of the alphabet, would pick wrongly if a third level were
# added, and raises IndexError on an empty set. A guard whose correctness is a coincidence of
# sort order is the shape this register keeps finding. The check below fails at import if the
# constant ever falls out of either set it has to belong to.
_MOST_SENSITIVE = "pii"
if _MOST_SENSITIVE not in SENSITIVE_PII or _MOST_SENSITIVE not in PII_LEVELS:
    raise RuntimeError(
        f"the fallback pii_level {_MOST_SENSITIVE!r} must be both a real level {PII_LEVELS} and "
        f"one the value-index gate treats as sensitive {sorted(SENSITIVE_PII)}"
    )

# object_type -> Snapshot list attribute, for standalone (non-column) records
_STANDALONE = {
    "definition": "definitions",
    "metric": "metrics",
    "dimension": "dimensions",
    "relationship": "relationships",
    "table_facts": "table_facts",
    "example": "examples",
}


# A record whose envelope will not yield an identity either. Named rather than dropped: a count
# with no name is worse than a name, and better than the silence this replaced.
_UNIDENTIFIED = "<unidentified record>"

# How much of a refusal body to read at all, and how much of it to log. The first bound is
# the one that matters: the second only shortens a string already in memory.
_REFUSAL_BODY_BYTES = 4096
_REFUSAL_LOG_CHARS = 500

_MAX_PAGES = 10000  # a guard against a misbehaving server; real corpora are far smaller
_CACHE_VERSION = 1
# How stale a merged set may get before it is reconciled in full. A withdrawal is invisible to
# a delta, so this window is the only thing that ever removes one locally.
_DEFAULT_RESYNC_SECS = 86400


@dataclass(frozen=True)
class CertifiedSet:
    """The certified corpus, and whether we could READ it.

    `records` alone could not distinguish a tenant that has certified nothing from a Verity that
    refused us: both were `[]`. That is M2's defect a second time -- `GrantSet.available` exists
    because "the policy could not be read" and "the policy grants nothing" both denied everything
    and were the same empty set -- and here it is worse, because the failure looks like health. A
    deployment configured to answer from a certified corpus, whose pull 401s, answers UNGROUNDED
    and is indistinguishable from one that is working.

    NOT iterable and NOT truthy, deliberately. `if certified:` used to mean "non-empty"; on an
    object it would be silently True even for an unreadable corpus, which is this defect wearing a
    different hat -- so it raises rather than answering. Both refusals push every call site into
    saying `.records` or `.available` explicitly, which is the decision the type exists to force.

    `available` is True when we have a corpus to stand on, which includes two states that are not
    failures: nothing has been certified yet, and no `verity_records_url` is configured at all. It
    is also True when a failed pull degraded to a POPULATED cache, because last-known-good is an
    answer (M18); it is False only when we asked and came away with nothing to ground on.
    """

    records: list[CertifiedRecord]
    available: bool = True
    # What arrived and could NOT be read, by `<object_type>:<object_id>` off the envelope. A
    # record that fails validation is dropped so one bad row cannot sink the batch -- right, and
    # silent: the column it described then keeps the LLM's GUESSED `pii_level`, which
    # `build_access_policy` reads, so the loss quietly UN-MASKS a column and shrinks `_protected`
    # so the LLM may re-guess a meaning a human certified. Carried here so the run can say so
    # (M10); `available` is untouched, because part of a corpus is not none of it.
    skipped: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        raise TypeError(
            "CertifiedSet has no truth value: `if certified:` used to mean 'non-empty' and would "
            "now be True even for a corpus we could not read. Say `.records` for what we have or "
            "`.available` for whether we could read it."
        )


def require_certified(certified, settings) -> None:
    """Refuse when a deployment ASKED for a certified corpus and has none to stand on.

    Fail-soft is right for freshness and wrong for grounding. With `verity_records_url` set, an
    unreadable corpus means every answer is composed without the meanings a human certified -- and
    it looks exactly like a healthy ungrounded engine, which is why nothing caught it. Beacon gates
    its own grounded arm on `expect_records` for this reason; this is the engine doing the same for
    itself.

    Not raised on emptiness: a tenant that has certified nothing, an unconfigured local run, and a
    failed pull that degraded to a populated cache all keep `available` True. Only "we asked and
    came away with nothing" refuses.
    """
    if getattr(settings, "verity_records_url", None) and not certified.available:
        raise RuntimeError(
            "certified records were configured but could not be read, so this run would be "
            "grounded on nothing while looking healthy. The per-event log above carries the cause "
            "verbatim (a 401 and an outage are different problems). Set no verity_records_url to "
            "run local-only on purpose."
        )


def fetch_certified_records(settings) -> CertifiedSet:
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
        # Unconfigured is a supported deployment, not an outage: a local run grounds on whatever
        # the local digest produced and never asked Verity for anything.
        return CertifiedSet([], available=True)

    complaint = _url_complaint(url)
    if complaint:
        # Said once, before the pull, because the symptom otherwise is either a 400 on every page
        # or -- worse -- a pull that drains and has quietly stopped being incremental. A WARNING
        # and not a refusal: the URL is the operator's to choose and a proxy in front of Verity
        # may legitimately serve this endpoint elsewhere.
        logger.warning("verity_records_url %r %s", url, complaint)

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
        # Degraded freshness is not absence. A populated cache still grounds the engine, so it
        # stays available; an EMPTY one means we asked for a corpus and have none, which is the
        # state a configured deployment must not answer from silently.
        records, skipped = _parse(cached)
        return CertifiedSet(records, available=bool(cached), skipped=skipped)

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
    parsed, skipped = _parse(records)
    return CertifiedSet(parsed, available=True, skipped=skipped)


def apply_certified_set(snapshot: Snapshot, certified: CertifiedSet
                        ) -> tuple[Snapshot, frozenset[str]]:
    """A pull result, laid onto a snapshot: the overlay, what the corpus PINS, and what it lost.

    Every product caller wrote these three steps out itself -- apply the records, derive
    `_protected` from the same list, and ignore the rest of the set -- so `cli`, `eval/run` and
    `eval/bird_runner` each carried the same five lines. That is why this exists rather than a
    keyword on `apply_certified`: a fact the pull learns has to reach the snapshot through ONE
    step, or it reaches two callers of three and reads as a guarantee (**M79**, and M11 one
    week later).

    `apply_certified` stays a pure overlay of the records it is HANDED. The unreadable ones are a
    fact about the pull, not about the records that arrived, and this is the seam where a pull
    meets a snapshot -- so the job is written here.
    """
    snapshot = apply_certified(snapshot, certified.records)
    if certified.skipped:
        # M2's lesson, and M10's: a security-relevant loss belongs in the structured run record.
        # A dropped column record leaves the LLM's guessed `pii_level` standing, which
        # `build_access_policy` reads -- so this is the difference between an un-masked column
        # somebody can be asked about afterwards and one nobody can.
        logger.warning(
            "%d certified record(s) could not be read and were dropped; the columns they "
            "described keep their locally guessed meaning: %s",
            len(certified.skipped), ", ".join(certified.skipped),
        )
        snapshot = snapshot.model_copy(update={"jobs": [*snapshot.jobs, Job(
            id="certified:record_unreadable",
            source_id=snapshot.source_id,
            kind="certified_record_unreadable",
            status="refused",
            checkpoints=list(certified.skipped),
        )]}, deep=True)
    protected = frozenset(r.envelope.object_id for r in certified.records
                          if r.envelope.object_type == "column")
    return snapshot, protected


def _url_complaint(url: str) -> str | None:
    """What is wrong with this records URL for an INCREMENTAL pull, if anything.

    Two different problems, and only one of them is about the path:

    * **the wrong endpoint.** Compared on the parsed PATH, not the whole string: `_page_url`
      appends its parameters with `"&" if "?" in url else "?"`, so a records URL that already
      carries a query string is a shape this module supports, and an `endswith` over the raw URL
      called every one of them misconfigured.
    * **a fragment.** MEASURED, not reasoned: `_page_url` appends `?since=...&limit=...` AFTER
      the `#`, and `urllib` cuts the request line at the first `#` -- so
      `.../records/open#frag` requests `/api/semantic/records/open` with no parameters at all,
      on every page. That is the silent full-dump-merged-as-a-delta this warning exists to
      announce, wearing the correct path, and the first version of this check called it fine.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.fragment:
        return (
            "carries a URL fragment; the pull appends `since`, `cursor` and `limit` after it and "
            "urllib drops everything from the `#`, so every page is requested unparameterised "
            "and the pull is a full dump wearing the shape of a delta"
        )
    if not parts.path.rstrip("/").endswith(CERTIFIED_RECORDS_PATH):
        return (
            f"has the path {parts.path!r}, not {CERTIFIED_RECORDS_PATH}; only that endpoint "
            "accepts `since`, so a pull against any other one is not incremental however well "
            "it appears to work"
        )
    return None


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


def _parse(items: list[dict]) -> tuple[list[CertifiedRecord], tuple[str, ...]]:
    """The records that validated, and the identities of the ones that did not.

    The identity is read off the ENVELOPE, which is a plain dict here and survives a payload that
    does not validate -- so the caller learns WHICH column silently kept its guess, rather than
    only that something was dropped. A record whose envelope is unusable too has no name to give
    and says so, because a count with no identity is still better than a log line nobody reads.
    """
    out: list[CertifiedRecord] = []
    skipped: list[str] = []
    for item in items:
        try:
            out.append(CertifiedRecord.model_validate(item))
        except Exception as exc:  # one malformed record must not sink the batch
            object_type, object_id = _record_identity(item) if isinstance(item, dict) else ("", "")
            identity = f"{object_type}:{object_id}" if object_type or object_id else _UNIDENTIFIED
            logger.warning("skipping unreadable certified record %s: %s", identity, exc)
            skipped.append(identity)
    return out, tuple(sorted(skipped))


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
            # The BODY, not just the status. `require_certified` tells the operator the log
            # "carries the cause verbatim", and for a 400 it did not: `HTTPError.__str__` is
            # "HTTP Error 400: Bad Request" while the sentence naming the bad parameter -- which
            # is how a deployment learns its `verity_records_url` points at the endpoint that
            # takes no `since` (M82) -- sits in a body nothing read.
            logger.warning("verity records unreachable; enriching local-only: %s%s",
                           exc, _refusal_detail(exc))
            return None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("verity records unreachable; enriching local-only: %s", exc)
            return None
    return None


def _refusal_detail(exc: "urllib.error.HTTPError") -> str:
    """Verity's own words for the refusal, or nothing at all.

    Read defensively and BOUNDED: this is a remote server's response reaching a log line, the body
    may already have been consumed, and an HTML error page from a proxy in front of Verity is not
    a diagnosis worth pasting whole.
    """
    try:
        # Bounded at the READ, not only at the log line: a proxy answering a large error
        # page to the records URL would otherwise be pulled into memory in full and then
        # thrown away, under a docstring promising it was not.
        body = exc.read(_REFUSAL_BODY_BYTES).decode("utf-8", "replace").strip()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            for key in ("detail", "message", "error"):
                value = parsed.get(key)
                if isinstance(value, str) and value:
                    body = value
                    break
    except ValueError:
        pass
    return f" -- verity said: {body[:_REFUSAL_LOG_CHARS]}"


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
    unrecognised_levels: list[str] = []
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
        # M41. `Column.pii_level` is `str | None` and this is the one producer that does not clamp
        # to the vocabulary (enrichment parses through `_in_vocabulary`), so an unrecognised value
        # used to land on the column -- and `_qualifies` gates harvesting on
        # `pii_level not in SENSITIVE_PII`, which anything unrecognised passes. That fails OPEN:
        # "personal", "sensitive", a typo'd "pii " all read as sensitive to a human and as
        # harvestable to the code. The value is recorded here and SUBSTITUTED BELOW, after M1.
        unrecognised = pii_level is not None and pii_level not in PII_LEVELS
        if unrecognised:
            logger.warning(
                "unrecognised pii_level %r on %r (not one of %s) -- an unrecognised level must "
                "not be read as non-sensitive",
                pii_level, col.id, "/".join(PII_LEVELS),
            )
            unrecognised_levels.append(col.id)
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
        elif unrecognised:
            # Substituting BEFORE the check above inverted the check's own predicate and was a
            # REGRESSION, caught in review: `"personal" not in SENSITIVE_PII` refuses, but the
            # substituted `"pii"` does not, so an UNATTESTED record could walk a column phi -> pii.
            # The harvest gate cannot see that (both are sensitive) and authorization can:
            # `pii_clearance` grants on the exact level, so a pii-cleared, non-phi-cleared role
            # then read a phi column raw. M1 therefore judges what the record actually SAID, and
            # the substitution only applies where M1 let the value through. Refusing the whole
            # record was rejected for M1's stated reason: the danger is one field, and dropping
            # meaning over it degrades enrichment.
            pii_level = _MOST_SENSITIVE
        new_cols.append(col.model_copy(update={
            "description": cert.description,
            "semantic_type": cert.semantic_type,
            "pii_level": pii_level,
            "coded_values": coded,
            "code_scheme": cert.code_scheme,
        }))

    updates: dict = {"columns": new_cols}
    # `snapshot.jobs` is this codebase's structured run record, so a security-relevant refusal goes
    # there rather than living only in a log line nobody reads (register M2's lesson). Accumulated
    # into ONE list rather than assigned per condition: two `updates["jobs"] = [*snapshot.jobs, ...]`
    # statements each rebuild from the original, so whichever ran second dropped the other's job
    # while both still logged -- the record gone, the logs reassuring. The two kinds are distinct
    # facts and both must survive: one is "the record was not trusted to lower this", the other is
    # "the record said something this field has no meaning for".
    new_jobs: list[Job] = []
    if unrecognised_levels:
        new_jobs.append(
            Job(
                id="certified:pii_level_unrecognised",
                source_id=snapshot.source_id,
                kind="certified_pii_level_unrecognised",
                status="refused",
                checkpoints=sorted(unrecognised_levels),
            )
        )
    if refused_downgrades:
        new_jobs.append(
            Job(
                id="certified:pii_downgrade_refused",
                source_id=snapshot.source_id,
                kind="certified_pii_downgrade_refused",
                status="refused",
                checkpoints=sorted(refused_downgrades),
            )
        )
    if new_jobs:
        updates["jobs"] = [*snapshot.jobs, *new_jobs]
    for object_type, attr in _STANDALONE.items():
        payloads = by_type.get(object_type, [])
        if payloads:
            updates[attr] = [*getattr(snapshot, attr), *payloads]

    # M24: keep WHICH certified version supplied each piece of meaning. The envelope was in hand
    # the whole time and was dropped here, so mnemiq carried certified content it could not
    # attribute -- and an emitted trace could never name what the answer relied on, leaving
    # Verity's D131 reader with no producer. Keyed on (object_type, object_id) because `object_id`
    # alone does not identify a record: `loss_ratio` is legitimately both a metric and a glossary
    # definition, which is the mistake D124 was.
    refs = {
        (rec.envelope.object_type, rec.envelope.object_id): CertifiedRef(
            object_type=rec.envelope.object_type,
            object_id=rec.envelope.object_id,
            version_hash=rec.envelope.version,
        )
        for rec in records
    }
    if refs or snapshot.certified_refs:
        existing = {(r.object_type, r.object_id): r for r in snapshot.certified_refs}
        existing.update(refs)
        updates["certified_refs"] = [existing[key] for key in sorted(existing)]

    return snapshot.model_copy(update=updates, deep=True)
