"""What the certified pull loses, it has to say it lost.

M10's live residue and M82. The pull is fail-soft by design and that is right -- an outage
degrades to last-known-good rather than losing every certified record. What is not right is
losing PART of the corpus in silence: a record `_parse` cannot validate is dropped with a log
line, and the column it described then keeps the LLM's GUESSED `pii_level`, which flows into
`build_access_policy`. So a malformed record silently un-masks a column, `_protected` shrinks
so the LLM is free to re-guess a meaning a human certified, and the run prints its success line
and exits 0. The register's M2 lesson is the fix: a security-relevant loss belongs in the
structured run record, not only in a log.

M82 is the same seam one step earlier -- a deployment configured from the field's own
description cannot pull at all, and the refusal that says why is thrown away.
"""
import io
import json as _json

import pytest


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _record(object_id):
    return {
        "envelope": {"object_type": "definition", "object_id": object_id, "version": "v1",
                     "source_system": "pg"},
        "payload": {"id": object_id, "term": object_id, "domain": "ops", "definition": "d"},
    }


def _column_record(col_id, *, malformed=False):
    table, name = col_id.split(".")
    payload = {"id": col_id, "object_id": table, "name": name, "description": "d",
               "pii_level": "none"}
    if malformed:
        payload = {"description": "d"}          # no `id`, no `name`: not a column
    return {
        "envelope": {"object_type": "column", "object_id": col_id, "version": "v1",
                     "source_system": "pg"},
        "payload": payload,
    }


def _settings(tmp_path, **extra):
    from mnemiq.config import Settings
    return Settings(verity_records_url="https://v/api/semantic/records/open",
                    verity_watermark_path=str(tmp_path / "wm.json"), **extra)


def _serving(payload):
    def fake_urlopen(req, timeout=0):
        return _Resp(_json.dumps(payload).encode())
    return fake_urlopen


# --- M10: a record that did not parse is a record the run has to name ---

def test_a_record_that_could_not_be_parsed_is_named_on_the_set(tmp_path, monkeypatch):
    from mnemiq.enrichment import certified as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen", _serving({
        "records": [_record("good"), _column_record("claim.ssn", malformed=True)],
        "watermark": "t1", "next_cursor": None}))

    got = mod.fetch_certified_records(_settings(tmp_path))
    assert [r.envelope.object_id for r in got.records] == ["good"]
    assert got.skipped == ("column:claim.ssn",), \
        "the identity comes off the ENVELOPE, which parses even when the payload does not -- " \
        "naming the column is the whole point, since that is what silently keeps its guess"
    assert got.available is True, "one bad record is not an unreadable corpus"


def test_a_clean_pull_reports_nothing_skipped(tmp_path, monkeypatch):
    from mnemiq.enrichment import certified as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen", _serving({
        "records": [_record("a")], "watermark": "t1", "next_cursor": None}))
    assert mod.fetch_certified_records(_settings(tmp_path)).skipped == ()


def test_a_record_that_stopped_parsing_IN_THE_CACHE_is_named_too(tmp_path, monkeypatch):
    """The cached set is served whenever a pull cannot be drained, and it is parsed by the same
    function. A record written by an older mnemiq that a newer contract rejects is lost exactly
    the same way, on the path taken when Verity is DOWN -- when nobody is watching."""
    from mnemiq.enrichment import certified as mod

    cache = tmp_path / "wm.json"
    cache.write_text(_json.dumps({"version": 1, "sources": {
        "https://v/api/semantic/records/open": {
            "watermark": "t1", "synced_at": mod._now_iso(),
            "records": [_record("good"), _column_record("claim.dob", malformed=True)],
        }}}))

    def dead(req, timeout=0):
        raise OSError("verity is down")

    monkeypatch.setattr(mod.urllib.request, "urlopen", dead)
    got = mod.fetch_certified_records(_settings(tmp_path))
    assert got.available is True and [r.envelope.object_id for r in got.records] == ["good"]
    assert got.skipped == ("column:claim.dob",)


def test_the_loss_reaches_the_run_record_and_not_only_the_log(tmp_path):
    """M2's lesson: a security-relevant loss goes in the structured record. `snapshot.jobs` is
    what `mnemiq enrich` reports from and what a later run can be asked about; a `logger.warning`
    is not, and the two `certified:` jobs beside this one exist for exactly that reason."""
    from mnemiq.contract import Column, Snapshot
    from mnemiq.enrichment.certified import CertifiedSet, apply_certified_set

    snap = Snapshot(version="v1", source_id="s", created_at="2026-01-01T00:00:00Z",
                    columns=[Column(id="claim.ssn", object_id="claim", name="ssn")])
    out, _protected = apply_certified_set(snap, CertifiedSet([], skipped=("column:claim.ssn",)))

    job = next((j for j in out.jobs if j.kind == "certified_record_unreadable"), None)
    assert job is not None, "a dropped certified record must be answerable for after the run"
    assert job.status == "refused"
    assert job.checkpoints == ["column:claim.ssn"]


def test_a_clean_set_writes_no_job(tmp_path):
    from mnemiq.contract import Column, Snapshot
    from mnemiq.enrichment.certified import CertifiedSet, apply_certified_set

    snap = Snapshot(version="v1", source_id="s", created_at="2026-01-01T00:00:00Z",
                    columns=[Column(id="claim.ssn", object_id="claim", name="ssn")])
    out, _ = apply_certified_set(snap, CertifiedSet([]))
    assert not [j for j in out.jobs if j.kind == "certified_record_unreadable"]


def test_the_shared_step_reports_what_the_corpus_PINS():
    """`_protected` is the other half every caller derived from the same set, written out three
    times. It is what stops the LLM re-guessing a certified meaning, so a caller that computed it
    from a different list than it applied would silently free the difference."""
    from mnemiq.contract import Column, Snapshot
    from mnemiq.enrichment.certified import CertifiedSet, _parse, apply_certified_set

    records, _ = _parse([_column_record("claim.ssn"), _record("gloss")])
    snap = Snapshot(version="v1", source_id="s", created_at="2026-01-01T00:00:00Z",
                    columns=[Column(id="claim.ssn", object_id="claim", name="ssn")])
    _out, protected = apply_certified_set(snap, CertifiedSet(records))
    assert protected == frozenset({"claim.ssn"}), "columns only -- a definition pins no column"


# --- M82: the documented endpoint, and the refusal that explains it ---

def test_every_verity_field_names_the_same_endpoint():
    """M82. Three settings describe this one pull and two of them said `/open` while the field
    holding the URL said `/api/semantic/records` -- which takes no `since`, so it cannot serve an
    incremental pull at all. A deployment configured from the field's own description gets a 400
    on every page and `mnemiq enrich` fails outright."""
    from mnemiq.config import Settings

    described = {
        name: field.description or ""
        for name, field in Settings.model_fields.items()
        if name.startswith("verity_") and "/api/semantic/records" in (field.description or "")
    }
    assert described, "this guard is worthless if it stops finding the fields it checks"
    for name, text in described.items():
        assert "/api/semantic/records/open" in text, (
            f"{name} names the records endpoint without `/open`. Only `/open` accepts `since`, "
            "so the other one silently stops being incremental rather than failing"
        )


@pytest.mark.parametrize("body,expected,absent", [
    # The SENTENCE, not the envelope it arrived in: asserting only that the sentence is present
    # cannot tell an unwrapped detail from the raw JSON, since the raw JSON contains it too.
    ('{"detail":"limit must be between 1 and 200"}', "limit must be between 1 and 200", '{"detail"'),
    ("plain text refusal", "plain text refusal", None),
    # A proxy in front of Verity answers HTML, and a whole error page is not a diagnosis.
    ("<html>" + "x" * 4000 + "</html>", "x" * 400, "x" * 600),
])
def test_the_refusal_verity_gave_reaches_the_operator(tmp_path, monkeypatch, caplog,
                                                     body, expected, absent):
    """`require_certified` tells the operator *'the per-event log above carries the cause
    verbatim'*. For a 401 it does. For a 400 it did not: `HTTPError.__str__` is
    'HTTP Error 400: Bad Request' and the sentence Verity wrote -- which names the misconfigured
    parameter -- is in the response BODY, which was read by nothing and dropped."""
    import urllib.error

    from mnemiq.enrichment import certified as mod

    def refuse(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(body.encode()))

    monkeypatch.setattr(mod.urllib.request, "urlopen", refuse)
    with caplog.at_level("WARNING"):
        got = mod.fetch_certified_records(_settings(tmp_path))
    assert got.available is False
    assert expected in caplog.text
    if absent is not None:
        assert absent not in caplog.text
