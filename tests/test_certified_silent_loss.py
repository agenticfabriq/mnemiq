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

def test_the_endpoint_this_pull_needs_is_a_VALUE_not_a_sentence():
    """M82, and the third shape of this guard.

    The first two checked the field's prose -- `/open` appears somewhere, then `/open` appears
    first -- and each round of review found a phrasing that defeated it: a description can warn
    against the endpoint it names first, or spell a path the pattern does not match. A rule about
    how prose goes wrong is prose. So the endpoint an operator is told to set is a constant the
    description is BUILT from, and this checks the constant.
    """
    from mnemiq.enrichment.certified import CERTIFIED_RECORDS_PATH

    assert CERTIFIED_RECORDS_PATH.startswith("/") and CERTIFIED_RECORDS_PATH.endswith(
        "/records/open"), (
        f"{CERTIFIED_RECORDS_PATH!r} is not the path Verity serves the incremental pull on. Only "
        "that one accepts `since`; the sibling silently stops being incremental. `/open` alone "
        "passes a suffix check and is not a configurable path"
    )
    # NOT "the description contains the constant": the description is an f-string built FROM the
    # constant, so that assertion cannot fail, and a guard that cannot fail is worse than none.
    # What the constant is worth is checked by the two behaviour tests below.


def test_a_url_that_is_not_the_open_endpoint_is_called_out_before_the_pull(tmp_path, monkeypatch,
                                                                          caplog):
    """The check that a docstring cannot make: what the deployment actually configured.

    A description only helps somebody reading it. This is the operator who already got it wrong
    -- and whose symptom, otherwise, is a 400 on every page or, worse, a pull that drains and
    quietly stops being incremental. A WARNING and not a refusal, because the path is the
    operator's to choose and a proxy in front of Verity may legitimately serve it elsewhere.
    """
    from mnemiq.enrichment import certified as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen", _serving({
        "records": [], "watermark": "t1", "next_cursor": None}))
    from mnemiq.config import Settings
    settings = Settings(verity_records_url="https://v/api/semantic/records",
                        verity_watermark_path=str(tmp_path / "wm.json"))
    with caplog.at_level("WARNING"):
        mod.fetch_certified_records(settings)
    assert "since" in caplog.text and mod.CERTIFIED_RECORDS_PATH in caplog.text


def test_the_open_endpoint_is_not_second_guessed(tmp_path, monkeypatch, caplog):
    from mnemiq.enrichment import certified as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen", _serving({
        "records": [], "watermark": "t1", "next_cursor": None}))
    with caplog.at_level("WARNING"):
        mod.fetch_certified_records(_settings(tmp_path))
    assert "since" not in caplog.text


# `verity_page_size`'s description also warns that 0 is not a way to make the sibling endpoint
# work -- deliberately NOT asserted here. It is prose, and a substring check over prose is the
# thing three rounds of review just took apart: `"0 = send no limit at all, which is how you make
# the sibling serve a pull"` satisfies every keyword such a check could name while prescribing the
# misconfiguration. The behaviour that actually catches it is the warning below, which fires on
# the URL regardless of page size.


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


def test_a_huge_error_page_is_not_read_into_memory_to_be_thrown_away():
    """The bound that matters is on the READ, not on the log line -- truncating a string already
    in memory saves nothing. Measured on the stream itself rather than on the log, because the
    log cannot tell 4KB read from 100KB read when both print the same 500 characters. That is why
    the mutation reverting `exc.read(N)` to `exc.read()` survived every other test here."""
    import io
    import urllib.error

    from mnemiq.enrichment.certified import _REFUSAL_BODY_BYTES, _refusal_detail

    stream = io.BytesIO(b"<html>" + b"x" * 200_000 + b"</html>")
    exc = urllib.error.HTTPError("https://v/x", 400, "Bad Request", {}, stream)
    detail = _refusal_detail(exc)

    # Against a LITERAL as well as the constant: comparing only against the constant lets the
    # guard agree with itself, since raising `_REFUSAL_BODY_BYTES` to a million reads the whole
    # page and still passes. The property is "a small fraction of what was offered".
    assert stream.tell() < 50_000, (
        f"read {stream.tell()} bytes of a 200KB error page into memory to log 500 of them"
    )
    assert stream.tell() <= _REFUSAL_BODY_BYTES, (
        f"read {stream.tell()} bytes; the cap this module declares is {_REFUSAL_BODY_BYTES}"
    )
    assert "verity said" in detail, "and it still says what it managed to read"


@pytest.mark.parametrize("url,expected", [
    ("https://v/api/semantic/records/open", None),
    ("https://v/api/semantic/records/open/", None),
    # A records URL that already carries a query string is a shape `_page_url` supports -- it
    # picks its separator with `"&" if "?" in url else "?"` -- so comparing the whole URL called
    # every correct deployment of that shape misconfigured, once per run.
    ("https://v/api/semantic/records/open?tenant=acme", None),
    ("https://v/api/semantic/records", "path"),
    ("https://v/api/semantic/records?limit=200", "path"),
    # The full path, not a `/open` suffix: nothing else here separates the two, so the check
    # could accept any path ending that way while the constant's own assertion stayed green.
    ("https://v/api/semantic/definitions/open", "path"),
    # MEASURED: `_page_url` appends after the `#` and urllib cuts the request line there, so this
    # shape requests every page with no `since`, `cursor` or `limit` at all. The correct path is
    # not enough to make it a working pull, and the first version of this check called it quiet.
    ("https://v/api/semantic/records/open#frag", "fragment"),
    # A BARE `#`: `urlsplit` calls that an EMPTY fragment, which is falsy, and the parameters are
    # dropped exactly the same. Measured on the built request, not reasoned from the parse.
    ("https://v/api/semantic/records/open#", "fragment"),
    ("https://v/api/semantic/records/open?tenant=acme#", "fragment"),
])
def test_the_warning_reads_the_PATH_and_not_the_whole_url(tmp_path, monkeypatch, caplog,
                                                         url, expected):
    from mnemiq.config import Settings
    from mnemiq.enrichment import certified as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen", _serving({
        "records": [], "watermark": "t1", "next_cursor": None}))
    with caplog.at_level("WARNING"):
        mod.fetch_certified_records(Settings(
            verity_records_url=url, verity_watermark_path=str(tmp_path / "wm.json")))
    # The WORDING each cause carries, not merely that something was logged: an operator whose
    # path is already right must not be told to change the path.
    if expected is None:
        assert "verity_records_url" not in caplog.text
    else:
        assert expected in caplog.text
        assert url in caplog.text, "and the URL it is complaining about"


def test_the_fragment_and_the_wrong_path_are_told_apart():
    """One warning with two causes is the shape this register keeps closing. An operator whose
    path is right and whose URL is unusable must not be told to change the path."""
    from mnemiq.enrichment.certified import _url_complaint

    assert _url_complaint("https://v/api/semantic/records/open") is None
    assert "fragment" in _url_complaint("https://v/api/semantic/records/open#f")
    assert "path" in _url_complaint("https://v/api/semantic/records")
    assert "fragment" not in _url_complaint("https://v/api/semantic/records")

