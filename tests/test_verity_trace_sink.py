"""The emitter sends what the tier table says and nothing else.

Asserted on the REQUEST BODY, never on what a receiver stored. Filtering at the receiver means the
bytes already crossed the wire, already sat in a log, already existed on a second machine -- "we do
not store it" and "we do not send it" are different claims and only the second survives a security
review (ui-1).
"""

from __future__ import annotations

import json
import logging
import time
import pytest
from dataclasses import dataclass, field


from mnemiq.contract.seams import IdentityContext
from mnemiq.contract.semantic import CertifiedRef
from mnemiq.observability import trace_sink as mod
from mnemiq.observability.trace_sink import AnswerEvent, VerityTraceSink


@dataclass
class _Preview:
    columns: list[str]
    rows: list[list[object]]
    row_count: int
    truncated: bool


@dataclass
class _Trace:
    target_sql: str = "select claimant from claims where name = 'Alice Smith'"
    enrichment_version: str = "v7"


@dataclass
class _Answer:
    answer: str = "Alice Smith has 3 claims"
    trace: _Trace = field(default_factory=_Trace)
    deferred: bool = False
    failed: bool = False
    reason_code: object = None
    mode: str = "agent"
    grant_fingerprint: str = "gf-abc"
    candidates_executed: int = 3
    preview: _Preview | None = None


@dataclass
class _Packet:
    definitions: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    dimensions: list = field(default_factory=list)


@dataclass
class _Obj:
    id: str


def _identity() -> IdentityContext:
    return IdentityContext(
        tenant_id="fspay",
        principal_id="idp:local:analyst",
        email="analyst@fspay.example",
        roles=["operator", "oncology_reviewer"],
        groups=["finance"],
        attributes={"cost_centre": "CC-42", "badge": "B-9911"},
    )


class _Settings:
    def __init__(self, **kw):
        self.verity_traces_url = "https://verity.example/api/traces/batch"
        self.verity_trace_send_text = False
        self.verity_trace_send_identity_detail = False
        self.verity_trace_send_rows = False
        self.verity_token_url = None
        self.verity_client_id = None
        self.verity_client_secret = None
        for k, v in kw.items():
            setattr(self, k, v)


def _event(**kw) -> AnswerEvent:
    event = AnswerEvent(
        source_id="fs_payments",
        question="how many claims does Alice Smith have",
        identity=_identity(),
        answer=kw.pop("answer", _Answer()),
        elapsed_ms=12.5,
        packet=kw.pop("packet", None),
        events=kw.pop("events", [{"stage": "retrieve", "ok": True, "ms": 4}]),
    )
    for k, v in kw.items():
        setattr(event, k, v)
    return event


def _sent(monkeypatch, settings, event) -> dict:
    """The request body as it would leave the process."""
    captured: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=0):
        captured["body"] = json.loads(request.data)
        return _Resp()

    from mnemiq.observability import trace_sink as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    VerityTraceSink(settings).record_answer(event)
    return captured["body"]["traces"][0]


def test_the_default_tier_sends_no_text_no_email_and_no_rows(monkeypatch):
    """The whole disclosure argument in one assertion. Every string below is a real leak: the
    question names a person, the SQL embeds her name as a literal, the prose restates the value,
    and `attributes` is an open dict a deployment filled."""
    answer = _Answer(preview=_Preview(["n"], [[3]], 1, False))
    body = _sent(monkeypatch, _Settings(), _event(answer=answer))
    blob = json.dumps(body)

    assert "Alice Smith" not in blob, f"a person's name left the process: {blob}"
    assert "analyst@fspay.example" not in blob, "an email address left the process"
    assert "CC-42" not in blob and "B-9911" not in blob, "deployment attributes rode along"
    assert "oncology_reviewer" not in blob, "group membership disclosed via roles"
    assert body["question"]["text"] == ""
    assert body["artifacts"] == []
    assert body.get("executed_sql") is None


def test_the_always_tier_still_answers_the_audit_question(monkeypatch):
    """Default-closed is only worth shipping if it still supports the claim the tier exists for:
    who asked, under what authorization, using which certified meaning, and what happened."""
    body = _sent(monkeypatch, _Settings(), _event())

    assert body["tenant_id"] == "fspay"
    assert body["identity"]["principal_id"] == "idp:local:analyst"
    assert body["collector_metadata"]["grant_fingerprint"] == "gf-abc"
    assert body["question"]["hash"].startswith("sha256:")
    assert body["resolved_intent"]["enrichment_version"] == "v7"
    assert body["events"], "stage events are metadata by construction and belong in every trace"


def test_opting_in_to_text_puts_the_answer_in_an_artifact_not_the_top_level(monkeypatch):
    """D132: the top-level `answer` is served unredacted while the artifact is gated, so the
    gradeable placement and the protected placement are the same one."""
    body = _sent(monkeypatch, _Settings(verity_trace_send_text=True), _event())

    assert body.get("answer") is None, "the top-level answer field is the UNPROTECTED copy"
    artifacts = {a["kind"]: a for a in body["artifacts"]}
    assert artifacts["answer_text"]["payload"]["answer"] == "Alice Smith has 3 claims"
    assert body["question"]["text"] == "how many claims does Alice Smith have"
    assert "Alice Smith" in body["executed_sql"]


def test_rows_need_their_own_opt_in(monkeypatch):
    """Text and rows are different decisions: prose restating a value is not the same disclosure as
    the customer's result set."""
    answer = _Answer(preview=_Preview(["claimant"], [["Alice Smith"]], 1, False))
    body = _sent(monkeypatch, _Settings(verity_trace_send_text=True), _event(answer=answer))

    assert not any(a["kind"] == "result_sample" for a in body["artifacts"])

    body = _sent(
        monkeypatch,
        _Settings(verity_trace_send_text=True, verity_trace_send_rows=True),
        _event(answer=_Answer(preview=_Preview(["claimant"], [["Alice Smith"]], 1, False))),
    )
    sample = next(a for a in body["artifacts"] if a["kind"] == "result_sample")
    assert sample["payload"]["rows"] == [["Alice Smith"]]


def test_refs_name_what_the_answer_used_not_what_was_available(monkeypatch):
    """`certified_refs` is SNAPSHOT-scoped and retrieval selects a subset per question. Shipping the
    snapshot's list would report what was AVAILABLE as what was USED -- and Verity's D131 reader
    would republish that as fact, which is worse than the empty list we started from."""
    packet = _Packet(definitions=[_Obj("loss_ratio")])
    event = _event(packet=packet)
    event.certified_refs = [
        CertifiedRef(object_type="definition", object_id="loss_ratio", version_hash="sha256:used"),
        CertifiedRef(object_type="definition", object_id="unused_term", version_hash="sha256:no"),
        CertifiedRef(object_type="metric", object_id="loss_ratio", version_hash="sha256:namesake"),
    ]

    body = _sent(monkeypatch, _Settings(), event)

    assert body["semantic_refs"] == [
        {"object_type": "definition", "object_id": "loss_ratio", "version_hash": "sha256:used"}
    ], "only the definition the packet selected -- not the unused one, and not its metric namesake"


def test_an_outage_does_not_fail_the_answer_and_is_counted(monkeypatch):
    """M18's lesson: fail-soft that silently produces nothing is indistinguishable from a system
    that had nothing to send."""
    import urllib.error

    from mnemiq.observability import trace_sink as mod

    def boom(request, timeout=0):
        raise urllib.error.URLError("verity is down")

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    sink = VerityTraceSink(_Settings())
    sink.record_answer(_event())  # must not raise

    assert sink.dropped == 1, "a drop that is not counted is a drop nobody can notice"


def test_no_verity_configured_sends_nothing(monkeypatch):
    from mnemiq.observability import trace_sink as mod

    def fail(request, timeout=0):  # pragma: no cover - must not be reached
        raise AssertionError("emitted with no Verity configured")

    monkeypatch.setattr(mod.urllib.request, "urlopen", fail)
    VerityTraceSink(_Settings(verity_traces_url=None)).record_answer(_event())


def test_a_deferral_sends_the_enum_and_not_the_message(monkeypatch):
    """The deferral MESSAGE is free text carrying schema and values -- "'salesperson' has no column
    'BusinessEntityID'" -- and now the provider's error string too."""

    class _Reason:
        value = "no_certified_meaning"

    answer = _Answer(
        deferred=True,
        reason_code=_Reason(),
        answer="I cannot answer: 'salesperson' has no column 'BusinessEntityID'",
    )
    body = _sent(monkeypatch, _Settings(), _event(answer=answer))

    assert body["resolved_intent"]["reason_code"] == "no_certified_meaning"
    assert "BusinessEntityID" not in json.dumps(body), "the deferral prose is opt-in, not always-on"


# --- M26: the emitter had no caller ------------------------------------------------------------
#
# The sink existed, `Runtime.trace_sink` existed, eight tests passed, and `build_runtime` never
# constructed one -- so every answer in production would have emitted nothing while the suite
# stayed green. Seventh instance in two days of a seam wired at one end, and I committed it in the
# slice that was about that species.
#
# Testing the SINK cannot catch this. The assertion has to be that the runtime HAS one.

def test_build_runtime_constructs_an_emitter_when_verity_is_configured(monkeypatch, tmp_path):
    # The assertion is that the field is WIRED from settings, not merely present on the dataclass.
    import inspect

    import mnemiq.runtime as rt

    source = inspect.getsource(rt.build_runtime)
    assert "trace_sink=trace_sink" in source, (
        "build_runtime does not pass trace_sink to Runtime -- the emitter has no caller and every "
        "answer emits nothing while its own tests pass"
    )
    assert "VerityTraceSink" in source, "build_runtime never constructs the emitter"


def test_the_emitter_is_off_unless_a_url_is_configured():
    """Byte-for-byte unchanged when Verity is not configured -- the emitter is opt-in."""
    import inspect

    import mnemiq.runtime as rt

    source = inspect.getsource(rt.build_runtime)
    assert "if settings.verity_traces_url:" in source, (
        "the emitter must be gated on configuration, not constructed unconditionally"
    )


# -- M54: one counter, three causes ---------------------------------------------------------------

# The counter told an operator that emission failed and never which of three unrelated things
# happened: a payload this engine could not BUILD (our bug, never self-heals), a request the
# receiver REJECTED (our config, permanent until someone changes it), or a server that could not be
# REACHED (their outage, self-healing). Measured live: a 401 from Verity's default auth mode
# incremented the same counter a dead socket does, so a deployment emitting zero traces forever
# looked exactly like one whose receiver was briefly down.
#
# The three want opposite responses -- fix the emitter, fix the config, wait -- which is why one
# number cannot serve them. `dropped` stays as the total, because M18's lesson is that a fail-soft
# producing nothing must be countable, and an operator alerting on "anything failing" should not
# have to sum three fields.


def _drop_counts(sink):
    return (sink.dropped, sink.dropped_unbuildable, sink.dropped_rejected, sink.dropped_unavailable)


def test_an_unbuildable_payload_is_counted_as_our_bug(monkeypatch):
    """Incremented before a request is ever sent, so it can never be an outage."""
    from mnemiq.observability import trace_sink as mod

    monkeypatch.setattr(mod.json, "dumps", lambda *a, **k: (_ for _ in ()).throw(TypeError("nope")))
    sink = VerityTraceSink(_Settings())
    sink.record_answer(_event())

    assert _drop_counts(sink) == (1, 1, 0, 0)


def test_a_rejected_request_is_counted_as_our_configuration(monkeypatch):
    """A 4xx is permanent and actionable: the receiver understood us and said no. Verity's DEFAULT
    auth mode answers 401 to the headers this sink sends, so this is the live case."""
    import urllib.error

    from mnemiq.observability import trace_sink as mod

    def unauthorized(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", unauthorized)
    sink = VerityTraceSink(_Settings())
    sink.record_answer(_event())

    assert _drop_counts(sink) == (1, 0, 1, 0)


@pytest.mark.parametrize("failure", ["url_error", "server_error"])
def test_an_unreachable_receiver_is_counted_as_their_outage(monkeypatch, failure):
    """Transport failure and a 5xx are the same class: the receiver is broken or absent, nothing
    here is wrong, and it heals without anyone acting. That is what makes it a different number
    from a 4xx rather than a different log line."""
    import urllib.error

    from mnemiq.observability import trace_sink as mod

    def boom(request, timeout=0):
        if failure == "url_error":
            raise urllib.error.URLError("verity is down")
        raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    sink = VerityTraceSink(_Settings())
    sink.record_answer(_event())

    assert _drop_counts(sink) == (1, 0, 0, 1)


def test_the_total_still_counts_every_cause(monkeypatch):
    """`dropped` is the sum, not a fourth cause. An operator alerting on "is anything failing"
    keeps one number to watch; only the operator asking "why" needs the breakdown."""
    import urllib.error

    from mnemiq.observability import trace_sink as mod

    sink = VerityTraceSink(_Settings())

    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda r, timeout=0: (_ for _ in ()).throw(
                            urllib.error.HTTPError(r.full_url, 401, "no", {}, None)))
    sink.record_answer(_event())
    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda r, timeout=0: (_ for _ in ()).throw(urllib.error.URLError("down")))
    sink.record_answer(_event())

    assert _drop_counts(sink) == (2, 0, 1, 1)


def _respond(monkeypatch, payload: bytes):
    """Answer the post with a real body, which the sink had never read.

    No `status` parameter: an earlier version took one, ignored it, and hardcoded 200 -- so a test
    written as `_respond(..., status=500)` to pin 5xx handling would have exercised the 200 path
    and passed green. The sink does not read `.status` either; a test that needs a 5xx must raise
    `HTTPError`, which is what the transport actually delivers.
    """
    class _Resp:

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        headers = {"Content-Length": str(len(payload))}

        def read1(self, *args):
            # `read1`, mirroring the real `HTTPResponse`: the sink reads only through it, because
            # `read` cannot be interrupted and so cannot be bounded.
            return self.read(*args)

        def read(self, *args):
            # Honours whatever `amt` it is handed. With `Content-Length` declared as exactly
            # `len(payload)` above, that is always the full body -- so this slice is a no-op on
            # every current test and is kept only so the stub is not WIDER than the thing it
            # stands in for. What actually catches a disabling cap is the refusal path in
            # `test_the_cap_is_large_enough_for_a_real_receipt`: a too-small cap now returns
            # before `read` is called at all, rather than truncating a body into invalid JSON as
            # it did before the declared-length check.
            amt = args[0] if args else None
            return payload if amt is None else payload[:amt]

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda request, timeout=0: _Resp())


def test_a_voided_lineage_claim_is_counted_not_swallowed(monkeypatch, caplog):
    """**D158's residue, from the producer's side.**

    Verity now answers a drifted claim with `accepted:1, lineage_voided:["..."]` -- the trace was
    kept and the claim it carried could not be read. The sink read no response body at all, so
    that 200 was byte-identical to a clean one and the report landed nowhere.

    Which is D158's own shape one repo over: the receiver was fixed to SAY it dropped something,
    and the producer still could not tell. Being told is worth nothing if nobody listens.
    """
    _respond(
        monkeypatch,
        json.dumps({
            "status": "accepted", "accepted": 1, "deduped": 0, "rejected": 0,
            "trace_ids": ["trace-abc"], "errors": [], "grading_jobs_created": 0,
            "lineage_voided": ["trace-abc"],
        }).encode(),
    )
    sink = VerityTraceSink(_Settings())
    with caplog.at_level(logging.WARNING):
        sink.record_answer(_event())

    assert sink.lineage_voided == 1, "a voided claim must be countable, like every other loss"
    assert sink.dropped == 0, "nothing was dropped: the trace is in the store"
    assert "trace-abc" in caplog.text, "the log must name the trace whose claim was voided"


def test_a_clean_acceptance_counts_no_voided_claim(monkeypatch):
    """A counter that fires on every success measures nothing."""
    _respond(
        monkeypatch,
        json.dumps({
            "status": "accepted", "accepted": 1, "deduped": 0, "rejected": 0,
            "trace_ids": ["trace-abc"], "errors": [], "grading_jobs_created": 0,
            "lineage_voided": [],
        }).encode(),
    )
    sink = VerityTraceSink(_Settings())
    sink.record_answer(_event())
    assert sink.lineage_voided == 0


def test_a_receiver_that_says_nothing_about_lineage_is_not_a_voided_claim(monkeypatch):
    """An older Verity, or any other receiver, answers without the field. Absent is not voided --
    the same distinction this whole finding is about, applied to the response instead of the claim.
    """
    _respond(monkeypatch, json.dumps({"status": "accepted", "accepted": 1}).encode())
    sink = VerityTraceSink(_Settings())
    sink.record_answer(_event())
    assert sink.lineage_voided == 0


def test_an_unreadable_response_body_does_not_fail_the_answer(monkeypatch):
    """The sink's whole contract is that Verity cannot break mnemiq. Reading the body is new
    surface for that promise to break on, so it is asserted rather than assumed."""
    _respond(monkeypatch, b"this is not json")
    sink = VerityTraceSink(_Settings())
    sink.record_answer(_event())
    assert sink.lineage_voided == 0
    assert sink.dropped == 0, "the trace was accepted; an unreadable receipt is not a drop"


@pytest.mark.parametrize(
    "body",
    [
        b'{"lineage_voided": 3}',
        b'{"lineage_voided": true}',
        b'{"lineage_voided": "trace-abc"}',
        b'{"lineage_voided": {"trace-abc": 1}}',
    ],
)
def test_a_receipt_whose_voided_field_is_not_a_list_cannot_break_the_answer(monkeypatch, body):
    """**The review gate caught a real fail-soft violation, and a probe confirmed it.**

    The guard covered only the parse; `len(voided)` and the join sat outside it. So a receipt that
    is valid JSON with a non-list `lineage_voided` escaped: `3` and `true` raised `TypeError` out
    of `record_answer` past the `except (URLError, OSError, ValueError)` whose own comment promises
    it never raises into `Runtime.ask`, and `"trace-abc"` did not raise but counted 9 and logged
    nine single characters as trace ids.

    A count-shaped value is not invented for this test: Verity writes this same field name AS A
    COUNT in its audit event, so both representations already exist in the feature and a receiver
    that ever answered with one would take mnemiq's answer path down.
    """
    _respond(monkeypatch, body)
    sink = VerityTraceSink(_Settings())

    sink.record_answer(_event())  # must not raise

    assert sink.lineage_voided == 0, "only a list of ids is a report; anything else is unreadable"
    assert sink.dropped == 0, "the trace was accepted; an unreadable receipt is not a drop"


def test_the_receipt_read_is_bounded_by_the_declared_length(monkeypatch):
    """The read asks for exactly what the receiver declared, never an open-ended slurp.

    Asserted as an EQUALITY against the declared length rather than as an upper bound. The first
    version of this test asserted only `limit <= 1 MiB`, which `_MAX_RECEIPT_BYTES = 64` satisfies
    -- so a cap that silently disabled the feature (every real receipt refused for declaring more
    than the cap allows) passed it.
    """
    body = json.dumps({"status": "accepted", "lineage_voided": ["trace-abc"]}).encode()
    seen = {}

    class _Resp:
        headers = {"Content-Length": str(len(body))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read1(self, *args):
            seen["limit"] = args[0] if args else None
            return body

        def read(self, *args):  # pragma: no cover - the sink reads only through read1
            seen["limit"] = args[0] if args else None
            return body

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda request, timeout=0: _Resp())
    sink = VerityTraceSink(_Settings())
    sink.record_answer(_event())

    assert seen["limit"] == len(body), "the read must be bounded by the declared length"
    assert sink.lineage_voided == 1, "and the receipt must still be understood"


def test_a_receipt_without_a_declared_length_is_not_waited_on(monkeypatch):
    """**A byte cap bounds bytes, not time.** `HTTPResponse.read(amt)` clips to `Content-Length`
    only when it is known; on a chunked or length-unknown body it blocks on the socket until `amt`
    bytes arrive, and `timeout=10` bounds each `recv` rather than the whole read. So a receiver
    dribbling one byte per window holds the SYNCHRONOUS answer path -- `Runtime.ask` does not
    return until `record_answer` does -- and the cap does nothing about it.

    Refusing to start removes the UNDECLARED and over-large cases. It does not bound time -- a
    receiver declaring a length within the cap can still dribble -- which is what
    `_read_within_deadline` is for; see `test_a_dribbling_receiver_cannot_hold_an_answer_open`.
    This pins the refusal, not the deadline.
    """
    reads: list = []

    class _Resp:
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read1(self, *args):
            reads.append(args)
            return b"{}"

        def read(self, *args):
            reads.append(args)
            return b"{}"

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda request, timeout=0: _Resp())
    sink = VerityTraceSink(_Settings())

    sink.record_answer(_event())

    assert reads == [], (
        "the body must not be read at all. Asserted by RECORDING rather than by raising: "
        "`_note_voided_claims` catches Exception, so a raising stub is swallowed and the test "
        "passes whether or not the read happened"
    )
    assert sink.lineage_voided == 0
    assert sink.dropped == 0, "declining to read a receipt is not a dropped trace"


def test_a_receipt_larger_than_the_cap_is_not_read(monkeypatch):
    """The same refusal, for a length that IS declared and is implausible for a batch receipt."""
    reads: list = []

    class _Resp:
        headers = {"Content-Length": str(64 * 1024 * 1024)}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read1(self, *args):
            reads.append(args)
            return b"{}"

        def read(self, *args):
            reads.append(args)
            return b"{}"

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda request, timeout=0: _Resp())
    sink = VerityTraceSink(_Settings())

    sink.record_answer(_event())

    assert reads == [], "a body larger than the cap must not be read"
    assert sink.lineage_voided == 0
    assert sink.dropped == 0


def test_the_cap_is_large_enough_for_a_real_receipt(monkeypatch):
    """Bounds the cap from BELOW. The previous version asserted only `<= 1 MiB`, so setting the cap
    to 64 bytes kept every test green while refusing every real receipt for declaring more than
    the cap allows -- `lineage_voided` would read 0 forever and the feature would silently do
    nothing. A cap that disables the feature must fail a test."""
    ids = [f"trace-{i:04d}" for i in range(50)]
    body = json.dumps({"status": "accepted", "accepted": 50, "lineage_voided": ids}).encode()
    assert len(body) > 700, "a 50-trace receipt should be substantial enough to be a real probe"

    _respond(monkeypatch, body)
    sink = VerityTraceSink(_Settings())
    sink.record_answer(_event())

    assert sink.lineage_voided == 50, (
        "a plausible real receipt must survive the cap intact; if this fails, "
        "_MAX_RECEIPT_BYTES is too small and the feature is silently disabled"
    )


@pytest.mark.parametrize("declared", ["-1", "-65537", "not-a-number", ""])
def test_a_nonsensical_declared_length_is_refused(monkeypatch, declared):
    """**A BLOCKER from the review gate: the previous version traded a hard cap for trust.**

    `read(_MAX_RECEIPT_BYTES)` capped memory on every response unconditionally. Replacing it with
    `read(declared)` put the receiver's own number in that position, and only the UPPER bound was
    checked -- so `Content-Length: -1` passed `-1 > 65536` and `read(-1)` takes `HTTPResponse`'s
    unbounded branch: `http/client.py` sets `self.length = None` for a negative header and falls
    through to `self.fp.read()`, which reads until EOF. A receiver declaring `-1` and dribbling
    would get an unbounded, uncapped buffer on the synchronous answer path.

    The lesson is narrower than "validate input": a guard that REPLACES a hard limit with a
    negotiated one has to re-establish every bound the hard limit gave for free, not just the one
    it was written to add.
    """
    reads: list = []

    class _Resp:
        headers = {"Content-Length": declared}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read1(self, *args):
            reads.append(args)
            return b"{}"

        def read(self, *args):
            reads.append(args)
            return b"{}"

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda request, timeout=0: _Resp())
    sink = VerityTraceSink(_Settings())

    sink.record_answer(_event())

    assert reads == [], f"a declared length of {declared!r} must not reach read()"
    assert sink.lineage_voided == 0
    assert sink.dropped == 0


def _dribbling_server(declared: int, per_byte: float) -> int:
    """A real HTTP server that declares a length and then trickles. Returns its port.

    A REAL server, not a stub, because a stub is what hid this defect twice. The first probe
    returned one byte per `read` call -- a shape a non-chunked `HTTPResponse` never produces -- so
    it exercised many loop iterations that the real dependency collapses into a single blocking
    call, and the mutation `_RECEIPT_CHUNK_BYTES = 64 * 1024` left it green. A stub WIDER than the
    thing it stands in for cannot observe a defect that lives in the narrowness.
    """
    import socket
    import threading

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve() -> None:
        try:
            conn, _ = srv.accept()
            conn.recv(65536)
            conn.sendall(f"HTTP/1.1 200 OK\r\nContent-Length: {declared}\r\n\r\n".encode())
            for _ in range(declared):
                conn.sendall(b"x")
                time.sleep(per_byte)
            conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return port


def test_a_dribbling_receiver_cannot_hold_an_answer_open(monkeypatch):
    """**Codex adversarial review, [high], then a review-gate BLOCKER on the first fix.**

    `Runtime.ask` does not return until `record_answer` does, and urllib's `timeout` is per socket
    OPERATION, not a total deadline -- so a receiver that declares an allowed length and then
    dribbles can hold an answer open and exhaust request workers after already accepting the trace.
    The exposure was NEW: before the receipt was read at all, the sink returned once headers
    arrived.

    The first fix was a deadline checked between `read()` calls, and it did nothing, because
    `HTTPResponse.read(amt)` does not return short -- it blocks until `amt` bytes arrive, so a
    receipt inside the cap is consumed in ONE call and the budget is never re-checked. Measured
    against this very server: `read` returned all 100 bytes after 5.60s under a 1.0s budget;
    `read1`, which returns after a single underlying read, returned 19 bytes after 1.01s.

    So this asserts against a real socket, and it is the mutation-checked shape: setting
    `_RECEIPT_CHUNK_BYTES` to the whole cap, or reverting `read1` to `read`, must fail it.
    """
    monkeypatch.setattr(mod, "_RECEIPT_DEADLINE_SECONDS", 0.5)
    port = _dribbling_server(declared=200, per_byte=0.05)  # 10s if read to completion
    sink = VerityTraceSink(
        _Settings(verity_traces_url=f"http://127.0.0.1:{port}/api/traces/batch")
    )

    started = time.monotonic()
    sink.record_answer(_event())
    elapsed = time.monotonic() - started

    assert elapsed < 3.0, (
        f"the answer path must not wait on a dribbling receiver; took {elapsed:.2f}s. "
        "Reading to completion would be ~10s."
    )
    assert sink.lineage_voided == 0, "an abandoned receipt reports nothing, like an older Verity"
    assert sink.dropped == 0, "the trace was accepted; abandoning its receipt is not a drop"
