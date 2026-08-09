"""The emitter sends what the tier table says and nothing else.

Asserted on the REQUEST BODY, never on what a receiver stored. Filtering at the receiver means the
bytes already crossed the wire, already sat in a log, already existed on a second machine -- "we do
not store it" and "we do not send it" are different claims and only the second survives a security
review (ui-1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from mnemiq.contract.seams import IdentityContext
from mnemiq.contract.semantic import CertifiedRef
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
