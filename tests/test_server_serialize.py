import dataclasses

from mnemiq.agent.loop import AgentAnswer, ResultPreview
from mnemiq.contract import DeferralReason, IdentityContext, Trace
from mnemiq.server.serialize import answer_payload


def _trace():
    ident = IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])
    return Trace(question="q", plan_sql="SELECT 1", target_sql="SELECT count(*) FROM claim",
                 result_shape="scalar", timing={"total_ms": 5.0}, enrichment_version="v1",
                 identity=ident, tables_used=["claim"])


def test_full_answer_payload():
    p = ResultPreview(columns=["n"], rows=[[2]], row_count=1, truncated=False)
    out = answer_payload(AgentAnswer(answer="2.", trace=_trace(), cached=True,
                                     agreement=0.8, mode="deep", preview=p))
    assert out["answer"] == "2."
    assert out["sql"] == "SELECT count(*) FROM claim"
    assert out["tables_used"] == ["claim"]
    assert out["timing"] == {"total_ms": 5.0}
    assert out["cached"] is True and out["agreement"] == 0.8 and out["mode"] == "deep"
    assert out["preview"]["rows"] == [[2]]


def test_deferral_payload_has_no_sql_no_preview():
    out = answer_payload(AgentAnswer(answer="No table holds salary.", deferred=True))
    assert out["deferred"] is True
    assert out["sql"] is None and out["preview"] is None and out["tables_used"] is None


# ---------------------------------------------------------------------------------------------
# M14 -- M6 split `failed` from `deferred`, and this serializer never learned. A failed answer
# went out as `deferred: false` with no `failed` key, so at the product surface an outage was
# indistinguishable from a successful answer -- strictly worse than the conflation M6 removed.
# ---------------------------------------------------------------------------------------------


def test_a_failed_answer_is_distinguishable_from_an_answer():
    out = answer_payload(
        AgentAnswer(
            answer="Could not answer this question: the database rejected every attempt.",
            failed=True,
            reason_code=DeferralReason.EXECUTION_FAILED,
        )
    )

    assert out["failed"] is True, (
        "before this, a source outage serialized as deferred:false with no failed key -- "
        "a caller checking `deferred` saw false and had every reason to call it an answer"
    )
    assert out["deferred"] is False
    assert out["reason_code"] == "execution_failed"


def test_a_deferral_carries_the_reason_the_deferral_card_needs():
    out = answer_payload(
        AgentAnswer(
            answer="Answering this would require access to 'salary'.",
            deferred=True,
            reason_code=DeferralReason.AUTHORIZATION,
        )
    )

    assert out["deferred"] is True and out["failed"] is False
    assert out["reason_code"] == "authorization", (
        "the UI section asks the DeferralCard to render a D5 reason state; the wire has to carry it"
    )


def test_a_plain_answer_says_so_in_both_fields():
    out = answer_payload(AgentAnswer(answer="2.", trace=_trace()))

    assert out["failed"] is False
    assert out["reason_code"] is None


def test_the_field_is_spelled_the_way_mcp_spells_it():
    # One spelling across MCP, /v1/ask and SSE. `deferral_reason` would also be wrong on its own
    # terms: EXECUTION_FAILED is not a deferral, and naming the field after one would re-merge the
    # two states M6 separated.
    out = answer_payload(AgentAnswer(answer="x", deferred=True,
                                     reason_code=DeferralReason.NO_TABLES))

    assert "reason_code" in out and "deferral_reason" not in out


# =============================================================================================
# CROSS-THREAD GUARD -- read this before adding a field to AgentAnswer.
#
# M14 happened because two threads were both correct: the engine thread gave AgentAnswer
# `failed` and `reason_code`, and the server thread's serializer -- written days earlier against
# an engine that had neither -- kept emitting the fields it knew about. Nobody was wrong, and a
# source outage went out looking like a successful answer for a day.
#
# Worktrees isolate code, not contracts, and the two threads do not share session memory. So the
# seam is enforced here instead of by anyone remembering: add a field to AgentAnswer and this
# test fails until you either put it on the wire or say out loud why it stays off.
# =============================================================================================

# Fields that reach the wire under other names, and the keys they become. `trace` is exploded
# rather than nested because the wire is flat by design (ui-eval-unify §5).
_EXPLODED: dict[str, tuple[str, ...]] = {
    "trace": ("sql", "tables_used", "enrichment_version", "timing"),
}

# Fields deliberately kept off the wire. Empty today: everything AgentAnswer carries is something
# a caller can act on. Add here ONLY with a reason -- an entry is a decision, not a shortcut.
_WITHHELD: dict[str, str] = {
    # Engine telemetry, added so an eval can ask whether the verifier's score RANKS wrong
    # answers below right ones. Deliberately not on the wire: a bare "0.62" beside an answer
    # reads to a user as a probability the answer is correct, which is not what it is and not
    # something we have shown it to be -- measured once, the judge caught 4 and killed 4.
    # Surfacing it is a UI decision with its own argument, not a side effect of measuring.
    "verify_confidence": "engine telemetry; a raw score beside an answer would read as an "
                         "accuracy claim we have not earned",
    "verify_layer": "engine telemetry; meaningless to a user without the confidence -- but see "
                    "`verified` on the payload, which is DERIVED from it and does ship: whether the "
                    "check ran needs no score to be meaningful, and is the half a reader is worse "
                    "off not knowing",
}


def test_every_agentanswer_field_reaches_the_wire_or_is_explicitly_withheld():
    payload = answer_payload(
        AgentAnswer(answer="x", trace=_trace(), preview=None)
    )

    missing = []
    for field in dataclasses.fields(AgentAnswer):
        name = field.name
        if name in _WITHHELD:
            continue
        if name in _EXPLODED:
            absent = [key for key in _EXPLODED[name] if key not in payload]
            if absent:
                missing.append(f"{name} -> {absent} (declared exploded, but those keys are absent)")
            continue
        if name not in payload:
            missing.append(name)

    assert not missing, (
        f"AgentAnswer fields that reach no caller: {missing}.\n"
        "Add them to answer_payload (it is the shared serializer for /v1/ask and the SSE CUSTOM "
        "event, so one edit covers both), or add them to _EXPLODED if they arrive under another "
        "name, or to _WITHHELD with a reason. Do not delete this assertion -- it exists because "
        "M14 shipped a failure that looked like a success for a day."
    )


def test_the_guard_notices_a_field_that_stops_being_serialized():
    # A guard that cannot fail is the register's pattern 6. This proves this one can, by asking
    # it about a field the payload genuinely does not have.
    payload = answer_payload(AgentAnswer(answer="x"))

    assert "not_a_real_field" not in payload
    assert "reason_code" in payload, "and that it is looking at the real payload"


def test_exploded_and_withheld_name_only_real_fields():
    # Otherwise a renamed field leaves a stale exemption behind that silently excuses its
    # replacement -- the register's own hand-maintained-mirror failure, inside the guard.
    known = {f.name for f in dataclasses.fields(AgentAnswer)}
    stale = (set(_EXPLODED) | set(_WITHHELD)) - known
    assert not stale, f"exemptions for fields that no longer exist: {sorted(stale)}"


def test_the_wire_carries_what_the_mode_actually_spent():
    """M33: `instant` and `thinking` run identical code whenever the first SQL is approved,
    so without these two the mode control is unfalsifiable from the outside."""
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.server.serialize import answer_payload

    body = answer_payload(AgentAnswer(answer="7.", mode="thinking", attempts=2, corrected=True))
    assert body["attempts"] == 2
    assert body["corrected"] is True


def test_an_answer_that_never_planned_reports_neither_rather_than_zero():
    """A deferral did no attempts; saying `0` would claim it measured something."""
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.server.serialize import answer_payload

    body = answer_payload(AgentAnswer(answer="No.", deferred=True))
    assert body["attempts"] is None and body["corrected"] is None
