"""AgentAnswer -> wire dict. Shared by /v1/ask and the SSE CUSTOM event."""

from __future__ import annotations

from mnemiq.agent.loop import AgentAnswer


# M89. NOT the confidence and NOT the layer name -- both stay withheld, and the argument for
# withholding them is untouched: a bare "0.62" beside an answer reads as an accuracy claim we have
# not earned, and a layer name means nothing without it. What ships is the one thing that is
# meaningful with no score attached and that a reader is worse off not knowing -- WHAT KIND of
# check ran, if any.
#
# Graded, not boolean, because "checked" would collapse two very different guarantees. `instant`
# and `thinking` -- and `thinking` is the default mode -- run the deterministic net only, which
# rejects empty and null results and never reads whether the answer is RIGHT. Serialising that
# identically to a judge-approved `deep` answer would make exactly the accuracy claim the withheld
# confidence exists to refuse, on the modes that get the most traffic.
_VERIFIED_STATE = {
    "judge": "judged",                  # a judge scored this answer
    "judge_unavailable": "unavailable",  # a judge was configured and could not be reached
    "sanity": "basic",                  # deterministic net only; says nothing about correctness
    "grounding": "basic",
    "pass": "basic",
}


def verified_state(layer: str | None) -> str | None:
    """What kind of verification this answer actually received, for the wire.

    `None` means NO verifier read produced a stamp, and it has several causes that a client must
    not conflate with "this mode does not verify": a deferral raised before verification, an
    execution failure, and a mode with no verifier all leave the field unset because the verifier
    never saw a table.

    FAILS CLOSED on an unrecognised layer, to `"unknown"` and NOT to `"unavailable"`. The layer
    vocabulary is a comment, not a type, so a future layer meaning "no judgement happened" -- the
    shape `judge_unavailable` itself had before it existed -- must not ship as though it were a
    check; defaulting to a check is how M89 stayed invisible. But `"unavailable"` means one
    specific thing an operator may page on, a judge that could not be reached, and folding an
    unrecognised layer into it buys a false outage signal. `"unknown"` says the true thing: this
    engine emitted a verification state this serializer has not been taught.
    """
    if layer is None:
        return None
    return _VERIFIED_STATE.get(layer, "unknown")


def answer_payload(ans: AgentAnswer) -> dict:
    t = ans.trace
    p = ans.preview
    return {
        "answer": ans.answer,
        "deferred": ans.deferred,
        # M14: M6 made `failed` a terminal state of its own, so a source outage carries
        # deferred=False. Without these two keys it went out looking like a successful answer --
        # worse than the conflation M6 removed. `reason_code` is spelled the way MCP spells it:
        # one name across three surfaces, and `deferral_reason` would be wrong anyway, since
        # EXECUTION_FAILED is not a deferral.
        "failed": ans.failed,
        "reason_code": str(ans.reason_code) if ans.reason_code else None,
        "mode": ans.mode,
        "grant_fingerprint": ans.grant_fingerprint,
        "cached": ans.cached,
        "agreement": ans.agreement,
        "verified": verified_state(ans.verify_layer),
        "judge_engaged": ans.judge_engaged,
        "judge_override": ans.judge_override,
        "candidates_executed": ans.candidates_executed,
        # M33: what the mode actually spent. Without these, `instant` and `thinking` are
        # indistinguishable on a question that succeeds first time -- which is most of them.
        "attempts": ans.attempts,
        "corrected": ans.corrected,
        "sql": t.target_sql if t else None,
        "tables_used": list(t.tables_used) if t else None,
        # M56: the list never ships without the marker. A bare `tables_used` is the misleading
        # artifact this engine now refuses to produce -- `[]` reads as "nothing was read" where
        # the truth may be "read through a function we could not classify". Emitting the list
        # here and the marker only in the Verity trace would have recreated it on the two
        # surfaces a customer actually reads.
        "lineage": ({"tables": list(t.tables_used),
                     "completeness": t.lineage_completeness,
                     "unresolved": list(t.lineage_unresolved),
                     "reasons": list(t.lineage_reasons)} if t else None),
        # Structured beside the sentence that is already in `answer`. `null` distinguishes "not
        # evaluated" from the `[]` that claims nothing was narrowed.
        "narrowed": ([n.model_dump() for n in t.narrowed]
                     if t and t.narrowed is not None else None),
        "enrichment_version": t.enrichment_version if t else None,
        "timing": t.timing if t else None,
        "preview": ({"columns": p.columns, "rows": p.rows, "row_count": p.row_count,
                     "truncated": p.truncated} if p else None),
    }
