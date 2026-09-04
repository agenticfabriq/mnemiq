"""AgentAnswer -> wire dict. Shared by /v1/ask and the SSE CUSTOM event."""

from __future__ import annotations

from mnemiq.agent.loop import AgentAnswer


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
