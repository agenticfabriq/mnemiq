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
        "cached": ans.cached,
        "agreement": ans.agreement,
        "judge_engaged": ans.judge_engaged,
        "judge_override": ans.judge_override,
        "candidates_executed": ans.candidates_executed,
        "sql": t.target_sql if t else None,
        "tables_used": list(t.tables_used) if t else None,
        "enrichment_version": t.enrichment_version if t else None,
        "timing": t.timing if t else None,
        "preview": ({"columns": p.columns, "rows": p.rows, "row_count": p.row_count,
                     "truncated": p.truncated} if p else None),
    }
