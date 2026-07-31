"""AgentAnswer -> wire dict. Shared by /v1/ask and the SSE CUSTOM event."""

from __future__ import annotations

from mnemiq.agent.loop import AgentAnswer


def answer_payload(ans: AgentAnswer) -> dict:
    t = ans.trace
    p = ans.preview
    return {
        "answer": ans.answer,
        "deferred": ans.deferred,
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
