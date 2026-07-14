from __future__ import annotations

from mnemiq.contract import IdentityContext, Trace
from mnemiq.sql.verdict import Approved


def build_trace(
    question: str,
    approved: Approved,
    identity: IdentityContext,
    enrichment_version: str | None,
    timing: dict[str, float],
    result_shape: str,
) -> Trace:
    """What the engine did, in the open contract's own words.

    The grading plane attaches to this by reference. It carries provenance -- never a
    judgement about the answer, which is not the engine's to make.
    """
    return Trace(
        question=question,
        plan_sql=approved.plan_sql,
        target_sql=approved.target_sql,
        result_shape=result_shape,
        timing=timing,
        enrichment_version=enrichment_version or "unknown",
        identity=identity,
        tables_used=list(approved.tables),
        definitions_used=[],  # the glossary lands in Plan 08
    )
