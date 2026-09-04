from __future__ import annotations

from mnemiq.contract import IdentityContext, Trace
from mnemiq.contract.seams import Narrowed
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
        # M56: the marker travels with the list from the decider to the audit store. Defaults
        # carry "unknown", so a path that forgets to thread it records that it cannot confirm
        # rather than asserting completeness it never established.
        lineage_completeness=getattr(approved.lineage, "completeness", "unknown"),
        lineage_unresolved=list(getattr(approved.lineage, "unresolved", []) or []),
        lineage_reasons=list(getattr(approved.lineage, "reasons", []) or []),
        # Narrowing rides to the audit store on the same rule as lineage: with the answer or not
        # at all. `getattr` default None, so a decider that never evaluated governance records
        # "not evaluated" rather than the claim that it narrowed nothing.
        narrowed=[Narrowed(object=n.object, rows=n.rows, columns=n.columns)
                  for n in (getattr(approved, "narrowed", None) or [])]
        if getattr(approved, "narrowed", None) is not None else None,
        definitions_used=[],  # the glossary lands in Plan 08
    )
