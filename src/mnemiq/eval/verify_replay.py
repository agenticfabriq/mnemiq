from __future__ import annotations

import json

import pyarrow as pa

from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved
from mnemiq.verify.verifier import Verifier

_ANSWERABLE = {"correct", "correct_facts", "wrong", "deferred_wrongly", "error"}


def _table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows) if rows else pa.table({})


def load_records(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def replay(records: list[dict], verifier: Verifier) -> dict:
    """Score a Verifier over saved eval records without re-running the engine. A verifier only
    ever turns an answer into a deferral (never fixes it), so ex_after <= ex_before."""
    ans = [r for r in records if r["outcome"] in _ANSWERABLE]
    n = len(ans)
    wrong_before = sum(r["outcome"] == "wrong" for r in ans)
    correct_before = sum(r["outcome"] == "correct" for r in ans)
    wrong_caught = correct_lost = correct_after = wrong_after = 0
    for r in ans:
        packet = ContextPacket(question=r["question"], cards=[], grant_fingerprint="", enrichment_version=None)
        approved = Approved(plan_sql=r.get("sql") or "", target_sql=r.get("sql") or "")
        deferred = verifier.verify(packet, approved, _table(r.get("engine_rows") or [])).defer
        if r["outcome"] == "wrong":
            wrong_caught += deferred
            wrong_after += not deferred
        elif r["outcome"] == "correct":
            correct_lost += deferred
            correct_after += not deferred
    return {
        "answerable": n,
        "wrong_before": wrong_before,
        "correct_before": correct_before,
        "wrong_caught": wrong_caught,
        "correct_lost": correct_lost,
        "ex_before": correct_before / n if n else 0.0,
        "ex_after": correct_after / n if n else 0.0,
        "wrong_before_rate": wrong_before / n if n else 0.0,
        "wrong_after_rate": wrong_after / n if n else 0.0,
    }
