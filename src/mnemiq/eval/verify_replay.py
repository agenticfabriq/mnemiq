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


def _render(rows: list[dict]) -> str:
    from mnemiq.execute.render import render_result

    return render_result(_table(rows), max_rows=5)


def judge_scores(records: list[dict], judge, cards_for) -> list[float]:
    """Score every answerable record once (deterministic layers off) so a threshold sweep is free.
    `cards_for(db_id) -> str` supplies the schema text the judge sees."""
    out: list[float] = []
    for r in records:
        if r["outcome"] not in _ANSWERABLE:
            continue
        preview = _render(r.get("engine_rows") or [])
        out.append(judge.score(r["question"], cards_for(r.get("db_id", "")), r.get("sql") or "", preview))
    return out


def _answerable(records: list[dict]) -> list[dict]:
    """The records a verifier can act on, in the ONE order every per-record list here uses.
    `judge_scores`, `layer_defers` and `tally` all walk this, which is what lets a judge score and
    a deterministic deferral for the same case be combined by position."""
    return [r for r in records if r["outcome"] in _ANSWERABLE]


def tally(records: list[dict], defers: list[bool]) -> dict:
    """Turn one deferral decision per answerable record into the wrong-caught/correct-lost trade.

    Every trade reported by this module is counted here, once. An earlier version counted the
    deterministic layers inside `replay` and the judge sweep inside `sweep`, which was fine until a
    THIRD caller wanted both layers at once: a combined row counted by a third copy of this loop
    can disagree with the two rows it is compared against, and the disagreement looks like a
    finding about the layers rather than about the counting."""
    ans = _answerable(records)
    if len(defers) != len(ans):
        raise ValueError(f"{len(defers)} deferrals for {len(ans)} answerable records")
    n = len(ans)
    wrong_before = sum(r["outcome"] == "wrong" for r in ans)
    correct_before = sum(r["outcome"] == "correct" for r in ans)
    wc = cl = ca = wa = 0
    for r, deferred in zip(ans, defers):
        if r["outcome"] == "wrong":
            wc += deferred
            wa += not deferred
        elif r["outcome"] == "correct":
            cl += deferred
            ca += not deferred
    return {
        "answerable": n, "wrong_before": wrong_before, "correct_before": correct_before,
        "wrong_caught": wc, "correct_lost": cl,
        "ex_before": correct_before / n if n else 0.0, "ex_after": ca / n if n else 0.0,
        "wrong_before_rate": wrong_before / n if n else 0.0, "wrong_after_rate": wa / n if n else 0.0,
    }


def sweep(records: list[dict], judge_score_list: list[float], thresholds,
          also_defer: list[bool] | None = None) -> list[dict]:
    """Given a fixed judge score per answerable record, compute the wrong-caught/correct-lost trade
    at each threshold (defer when score < threshold). One tally() dict per threshold.

    `also_defer` adds a deterministic layer running BESIDE the judge: a case is deferred if either
    catches it. A combined figure is therefore not the sum of the two separate ones -- they
    overlap, and reading a published combined row as the sum is what made the number card's
    `sanity + judge` line irreproducible.

    The OR matches the product, which SHORT-CIRCUITS instead: `Verifier.verify` returns on the
    first layer that produces a verdict and never reaches the judge. The two agree only because
    every verdict `sanity_check` can return has `defer=True` -- a sanity layer that could return an
    APPROVAL would make the short-circuit and the OR disagree, and this row would quietly stop
    describing the product. `test_sanity_verdicts_all_defer` pins that."""
    ans = _answerable(records)
    if len(judge_score_list) != len(ans):
        raise ValueError(f"{len(judge_score_list)} judge scores for {len(ans)} answerable records")
    extra = also_defer if also_defer is not None else [False] * len(ans)
    return [{"threshold": t,
             **tally(records, [s < t or e for s, e in zip(judge_score_list, extra)])}
            for t in thresholds]


def layer_defers(records: list[dict], verifier: Verifier) -> list[bool]:
    """One deterministic deferral decision per answerable record, positionally aligned with
    `judge_scores` so the two can be combined."""
    out: list[bool] = []
    for r in _answerable(records):
        packet = ContextPacket(question=r["question"], cards=[], grant_fingerprint="", enrichment_version=None)
        approved = Approved(plan_sql=r.get("sql") or "", target_sql=r.get("sql") or "")
        out.append(verifier.verify(packet, approved, _table(r.get("engine_rows") or [])).defer)
    return out


def replay(records: list[dict], verifier: Verifier) -> dict:
    """Score a Verifier over saved eval records without re-running the engine. A verifier only
    ever turns an answer into a deferral (never fixes it), so ex_after <= ex_before."""
    return tally(records, layer_defers(records, verifier))
