"""Replay the verifier over a saved mini_dev run -- no engine re-runs.

Deterministic layers (default): print the wrong-caught vs correct-lost trade for sanity /
grounding / both.

  .venv/bin/python scripts/run_verify_replay.py eval-reports/minidev-pg-14b-guided-fixed.jsonl

Judge sweep (--judge): score each case with an LLM judge and sweep thresholds. The judge endpoint
is MNEMIQ_VERIFY_BASE_URL/MODEL/API_KEY (hosted gpt-5.5 ceiling) or falls back to MNEMIQ_LLM_*
(the served local model). Schema cards are rebuilt per db from the cached enrichment.

  MNEMIQ_VERIFY_BASE_URL=... MNEMIQ_VERIFY_MODEL=openai.gpt-5.5 \
    .venv/bin/python scripts/run_verify_replay.py <run.jsonl> --judge
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os

from mnemiq.config import Settings
from mnemiq.eval.bird_runner import enrich_bird_db
from mnemiq.eval.verify_replay import judge_scores, load_records, replay, sweep
from mnemiq.llm.client import LLMClient
from mnemiq.semantic.cards import build_cards
from mnemiq.verify.judge import SemanticJudge
from mnemiq.verify.verifier import Verifier

_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class _CachingJudge:
    """Wrap a judge with a persistent per-case score cache so a wall-clock kill mid-sweep doesn't
    re-pay for hosted calls -- a re-run resumes. Keyed on (question, sql, model)."""

    def __init__(self, judge, path: str, model: str) -> None:
        self._judge = judge
        self._path = path
        self._model = model
        self._cache: dict[str, float] = {}
        if os.path.exists(path):
            self._cache = json.load(open(path))

    def score(self, question: str, schema: str, sql: str, preview: str) -> float:
        key = hashlib.sha1(f"{self._model}\x00{question}\x00{sql}".encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        s = self._judge.score(question, schema, sql, preview)
        self._cache[key] = s
        json.dump(self._cache, open(self._path, "w"))
        return s


def _line(o: dict) -> str:
    return (f"  wrong-caught {o['wrong_caught']}/{o['wrong_before']}  "
            f"correct-lost {o['correct_lost']}/{o['correct_before']}  "
            f"EX {o['ex_before']:.1%}->{o['ex_after']:.1%}  "
            f"wrong {o['wrong_before_rate']:.1%}->{o['wrong_after_rate']:.1%}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--minidev", default=os.environ.get(
        "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/MINIDEV")))
    p.add_argument("--cache", default="eval-reports/minidev-pg-cache")
    args = p.parse_args()

    records = load_records(args.run)
    print(f"loaded {len(records)} records from {args.run}")

    if not args.judge:
        for name, vf in (("sanity+grounding", Verifier(grounding=True)),
                         ("sanity-only (default)", Verifier()),
                         ("grounding-only", Verifier(sanity=False, grounding=True))):
            print(f"\n== {name} ==\n{_line(replay(records, vf))}")
        return 0

    settings = Settings.from_env()
    base = os.getenv("MNEMIQ_VERIFY_BASE_URL") or settings.llm_base_url
    key = os.getenv("MNEMIQ_VERIFY_API_KEY") or settings.llm_api_key
    model = os.getenv("MNEMIQ_VERIFY_MODEL") or settings.llm_model
    client = LLMClient(settings.model_copy(update={"llm_base_url": base, "llm_api_key": key, "llm_model": model}))
    tag = "".join(ch if ch.isalnum() else "_" for ch in model)
    judge = _CachingJudge(SemanticJudge(client), f"{args.run}.judgecache.{tag}.json", model)

    @functools.lru_cache(maxsize=None)
    def cards_for(db_id: str) -> str:
        if not db_id:
            return ""
        snap = enrich_bird_db(args.minidev, db_id, settings, cache_dir=args.cache)
        return "\n".join(c.text for c in build_cards(snap))

    print(f"judge endpoint: {base} | model: {model}")
    inner = judge._judge  # the instrumented SemanticJudge sitting under the cache wrapper
    scores = judge_scores(records, judge, cards_for)

    # How many of those "scores" are the fail-open constant rather than a judgement. Nothing in the
    # cache can answer this: `SemanticJudge` clamps a real reply to 1.0 and returns 1.0 on error, so
    # a dead endpoint and a judge that approved everything write identical files. Only the judge's
    # own counters separate them, and a sweep that cannot be told from an outage must not be
    # published as a measurement. Written beside the cache; counts THIS process's calls, so a run
    # resumed over a warm cache reports only what it re-scored.
    errpath = f"{args.run}.judgeerrors.{tag}.json"
    json.dump({"model": model, "calls": inner.calls, "errors": inner.errors,
               "unparsed": inner.unparsed, "fallbacks": inner.fallbacks,
               "scored_cases": len(scores)}, open(errpath, "w"), indent=2)
    print(f"judge calls {inner.calls}: {inner.errors} endpoint errors, {inner.unparsed} unreadable "
          f"replies -> {inner.fallbacks} fail-open constants (recorded in {errpath})")
    if inner.fallbacks:
        print("WARNING: some scores are the fail-open constant, not judgements. Any figure taken "
              "from this sweep is contaminated by that many cases.")

    print(f"scored {len(scores)} answerable cases; sweeping thresholds:")
    for row in sweep(records, scores, _THRESHOLDS):
        print(f"thr {row['threshold']:.1f}:{_line(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
