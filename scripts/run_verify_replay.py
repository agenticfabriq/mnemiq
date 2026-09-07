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
import time

from mnemiq.config import Settings
from mnemiq.eval.bird_runner import enrich_bird_db
from mnemiq.eval.verify_replay import (_ANSWERABLE, judge_scores, layer_defers, load_records,
                                       replay, sweep)
from mnemiq.llm.client import LLMClient
from mnemiq.semantic.cards import build_cards
from mnemiq.verify.judge import JudgeRead, SemanticJudge
from mnemiq.verify.verifier import Verifier

_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class _RetryingJudge:
    """Retry a judge call that FAILED, which its score alone cannot distinguish from one that
    scored 1.0.

    The product's judge fails open on the first error, deliberately -- a dead judge must not stop
    an answer. A MEASUREMENT wants the opposite: a fail-open score is not a judgement, and one
    flaky call should not become a data point. This sits between the cache and the judge and tries
    again with backoff.

    It learns what happened from `JudgeRead`, which the inner judge returns WITH the score. An
    earlier design read the inner judge's `errors`/`unparsed` counters before and after each call
    and treated the delta as the signal -- and this docstring described that for three commits
    after it was removed. Those counters are cumulative and the judge is shared, so under
    concurrency the delta includes another caller's failure: the call retries a judgement it
    already had, and can report a fail-open for an answer that was judged. Nothing here reads them
    now, and the constructor refuses a judge that cannot answer `read`.
    """

    # Retries exist because this is a network call, and for no stronger reason than that. An
    # earlier version of this comment claimed the window had to outlast an OUTAGE, and widened the
    # backoff to 8 on the strength of sweeps that each lost one case -- which that comment read
    # as a different case each time, and which later turned out to be the same hard case. That
    # explanation was wrong. Raising the reasoning reserve to 4096 took endpoint errors from
    # 7, 6 and 4 per ~490 calls to 0 in 487 -- under the old rate the chance of a clean run is
    # 0.4% -- so every "endpoint error" measured here was this code truncating its own request,
    # not the provider faltering. No genuine outage was ever demonstrated, and a window sized for
    # one would be sized for nothing.
    #
    # 1.5 restored deliberately. The wider value cost nothing on the success path, which is exactly
    # why it could have stayed: an unfalsifiable comfort, justified by a story already known to be
    # false.
    def __init__(self, judge, attempts: int = 4, backoff: float = 1.5) -> None:
        # `attempts < 1` makes the loop body never run, so every score is the fail-open constant
        # and the judge is never called at all -- `calls` stays 0. It would NOT certify silently
        # (control falls through to `gave_up += 1`, and the gate refuses on any non-zero
        # `unrecovered` -- not on its magnitude, which the cache in front of this wrapper makes a
        # distinct-pair count rather than a record count); the reason to refuse it HERE is that a
        # sweep whose judge was never invoked
        # is not a measurement, and finding that out after paying for the run and reading a
        # contamination refusal tells the operator the wrong thing about why.
        # Enforced in the constructor rather than at the arg parser so no caller can reach the
        # state -- though `_attempts` stays rebindable afterwards; this is a construction check,
        # not an invariant.
        if attempts < 1:
            raise ValueError(f"attempts must be >= 1, got {attempts}: fewer means the judge is "
                             "never called and every score is the fail-open constant")
        # Checked HERE, beside the `attempts` guard above and for its stated reason: so no caller
        # can reach the state. `read` is the only way this wrapper learns whether a call produced a
        # judgement -- deriving it from the shared judge's counters is the race this class was
        # fixed twice to stop -- and raising at call time would abort a sweep only after
        # `cards_for` had paid for that database's enrichment. `FakeJudge` has just `score`.
        if not hasattr(judge, "read"):
            raise TypeError(
                f"{type(judge).__name__} has no `read`. Without it this wrapper cannot tell a "
                "judgement from the fail-open constant except by diffing the shared judge's "
                "counters, which is the race this class was fixed twice to stop. Wrap a judge "
                "that answers `read`."
            )
        self._judge, self._attempts, self._backoff = judge, attempts, backoff
        # A failure that RETRIED SUCCESSFULLY is not contamination -- the score that survives is a
        # real judgement. Only a case that exhausted its attempts leaves the fail-open constant in
        # the cache, so that is what the gate must key on. Counting raw `errors` there would refuse
        # every sweep against a flaky endpoint even when every case was recovered.
        self.gave_up = 0

    def __getattr__(self, name):          # calls/errors/unparsed/fallbacks read through
        return getattr(self._judge, name)

    def read(self, *args, **kwargs) -> JudgeRead:
        """Retry a failed judge call, and report whether a judgement was ever obtained.

        NO COUNTER IS READ HERE. Two earlier versions of this method diffed the inner judge's
        `errors`/`unparsed` around its own call -- the identical shared-state pattern that
        `SemanticJudge.read` exists to retire, moved one level down and left there while the
        docstring claimed it was gone. That judge is shared, so under concurrency another thread's
        failure lands between the two reads and this call retries a judgement it already had, and
        can end up reporting `fell_open` for an answer that was judged.

        `JudgeRead.reason` carries the cause instead, which is what the retry policy actually needs:
        `unparsed` is deterministic for a model that cannot emit the JSON, so it is not retried and
        counts as unrecovered at once; `error` is worth another attempt. Keying the retry on errors
        alone WITHOUT the unparsed case is the trap the counter version documented -- unparsed then
        reaches neither the retry nor `gave_up`, and a judge answering unreadably every time
        certifies with `unrecovered` at zero.
        """
        reader = self._judge.read
        got = JudgeRead(1.0, fell_open=True, reason="error")
        for attempt in range(self._attempts):
            got = reader(*args, **kwargs)
            if not got.fell_open:
                return got                       # answered, and readable -- a judgement
            if got.reason == "unparsed":
                self.gave_up += 1
                return got                       # deterministic: not retried, and not forgiven
            if attempt + 1 < self._attempts:
                time.sleep(self._backoff * (2 ** attempt))
        self.gave_up += 1
        return got                               # the constant, and recorded as such

    def score(self, *args, **kwargs) -> float:
        """The float protocol the cache and `judge_scores` speak. Unchanged, fail-open included."""
        return self.read(*args, **kwargs).score


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
        gave_up_before = getattr(self._judge, "gave_up", 0)
        s = self._judge.score(question, schema, sql, preview)
        # NEVER PERSIST A NON-JUDGEMENT. A fail-open score is exactly 1.0, which is also what a
        # judge returns when it approves, so a cached constant cannot be told from a verdict by
        # inspection afterwards -- and the whole cache then has to be thrown away and re-paid for
        # to remove one case. MEASURED: a single transient burst left 1 unrecovered case in a
        # 487-call sweep, and repairing it any other way meant re-scoring all 487. Skipping the
        # write leaves that case ABSENT, which a re-run refills by scoring exactly it.
        #
        # A delta, and safe here for a reason `_RetryingJudge` no longer shares: `gave_up` belongs
        # to the wrapper this cache was handed, not to the judge behind it, so no other caller can
        # move it. (`_RetryingJudge` used to diff the SHARED judge's counters and does not any
        # more -- do not read this as precedent for that.) A
        # recovered retry increments `errors` while still producing a real judgement, so `errors`
        # is the wrong counter to key on here and `gave_up` is the right one.
        if getattr(self._judge, "gave_up", 0) != gave_up_before:
            return s
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
    p.add_argument("--judge-attempts", type=int, default=4,
                   help="total ATTEMPTS per judge call that errors (1 = no retry); a low score is never retried")
    p.add_argument("--minidev", default=os.environ.get(
        "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/MINIDEV")))
    p.add_argument("--cache", default="eval-reports/minidev-pg-cache")
    p.add_argument("--with-sanity", action="store_true",
                   help="also sweep the judge WITH the sanity layer beside it (the product's shape: "
                        "either layer defers). The combined row is not the sum of the two separate "
                        "rows -- they overlap")
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
    inner_judge = SemanticJudge(client)
    judge = _CachingJudge(_RetryingJudge(inner_judge, attempts=args.judge_attempts),
                          f"{args.run}.judgecache2.{tag}.json", model)

    @functools.lru_cache(maxsize=None)
    def cards_for(db_id: str) -> str:
        if not db_id:
            return ""
        snap = enrich_bird_db(args.minidev, db_id, settings, cache_dir=args.cache)
        return "\n".join(c.text for c in build_cards(snap))

    print(f"judge endpoint: {base} | model: {model}")
    inner = inner_judge  # the instrumented SemanticJudge, under the retry and cache wrappers
    scores = judge_scores(records, judge, cards_for)

    # How many of those "scores" are the fail-open constant rather than a judgement. Nothing in the
    # cache can answer this: `SemanticJudge` clamps a real reply to 1.0 and returns 1.0 on error, so
    # a dead endpoint and a judge that approved everything write identical files. Only the judge's
    # own counters separate them, and a sweep that cannot be told from an outage must not be
    # published as a measurement. Written beside the cache; counts THIS process's calls, so a run
    # resumed over a warm cache reports only what it re-scored.
    errpath = f"{args.run}.judgeerrors.{tag}.json"
    gave_up = judge._judge.gave_up
    json.dump({"model": model, "calls": inner.calls, "errors": inner.errors,
               "unparsed": inner.unparsed, "fallbacks": inner.fallbacks,
               "unrecovered": gave_up, "attempts_per_case": args.judge_attempts,
               "scored_cases": len(scores)}, open(errpath, "w"), indent=2)
    print(f"judge calls {inner.calls} for {len(scores)} cases: {inner.errors} endpoint errors, "
          f"{inner.unparsed} unreadable replies, retried to {gave_up} UNRECOVERED fail-open "
          f"constants (recorded in {errpath})")
    if gave_up:
        print(f"WARNING: {gave_up} scores are the fail-open constant after {args.judge_attempts} "
              "attempts each, not judgements. Any figure from this sweep is contaminated by that "
              "many cases.")
    # Against DISTINCT pairs, not the record count: `_CachingJudge` dedupes in memory too, so a
    # repeated (question, sql) makes the second record a hit that never reaches the judge. Comparing
    # to `len(scores)` would warn "came from the CACHE" about a case this run judged, and tell the
    # operator to delete a cache that cannot fix it.
    distinct_pairs = len({(r["question"], r.get("sql") or "") for r in records
                          if r["outcome"] in _ANSWERABLE})
    if inner.calls < distinct_pairs:
        # These counters see only THIS process's calls. `_CachingJudge` returns a persisted score
        # without touching the judge, and it writes each score as it goes -- so a sweep killed
        # mid-run leaves fail-open constants on disk, and the resumed run this script's own
        # docstring prescribes would report zero of them over a cache that is mostly constants.
        print(f"WARNING: {distinct_pairs - inner.calls} distinct cases came from the CACHE on "
              "disk and were not "
              "re-judged, so the counts above say nothing about them. Delete the cache to make "
              "this a complete measurement, or read the counts as covering only what was re-scored.")

    print(f"scored {len(scores)} answerable cases; sweeping thresholds:")
    for row in sweep(records, scores, _THRESHOLDS):
        print(f"thr {row['threshold']:.1f}:{_line(row)}")

    if args.with_sanity:
        # The sanity layer is deterministic and free, so it re-runs here rather than being read
        # from anywhere. It is the same `Verifier()` the no-`--judge` invocation reports as
        # "sanity-only", which is a SEPARATE run of this script -- those rows are not printed
        # above, because `--judge` takes the other branch.
        sanity = layer_defers(records, Verifier())
        print(f"\nsanity layer alone defers {sum(sanity)} of {len(sanity)} answerable cases; "
              "sweeping the judge BESIDE it (either layer defers):")
        for row in sweep(records, scores, _THRESHOLDS, also_defer=sanity):
            print(f"thr {row['threshold']:.1f}:{_line(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
