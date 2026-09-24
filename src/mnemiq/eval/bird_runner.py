from __future__ import annotations

import hashlib
import json
import os
import warnings
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.config import Settings
from mnemiq.contract import EvaluationCase, Snapshot
from mnemiq.enrichment.enricher import LLMEnricher
from mnemiq.enrichment.examples import LLMExampleGenerator, enrich_examples
from mnemiq.enrichment.facts import LLMFactsEnricher, enrich_table_facts
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.enrichment.semantic import enrich_semantic
from mnemiq.eval.bird import bird_db_path
from mnemiq.eval.engine import build_engine
from mnemiq.eval.harness import CaseResult, Outcome, run_case
from mnemiq.llm.client import LLMClient


def _append_result(path: str, result: CaseResult) -> None:
    """Checkpoint one result immediately -- a 500-question live run must survive a kill."""
    with open(path, "a") as fh:
        fh.write(json.dumps(asdict(result), default=str) + "\n")


def _load_done(path: str) -> dict[str, CaseResult]:
    """Reload checkpointed results so a resumed run skips what it already answered."""
    if not os.path.isfile(path):
        return {}
    done: dict[str, CaseResult] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["outcome"] = Outcome(d["outcome"])
            done[d["case_id"]] = CaseResult(**d)
    return done


def _meta_path(results_path: str) -> str:
    return results_path + ".meta.json"


def _load_meta(results_path: str) -> tuple[int, int, list[str]]:
    """The totals and exclusions this results file's own run has accumulated.

    NO staleness rule here. It used to carry one -- return zeros unless something was
    restored -- and that conflated two states it could not tell apart: an EXCLUDED case
    appends nothing to the results file, so a run whose progress is entirely exclusions has
    real state in the meta and no result rows, and the zero-rows rule threw it away.
    Measured: two exclusions and 500 tokens lost on resume, and those cases re-probed
    against the gold to rediscover they are too big.

    Staleness is `_meta_is_orphaned`'s single decision, against a recorded row count. When
    it says the meta belongs to another run, `_claim_meta` replaces the file before this
    reads it, so this returns that run's own zeros without a rule of its own.
    """
    path = _meta_path(results_path)
    if not os.path.isfile(path):
        return 0, 0, []
    with open(path) as fh:
        m = json.load(fh)
    return m.get("tokens", 0), m.get("llm_calls", 0), m.get("excluded", [])


class MixedGradingRules(RuntimeError):
    """A resumed results file was graded under a different rule than this run uses."""


def assert_grading_rule_unchanged(
    results_path: str | None, duplicate_rows_insignificant: bool, restored: int
) -> None:
    """Refuse to resume a results file whose stored rows were graded under another rule.

    `_load_done` restores earlier outcomes VERBATIM, so a file resumed across a change of
    grading rule holds rows decided two ways while the meta -- rewritten whole on each save --
    records only the last invocation's. The number that comes out is not one metric, and
    nothing in the artifact says so. That is the shape M34 already cost this project once: a
    gate that could not fail while still ratcheting.

    Recording the rule made this WORSE before it made it better. Silence at least said
    nothing; a meta stamped `true` over rows graded multiset is a confident false claim.

    Three states, and the third is the one worth spelling out:

      * stored == current -- resume, nothing mixed
      * stored != current -- REFUSE
      * stored is None -- REFUSE, and this covers TWO provenances. Either the meta predates
        the field, or there is no meta file at all: results are appended per case and the
        meta written after, so a killed run leaves one, and so does deleting the meta by
        hand. Both mean the stored rows cannot be confirmed either way, which is not the
        same as matching, so neither may assume the convenient answer. The message names
        which of the two it found.

    Only when rows would actually be restored: an empty or absent results file has nothing to
    mix, and a fresh run must not be blocked by a stale meta beside it.
    """
    if not results_path or restored == 0:
        return
    # A MISSING meta is the same claim as an unrecorded one: both land in `stored is None`,
    # which this function's contract says must refuse.
    # Returning early here instead let the guard go silent on that file shape.
    has_meta = os.path.isfile(_meta_path(results_path))
    if has_meta:
        with open(_meta_path(results_path)) as fh:
            stored = json.load(fh).get("duplicate_rows_insignificant")
    else:
        stored = None
    if stored is duplicate_rows_insignificant:
        return
    if os.environ.get("MNEMIQ_ALLOW_MIXED_GRADING") == "1":
        return
    if stored is not None:
        was = f"duplicate_rows_insignificant={stored}"
    else:
        was = "not recorded" if has_meta else "no metadata file at all"
    raise MixedGradingRules(
        f"{results_path} holds {restored} results graded with {was}, and this run grades with "
        f"duplicate_rows_insignificant={duplicate_rows_insignificant}. Resuming would mix two "
        f"rules into one number.\n"
        f"Use a fresh --results path -- always sufficient, and the only thing that is when a "
        f"run's progress was exclusions, which write no results file to delete. Deleting "
        f"{results_path} also works when it exists: the meta beside it is then replaced "
        f"rather than read.\n"
        f"MNEMIQ_ALLOW_MIXED_GRADING=1 overrides this if you know the difference cannot reach "
        f"these cases."
    )


def resume_state(
    results_path: str | None, duplicate_rows_insignificant: bool
) -> tuple[dict[str, CaseResult], int, int, list[str]]:
    """Everything a resumed run restores, decided once: prior results, prior totals, prior
    exclusions -- and the refusal when the stored rows were graded under another rule.

    One function because the three runners had the same four lines each and the JOIN between
    them is where this went wrong twice. `_load_meta` is correct on its own and was pinned on
    its own, and passing it a literal instead of `len(done)` restored the stale-metadata bug
    with all 2143 tests green. A helper the runners share is a seam a test can hold.
    """
    done = _load_done(results_path) if results_path else {}
    assert_grading_rule_unchanged(results_path, duplicate_rows_insignificant, len(done))
    if results_path and not done:
        # TWO decisions, not one. Whether to DISCARD the meta's state is
        # `_meta_is_orphaned`'s; whether to stamp THIS run's rule into it is separate and
        # unconditional here, because nothing has been graded yet under the old one. An
        # exclusion-only run keeps its exclusions -- they are gold-too-big verdicts and owe
        # nothing to the grading rule -- while still recording the rule about to be used.
        if _meta_is_orphaned(results_path, len(done)):
            _claim_meta(results_path, duplicate_rows_insignificant)
        else:
            _restamp_rule(results_path, duplicate_rows_insignificant)
    tokens, calls, excluded = _load_meta(results_path) if results_path else (0, 0, [])
    if not done and (tokens or calls or excluded):
        # SAY SO. This is the one case the file cannot decide: a meta with state and no
        # result rows is an exclusion-only run mid-flight AND a finished one the operator
        # may have meant to start over from, and nothing on disk tells them apart -- an
        # excluded case writes no row, so there is no results file to delete as a signal.
        # Inheriting is the right default (it is usually the same run), but inheriting in
        # SILENCE is what makes the wrong case invisible: `excluded` is computed against
        # `max_rows_cap`, so a re-run under a different cap would skip those cases without
        # ever probing them.
        warnings.warn(
            f"resuming {results_path} with no result rows but prior state from its meta: "
            f"{tokens} tokens, {calls} calls, {len(excluded)} excluded "
            f"({', '.join(excluded[:3])}{'...' if len(excluded) > 3 else ''}). "
            f"That is an exclusion-only run continuing. To start clean instead, use a fresh "
            f"--results path -- deleting the results file does nothing here, because an "
            f"excluded case never wrote one. Note `excluded` was computed against the "
            f"earlier run's --max-rows.",
            stacklevel=2,
        )
    return done, tokens, calls, excluded


def _claim_meta(results_path: str, duplicate_rows_insignificant: bool) -> None:
    """Stamp this run's rule into the meta BEFORE it grades anything, replacing any stale one.

    Two problems, one write.

    IGNORING A STALE META IS NOT ENOUGH, because the condition that makes it ignorable is
    temporary. `_load_meta` skips it when nothing was restored, which is right for the fresh
    run -- but the moment that run writes its first result and is interrupted, the resume
    restores rows, `restored` is no longer zero, and the OLD meta reattaches with its token
    totals and its exclusions. Reproduced: delete the results file, resume once (totals
    correctly zero), answer one case, kill, resume again -- 9999 tokens and a stale excluded
    id come back.

    And DELETING it instead is not enough either, because results are appended per case while
    the meta was only written at a checkpoint: an interrupted fresh run legitimately leaves
    rows with no meta, which `assert_grading_rule_unchanged` then refuses as unconfirmable.
    That refusal is right -- rows whose rule was never recorded ARE unconfirmable -- so the
    fix is to leave no such window rather than to soften it.

    Writing the rule before the first row closes both: a stale meta is gone, and every row
    this run appends has its rule already on disk. Totals start at zero because this run has
    spent nothing yet; the first checkpoint overwrites them.
    """
    _save_meta(results_path, 0, 0, [], duplicate_rows_insignificant, results_rows=0)


def _restamp_rule(results_path: str, duplicate_rows_insignificant: bool) -> None:
    """Record the rule this run will grade under, keeping the meta's accumulated state.

    Reached when nothing was restored and the meta is NOT orphaned -- an exclusion-only run
    mid-flight, or one whose rule was claimed and which has not produced a row yet. Its
    totals and exclusions are real and stay; only the rule is this run's to set, and it is
    safe to set because no row has been graded under the old one.

    Without this, deleting the results file of an exclusion-only run and re-running under
    the other rule left a meta claiming the rule that was NOT used.
    """
    path = _meta_path(results_path)
    if not os.path.isfile(path):
        # A brand-new --results path. Writing the rule here is the whole point of stamping
        # before the first row: without it the run appends rows whose rule was never
        # recorded, and `assert_grading_rule_unchanged` refuses that on the next resume.
        # `run_bird` checkpoints only after a whole db group, so a single-db run or a kill
        # inside the first group would be unresumable.
        _save_meta(results_path, 0, 0, [], duplicate_rows_insignificant, results_rows=0)
        return
    with open(path) as fh:
        m = json.load(fh)
    _save_meta(results_path, m.get("tokens", 0), m.get("llm_calls", 0), m.get("excluded", []),
               duplicate_rows_insignificant, results_rows=m.get("results_rows"))


def _meta_is_orphaned(results_path: str, restored: int) -> bool:
    """Whether the meta beside this results file belongs to a DIFFERENT run.

    Emptiness cannot answer this, and reading it as if it could destroyed real state. An
    EXCLUDED case appends nothing to the results file -- it lands only in the meta's
    `excluded` -- so a run whose progress so far is entirely exclusions has a populated meta
    and no result rows, which looks exactly like a fresh run beside a leftover meta.

    So the meta records how many result rows it was written beside, and rows DISAPPEARING is
    what makes it stale -- `recorded > restored`, not `!=`. The meta is written at checkpoint
    boundaries while results are appended per case, so a meta that lags behind the file is
    the ordinary mid-flight state, not evidence of anything: 5 recorded against 7 restored is
    a run that progressed since its last checkpoint. 1 against 0 is a results file that was
    deleted, and 0 against 0 is an exclusion-only run.

    Getting this wrong as `!=` would have discarded the last checkpoint's totals and
    exclusions on every resume of a run killed between a row and a checkpoint -- caught by
    the seam test walking that exact sequence.

    An older meta carries no such count, and unknown is not a match -- the same reading the
    grading rule gets, for the same reason.
    """
    if not os.path.isfile(_meta_path(results_path)):
        return False  # no meta to orphan; `_restamp_rule` writes the first one
    with open(_meta_path(results_path)) as fh:
        recorded = json.load(fh).get("results_rows")
    if recorded is None:
        return True
    return recorded > restored


def source_rev() -> str:
    """The mnemiq revision that produced a run, for a consumer's provenance record.

    A results file says what the engine answered and how this runner graded it, and
    "how this runner graded it" is a moving target -- the grading rules changed twice
    in one day. Without a revision, a consumer importing the file can record the script
    path and nothing that actually pins the behaviour.

    Fail-soft, and honest when uncertain: an eval run must never die for want of a
    version string, and a bare rev that hides uncommitted changes is worse than no rev,
    because it looks precise. A tree with MODIFIED TRACKED FILES is marked as such.

    Untracked files deliberately do not count, and the distinction is not pedantry. This
    marked `-dirty` on any porcelain output, so three orphan scratch files in the checkout
    stamped every artifact produced there as unreproducible -- including the Spider2 k=24 run,
    whose code was in fact exactly its commit. A flag that fires when nothing is wrong is a
    flag nobody reads when something is, which is M19's lesson about a guard that cries wolf.

    The signal is not discarded, because an untracked file can be something a run READ -- an
    authz policy or a golden set sitting beside the code. It moves to `untracked_files` in the
    meta, where it says what it means instead of impersonating a modified tree. Two states,
    two fields: collapsing them is the defect M2, M6 and M34 all turned out to be.
    """
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(  # noqa: S603 -- fixed argv, no shell
                ["git", "-C", here, *args], capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    rev = git("rev-parse", "--short", "HEAD")
    if rev is None:  # not a checkout: an installed wheel still has a version
        try:
            from importlib.metadata import version

            return f"mnemiq {version('mnemiq')}"
        except Exception:  # noqa: BLE001 -- provenance is never worth failing a run for
            return ""
    return f"{rev}-dirty" if git("status", "--porcelain", "--untracked-files=no") else rev


def untracked_count() -> int:
    """How many untracked files sat beside the code. Provenance, not a warning: a consumer
    comparing two artifacts can see whether the working directory differed at all."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        done = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            ["git", "-C", here, "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    return len([ln for ln in done.stdout.splitlines() if ln.strip()]) if done.returncode == 0 else 0


def _save_meta(results_path: str, tokens: int, calls: int, excluded: list[str],
               duplicate_rows_insignificant: bool | None = None,
               results_rows: int | None = None) -> None:
    """Write the run's metadata beside its results.

    `duplicate_rows_insignificant` is recorded because the results file is RESUMABLE and the
    grading rule can change between segments: `_load_done` restores earlier outcomes verbatim,
    so a file resumed across the M105 change holds multiset-graded rows beside set-graded ones
    with nothing saying so. `source_rev` does not cover it -- only the last segment's rev
    survives. `None` means the runner did not say, which is what an older meta file looks like
    and is not the same claim as `false`.
    """
    with open(_meta_path(results_path), "w") as fh:
        json.dump(
            {
                "tokens": tokens,
                "llm_calls": calls,
                "excluded": excluded,
                "source_rev": source_rev(),
                "untracked_files": untracked_count(),
                "duplicate_rows_insignificant": duplicate_rows_insignificant,
                # How many result ROWS stood beside this meta when it was written. It is
                # what tells a stale meta from a live one whose progress is entirely
                # EXCLUSIONS -- an excluded case appends nothing to the results file, so
                # both look like "no results" and only this says which. `None` is an older
                # file.
                "results_rows": results_rows,
            },
            fh,
        )


# Facts/examples default OFF (settings.enrich_facts/enrich_examples): the plan-20 A/B measured both
# phases as regressions on a strong frontier model (facts -2.9, facts+examples -8.0 strict). Parked
# as opt-in plumbing; set MNEMIQ_ENRICH_FACTS/EXAMPLES=1 (a weaker/local model may need scaffolding).
def _enrich_cache_suffix(
    settings: Settings, certified_digest: str = "", semantic: bool = True
) -> str:
    # `semantic` belongs in the KEY, not just in the caller's choice of directory. The
    # structural and semantic snapshots of one database differ only in whether the
    # descriptions are LLM-written, so a tier-1 run pointed at a tier-2 cache silently
    # inherits tier-2's descriptions and reports a tier-1 number that never happened.
    # Separate directories prevent that by convention; this prevents it by construction.
    parts = [] if semantic else ["structural"]
    if settings.enrich_facts:
        parts.append("facts")
    if settings.enrich_examples:
        # the fan-out guard decides which proposed examples survive -> a guard-off example set
        # must not be reused with the guard on (M109)
        parts.append("examples_fanout" if settings.guard_fanout else "examples")
    if settings.dictionary_path:
        # a dictionary changes grounded meanings -> must not reuse a no-dict (or other-dict)
        # snapshot. Content edits to the same path still need --refresh (as grounding itself does).
        import hashlib
        parts.append("dict" + hashlib.sha256(settings.dictionary_path.encode()).hexdigest()[:6])
    if settings.ontology_records_path:
        # ontology bindings change grounded meanings AND which columns carry a scheme -> a
        # no-ontology snapshot must never be reused. Content edits still need --refresh.
        import hashlib

        parts.append(
            "onto" + hashlib.sha256(settings.ontology_records_path.encode()).hexdigest()[:6]
        )
    if certified_digest:
        # a changed set of certified records must not reuse a snapshot built from another set
        parts.append("cert" + certified_digest[:6])
    return f"__{'_'.join(parts)}" if parts else ""


def enrich_bird_db(
    minidev_dir: str,
    db_id: str,
    settings: Settings,
    cache_dir: str | None = None,
    refresh: bool = False,
    semantic: bool = True,
    db_path_fn: Callable[[str, str], str] = bird_db_path,
) -> Snapshot:
    """Enrich one BIRD database. Cached to disk: BIRD DBs never change, so (db_id, model)
    is the key -- the enriched snapshot depends on the model, so switching models must not
    silently reuse another model's enrichment. The facts/examples toggles enter the key too,
    so an A/B run never reuses another config's enrichment."""
    from mnemiq.enrichment.certified import (
        apply_certified_set, fetch_certified_records, require_certified)

    _certified_set = fetch_certified_records(settings)
    require_certified(_certified_set, settings)
    _certified = _certified_set.records
    _cert_digest = hashlib.sha256(
        "".join(sorted(r.envelope.version for r in _certified)).encode()
    ).hexdigest() if _certified else ""

    model_slug = (settings.llm_model or "default").replace("/", "_")
    cache_path = (
        os.path.join(cache_dir,
                     f"{db_id}__{model_slug}{_enrich_cache_suffix(settings, _cert_digest, semantic)}.json")
        if cache_dir else None
    )
    if cache_path and not refresh and os.path.isfile(cache_path):
        with open(cache_path) as fh:
            return Snapshot.model_validate_json(fh.read())

    from mnemiq.enrichment.dictionary import load_dictionary
    from mnemiq.enrichment.grounding import apply_dictionary, ground_codes
    from mnemiq.enrichment.pipeline import content_version
    from mnemiq.ontology.records import load_records

    adapter = SQLiteAdapter(db_path_fn(minidev_dir, db_id))
    snapshot = enrich_structural(adapter, db_id)
    _dict = load_dictionary(settings.dictionary_path) if settings.dictionary_path else None
    _onto = load_records(settings.ontology_records_path) if settings.ontology_records_path else None
    # precedence: ontology < correlated < lookup < certified < dictionary
    snapshot = ground_codes(adapter, snapshot, dictionary=None, ontology=_onto)
    snapshot, _protected = apply_certified_set(snapshot, _certified_set)
    if semantic:
        snapshot = enrich_semantic(snapshot, LLMEnricher(LLMClient(settings)), protected=_protected)
        if settings.enrich_facts:
            snapshot = enrich_table_facts(snapshot, LLMFactsEnricher(LLMClient(settings)))
        if settings.enrich_examples:
            snapshot = enrich_examples(
                snapshot, LLMExampleGenerator(LLMClient(settings)),
                adapter, dialect=adapter.dialect, guard_fanout=settings.guard_fanout,
            )

    if _dict:
        snapshot = apply_dictionary(snapshot, _dict)  # the operator's final override
        snapshot.version = content_version(snapshot)  # re-version: dict landed after enrich_semantic

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as fh:
            fh.write(snapshot.model_dump_json(by_alias=True))
    return snapshot


def _run_grouped(
    cases: list[EvaluationCase],
    build: Callable[[str], tuple[Callable, object]],
    gold_sql_sentinel: str | None = None,
    dupes_ok: bool = False,
) -> list[CaseResult]:
    """Route each case to its DB's engine+adapter (built once per DB). Pure: engine injected."""
    by_db: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        by_db.setdefault(case.db_id, []).append(case)

    results: list[CaseResult] = []
    for db_id, db_cases in by_db.items():
        ask, adapter = build(db_id)
        for case in db_cases:
            if gold_sql_sentinel is not None:  # test hook: fake adapter keys off a sentinel
                case = case.model_copy(update={"gold_sql": gold_sql_sentinel})
            results.append(run_case(case, ask, adapter,
                                    duplicate_rows_insignificant=dupes_ok))
    return results


def _process_db(cases, build_engine_fn, max_rows_cap: int, workers: int,
                dupes_ok: bool = False):
    """Run one database's cases, optionally across worker threads.

    Thread safety: each worker builds its OWN engine on first use (thread-local) -- its own
    DuckDB store connection, its own SQLite adapter, its own LLM client -- so no non-thread-safe
    connection is ever shared. The OpenAI HTTP client is concurrency-safe. Results are
    consumed single-threaded by the caller (ThreadPoolExecutor.map preserves order), so the
    checkpoint append needs no lock. Returns (list of (kind, payload), per-worker clients).
    """
    local = threading.local()
    clients: list = []
    clients_lock = threading.Lock()

    def engine():
        if not hasattr(local, "e"):
            ask, engine_adapter, gold_adapter, client = build_engine_fn()
            local.e = (ask, engine_adapter, gold_adapter)
            with clients_lock:
                clients.append(client)
        return local.e

    def work(case):
        ask, engine_adapter, gold_adapter = engine()
        if _gold_too_big(gold_adapter, case.gold_sql, max_rows_cap):
            return ("excluded", case.id)
        return ("result", run_case(case, ask, engine_adapter, gold_adapter,
                                   duplicate_rows_insignificant=dupes_ok))

    if workers <= 1:
        out = [work(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(work, cases))
    return out, clients


def _gold_too_big(adapter, gold_sql: str, cap: int) -> bool:
    probe = f"SELECT count(*) FROM (SELECT 1 FROM ({gold_sql}) LIMIT {cap + 1})"
    try:
        return adapter.execute(probe)[0][0] > cap
    except Exception:
        return False  # let run_case surface a real error rather than pre-judging


def run_bird(
    cases: list[EvaluationCase],
    minidev_dir: str,
    settings: Settings,
    *,
    cache_dir: str | None = None,
    max_rows_cap: int = 1000,
    on_case: Callable[[int, int, CaseResult], None] | None = None,
    results_path: str | None = None,
    workers: int = 1,
    candidates: int = 1,
    semantic: bool = True,
    db_path_fn: Callable[[str, str], str] = bird_db_path,
    duplicate_rows_insignificant: bool = False,
) -> tuple[list[CaseResult], dict]:
    """Run BIRD cases grouped by database. Resumable: with results_path, each result is
    checkpointed as it completes and a re-run skips everything already answered -- a long
    live run survives a kill without re-paying for the questions it already got through.

    `duplicate_rows_insignificant` is the BENCHMARK's declaration and defaults OFF, because
    despite the name THIS RUNNER IS NOT BIRD-ONLY -- `scripts/run_spider.py` drives Spider 1.0
    through it with `db_path_fn=spider_db_path`. BIRD publishes `set(pred) == set(gold)` and
    declares it; Spider does not. Hardcoding it here graded every Spider run under a rule
    Spider does not publish, and `regrade_engine_runs.py --benchmark spider` would then have
    reported those cases as verdicts changed, blaming grader drift for a runner's hardcode.
    See register M105."""
    by_db: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        by_db.setdefault(case.db_id, []).append(case)

    done_results, tokens, calls, excluded = resume_state(
        results_path, duplicate_rows_insignificant)
    skip = set(done_results) | set(excluded)

    results: list[CaseResult] = list(done_results.values())
    processed = len(skip)

    for db_id, db_cases in by_db.items():
        remaining = [c for c in db_cases if c.id not in skip]
        if not remaining:
            continue  # whole DB already done in a prior segment -- no enrichment, no client

        snapshot = enrich_bird_db(
            minidev_dir, db_id, settings, cache_dir=cache_dir,
            semantic=semantic, db_path_fn=db_path_fn,
        )

        def _build():  # each worker builds its own isolated engine (thread-safe connections)
            # BIRD grades single-engine on native SQLite: the engine generates + executes
            # SQLite and gold runs on the same engine, so a wrong answer is a real error,
            # never a cross-engine artifact (parity measured 1.5% otherwise). DuckDB is the
            # executor in the product path (DuckDBAdapter); the benchmark stays apples-to-apples.
            adapter = SQLiteAdapter(db_path_fn(minidev_dir, db_id))
            ask, client = build_engine(snapshot, adapter, settings, candidates=candidates)
            return ask, adapter, adapter, client  # engine + gold: same native SQLite executor

        out, clients = _process_db(remaining, _build, max_rows_cap, workers,
                                   duplicate_rows_insignificant)

        # collection is single-threaded here -> checkpoint append needs no lock
        for kind, payload in out:
            processed += 1
            if kind == "excluded":
                excluded.append(payload)
                continue
            results.append(payload)
            if results_path is not None:
                _append_result(results_path, payload)
            if on_case is not None:
                on_case(processed, len(cases), payload)

        for client in clients:
            tokens += client.total_tokens
            calls += client.calls
        if results_path is not None:
            _save_meta(results_path, tokens, calls, excluded,
                       duplicate_rows_insignificant,
                       results_rows=len(results))

    return results, {"tokens": tokens, "llm_calls": calls, "excluded": excluded}
