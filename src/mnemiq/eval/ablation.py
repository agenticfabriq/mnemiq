"""Part 2's ablation: does certified meaning change the answer?

Two arms over the SAME database, the same questions, the same model and the same engine. The only
difference is whether Verity's certified records were applied to the snapshot. The claim under test
is that grounding wins where meaning is not recoverable from the schema, and does NOT win where it
is -- the second half is what separates "the semantic layer carries meaning" from "any extra prose
in the prompt helps".

**What is reported, and what is not.** Exact match and deferral, per arm and per band, with
deferral beside accuracy rather than folded into it. Unportability -- whether the engine's SQL also
runs on the gold's engine -- is deliberately absent: it is only meaningful when the two sides run on
different engines, and here one adapter executes both, so `run_case` never sets it and a column of
`None` would suggest a measurement that was not taken rather than one that does not apply.

**Why this runs the PRODUCT path and not `build_engine`.** Two reasons, and only one of them is
still about grounding. When this was written the eval engine's `ask` passed neither `metrics` nor
`dimensions` to retrieval while `Runtime.ask` passed both -- so an ablation driven through
`build_engine` would have grounded the grounded arm less than production does, and a null result
would have been indistinguishable from the layer never having been connected. That was register
**M81**, and it is CLOSED: both doors now pass the same grounding, pinned by a test that compares
the two call sites rather than either one alone.

What remains is the reason that does not expire: `Runtime` is the code the product runs, so the
traces Part 2 grades are emitted by the emitter a customer would use rather than by a second one
written for the experiment. An ablation through `build_engine` would measure the engine correctly
today and still have to grow its own trace emission to be Part 2.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mnemiq.config import Settings
from mnemiq.contract import EvaluationCase, IdentityContext
from mnemiq.eval.harness import CaseResult, Outcome, run_case

# The identity every arm answers as. Fixed across arms for the same reason the model and the
# questions are: it must not be a way the two runs differ.
IDENTITY = IdentityContext(tenant_id="fspay", principal_id="part2", roles=["analyst"])

BARE = "bare"
GROUNDED = "grounded"


@dataclass
class ArmOutcome:
    """What one arm was, and what it answered."""

    name: str
    snapshot_version: str
    certified_refs: int
    metrics: int
    dimensions: int
    # The certified objects THEMSELVES, because the gate cannot use a digest to compare them --
    # see `check_gate`. Ids only; the definitions are in the snapshot.
    certified_objects: frozenset[str] = frozenset()
    results: list[CaseResult] = field(default_factory=list)

    def by_id(self) -> dict[str, CaseResult]:
        return {result.case_id: result for result in self.results}


@dataclass
class Gate:
    """The instrument check, run BEFORE any number is reported.

    `fetch_certified_records` is fail-soft by design: a 401, a moved URL or a shape change leaves
    the run going, serving whatever was cached locally -- an empty set on a first run, and
    LAST-KNOWN-GOOD on any later one. A grounded arm that grounded nothing scores exactly like
    bare, and "the semantic layer does not help" would be indistinguishable from "the semantic
    layer was not plugged in". Every check here is about the INSTRUMENT, never about the result.

    Note what the cache means for check 1: `certified_refs > 0` says records were APPLIED, not that
    they were fetched fresh. `run_ablation` clears each arm's store and its watermark sidecar for
    exactly this reason -- otherwise a second run into the same directory greens this check from
    cache with Verity unreachable, which is the one state the gate exists to refuse.
    """

    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append((name, passed, detail))

    @property
    def passed(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def render(self) -> str:
        lines = ["gate:"]
        for name, passed, detail in self.checks:
            lines.append(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")
        return "\n".join(lines)


def check_gate(bare: ArmOutcome, grounded: ArmOutcome) -> Gate:
    """Did grounding actually reach the grounded arm?

    Measured from the two persisted snapshots rather than by instrumenting the enrichment, because
    the snapshot is what the engine answers FROM. An arm that read the records and then failed to
    apply them is not grounded, however healthy the fetch looked.
    """
    gate = Gate()
    gate.record(
        "records were read",
        grounded.certified_refs > 0,
        f"the grounded arm carries {grounded.certified_refs} certified ref(s); "
        "0 means the fetch degraded to empty and this arm is bare with a different name",
    )
    # Certified metrics and dimensions are what `apply_certified` APPENDS. The bare arm profiles
    # the same database and finds none, because no column name says "gross includes the tip".
    gained = (grounded.metrics - bare.metrics) + (grounded.dimensions - bare.dimensions)
    gate.record(
        "the snapshot changed",
        gained > 0,
        f"grounding added {gained} certified object(s) "
        f"(metrics {bare.metrics}->{grounded.metrics}, "
        f"dimensions {bare.dimensions}->{grounded.dimensions})",
    )
    # NOT `snapshot.version`. That is `content_version`, whose body is source_id, columns,
    # relationships, source_bindings, views and ontology_version -- and never `metrics` or
    # `dimensions`. Measured: a snapshot with 0 metrics and 0 dimensions and one with 5 of each
    # hash IDENTICALLY (`acdc9b95cd5b` both). So a digest comparison is blind to exactly the
    # objects grounding adds and check 2 counts: it would refuse a genuinely grounded run whose
    # certified corpus is metrics and dimensions only, and it passes on nothing more than the two
    # arms' independent enrichment passes describing a column differently. Comparing the object
    # sets asks the question the digest cannot.
    only_grounded = sorted(grounded.certified_objects - bare.certified_objects)
    gate.record(
        "the arms carry different meaning",
        bool(only_grounded),
        f"the grounded arm carries {len(only_grounded)} certified object(s) the bare arm does not "
        f"({only_grounded[:4]}{'...' if len(only_grounded) > 4 else ''}); none would mean both "
        "arms answered from the same meaning",
    )
    return gate


# -- cuts -----------------------------------------------------------------------------------------

# The three ways the answer is read. `meaning` is where the layer can pay: the question turns on a
# rule the schema cannot state. `control` is the half that makes the result falsifiable -- questions
# a reader of the DDL alone can answer. If grounding moves the control band as much as the meaning
# band, it is acting as a general prompt improvement and the ablation has NOT shown what it claims.
CUTS = (
    ("all questions", None),
    ("meaning-dependent", "meaning"),
    ("schema-recoverable (control)", "control"),
)


def _cut(results: Sequence[CaseResult], cases: dict[str, EvaluationCase], tag: str | None
         ) -> list[CaseResult]:
    if tag is None:
        return list(results)
    return [r for r in results if tag in (cases[r.case_id].tags if r.case_id in cases else [])]


@dataclass
class AblationReport:
    bare: ArmOutcome
    grounded: ArmOutcome
    gate: Gate
    cases: dict[str, EvaluationCase]

    def render(self) -> str:
        lines = [self.gate.render(), ""]
        if not self.gate.passed:
            lines.append(
                "NO RESULT. The gate failed, so the arms are not comparable and no accuracy "
                "number below would mean what it appears to mean."
            )
            return "\n".join(lines)

        header = f"{'cut':32s} {'n':>3s}  {'arm':9s} {'correct':>8s} {'facts':>6s} {'wrong':>6s} {'defer':>6s} {'error':>6s}"
        lines.append(header)
        lines.append("-" * len(header))
        for label, tag in CUTS:
            bare_cut = _cut(self.bare.results, self.cases, tag)
            grounded_cut = _cut(self.grounded.results, self.cases, tag)
            if not bare_cut and not grounded_cut:
                lines.append(f"{label:32s} {'0':>3s}  (no cases carry this tag)")
                continue
            for arm_name, cut in ((BARE, bare_cut), (GROUNDED, grounded_cut)):
                counts = _counts(cut)
                lines.append(
                    f"{label if arm_name == BARE else '':32s} {len(cut):>3d}  {arm_name:9s} "
                    f"{counts[Outcome.CORRECT]:>8d} {counts[Outcome.CORRECT_FACTS]:>6d} "
                    f"{counts[Outcome.WRONG]:>6d} "
                    f"{counts[Outcome.DEFERRED_CORRECTLY] + counts[Outcome.DEFERRED_WRONGLY]:>6d} "
                    f"{counts[Outcome.ERROR]:>6d}"
                )
            lines.append("")

        # How much grounding CHANGED the control band, which is a sharper question than whether it
        # scored better there. Both arms sitting at 5/5 is a ceiling, and a ceiling cannot tell
        # "the records were irrelevant here" from "the records helped and the questions were too
        # easy to show it". Identical SQL can: if the engine emits the same query with and without
        # the certified records, they were inert on that question rather than merely harmless.
        control_ids = [r.case_id for r in _cut(self.bare.results, self.cases, "control")]
        if control_ids:
            bare_sql, grounded_sql = self.bare.by_id(), self.grounded.by_id()
            # Only cases where BOTH arms actually emitted SQL. `CaseResult.sql` is `""` on the
            # defer and failure paths, so comparing blindly counts two silences as agreement --
            # and a run where every control case deferred would print "inert in both arms" as its
            # strongest claim while no SQL was produced at all.
            comparable = [
                case_id for case_id in control_ids
                if bare_sql.get(case_id) and grounded_sql.get(case_id)
                and bare_sql[case_id].sql.strip() and grounded_sql[case_id].sql.strip()
            ]
            identical = sum(
                1 for case_id in comparable
                if _normalised(bare_sql[case_id].sql) == _normalised(grounded_sql[case_id].sql)
            )
            silent = len(control_ids) - len(comparable)
            lines.append(
                f"control-band SQL identical in both arms: {identical}/{len(comparable)} "
                f"compared" + (f", {silent} not compared (an arm emitted no SQL)" if silent else "")
                + " -- where both arms answered, identical SQL means grounding was inert, which is "
                "the claim the control exists to test; a general prompt improvement would have "
                "changed it"
            )
            lines.append("")

        # The caveat travels WITH the number, because the number is what gets quoted. Three weeks
        # later a table in a scrollback has no context, and this one reads as a market claim unless
        # it says otherwise on its own face.
        # Structural only. An earlier version stated this run's conclusions here -- that the bare
        # arm was "wrong in defensible ways" and the control band "unchanged" -- which printed
        # unconditionally, including on a run whose control band held no cases at all. A caveat
        # that asserts results it has not read is a worse lie than no caveat, so what stays is
        # what is true of the DESIGN, and the reader is pointed at the rows above for the rest.
        lines.append(
            "what this shows: certified meaning reaches the answer, end to end -- fetch, apply,\n"
            "  retrieve, generate. What it does NOT show: how a model fares on meaning it has not\n"
            "  been given. We authored the definitions AND the questions, and the meaning band's\n"
            "  gold is the certified expression itself, so a grounded arm carrying that definition\n"
            "  is expected to match it. The parts that are NOT circular are in the rows above: how\n"
            "  the bare arm failed (wrong beats deferred -- read the per-case list), and whether\n"
            "  the control band moved at all."
        )
        lines.append("")

        # Deferral is reported BESIDE accuracy and never folded into it: an arm that answers fewer
        # questions and gets more of them right is a different product decision, not a better score.
        lines.append("per case (correct / facts / wrong / deferred / error):")
        for case_id in sorted(self.cases):
            bare_r = self.bare.by_id().get(case_id)
            grounded_r = self.grounded.by_id().get(case_id)
            tags = ",".join(self.cases[case_id].tags) or "-"
            lines.append(
                f"  {case_id:34s} {tags:18s} "
                f"bare={bare_r.outcome if bare_r else 'missing':<18s} "
                f"grounded={grounded_r.outcome if grounded_r else 'missing'}"
            )
        return "\n".join(lines)


def _normalised(sql: str) -> str:
    """Whitespace and case folded away. Not a SQL parser, and not trying to be: the question is
    whether the engine emitted the SAME query, and two spellings that differ by a line break are
    the same query while `count(1)` and `count(*)` are deliberately left as different."""
    return " ".join(sql.split()).lower().rstrip(";")


def _counts(results: Sequence[CaseResult]) -> dict[Outcome, int]:
    counts = dict.fromkeys(Outcome, 0)
    for result in results:
        counts[result.outcome] += 1
    return counts


# -- running an arm -------------------------------------------------------------------------------


def source_mismatch(snapshot, cases: Sequence[EvaluationCase]) -> str | None:
    """Is this snapshot even about the database the questions ask about?

    Enriching the wrong source SUCCEEDS -- any populated Postgres profiles fine -- and then every
    answer is wrong for a reason that has nothing to do with grounding, which is the most expensive
    way to learn that a DSN was stale. The gold SQL names the tables the corpus is about, so a
    snapshot sharing none of them is pointed somewhere else.

    Returns the complaint, or None when the source looks right. Deliberately not an assertion about
    ALL tables matching: a bundle may carry tables no question touches, and that is normal.
    """
    tables = {
        column.object_id.split(".")[-1].lower()
        for column in (snapshot.columns if snapshot else [])
    }
    if not tables:
        return None  # nothing profiled; `enrich` has its own, better, complaint for that
    wanted = " ".join(case.gold_sql or "" for case in cases).lower()
    # Whole words. A plain `in` makes a short table name match almost any query -- a table called
    # `c` is a substring of `count(*)` -- so the check would pass on every snapshot and guard
    # nothing, which is the failure mode it was written to prevent, one level up.
    if any(re.search(rf"\b{re.escape(table)}\b", wanted) for table in tables):
        return None
    return (
        f"the snapshot carries {sorted(tables)[:6]} and no gold query in this corpus names any of "
        "them, so this arm is pointed at a different database than the questions are about"
    )


def arm_settings(base: Settings, *, name: str, store_dir: Path, records_url: str | None,
                 traces_url: str | None, authz_path: str) -> Settings:
    """One arm's configuration: its own store, and records only if it is the grounded arm.

    A separate store per arm is not tidiness. The snapshot is keyed by source id, so two arms
    sharing a store would overwrite each other's snapshot and the second run would silently measure
    the first one's meaning.
    """
    return base.model_copy(update={
        "store_path": str(store_dir / f"{name}.duckdb"),
        "verity_records_url": records_url,
        "verity_traces_url": traces_url,
        # Its OWN watermark sidecar. The default is `dirname(store_path)/verity-watermark.json`,
        # which both arms would share -- so the grounded arm's incremental-sync state would sit in
        # the same file the bare arm's store lives beside, and a later run could serve one arm from
        # the other's cache.
        "verity_watermark_path": str(store_dir / f"{name}-verity-watermark.json"),
        "authz_path": authz_path,
    })


def run_arm(name: str, settings: Settings, cases: Sequence[EvaluationCase]) -> ArmOutcome:
    """Enrich, build and answer -- through the same three doors the product uses.

    The CLI's own `_cmd_enrich` and `_cmd_build` are imported despite the underscore, deliberately.
    They ARE the enrichment: the certified fetch, the grounding precedence, the protected-column
    set and the snapshot digest all live inside them. Re-assembling that here would be a second
    spelling of the pipeline whose whole purpose is to be the thing being measured, and the two
    would drift on the first change to either -- with the experiment quietly measuring the older
    one. If this needs a public seam, the fix is to give the CLI one, not to copy it.
    """
    from mnemiq.cli import _cmd_build, _cmd_enrich
    from mnemiq.runtime import build_runtime

    if _cmd_enrich(settings) != 0:
        raise RuntimeError(f"{name}: enrich failed; see the log above")
    if _cmd_build(settings) != 0:
        raise RuntimeError(f"{name}: build failed; see the log above")

    runtime = build_runtime(settings)
    snapshot = runtime.snapshot
    outcome = ArmOutcome(
        name=name,
        snapshot_version=(snapshot.version if snapshot else "none"),
        certified_refs=len(snapshot.certified_refs) if snapshot else 0,
        metrics=len(snapshot.metrics) if snapshot else 0,
        dimensions=len(snapshot.dimensions) if snapshot else 0,
        certified_objects=frozenset(
            [f"metric:{m.id}" for m in (snapshot.metrics if snapshot else [])]
            + [f"dimension:{d.id}" for d in (snapshot.dimensions if snapshot else [])]
        ),
    )

    # THE SOURCE MUST BE THE ONE THE QUESTIONS ARE ABOUT. Enriching the wrong database succeeds --
    # any populated Postgres profiles fine -- and then every answer is wrong for a reason that has
    # nothing to do with grounding, which is the most expensive way to learn a DSN was stale. The
    # gold SQL names the tables the corpus asks about, so requiring the snapshot to carry at least
    # one of them separates "pointed at the wrong database" from "answered badly".
    mismatch = source_mismatch(snapshot, cases)
    if mismatch:
        raise RuntimeError(f"{name}: source {settings.source_id!r} -- {mismatch}")

    def engine(question: str):
        return runtime.ask(question, IDENTITY)

    # `run_case` is the grader the rest of this repo uses: it executes the gold SQL and the
    # engine's SQL and compares RESULT SETS. Reused rather than re-implemented so this experiment
    # is scored the same way every other mnemiq eval is.
    outcome.results = [run_case(case, engine, runtime.adapter) for case in cases]
    return outcome


def run_ablation(base: Settings, cases: Sequence[EvaluationCase], *, store_dir: Path,
                 records_url: str, traces_url: str | None, authz_path: str) -> AblationReport:
    """Both arms, then the gate, then the cuts -- in that order, deliberately.

    The gate is computed from the arms and reported ABOVE the numbers, and `render` refuses to
    print any accuracy at all when it fails. A run that cannot show its instrument was connected
    has not measured anything, and printing a table anyway is how it gets quoted later.
    """
    # A FRESH pull, every run. `fetch_certified_records` serves last-known-good from the sidecar
    # when Verity is unreachable, so a second run into the same directory would green the gate's
    # "records were read" check from cache with nothing listening on the other end -- the one state
    # the gate exists to refuse. Clearing the store too, because a surviving snapshot would let an
    # arm answer from the previous run's meaning.
    for arm_name in (BARE, GROUNDED):
        for stale in (store_dir / f"{arm_name}.duckdb",
                      store_dir / f"{arm_name}-verity-watermark.json"):
            if stale.exists():
                stale.unlink()

    bare = run_arm(
        BARE,
        arm_settings(base, name=BARE, store_dir=store_dir, records_url=None,
                     traces_url=traces_url, authz_path=authz_path),
        cases,
    )
    grounded = run_arm(
        GROUNDED,
        arm_settings(base, name=GROUNDED, store_dir=store_dir, records_url=records_url,
                     traces_url=traces_url, authz_path=authz_path),
        cases,
    )
    return AblationReport(
        bare=bare,
        grounded=grounded,
        gate=check_gate(bare, grounded),
        cases={case.id: case for case in cases},
    )


def load_golden(path: str | Path) -> list[EvaluationCase]:
    """Read the exported golden set: questions paired with the certified SQL that answers them.

    The file is what `semantic_cli export-golden` writes. Its field names are already
    `EvaluationCase`'s, so this is a validation step and not a translation -- a mismatch is a
    contract change between the two repos and should fail here, loudly, rather than be papered over
    with a mapping that hides it.
    """
    raw = json.loads(Path(path).read_text())
    cases = [EvaluationCase.model_validate(case) for case in raw]
    # `EvaluationCase` does not forbid extra fields, so a RENAMED field on the exporter -- the
    # likeliest cross-repo contract change -- is dropped in silence and the case arrives with
    # `gold_sql=None`. Validation alone therefore does not deliver the loud failure this docstring
    # promises; the fields this file actually depends on have to be required by name.
    missing = [
        case.id for case in cases
        if not (case.gold_sql or "").strip() or not (case.question or "").strip()
    ]
    if missing:
        raise ValueError(
            f"{Path(path).name}: {len(missing)} case(s) carry no question or no gold_sql "
            f"({missing[:5]}). A case with no gold query cannot be graded, and the usual cause is "
            "a field renamed on the producing side -- which validation drops silently, because "
            "unknown keys are ignored."
        )
    return cases
