"""The ablation's gate and its report.

What is guarded here is the EXPERIMENT's honesty, not the engine's accuracy. Running the arms needs
a database, an LLM and a live Verity; deciding whether a run may be reported does not, and that
decision is the part that goes wrong silently.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mnemiq.contract import EvaluationCase
from mnemiq.eval.ablation import (
    BARE,
    GROUNDED,
    AblationReport,
    ArmOutcome,
    check_gate,
    load_golden,
)
from mnemiq.eval.harness import CaseResult, Outcome


def _arm(name: str, *, version: str, refs: int, metrics: int, dimensions: int,
         objects: frozenset[str] | None = None,
         results: list[CaseResult] | None = None) -> ArmOutcome:
    return ArmOutcome(
        name=name,
        snapshot_version=version,
        certified_refs=refs,
        metrics=metrics,
        dimensions=dimensions,
        certified_objects=objects if objects is not None else frozenset(
            [f"metric:m{i}" for i in range(metrics)]
            + [f"dimension:d{i}" for i in range(dimensions)]
        ),
        results=results or [],
    )


def _grounded_run() -> tuple[ArmOutcome, ArmOutcome]:
    """The shape a healthy run has: the grounded arm gained the certified objects."""
    bare = _arm(BARE, version="aaa", refs=0, metrics=0, dimensions=0)
    grounded = _arm(GROUNDED, version="bbb", refs=10, metrics=5, dimensions=5)
    return bare, grounded


def test_a_healthy_run_passes_every_gate_check():
    assert check_gate(*_grounded_run()).passed


def test_the_gate_fails_when_the_records_fetch_degraded_to_empty():
    """The failure the gate exists for: `fetch_certified_records` is fail-soft.

    A 401, a moved URL or a shape change returns an empty set and the run continues. That arm then
    scores exactly like bare, and the honest-looking conclusion "the layer does not help" would be
    indistinguishable from "the layer was never connected".
    """
    bare, grounded = _grounded_run()
    grounded.certified_refs = 0
    gate = check_gate(bare, grounded)
    assert not gate.passed
    assert any("records were read" == name and not passed for name, passed, _ in gate.checks)


def test_the_gate_fails_when_grounding_added_nothing():
    bare, grounded = _grounded_run()
    grounded.metrics, grounded.dimensions = 0, 0
    gate = check_gate(bare, grounded)
    assert not gate.passed
    assert any("the snapshot changed" == name and not passed for name, passed, _ in gate.checks)


def test_the_gate_fails_when_both_arms_carry_the_same_meaning():
    """Two arms holding the same certified objects are one arm run twice.

    Deliberately NOT a digest comparison. `snapshot.version` is `content_version`, whose body
    excludes `metrics` and `dimensions` -- measured, a snapshot with five of each hashes the same
    as one with none -- so a digest check is blind to exactly what grounding adds and would pass on
    two enrichment passes wording a column differently.
    """
    bare, grounded = _grounded_run()
    grounded.certified_objects = bare.certified_objects
    gate = check_gate(bare, grounded)
    assert not gate.passed
    assert any(name == "the arms carry different meaning" and not passed
               for name, passed, _ in gate.checks)


def test_the_gate_is_not_satisfied_by_differing_digests_alone():
    """The check the digest comparison used to make must no longer be enough to pass.

    This is the regression the old check hid: two arms whose snapshots hash differently -- which
    two independent LLM enrichment passes produce on their own -- while carrying identical
    certified meaning.
    """
    bare, grounded = _grounded_run()
    grounded.certified_objects = bare.certified_objects
    assert bare.snapshot_version != grounded.snapshot_version  # digests differ...
    assert not check_gate(bare, grounded).passed                # ...and it still fails


def _report(gate_ok: bool) -> AblationReport:
    bare, grounded = _grounded_run()
    if not gate_ok:
        grounded.certified_refs = 0
    cases = {
        "m1": EvaluationCase(id="m1", question="q", gold_sql="select 1", tags=["meaning"]),
        "c1": EvaluationCase(id="c1", question="q", gold_sql="select 1", tags=["control"]),
    }
    bare.results = [
        CaseResult(case_id="m1", outcome=Outcome.WRONG),
        CaseResult(case_id="c1", outcome=Outcome.CORRECT),
    ]
    grounded.results = [
        CaseResult(case_id="m1", outcome=Outcome.CORRECT),
        CaseResult(case_id="c1", outcome=Outcome.CORRECT),
    ]
    return AblationReport(bare=bare, grounded=grounded, gate=check_gate(bare, grounded),
                          cases=cases)


def test_a_failed_gate_prints_no_accuracy_at_all():
    """The number must not survive the check that says it is meaningless.

    Printing the table with a warning above it is not enough: the table is what gets quoted, and a
    reader three weeks later has the number without the caveat.
    """
    rendered = _report(gate_ok=False).render()
    assert "NO RESULT" in rendered
    assert "meaning-dependent" not in rendered
    assert "correct" not in rendered.split("NO RESULT")[1]


def test_the_report_shows_each_band_its_own_accuracy():
    """Both bands must be SCORED, not merely named.

    Asserting the labels appear proves nothing: `CUTS` is static and `render` prints every label
    unconditionally, adding "(no cases carry this tag)" when a band is empty. Stripping every tag
    from both cases left the old assertions passing over a report with no per-band accuracy at all.
    So this reads the numbers, and checks the untagged case is the one that stops being scored.
    """
    report = _report(gate_ok=True)
    rendered = report.render()
    # One meaning case, wrong in bare and correct in grounded; one control case, correct in both.
    assert "meaning-dependent                  1  bare             0" in rendered
    assert "schema-recoverable (control)       1  bare             1" in rendered

    for case in report.cases.values():  # the mutation: no case belongs to any band
        case.tags = []
    stripped = report.render()
    assert "(no cases carry this tag)" in stripped
    assert "meaning-dependent                  1  bare" not in stripped


def test_loading_a_golden_set_rejects_a_case_the_contract_does_not_accept(tmp_path):
    """The exported file is validated, not adapted.

    Its field names are already `EvaluationCase`'s. A mismatch means the two repos' contracts have
    parted company, and a tolerant loader would paper over exactly the change worth hearing about.
    """
    good = tmp_path / "good.json"
    good.write_text(json.dumps([
        {"id": "a", "question": "q?", "gold_sql": "select 1", "expected_answer": "1"},
    ]))
    assert [case.id for case in load_golden(good)] == ["a"]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"question": "q?", "gold_sql": "select 1"}]))  # no id
    with pytest.raises(ValidationError):
        load_golden(bad)

    # The likelier cross-repo change: a field RENAMED on the exporter. Pydantic ignores unknown
    # keys, so this validates cleanly and arrives with `gold_sql=None` -- a case that can never be
    # graded, loaded without complaint. The fields this module depends on are required by name.
    renamed = tmp_path / "renamed.json"
    renamed.write_text(json.dumps([
        {"id": "m1", "question": "q?", "certified_sql": "select 1"},
    ]))
    with pytest.raises(ValueError, match="no gold_sql"):
        load_golden(renamed)


def test_the_control_band_reports_whether_grounding_changed_the_sql():
    """Both arms at 5/5 is a ceiling; identical SQL is what the ceiling cannot say.

    A control band where both arms score full marks cannot distinguish "the certified records were
    irrelevant to these questions" from "they helped and the questions were too easy to show it".
    Whether the engine emitted the SAME query with and without them can.
    """
    bare, grounded = _grounded_run()
    cases = {
        "c1": EvaluationCase(id="c1", question="q", gold_sql="select 1", tags=["control"]),
        "c2": EvaluationCase(id="c2", question="q", gold_sql="select 1", tags=["control"]),
    }
    bare.results = [
        CaseResult(case_id="c1", outcome=Outcome.CORRECT, sql="SELECT count(*)  FROM t"),
        CaseResult(case_id="c2", outcome=Outcome.CORRECT, sql="select a from t"),
    ]
    grounded.results = [
        # Same query, different whitespace and case -- the engine did the same thing.
        CaseResult(case_id="c1", outcome=Outcome.CORRECT, sql="select COUNT(*) from t"),
        # A genuinely different query.
        CaseResult(case_id="c2", outcome=Outcome.CORRECT, sql="select distinct a from t"),
    ]
    report = AblationReport(bare=bare, grounded=grounded, gate=check_gate(bare, grounded),
                            cases=cases)
    assert "control-band SQL identical in both arms: 1/2 compared" in report.render()


def test_two_silent_arms_are_not_counted_as_agreeing():
    """`CaseResult.sql` is `""` when an arm defers or errors, and two blanks are equal.

    Counted blindly, a run where every control case deferred prints "inert in both arms" as its
    strongest falsifiability claim while no SQL was produced at all.
    """
    bare, grounded = _grounded_run()
    cases = {"c1": EvaluationCase(id="c1", question="q", gold_sql="select 1", tags=["control"])}
    bare.results = [CaseResult(case_id="c1", outcome=Outcome.DEFERRED_WRONGLY)]
    grounded.results = [CaseResult(case_id="c1", outcome=Outcome.ERROR)]
    rendered = AblationReport(bare=bare, grounded=grounded,
                              gate=check_gate(bare, grounded), cases=cases).render()
    assert "identical in both arms: 0/0 compared" in rendered
    assert "1 not compared (an arm emitted no SQL)" in rendered


def test_a_snapshot_from_the_wrong_database_is_refused():
    """Enriching the wrong source succeeds, which is what makes this worth checking.

    Any populated Postgres profiles without complaint, so a stale DSN produces a full run in which
    every answer is wrong for a reason that has nothing to do with grounding.
    """
    from mnemiq.contract import Column, Snapshot
    from mnemiq.eval.ablation import source_mismatch

    cases = [EvaluationCase(id="c", question="q",
                            gold_sql="select count(*) from payment_transaction")]

    def _snapshot(*names):
        return Snapshot(
            version="v", source_id="s", created_at="now",
            # `object_id` is the TABLE the column belongs to; `id` is the column.
            columns=[Column(id=f"{n}.c", object_id=n, name="c") for n in names],
        )

    assert source_mismatch(_snapshot("payment_transaction", "payment_split"), cases) is None
    # The real trap: mnemiq's default source is the ACME insurance database.
    complaint = source_mismatch(_snapshot("claim", "policy", "fireclaim"), cases)
    assert complaint and "different database" in complaint
    # Nothing profiled is `enrich`'s complaint to make, not this one's.
    assert source_mismatch(_snapshot(), cases) is None
