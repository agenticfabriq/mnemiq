"""The governed measurement arm narrows something, or it is not a measurement.

Spec item 5. Every other eval arm grants everything by design, so the disclosure path has never
run end to end under a real policy -- which is how four defects reached review in a feature whose
unit tests all passed.

The trap this file exists for: a governed arm that grants everything anyway, or filters a table
nobody queries, or masks a level no column carries. It runs, it reports zero disclosures, and zero
is indistinguishable from a disclosure path that is broken. So the plan is resolved against the
snapshot BEFORE the arm runs, and refuses to look governed when it is not.
"""

from mnemiq.contract import Column, Snapshot
from mnemiq.eval.governed import governed_grants


def _snap():
    return Snapshot(
        version="v", source_id="s", created_at="t",
        columns=[
            Column(id="claim.id", object_id="claim", name="id"),
            Column(id="claim.ssn", object_id="claim", name="ssn", pii_level="high"),
            Column(id="person.email", object_id="person", name="email", pii_level="low"),
        ],
    )


def test_the_masked_level_is_withheld_from_clearance():
    """A level in BOTH `pii_clearance` and `pii_mask` is seen raw. Putting it in both produces a
    policy that reads as governed and masks nothing -- the arm would run and measure the
    ungoverned engine under a governed name."""
    plan = governed_grants(_snap(), ["claim", "person"], mask_level="high")
    assert "high" in plan.grants.pii_mask
    assert "high" not in plan.grants.pii_clearance
    assert "low" in plan.grants.pii_clearance, "unrelated levels stay cleared"


def test_it_names_the_columns_the_mask_actually_reaches():
    plan = governed_grants(_snap(), ["claim", "person"], mask_level="high")
    assert plan.masked_columns == ("claim.ssn",)
    assert plan.narrows_something


def test_a_level_NO_column_carries_does_not_look_governed():
    """The silent-null-arm trap: a policy that cannot narrow this corpus must say so rather than
    produce a zero the kill criterion would read as a result."""
    plan = governed_grants(_snap(), ["claim", "person"], mask_level="nonexistent")
    assert plan.masked_level is None and plan.masked_columns == ()
    assert not plan.narrows_something


def test_a_filter_on_a_table_outside_the_corpus_does_not_look_governed():
    plan = governed_grants(_snap(), ["claim", "person"], filter_table="not_queried")
    assert plan.filtered_table is None
    assert not plan.narrows_something


def test_a_filter_on_a_real_table_reaches_the_grants():
    plan = governed_grants(_snap(), ["claim", "person"],
                           filter_table="claim", filter_predicate="amount > 0")
    assert plan.grants.row_filters == {"claim": "amount > 0"}
    assert plan.narrows_something


def test_build_engine_still_grants_everything_when_no_plan_is_given():
    """The existing arms must not move. Their comment is explicit that full access is deliberate,
    and a benchmark that quietly started denying PII would under-count every attempt."""
    import inspect

    from mnemiq.eval import engine

    src = inspect.getsource(engine.build_engine)
    assert "if grants is None:" in src
    assert "pii_clearance=levels" in src, "the ungoverned default must still clear every level"
