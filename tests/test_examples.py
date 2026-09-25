import pyarrow as pa
import pytest

from mnemiq.contract import Column, Snapshot
from mnemiq.enrichment.examples import FakeExampleGenerator, enrich_examples


def _snapshot():
    return Snapshot(
        version="v1", source_id="acme", created_at="t",
        columns=[
            Column(id="claim.id", object_id="claim", name="id", data_type="BIGINT"),
            Column(id="claim.amount", object_id="claim", name="amount", data_type="DECIMAL"),
        ],
    )


class _Adapter:
    """Approves any SELECT via EXPLAIN; returns rows for the 'amount' query, empty otherwise."""

    def execute(self, sql):
        return []  # EXPLAIN inside decide()

    def execute_arrow(self, sql, timeout_s=None):
        if "amount" in sql:
            return pa.table({"n": [3]})
        return pa.table({"n": []})  # 0 rows -> dropped


def test_enrich_examples_keeps_only_approved_executed_nonempty():
    gen = FakeExampleGenerator({"claim": [
        {"question": "total amount?", "sql": "SELECT sum(amount) AS n FROM claim"},   # keep
        {"question": "empty one", "sql": "SELECT count(id) AS n FROM claim WHERE id < 0"},  # 0 rows
        {"question": "bad col", "sql": "SELECT nope FROM claim"},                     # decider refuses
    ]})
    out = enrich_examples(_snapshot(), gen, _Adapter(), dialect="duckdb")
    assert len(out.examples) == 1
    ex = out.examples[0]
    assert ex.object_id == "claim" and "amount" in ex.sql and "claim" in ex.tables


def test_enrich_examples_caps_per_table():
    gen = FakeExampleGenerator({"claim": [
        {"question": f"q{i}", "sql": "SELECT sum(amount) AS n FROM claim"} for i in range(5)
    ]})
    out = enrich_examples(_snapshot(), gen, _Adapter(), dialect="duckdb", per_table=3)
    assert len(out.examples) == 3


# -- the fan-out guard (M109) ---------------------------------------------------------------------
# Examples are decider-screened and then shown to the model as verified patterns, so an inflated
# example the guard would refuse at ask time must not survive enrichment with the guard on.

_FANOUT_SQL = (
    "SELECT SUM(c.paid_amount_cents) / SUM(p.earned_premium_cents) AS loss_ratio "
    "FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id"
)


def _two_fact_snapshot():
    """Both facts reference dim_policy, so its card (and its visible set) reaches both."""
    from mnemiq.contract import Relationship

    profile = {
        "fact_claim": {"policy_id": (30000, 23322, 0), "paid_amount_cents": (30000, 29000, 0)},
        "fact_premium": {"policy_id": (125000, 50000, 0),
                         "earned_premium_cents": (125000, 90000, 0)},
        "dim_policy": {"policy_id": (50000, 50000, 0)},
    }
    return Snapshot(
        version="v1", source_id="acme", created_at="t",
        columns=[Column(id=f"{t}.{c}", object_id=t, name=c, data_type="BIGINT",
                        row_count=r, distinct_count=d, null_count=n)
                 for t, cols in profile.items() for c, (r, d, n) in cols.items()],
        relationships=[
            Relationship(id="r1", from_="fact_claim", to="dim_policy", cardinality="many_to_one"),
            Relationship(id="r2", from_="fact_premium", to="dim_policy", cardinality="many_to_one"),
        ],
    )


def _fanout_examples(guard_fanout: bool | None):
    gen = FakeExampleGenerator({"dim_policy": [{"question": "loss ratio?", "sql": _FANOUT_SQL}]})
    return enrich_examples(_two_fact_snapshot(), gen, _Adapter(), dialect="duckdb",
                           guard_fanout=guard_fanout).examples


@pytest.mark.parametrize("guard", [True, None])
def test_the_fanout_guard_keeps_no_inflated_example(guard):
    """Auto screens too, though this snapshot records no partition profiling: a missing partition
    fact can only drop a valid example, and the examples are cached as screened."""
    assert _fanout_examples(guard_fanout=guard) == []


def test_without_the_fanout_guard_the_inflated_example_is_kept():
    """The control: the fakes do execute it, so the guard is what removes it."""
    (kept,) = _fanout_examples(guard_fanout=False)
    assert "fact_premium" in kept.sql
