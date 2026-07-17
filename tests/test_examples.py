import pyarrow as pa

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
