import json

from mnemiq.contract import Column, Snapshot
from mnemiq.enrichment.facts import FakeFactsEnricher, enrich_table_facts, parse_facts


def _snapshot():
    return Snapshot(
        version="v1", source_id="acme", created_at="t",
        columns=[
            Column(id="claim.amount", object_id="claim", name="amount", data_type="DECIMAL"),
            Column(id="claim.id", object_id="claim", name="id", data_type="BIGINT"),
            Column(id="claim.created", object_id="claim", name="created", data_type="DATE"),
        ],
    )


def test_parse_facts_screens_measures_and_time_column():
    raw = json.dumps({
        "grain": "one row per claim",
        "gotchas": ["  amount is stored as text; cast first  ", ""],
        "canonical_measures": {
            "total_amount": "sum(amount)",            # real column -> kept
            "bogus": "sum(nonexistent_col)",          # unknown column -> dropped
            "claim_count": "count(distinct id)",      # real -> kept
        },
        "default_time_column": "created",
    })
    tf = parse_facts(raw, "claim", columns={"amount", "id", "created"}, related=set())
    assert tf.grain == "one row per claim"
    assert tf.gotchas == ["amount is stored as text; cast first"]  # blank dropped, trimmed
    assert set(tf.canonical_measures) == {"total_amount", "claim_count"}  # bogus dropped
    assert tf.default_time_column == "created"


def test_parse_facts_rejects_time_column_not_in_table():
    tf = parse_facts(json.dumps({"default_time_column": "not_a_col"}), "claim",
                     columns={"amount"}, related=set())
    assert tf.default_time_column is None


def test_enrich_table_facts_merges_and_is_fail_soft():
    enricher = FakeFactsEnricher({"claim": json.dumps({"grain": "one row per claim"})})
    out = enrich_table_facts(_snapshot(), enricher)
    facts = {t.object_id: t for t in out.table_facts}
    assert facts["claim"].grain == "one row per claim"

    class _Boom:
        def facts(self, table, context, columns, related):
            raise RuntimeError("rate limited")

    out2 = enrich_table_facts(_snapshot(), _Boom())  # fail-soft: no facts, no raise
    assert out2.table_facts == []
