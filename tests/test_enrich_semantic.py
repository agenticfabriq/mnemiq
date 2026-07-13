from mnemiq.contract import CodedValue, Column, Snapshot
from mnemiq.enrichment.enricher import FakeEnricher
from mnemiq.enrichment.semantic import enrich_semantic


def _snapshot() -> Snapshot:
    return Snapshot(
        version="structural",
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        columns=[
            Column(
                id="fireclaim.fireplace",
                object_id="fireclaim",
                name="fireplace",
                data_type="text",
                coded_values=[CodedValue(code="yes"), CodedValue(code="no")],
            ),
            Column(
                id="fireclaim.claim_identifier",
                object_id="fireclaim",
                name="claim_identifier",
                data_type="integer",
            ),
        ],
    )


GOOD = """{"columns": [
    {"name": "fireplace", "description": "Whether the property has a fireplace.",
     "semantic_type": "boolean", "pii_level": "none",
     "code_meanings": {"yes": "has a fireplace", "no": "has no fireplace"}},
    {"name": "claim_identifier", "description": "Reference to the claim.",
     "semantic_type": "identifier", "pii_level": "none"}
]}"""


def test_annotations_land_on_the_columns():
    out = enrich_semantic(_snapshot(), FakeEnricher({"fireclaim": GOOD}))

    fireplace = next(c for c in out.columns if c.name == "fireplace")
    assert fireplace.description == "Whether the property has a fireplace."
    assert fireplace.semantic_type == "boolean"
    assert fireplace.pii_level == "none"
    assert {cv.code: cv.meaning for cv in fireplace.coded_values} == {
        "yes": "has a fireplace",
        "no": "has no fireplace",
    }

    claim = next(c for c in out.columns if c.name == "claim_identifier")
    assert claim.semantic_type == "identifier"


def test_structural_facts_are_never_overwritten():
    out = enrich_semantic(_snapshot(), FakeEnricher({"fireclaim": GOOD}))
    fireplace = next(c for c in out.columns if c.name == "fireplace")
    assert fireplace.data_type == "text"  # the LLM does not get to change types
    assert {cv.code for cv in fireplace.coded_values} == {"yes", "no"}  # nor the codes


def test_the_enricher_sees_the_codes_it_must_explain():
    fake = FakeEnricher({"fireclaim": GOOD})
    enrich_semantic(_snapshot(), fake)
    assert fake.calls == ["fireclaim"]  # one call per table


def test_the_input_snapshot_is_not_mutated():
    original = _snapshot()
    enrich_semantic(original, FakeEnricher({"fireclaim": GOOD}))
    assert all(c.description is None for c in original.columns)
    assert all(cv.meaning is None for c in original.columns for cv in c.coded_values)
    assert original.version == "structural"


def test_version_changes_because_content_changed():
    original = _snapshot()
    out = enrich_semantic(original, FakeEnricher({"fireclaim": GOOD}))
    assert out.version != original.version
    assert out.source_id == original.source_id


def test_a_table_the_enricher_fails_is_marked_failed_and_left_alone():
    out = enrich_semantic(_snapshot(), FakeEnricher({}))  # no reply for any table

    assert all(c.description is None for c in out.columns)
    assert all(cv.meaning is None for c in out.columns for cv in c.coded_values)
    job = next(j for j in out.jobs if j.id == "semantic:fireclaim")
    assert job.status == "failed"


def test_a_raising_enricher_does_not_sink_the_run():
    class _Exploding:
        def annotate(self, table, facts):
            raise RuntimeError("429 rate limited")

    out = enrich_semantic(_snapshot(), _Exploding())
    assert next(j for j in out.jobs if j.id == "semantic:fireclaim").status == "failed"
    assert all(c.description is None for c in out.columns)


def test_pii_classification_scrubs_the_harvested_values():
    # Second line of defense: the column name gave nothing away, so profiling harvested its
    # values -- but the model recognized them as personal data. Those values ARE the PII.
    snapshot = Snapshot(
        version="structural",
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        columns=[
            Column(
                id="t.handle",
                object_id="t",
                name="handle",
                data_type="text",
                coded_values=[CodedValue(code="alice99"), CodedValue(code="bob42")],
            )
        ],
    )
    reply = """{"columns": [{"name": "handle", "description": "The person's login handle.",
                "semantic_type": "name", "pii_level": "pii",
                "code_meanings": {"alice99": "the user Alice", "bob42": "the user Bob"}}]}"""

    out = enrich_semantic(snapshot, FakeEnricher({"t": reply}))
    handle = out.columns[0]

    assert handle.pii_level == "pii"
    assert handle.description  # we still document it
    assert handle.coded_values == []  # but we do not carry real people in the snapshot


def test_a_successful_table_is_marked_done():
    out = enrich_semantic(_snapshot(), FakeEnricher({"fireclaim": GOOD}))
    job = next(j for j in out.jobs if j.id == "semantic:fireclaim")
    assert job.status == "done"
    assert job.kind == "semantic"
