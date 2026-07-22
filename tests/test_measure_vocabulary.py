from mnemiq.contract import CodedValue, Column, Snapshot
from mnemiq.enrichment.enricher import FakeEnricher
from mnemiq.enrichment.semantic import enrich_semantic


def _snapshot(name: str) -> Snapshot:
    return Snapshot(
        version="structural",
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        columns=[
            Column(
                id=f"fireclaim.{name}",
                object_id="fireclaim",
                name=name,
                data_type="text",
                coded_values=[CodedValue(code="0.7"), CodedValue(code="0.8")],
            )
        ],
    )


def _reply(name: str, semantic_type: str) -> str:
    return (
        f'{{"columns": [{{"name": "{name}", "description": "A number.", '
        f'"semantic_type": "{semantic_type}", "pii_level": "none", '
        f'"code_meanings": {{"0.7": "the value 0.7", "0.8": "the value 0.8"}}}}]}}'
    )


def test_measures_carry_no_coded_values():
    # a sample of 11 distinct ratios is not a controlled vocabulary; "0.7 means 0.7" is noise
    for measure in ("ratio", "amount", "count"):
        out = enrich_semantic(
            _snapshot("loss_ratio"), FakeEnricher({"fireclaim": _reply("loss_ratio", measure)})
        )
        column = out.columns[0]
        assert column.semantic_type == measure
        assert column.description  # still documented
        assert column.coded_values == [], f"{measure} must not carry a vocabulary"


def test_real_vocabularies_survive():
    # A real vocabulary is kept (unlike a measure, which is dropped) -- but the LLM no longer
    # invents code meanings, so ungrounded codes survive bare (grounded-or-bare).
    out = enrich_semantic(
        _snapshot("status"), FakeEnricher({"fireclaim": _reply("status", "code")})
    )
    coded = out.columns[0].coded_values
    assert {cv.code for cv in coded} == {"0.7", "0.8"}   # the vocabulary survives
    assert all(cv.meaning is None for cv in coded)        # LLM no longer assigns meanings
