"""M1 -- an unverified trust marker must not open a PII gate.

`apply_certified` used to drop the envelope on its first line and overlay `pii_level` from whatever
arrived. `_qualifies` keeps a column out of the value index when `pii_level` is in `_SENSITIVE_PII`,
so a record setting it to "none" put the column back in scope and `build_value_index` stored every
distinct value. The ordering compounded it: the overlaid column is added to `_protected`, so the LLM
is then forbidden from correcting the level that was just overridden.

The producer half is Verity's D56 -- it hardcoded `"status": "certified"` and left the certifier null
for a whole class of records. Each half is defensible alone; only the seam is a defect.
"""

import logging

from mnemiq.contract import Column, Snapshot
from mnemiq.contract.records import CertifiedRecord, Provenance, RecordEnvelope
from mnemiq.enrichment.certified import apply_certified
from mnemiq.semantic.values import _qualifies


def _snapshot(pii_level: str = "phi") -> Snapshot:
    return Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-31T00:00:00Z",
        columns=[
            Column(
                id="patient.dx_code",
                object_id="patient",
                name="dx_code",
                data_type="varchar",
                distinct_count=12,
                pii_level=pii_level,
                description="local guess",
            )
        ],
    )


def _record(provenance: Provenance | None, pii_level: str = "none") -> CertifiedRecord:
    return CertifiedRecord.model_validate(
        {
            "envelope": {
                "object_type": "column",
                "object_id": "patient.dx_code",
                "version": "h1",
                "source_system": "verity",
                **({"provenance": provenance.model_dump()} if provenance else {}),
            },
            "payload": {
                "id": "patient.dx_code",
                "object_id": "patient",
                "name": "dx_code",
                "pii_level": pii_level,
                "description": "governed meaning",
                "semantic_type": "code",
            },
        }
    )


_ATTESTED = Provenance(
    status="certified", certifier="reviewer@acme.example", certified_at="2026-07-30T00:00:00Z"
)
# Exactly what Verity published before D56: the status asserted, nobody named.
_UNATTESTED = Provenance(status="certified", certifier=None, certified_at=None)


def test_an_unattested_record_cannot_open_a_pii_gate():
    updated = apply_certified(_snapshot("phi"), [_record(_UNATTESTED, pii_level="none")])
    column = updated.columns[0]

    assert column.pii_level == "phi", (
        "a record asserting certification with nobody attesting it must not weaken a PII level"
    )
    assert not _qualifies(column, max_distinct=200), (
        "and the column must stay out of the value index, which is the actual consequence"
    )


def test_a_record_with_no_provenance_at_all_cannot_open_a_pii_gate():
    updated = apply_certified(_snapshot("pii"), [_record(None, pii_level="none")])

    assert updated.columns[0].pii_level == "pii"


def test_an_unattested_record_still_contributes_meaning():
    # Deliberately narrow: the seam's danger is one field. Refusing whole records would degrade
    # enrichment for a provenance shape Verity emitted for two months.
    updated = apply_certified(_snapshot("phi"), [_record(_UNATTESTED, pii_level="none")])
    column = updated.columns[0]

    assert column.description == "governed meaning"
    assert column.semantic_type == "code"


def test_an_attested_record_may_set_the_level_in_either_direction():
    # Correcting a wrong local guess is exactly what governed meaning is for.
    relaxed = apply_certified(_snapshot("phi"), [_record(_ATTESTED, pii_level="none")])
    assert relaxed.columns[0].pii_level == "none"
    assert _qualifies(relaxed.columns[0], max_distinct=200)

    tightened = apply_certified(_snapshot("none"), [_record(_ATTESTED, pii_level="phi")])
    assert tightened.columns[0].pii_level == "phi"


def test_an_unattested_record_may_still_tighten():
    updated = apply_certified(_snapshot("none"), [_record(_UNATTESTED, pii_level="phi")])

    assert updated.columns[0].pii_level == "phi", "raising sensitivity is never the risk"


def test_a_refused_downgrade_is_recorded_as_a_job_and_logged(caplog):
    # M2's lesson, applied the same day: a silent security downgrade is worse than the downgrade.
    # `snapshot.jobs` is the codebase's own structured run record, so the refusal goes there rather
    # than into a new field invented on the contract.
    with caplog.at_level(logging.WARNING):
        updated = apply_certified(_snapshot("phi"), [_record(_UNATTESTED, pii_level="none")])

    refusals = [job for job in updated.jobs if job.kind == "certified_pii_downgrade_refused"]
    assert len(refusals) == 1, f"no job recorded the refusal: {updated.jobs}"
    assert refusals[0].status == "refused"
    assert "patient.dx_code" in refusals[0].checkpoints

    messages = [record.getMessage() for record in caplog.records]
    assert any("patient.dx_code" in message for message in messages), (
        f"the refusal must name the column: {messages}"
    )


def test_a_clean_overlay_records_no_refusal_job():
    updated = apply_certified(_snapshot("none"), [_record(_ATTESTED, pii_level="none")])

    assert not [job for job in updated.jobs if job.kind == "certified_pii_downgrade_refused"]
