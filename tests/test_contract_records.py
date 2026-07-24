import json


def test_envelope_defaults_provenance_to_none():
    from mnemiq.contract import RecordEnvelope

    env = RecordEnvelope(object_type="column", object_id="patient.icd10_cd",
                         version="abc123", source_system="pg-prod")
    assert env.provenance is None  # the open path leaves certification unset


def test_envelope_carries_certification_provenance_and_round_trips():
    from mnemiq.contract import Provenance, RecordEnvelope

    env = RecordEnvelope(
        object_type="metric", object_id="net_revenue", version="v1", source_system="pg-prod",
        provenance=Provenance(status="certified", certifier="alice", certified_at="2026-07-23"),
    )
    again = RecordEnvelope.model_validate(json.loads(env.model_dump_json()))
    assert again.provenance.status == "certified"
    assert again.provenance.certifier == "alice"
    assert again.provenance.evidence_ref is None
