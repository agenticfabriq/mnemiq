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


def _column():
    from mnemiq.contract import Column
    return Column(id="patient.icd10_cd", object_id="patient", name="icd10_cd")


def _env(object_type, object_id):
    from mnemiq.contract import RecordEnvelope
    return RecordEnvelope(object_type=object_type, object_id=object_id,
                          version="v1", source_system="pg")


def test_record_pairs_object_type_with_its_payload():
    from mnemiq.contract import CertifiedRecord

    rec = CertifiedRecord(envelope=_env("column", "patient.icd10_cd"), payload=_column())
    assert rec.envelope.object_type == "column"
    assert rec.payload.name == "icd10_cd"


def test_record_rejects_type_payload_mismatch():
    import pytest
    from mnemiq.contract import CertifiedRecord

    with pytest.raises(ValueError):
        CertifiedRecord(envelope=_env("metric", "m1"), payload=_column())


def test_record_coerces_payload_by_object_type_on_parse():
    import json

    from mnemiq.contract import CertifiedRecord

    rec = CertifiedRecord(envelope=_env("column", "patient.icd10_cd"), payload=_column())
    reparsed = CertifiedRecord.model_validate(json.loads(rec.model_dump_json()))
    from mnemiq.contract import Column
    assert isinstance(reparsed.payload, Column)  # not mis-parsed as another union member
    assert reparsed.payload.object_id == "patient"


def test_all_eight_object_types_are_registered():
    from mnemiq.contract.records import PAYLOAD_TYPES

    assert set(PAYLOAD_TYPES) == {
        "column", "definition", "metric", "dimension",
        "relationship", "concept_scheme", "table_facts", "example",
    }


def test_unknown_object_type_is_rejected_on_parse():
    import pytest
    from mnemiq.contract import CertifiedRecord

    payload = {"id": "x.y", "object_id": "x", "name": "y"}
    with pytest.raises(ValueError):
        CertifiedRecord.model_validate({
            "envelope": {"object_type": "banana", "object_id": "x.y",
                         "version": "v1", "source_system": "pg"},
            "payload": payload,
        })
