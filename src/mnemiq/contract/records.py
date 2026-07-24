from __future__ import annotations

from typing import Union

from pydantic import BaseModel, model_validator

from mnemiq.contract.semantic import (
    Column,
    Definition,
    Dimension,
    Example,
    Metric,
    Relationship,
    TableFacts,
)
from mnemiq.ontology.records import ConceptScheme


class Provenance(BaseModel):
    """A portable trust marker: who vouched for a record and when. Optional -- unset on the
    local/open path, filled by the governed producer. Deliberately holds no governance
    machinery; `evidence_ref` is an opaque pointer, never the evidence content."""
    status: str                        # "certified" | "imported"
    certifier: str | None = None
    certified_at: str | None = None
    evidence_ref: str | None = None


class RecordEnvelope(BaseModel):
    """The thin wrapper around a per-object payload. `version` is a content hash of the payload
    (supplied by the producer); `provenance` is None until a governed producer sets it."""
    object_type: str
    object_id: str
    version: str
    source_system: str
    provenance: Provenance | None = None


PAYLOAD_TYPES: dict[str, type] = {
    "column": Column,
    "definition": Definition,
    "metric": Metric,
    "dimension": Dimension,
    "relationship": Relationship,
    "concept_scheme": ConceptScheme,
    "table_facts": TableFacts,
    "example": Example,
}

_Payload = Union[
    Column, Definition, Metric, Dimension, Relationship, ConceptScheme, TableFacts, Example
]


class CertifiedRecord(BaseModel):
    """One certifiable object plus its envelope. `object_type` in the envelope discriminates the
    payload; the payload is a reused domain type, never a parallel record type."""
    envelope: RecordEnvelope
    payload: _Payload

    @model_validator(mode="before")
    @classmethod
    def _coerce_payload(cls, data):
        # Resolve the payload to the exact type named by object_type, so a round-trip is
        # unambiguous rather than left to union guessing.
        if isinstance(data, dict) and isinstance(data.get("payload"), dict):
            object_type = (data.get("envelope") or {}).get("object_type")
            payload_type = PAYLOAD_TYPES.get(object_type)
            if payload_type is None:
                raise ValueError(f"unknown object_type: {object_type!r}")
            data = dict(data)
            data["payload"] = payload_type.model_validate(data["payload"])
        return data

    @model_validator(mode="after")
    def _payload_matches_object_type(self):
        expected = PAYLOAD_TYPES.get(self.envelope.object_type)
        if expected is None:
            raise ValueError(f"unknown object_type: {self.envelope.object_type!r}")
        if not isinstance(self.payload, expected):
            raise ValueError(
                f"payload {type(self.payload).__name__} does not match "
                f"object_type {self.envelope.object_type!r}"
            )
        return self
