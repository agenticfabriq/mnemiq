from __future__ import annotations

from pydantic import BaseModel


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
