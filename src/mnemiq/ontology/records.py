from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field, ValidationError

from mnemiq.contract.semantic import Definition


class Concept(BaseModel):
    """One member of a code system. `notation` is the ONLY join key to stored data -- every
    match downstream compares stored values against it, never against a label."""
    id: str
    notation: str
    pref_label: str
    alt_labels: list[str] = Field(default_factory=list)
    definition: str | None = None
    broader: list[str] = Field(default_factory=list)


class ConceptScheme(BaseModel):
    id: str
    label: str
    description: str | None = None
    concepts: list[Concept] = Field(default_factory=list)


class OntologyRecords(BaseModel):
    """The portable artifact. Two producers -- mnemiq's lean digest and the governed plane --
    emit this same shape; the engine cannot tell them apart, which is the point.

    `bindings` carries EXPLICIT operator bindings only. Auto-binding needs observed values and
    therefore cannot happen at digest time; it runs at enrich time (see binder.py).
    """
    version: str = ""
    schemes: list[ConceptScheme] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)
    bindings: dict[str, str] = Field(default_factory=dict)  # Column.id -> scheme id


def records_version(records: OntologyRecords) -> str:
    """Hash what the records assert, not when they were built."""
    payload = records.model_dump(exclude={"version"})
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def merge_records(local, certified_schemes):
    """Union local-digest schemes with governed (Verity-certified) schemes by scheme id; certified
    wins on a collision -- the governed source outranks the local digest, matching the
    certified > ontology precedence. Local definitions and explicit bindings are preserved."""
    by_id = {s.id: s for s in (local.schemes if local else [])}
    for scheme in certified_schemes:
        by_id[scheme.id] = scheme  # certified overrides local on id collision
    records = OntologyRecords(
        schemes=sorted(by_id.values(), key=lambda s: s.id),
        definitions=list(local.definitions) if local else [],
        bindings=dict(local.bindings) if local else {},
    )
    records.version = records_version(records)
    return records


def load_records(path: str) -> OntologyRecords:
    """Parse an ontology records artifact (JSON). Raises ValueError on malformed input."""
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return OntologyRecords.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, OSError) as exc:
        raise ValueError(f"invalid ontology records at {path}: {exc}") from exc
