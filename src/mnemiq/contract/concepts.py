"""Code systems as the contract carries them: a scheme and its concepts (M129).

They lived in `mnemiq.ontology.records`, and the contract's certified-record union needs
`ConceptScheme` -- so `mnemiq.contract` imported the ontology package while the ontology package
imported `mnemiq.contract`, and whichever was imported first decided whether the import worked:
`import mnemiq.ontology` on its own raised ImportError. The contract is the lower layer; the types
it carries live in it, and `mnemiq.ontology.records` re-exports them unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
