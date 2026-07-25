from __future__ import annotations

import logging
from collections.abc import Sequence

from mnemiq.contract import Definition
from mnemiq.ontology.records import Concept, ConceptScheme, OntologyRecords, records_version

logger = logging.getLogger(__name__)

# Standard vocabularies ONLY. A source-specific namespace must never appear here: the same
# digest has to work against any publisher's TTL.
SKOS = "http://www.w3.org/2004/02/skos/core#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"


def _first(graph, subject, *predicates) -> str | None:
    """First literal found across predicates, in preference order."""
    from rdflib import URIRef

    for predicate in predicates:
        for value in graph.objects(subject, URIRef(predicate)):
            text = str(value).strip()
            if text:
                return text
    return None


def _all(graph, subject, predicate) -> list[str]:
    from rdflib import URIRef

    return [str(v).strip() for v in graph.objects(subject, URIRef(predicate)) if str(v).strip()]


def _label(graph, subject) -> str:
    return _first(graph, subject, SKOS + "prefLabel", RDFS + "label") or str(subject)


def _concept(graph, subject) -> Concept | None:
    notation = _first(graph, subject, SKOS + "notation")
    if not notation:
        return None  # nothing to match stored values against -- it can never ground a code
    return Concept(
        id=str(subject),
        notation=notation,
        pref_label=_label(graph, subject),
        alt_labels=sorted(_all(graph, subject, SKOS + "altLabel")),
        definition=_first(graph, subject, SKOS + "definition", RDFS + "comment"),
        broader=sorted(_all(graph, subject, SKOS + "broader")),
    )


def digest_ontology(paths: Sequence[str], public: bool = False) -> OntologyRecords:
    """Deterministic TTL/SKOS/OWL -> records. No LLM, no network, no inference.

    Two shapes are recognised: an explicit skos:ConceptScheme whose members declare
    skos:inScheme, and (fallback) an owl:Class whose rdfs:subClassOf children carry notations.

    `public` marks the produced definitions as a public standard (visible to every identity).
    It defaults False -- fail-closed -- so a confidential taxonomy is not globally visible unless
    the operator declares the ontology public (`mnemiq digest-ontology --public`).
    """
    from rdflib import RDF, Graph, URIRef  # lazy: rdflib is the optional `ontology` extra

    graph = Graph()
    for path in paths:
        graph.parse(path, format="turtle")

    schemes: list[ConceptScheme] = []
    members: set[str] = set()

    # -- shape 1: explicit SKOS concept schemes
    for subject in graph.subjects(RDF.type, URIRef(SKOS + "ConceptScheme")):
        concepts = []
        for member in graph.subjects(URIRef(SKOS + "inScheme"), subject):
            members.add(str(member))
            concept = _concept(graph, member)
            if concept is not None:
                concepts.append(concept)
        schemes.append(ConceptScheme(
            id=str(subject),
            label=_label(graph, subject),
            description=_first(graph, subject, SKOS + "definition", RDFS + "comment"),
            concepts=sorted(concepts, key=lambda c: c.notation),
        ))

    # -- shape 2: OWL fallback -- a class whose subclasses carry notations is a scheme
    known = {s.id for s in schemes}
    for parent in graph.subjects(RDF.type, URIRef(OWL + "Class")):
        if str(parent) in known:
            continue
        concepts = []
        for child in graph.subjects(URIRef(RDFS + "subClassOf"), parent):
            concept = _concept(graph, child)
            if concept is not None:
                members.add(str(child))
                concepts.append(concept)
        if concepts:
            schemes.append(ConceptScheme(
                id=str(parent),
                label=_label(graph, parent),
                description=_first(graph, parent, SKOS + "definition", RDFS + "comment"),
                concepts=sorted(concepts, key=lambda c: c.notation),
            ))

    # -- definitions: one per scheme, plus classes that belong to no scheme.
    # A scheme MEMBER never becomes a Definition: select_definitions runs a regex per
    # definition per question, so one Definition per code would be slow and noisy. Member
    # meanings stay reachable through the concept index, which is keyed by scheme.
    definitions: list[Definition] = []
    for scheme in schemes:
        if scheme.description:
            definitions.append(Definition(
                id=f"ontology:scheme:{scheme.id}",
                term=scheme.label,
                domain="ontology",
                definition=scheme.description,
                public=public,
            ))
    scheme_ids = {s.id for s in schemes}
    for subject in graph.subjects(RDF.type, URIRef(OWL + "Class")):
        key = str(subject)
        if key in members or key in scheme_ids:
            continue
        text = _first(graph, subject, SKOS + "definition", RDFS + "comment")
        if text:
            definitions.append(Definition(
                id=f"ontology:term:{key}",
                term=_label(graph, subject),
                domain="ontology",
                definition=text,
                public=public,
            ))

    records = OntologyRecords(
        schemes=sorted(schemes, key=lambda s: s.id),
        definitions=sorted(definitions, key=lambda d: d.id),
    )
    records.version = records_version(records)
    logger.info("digested %d scheme(s), %d definition(s)", len(schemes), len(definitions))
    return records
