from __future__ import annotations

import logging
from dataclasses import dataclass, field

from mnemiq.catalog import is_key_like, is_sensitive_name
from mnemiq.contract import CodeScheme, Snapshot
from mnemiq.ontology.records import ConceptScheme, OntologyRecords
from mnemiq.semantic.textmatch import name_affinity

logger = logging.getLogger(__name__)

CONTAINMENT_MIN = 0.95   # tolerates residual dirt after sentinel forgiveness
MIN_DISTINCT = 4         # three or fewer values match a scheme by coincidence too easily
SAMPLE_LIMIT = 1000      # bounds the large-code-system probe
AFFINITY_MIN = 0.50      # overlap-coefficient floor; see textmatch.name_affinity

# Junk that appears in essentially every real coded column. Forgiven from containment so a
# sparsely-populated standard-code column still binds. Extensible per deployment.
DEFAULT_SENTINELS = frozenset({"", "n/a", "na", "unknown", "null", "none", "-1", "?"})


@dataclass
class BindEvidence:
    """Why a column bound. Recorded so a wrong bind is diagnosable after the fact."""
    column_id: str
    scheme_id: str
    method: str  # "auto" | "explicit"
    containment: float | None = None
    matched: int | None = None
    sample_size: int | None = None
    affinity: float | None = None
    unmatched: list[str] = field(default_factory=list)


def _observed(adapter, snapshot: Snapshot, column) -> list[str]:
    """The column's value set. Regime 1 reuses the harvested vocabulary; regime 2 probes.

    Regime 2 exists because profiling harvests coded_values only below code_max_distinct, so
    a large standard code system arrives with nothing harvested at all.
    """
    if column.coded_values:
        return [cv.code for cv in column.coded_values]
    if not column.distinct_count:
        return []
    physical = {sb.object_id: sb.source_object for sb in snapshot.source_bindings}
    table = physical.get(column.object_id, column.object_id)
    try:
        rows = adapter.execute(
            f'SELECT DISTINCT "{column.name}" FROM "{table}" '
            f'WHERE "{column.name}" IS NOT NULL LIMIT {SAMPLE_LIMIT}'
        )
    except Exception as exc:
        logger.warning("ontology probe failed for %s: %s", column.id, exc)
        return []
    return [str(r[0]) for r in rows]


def _gate(column, values: list[str], scheme: ConceptScheme, notations: set[str],
          sentinels: frozenset[str]) -> BindEvidence | None:
    """Containment (sentinel-forgiving) AND name affinity. Either failing binds nothing."""
    matched, unmatched = 0, []
    for value in values:
        cleaned = value.strip().casefold()
        if cleaned in notations:
            matched += 1
        elif cleaned in sentinels:
            continue  # forgiven entirely -- junk is not evidence against the scheme
        else:
            unmatched.append(value)

    considered = matched + len(unmatched)
    if considered == 0 or matched < MIN_DISTINCT:
        return None  # the minimum counts NON-SENTINEL values: padding must not carry a column
    containment = matched / considered
    if containment < CONTAINMENT_MIN:
        return None
    affinity = name_affinity(column.name, scheme.label)
    if affinity < AFFINITY_MIN:
        return None  # containment proves the values fit; affinity makes the fit mean something
    return BindEvidence(
        column_id=column.id, scheme_id=scheme.id, method="auto",
        containment=containment, matched=matched, sample_size=len(values),
        affinity=affinity, unmatched=unmatched[:10],
    )


def bind_schemes(adapter, snapshot: Snapshot, records: OntologyRecords,
                 sentinels: frozenset[str] = DEFAULT_SENTINELS) -> Snapshot:
    """Attach a code scheme to each column that demonstrably draws from one.

    Explicit bindings are taken on the operator's word. Auto-binding is precision-first: when
    two schemes both fit, bind NEITHER -- the same rule the in-DB grounders use.
    """
    by_scheme = {s.id: s for s in records.schemes}
    notations = {
        s.id: {c.notation.strip().casefold() for c in s.concepts} for s in records.schemes
    }
    known_columns = {c.id for c in snapshot.columns}
    bound: dict[str, CodeScheme] = {}
    evidence: list[BindEvidence] = []

    for column_id, scheme_id in records.bindings.items():
        scheme = by_scheme.get(scheme_id)
        if scheme is None:
            logger.warning("explicit binding names unknown scheme %r; skipped", scheme_id)
            continue
        if column_id not in known_columns:
            logger.warning("explicit binding column %r not in schema; skipped", column_id)
            continue
        bound[column_id] = CodeScheme(id=scheme.id, label=scheme.label)
        evidence.append(BindEvidence(column_id=column_id, scheme_id=scheme.id, method="explicit"))

    for column in snapshot.columns:
        if column.id in bound or is_key_like(column.name) or is_sensitive_name(column.name):
            continue
        values = _observed(adapter, snapshot, column)
        if not values:
            continue
        passing = [
            ev for s in records.schemes
            if (ev := _gate(column, values, s, notations[s.id], sentinels)) is not None
        ]
        if len(passing) != 1:
            if len(passing) > 1:
                logger.info("ontology: %s matches %d schemes; binding nothing",
                            column.id, len(passing))
            continue
        scheme = by_scheme[passing[0].scheme_id]
        bound[column.id] = CodeScheme(id=scheme.id, label=scheme.label)
        evidence.append(passing[0])

    for ev in evidence:
        logger.info("ontology bind %s -> %s (%s, containment=%s, affinity=%s, unmatched=%s)",
                    ev.column_id, ev.scheme_id, ev.method, ev.containment, ev.affinity,
                    ev.unmatched)

    columns = [
        c.model_copy(update={"code_scheme": bound[c.id]}) if c.id in bound else c
        for c in snapshot.columns
    ]

    # Ontology definitions stay UNBOUND, i.e. visible to every identity.
    #
    # Binding them to the tables that use the scheme was backwards. select_definitions requires
    # ALL bound objects to be granted, so the more widely a scheme was used the FEWER identities
    # could see what it means -- on Pagila the MPAA definition bound to `film` plus two views
    # over it, and an analyst granted only `film` was shown the coded column and denied the
    # definition of the very scheme named on its card.
    #
    # That rule protects against a definition whose TEXT names tables ("join premium to
    # policy_amount on ..."), which leaks their existence. A scheme definition describes the
    # scheme, not the tables, and names none of them, so it discloses nothing. Correct for a
    # public standard; a proprietary taxonomy needs a public/proprietary marker in the record
    # format, which SP2 will add when the format is revised for certified records.
    definitions = list(snapshot.definitions) + list(records.definitions)

    return snapshot.model_copy(update={"columns": columns, "definitions": definitions}, deep=True)
