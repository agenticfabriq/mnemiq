from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from mnemiq.catalog import is_key_like, is_sensitive_name
from mnemiq.contract import CodeScheme, Snapshot
from mnemiq.ontology.records import ConceptScheme, OntologyRecords
from mnemiq.semantic.textmatch import name_affinity

logger = logging.getLogger(__name__)

CONTAINMENT_MIN = 0.95   # tolerates residual dirt after sentinel forgiveness
MIN_DISTINCT = 4         # three or fewer values match a scheme by coincidence too easily
SAMPLE_LIMIT = 1000      # bounds the large-code-system probe
AFFINITY_MIN = 0.50      # overlap-coefficient floor; see textmatch.name_affinity
SUGGEST_CONTAINMENT_MIN = 0.80   # near-miss containment floor for a suggestion (below strict 0.95)
SUGGEST_AFFINITY_MIN = 0.40      # near-miss affinity floor for a suggestion (below strict 0.50)

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


def _score(column, values: list[str], scheme: ConceptScheme, notations: set[str],
           sentinels: frozenset[str]) -> BindEvidence | None:
    """Containment (sentinel-forgiving) + name affinity, computed WITHOUT applying the gate.
    Returns evidence whenever there is any non-sentinel signal; None only when nothing counts."""
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
    if considered == 0:
        return None
    return BindEvidence(
        column_id=column.id, scheme_id=scheme.id, method="scored",
        containment=matched / considered, matched=matched, sample_size=len(values),
        affinity=name_affinity(column.name, scheme.label), unmatched=unmatched[:10],
    )


def _gate(column, values: list[str], scheme: ConceptScheme, notations: set[str],
          sentinels: frozenset[str]) -> BindEvidence | None:
    """Strict auto-bind gate: containment AND affinity AND minimum distinct. Either failing binds
    nothing. (Thin wrapper over _score so suggestions can reuse the same math.)"""
    ev = _score(column, values, scheme, notations, sentinels)
    if ev is None or ev.matched < MIN_DISTINCT:
        return None  # the minimum counts NON-SENTINEL values: padding must not carry a column
    if ev.containment < CONTAINMENT_MIN:
        return None
    if ev.affinity < AFFINITY_MIN:
        return None  # containment proves the values fit; affinity makes the fit mean something
    ev.method = "auto"
    return ev


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


@dataclass
class BindingSuggestion:
    """A concept-scheme binding the precision-first binder declined to auto-make, surfaced for
    operator review. Promoting it into records.bindings makes it bind on the next run."""
    column_id: str
    scheme_id: str
    scheme_label: str
    containment: float
    matched: int
    sample_size: int
    affinity: float
    unmatched: list[str] = field(default_factory=list)
    reason: str = ""  # near_miss_containment | near_miss_affinity | ambiguous


def _suggestion(ev: BindEvidence, scheme: ConceptScheme, reason: str) -> BindingSuggestion:
    return BindingSuggestion(
        column_id=ev.column_id, scheme_id=ev.scheme_id, scheme_label=scheme.label,
        containment=ev.containment, matched=ev.matched, sample_size=ev.sample_size,
        affinity=ev.affinity, unmatched=list(ev.unmatched), reason=reason,
    )


def suggest_bindings(adapter, snapshot: Snapshot, records: OntologyRecords,
                     sentinels: frozenset[str] = DEFAULT_SENTINELS) -> list[BindingSuggestion]:
    """Reviewable binding candidates for unbound columns: near-misses (relaxed thresholds) and the
    ambiguous >1-strict-match case the auto-binder deliberately declines. Never mutates the snapshot.
    Run AFTER bind_schemes -- already-bound columns carry a code_scheme and are skipped."""
    by_scheme = {s.id: s for s in records.schemes}
    notations = {
        s.id: {c.notation.strip().casefold() for c in s.concepts} for s in records.schemes
    }
    out: list[BindingSuggestion] = []
    for column in snapshot.columns:
        if column.code_scheme is not None or is_key_like(column.name) or is_sensitive_name(column.name):
            continue
        values = _observed(adapter, snapshot, column)
        if not values:
            continue

        scored = [
            ev for s in records.schemes
            if (ev := _score(column, values, s, notations[s.id], sentinels)) is not None
        ]
        strict = [
            ev for ev in scored
            if ev.matched >= MIN_DISTINCT and ev.containment >= CONTAINMENT_MIN
            and ev.affinity >= AFFINITY_MIN
        ]
        if len(strict) > 1:  # auto-binder declined on ambiguity; surface all strict matches
            out.extend(_suggestion(ev, by_scheme[ev.scheme_id], "ambiguous") for ev in strict)
            continue
        if len(strict) == 1:
            continue  # bind_schemes already auto-bound this uniquely
        for ev in scored:  # no strict pass -> near-miss suggestions
            if ev.matched < MIN_DISTINCT:
                continue
            if ev.containment >= SUGGEST_CONTAINMENT_MIN:
                out.append(_suggestion(ev, by_scheme[ev.scheme_id], "near_miss_containment"))
            elif ev.affinity >= SUGGEST_AFFINITY_MIN:
                out.append(_suggestion(ev, by_scheme[ev.scheme_id], "near_miss_affinity"))

    out.sort(key=lambda s: (-s.containment, -s.affinity, s.column_id, s.scheme_id))
    return out


def binding_suggestions_document(source_id: str, suggestions: list[BindingSuggestion]) -> dict:
    """Serialize suggestions to the reviewable artifact shape."""
    from dataclasses import asdict
    return {
        "source_id": source_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "suggestions": [asdict(s) for s in suggestions],
    }
