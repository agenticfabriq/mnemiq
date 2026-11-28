"""What a request in Superset mode may use of the values enrichment harvested from a table's rows.

A snapshot enriched directly from Oracle for Superset mode (`mnemiq enrich --for-superset`) was read with the
enrichment login, past every Superset row rule, and its `superset:oracle` job says whether values were harvested
(`values_harvested`). The grants of each request withhold the values of every table Superset does not say, asked as the
person, no row rule applies to for them (`superset.grants.resolve_grants`, `GrantSet.values_withheld`), and what the
request reads follows them, by one predicate (`values_shown`):

- the snapshot it reads (`values_scoped`), with each withheld table's harvested content taken out: its columns' code
  lists and the meanings grounded for them, the code scheme its values bound a column to, its column descriptions,
  which the model wrote from its code lists and so may quote a value, its table facts, which the model wrote from those
  descriptions, and every example that reads it, whose SQL was written from them and kept because it returned rows. A
  code list grounded from a lookup table carries that table's labels, so a table holding one is shown only while every
  table that column joins is shown, and withheld where no join of it is recorded. What enrichment read from the
  database's dictionary or decided for itself stays: names, types, comments, classifications and counts.
- the value index it reads (`ShownValues`): the rows enrichment wrote for the snapshot's own source, of the tables the
  snapshot names whose values are shown. Every other column reads as one the index does not hold, so the value check
  checks nothing on it, and where the answer path asks whether a value exists it asks Superset, as the person, under
  their rules.

A table's values are shown only where the grants give the table to the person and Superset, asked in this request,
says no rule applies to it; a lookup's labels only where the same holds of each table its column joins. The runtime
hands the grants, the scoping and the index one snapshot per request (`Runtime._reloaded`), the one the grants were
decided for, and applies both in Superset mode alone: a column the index reads as one it does not hold is asked of
Superset, as the person, under their row rules.
"""

from __future__ import annotations

from mnemiq.authz.grants import GrantSet, values_shown
from mnemiq.contract import Snapshot
from mnemiq.contract.semantic import Example
from mnemiq.semantic.values import NoValueIndex, ValueIndex
from mnemiq.superset.settings import NOT_HARVESTED, SUPERSET_ORACLE_JOB, oracle_source

__all__ = ["ShownValues", "served_values", "snapshot_objects", "values_harvested", "values_scoped"]

# Where a meaning was grounded from a lookup table's label column (`enrichment.grounding.ground_from_lookup`).
_LOOKUP = "lookup"


def values_harvested(snapshot: Snapshot) -> bool:
    """Whether `snapshot` holds values enrichment harvested from rows, as its `superset:oracle` job records it: no only
    where the job says none were; a job whose record cannot be read cannot say so, and reads as yes. A snapshot without
    the job (one enriched through Superset, which harvests none) holds none."""
    job = next((job for job in snapshot.jobs if job.id == SUPERSET_ORACLE_JOB), None)
    if job is None:
        return False
    source = oracle_source(job.detail)
    return source is None or source.values != NOT_HARVESTED


def snapshot_objects(snapshot: Snapshot) -> frozenset[str]:
    """Every object id `snapshot` names: its columns', bindings', views', table facts' and relationships' tables, and
    each table an example reads."""
    names = {column.object_id for column in snapshot.columns}
    names |= {binding.object_id for binding in snapshot.source_bindings}
    names |= {view.object_id for view in snapshot.views}
    names |= {facts.object_id for facts in snapshot.table_facts}
    names |= {end for relationship in snapshot.relationships for end in (relationship.from_, relationship.to)}
    names |= {table for example in snapshot.examples for table in _reads(example)}
    return frozenset(name for name in names if name)


def _tables(snapshot: Snapshot) -> frozenset[str]:
    """The object ids of `snapshot`'s table bindings: the only objects whose harvested values a request may be shown,
    since the grants ask Superset of no other (`superset.grants._values_withheld`)."""
    return frozenset(binding.object_id for binding in snapshot.source_bindings if binding.binding_type == "table")


def _reads(example: Example) -> set[str]:
    """The tables an example reads, as `compose_starters` reads them: its own, and those it names."""
    return {*example.tables, *([example.object_id] if example.object_id else [])}


def _lookup_sources(snapshot: Snapshot) -> dict[str, set[str | None]]:
    """For each table holding a code list grounded from a lookup table, the tables those columns join, as the snapshot's
    relationships record them (`ground_from_lookup` reads a column's labels across each join of it); None where a
    column has no join recorded, which cannot say which table its labels came from."""
    joins: dict[tuple[str, str], set[str]] = {}
    for relationship in snapshot.relationships:
        for key in relationship.join_keys:
            joins.setdefault((relationship.from_, key.left), set()).add(relationship.to)
    sources: dict[str, set[str | None]] = {}
    for column in snapshot.columns:
        if any(value.source == _LOOKUP for value in column.coded_values):
            joined = joins.get((column.object_id, column.name), set())
            sources.setdefault(column.object_id, set()).update(joined or {None})
    return sources


def _shown_tables(snapshot: Snapshot, grants: GrantSet) -> frozenset[str]:
    """The tables of `snapshot` whose harvested content a request under `grants` may read: each whose values are shown
    (`values_shown`: granted, and not withheld), and whose lookup-grounded code lists join only tables whose values are
    shown too, and are joined. No other object is ever one: a name the snapshot binds to no table cannot be told to be
    a shown one."""
    lookups = _lookup_sources(snapshot)
    return frozenset(name for name in _tables(snapshot)
                     if values_shown(grants, name)
                     and all(source is not None and values_shown(grants, source) for source in lookups.get(name, ())))


def values_scoped(snapshot: Snapshot, grants: GrantSet) -> Snapshot:
    """`snapshot` as a request under `grants` reads it: each object whose harvested content it may not read (every one
    but `_shown_tables`) with its columns' code lists, code schemes and descriptions taken out, its table facts dropped,
    and every example reading it dropped. Everything else is the snapshot's own. A snapshot that harvested no values,
    and one with nothing to take out, is returned itself; it is never changed."""
    if not values_harvested(snapshot):
        return snapshot
    shown = _shown_tables(snapshot, grants)
    columns = [column if column.object_id in shown or not _harvested_fields(column)
               else column.model_copy(update={"coded_values": [], "code_scheme": None, "description": None})
               for column in snapshot.columns]
    facts = [facts for facts in snapshot.table_facts if facts.object_id in shown]
    examples = [example for example in snapshot.examples if (read := _reads(example)) and read <= shown]
    if (all(new is old for new, old in zip(columns, snapshot.columns)) and len(facts) == len(snapshot.table_facts)
            and len(examples) == len(snapshot.examples)):
        return snapshot
    return snapshot.model_copy(update={"columns": columns, "table_facts": facts, "examples": examples})


def _harvested_fields(column) -> bool:
    """Whether `column` carries anything `values_scoped` takes out of a withheld table."""
    return bool(column.coded_values or column.code_scheme is not None or column.description)


class ShownValues:
    """The value index as a request in Superset mode reads it: `index`'s rows of the objects in `objects` (the
    snapshot's tables), each only while `values_shown` says a request under `grants` may use its values. Every other
    column reads as one the index does not hold, every way it is read."""

    def __init__(self, index, grants: GrantSet, objects: frozenset[str]) -> None:
        self._index = index
        self._grants = grants
        self._objects = frozenset(objects)

    def _shown(self, object_id: str) -> bool:
        return object_id in self._objects and values_shown(self._grants, object_id)

    def has(self, object_id: str, column: str) -> bool:
        return self._shown(object_id) and self._index.has(object_id, column)

    def contains(self, object_id: str, column: str, literal: str) -> bool:
        return self._shown(object_id) and self._index.contains(object_id, column, literal)

    def nearest(self, object_id: str, column: str, literal: str, k: int = 8) -> list[str]:
        return self._index.nearest(object_id, column, literal, k) if self._shown(object_id) else []

    def values(self, object_id: str, column: str) -> list[str]:
        return self._index.values(object_id, column) if self._shown(object_id) else []

    def holders(self, literal: str) -> set[tuple[str, str]]:
        return {(table, column) for table, column in self._index.holders(literal) if self._shown(table)}


def served_values(index: ValueIndex | None, snapshot: Snapshot | None, grants: GrantSet) -> ShownValues | NoValueIndex:
    """The value index a request in Superset mode reads: `index`, the store's index held to one source's rows, as
    `grants` let the request read it (`ShownValues`), for a snapshot of that source that harvested values; for any
    other, none, whatever the store holds (`NoValueIndex`)."""
    if index is None or snapshot is None or index.source_id != snapshot.source_id or not values_harvested(snapshot):
        return NoValueIndex()
    return ShownValues(index, grants, _tables(snapshot))
