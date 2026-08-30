from __future__ import annotations

from mnemiq.catalog import KEY_SUFFIXES, TableInfo, is_key_like
from mnemiq.contract import JoinKey, Relationship
from mnemiq.enrichment.profiling import ColumnStats, profile_column


def _stem(column: str) -> str:
    """`claim_identifier` -> `claim`: the entity an identifier column names.

    The suffix is matched case-INSENSITIVELY and the original case is returned, because the stem
    is looked up against table names from the same source and must keep that source's folding.

    This compared raw case, so on Oracle -- which folds unquoted identifiers to upper --
    `_stem("DOC_ID")` returned `"DOC_ID"` unchanged, `tables.get` found nothing, and **inferred
    relationships were impossible on any Oracle source**. It fails silently: declared foreign keys
    still arrive from the catalog, so a model comes back with relationships in it and nothing looks
    wrong. Second instance of this bug in one family, after `is_key_like`.
    """
    lowered = column.lower()
    for suffix in KEY_SUFFIXES:
        if lowered.endswith(suffix):
            return column[: -len(suffix)]
    return column


def infer_relationships(adapter, catalog: list[TableInfo]) -> list[Relationship]:
    """Infer foreign keys: naming gives direction, data gives the verdict.

    The source declares no usable keys (ACME's own DDL does not survive its sample
    data), so both sides are established from the data. Uniqueness alone cannot say
    which side is the parent -- `claim_identifier` is unique in both `claim` and
    `claim_coverage` -- so the parent is the table the column *names*, and that claim
    is then checked: the parent's column must be a real key, and the child's values
    must all exist in it.
    """
    tables = {t.name: t for t in catalog}
    stats_cache: dict[tuple[str, str], ColumnStats] = {}

    def _declared_type(table: str, col: str) -> str | None:
        info = tables.get(table)
        return next((c.data_type for c in info.columns if c.name == col), None) if info else None

    def stats(table: str, col: str) -> ColumnStats:
        if (table, col) not in stats_cache:
            # k=0: key detection needs counts only, never the value list. The declared type goes
            # with it so a column the source cannot aggregate is skipped rather than raising --
            # on Oracle a CLOB named `*_id` otherwise takes EVERY relationship down with it.
            stats_cache[(table, col)] = profile_column(
                adapter, table, col, k=0, data_type=_declared_type(table, col))
        return stats_cache[(table, col)]

    def is_key_of(table: str, col: str) -> bool:
        s = stats(table, col)
        return s.row_count > 0 and s.distinct_count == s.row_count and s.null_count == 0

    rels: list[Relationship] = []
    for child in catalog:
        for col in child.columns:
            if not is_key_like(col.name):
                continue
            parent = tables.get(_stem(col.name))
            # the column must name some *other* table, and be that table's key
            if parent is None or parent.name == child.name:
                continue
            if not any(c.name == col.name for c in parent.columns):
                continue
            if not is_key_of(parent.name, col.name):
                continue  # parent side is not unique: a join here would fan out

            # inclusion dependency. Compared as text because the catalog mixes typed and
            # CSV-inferred all-text tables, so INTEGER 1 and TEXT '1' must still match.
            # A NULL child value means "no parent", not a broken reference.
            orphans = adapter.execute(
                f'SELECT count(*) FROM "{child.name}" c '
                f'WHERE c."{col.name}" IS NOT NULL AND NOT EXISTS ('
                f'  SELECT 1 FROM "{parent.name}" p '
                f'  WHERE CAST(p."{col.name}" AS VARCHAR) = CAST(c."{col.name}" AS VARCHAR))'
            )[0][0]
            if orphans == 0:
                rels.append(
                    Relationship(
                        id=f"{child.name}.{col.name}->{parent.name}",
                        from_=child.name,
                        to=parent.name,
                        cardinality="many_to_one",
                        join_keys=[JoinKey(left=col.name, right=col.name)],
                    )
                )
    return rels


def relationships_from_foreign_keys(adapter, catalog: list[TableInfo]) -> list[Relationship]:
    """Declared foreign keys as Relationships. The catalog is authoritative -- unlike the
    naming inference, this captures differently-named and non-_id keys."""
    known = {t.name for t in catalog}
    # Group by constraint, not by (child, parent): a composite FK's columns share a
    # constraint id, while independent FKs to the same parent do not -- so eye/hair/skin
    # -> colour stay three relationships, not one bogus composite.
    grouped: dict[str, tuple[str, str, list[JoinKey]]] = {}
    order: list[str] = []
    for from_table, from_col, to_table, to_col, cid in adapter.foreign_keys():
        if from_table not in known or to_table not in known:
            continue
        if cid not in grouped:
            grouped[cid] = (from_table, to_table, [])
            order.append(cid)
        grouped[cid][2].append(JoinKey(left=from_col, right=to_col))

    rels: list[Relationship] = []
    for cid in order:
        child, parent, keys = grouped[cid]
        cols = "+".join(k.left for k in keys)
        rels.append(
            Relationship(
                id=f"{child}.{cols}->{parent}",
                from_=child,
                to=parent,
                cardinality="many_to_one",  # the FK side is the "many"
                join_keys=keys,
            )
        )
    return rels


def build_relationships(adapter, catalog: list[TableInfo]) -> list[Relationship]:
    """Declared-first, inference-fallback, per child table. A table that declares any FK is
    described by the catalog; inference runs only for tables the catalog is silent on."""
    declared = relationships_from_foreign_keys(adapter, catalog)
    covered = {r.from_ for r in declared}
    inferred = [r for r in infer_relationships(adapter, catalog) if r.from_ not in covered]
    return declared + inferred
