from __future__ import annotations

from mnemiq.catalog import KEY_SUFFIXES, TableInfo, is_key_like
from mnemiq.contract import JoinKey, Relationship
from mnemiq.enrichment.profiling import ColumnStats, profile_column


def _stem(column: str) -> str:
    """`claim_identifier` -> `claim`: the entity an identifier column names."""
    for suffix in KEY_SUFFIXES:
        if column.endswith(suffix):
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

    def stats(table: str, col: str) -> ColumnStats:
        if (table, col) not in stats_cache:
            # k=0: key detection needs counts only, never the value list
            stats_cache[(table, col)] = profile_column(adapter, table, col, k=0)
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
