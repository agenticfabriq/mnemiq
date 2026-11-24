from __future__ import annotations

import logging

from mnemiq.catalog import KEY_SUFFIXES, TableInfo, is_key_like
from mnemiq.contract import JoinKey, Relationship
from mnemiq.enrichment.profiling import ColumnStats, profile_column, unharvested

logger = logging.getLogger(__name__)


# The first word of a declared type, before any "(", that holds a number on the sources compared as text below:
# DuckDB's, Postgres's (`double precision`), MySQL's (`int unsigned`), SQLite's and the warehouses'.
_NUMERIC_BASES = frozenset({
    "TINYINT", "SMALLINT", "MEDIUMINT", "INT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER",
    "UBIGINT", "UHUGEINT", "INT1", "INT2", "INT4", "INT8", "SERIAL", "SMALLSERIAL", "BIGSERIAL", "DECIMAL", "NUMERIC",
    "NUMBER", "REAL", "FLOAT", "FLOAT4", "FLOAT8", "DOUBLE"})


def _numeric(data_type: str | None) -> bool:
    words = (data_type or "").split("(", 1)[0].upper().split()
    return bool(words) and words[0] in _NUMERIC_BASES


def _comparable(expr: str, dialect: str, *, numbers: bool = False) -> str:
    """`expr` as the orphan check compares it.

    Natively where both sides are numbers (`numbers`), as the join the relationship describes will compare them: as
    text one number is 1, 1.00 or 1.0 by its type, and a DuckDB copy of Oracle lands each NUMBER as BIGINT,
    DECIMAL(p, s) or DOUBLE by its declaration. A source may compare two keys in a wider type or a collation, so
    `infer_relationships` also asks whether a child value meets two parents. Elsewhere as text, because a catalog
    can mix typed and CSV-inferred all-text tables, where INTEGER 1 and TEXT '1' must still match. Oracle natively
    always: it refuses
    `CAST(x AS VARCHAR)` with ORA-00906 (a VARCHAR2 needs a length, measured on 23.26), sqlglot's transpile gives
    `CAST(x AS CLOB)`, which it cannot compare with `=`, and `TO_CHAR` of a DATE follows the session's format, which
    can drop the time and make two different keys one. Types that do not compare raise, and the candidate is skipped."""
    return expr if dialect == "oracle" or numbers else f"CAST({expr} AS VARCHAR)"


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


def infer_relationships(adapter, catalog: list[TableInfo], *, harvest_values: bool = True,
                        failures: list[str] | None = None) -> list[Relationship]:
    """Infer foreign keys: naming gives direction, data gives the verdict. Key detection reads counts only, and with
    `harvest_values` False each profile is told so too.

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

    failed: dict[tuple[str, str], Exception] = {}

    def stats(table: str, col: str) -> ColumnStats:
        if (table, col) in failed:
            # A count that failed once fails again, at the same cost: a parent named by many children is counted once,
            # and a count that reached the statement time limit is not sent again for the next child.
            raise failed[(table, col)]
        if (table, col) not in stats_cache:
            # k=0: key detection needs counts only, never the value list. The declared type goes
            # with it so a column the source cannot aggregate is skipped rather than raising --
            # on Oracle a CLOB named `*_id` otherwise takes EVERY relationship down with it.
            try:
                stats_cache[(table, col)] = profile_column(
                    adapter, table, col, k=0, data_type=_declared_type(table, col),
                    **unharvested(harvest_values))
            except Exception as exc:
                failed[(table, col)] = exc
                raise
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
            # A candidate whose check cannot run is skipped, not inferred, and said. Raising here took
            # every relationship of the source with it, the declared ones included, because
            # `enrich_structural` keeps none when `build_relationships` raises.
            try:
                if not is_key_of(parent.name, col.name):
                    continue  # parent side is not unique: a join here would fan out

                # inclusion dependency, compared as `_comparable` says. A NULL child value means "no
                # parent", not a broken reference.
                dialect = getattr(adapter, "dialect", "duckdb")
                numbers = _numeric(_declared_type(parent.name, col.name)) and _numeric(col.data_type)
                in_parent = _comparable(f'p."{col.name}"', dialect, numbers=numbers)
                in_child = _comparable(f'c."{col.name}"', dialect, numbers=numbers)
                orphans = adapter.execute(
                    f'SELECT count(*) FROM "{child.name}" c '
                    f'WHERE c."{col.name}" IS NOT NULL AND NOT EXISTS ('
                    f'  SELECT 1 FROM "{parent.name}" p '
                    f'  WHERE {in_parent} = {in_child})'
                )[0][0]
                # The parent was counted unique in its own type and collation, and the join may compare in another
                # that makes two parent keys one: a DOUBLE child against DECIMAL(38, 0) parents past 2^53, DuckDB's
                # HUGEINT against UHUGEINT, a NOCASE collation's 'a' and 'A' (Codex's second to fourth reviews). So
                # the source is asked, under the very predicate the join will use, whether a child value meets two.
                if orphans == 0:
                    in_values = _comparable("d.v", dialect, numbers=numbers)
                    fanned = adapter.execute(
                        f'SELECT count(*) FROM (SELECT d.v FROM (SELECT DISTINCT c."{col.name}" AS v '
                        f'FROM "{child.name}" c WHERE c."{col.name}" IS NOT NULL) d '
                        f'JOIN "{parent.name}" p ON {in_parent} = {in_values} GROUP BY d.v HAVING count(*) > 1) f'
                    )[0][0]
                    if fanned:
                        continue  # a child value meets two parents: a join here would fan out
            except Exception as exc:  # noqa: BLE001 -- one candidate failing costs only itself
                first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
                logger.warning("relationship %s.%s -> %s not inferred: its check failed: %s",
                               child.name, col.name, parent.name, first)
                if failures is not None:
                    failures.append(f"{child.name}.{col.name} -> {parent.name}: {first}")
                continue
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
    # -> colour stay three relationships, not one bogus composite. And by the child table with
    # it: Postgres keeps a constraint's name unique on its table, not in its schema, so two
    # tables may each declare `fk_customer`. A constraint has one target, so the pair is enough.
    grouped: dict[tuple[str, str], tuple[str, str, list[JoinKey]]] = {}
    for from_table, from_col, to_table, to_col, cid in adapter.foreign_keys():
        if from_table not in known or to_table not in known:
            continue
        grouped.setdefault((from_table, cid), (from_table, to_table, []))[2].append(
            JoinKey(left=from_col, right=to_col))

    rels: list[Relationship] = []
    for child, parent, keys in grouped.values():
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


def build_relationships(adapter, catalog: list[TableInfo], *, harvest_values: bool = True,
                        failures: list[str] | None = None,
                        estimated: frozenset[str] = frozenset()) -> list[Relationship]:
    """Declared-first, inference-fallback, per child table. A table that declares any FK is
    described by the catalog; inference runs only for tables the catalog is silent on. Each
    candidate whose check could not run is appended to `failures`, when given. A table in
    `estimated` (profiled from the source's statistics) is left out of inference on both sides,
    since checking a candidate reads both tables' rows; its declared keys are kept."""
    declared = relationships_from_foreign_keys(adapter, catalog)
    covered = {r.from_ for r in declared}
    measured = [t for t in catalog if t.name not in estimated]
    inferred = [r for r in infer_relationships(adapter, measured, **unharvested(harvest_values),
                                               failures=failures)
                if r.from_ not in covered]
    return declared + inferred
