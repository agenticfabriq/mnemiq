from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import duckdb

from mnemiq.authz.grants import AuthzProvider, GrantSet
from mnemiq.contract import (
    Column, Definition, Dimension, Example, HistoryTurn, IdentityContext, Metric,
    TableFacts,
)
from mnemiq.llm.embeddings import Embedder
from mnemiq.semantic.cards import render_facts_block
from mnemiq.semantic.embedding_width import refuse_if_mismatched
from mnemiq.semantic.glossary import select_definitions
from mnemiq.semantic.measures import (
    expression_reads_raw,
    read_raw,
    select_dimensions,
    select_metrics,
)
from mnemiq.sql.identifiers import ascii_lower

logger = logging.getLogger(__name__)

_RRF_K = 60  # the standard RRF constant (reciprocal-rank blending)


@dataclass
class RetrievedCard:
    object_id: str
    card: str
    score: float


@dataclass(frozen=True)
class ResolvedConcept:
    """A candidate code for the asked question, on a scheme-bound column."""
    column_id: str
    scheme_label: str
    notation: str
    label: str


@dataclass
class ContextPacket:
    question: str
    cards: list[RetrievedCard]
    grant_fingerprint: str
    enrichment_version: str | None
    definitions: list[Definition] = field(default_factory=list)
    # Certified metrics and dimensions over the tables in `cards`. They rode in the snapshot and
    # were read by nothing until this; see `semantic.measures` for why they are selected by table
    # rather than by the words of the question.
    metrics: list[Metric] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)
    concepts: list[ResolvedConcept] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)  # Plan 08 seam; filled by retrieve()
    # Prior turns, already scoped to this identity's authorization boundary by the caller.
    history: list[HistoryTurn] = field(default_factory=list)
    # (sent, described) when `mnemiq.semantic.fit` withheld column descriptions to fit the window
    # (M127); None when every card is as retrieval rendered it.
    descriptions: tuple[int, int] | None = None


def _rank(rows: list[tuple[str, float]]) -> dict[str, int]:
    return {object_id: rank for rank, (object_id, _score) in enumerate(rows, start=1)}


def _attach_facts(cards: list[RetrievedCard], table_facts: Sequence[TableFacts],
                  card_style: str = "cards", *, grants, columns: Sequence[Column],
                  relationships) -> None:
    """Attach each retrieved table's structural-facts block. Facts live here, NOT in the
    indexed card, so they never perturb retrieval top-k.

    Where the block goes depends on the card form. Splicing after the first line is right
    for `TABLE x` and WRONG for `CREATE TABLE x (` -- it drops `GRAIN:`/`GOTCHAS:` between
    the header and the first column and hands the generator malformed DDL, which is the one
    thing the ddl form exists to avoid. In ddl style the block follows the closing paren,
    commented, so it is legal SQL either way.

    `relationships` is required, and None means unknown: a fact with no dependencies recorded
    is shown only to an identity holding every related table, which cannot be established
    without them (M143). A default of none would show such facts wherever a caller forgot.
    """
    by_id = {tf.object_id: tf for tf in table_facts}
    for c in cards:
        tf = by_id.get(c.object_id)
        block = (render_facts_block(_scoped_facts(tf, grants, columns, relationships))
                 if tf else "")
        if not block:
            continue
        if card_style == "ddl":
            commented = "\n".join(f"-- {line}" if line else "--" for line in block.split("\n"))
            c.card = f"{c.card}\n{commented}"
            continue
        head, _, rest = c.card.partition("\n")
        c.card = f"{head}\n{block}\n{rest}" if rest else f"{head}\n{block}"


def known_relationships(snapshot):
    """The snapshot's relationships, or None where they are not known. A merged federated
    snapshot carries none -- `merge_snapshots` does not merge them -- and its empty list would
    read as "no related tables", showing a fact with no dependencies recorded to an identity
    that does not hold one (review gate, M143)."""
    if snapshot is None or getattr(snapshot, "registry", None):
        return None
    return snapshot.relationships


def _reads_unnamed_columns(ast) -> bool:
    """Does the query read columns it never names: a star in a projection (`*`, `o.*`, with or
    without EXCLUDE), DuckDB's `COLUMNS(...)`, or a table or alias used as a value -- DuckDB
    returns the whole row as a struct for `SELECT o FROM orders o`, measured. `count(*)` names
    nothing and reads nothing, so a star whose parent is COUNT does not count."""
    from sqlglot import exp

    if ast.find(exp.Columns) is not None:
        return True
    if any(not isinstance(star.parent, exp.Count) for star in ast.find_all(exp.Star)):
        return True
    relations = {name.lower() for t in ast.find_all(exp.Table)
                 for name in (t.name, t.alias_or_name) if name}
    return any(not column.table and column.name.lower() in relations
               for column in ast.find_all(exp.Column))


def _scoped_facts(tf: TableFacts, grants, columns: Sequence[Column],
                  relationships) -> TableFacts:
    """A table's facts with what this identity may not see taken out (M143).

    A canonical measure or the default time column is kept only where `expression_reads_raw`
    keeps a metric (Codex's review of #121 found the block naming `count(secret_code)` after the
    metric selector had withheld it). The grain and each gotcha are prose, kept only where the
    identity can see every dependency enrichment recorded for that text -- a column read raw, a
    table granted, and anything the snapshot does not hold seen by no one. A fact with none
    recorded is kept only for an identity that reads every column of the table raw and holds
    every related table: the owner's rule (2026-10-14) for facts enriched before dependencies
    were recorded, or reworded since.
    """
    measures = {name: expr for name, expr in tf.canonical_measures.items()
                if expression_reads_raw(expr, tf.object_id, columns, grants)}
    time = tf.default_time_column
    if time and not expression_reads_raw(time, tf.object_id, columns, grants):
        time = None

    # Every name here is matched to decide what is GRANTED, so it is folded no wider than the
    # engine: ASCII letters only (M154). Folding every letter let a grant on `Ä` show a fact that
    # depends on an ungranted `ä`. Narrower only makes a dependency name nothing, which is withheld.
    fold = ascii_lower
    granted = {fold(o) for o in grants.objects}
    by_key: dict[tuple[str, str], list[Column]] = {}
    tables: set[str] = set()
    for column in columns:
        by_key.setdefault((fold(column.object_id), fold(column.name)), []).append(column)
        tables.add(fold(column.object_id))

    def visible(dep: str) -> bool:
        # `table.column` or `table`; a dotted table makes the two readings ambiguous, so each
        # one that names something must be visible, and one naming nothing sees no one.
        table, _, name = fold(dep).rpartition(".")
        as_column = by_key.get((table, name)) if table else None
        as_table = fold(dep) in tables
        if not as_column and not as_table:
            return False
        if as_column and not (table in granted and all(read_raw(c, grants) for c in as_column)):
            return False
        return not as_table or fold(dep) in granted

    def unrestricted() -> bool:
        if relationships is None:
            return False
        own = fold(tf.object_id)
        related = {fold(r.to if fold(r.from_) == own else r.from_)
                   for r in relationships if own in (fold(r.from_), fold(r.to))}
        return (own in granted and related <= granted
                and all(read_raw(c, grants) for c in columns if fold(c.object_id) == own))

    def shown(text: str) -> bool:
        deps = tf.reads.get(text)
        return unrestricted() if deps is None else all(visible(d) for d in deps)

    grain = tf.grain if tf.grain and shown(tf.grain) else None
    gotchas = [g for g in tf.gotchas if shown(g)]
    return tf.model_copy(update={"canonical_measures": measures, "default_time_column": time,
                                 "grain": grain, "gotchas": gotchas})


def _retrieve_examples(con, embedding, allowed: set[str], k: int = 5, *,
                       not_raw: frozenset[tuple[str, str]]) -> list[Example]:
    """Top-k validated examples by QUESTION similarity to the asked question -- on-target
    few-shot, not per-table. Never surfaces an example touching a table outside the grants, nor
    one reading a column the identity does not read raw (`not_raw`, M144): its SQL would show the
    model that column's name and use, which the card withheld."""
    placeholders = ", ".join("?" for _ in allowed)
    # Never built -- checked explicitly, by name, schema AND catalog (matching
    # DefinitionIndex.nearest's identical check; a same-named table in a different attached
    # catalog must not pass this, the mistake `declared_width` made before it was fixed to use
    # DESCRIBE), rather than inferred from catching duckdb.CatalogException around the query
    # below: a missing TABLE and a missing FUNCTION raise that same exception type, so catching
    # it there would also silently swallow a genuinely broken query. This is the common,
    # unremarkable state (examples were never built, or enrich_examples is off), so it is not
    # logged -- every one of retrieve()'s callers would otherwise warn on every call.
    table_exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'example' "
        "AND table_schema = current_schema() AND table_catalog = current_database()"
    ).fetchone()
    if table_exists is None:
        return []
    try:
        # Ties broken on every selected column but the score: `example` has no key, and DuckDB returns tied rows in no fixed
        # order, so tied examples would otherwise change order, or which ones make the k * 3 cut, per call (M149).
        rows = con.execute(
            "SELECT question, sql, tables, object_id, "
            f"array_cosine_similarity(embedding, ?::FLOAT[{len(embedding)}]) AS s "
            f"FROM example WHERE object_id IN ({placeholders}) "
            "ORDER BY s DESC, object_id, question, sql, tables LIMIT ?",
            [embedding, *allowed, k * 3],
        ).fetchall()
    except Exception as exc:
        # The table exists but the query still failed. The case this branch is FOR is an
        # embedding-width mismatch (duckdb.BinderException) -- example is sized and refused
        # independently of semantic_object (see refuse_if_mismatched), so retrieve()'s own guard
        # on semantic_object does not catch a mismatch confined to example alone -- but this also
        # catches an unavailable array function or similar, same as DefinitionIndex.nearest's
        # identical broad catch. Logged, unlike the missing-table case above: this is not the
        # ordinary "not built yet" state, and the build path would refuse the identical mismatch
        # loudly (build_example_index, federated_build). Still returns [] rather than raising --
        # examples are a supplementary few-shot enhancement, never load-bearing for an answer, so
        # losing them is the correct degrade even though it is worth a line in the log.
        logger.warning("example retrieval failed; few-shot examples skipped: %s", exc)
        return []
    out: list[Example] = []
    for question, sql, tables_json, object_id, _s in rows:
        tables = json.loads(tables_json)
        # An example must not reveal a forbidden table, nor a column this identity may not read.
        if set(tables) <= allowed and _reads_only_raw(sql, not_raw):
            out.append(Example(question=question, sql=sql, tables=tables, object_id=object_id))
        if len(out) >= k:
            break
    return out


def _reads_only_raw(sql: str, not_raw: frozenset[tuple[str, str]]) -> bool:
    """Does an example's SQL read no column in `not_raw`? Asked of `check_cls`, the decider's own
    column resolution, so an alias, a CTE or an unqualified name resolves as it would on a real
    query. Enrichment stores an example as an approved plan in the deciding dialect -- `duckdb`
    for every source `adapters/resolve.py` builds but Oracle, which is `oracle`; an adapter kind
    added there with its own dialect would land here -- while feedback capture stores the
    SQL an operator wrote, in any spelling. One neither parser reads is withheld, since what it
    reads cannot be checked, and logged, since an operator would otherwise see only examples
    missing from the prompts of identities with something withheld."""
    if not not_raw:
        return True
    import sqlglot

    from mnemiq.sql.cls import check_cls
    from mnemiq.sql.policy import AccessPolicy

    from sqlglot import exp

    from mnemiq.semantic.expr_columns import keyword_columns

    withheld_tables = {obj.lower().rsplit(".", 1)[-1] for obj, _ in not_raw}
    tokens = keyword_columns(sql)
    for dialect in ("duckdb", "oracle"):
        try:
            ast = sqlglot.parse_one(sql, read=dialect)
        except Exception:  # noqa: BLE001 -- try the other dialect; neither means not shown
            continue
        # A read of columns no node names cannot be seen by `check_cls`, which checks named ones
        # (Codex's review of #121): `SELECT COLUMNS('secret_code') FROM orders` returns the column.
        # Withheld when any table the example reads has a withheld column.
        # The tables read: the parse's names, and every table the text names anywhere, by the
        # last part of a dotted token, since the parse names some wrongly or not at all:
        # `query_table('orders')` is a Table with no name to sqlglot and `query_table('main.orders')`
        # names the table inside a dotted string, `TABLE orders` is a Table named `TABLE` and a
        # CTE over it is its own name, and DuckDB reads orders through each (the review gate on
        # the second and third commits); a name quoted inside the string (`query_table('"orders"')`)
        # keeps its quotes as a token, stripped here (the gate on the third).
        read = {t.name.lower() for t in ast.find_all(exp.Table)} | {
            token.rsplit(".", 1)[-1].strip('"`') for token in tokens}
        if "" in read:
            # A source the parse cannot name at all, a table function: DuckDB's `query_table`
            # reads the table named inside the string with quote characters dropped anywhere in
            # it (`'"ord""ers"'` is orders), so no token matches, and any withheld table may be the
            # one read (the review gate on the fourth commit).
            read |= withheld_tables
        if _reads_unnamed_columns(ast) and read & withheld_tables:
            return False
        # A column list renames by position (`FROM orders AS q(id, unit)`, a derived table's or
        # a CTE's list the same), so a withheld column reaches the text under a name no token
        # matches: `date_trunc(unit, id)` read orders.secret_code on DuckDB and passed (Codex's
        # first review of #181). Withheld wherever a list stands over a withheld table read.
        if read & withheld_tables and any(alias.columns for alias in ast.find_all(exp.TableAlias)):
            return False
        # DuckDB reads a column by its position too, `#2`, `# 2` or `#/**/2`, which no token
        # names (the review gate on the sixth, seventh and eighth commits): a `#` has no place in an
        # example over a withheld table, a positional reference there or MySQL's comment opener
        # elsewhere, so the character alone withholds, a string literal holding it included. A
        # table's name or alias as a value, the whole row as a struct (`SELECT orders FROM
        # orders`), the unnamed-columns rule withholds already.
        if read & withheld_tables and "#" in sql:
            return False
        # Any token of the example's text the engine reads as a column where a table the example
        # reads has a withheld column so named (M210, Codex's twenty-third review of #172):
        # `date_trunc(secret_code, id)` reads the column on DuckDB while sqlglot makes a string of
        # the unit, so the parse saw `id` alone and the example naming and using the withheld
        # column reached the uncleared's prompt. The selectors' rule (`keyword_columns`), held
        # here to the tables the example reads.
        if any(obj.lower().rsplit(".", 1)[-1] in read and col.lower() in tokens for obj, col in not_raw):
            return False
        return check_cls(ast, AccessPolicy(denied=set(not_raw)), dialect) is None
    logger.info("an example was withheld from an identity with columns withheld: its SQL parses as "
                "neither duckdb nor oracle, so the columns it reads cannot be checked: %.80s", sql)
    return False


_CONCEPT_CAP = 15  # total across all columns: a prompt block, not a data dump


def _resolve_concepts(question, columns, shown: set[str], index, grants) -> list[ResolvedConcept]:
    """Candidate codes for the question, for scheme-bound columns on RETRIEVED tables only, that
    the identity reads raw.

    TABLE safety is inherited: `shown` comes from cards already scoped to the identity's grants.
    COLUMN safety is not, and the docstring that said it was covered only tables (M144): a denied
    or masked column on a shown table had its code vocabulary resolved into the prompt.
    """
    out: list[ResolvedConcept] = []
    for column in columns:
        if column.object_id not in shown or column.code_scheme is None:
            continue
        if not read_raw(column, grants):
            continue
        for notation, label in index.nearest(column.code_scheme.id, question):
            out.append(ResolvedConcept(column.id, column.code_scheme.label, notation, label))
            if len(out) >= _CONCEPT_CAP:
                return out
    return out


def retrieve(
    con: duckdb.DuckDBPyConnection,
    question: str,
    identity: IdentityContext,
    authz: AuthzProvider,
    embedder: Embedder,
    k: int = 5,
    definitions: Sequence[Definition] = (),
    metrics: Sequence[Metric] = (),
    dimensions: Sequence[Dimension] = (),
    table_facts: Sequence[TableFacts] = (),
    columns: Sequence[Column] = (),
    ontology_index=None,
    snapshot=None,
    card_style: str = "cards",
    grants: GrantSet | None = None,
) -> ContextPacket:
    """Hybrid retrieval, scoped to the identity's grants *before* anything is ranked.

    An object the identity may not access is never scored, never ranked, never counted and
    never returned: the model cannot be tempted by a table it was never shown, and no packet
    can disclose that the table exists. Metadata is itself confidential -- a table named
    `layoff_plans` discloses by its mere existence.

    **That was true of tables and false of columns until M4.** The stored card is rendered once at
    enrich time from the whole snapshot, so a granted table's card carried every column, its
    pii_level and its harvested coded values. Given `snapshot`, the cards are re-rendered against
    the identity's column policy before they go in the packet. Ranking still runs on the stored,
    identity-independent card -- an index per grant set is combinatorial, and ranking is not a
    channel: nothing about a hidden column reaches the caller.

    `grants`, when given, are the identity's grants as the caller already resolved them for this
    question, and `authz` is not asked again (M13): a provider that re-reads its policy file per
    call could otherwise scope these cards against one version of it and the agent's decision
    against another.
    """
    if grants is None:
        grants = authz.grants_for(identity)
    packet = ContextPacket(
        question=question,
        cards=[],
        grant_fingerprint=grants.fingerprint,
        enrichment_version=None,
    )
    if not grants.objects:
        return packet  # nothing is visible: do not even touch the index


    allowed = list(grants.objects)
    placeholders = ", ".join("?" for _ in allowed)

    # Both rankings break ties by object_id. DuckDB returns tied rows in no fixed order, and `_rank` numbers rows by
    # position, so without it the same question could reach the generator with its cards in another order (M149).
    lexical = con.execute(
        f"""
        SELECT object_id, score FROM (
          SELECT object_id,
                 fts_main_semantic_object.match_bm25(object_id, ?) AS score
          FROM semantic_object
          WHERE object_id IN ({placeholders})
        ) WHERE score IS NOT NULL
        ORDER BY score DESC, object_id LIMIT ?
        """,
        [question, *allowed, k],
    ).fetchall()

    (embedding,) = embedder.embed([question])
    # The read-side twin of the check every destructive rebuild runs before its own statement
    # (build_index, build_example_index, definition_index, federated_build): an unguarded
    # array_cosine_similarity below raises DuckDB's own
    # duckdb.BinderException ("Array arguments must be of the same size") for the identical
    # width mismatch, with none of the write path's named remedy -- an operator who never rebuilt
    # would see only the crash, never told this store must be rebuilt for the embedder now
    # configured. Reuses refuse_if_mismatched rather than a second copy of the wording.
    refuse_if_mismatched(con, "semantic_object", len(embedding))
    semantic = con.execute(
        f"""
        SELECT object_id, array_cosine_similarity(embedding, ?::FLOAT[{len(embedding)}]) AS score
        FROM semantic_object
        WHERE object_id IN ({placeholders})
        ORDER BY score DESC, object_id LIMIT ?
        """,
        [embedding, *allowed, k],
    ).fetchall()

    lexical_rank, semantic_rank = _rank(lexical), _rank(semantic)
    fused: dict[str, float] = {}
    for object_id in {*lexical_rank, *semantic_rank}:
        score = 0.0
        if object_id in lexical_rank:
            score += 1.0 / (_RRF_K + lexical_rank[object_id])
        if object_id in semantic_rank:
            score += 1.0 / (_RRF_K + semantic_rank[object_id])
        fused[object_id] = score

    top = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    if not top:
        return packet

    rows = con.execute(
        "SELECT object_id, card, version FROM semantic_object WHERE object_id IN "
        f"({', '.join('?' for _ in top)})",
        [object_id for object_id, _ in top],
    ).fetchall()
    cards = {object_id: (card, version) for object_id, card, version in rows}

    scoped: dict[str, str] = {}
    if snapshot is not None:
        from mnemiq.semantic.cards import build_cards
        from mnemiq.sql.policy import build_access_policy

        # `card_style` reaches only THIS re-render, the identity-scoped copy handed to the
        # generator. `build_index` keeps the default, so the embedded card stays the form
        # top-k was measured on and retrieval is untouched by the generator's preference.
        scoped = {
            card.object_id: card.text
            for card in build_cards(
                snapshot, policy=build_access_policy(snapshot, grants), style=card_style)
        }
    packet.cards = [
        RetrievedCard(
            object_id=object_id,
            card=scoped.get(object_id, cards[object_id][0]),
            score=score,
        )
        for object_id, score in top
        if object_id in cards
    ]
    packet.enrichment_version = next(iter(cards.values()))[1] if cards else None
    _attach_facts(packet.cards, table_facts, card_style, grants=grants, columns=columns,
                  relationships=known_relationships(snapshot))
    # After the cards are chosen, because a certified measure rides with its table -- see
    # `semantic.measures` for why that rule differs from the glossary's word-matching one.
    table_ids = [c.object_id for c in packet.cards]
    # After the cards, not before: a bound definition rides with its table, so which tables were
    # retrieved has to be known first. It used to be selected before anything was ranked.
    packet.definitions = select_definitions(question, definitions, grants, table_ids)
    packet.metrics = select_metrics(table_ids, metrics, grants, columns=columns)
    packet.dimensions = select_dimensions(table_ids, dimensions, grants, columns=columns)
    not_raw = frozenset((c.object_id, c.name) for c in columns if not read_raw(c, grants))
    packet.examples = _retrieve_examples(con, embedding, set(grants.objects), k=k,
                                         not_raw=not_raw)
    if ontology_index is not None and packet.cards:
        packet.concepts = _resolve_concepts(
            question, columns, {c.object_id for c in packet.cards}, ontology_index, grants
        )
    return packet
