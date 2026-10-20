"""M144: certified metrics, worked examples and code concepts reach the prompt by COLUMN, not table.

Each was selected by table grant alone. The card for `orders` withheld `secret_code` (phi) from an
identity without phi clearance, and then the prompt named it anyway: a metric's `count(secret_code)`,
an example's `SELECT secret_code`, the column's code vocabulary. The decider would still refuse
SQL reading it, so the leak was metadata -- the column's existence, its use and its codes -- to a
model and a caller the card had already decided should not see them.

The rule is the one a certified dimension already meets (M136): offered only where the identity
reads every column it reads raw.
"""

from __future__ import annotations

import duckdb
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import CodeScheme, Column, Example, IdentityContext, Metric, Snapshot
from mnemiq.contract.values import MeasureExpr
from mnemiq.llm.embeddings import FakeEmbedder
from mnemiq.semantic.measures import select_metrics
from mnemiq.semantic.ontology_index import OntologyIndex
from mnemiq.semantic.retrieval import _resolve_concepts, _retrieve_examples, retrieve
from mnemiq.semantic.store import build_example_index, build_index
from mnemiq.store.bootstrap import init_store

_SCHEME = CodeScheme(id="urn:icd10", label="ICD-10-CM")
_COLUMNS = [
    Column(id="orders.id", object_id="orders", name="id", data_type="int"),
    Column(id="orders.secret_code", object_id="orders", name="secret_code", data_type="text",
           pii_level="phi", code_scheme=_SCHEME),
]
_UNCLEARED = GrantSet(frozenset({"orders"}))
_CLEARED = GrantSet(frozenset({"orders"}), pii_clearance=frozenset({"phi"}))


def _metric(expr: str) -> Metric:
    return Metric(id=f"m:{expr}", label=expr, status="certified", owner="o", grain="order",
                  time_dimension="orders.id", measure=MeasureExpr(expr=expr, source="orders"))


_SECRET_METRIC, _PLAIN_METRIC = _metric("count(secret_code)"), _metric("count(id)")
_EXAMPLES = [
    Example(question="how many orders", sql="SELECT count(id) FROM orders",
            tables=["orders"], object_id="orders"),
    Example(question="how many orders by code", sql="SELECT o.secret_code, count(*) FROM orders o "
            "GROUP BY o.secret_code", tables=["orders"], object_id="orders"),
]


def _snapshot() -> Snapshot:
    return Snapshot(version="v1", source_id="s", created_at="t", columns=_COLUMNS,
                    examples=_EXAMPLES)


def test_a_metric_reading_a_column_the_identity_cannot_read_raw_is_not_offered():
    offered = select_metrics(["orders"], [_SECRET_METRIC, _PLAIN_METRIC], _UNCLEARED,
                             columns=_COLUMNS)
    assert offered == [_PLAIN_METRIC]
    assert select_metrics(["orders"], [_SECRET_METRIC, _PLAIN_METRIC], _CLEARED,
                          columns=_COLUMNS) == [_SECRET_METRIC, _PLAIN_METRIC]


def _examples_store(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    embedder = FakeEmbedder()
    build_example_index(con, _snapshot(), embedder)
    (qvec,) = embedder.embed(["how many orders"])
    return con, qvec


def test_an_example_reading_a_withheld_column_is_not_shown_through_its_alias(tmp_path):
    """The column is read through a table alias, which the decider's own resolution follows."""
    con, qvec = _examples_store(tmp_path)
    shown = _retrieve_examples(con, qvec, {"orders"}, k=5,
                               not_raw=frozenset({("orders", "secret_code")}))
    assert [e.sql for e in shown] == ["SELECT count(id) FROM orders"]
    assert len(_retrieve_examples(con, qvec, {"orders"}, k=5, not_raw=frozenset())) == 2


def _concept_index() -> OntologyIndex:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE ontology_concept (source_id TEXT, scheme_id TEXT, notation TEXT, "
                "label TEXT, definition TEXT)")
    con.execute("INSERT INTO ontology_concept VALUES "
                "('s','urn:icd10','E11','Type 2 diabetes mellitus','')")
    return OntologyIndex(con)


def test_a_code_vocabulary_on_a_withheld_column_is_not_resolved():
    question = "how many with type 2 diabetes"
    assert _resolve_concepts(question, _COLUMNS, {"orders"}, _concept_index(), _UNCLEARED) == []
    assert _resolve_concepts(question, _COLUMNS, {"orders"}, _concept_index(), _CLEARED)


class _Authz:
    def __init__(self, grants: GrantSet):
        self._grants = grants

    def grants_for(self, identity):
        return self._grants


def test_retrieve_scopes_all_three_to_the_identity(tmp_path):
    """The join: `retrieve` is what a deployment calls, and each selector must be handed what
    it needs to apply the rule -- a selector that is right in isolation and never given the
    identity's columns leaves the prompt as it was."""
    con = init_store(str(tmp_path / "s.duckdb"))
    embedder = FakeEmbedder()
    build_index(con, _snapshot(), embedder)
    build_example_index(con, _snapshot(), embedder)
    con.execute("CREATE TABLE ontology_concept (source_id TEXT, scheme_id TEXT, notation TEXT, "
                "label TEXT, definition TEXT)")
    con.execute("INSERT INTO ontology_concept VALUES "
                "('s','urn:icd10','E11','Type 2 diabetes mellitus','')")

    def ask(grants: GrantSet):
        return retrieve(con, "how many orders with type 2 diabetes", IdentityContext(
                            tenant_id="t", principal_id="p", roles=["r"]),
                        _Authz(grants), embedder, metrics=[_SECRET_METRIC, _PLAIN_METRIC],
                        columns=_COLUMNS, ontology_index=OntologyIndex(con), snapshot=_snapshot())

    uncleared, cleared = ask(_UNCLEARED), ask(_CLEARED)
    assert [m.id for m in uncleared.metrics] == [_PLAIN_METRIC.id]
    assert all("secret_code" not in e.sql for e in uncleared.examples)
    assert uncleared.concepts == []
    # The control: a cleared identity is shown all three, so the above is the rule, not an outage.
    assert len(cleared.metrics) == 2
    assert any("secret_code" in e.sql for e in cleared.examples)
    assert cleared.concepts


def test_an_example_whose_columns_cannot_be_read_is_not_shown_and_says_so(caplog):
    """Fail closed: SQL neither dialect parses cannot be checked, so it is not shown -- but only
    when something is withheld; with nothing to withhold there is nothing to check. The log line
    is how an operator learns why an example is missing from some identities' prompts."""
    import logging

    from mnemiq.semantic.retrieval import _reads_only_raw

    broken = "SELEC secret_code FRM orders"
    with caplog.at_level(logging.INFO, logger="mnemiq.semantic.retrieval"):
        assert _reads_only_raw(broken, frozenset({("orders", "secret_code")})) is False
    assert any("withheld" in r.getMessage() and r.levelno >= logging.INFO
               for r in caplog.records), [r.getMessage() for r in caplog.records]
    assert _reads_only_raw(broken, frozenset()) is True


# -- Codex's review of #121: three ways past the selectors, each closed ------------------------

_UNCHECKABLE = [
    "(SELECT count(p.secret_code) FROM patients p)",  # a subquery: reads a relation of its own
    "count(patients.secret_code)",                     # a column another table qualifies
    "count(secret_code) FILTER (WHERE",                # parses in no dialect, names no column
    "count(patients.secret_code) FILTER (WHERE",       # parses nowhere, its spelling not ours
    "(SELECT count(p.secret_code) FROM patients p",    # the same, unbalanced
]


@pytest.mark.parametrize("expr", _UNCHECKABLE)
def test_an_expression_the_check_cannot_read_is_withheld_from_everyone(expr):
    """`columns_read` reports the columns of the metric's own source, so these read columns it
    never names, and a check built on it passed them. Withheld even from the cleared: what they
    read cannot be established for anyone."""
    for grants in (_UNCLEARED, _CLEARED):
        assert select_metrics(["orders"], [_metric(expr)], grants, columns=_COLUMNS) == []


# -- a subquery over the source alone is the source's own read (the ui lane, 2026-10-20) ---------

_LATEST = "count(id) FILTER (WHERE orders.id = (SELECT max(r.id) FROM orders r WHERE r.id = orders.id))"
_OWN_SUBQUERY = [
    (_LATEST, False),                                                   # the latest-version filter: reads id alone
    ("count(id) FILTER (WHERE id = (SELECT max(id) FROM orders))", False),
    ("(SELECT max(r.secret_code) FROM orders r)", True),                 # aliased: reads the withheld column
    ("(SELECT max(orders.secret_code) FROM orders)", True),
    ("(SELECT max(secret_code) FROM orders)", True),
    ("(SELECT max(x) FROM (SELECT secret_code AS x FROM orders) q)", True),  # through a derived table
    ("(SELECT max(q.x) FROM (SELECT secret_code AS x FROM orders) q)", True),
    ("(SELECT count(*) FROM orders)", False),                            # COUNT's star reads no column
    ("(SELECT count(*) FROM orders o JOIN orders r ON o.secret_code = r.secret_code)", True),  # keys named
]


@pytest.mark.parametrize("expr, reads_the_secret", _OWN_SUBQUERY)
def test_a_subquery_over_the_source_alone_is_checked_as_the_sources_own_read(expr, reads_the_secret):
    """Read as foreign for any subquery, a certified metric with the latest-version filter was
    withheld from everyone, and four of a pilot bundle's five metrics never reached the model
    (the ui lane's measurement). Every column such a subquery reads is a column of the source,
    aliased, bare or through a derived table, so the raw-read check covers each as before."""
    from mnemiq.semantic.expr_columns import columns_read, expression_scope
    assert expression_scope(expr, "orders") == "own"
    read = {name for ref in columns_read(expr, "orders") for name in ref}
    assert ("secret_code" in read) is reads_the_secret, read
    uncleared = select_metrics(["orders"], [_metric(expr)], _UNCLEARED, columns=_COLUMNS)
    assert uncleared == ([] if reads_the_secret else [_metric(expr)]), expr
    assert select_metrics(["orders"], [_metric(expr)], _CLEARED, columns=_COLUMNS) == [_metric(expr)]


@pytest.mark.parametrize("expr", [
    "(SELECT count(p.secret_code) FROM patients p)",
    "(SELECT count(secret_code) FROM patients)",   # no qualifier: the table alone says it is foreign
    "(SELECT max(id) FROM patients)",
    "(SELECT max(id) FROM orders o JOIN patients p ON p.id = o.id)",
    "(SELECT max(q.secret_code) FROM (SELECT * FROM orders) q)",  # a star names no column (the gate)
    "(SELECT max(to_json(r)::text) FROM orders r)",               # the whole row as a value (the gate)
    "(SELECT max(to_json(orders)::text) FROM orders)",
    "(SELECT max(q.x) FROM (SELECT COLUMNS('s.*') AS x FROM orders) q)",
    "(SELECT max(q.c) FROM orders q(a, c))",                       # renamed by position (the gate)
    "(SELECT count(*) FROM orders o JOIN orders r USING (secret_code))",  # a key no node names (Codex)
    "(SELECT count(*) FROM orders o NATURAL JOIN orders r)",
    "(SELECT max(id) FROM private.orders)",                        # another schema's table (Codex)
    "(SELECT max(q.c) FROM (SELECT id, secret_code FROM orders) q(a, c))",
    "(WITH w AS (SELECT id FROM orders) SELECT max(id) FROM w)",
    "(SELECT max(id) FROM orders, LATERAL (SELECT 1 AS one) l)",
    "(SELECT max(v.c) FROM (VALUES (1)) v(c))",
    "(SELECT max(id) FROM read_csv('x'))",
])
def test_a_subquery_reading_any_other_relation_stays_foreign(expr):
    from mnemiq.semantic.expr_columns import expression_scope
    assert expression_scope(expr, "orders") == "foreign"
    for grants in (_UNCLEARED, _CLEARED):
        assert select_metrics(["orders"], [_metric(expr)], grants, columns=_COLUMNS) == []


def test_a_keyword_named_column_still_reads_through_its_spelling():
    """The control for the rule above: an expression no dialect parses is withheld only when its
    spelling names no column. `orders.select` names one, as M136's gate measured."""
    columns = [*_COLUMNS, Column(id="orders.select", object_id="orders", name="select")]
    kept = select_metrics(["orders"], [_metric("orders.select"), _metric("count(*)")],
                          _UNCLEARED, columns=columns)
    assert [m.measure.expr for m in kept] == ["orders.select", "count(*)"]


@pytest.mark.parametrize("sql", ["SELECT * FROM orders", "SELECT o.* FROM orders o",
                                 "SELECT COLUMNS('secret_code') FROM orders",
                                 "SELECT * EXCLUDE (id) FROM orders",
                                 # DuckDB returns the whole row for a relation used as a value.
                                 "SELECT o FROM orders o", "SELECT to_json(o) FROM orders o"])
def test_an_example_reading_through_a_star_is_withheld_where_a_column_is(sql):
    """A star or COLUMNS(...) reads columns no node names, so `check_cls` cannot see them."""
    from mnemiq.semantic.retrieval import _reads_only_raw

    assert _reads_only_raw(sql, frozenset({("orders", "secret_code")})) is False
    assert _reads_only_raw(sql, frozenset()) is True
    # and only where it touches a table that has a withheld column
    assert _reads_only_raw(sql.replace("orders", "customers"),
                           frozenset({("orders", "secret_code")})) is True


def _facts():
    from mnemiq.contract import TableFacts

    return TableFacts(object_id="orders", canonical_measures={
        "coded orders": "count(secret_code)", "orders": "count(id)"},
        default_time_column="secret_code")


def test_the_facts_block_withholds_what_the_metric_selector_withholds():
    from mnemiq.semantic.retrieval import RetrievedCard, _attach_facts

    def card_for(grants):
        card = RetrievedCard(object_id="orders", card="TABLE orders\n  id int", score=1.0)
        _attach_facts([card], [_facts()], grants=grants, columns=_COLUMNS, relationships=[])
        return card.card

    uncleared, cleared = card_for(_UNCLEARED), card_for(_CLEARED)
    assert "secret_code" not in uncleared and "count(id)" in uncleared
    assert "count(secret_code)" in cleared  # the control


def test_retrieve_scopes_the_facts_block_too(tmp_path):
    """The join for the facts block: `retrieve` must hand it the identity's columns."""
    con = init_store(str(tmp_path / "s.duckdb"))
    embedder = FakeEmbedder()
    build_index(con, _snapshot(), embedder)
    packet = retrieve(con, "how many orders", IdentityContext(tenant_id="t", principal_id="p",
                                                               roles=["r"]),
                      _Authz(_UNCLEARED), embedder, columns=_COLUMNS, table_facts=[_facts()],
                      snapshot=_snapshot())
    text = "\n".join(c.card for c in packet.cards)
    assert "count(id)" in text, "the facts block has to arrive, or this proves nothing"
    assert "secret_code" not in text


def test_counting_rows_reads_no_column_and_is_shown():
    """The review gate's blocker on the first version: `count(*)` holds a star and was withheld,
    taking every row-count example from any identity with a withheld column."""
    from mnemiq.semantic.retrieval import _reads_only_raw

    assert _reads_only_raw("SELECT count(*) FROM orders", frozenset({("orders", "secret_code")}))


def test_a_dimension_reading_another_table_is_withheld_from_everyone():
    """Dimensions share the expression check, so they share its stricter rule: one qualified by
    another table cannot be checked against this source's columns."""
    from mnemiq.contract import Dimension
    from mnemiq.semantic.measures import select_dimensions

    def dim(expr):
        return Dimension(id=f"d:{expr}", label=expr, source="orders", expr=expr)

    for grants in (_UNCLEARED, _CLEARED):
        kept = select_dimensions(["orders"], [dim("customer.region"), dim("id")], grants,
                                 columns=_COLUMNS)
        assert [d.expr for d in kept] == ["id"]


def test_a_qualified_source_is_its_own_relation_exactly():
    # Codex's review of #172: the column qualifier's suffix match is no relation test. The
    # source's own dotted name, or a bare name equal to its last part, is the source; another
    # schema's same-named table is not, and a qualified source read bare still is.
    from mnemiq.semantic.expr_columns import expression_scope
    assert expression_scope("(SELECT max(id) FROM pg.orders)", "pg.orders") == "own"
    assert expression_scope("(SELECT max(id) FROM orders)", "pg.orders") == "own"
    assert expression_scope("(SELECT max(id) FROM other.orders)", "pg.orders") == "foreign"
    assert expression_scope("(SELECT max(id) FROM private.orders)", "orders") == "foreign"
    assert expression_scope("(SELECT max(r.id) FROM pg.orders r)", "pg.orders") == "own"
    # One direction only (the review gate): a bare source claims the unqualified table, so a
    # qualified relation over it may be another schema's and is withheld; a bundle qualifying
    # its tables registers its sources qualified.
    assert expression_scope("(SELECT max(id) FROM pg.orders)", "orders") == "foreign"
    # A quoted name keeps its case where quoting preserves it, and must match as written.
    assert expression_scope('(SELECT max(id) FROM "ORDERS")', "orders") == "foreign"
    assert expression_scope('(SELECT max(id) FROM "orders")', "orders") == "own"
    assert expression_scope("(SELECT max(id) FROM ORDERS)", "orders") == "own"
