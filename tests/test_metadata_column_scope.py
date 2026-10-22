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
    ("(SELECT max(`r.secret_code`) FROM orders r)", True),              # one quoted identifier to one reader (Codex)
    ("(SELECT max(`r.secret_code.value`) FROM orders r)", True),        # a nested field path (Codex's sixth)
    ("(SELECT max(`orders.secret_code.value`) FROM orders)", True),
    ("(SELECT max(`secret_code.value`) FROM orders)", True),            # a struct column's field (Codex's seventh)
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
    "(SELECT count(* EXCLUDE (id)) FROM orders)",                   # a modified star reads the rest (Codex's ninth)
    "(SELECT count(* REPLACE (id AS secret_code)) FROM orders)",
    "(SELECT count(*) FROM orders o JOIN orders r USING (secret_code))",  # a key no node names (Codex)
    "(SELECT count(*) FROM orders o NATURAL JOIN orders r)",
    "(SELECT max(id) FROM private.orders)",                        # another schema's table (Codex)
    "(SELECT max(q.c) FROM (SELECT id, secret_code FROM orders) q(a, c))",
    "(WITH w AS (SELECT id FROM orders) SELECT max(id) FROM w)",
    "(SELECT max(id) FROM orders, LATERAL (SELECT 1 AS one) l)",
    "(SELECT max(v.c) FROM (VALUES (1)) v(c))",
    "(SELECT max(id) FROM read_csv('x'))",
    "(SELECT count(*) FROM orders PIVOT (sum(id) FOR status IN ('x')))",  # grouping columns no node names (Codex's tenth)
    "(SELECT max(to_json(p)) FROM orders PIVOT (sum(id) FOR status IN ('x')) p)",  # the whole row through the pivot's alias
    "(SELECT count(*) FROM (PIVOT orders ON status USING sum(id)))",     # DuckDB's own spelling
    "(SELECT max(v) FROM orders UNPIVOT (v FOR k IN (id, status)))",
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


def test_an_example_reading_a_withheld_column_through_a_unit_argument_is_not_shown(tmp_path):
    # M210 (Codex's twenty-third review of #172): sqlglot makes a string of `date_trunc`'s unit
    # argument, so the parse-based check saw `id` alone where DuckDB reads `secret_code` (the
    # values `year` and `day` give different dates), and the example naming and using the
    # withheld column reached the uncleared's prompt. Held to the raw text as the selectors are.
    from mnemiq.semantic.retrieval import _reads_only_raw
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders(id DATE, secret_code TEXT)")
    con.execute("INSERT INTO orders VALUES ('2026-03-15', 'year'), ('2026-03-15', 'day')")
    dates = con.execute("SELECT date_trunc(secret_code, id) FROM orders ORDER BY secret_code").fetchall()
    assert dates[0] != dates[1]
    sql = "SELECT date_trunc(secret_code, id) AS period, count(*) FROM orders GROUP BY 1"
    assert _reads_only_raw(sql, frozenset({("orders", "secret_code")})) is False
    assert _reads_only_raw(sql, frozenset({("other", "secret_code")})) is True   # a table the example does not read
    assert _reads_only_raw("SELECT count(id) FROM orders", frozenset({("orders", "secret_code")})) is True
    # a table named inside a string is no Table node to sqlglot, and DuckDB reads it (the gate):
    # a source the parse cannot name may be any withheld table, under the token rule and the
    # unnamed-columns rule alike, however the string spells it
    assert con.execute("SELECT date_trunc(secret_code, id) FROM query_table('orders') ORDER BY secret_code").fetchall() == dates
    assert len(con.execute("SELECT * FROM query_table('main.orders')").fetchall()[0]) == 2
    for example in ("SELECT date_trunc(secret_code, id) FROM query_table('orders')",
                    "SELECT date_trunc(secret_code, id) FROM query_table('main.orders')",
                    "SELECT * FROM query_table('orders')",
                    # `TABLE orders` is a Table named `TABLE` to sqlglot, a CTE over it its own name
                    "SELECT date_trunc(secret_code, id) FROM (TABLE orders)",
                    "WITH q AS (TABLE orders) SELECT date_trunc(secret_code, id) FROM q",
                    "WITH q AS (TABLE orders) SELECT * FROM q",
                    # a name quoted inside the string keeps its quotes as a token (the gate)
                    "SELECT date_trunc(secret_code, id) FROM query_table('\"orders\"')",
                    "SELECT * FROM query_table('main.\"orders\"')"):
        assert con.execute(example).fetchall(), example
        assert _reads_only_raw(example, frozenset({("orders", "secret_code")})) is False, example
    store = init_store(str(tmp_path / "s.duckdb"))
    embedder = FakeEmbedder()
    snapshot = Snapshot(version="v1", source_id="s", created_at="t", columns=_COLUMNS, examples=[
        Example(question="how many orders by period", sql=sql, tables=["orders"], object_id="orders"),
        Example(question="how many orders", sql="SELECT count(id) FROM orders", tables=["orders"], object_id="orders")])
    build_example_index(store, snapshot, embedder)
    (qvec,) = embedder.embed(["how many orders"])
    shown = _retrieve_examples(store, qvec, {"orders"}, k=5, not_raw=frozenset({("orders", "secret_code")}))
    assert [e.sql for e in shown] == ["SELECT count(id) FROM orders"]
    assert len(_retrieve_examples(store, qvec, {"orders"}, k=5, not_raw=frozenset())) == 2


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
    # Each dotted part on its own quoting (the gate's third advisory): a quoted schema part too.
    assert expression_scope('(SELECT max(id) FROM "PG".orders)', "pg.orders") == "foreign"
    assert expression_scope('(SELECT max(id) FROM pg."ORDERS")', "pg.orders") == "foreign"
    assert expression_scope('(SELECT max(id) FROM "pg".orders)', "pg.orders") == "own"
    assert expression_scope("(SELECT max(id) FROM PG.ORDERS)", "pg.orders") == "own"
    # A four-part name's tail sits in the table slot as a dotted node: foreign, not shrunk.
    assert expression_scope("(SELECT max(id) FROM pg.orders.x.y)", "pg.orders") == "foreign"
    # The aliased path reads the relation the same way.
    assert expression_scope('(SELECT max(r.id) FROM "ORDERS" r)', "orders") == "foreign"
    assert expression_scope('(SELECT max(r.id) FROM "orders" r)', "orders") == "own"


def test_a_keyword_named_column_inside_a_subquery_still_requires_its_clearance():
    # Codex's third review of #172: `true` and `current_date` parse as constants, not columns,
    # and its fourth: `1e` and `123e` parse as numbers where MySQL reads a column so named;
    # the gate: `1ea`, `123abc` and `1_e` parse as a number and an implicit alias, so the
    # digit-led token is read off the expression's own text;
    # and inside a subquery the spelling fallback (which covers a bare expression alone) did
    # not reach them, so a phi column named `true` was offered to the uncleared. Each keyword
    # spelling is reported as a column it may be.
    from mnemiq.semantic.expr_columns import columns_read, keyword_columns
    for spelling in ("true", "current_date", "1e", "123e"):
        columns = [Column(id="orders.id", object_id="orders", name="id", data_type="int"),
                   Column(id=f"orders.{spelling}", object_id="orders", name=spelling, data_type="text",
                          pii_level="phi")]
        for expr in (f"(SELECT max({spelling}) FROM orders)",
                     f"(SELECT id FROM orders ORDER BY {spelling} LIMIT 1)",
                     f"(SELECT count(*) FROM orders WHERE {spelling})"):
            assert spelling in keyword_columns(expr), expr
            assert spelling not in {name for ref in columns_read(expr, "orders") for name in ref}, expr  # not a ref (the gate)
            assert select_metrics(["orders"], [_metric(expr)], _UNCLEARED, columns=columns) == [], expr
            assert select_metrics(["orders"], [_metric(expr)], _CLEARED, columns=columns) == [_metric(expr)], expr
    # A digit-led name sqlglot reads as a number and an implicit alias (`1ea` is `1e AS a`,
    # `123abc` is `123 AS abc`: the gate) parses as a bare projection alone; the token is read
    # off the expression's own text. Inside a call or a WHERE it parses nowhere, and an
    # unparsed expression is withheld from everyone already.
    # MySQL's own identifier characters (Codex's fifth review; the gate on the tenth commit):
    # letters of any script, `$`, a currency symbol, a combining accent in the decomposed form.
    for spelling in ("1ea", "123abc", "1_e", "1é", "1你好", "1$", "1a$b", "1€", "1e\u0301x"):
        columns = [Column(id="orders.id", object_id="orders", name="id", data_type="int"),
                   Column(id=f"orders.{spelling}", object_id="orders", name=spelling, data_type="text",
                          pii_level="phi")]
        for expr in (f"(SELECT {spelling} FROM orders)", f"(SELECT r.{spelling} FROM orders r)"):
            assert spelling in keyword_columns(expr), expr
            assert select_metrics(["orders"], [_metric(expr)], _UNCLEARED, columns=columns) == [], expr
            assert select_metrics(["orders"], [_metric(expr)], _CLEARED, columns=columns) == [_metric(expr)], expr
        unparsed = f"(SELECT max({spelling}) FROM orders)"
        for grants in (_UNCLEARED, _CLEARED):
            assert select_metrics(["orders"], [_metric(unparsed)], grants, columns=columns) == [], unparsed
    # Where no column is so named the keyword is the constant it reads as, and decides nothing,
    # for the selector and for the enrichment classifier alike (the gate: a ref naming no
    # column would mark a personal dimension unresolved).
    plain = "(SELECT count(*) FROM orders WHERE true)"
    assert select_metrics(["orders"], [_metric(plain)], _UNCLEARED, columns=_COLUMNS) == [_metric(plain)]
    assert columns_read("CASE WHEN secret_code IS NOT NULL THEN true ELSE false END", "orders") == {frozenset({"secret_code"})}


def test_a_pivot_carries_the_rest_of_the_row_under_an_alias_that_is_no_source_alias():
    # Codex's tenth review of #172: a PIVOT keeps every column it neither aggregates nor pivots
    # on as a grouping column, which no node names, and its alias is no alias of the source,
    # so `to_json(p)` passed the whole-row guard: the expression was own, `id`, `status` and a
    # bare `p` reported, the metric offered to the uncleared, and DuckDB returned the phi
    # column's value inside the JSON. An UNPIVOT carries the rest of the row the same way. Both
    # are foreign, in either spelling, and withheld from everyone.
    from mnemiq.semantic.expr_columns import expression_scope
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders(id INT, status TEXT, secret_code TEXT)")
    con.execute("INSERT INTO orders VALUES (1, 'x', 'WITHHELD')")
    columns = _COLUMNS + [Column(id="orders.status", object_id="orders", name="status", data_type="text")]
    for expr in ("(SELECT max(to_json(p)) FROM orders PIVOT (sum(id) FOR status IN ('x')) p)",
                 "(SELECT max(to_json(p)) FROM (PIVOT orders ON status USING sum(id)) p)",
                 "(SELECT max(to_json(u)) FROM orders UNPIVOT (v FOR k IN (id)) u)"):
        assert "WITHHELD" in con.execute(f"SELECT {expr}").fetchone()[0], expr  # the leak, live
        assert expression_scope(expr, "orders") == "foreign", expr
        for grants in (_UNCLEARED, _CLEARED):
            assert select_metrics(["orders"], [_metric(expr)], grants, columns=columns) == [], expr


def test_a_character_duckdb_keeps_in_a_name_is_no_boundary():
    # Codex's twenty-second review of #172: U+0085 is part of a name to DuckDB and a no-break
    # space ends it, so `secret_code<U+0085><NBSP>` reads the column `secret_code<U+0085>`; split
    # at U+0085 as Python's `\s` splits, the scan read `secret_code`, and MySQL's class the whole
    # spelling, neither the column, and the phi column so named reached the uncleared through
    # both selectors. The boundary set is the measured one and nothing else: a split DuckDB does
    # not make withholds less, not more.
    from mnemiq.contract import Dimension
    from mnemiq.semantic.expr_columns import expression_scope, keyword_columns
    from mnemiq.semantic.measures import select_dimensions
    for kept in ("\u0085", "\u1680", "\u2028", "\u2029"):
        name = f"secret_code{kept}"
        con = duckdb.connect(":memory:")
        con.execute(f'CREATE TABLE orders(id DATE, "{name}" TEXT)')
        con.execute("INSERT INTO orders VALUES ('2026-03-15', 'year'), ('2026-03-15', 'day')")
        expr = f"(SELECT max(date_trunc({name}\u00a0, id)) FROM orders)"
        assert con.execute(f"SELECT {expr}").fetchall()[0][0] is not None, repr(kept)
        columns = [_COLUMNS[0], Column(id=f"orders.{name}", object_id="orders", name=name, data_type="text", pii_level="phi")]
        assert expression_scope(expr, "orders") == "own", repr(kept)
        assert name in keyword_columns(expr), repr(kept)
        dim = Dimension(id=f"d:{expr}", label=expr, source="orders", expr=expr)
        assert select_metrics(["orders"], [_metric(expr)], _UNCLEARED, columns=columns) == [], repr(kept)
        assert select_metrics(["orders"], [_metric(expr)], _CLEARED, columns=columns) == [_metric(expr)], repr(kept)
        assert select_dimensions(["orders"], [dim], _UNCLEARED, columns=columns) == [], repr(kept)
        assert select_dimensions(["orders"], [dim], _CLEARED, columns=columns) == [dim], repr(kept)


def test_a_mysql_executable_comment_is_withheld_from_everyone():
    # Codex's twenty-first review of #172: MySQL runs `/*! ... */` (and `/*!50000 ... */` from
    # that version) as code, measured, where every other engine reads a comment, so the parse
    # said `own` for a subquery over another table inside one and the selector offered it to a
    # grant on `orders` alone. MySQL lexes the span with comments of its own inside, so it is
    # not expanded here: the opener anywhere, a string literal included, withholds.
    from mnemiq.contract import Dimension
    from mnemiq.semantic.expr_columns import expression_scope
    from mnemiq.semantic.measures import select_dimensions
    for expr in ("(SELECT max(id) /*! + (SELECT max(secret_code) FROM patients) */ FROM orders)",
                 "max(id) /*!50000 + (SELECT max(secret_code) FROM patients) */",
                 "max(id) /*!+(SELECT max(secret_code) FROM patients)*/",
                 "max(id) /*! 1 /* x */ + (SELECT max(secret_code) FROM patients) */",
                 "max(id) /*M! + (SELECT max(secret_code) FROM patients) */",
                 "max(id) /*! + max(secret_code) */",
                 "CASE WHEN id = '/*!' THEN 1 END"):
        assert expression_scope(expr, "orders") == "foreign", expr
        dim = Dimension(id=f"d:{expr}", label=expr, source="orders", expr=expr)
        for grants in (_UNCLEARED, _CLEARED):
            assert select_metrics(["orders"], [_metric(expr)], grants, columns=_COLUMNS) == [], expr
            assert select_dimensions(["orders"], [dim], grants, columns=_COLUMNS) == [], expr
    # a plain comment is one to every engine, its text read raw as before
    plain = "(SELECT max(id) /* + (SELECT max(secret_code) FROM patients) */ FROM orders)"
    assert expression_scope(plain, "orders") == "own"
    assert select_metrics(["orders"], [_metric(plain)], _UNCLEARED, columns=_COLUMNS) == []
    assert select_metrics(["orders"], [_metric(plain)], _CLEARED, columns=_COLUMNS) == [_metric(plain)]


def test_a_column_the_parser_reads_as_a_unit_or_a_type_still_requires_its_clearance():
    # Codex's eleventh review of #172: sqlglot reads `date_trunc(secret_code, id)` as
    # `DATE_TRUNC('SECRET_CODE', id)`, a string where the column stood, so the parse held no
    # node naming the phi column, the expression was own with `id` alone reported, and the
    # metric was offered to the uncleared, where DuckDB reads the unit column: a withheld
    # `year` and `day` give different dates (measured here). The same for the quoted name,
    # EXTRACT's part, an INTERVAL's unit, LAST_DAY's and TIMESTAMP_TRUNC's unit and a CAST's
    # type name. Every token of the text is a column it may be; a token naming none decides
    # nothing.
    from mnemiq.semantic.expr_columns import expression_scope, keyword_columns
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders(id DATE, secret_code TEXT)")
    con.execute("INSERT INTO orders VALUES ('2026-03-15', 'year'), ('2026-03-15', 'day')")
    dates = con.execute("SELECT date_trunc(secret_code, id) FROM orders ORDER BY secret_code").fetchall()
    assert dates[0] != dates[1]  # the withheld value decides the result
    for expr in ("(SELECT max(date_trunc(secret_code, id)) FROM orders)",
                 '(SELECT max(date_trunc("secret_code", id)) FROM orders)',
                 # DuckDB ends a name at eighteen characters MySQL's identifier class holds
                 # (Codex's twentieth review), three of them outside Python's `\s` (the gate on
                 # the twenty-ninth commit): the token under both classes
                 "(SELECT max(date_trunc(secret_code\u00a0, id)) FROM orders)",
                 "(SELECT max(date_trunc(secret_code\u2003, id)) FROM orders)",
                 "(SELECT max(date_trunc(secret_code\u200b, id)) FROM orders)",
                 "(SELECT max(date_trunc(secret_code\u2060, id)) FROM orders)",
                 "(SELECT max(date_trunc(secret_code\ufeff, id)) FROM orders)",
                 "(SELECT max(extract(secret_code FROM id)) FROM orders)",
                 "(SELECT max(date_add(id, INTERVAL 1 secret_code)) FROM orders)",
                 "(SELECT max(last_day(id, secret_code)) FROM orders)",
                 "(SELECT max(timestamp_trunc(id, secret_code)) FROM orders)",
                 "(SELECT max(cast(id AS secret_code)) FROM orders)"):
        assert expression_scope(expr, "orders") == "own", expr
        assert "secret_code" in keyword_columns(expr), expr
        assert select_metrics(["orders"], [_metric(expr)], _UNCLEARED, columns=_COLUMNS) == [], expr
        assert select_metrics(["orders"], [_metric(expr)], _CLEARED, columns=_COLUMNS) == [_metric(expr)], expr
    # An open comment runs to the end of the text, its last characters kept (the gate on the
    # twenty-first commit: cut as if closed, `secret_code` read `secret_cod`).
    for expr in ("id /* /* */ /*! + secret_code */", "id /* /* */ /*!+secret_code*/", "id /* /* */ + secret_code",
                 "id /* /* */ /*!+secret_code"):   # open under both readings
        assert "secret_code" in keyword_columns(expr), expr
    # Digits alone are a number to every engine, never a column: a phi column named `123`
    # withholds nothing from `LIMIT 123`.
    columns = _COLUMNS + [Column(id="orders.123", object_id="orders", name="123", data_type="text",
                                 pii_level="phi")]
    limited = "(SELECT count(*) FROM (SELECT id FROM orders LIMIT 123) q)"
    assert "123" not in keyword_columns(limited)
    assert select_metrics(["orders"], [_metric(limited)], _UNCLEARED, columns=columns) == [_metric(limited)]


def test_a_quoted_name_the_parser_reads_as_a_unit_is_read_whole():
    # Codex's twelfth review of #172: `date_trunc("secret code", id)` parses as the unit string
    # too, and the token rule split the quoted name at its space (`"secret.code"` at its dot),
    # dropped `"123"` as digits and `"null"` as the constant, and knew no character of `"💩"`,
    # so each phi column so named was offered to the uncleared through the metric and the
    # dimension selectors alike, DuckDB reading the unit column. A quoted span is a column it
    # may be as written, doubled quotes undoubled; an unquoted astral character is a token.
    from mnemiq.contract import Dimension
    from mnemiq.semantic.expr_columns import expression_scope, keyword_columns
    from mnemiq.semantic.measures import select_dimensions
    con = duckdb.connect(":memory:")
    con.execute('CREATE TABLE orders(id DATE, "secret code" TEXT)')
    con.execute("INSERT INTO orders VALUES ('2026-03-15', 'year'), ('2026-03-15', 'day')")
    dates = con.execute('SELECT date_trunc("secret code", id) FROM orders ORDER BY "secret code"').fetchall()
    assert dates[0] != dates[1]
    for name, spelling in (("secret code", '"secret code"'), ("secret.code", '"secret.code"'), ("💩", '"💩"'),
                           ("💩", "💩"), ("123", '"123"'), ("null", '"null"'), ('se"cret', '"se""cret"'),
                           ("secret code", "`secret code`"), ("secret code", "'secret code'"),
                           # a quote inside a comment pairs with nothing (Codex's thirteenth review)
                           ("secret code", '/* " */ "secret code"'), ("secret code", '"secret code" /* " */'),
                           ("secret code", "-- '\n'secret code'"),
                           # a line comment ends at a carriage return to DuckDB and Postgres (Codex's
                           # eighteenth review) and at a line feed alone to MySQL and SQLite (the gate):
                           # both readings, joined
                           ("secret code", '-- "\r"secret code"'), ("secret code", "-- '\r\n'secret code'"),
                           # a comment's text is code to an engine that runs it (the gate): MySQL reads
                           # `--secret_code` as two minus signs and the column (an executable comment,
                           # `/*! ... */`, is withheld from everyone since Codex's twenty-first review)
                           ("secret_code", "id --secret_code\n"),
                           ("secret code", 'id /* " */ --"secret code"\n'),
                           # a block comment nests, as DuckDB and Postgres read it (Codex's fourteenth)
                           ("secret code", '/* /* */ " */ "secret code"'), ("secret code", '/* /* /* */ */ " */ "secret code"'),
                           ("secret code", '/* /* " */ "secret code"'),
                           # and does not nest to MySQL, which reads a name across the nested region whole
                           ("secret */ code", "/* /* */ `secret */ code`"),
                           # a Postgres Unicode-escaped span decodes to the name (Codex's fifteenth review):
                           # `\XXXX` and `\+XXXXXX` a code point each, a surrogate pair one character,
                           # the escape doubled itself; `U&` touching the quote, in either case
                           ("secret_code", 'U&"secret\\005fcode"'), ("secret_code", "u&'secret\\005fcode'"),
                           ("secret_code", 'U&"secret\\+00005fcode"'), ("😄", 'U&"\\d83d\\de04"'),
                           ("secret\\code", 'U&"secret\\\\code"'), ("secret_code", 'U&"secret_code"')):
        columns = [Column(id="orders.id", object_id="orders", name="id", data_type="date"),
                   Column(id=f"orders.{name}", object_id="orders", name=name, data_type="text", pii_level="phi")]
        expr = f"(SELECT max(date_trunc({spelling}, id)) FROM orders)"
        assert expression_scope(expr, "orders") == "own", expr
        assert name in keyword_columns(expr), expr
        assert select_metrics(["orders"], [_metric(expr)], _UNCLEARED, columns=columns) == [], expr
        assert select_metrics(["orders"], [_metric(expr)], _CLEARED, columns=columns) == [_metric(expr)], expr
        dim = Dimension(id="d", label=expr, source="orders", expr=expr)
        assert select_dimensions(["orders"], [dim], _UNCLEARED, columns=columns) == [], expr
        assert select_dimensions(["orders"], [dim], _CLEARED, columns=columns) == [dim], expr
    # A `UESCAPE 'c'` after the closing quote changes the escape character; sqlglot cannot parse
    # the clause, so the expression is unparsed and withheld from everyone, and the spelling is
    # still read. `U& "x"`, the quote not touched, is the `&` operator over a column `U`, which
    # DuckDB binds (measured: a binder error naming `U`), and decodes to nothing.
    assert "secret_code" in keyword_columns("U&\"secret!005fcode\" UESCAPE '!'")
    assert "secret_code" in keyword_columns("u&'secret!005fcode'uescape'!'")
    assert "secret!code" in keyword_columns("U&\"secret!!code\" UESCAPE '!'")
    assert "secret_code" not in keyword_columns('U& "secret\\005fcode"')
    assert "secret_code" not in keyword_columns('U&"secret\\zzzzcode"')   # an escape Postgres refuses
    # A comment may stand between the closing quote and UESCAPE, or between it and its
    # character (the gate on the twenty-second commit); an escape short of its digits is none
    assert "secret_code" in keyword_columns("U&\"secret!005fcode\" /* c */ UESCAPE '!'")
    assert "secret_code" in keyword_columns("U&\"secret!005fcode\" -- c\r UESCAPE '!'")   # the comment ends at the carriage return
    # MySQL's reading, the comment running past the carriage return to the line feed, reads the
    # name whole (the gate); sqlglot parses neither spelling, so the expression is withheld
    # from everyone, and the spelling is still read
    assert "secret code" in keyword_columns("-- \r` \n`secret code`")
    assert "secret code" in keyword_columns('-- \r" \n"secret code"')
    # MySQL opens a line comment with `-- ` and `#`, and `--` before a quote is subtraction and
    # negation, the quoted name running on across the line feed (Codex's nineteenth review):
    # the name with the line feed and the `--` inside it is a column it may be, under both
    # selectors
    name = "secret\n-- code"
    columns = [Column(id="orders.id", object_id="orders", name="id", data_type="date"),
               Column(id=f"orders.{name}", object_id="orders", name=name, data_type="text", pii_level="phi")]
    expr = "(SELECT max(id --`secret\n-- code`\n) FROM orders)"
    assert expression_scope(expr, "orders") == "own"
    assert name in keyword_columns(expr)
    assert select_metrics(["orders"], [_metric(expr)], _UNCLEARED, columns=columns) == []
    assert select_metrics(["orders"], [_metric(expr)], _CLEARED, columns=columns) == [_metric(expr)]
    dim = Dimension(id="d", label=expr, source="orders", expr=expr)
    assert select_dimensions(["orders"], [dim], _UNCLEARED, columns=columns) == []
    assert "secret code" in keyword_columns("id # `\n`secret code`")   # the `#` opener: a quote inside its comment pairs with nothing (the gate)
    assert "secret\n-- code" in keyword_columns("id --\u00a0`secret\n-- code`\n")   # a no-break space opens no MySQL comment: the name read whole (the gate)
    # an engine not measured, reading `#` comments and ending a name as DuckDB does: the token
    # classes pair with every comment reading (the gate on the twenty-ninth commit)
    assert "secret_code" in keyword_columns('id # "\n secret_code\u200b"')
    assert "secret_code" in keyword_columns("U&\"secret!005fcode\" -- c\n UESCAPE /* d */ '!'")
    assert keyword_columns('U&"secret\\5f"') == {"u", "secret\\5f"}
    assert keyword_columns('U&"secret\\+00005"') == {"u", "secret\\+00005"}


def test_a_spelling_past_postgres_identifier_limit_reads_the_column_named_by_its_first_63_bytes():
    # Codex's sixteenth review of #172: Postgres truncates an identifier, quoted or not, to 63
    # bytes, so a 64-character reference reads the phi column named by its first 63 and the
    # spelling as written named nothing. Whole characters only: 32 two-byte characters are 64
    # bytes and cut to 31. A Unicode-escaped span is cut after decoding.
    from mnemiq.contract import Dimension
    from mnemiq.semantic.expr_columns import expression_scope, keyword_columns
    from mnemiq.semantic.measures import select_dimensions
    # The server encoding is unknown here and the cut falls where it says: `あ` is three bytes
    # in UTF-8 and two in EUC_JP, so a name of 31 `あ` and an `a` (63 bytes there) is reached
    # by that name and one more `あ` (Codex's seventeenth review); every prefix the cut may
    # fall on is read, which for an ASCII spelling is its first 63 characters alone.
    ascii_name, wide_name, euc_name = "s" * 63, "é" * 31, "あ" * 31 + "a"
    for name, spelling in ((ascii_name, '"' + ascii_name + 'x"'), (ascii_name, ascii_name + "x"),
                           (wide_name, '"' + wide_name + "é" + '"'), (wide_name, wide_name + "é"),
                           (ascii_name, 'U&"' + "s" * 62 + '\\0073x"'),
                           (euc_name, '"' + euc_name + "あ" + '"'), (euc_name, euc_name + "あ")):
        columns = [Column(id="orders.id", object_id="orders", name="id", data_type="date"),
                   Column(id=f"orders.{name}", object_id="orders", name=name, data_type="text", pii_level="phi")]
        expr = f"(SELECT max({spelling}) FROM orders)"
        assert expression_scope(expr, "orders") == "own", expr
        assert name in keyword_columns(expr), expr
        assert select_metrics(["orders"], [_metric(expr)], _UNCLEARED, columns=columns) == [], expr
        assert select_metrics(["orders"], [_metric(expr)], _CLEARED, columns=columns) == [_metric(expr)], expr
        dim = Dimension(id="d", label=expr, source="orders", expr=expr)
        assert select_dimensions(["orders"], [dim], _UNCLEARED, columns=columns) == [], expr
        assert select_dimensions(["orders"], [dim], _CLEARED, columns=columns) == [dim], expr
    # a spelling that fits in every encoding is not cut, and an ASCII spelling one past the
    # limit reads one prefix; a non-ASCII spelling reads its prefixes down to where four bytes
    # a character would cut it, each a prefix of the spelling and nothing else
    assert keyword_columns('"' + ascii_name + '"') == {ascii_name}
    assert keyword_columns('"' + ascii_name + 'x"') == {ascii_name + "x", ascii_name}
    wide = keyword_columns('"' + wide_name + 'x"')
    assert wide_name + "x" in wide and all((wide_name + "x").startswith(w) and len(w) >= 15 for w in wide), wide
    # a character four bytes wide in some server encoding (EUC_TW's later planes,
    # MULE_INTERNAL) cuts sixteen of them at fifteen, where UTF-8 alone, three bytes each,
    # would cut nothing
    assert "あ" * 15 in keyword_columns('"' + "あ" * 16 + '"')
    assert "あ" * 14 not in keyword_columns('"' + "あ" * 16 + '"')
    # cut as written, lowercased after (the gate): `İ` is two bytes, its lowercase three, so
    # Postgres reads `s` * 62 (the `İ` not whole at 63) where the lowercased spelling cut at
    # another boundary; a lone surrogate, which the identifier class admits, raises nothing
    assert "s" * 62 in keyword_columns('"' + "s" * 62 + 'İx"')
    assert "a\ud800" in keyword_columns('max("a\ud800")')
    assert "a" * 63 in keyword_columns('max("' + "a" * 63 + '\ud800")')


def test_a_struct_column_named_like_the_source_keeps_its_root_reading():
    # Codex's eighth review of #172: with a phi struct column named `orders` on `orders`, the
    # parser reads `orders.value` as a qualifier and a column, and the root reading was kept
    # for unqualified paths alone, so the struct reached the uncleared.
    columns = [Column(id="orders.id", object_id="orders", name="id", data_type="int"),
               Column(id="orders.orders", object_id="orders", name="orders", data_type="struct",
                      pii_level="phi")]
    for expr in ("(SELECT max(orders.value) FROM orders r)", "(SELECT max(orders.value) FROM orders)",
                 "(SELECT max(r.orders.value) FROM orders r)"):
        assert select_metrics(["orders"], [_metric(expr)], _UNCLEARED, columns=columns) == [], expr
        assert select_metrics(["orders"], [_metric(expr)], _CLEARED, columns=columns) == [_metric(expr)], expr
    # Without such a column the root names nothing and decides nothing: offered to everyone.
    plain = "(SELECT max(orders.id) FROM orders)"
    assert select_metrics(["orders"], [_metric(plain)], _UNCLEARED, columns=_COLUMNS) == [_metric(plain)]
