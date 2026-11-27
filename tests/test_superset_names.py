"""One name mapping at the Superset boundary (`mnemiq.superset.names`): a name as Superset shows it and as the database
stores it, and the statements mnemiq sends with each table written as Superset names it.

Superset reads Oracle through SQLAlchemy, whose Oracle dialect shows a stored `ORDERS` as `orders` and keeps a quoted
`"OrderItems"` as it is (`normalize_name`), and reads a name it is given back to the stored one (`denormalize_name`).
The mapping follows that rule exactly: the names below were given to SQLAlchemy 2.0.52's Oracle dialect, the one
Superset 7.0.0rc2 runs, and are held here as it answered. On PostgreSQL every function here is the identity.

The snapshot, the prompts and every statement mnemiq decides keep the database's spelling; the mapping is applied only
where a name crosses to or from Superset.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
import sqlglot
from sqlglot import exp

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.superset import names
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.chart_form import DatasetFacts
from mnemiq.superset.charts import ChartSpec, Family, Filter, WrittenRef, config_chart_request
from mnemiq.superset.grants import ListedDataset
from mnemiq.superset.measures import Measure, MeasureKind
from mnemiq.superset.names import UnreadableStatement, for_superset, from_superset, stored_name, superset_name
from mnemiq.superset.payload import SupersetQueryFailed
from mnemiq.superset.planner import BuildPlan, PlannedDataset
from mnemiq.sql.identifiers import resolve_identifier
from mnemiq.sql.scope import base_tables

# SQLAlchemy 2.0.52's Oracle dialect (Superset 7.0.0rc2's) on each name: `normalize_name` and `denormalize_name`, as it
# answered them, not as this module computes them. `test_the_mapping_is_sqlalchemys_where_it_is_installed` asks it
# again wherever SQLAlchemy can be imported.
SQLALCHEMY_ORACLE = [
    ("ORDERS", "orders", "ORDERS"),
    ("orders", "orders", "ORDERS"),
    ("ORDER_ITEMS", "order_items", "ORDER_ITEMS"),
    ("order_items", "order_items", "ORDER_ITEMS"),
    ("OrderItems", "OrderItems", "OrderItems"),
    ("ORDER", "ORDER", "ORDER"),
    ("order", "order", "order"),
    ("Order", "Order", "Order"),
    ("SHOP", "shop", "SHOP"),
    ("shop", "shop", "SHOP"),
    ("PUBLIC", "PUBLIC", "PUBLIC"),
    ("public", "public", "public"),
    ("ORDER#", "ORDER#", "ORDER#"),
    ("order#", "order#", "order#"),
    ("A$B", "a$b", "A$B"),
    ("a$b", "a$b", "A$B"),
    ("$A", "$A", "$A"),
    ("$a", "$a", "$a"),
    ("_X", "_X", "_X"),
    ("_x", "_x", "_x"),
    ("X_", "x_", "X_"),
    ("x_", "x_", "X_"),
    ("1A", "1A", "1A"),
    ("1a", "1a", "1a"),
    ("A1", "a1", "A1"),
    ("a1", "a1", "A1"),
    ("123", "123", "123"),
    ("_", "_", "_"),
    ("$", "$", "$"),
    ("#", "#", "#"),
    ("", "", ""),
    ("订单", "订单", "订单"),
    ("ORDERS订单", "ORDERS订单", "ORDERS订单"),
    ("orders订单", "orders订单", "orders订单"),
    ("STRA\xdfE", "STRA\xdfE", "STRA\xdfE"),
    ("stra\xdfe", "stra\xdfe", "stra\xdfe"),
    ("\xc4PFEL", "\xc4PFEL", "\xc4PFEL"),
    ("\xe4pfel", "\xe4pfel", "\xe4pfel"),
    ("\xc4pfel", "\xc4pfel", "\xc4pfel"),
    ("K", "k", "K"),
    ("K", "k", "K"),
    ("k", "k", "K"),
    ("ſUM", "ſUM", "ſUM"),
    ("ſum", "ſum", "SUM"),
    ("SUM", "sum", "SUM"),
    ("ıD", "ıD", "ıD"),
    ("ID", "id", "ID"),
    ("id", "id", "ID"),
    ("İD", "İD", "İD"),
    ("ORDER ITEMS", "ORDER ITEMS", "ORDER ITEMS"),
    ("order items", "order items", "order items"),
    ("ORDERS\n", "orders\n", "ORDERS\n"),
    ("orders\n", "orders\n", "ORDERS\n"),
    ("SYSDATE", "sysdate", "SYSDATE"),
    ("sysdate", "sysdate", "SYSDATE"),
    ("ROWNUM", "rownum", "ROWNUM"),
    ("rownum", "rownum", "ROWNUM"),
    ("USER", "USER", "USER"),
    ("user", "user", "user"),
    ("LEVEL", "LEVEL", "LEVEL"),
    ("level", "level", "level"),
    ("UID", "UID", "UID"),
    ("uid", "uid", "uid"),
    ("SESSION", "session", "SESSION"),
    ("session", "session", "SESSION"),
    ("FILE", "file", "FILE"),
    ("file", "file", "FILE"),
    ("DUAL", "dual", "DUAL"),
    ("dual", "dual", "DUAL"),
    ("VIRTUAL_TABLE", "virtual_table", "VIRTUAL_TABLE"),
    ("virtual_table", "virtual_table", "VIRTUAL_TABLE"),
    ("A-B", "A-B", "A-B"),
    ("a-b", "a-b", "a-b"),
    ("ΣΙΓΜΑ", "ΣΙΓΜΑ", "ΣΙΓΜΑ"),
    ("σιγμα", "σιγμα", "σιγμα"),
    ("ǅ", "ǅ", "ǅ"),
    ("ǆ", "ǆ", "ǆ"),
    ("Ǆ", "Ǆ", "Ǆ"),
    ("ﬁ", "ﬁ", "ﬁ"),
    ("FI", "fi", "FI"),
    ("ǈ", "ǈ", "ǈ"),
]
# The Oracle dialect's `RESERVED_WORDS` in SQLAlchemy 2.0.52: none of them is shown in lower case, whatever its case.
SQLALCHEMY_RESERVED = frozenset(
    "ALL ALTER AND ANY AS ASC BETWEEN BY CHAR CHECK CLUSTER COMMENT COMPRESS CONNECT CREATE CURRENT DATE DECIMAL "
    "DEFAULT DELETE DESC DISTINCT DROP ELSE EXCLUSIVE EXISTS FLOAT FOR FROM GRANT GROUP HAVING IDENTIFIED IN INDEX "
    "INSERT INTEGER INTERSECT INTO IS LEVEL LIKE LOCK LONG MINUS MODE NOCOMPRESS NOT NOWAIT NULL NUMBER OF ON OPTION "
    "OR ORDER PCTFREE PRIOR PUBLIC RAW RENAME RESOURCE REVOKE SELECT SET SHARE SIZE SMALLINT START SYNONYM TABLE THEN "
    "TO TRIGGER UID UNION UNIQUE UPDATE USER VALUES VARCHAR VARCHAR2 VIEW WHERE WITH".split())


def _every_case(words) -> list[str]:
    return [spelled for word in sorted(words) for spelled in (word, word.lower(), word.capitalize())]


# ---------------------------------------------------------------- the mapping

@pytest.mark.parametrize(("stored", "shown"), [
    ("ORDERS", "orders"), ("ORDER_ITEMS", "order_items"), ("OrderItems", "OrderItems"), ("ORDER", "ORDER"),
    ("order", "order"), ("ORDER#", "ORDER#"), ("A$B", "a$b"), ("SHOP", "shop")])
def test_a_stored_name_round_trips_through_supersets_spelling(superset_oracle, stored, shown):
    """A name stored all upper case that Oracle reads unquoted is shown in lower case; a reserved word (`ORDER`,
    `order`), a name with a character an unquoted name may not hold for SQLAlchemy (`#`) and a mixed-case one are shown
    as stored. Each is read back to the name it came from."""
    assert superset_name(stored) == shown
    assert stored_name(shown) == stored


def test_a_mixed_case_name_is_never_lower_cased(superset_oracle):
    """A quoted mixed-case `"OrderItems"` round-trips through Superset quoted and as stored, in every direction and in
    a statement; Superset's `orderitems` is another table, which Oracle reads, unquoted, as ORDERITEMS."""
    assert superset_name("OrderItems") == "OrderItems" and stored_name("OrderItems") == "OrderItems"
    assert stored_name("orderitems") == "ORDERITEMS"
    assert for_superset('SELECT "ID" FROM "OrderItems"') == 'SELECT "ID" FROM "OrderItems"'
    assert from_superset('SELECT "ID" FROM "OrderItems"') == 'SELECT "ID" FROM "OrderItems"'


def test_a_name_stored_in_lower_case_reads_back_as_the_upper_case_one(superset_oracle):
    """SQLAlchemy's one lossy turn: a quoted lower-case `orders` is shown as itself, while a name Superset shows as
    `orders` reads ORDERS, as SQLAlchemy writes it unquoted; so the mapping says what Superset reads, not where a name
    came from."""
    assert superset_name("orders") == "orders"
    assert stored_name("orders") == "ORDERS"


def test_the_mapping_is_sqlalchemys(superset_oracle):
    wrong = [(name, superset_name(name), stored_name(name)) for name, normalized, denormalized in SQLALCHEMY_ORACLE
             if (superset_name(name), stored_name(name)) != (normalized, denormalized)]
    assert wrong == []
    assert names._RESERVED == {word.lower() for word in SQLALCHEMY_RESERVED}
    assert [word for word in _every_case(SQLALCHEMY_RESERVED)
            if (superset_name(word), stored_name(word)) != (word, word)] == []


def test_the_mapping_is_sqlalchemys_where_it_is_installed(superset_oracle):
    """mnemiq does not depend on SQLAlchemy; where it can be imported (Superset's own environment), it is asked again,
    so the table above and the rule cannot drift from it unseen."""
    oracle = pytest.importorskip("sqlalchemy.dialects.oracle.base")
    dialect = oracle.OracleDialect()
    asked = [name for name, _, _ in SQLALCHEMY_ORACLE] + _every_case(oracle.RESERVED_WORDS)
    wrong = [name for name in asked if (superset_name(name), stored_name(name))
             != (str(dialect.normalize_name(name)), str(dialect.denormalize_name(name)))]
    assert wrong == []


@pytest.mark.parametrize("name", ["ORDERS", "orders", "OrderItems", "ORDER", "order", "A$B", "K", ""])
def test_on_postgres_a_name_is_shown_as_stored(name):
    assert superset_name(name) == name and stored_name(name) == name


def test_on_postgres_a_statement_is_sent_as_written():
    """Returned as it came, not printed again, whatever it holds: PostgreSQL's statements are byte-identical."""
    for sql in ('SELECT "REVENUE" FROM "ORDERS"', "SELECT * FROM ORDERS", "SELECT * FROM ORDERS WHERE (", ""):
        assert for_superset(sql) is sql and from_superset(sql) is sql


# ---------------------------------------------------------------- statements

def test_for_superset_writes_supersets_spelling(superset_oracle):
    """Each table the statement reads, as Superset names its dataset (rc2's access check matches a statement's tables
    to dataset names exactly, and its row rules attach by them), and each column as written: Superset matches no column
    by name, and an unquoted `sysdate` would read the function, not a column stored as SYSDATE. A table renamed so is
    aliased by its own name (`test_a_renamed_table_keeps_its_name_under_a_row_rule`)."""
    assert for_superset('SELECT "REVENUE" FROM "ORDERS"') == 'SELECT "REVENUE" FROM orders "ORDERS"'
    assert for_superset('SELECT "SYSDATE", "ROWNUM" FROM "ORDERS"') == 'SELECT "SYSDATE", "ROWNUM" FROM orders "ORDERS"'


@pytest.mark.parametrize(("written", "sent"), [
    ('SELECT "ID" FROM ORDERS', 'SELECT "ID" FROM orders "ORDERS"'),
    ('SELECT "ID" FROM Orders', 'SELECT "ID" FROM orders "ORDERS"'),
    ('SELECT "ID" FROM orders', 'SELECT "ID" FROM orders'),
    ('SELECT "ID" FROM "ORDER_ITEMS"', 'SELECT "ID" FROM order_items "ORDER_ITEMS"'),
    ('SELECT "ID" FROM OrderItems', 'SELECT "ID" FROM orderitems "ORDERITEMS"'),
    ('SELECT "ID" FROM "orders"', 'SELECT "ID" FROM "orders"'),
    ('SELECT "ID" FROM "ORDER"', 'SELECT "ID" FROM "ORDER"'),
    ('SELECT "ID" FROM "ORDER#"', 'SELECT "ID" FROM "ORDER#"'),
    ('SELECT "ID" FROM order#', 'SELECT "ID" FROM "ORDER#" "ORDER#"'),
    ('SELECT "ID" FROM "A$B"', 'SELECT "ID" FROM a$b "A$B"'),
    ('SELECT "ID" FROM "SHOP"."ORDERS"', 'SELECT "ID" FROM shop.orders "ORDERS"'),
    # SQLAlchemy shows the Kelvin sign's table as `k`, which Oracle reads as another table, K: it stays quoted.
    ('SELECT "ID" FROM "K"', 'SELECT "ID" FROM "K"'),
])
def test_each_table_is_written_as_superset_names_it(superset_oracle, written, sent):
    assert for_superset(written) == sent


# Statements holding names that are no table of the source: a CTE, a derived table's alias, a table's alias, an output
# alias, a column qualifier, a function, a literal and a comment. Each with what is sent for it.
NOT_TABLES = [
    ('WITH "TOTALS" AS (SELECT "CUSTOMER_ID", COUNT(*) AS "ORDERS" FROM "ORDERS" GROUP BY "CUSTOMER_ID") '
     'SELECT "T"."ORDERS", NVL("C"."NAME", \'"ORDERS"\') AS "NAME", SYSDATE FROM "TOTALS" "T" '
     'JOIN "CUSTOMERS" "C" ON "C"."ID" = "T"."CUSTOMER_ID" '
     'WHERE "C"."ID" IN (SELECT "CUSTOMER_ID" FROM "RETURNS") AND "C"."NOTE" <> \'FROM ORDERS\'',
     'WITH "TOTALS" AS (SELECT "CUSTOMER_ID", COUNT(*) AS "ORDERS" FROM orders "ORDERS" GROUP BY "CUSTOMER_ID") '
     'SELECT "T"."ORDERS", NVL("C"."NAME", \'"ORDERS"\') AS "NAME", SYSDATE FROM "TOTALS" "T" '
     'JOIN customers "C" ON "C"."ID" = "T"."CUSTOMER_ID" '
     'WHERE "C"."ID" IN (SELECT "CUSTOMER_ID" FROM returns "RETURNS") AND "C"."NOTE" <> \'FROM ORDERS\''),
    ('SELECT "D"."N" FROM (SELECT COUNT(*) AS "N" FROM "ORDERS") "D"',
     'SELECT "D"."N" FROM (SELECT COUNT(*) AS "N" FROM orders "ORDERS") "D"'),
    ('SELECT "ORDERS"."ID" FROM "ORDERS" WHERE "ORDERS"."ID" > 1',
     'SELECT "ORDERS"."ID" FROM orders "ORDERS" WHERE "ORDERS"."ID" > 1'),
    ('-- reads "ORDERS"\nSELECT 1 FROM "ORDERS"', '-- reads "ORDERS"\nSELECT 1 FROM orders "ORDERS"'),
]


@pytest.mark.parametrize(("written", "sent"), NOT_TABLES)
def test_only_the_tables_a_statement_reads_are_rewritten(superset_oracle, written, sent):
    assert for_superset(written) == sent


def test_writing_a_statement_twice_changes_nothing_more(superset_oracle):
    for written, _ in NOT_TABLES:
        once = for_superset(written)
        assert for_superset(once) == once


def test_a_table_whose_reading_is_not_known_is_left_as_written(superset_oracle):
    """An unquoted name holding a letter whose fold on Oracle mnemiq has not measured (a Greek letter with a subscript
    iota, `resolve_exactly`) names a table mnemiq cannot spell another way and be sure it is the same one."""
    sql = 'SELECT "ID" FROM ᾀx'
    assert for_superset(sql) == sql and from_superset(sql) == sql


def test_a_rewrite_that_does_not_read_back_is_refused(superset_oracle):
    """The text is rewritten where each table's name stands, then read again: a rewrite that reads as another
    statement (here a name written bare that reads as a table and an alias) is refused, never sent."""
    with pytest.raises(UnreadableStatement, match="does not read as"):
        names._rewritten('SELECT 1 FROM "ORDERS"', lambda stored: names.exp.Identifier(this="x y", quoted=False),
                         aliased=False)


def test_two_parses_are_one_statement_only_node_for_node():
    """The read-back compares the trees themselves, not their printing: two nodes of different kinds over the same
    name differ, and so do two names in different case; where a node stands in the text does not count."""
    same = names._same_tree
    assert same(sqlglot.parse_one('SELECT 1 FROM "ORDERS"', read="oracle"),
                sqlglot.parse_one('SELECT  1  FROM "ORDERS"', read="oracle"))
    assert not same(exp.Column(this=exp.to_identifier("A")), exp.Table(this=exp.to_identifier("A")))
    assert not same(exp.to_identifier("ORDERS", quoted=True), exp.to_identifier("orders", quoted=True))


def test_a_table_name_with_no_place_in_the_text_is_refused(superset_oracle, monkeypatch):
    unplaced = names.exp.Table(this=names.exp.Identifier(this="ORDERS", quoted=True))
    monkeypatch.setattr(names, "base_tables", lambda tree, dialect: [unplaced])
    with pytest.raises(UnreadableStatement, match="no place in the text"):
        for_superset('SELECT 1 FROM "ORDERS"')


def _superset_tables(sql: str) -> list[tuple[str, str]]:
    """The tables a statement reads as rc2's access check names them, `Table(source.name, source.db)`: each table
    source's schema and name as written, quoting dropped, CTE references left out."""
    return [(table.db, table.name) for table in base_tables(sqlglot.parse_one(sql, read="oracle"), "oracle")]


def _row_ruled(sql: str) -> str:
    """`sql` with every table it reads wrapped as rc2 applies a row rule on Oracle (`RLSAsSubqueryTransformer` in
    `superset/sql/parse.py`, Oracle's `get_rls_method` being AS_SUBQUERY): `(SELECT * FROM <table> WHERE <rule>)`,
    aliased by the table's own alias where it has one, else by its name quoted as written there. A copy of rc2's rule,
    since mnemiq does not install Superset; checked against rc2's own in the task's live notes."""
    tree = sqlglot.parse_one(sql, read="oracle")
    for table in base_tables(tree, "oracle"):
        alias = table.args.get("alias") or exp.TableAlias(this=exp.Identifier(this=table.name, quoted=True))
        bare = table.copy()
        bare.set("alias", None)
        table.replace(exp.Subquery(this=exp.select("*").from_(bare).where("1 = 1"), alias=alias.copy()))
    return tree.sql(dialect="oracle")


def _unresolved(sql: str) -> list[str]:
    """Each column qualifier in `sql` that names no source it can see, as Oracle resolves names: a source in its own
    query's FROM or one enclosing it, by its alias, else by its table's name."""
    tree = sqlglot.parse_one(sql, read="oracle")
    unresolved = []
    for column in tree.find_all(exp.Column):
        qualifier = column.args.get("table")
        if qualifier is None:
            continue
        wanted, select, found = resolve_identifier(qualifier, "oracle"), column.parent_select, False
        while select is not None and not found:
            sources = [select.args["from_"].this] if select.args.get("from_") else []
            sources += [join.this for join in select.args.get("joins") or []]
            exposed = {resolve_identifier(source.args["alias"].this if source.args.get("alias") else source.this,
                                          "oracle") for source in sources}
            found = wanted in exposed
            select = select.parent_select
        if not found:
            unresolved.append(column.sql(dialect="oracle"))
    return unresolved


@pytest.mark.parametrize("decided", [
    'SELECT "ORDERS"."REVENUE" FROM "ORDERS"',
    "SELECT ORDERS.REVENUE FROM ORDERS",
    'SELECT "ORDERS"."ID" FROM "ORDERS" WHERE EXISTS (SELECT 1 FROM "RETURNS" '
    'WHERE "RETURNS"."ORDER_ID" = "ORDERS"."ID")',
])
def test_a_renamed_table_keeps_its_name_under_a_row_rule(superset_oracle, decided):
    """rc2 wraps a table a row rule applies to as a subquery aliased by the table's name quoted as written, `"orders"`
    for a bare `orders`, which Oracle reads as another name than ORDERS: a column the statement qualifies by its table
    would name nothing. So a table the rewrite renames and that has no alias is aliased by its stored name, quoted,
    which the wrap keeps: every qualifier resolves before and after the wrap, and the access check still reads
    `orders`. Unaliased, it would not."""
    sent = for_superset(decided)
    assert sorted(_superset_tables(sent)) == [("", "orders")] + ([("", "returns")] if "RETURNS" in decided else [])
    assert _unresolved(sent) == [] and _unresolved(_row_ruled(sent)) == []
    unaliased = sent.replace('orders "ORDERS"', "orders").replace('returns "RETURNS"', "returns")
    assert unaliased != sent
    assert _unresolved(unaliased) == [] and _unresolved(_row_ruled(unaliased)) != []


def test_a_made_datasets_join_keeps_its_columns_under_a_row_rule(superset_oracle):
    """The SQL of a joined dataset the build makes qualifies every column by its table, as the plan's join names it;
    sent as Superset names its tables, each is aliased by its own name, so the join's columns resolve under a row rule
    on either table, as Superset applies the rule to every query of the dataset."""
    from mnemiq.superset.planner import _Catalogue, _joined

    columns = {"ORDERS": frozenset({"ID", "CUSTOMER_ID"}), "CUSTOMERS": frozenset({"ID", "NAME"})}
    catalogue = _Catalogue(columns=columns, stored={t: t for t in columns}, datasets={}, saved={},
                           types={t: {} for t in columns})
    made = _joined(PlannedDataset(key="dataset_1", placeholder=-1,
                                  join="ORDERS LEFT JOIN CUSTOMERS ON CUSTOMERS.ID = ORDERS.CUSTOMER_ID"), catalogue)
    assert not isinstance(made, str), made
    sent = for_superset(made.planned.sql)
    assert 'FROM orders "ORDERS" LEFT JOIN customers "CUSTOMERS" ON' in sent, sent
    assert _superset_tables(sent) == [("", "orders"), ("", "customers")]
    assert _unresolved(made.planned.sql) == [] and _unresolved(_row_ruled(sent)) == []


@pytest.mark.parametrize(("decided", "sent"), [
    # Its own alias, which the wrap keeps.
    ('SELECT "O"."ID" FROM "ORDERS" "O"', 'SELECT "O"."ID" FROM orders "O"'),
    # One table twice, each with its alias.
    ('SELECT "A"."ID", "B"."ID" FROM "ORDERS" "A" JOIN "ORDERS" "B" ON "A"."ID" = "B"."PARENT_ID"',
     'SELECT "A"."ID", "B"."ID" FROM orders "A" JOIN orders "B" ON "A"."ID" = "B"."PARENT_ID"'),
    # One table twice, one aliased: the other is the only source of its name, and takes it.
    ('SELECT "O"."ID" FROM "ORDERS" "O", "ORDERS"', 'SELECT "O"."ID" FROM orders "O", orders "ORDERS"'),
    # Two references one query exposes by one name already: mnemiq writes no alias two of them would share.
    ('SELECT COUNT(*) FROM "ORDERS", "ORDERS"', 'SELECT COUNT(*) FROM orders, orders'),
    ('SELECT COUNT(*) FROM "CUSTOMERS" "ORDERS", "ORDERS"', 'SELECT COUNT(*) FROM customers "ORDERS", orders'),
    # A table the rewrite leaves as it is.
    ('SELECT "ID" FROM "OrderItems"', 'SELECT "ID" FROM "OrderItems"'),
    ('SELECT orders.revenue FROM orders', 'SELECT orders.revenue FROM orders'),
    # An alias would not follow the name there, or would hide a column named by schema and table.
    ('SELECT * FROM "ORDERS" SAMPLE (10)', 'SELECT * FROM orders SAMPLE (10)'),
    ('SELECT * FROM "ORDERS" PARTITION (P1)', 'SELECT * FROM orders PARTITION (P1)'),
    ('SELECT "SHOP"."ORDERS"."ID" FROM "SHOP"."ORDERS"', 'SELECT "SHOP"."ORDERS"."ID" FROM shop.orders'),
])
def test_only_a_renamed_table_of_its_own_name_is_aliased(superset_oracle, decided, sent):
    assert for_superset(decided) == sent
    assert for_superset(sent) == sent


@pytest.mark.parametrize(("sql", "why"), [("SELECT * FROM ORDERS WHERE (", "does not parse"),
                                          ("", "no statement"),
                                          ("SELECT 1 FROM DUAL; SELECT 2 FROM ORDERS", "more than one")])
def test_a_statement_mnemiq_cannot_read_as_one_is_never_rewritten(superset_oracle, sql, why):
    """A statement whose tables mnemiq cannot find, or one of several, is refused rather than sent with a table in
    the database's spelling, which would keep a dataset's row rules from attaching to it."""
    with pytest.raises(UnreadableStatement, match=why):
        for_superset(sql)
    with pytest.raises(UnreadableStatement, match=why):
        from_superset(sql)


@pytest.mark.parametrize(("held", "decided"), [
    ("SELECT * FROM orders", 'SELECT * FROM "ORDERS"'),
    ("SELECT * FROM ORDERS", 'SELECT * FROM "ORDERS"'),
    ('SELECT * FROM "OrderItems"', 'SELECT * FROM "OrderItems"'),
    ('SELECT * FROM "ORDERS"', 'SELECT * FROM "ORDERS"'),
    ("SELECT o.id FROM shop.orders o", 'SELECT o.id FROM "SHOP"."ORDERS" o'),
    ("WITH t AS (SELECT * FROM orders) SELECT * FROM t", 'WITH t AS (SELECT * FROM "ORDERS") SELECT * FROM t'),
    ("SELECT * FROM order#", 'SELECT * FROM "ORDER#"'),
])
def test_from_superset_writes_each_table_as_the_database_reads_it(superset_oracle, held, decided):
    """SQL Superset holds (a virtual dataset's), with each table quoted as Oracle reads it, as mnemiq's deciders read a
    table only by the name the engine reads; sent, it names the tables Superset's access check reads as the statement
    Superset holds would, sent, name them."""
    assert from_superset(held) == decided
    assert _superset_tables(for_superset(decided)) == _superset_tables(for_superset(held))


# ---------------------------------------------------------------- the adapter

class _Recorder:
    """A Superset client answering every `execute_sql` with no rows, each request kept."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def call(self, tool: str, request: dict, *, timeout_s: float | None = None) -> dict:
        assert tool == "execute_sql"
        self.requests.append(dict(request))
        return {"success": True, "columns": [], "rows": []}


def test_the_adapter_sends_each_statement_in_supersets_spelling(superset_oracle):
    """Every statement through `execute_sql`, whichever read sends it: a result, an answer, the proof's probe and a dry
    run. The statement's row bound is read from it as written, and the same either way."""
    client = _Recorder()
    adapter = SupersetSqlAdapter(client, database_id=7, schema="shop")
    adapter.execute_arrow('SELECT "REVENUE" FROM "ORDERS"')
    adapter.execute_answer('SELECT "REVENUE" FROM "ORDERS" FETCH FIRST 5 ROWS ONLY')
    adapter.validate('SELECT "REVENUE" FROM "ORDERS"')
    adapter.dry_run('SELECT 1 FROM "ORDERS", "OrderItems" WHERE 1 = 0')
    assert [request["sql"] for request in client.requests] == [
        'SELECT "REVENUE" FROM orders "ORDERS"',
        'SELECT "REVENUE" FROM orders "ORDERS" FETCH FIRST 5 ROWS ONLY',
        'SELECT * FROM (SELECT "REVENUE" FROM orders "ORDERS") mnemiq_probe WHERE 1 = 0',
        'SELECT 1 FROM orders "ORDERS", "OrderItems" WHERE 1 = 0']
    assert {request["schema"] for request in client.requests} == {"shop"}


def test_a_statement_the_adapter_cannot_rewrite_is_not_sent(superset_oracle):
    client = _Recorder()
    with pytest.raises(SupersetQueryFailed, match="not sent"):
        SupersetSqlAdapter(client, database_id=7, schema="shop").execute_arrow("SELECT * FROM ORDERS WHERE (")
    assert client.requests == []


def test_on_postgres_the_adapter_sends_each_statement_as_written():
    client = _Recorder()
    sql = 'SELECT "REVENUE" FROM "ORDERS"'
    SupersetSqlAdapter(client, database_id=7, schema="public").execute_arrow(sql)
    assert client.requests[0]["sql"] is sql


# ---------------------------------------------------------------- the boundaries

ADAPTER = SimpleNamespace(database_id=7, schema="shop")
ORACLE_SNAPSHOT = Snapshot(version="v1", source_id="only", created_at="t", columns=[
    Column(id="ORDERS.REGION", object_id="ORDERS", name="REGION", data_type="VARCHAR2"),
    Column(id="ORDERS.CHANNEL", object_id="ORDERS", name="CHANNEL", data_type="VARCHAR2"),
    Column(id="ORDERS.PREMIUM", object_id="ORDERS", name="PREMIUM", data_type="NUMBER"),
])
ORACLE_GRANTS = GrantSet(frozenset({"ORDERS"}))


def _listed(dataset_id: int, table_name: str, *, schema: str = "shop", sql: str | None = None) -> ListedDataset:
    return ListedDataset(dataset_id, table_name, schema, 7, sql)


def _facts(**changes) -> DatasetFacts:
    fields = {"dataset_id": 5, "is_virtual": False, "table": "orders", "sql": None, "saved_metric_names": frozenset(),
              "database_id": 7, "schema": "shop"}
    return DatasetFacts(**(fields | changes))


def test_every_boundary_uses_the_mapping(superset_oracle):
    """The planner finds snapshot table ORDERS's dataset `orders`; grants ask Superset about the tables its datasets
    are read as; the builder names a table's dataset as Superset lists the table; and a chart names a column as its
    dataset does, which for a physical dataset rc2 reads is as the database stores it (`get_physical_table_metadata`
    denormalizes each name unless the dataset is set to normalize its columns, and the build's datasets are not), and
    for a joined one as the database returns its quoted aliases."""
    from mnemiq.superset.builder import _dataset_name
    from mnemiq.superset.grants import _candidates
    from mnemiq.superset.planner import _datasets_by_table

    listed = [_listed(5, "orders"), _listed(6, "OrderItems"), _listed(4, "orderitems"), _listed(3, "orders",
                                                                                             schema="sales")]
    stored = {"ORDERS": "ORDERS", "OrderItems": "OrderItems", "ORDER_ITEMS": "ORDER_ITEMS"}
    assert _datasets_by_table(listed, ADAPTER, stored) == {"ORDERS": 5, "OrderItems": 6}

    datasets = [_listed(5, "orders"), _listed(6, "OrderItems"), _listed(8, "order_items", sql="SELECT 1 FROM dual")]
    tables = {"ORDERS": {"ORDERS"}, "OrderItems": {"OrderItems"}, "ORDERITEMS": {"ORDERITEMS"},
              "ORDER_ITEMS": {"ORDER_ITEMS"}}
    assert _candidates(datasets, tables, 7, "shop") == ["ORDERS", "OrderItems"]

    build = SimpleNamespace(build_id=str(uuid.uuid4()))
    assert _dataset_name(build, PlannedDataset(key="dataset_1", placeholder=-1, table="ORDERS",
                                               table_name="ORDERS")) == "orders"
    assert _dataset_name(build, PlannedDataset(key="dataset_1", placeholder=-1, table="OrderItems",
                                               table_name="OrderItems")) == "OrderItems"

    premium = Measure(name="premium", label="Premium", expression='SUM("PREMIUM")', kind=MeasureKind.NUMBER,
                      dataset_id=5, description="", asked_by="")
    spec = ChartSpec(key="by_region", family=Family.PIE, title="By region", dataset_id=5,
                     measures=(WrittenRef(premium),), dimensions=("REGION",), time_column=None, time_grain=None,
                     filters=(Filter("CHANNEL", "IN", ["web"]),), row_limit=None, source=None, target=None)
    config = config_chart_request(spec, {"premium": premium}, {}, currency="USD")["config"]
    assert config["dimension"] == {"name": "REGION"}
    assert config["filters"] == [{"column": "CHANNEL", "op": "IN", "value": ["web"]}]


def test_the_builder_sends_a_joined_datasets_sql_in_supersets_spelling(superset_oracle, monkeypatch):
    """rc2 applies the row rules of a virtual dataset's tables to every query of it, matched by their names, so the
    SQL a joined dataset is made from names them as Superset does; the plan keeps the SQL mnemiq decided, which the
    check reads the dataset with (`made_virtual_sql`)."""
    import mnemiq.superset.builder as builder

    sent: list[dict] = []
    monkeypatch.setattr(builder, "_created", lambda run, tool, request, **more: sent.append(request) or 21)
    decided = 'SELECT "O"."ID" AS "ID" FROM "ORDERS" "O" LEFT JOIN "CUSTOMERS" "C" ON "C"."ID" = "O"."CUSTOMER_ID"'
    joined = PlannedDataset(key="dataset_1", placeholder=-1, join='"ORDERS" "O" LEFT JOIN "CUSTOMERS" "C" ON 1 = 1',
                            sql=decided, name="mnemiq_dataset_1:ORDERS_CUSTOMERS")
    table = PlannedDataset(key="dataset_2", placeholder=-2, table="ORDERS", table_name="ORDERS")
    plan = BuildPlan(charts=(), measures=(), datasets=(joined, table))
    build = SimpleNamespace(build_id=str(uuid.uuid4()), dashboard_id=3)
    for key in ("dataset_1", "dataset_2"):
        builder._run_dataset(SimpleNamespace(build=build, plan=plan, key=key, adapter=ADAPTER))
    assert sent[0]["sql"] == ('SELECT "O"."ID" AS "ID" FROM orders "O" LEFT JOIN customers "C" '
                              'ON "C"."ID" = "O"."CUSTOMER_ID"')
    assert sent[1]["table_name"] == "orders" and "sql" not in sent[1]
    assert (sent[0]["schema"], sent[1]["schema"]) == ("shop", "shop")


def test_the_planner_reads_a_dataset_as_superset_reads_it(superset_oracle):
    """A dataset Superset answers, as the FROM of a statement mnemiq decides: a physical one's table by the stored
    name Superset reads for it, quoted, and only one the builder's catalogue holds; a virtual one's SQL with each table
    as the database reads it (`from_superset`), and none when mnemiq cannot read it."""
    from mnemiq.superset.planner import _source_for_builder

    stored = {"ORDERS", "OrderItems"}
    assert _source_for_builder(_facts(), ADAPTER, stored) == '"ORDERS"'
    assert _source_for_builder(_facts(table="OrderItems"), ADAPTER, stored) == '"OrderItems"'
    assert _source_for_builder(_facts(table="orderitems"), ADAPTER, stored) is None
    assert _source_for_builder(_facts(table=None), ADAPTER, stored) is None
    assert _source_for_builder(_facts(is_virtual=True, table="joined", sql="SELECT * FROM orders"), ADAPTER,
                               stored) == '(SELECT * FROM "ORDERS") AS "virtual_table"'
    assert _source_for_builder(_facts(is_virtual=True, table="joined", sql="SELECT * FROM orders WHERE ("), ADAPTER,
                               stored) is None


def test_a_measures_dataset_is_read_as_superset_reads_it(superset_oracle):
    from mnemiq.superset.measure_actions import ActionRefused, _source

    row = SimpleNamespace(measure=SimpleNamespace(dataset_id=5), made_dataset_ids=(5,))
    assert _source(None, row, _facts()) == '"ORDERS"'
    assert _source(None, row, _facts(is_virtual=True, sql="SELECT * FROM orders")) == \
        '(SELECT * FROM "ORDERS") AS "virtual_table"'
    with pytest.raises(ActionRefused, match="could not read"):
        _source(None, row, _facts(is_virtual=True, sql="SELECT * FROM orders WHERE ("))


def test_a_charts_table_offers_the_columns_of_the_table_superset_reads(superset_oracle):
    from mnemiq.superset.chart_change import _table_columns

    asking = SimpleNamespace(snapshot=ORACLE_SNAPSHOT, grants=ORACLE_GRANTS)
    assert _table_columns(asking, _facts()) == ["CHANNEL", "PREMIUM", "REGION"]
    assert _table_columns(asking, _facts(table="ORDERS")) == ["CHANNEL", "PREMIUM", "REGION"]
    assert _table_columns(asking, _facts(table="orderitems")) == []


# ---------------------------------------------------------------- the value check, on Oracle

def _oracle_chart(columns: dict[str, str], *, virtual_sql: str | None = None) -> dict:
    """A bar chart of the premium by region on the web channel, on a dataset `orders` whose columns Superset names as
    `columns` gives each stored column's name: as stored (rc2's default), or normalized."""
    region, channel, premium = columns["REGION"], columns["CHANNEL"], columns["PREMIUM"]
    return {
        "form": {"viz_type": "echarts_timeseries_bar", "x_axis": region, "groupby": [],
                 "metrics": [{"expressionType": "SIMPLE", "aggregate": "SUM", "label": "Total premium",
                              "column": {"column_name": premium}, "hasCustomLabel": True}],
                 "adhoc_filters": [{"clause": "WHERE", "expressionType": "SIMPLE", "subject": channel,
                                    "operator": "IN", "comparator": ["web"]}],
                 "row_limit": 100, "y_axis_format": ",.2f", "x_axis_sort_series_type": "name"},
        "dataset": {"id": 12, "table_name": "orders", "schema": "shop", "database_id": 7,
                    "is_virtual": virtual_sql is not None, "sql": virtual_sql, "metrics": [],
                    "columns": [{"column_name": name, "type": "VARCHAR2", "is_dttm": False}
                                for name in columns.values()]},
        "region": region,
    }


@pytest.fixture
def fake():
    from tests.superset_fake import FakeSuperset

    with FakeSuperset() as served:
        yield served


def _checked(fake, monkeypatch, chart: dict, *, made: dict | None = None):
    """The chart checked as the builder on Oracle: what mnemiq decided, what it sent, and the check."""
    from mnemiq.superset import chart_check
    from mnemiq.superset.client import DelegatedToken, SupersetMcp
    from tests.test_superset_chart_check import _PROBE, _chart_data, _info, _no_rule, _result, _rules

    rows = (("east", 1200), ("west", 800))
    fake.reply("get_chart_info", lambda request, authorization: _info(chart["form"]))
    fake.reply("get_dataset_info", lambda request, authorization: chart["dataset"])
    fake.reply("get_chart_data", lambda request, authorization: _chart_data([chart["region"], "Total premium"],
                                                                           *rows))
    # Oracle names an unquoted alias in upper case (`AS m0` comes back M0): mnemiq reads its rows by position.
    answer = _result(*rows, columns=(("REGION", "object"), ("M0", "int64")))
    fake.reply("execute_sql", lambda request, authorization: {"success": True, "columns": [], "rows": []}
               if request.get("force_refresh") or _PROBE.search(request["sql"]) else answer)
    _rules(fake, _no_rule)
    decided: list[str] = []
    real = chart_check.run_decided
    monkeypatch.setattr(chart_check, "run_decided", lambda sql, **more: decided.append(sql) or real(sql, **more))
    adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok-builder")), database_id=7, schema="shop")
    check = chart_check.check_chart(adapter, chart_id=41, dataset_id=12,
                                    kinds={"Total premium": MeasureKind.NUMBER}, saved={}, snapshot=ORACLE_SNAPSHOT,
                                    grants=ORACLE_GRANTS, guard_fanout=False, made=made or {})
    sent = [call.request["sql"] for call in fake.calls if call.tool == "execute_sql"
            and not call.request.get("force_refresh") and not _PROBE.search(call.request["sql"])]
    return decided, sent, check


@pytest.mark.parametrize("columns", [{"REGION": "REGION", "CHANNEL": "CHANNEL", "PREMIUM": "PREMIUM"},
                                     {"REGION": "region", "CHANNEL": "channel", "PREMIUM": "premium"}],
                         ids=["as stored", "normalized"])
def test_the_value_check_names_what_superset_reads(fake, monkeypatch, superset_oracle, columns):
    """The chart's dataset `orders`, its x-axis, its filter's column and its metric's column, each as the stored name
    Superset reads for it, in the statement mnemiq decides: a dataset's columns as stored (rc2's default) or normalized
    read the same. The statement sent names the table as Superset does, and the chart is checked."""
    from mnemiq.superset.chart_check import ChartStatus

    decided, sent, check = _checked(fake, monkeypatch, _oracle_chart(columns))
    assert check.status is ChartStatus.CHECKED, check.reason
    [mine] = decided
    assert 'FROM "ORDERS"' in mine and 'SUM("PREMIUM")' in mine and '"CHANNEL" IN (' in mine, mine
    assert mine.startswith('SELECT "REGION", '), mine
    [statement] = sent
    assert 'FROM orders "ORDERS"' in statement, statement


def test_a_made_dataset_is_checked_with_the_sql_the_build_decided(fake, monkeypatch, superset_oracle):
    """Superset holds a joined dataset the build made in Superset's spelling, as the build sent it: that text is the
    one the check holds Superset's to, and the SQL the build decided, in the database's spelling, is the one it reads
    the dataset with. A dataset still holding the decided text was not made as this build sends one: not checked."""
    from mnemiq.superset.chart_check import ChartStatus
    from tests.test_superset_chart_check import SQL_CHANGED

    made = 'SELECT "REGION", "CHANNEL", "PREMIUM" FROM "ORDERS"'
    stored = {"REGION": "REGION", "CHANNEL": "CHANNEL", "PREMIUM": "PREMIUM"}
    decided, sent, check = _checked(fake, monkeypatch, _oracle_chart(stored, virtual_sql=for_superset(made)),
                                    made={12: made})
    assert check.status is ChartStatus.CHECKED, check.reason
    [mine] = decided
    assert f"FROM ({made}) virtual_table" in mine, mine
    [statement] = sent
    assert f"FROM ({for_superset(made)}) virtual_table" in statement, statement
    fake.calls.clear()
    _, _, check = _checked(fake, monkeypatch, _oracle_chart(stored, virtual_sql=made), made={12: made})
    assert check.status is ChartStatus.NOT_CHECKED and check.reason == SQL_CHANGED
