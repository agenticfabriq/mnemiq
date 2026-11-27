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

import re
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
from mnemiq.superset.names import (
    RefusedStatement,
    UnreadableStatement,
    for_superset,
    from_superset,
    stored_name,
    superset_name,
)
from mnemiq.superset.payload import SupersetQueryFailed
from mnemiq.superset.planner import BuildPlan, PlannedDataset
from mnemiq.superset.superset_reads import superset_reads
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
    assert for_superset('SELECT "ID" FROM "OrderItems"') == 'SELECT "ID" FROM "OrderItems" "OrderItems"'
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
    """Returned as it came, not printed again, whatever it holds, and nothing refused: PostgreSQL's statements are
    byte-identical."""
    for sql in ('SELECT "REVENUE" FROM "ORDERS"', "SELECT * FROM ORDERS", "SELECT * FROM ORDERS WHERE (", "",
                'SELECT COUNT(*) FROM "ORDERS" SAMPLE (10), (SELECT 1 AS "X" FROM "DUAL") "orders"',
                'WITH "orders" AS (SELECT 1 AS "ID" FROM "DUAL") SELECT "ORDERS"."ID" FROM "ORDERS"',
                "SELECT 1 FROM orders@lnk", "SELECT q'[x]' FROM ORDERS", "SELECT SHOP.ORDERS.ID FROM SHOP.ORDERS",
                "SELECT 1 FROM (ORDERS JOIN CUSTOMERS ON 1 = 1)"):
        assert for_superset(sql) is sql and from_superset(sql) is sql


# ---------------------------------------------------------------- statements

def test_for_superset_writes_supersets_spelling(superset_oracle):
    """Each table the statement reads, as Superset names its dataset, and each column as written: Superset matches no
    column by name, and an unquoted `sysdate` would read the function, not a column stored as SYSDATE. A table renamed
    so is aliased by its own name (`test_an_unaliased_table_keeps_its_name_under_a_row_rule`)."""
    assert for_superset('SELECT "REVENUE" FROM "ORDERS"') == 'SELECT "REVENUE" FROM orders "ORDERS"'
    assert for_superset('SELECT "SYSDATE", "ROWNUM" FROM "ORDERS"') == 'SELECT "SYSDATE", "ROWNUM" FROM orders "ORDERS"'


@pytest.mark.parametrize(("written", "sent"), [
    ('SELECT "ID" FROM ORDERS', 'SELECT "ID" FROM orders "ORDERS"'),
    ('SELECT "ID" FROM Orders', 'SELECT "ID" FROM orders "ORDERS"'),
    ('SELECT "ID" FROM orders', 'SELECT "ID" FROM orders "ORDERS"'),
    ('SELECT "ID" FROM "ORDER_ITEMS"', 'SELECT "ID" FROM order_items "ORDER_ITEMS"'),
    ('SELECT "ID" FROM OrderItems', 'SELECT "ID" FROM orderitems "ORDERITEMS"'),
    ('SELECT "ID" FROM "orders"', 'SELECT "ID" FROM "orders" "orders"'),
    ('SELECT "ID" FROM "ORDER"', 'SELECT "ID" FROM "ORDER" "ORDER"'),
    ('SELECT "ID" FROM "ORDER#"', 'SELECT "ID" FROM "ORDER#" "ORDER#"'),
    ('SELECT "ID" FROM order#', 'SELECT "ID" FROM "ORDER#" "ORDER#"'),
    ('SELECT "ID" FROM "A$B"', 'SELECT "ID" FROM a$b "A$B"'),
    ('SELECT "ID" FROM "SHOP"."ORDERS"', 'SELECT "ID" FROM shop.orders "ORDERS"'),
])
def test_each_table_is_written_as_superset_names_it(superset_oracle, written, sent):
    assert for_superset(written) == sent


@pytest.mark.parametrize("written", [
    # SQLAlchemy shows the Kelvin sign's table as `k`, which Oracle reads as another table, K: it can be sent only
    # quoted, by a name Superset does not give it.
    'SELECT "ID" FROM "\u212a"',
    # A letter whose fold on Oracle mnemiq has not measured: the table's stored name, so Superset's, is not known.
    'SELECT "ID" FROM \u1f80x',
    'SELECT "ID" FROM \u1f80x.orders',
    # More parts than a catalog, a schema and a name.
    'SELECT "ID" FROM shop.orders.x.y',
])
def test_a_table_not_sent_by_the_name_superset_gives_it_is_refused(superset_oracle, written):
    """A statement is sent only when Superset reads each table in it by the name Superset gives the table: one that
    can be written only by another name, or whose name mnemiq cannot tell, is refused, nothing sent."""
    with pytest.raises(RefusedStatement, match="could not confirm"):
        for_superset(written)


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
    iota, `resolve_exactly`) names a table mnemiq cannot spell another way and be sure it is the same one: Superset's
    SQL is read with it as written, and a statement naming it is not sent
    (`test_a_table_not_sent_by_the_name_superset_gives_it_is_refused`)."""
    sql = 'SELECT "ID" FROM ᾀx'
    assert from_superset(sql) == sql
    with pytest.raises(RefusedStatement):
        for_superset(sql)


def test_a_rewrite_that_does_not_read_back_is_refused(superset_oracle):
    """The text is rewritten where each table's name stands, then read again: a rewrite that reads as another
    statement (here a name written bare that reads as a table and an alias) is refused, never sent."""
    with pytest.raises(UnreadableStatement, match="does not read as"):
        names._rewritten('SELECT 1 FROM "ORDERS"', lambda stored: names.exp.Identifier(this="x y", quoted=False),
                         aliased=False, refusing=False)


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
    aliased by the table's own alias where it has one, else by its name quoted as written there, the table moved in
    with all sqlglot hangs on it (a parenthesized join's first table takes its joins along, as rc2's does), the
    deepest table first, as rc2 wraps them (`apply_rls` sorts its reads by depth), so a table inside another's joins is
    wrapped before that table carries it in. A copy of rc2's rule, since mnemiq does not install Superset; checked
    against rc2's own in the task's live notes. It models only the subquery wrap, over the reads mnemiq's resolver
    finds; which reads Superset finds is `superset_reads`'s."""
    tree = sqlglot.parse_one(sql, read="oracle")
    for table in sorted(base_tables(tree, "oracle"), key=lambda node: node.depth, reverse=True):
        alias = table.args.get("alias") or exp.TableAlias(this=exp.Identifier(this=table.name, quoted=True))
        bare = table.copy()
        bare.set("alias", None)
        table.replace(exp.Subquery(this=exp.select("*").from_(bare).where("1 = 1"), alias=alias.copy()))
    return tree.sql(dialect="oracle")


def _unresolved(sql: str) -> list[str]:
    """Each column qualifier in `sql` that names no source it can see, as Oracle resolves names: a source of its own
    query or of one enclosing it (a table, at any depth of parenthesized joins, or a derived table), by its alias, else
    by its table's name."""
    tree = sqlglot.parse_one(sql, read="oracle")
    unresolved = []
    for column in tree.find_all(exp.Column):
        qualifier = column.args.get("table")
        if qualifier is None:
            continue
        wanted, select, found = resolve_identifier(qualifier, "oracle"), column.parent_select, False
        while select is not None and not found:
            exposed = {resolve_identifier(node.args["alias"].this if node.args.get("alias") else node.this, "oracle")
                       for node in select.find_all(exp.Table, exp.Subquery) if node.parent_select is select
                       and (isinstance(node, exp.Table) or node.args.get("alias") is not None)}
            found = wanted in exposed
            select = select.parent_select
        if not found:
            unresolved.append(column.sql(dialect="oracle"))
    return unresolved


# A bare name followed by the alias the rewrite gives it, its stored name quoted.
_OWN_ALIAS = re.compile(r'\b([a-z_$#][a-z0-9_$#]*) "([A-Z_$#][A-Z0-9_$#]*)"')


@pytest.mark.parametrize(("decided", "read"), [
    ('SELECT "ORDERS"."REVENUE" FROM "ORDERS"', [("", "orders")]),
    ("SELECT ORDERS.REVENUE FROM ORDERS", [("", "orders")]),
    ("SELECT orders.revenue FROM orders", [("", "orders")]),
    ('SELECT "ORDERS"."ID" FROM "ORDERS" WHERE EXISTS (SELECT 1 FROM "RETURNS" '
     'WHERE "RETURNS"."ORDER_ID" = "ORDERS"."ID")', [("", "orders"), ("", "returns")]),
    ('SELECT "ORDERS"."ID", "RETURNS"."ID" FROM "ORDERS" JOIN "CUSTOMERS" ON "CUSTOMERS"."ID" = "ORDERS"."CID" '
     'JOIN "RETURNS" ON "RETURNS"."ORDER_ID" = "ORDERS"."ID"', [("", "customers"), ("", "orders"), ("", "returns")]),
    ("SELECT ORDERS.ID FROM (ORDERS)", [("", "orders")]),
])
def test_an_unaliased_table_keeps_its_name_under_a_row_rule(superset_oracle, decided, read):
    """rc2 wraps a table a row rule applies to as a subquery aliased by the table's name quoted as written, `"orders"`
    for a bare `orders`, which Oracle reads as another name than ORDERS: a column the statement qualifies by its table
    would name nothing. So every table with no alias is aliased by its stored name, quoted, whether the rewrite renamed
    it or Superset's spelling was already written, and in a parenthesized join as anywhere: the wrap keeps it, every
    qualifier resolves before and after the wrap, and the access check still reads `orders`. Unaliased, it would
    not."""
    sent = for_superset(decided)
    assert sorted(_superset_tables(sent)) == read == sorted(_rc2_reads(sent))
    assert _unresolved(sent) == [] and _unresolved(_row_ruled(sent)) == []
    unaliased = _OWN_ALIAS.sub(lambda m: m[1] if m[1].upper() == m[2] else m[0], sent)
    assert unaliased != sent
    assert _unresolved(unaliased) == [] and _unresolved(_row_ruled(unaliased)) != []


def _rc2_reads(sql: str) -> list[tuple[str, str]]:
    """The tables Superset reads in `sql`, each (schema, table) as Superset names it: `superset_reads`, the copy of
    rc2's own table extraction that `for_superset` holds every statement it sends to."""
    return [(table.db, table.name) for table in superset_reads(sqlglot.parse_one(sql, read="oracle"))]


# A statement whose CTE has the name of a table it reads, in any case and quoting, in any scope of the statement: as
# the table is written, as it is stored, or as Superset names it. mnemiq refuses each.
CTE_TWINS = [
    'WITH "orders" AS (SELECT 1 AS "ID" FROM "DUAL") SELECT "ORDERS"."ID" FROM "ORDERS"',
    'WITH "Orders" AS (SELECT 1 AS "ID" FROM "DUAL") SELECT "ORDERS"."ID" FROM "ORDERS"',
    'WITH orders AS (SELECT "ID" FROM "ORDERS") SELECT "ID" FROM orders',
    'WITH ORDERS AS (SELECT "ID" FROM "ORDERS") SELECT "ID" FROM ORDERS',
    'SELECT * FROM (WITH "orders" AS (SELECT 1 AS "ID" FROM "DUAL") SELECT "ID" FROM "orders") "D", "ORDERS"',
    'WITH "orders" AS (SELECT 1 AS "ID" FROM "DUAL") SELECT "X"."ID" FROM "X" '
    'WHERE "X"."ID" IN (SELECT "ID" FROM "ORDERS")',
    # Apart from the CTE's in any case as written, but not as stored: a bare dotless i is stored, and sent, as I.
    'WITH "id" AS (SELECT 1 AS "X" FROM "DUAL") SELECT "X" FROM \u0131D',
    # A mixed-case table, which Superset names as stored, and a CTE of its name in another case.
    'WITH "orderitems" AS (SELECT 1 AS "ID" FROM "DUAL") SELECT "ID" FROM "OrderItems"',
]


@pytest.mark.parametrize("decided", CTE_TWINS)
def test_a_cte_that_shares_a_tables_name_is_refused(superset_oracle, decided):
    """A CTE whose name, in any case and quoting, is the name of a table the statement reads (as written, as stored or
    as Superset names it) is refused, nothing sent, and the model is told how to repair it: mnemiq sends a CTE only
    under a name no table of the statement has."""
    with pytest.raises(RefusedStatement, match="give the CTE a name no table of the statement has"):
        for_superset(decided)


@pytest.mark.parametrize(("decided", "said"), [
    ('WITH W AS (SELECT * FROM "CLAIM") SELECT w."SECRET" FROM w', "the CTE W as w"),
    ('WITH "W" AS (SELECT * FROM "CLAIM") SELECT w."SECRET" FROM w', "the CTE W as w"),
    ('WITH w AS (SELECT * FROM "CLAIM") SELECT "ID" FROM "CLAIM" WHERE "ID" IN (SELECT "ID" FROM W)', "the CTE w as W"),
    ('WITH W AS (SELECT * FROM "CLAIM") SELECT "SECRET" FROM (SELECT w.* FROM w) "D"', "the CTE W as w"),
])
def test_a_cte_named_otherwise_than_its_with_clause_is_refused(superset_oracle, decided, said):
    """A CTE named, where the statement reads it, otherwise than its WITH clause writes its name (quotes aside) is
    refused, nothing sent, though Oracle reads both as one name; the model is told to name it as the WITH clause does.
    Named so, it is sent."""
    with pytest.raises(RefusedStatement, match=f"names {re.escape(said)}.+name each CTE as its WITH clause does"):
        for_superset(decided)
    assert for_superset('WITH "W" AS (SELECT * FROM "CLAIM") SELECT W."SECRET" FROM W').endswith("FROM W")


def test_a_cte_that_reads_as_the_cte_is_no_twin(superset_oracle):
    """A reference Oracle reads as the CTE reads no table of that name: it is no twin, and is sent, Superset reading
    the same tables in it as mnemiq does."""
    sent = for_superset('WITH orders AS (SELECT 1 AS "ID" FROM "DUAL") SELECT "ID" FROM orders')
    assert sent == 'WITH orders AS (SELECT 1 AS "ID" FROM dual "DUAL") SELECT "ID" FROM orders'
    assert _superset_tables(sent) == _rc2_reads(sent) == [("", "dual")]


# Statements this module's tests send, written or decided: each is sent only when Superset reads the tables in it that
# mnemiq reads.
SENT_HERE = [decided for decided, _ in NOT_TABLES] + CTE_TWINS + [
    'SELECT COUNT(*) FROM "ORDERS" "X", "CUSTOMERS" "X"', 'SELECT COUNT(*) FROM "ORDERS", "ORDERS"',
    'WITH t AS (SELECT * FROM "ORDERS") SELECT * FROM t',
    'SELECT * FROM "SHOP"."ORDERS" "O" WHERE "O"."ID" IN (SELECT "ORDER_ID" FROM "RETURNS")',
]


@pytest.mark.parametrize("decided", SENT_HERE)
def test_superset_finds_the_tables_mnemiq_reads(superset_oracle, decided):
    """A statement sent is one in which Superset reads (`superset_reads`) the tables mnemiq's resolver finds it reads,
    name for name as Superset names them; the rest are refused."""
    try:
        sent = for_superset(decided)
    except RefusedStatement:
        assert decided in CTE_TWINS
        return
    assert sorted(_rc2_reads(sent)) == sorted(_superset_tables(sent))


# Statements in which mnemiq cannot confirm that Superset reads every table mnemiq reads, each by the name Superset
# gives it. mnemiq refuses each.
NOT_READ_ALIKE = [
    'SELECT COUNT(*) FROM "ORDERS" "O", (SELECT 1 AS "X" FROM "DUAL") "O"',
    'WITH "T" AS (SELECT 1 AS "X" FROM "DUAL") SELECT COUNT(*) FROM "ORDERS" "O", "T" "O"',
    'SELECT COUNT(*) FROM "ORDERS" "O" CROSS JOIN (SELECT COUNT(*) AS "N" FROM "DUAL") "O"',
    'SELECT COUNT(*) FROM "ORDERS" PARTITION (P1), (SELECT 1 AS "X" FROM "DUAL") "orders"',
    'SELECT COUNT(*) FROM "ORDERS" "O", (SELECT 1 AS "X" FROM "ORDERS" "P") "O"',
    # A sample whose table cannot take its alias, another source going by that name.
    'SELECT COUNT(*) FROM "ORDERS" SAMPLE (10), "CUSTOMERS" "ORDERS", (SELECT 1 AS "X" FROM "DUAL") "orders"',
    'SELECT COUNT(*) FROM "ORDERS" SAMPLE (10), "ORDERS"',
    # A flashback clause, bare and aliased, and an @ in its literal, which is no link.
    'SELECT "ID" FROM "ORDERS" AS OF SCN 1',
    'SELECT "O"."ID" FROM "ORDERS" AS OF TIMESTAMP SYSTIMESTAMP "O"',
    "SELECT \"ID\" FROM \"ORDERS\" AS OF TIMESTAMP TO_TIMESTAMP('2026-01-01 a@b', 'YYYY-MM-DD')",
    # A clause on a CTE's reference, as on a table.
    'WITH "W" AS (SELECT 1 AS "X" FROM "DUAL") SELECT COUNT(*) FROM "ORDERS" "PARTITION", "W" PARTITION (P1)',
    'WITH "W" AS (SELECT 1 AS "X" FROM "DUAL") SELECT COUNT(*) FROM "ORDERS" "O", "W" AS OF SCN 1',
    # A pivoted or unpivoted reference, to a table or to a CTE.
    'SELECT * FROM "ORDERS" PIVOT (SUM("AMOUNT") FOR "STATUS" IN (\'A\' AS "A"))',
    'WITH "W" AS (SELECT 1 AS "X", 2 AS "Y" FROM "DUAL") SELECT COUNT(*) FROM "W" PIVOT (SUM("X") FOR "Y" IN '
    '(1 AS "A")) "P", "ORDERS" "W", (SELECT 1 AS "Z" FROM "DUAL") "W"',
    'WITH "W" AS (SELECT 1 AS "X", 2 AS "Y" FROM "DUAL") SELECT COUNT(*) FROM "W" UNPIVOT ("V" FOR "K" IN ("X", '
    '"Y")) "P", "ORDERS" "W", (SELECT 1 AS "Z" FROM "DUAL") "W"',
    'SELECT COUNT(*) FROM "ORDERS" PARTITION (P1), (SELECT 1 AS "X" FROM "DUAL") "PARTITION"',
    'WITH "PARTITION" AS (SELECT 1 AS "X" FROM "DUAL") SELECT COUNT(*) FROM "ORDERS" PARTITION (P1), "PARTITION"',
    'SELECT COUNT(*) FROM "ORDERS" SUBPARTITION (P1) CROSS JOIN (SELECT 1 AS "X" FROM "DUAL") "SUBPARTITION"',
    'SELECT * FROM "ORDERS" PARTITION (P1)',
    "(SELECT count(*) FROM (PIVOT orders ON status USING sum(id)))",
    "(SELECT max(to_json(p)) FROM (PIVOT orders ON status USING sum(id)) p)",
    "SELECT * FROM (PIVOT film ON title USING count(film_id)) p",
    "SELECT * FROM (PIVOT query_table('customer') ON c USING sum(x)) p",
    "SELECT id FROM (PIVOT claim ON id USING first(secret)) p",
    "SELECT k FROM (UNPIVOT claim ON secret INTO NAME k VALUE v) u",
    "SELECT * FROM LATERAL (VALUES ((SELECT max(store_id) FROM customer))) AS lv(x)",
    "SELECT * FROM film f, LATERAL (VALUES ((SELECT max(store_id) FROM customer))) AS v(x)",
    "SELECT 1 AS one FROM LATERAL (VALUES ((SELECT max(amount) FROM claim))) AS lv(x)",
    "SELECT id, lv.x FROM claim CROSS JOIN LATERAL (VALUES ((SELECT max(ssn) FROM customer))) AS lv(x)",
    "SELECT x FROM LATERAL (VALUES ((SELECT max(ssn) FROM customer))) AS lv(x)",
    "SELECT x FROM LATERAL (VALUES ((SELECT max(store_id) FROM customer))) AS lv(x)",
    "SELECT 1 FROM claim c, (SELECT 1) c",
    "SELECT COLUMN_VALUE FROM TABLE(SYS.ODCINUMBERLIST(1, 2))",
    "SELECT j.a FROM JSON_TABLE('[]', '$' COLUMNS (a NUMBER PATH '$.a')) j",
    "SELECT id INTO flags FROM kw",
]


@pytest.mark.parametrize("decided", NOT_READ_ALIKE)
def test_a_statement_is_sent_only_when_superset_reads_the_tables_mnemiq_reads(superset_oracle, decided):
    """mnemiq sends a statement only when Superset reads (`superset_reads`) each table mnemiq reads in it, by the name
    Superset gives the table. In each of these it cannot confirm that for a table: refused, nothing sent, and the
    model told how to repair it."""
    with pytest.raises(RefusedStatement, match="could not confirm for .+ so it was not sent; read each table in a "
                                               "plain FROM or JOIN, under an alias of its own"):
        for_superset(decided)


@pytest.mark.parametrize(("decided", "named"), [(NOT_READ_ALIKE[0], '"ORDERS"'), (NOT_READ_ALIKE[4], '"ORDERS"'),
                                                (NOT_READ_ALIKE[-1], "flags")])
def test_the_refusal_names_each_table_as_written(superset_oracle, decided, named):
    with pytest.raises(RefusedStatement) as refused:
        for_superset(decided)
    assert f"could not confirm for {named}, so" in str(refused.value)


@pytest.mark.parametrize(("decided", "sent"), [
    ('SELECT ORDERS.ID FROM ORDERS SAMPLE (10)', 'SELECT ORDERS.ID FROM orders SAMPLE (10) "ORDERS"'),
    ('SELECT "ORDERS"."ID" FROM "ORDERS" sample(10) WHERE "ORDERS"."ID" > 0',
     'SELECT "ORDERS"."ID" FROM orders sample(10) "ORDERS" WHERE "ORDERS"."ID" > 0'),
    ('SELECT "ORDERS"."ID" FROM "ORDERS" SAMPLE BLOCK (10) SEED (3)',
     'SELECT "ORDERS"."ID" FROM orders SAMPLE BLOCK (10) SEED (3) "ORDERS"'),
    ('SELECT "ORDERS"."ID" FROM "ORDERS" SAMPLE (.5) WHERE "ORDERS"."ID" IN (SELECT "ID" FROM "RETURNS")',
     'SELECT "ORDERS"."ID" FROM orders SAMPLE (.5) "ORDERS" WHERE "ORDERS"."ID" IN '
     '(SELECT "ID" FROM returns "RETURNS")'),
    ('SELECT COUNT(*) FROM "ORDERS" SAMPLE (10), (SELECT 1 AS "X" FROM "DUAL") "orders"',
     'SELECT COUNT(*) FROM orders SAMPLE (10) "ORDERS", (SELECT 1 AS "X" FROM dual "DUAL") "orders"'),
    ('WITH "T" AS (SELECT 1 AS "X" FROM "DUAL") SELECT COUNT(*) FROM "ORDERS" SAMPLE (10), "T" "orders"',
     'WITH "T" AS (SELECT 1 AS "X" FROM dual "DUAL") SELECT COUNT(*) FROM orders SAMPLE (10) "ORDERS", "T" "orders"'),
    ('SELECT COUNT(*) FROM "ORDERS" SAMPLE (10) CROSS JOIN (SELECT COUNT(*) AS "N" FROM "DUAL") "orders"',
     'SELECT COUNT(*) FROM orders SAMPLE (10) "ORDERS" CROSS JOIN (SELECT COUNT(*) AS "N" FROM dual "DUAL") "orders"'),
    ('SELECT COUNT(*) FROM "ORDERS" SAMPLE (10), (SELECT 1 AS "X" FROM "ORDERS" "O") "orders"',
     'SELECT COUNT(*) FROM orders SAMPLE (10) "ORDERS", (SELECT 1 AS "X" FROM orders "O") "orders"'),
])
def test_a_sampled_table_is_aliased_after_its_sample(superset_oracle, decided, sent):
    """Oracle writes a table's alias after its sample clause, so a table with a sample takes its stored name as alias
    there, as any other table takes it after its name. Superset reads every table in it, and under a row rule's wrap
    (rc2's, as `_row_ruled` copies it) a column qualified by the table keeps resolving; with no alias it would not."""
    assert for_superset(decided) == sent and for_superset(sent) == sent
    assert sorted(_rc2_reads(sent)) == sorted(_superset_tables(sent))
    assert _unresolved(sent) == [] and _unresolved(_row_ruled(sent)) == []
    unaliased = sent.replace(') "ORDERS"', ")", 1)
    if '"ORDERS".' in sent or "ORDERS." in sent:
        assert _unresolved(unaliased) == [] and _unresolved(_row_ruled(unaliased)) != []


def test_a_statement_repaired_as_the_refusal_asks_is_sent(superset_oracle):
    """Each table under an alias of its own, the one with a sample too: Superset reads every table in it."""
    sent = for_superset('SELECT COUNT(*) FROM "ORDERS" SAMPLE (10) "O", (SELECT 1 AS "X" FROM "DUAL") "orders"')
    assert sent == 'SELECT COUNT(*) FROM orders SAMPLE (10) "O", (SELECT 1 AS "X" FROM dual "DUAL") "orders"'
    assert sorted(_rc2_reads(sent)) == sorted(_superset_tables(sent)) == [("", "dual"), ("", "orders")]


def test_a_statement_whose_reading_by_superset_mnemiq_cannot_find_is_not_sent(superset_oracle, monkeypatch):
    from sqlglot.errors import OptimizeError

    def unreadable(statement):
        raise OptimizeError("no scope")

    monkeypatch.setattr(names, "superset_reads", unreadable)
    with pytest.raises(UnreadableStatement, match="which tables Superset reads"):
        for_superset('SELECT 1 FROM "ORDERS"')


@pytest.mark.parametrize(("sql", "read"), [
    ("SELECT 1 FROM orders o JOIN shop.customers c ON 1 = 1", [("", "orders"), ("shop", "customers")]),
    ("WITH t AS (SELECT * FROM orders) SELECT * FROM t", [("", "orders")]),
    ("WITH orders AS (SELECT 1 AS x FROM dual) SELECT * FROM shop.orders", [("", "dual"), ("shop", "orders")]),
    ("SELECT 1 FROM orders WHERE id IN (SELECT id FROM returns)", [("", "orders"), ("", "returns")]),
    ("SELECT 1 FROM orders UNION ALL SELECT 1 FROM returns", [("", "orders"), ("", "returns")]),
    ("SELECT n FROM (SELECT COUNT(*) AS n FROM orders) d", [("", "orders")]),
    ("WITH t AS (SELECT 1 AS x FROM dual) SELECT * FROM t x, orders t", [("", "dual"), ("", "orders")]),
    ("WITH t AS (SELECT 1 AS x, 2 AS y FROM dual) SELECT * FROM t PIVOT (SUM(x) FOR y IN (1 AS a))", [("", "dual")]),
    ("WITH t AS (SELECT 1 AS x, 2 AS y FROM dual) SELECT * FROM t PIVOT (SUM(x) FOR y IN (1 AS a)) p, orders t",
     [("", "dual"), ("", "orders")]),
    ("WITH t AS (SELECT 1 AS x, 2 AS y FROM dual) SELECT * FROM t PIVOT (SUM(x) FOR y IN (1 AS a)) p, "
     "(SELECT 1 AS z FROM dual) t", [("", "dual"), ("", "dual")]),
    ("SELECT COUNT(*) FROM orders, orders", [("", "orders"), ("", "orders")]),
])
def test_superset_reads_each_table_source_of_each_scope(sql, read):
    """Each table among the sources of each of the statement's scopes, each reference once, a CTE's reference not
    among them and a table named with its schema always among them: as rc2 7.0.0rc2's own extraction reads these."""
    assert sorted(_rc2_reads(sql)) == sorted(read)


@pytest.mark.parametrize("decided", [
    "SELECT 1 FROM orders@lnk", 'SELECT 1 FROM "ORDERS"@lnk', 'SELECT 1 FROM "ORDERS"@"LNK"',
    "SELECT 1 FROM orders @ lnk",
    "SELECT 1 FROM orders@ lnk", "SELECT 1 FROM orders o@lnk", 'SELECT 1 FROM "SHOP"."ORDERS"@lnk',
    "SELECT 1 FROM shop.orders@lnk.example.com", 'SELECT 1 FROM "A@B"', "SELECT 1 FROM @lnk",
    'SELECT 1 FROM "ORDERS" JOIN returns@lnk ON 1 = 1', 'SELECT 1 FROM ("ORDERS" JOIN "RETURNS"@lnk ON 1 = 1)',
    'SELECT 1 FROM (orders JOIN returns @ lnk ON 1 = 1)',
    'SELECT 1 FROM "ORDERS" WHERE "ID" IN (SELECT "ID" FROM returns@lnk)'])
def test_a_table_over_a_database_link_is_refused(superset_oracle, decided):
    """A table read over a database link (an `@` in a part of its name, quoted or not, or a link sqlglot reads as its
    alias) is another database's, which no Superset dataset names: refused, nothing sent."""
    with pytest.raises(RefusedStatement, match="reads .+ over a database link"):
        for_superset(decided)


@pytest.mark.parametrize(("decided", "named"), [
    ('SELECT seq@lnk.nextval FROM "DUAL"', "seq@lnk"),
    ('SELECT shop.seq@lnk.currval + 1 AS "N" FROM "DUAL"', "seq@lnk"),
    ('SELECT f@lnk(1) FROM "DUAL"', "f@lnk"), ('SELECT shop.f@lnk(1) FROM "DUAL"', "f@lnk"),
    ('SELECT "ID" FROM "ORDERS" WHERE "ID" = f@lnk("ID")', "f@lnk"), ('SELECT f@lnk FROM "DUAL"', "f@lnk"),
    ('SELECT @lnk FROM "DUAL"', "@lnk")])
def test_a_function_or_sequence_over_a_database_link_is_refused(superset_oracle, decided, named):
    """A function or a sequence named over a database link is another database's, as a table there is: refused with
    the same plain reason, nothing sent."""
    with pytest.raises(RefusedStatement, match=f"names {re.escape(named)} over a database link"):
        for_superset(decided)


@pytest.mark.parametrize("decided", [
    'SELECT "ORDERS"."ID", "CUSTOMERS"."NAME" FROM ("ORDERS" JOIN "CUSTOMERS" ON "CUSTOMERS"."ID" = "ORDERS"."CID")',
    'SELECT 1 FROM (("ORDERS" JOIN "CUSTOMERS" ON 1 = 1) JOIN "ITEMS" ON 1 = 1)',
    'SELECT COUNT(*) FROM "ORDERS", ("ORDERS" JOIN "CUSTOMERS" ON 1 = 1)',
    'SELECT 1 FROM ("ORDERS" JOIN "CUSTOMERS" ON 1 = 1) JOIN "ITEMS" ON 1 = 1',
    'SELECT 1 FROM "ORDERS" JOIN ("CUSTOMERS" JOIN "ITEMS" ON 1 = 1) ON 1 = 1',
    'SELECT 1 FROM "ORDERS" WHERE "ID" IN (SELECT 1 FROM ("CUSTOMERS" JOIN "ITEMS" ON 1 = 1))',
    'SELECT 1 FROM (("ORDERS") JOIN "CUSTOMERS" ON 1 = 1)',
    'SELECT 1 FROM ((SELECT 1 AS "X" FROM "DUAL") "D" JOIN "CUSTOMERS" ON 1 = 1)'])
def test_a_parenthesized_join_is_refused(superset_oracle, decided):
    """A join inside parentheses, at any depth, is refused, nothing sent; the model is told to write the join without
    them."""
    with pytest.raises(RefusedStatement, match="a join inside parentheses.+write the join without the parentheses"):
        for_superset(decided)


@pytest.mark.parametrize("decided", [
    'SELECT "SHOP"."ORDERS"."ID" FROM "SHOP"."ORDERS"', "SELECT SHOP.ORDERS.ID FROM SHOP.ORDERS",
    "SELECT shop.orders.id FROM shop.orders",
    'SELECT COUNT(*) FROM "SHOP"."ORDERS", "SHOP" WHERE "SHOP"."ORDERS"."ID" > 0',
    'SELECT "O"."ID" FROM "SHOP"."ORDERS" "O" WHERE "O"."ID" IN (SELECT "SHOP"."RETURNS"."ID" FROM "SHOP"."RETURNS")'])
def test_a_column_named_by_its_schema_and_table_is_refused(superset_oracle, decided):
    """A column qualified by its table's schema and name is refused, nothing sent; the model is told to alias the table
    and qualify the column by the alias, which is sent (`SELECT "O"."ID" FROM shop.orders "O"`)."""
    with pytest.raises(RefusedStatement, match="by its table's schema.+alias the table and qualify the column by the "
                                               "alias"):
        for_superset(decided)
    assert for_superset('SELECT "O"."ID" FROM "SHOP"."ORDERS" "O"') == 'SELECT "O"."ID" FROM shop.orders "O"'


@pytest.mark.parametrize("decided", [
    """SELECT q'[x]' FROM "ORDERS\"""", """SELECT nq'[x]' FROM "ORDERS\"""", """SELECT Q'<x>' FROM "ORDERS\"""",
    """SELECT "ID" FROM "ORDERS" WHERE "NOTE" = q'{it's}'""",
    """SELECT "ID" FROM "ORDERS" WHERE "NOTE" BETWEEN q'[a]' AND q'[b]'""",
    """SELECT "ID" FROM "ORDERS" WHERE "AT" > SYSDATE - INTERVAL q'[1]' DAY"""])
def test_a_q_quoted_literal_is_refused(superset_oracle, decided):
    """mnemiq never writes a q-quoted literal, and refuses a statement holding one, nothing sent; the model is told to
    write it as an ordinary literal. An ordinary literal holding q-quote text is sent."""
    with pytest.raises(RefusedStatement, match="q-quoted literal.+write it as an ordinary quoted literal"):
        for_superset(decided)
    plain = """SELECT "ID" FROM "ORDERS" WHERE "NOTE" = 'q''[x]''' OR "NOTE" = N'y'"""
    assert for_superset(plain) == plain.replace('FROM "ORDERS"', 'FROM orders "ORDERS"')


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
    # A table the rewrite leaves as it is, aliased all the same.
    ('SELECT "ID" FROM "OrderItems"', 'SELECT "ID" FROM "OrderItems" "OrderItems"'),
    ('SELECT orders.revenue FROM orders', 'SELECT orders.revenue FROM orders "ORDERS"'),
    # A table in parentheses, alone, and one with a sample, which the alias follows.
    ('SELECT 1 FROM ("ORDERS")', 'SELECT 1 FROM (orders "ORDERS")'),
    ('SELECT * FROM "ORDERS" SAMPLE (10)', 'SELECT * FROM orders SAMPLE (10) "ORDERS"'),
])
def test_only_an_unaliased_table_of_its_own_name_is_aliased(superset_oracle, decided, sent):
    assert for_superset(decided) == sent
    assert for_superset(sent) == sent


@pytest.mark.parametrize(("sql", "why"), [("SELECT * FROM ORDERS WHERE (", "does not parse"),
                                          ("", "no statement"),
                                          ("SELECT 1 FROM DUAL; SELECT 2 FROM ORDERS", "more than one")])
def test_a_statement_mnemiq_cannot_read_as_one_is_never_rewritten(superset_oracle, sql, why):
    """A statement whose tables mnemiq cannot find, or one of several, is refused, never sent: a statement is sent only
    with each table it reads named as Superset names it."""
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
        'SELECT 1 FROM orders "ORDERS", "OrderItems" "OrderItems" WHERE 1 = 0']
    assert {request["schema"] for request in client.requests} == {"shop"}


@pytest.mark.parametrize("sql", [
    "SELECT '@' FROM \"ORDERS\"", 'SELECT 1 FROM "ORDERS" JOIN "CUSTOMERS" ON "CUSTOMERS"."NOTE" = \'@x\'',
    'SELECT "ID" FROM "ORDERS" /* by a@b */', 'SELECT "ID" FROM "ORDERS" "O" -- see x@y',
    'SELECT "ID" FROM "ORDERS" "A@B"', 'SELECT "A@B" FROM "ORDERS"', 'SELECT "ID" AS "A@B" FROM "ORDERS"'])
def test_an_at_sign_outside_a_tables_name_is_no_link(superset_oracle, sql):
    """An `@` in a literal, a comment, a quoted alias or a quoted column's name is no database link: sent. (One in a
    clause after a table's name is none either; such a clause is refused for itself, `NOT_READ_ALIKE`.)"""
    assert "@" in for_superset(sql)


def test_a_statement_the_adapter_refuses_is_not_sent_and_says_why(superset_oracle):
    client = _Recorder()
    adapter = SupersetSqlAdapter(client, database_id=7, schema="shop")
    with pytest.raises(SupersetQueryFailed, match="give the CTE a name no table of the statement has"):
        adapter.execute_arrow(CTE_TWINS[0])
    with pytest.raises(SupersetQueryFailed, match="database link"):
        adapter.execute_arrow("SELECT 1 FROM orders@lnk")
    with pytest.raises(SupersetQueryFailed, match="could not confirm for"):
        adapter.execute_arrow(NOT_READ_ALIKE[0])
    assert client.requests == []


def test_a_statement_the_adapter_cannot_rewrite_is_not_sent(superset_oracle):
    client = _Recorder()
    with pytest.raises(SupersetQueryFailed, match="not sent"):
        SupersetSqlAdapter(client, database_id=7, schema="shop").execute_arrow("SELECT * FROM ORDERS WHERE (")
    assert client.requests == []


def test_the_text_sent_is_the_text_checked_for_template_markers(superset_oracle, monkeypatch):
    """`execute_sql` renders a statement as a template before it checks and runs it, so the text it is sent, written
    in Superset's spelling, is the text held to hold no template marker. The rewrite writes only names, so here it is
    made to add a marker, which is refused and nothing is sent; a marker in the text as written is named as before."""
    import mnemiq.superset.adapter as adapter_module

    client = _Recorder()
    adapter = SupersetSqlAdapter(client, database_id=7, schema="shop")
    monkeypatch.setattr(adapter_module, "for_superset", lambda sql: sql + " -- {{ x }}")
    with pytest.raises(SupersetQueryFailed, match="holds {{, which Superset reads as a template marker"):
        adapter.execute_arrow('SELECT 1 FROM "ORDERS"')
    monkeypatch.undo()
    with pytest.raises(SupersetQueryFailed, match="holds {%, which Superset reads as a template marker"):
        adapter.execute_arrow("SELECT {% x %} FROM ORDERS")
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
    """rc2 applies the row rules of a virtual dataset's tables to every query of it, so the SQL a joined dataset is made
    from names them as Superset does; the plan keeps the SQL mnemiq decided, which the check reads the dataset with
    (`made_virtual_sql`)."""
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


@pytest.mark.parametrize("dialect", ["postgres", "oracle"])
def test_a_charts_summary_names_its_table_as_the_snapshot_does(request, dialect):
    """The model is shown a chart's table beside the snapshot's columns, so in the snapshot's spelling: on Oracle the
    table Superset reads for the dataset's name, ORDERS for `orders`; on PostgreSQL the dataset's name as it was."""
    from mnemiq.superset.chart_change import _chart_summary
    from mnemiq.superset.chart_check import ChartRead

    if dialect == "oracle":
        request.getfixturevalue("superset_oracle")
    read = ChartRead(chart_id=41, title="By region", viz_type="pie", form_data=None, changed_on=None)
    assert _chart_summary(read, _facts())["table"] == ("ORDERS" if dialect == "oracle" else "orders")
    assert _chart_summary(read, _facts(is_virtual=True, sql="SELECT 1 FROM dual"))["table"] is None


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


@pytest.mark.parametrize(("made", "said"), [
    ('WITH "orders" AS (SELECT 1 AS "ID" FROM "DUAL") SELECT "REGION", "CHANNEL", "PREMIUM" FROM "ORDERS"',
     "give the CTE a name no table of the statement has"),
    ('SELECT "REGION", "CHANNEL", "PREMIUM" FROM "ORDERS" WHERE (', "the statement does not parse"),
], ids=["refused", "unreadable"])
def test_a_made_dataset_whose_sql_mnemiq_does_not_send_is_not_checked(fake, monkeypatch, superset_oracle, made, said):
    """The check holds Superset's copy of a made dataset's SQL to the text the build sends for it (`for_superset`).
    SQL that `for_superset` refuses, or cannot read, leaves the chart not checked, saying why; nothing is run."""
    from mnemiq.superset.chart_check import ChartStatus

    stored = {"REGION": "REGION", "CHANNEL": "CHANNEL", "PREMIUM": "PREMIUM"}
    decided, sent, check = _checked(fake, monkeypatch, _oracle_chart(stored, virtual_sql=made), made={12: made})
    assert check.status is ChartStatus.NOT_CHECKED, check
    assert "made with SQL mnemiq does not send to Superset" in check.reason and said in check.reason, check.reason
    assert decided == [] and sent == []


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
