"""How a plan whose columns are named with their tables becomes rows (`plan_rows`): each measure's base table, the LEFT
JOIN chain from that base to every other table a part needs, and the name the joined dataset gives each column.

The tables are the demo shop's, every relationship many to one: order_items to orders to customers, order_items to
products, and returns to order_items. Each table's own key holds each value once, as enrichment's profile reads it, and
a key pointing at another table repeats, except a return's item: an item is returned at most once.
"""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from mnemiq.contract import Relationship
from mnemiq.contract.values import JoinKey
from mnemiq.superset.plan_rows import (Link, Refusal, Step, aggregated, base_among, base_of, chain, column_of,
                                       column_text, comparable, edges, far_side, join_text, links_of, qualified,
                                       qualify, read_join, rewrite)
from mnemiq.superset.planner import joined_names
from mnemiq.superset.saved_metrics import normalized
from mnemiq.superset.settings import SUPERSET_DIALECT


def _relationship(source: str, target: str, *keys: tuple[str, str], name: str | None = None) -> Relationship:
    return Relationship(id=name or f"{source}->{target}", from_=source, to=target, cardinality="many_to_one",
                        join_keys=[JoinKey(left=left, right=right) for left, right in keys])


SHOP = [_relationship("order_items", "orders", ("order_id", "order_id")),
        _relationship("order_items", "products", ("product_id", "product_id")),
        _relationship("orders", "customers", ("customer_id", "customer_id")),
        _relationship("returns", "order_items", ("order_item_id", "order_item_id"))]
TABLES = ("orders", "order_items", "products", "customers", "returns")
NAMES = {table: table for table in TABLES}
COLUMNS = {"orders": ("order_id", "customer_id", "order_date", "channel"),
           "order_items": ("order_item_id", "order_id", "product_id", "revenue", "unit_price", "quantity"),
           "products": ("product_id", "unit_price", "category"),
           "customers": ("customer_id", "region")}
UNIQUE = {("orders", "order_id"): True, ("order_items", "order_item_id"): True, ("products", "product_id"): True,
          ("customers", "customer_id"): True, ("returns", "return_id"): True, ("returns", "order_item_id"): True,
          ("order_items", "order_id"): False, ("order_items", "product_id"): False,
          ("orders", "customer_id"): False}

ITEM_ORDER = Link(("order_items", "order_id"), ("orders", "order_id"), "order_items->orders")
ITEM_PRODUCT = Link(("order_items", "product_id"), ("products", "product_id"), "order_items->products")
ORDER_CUSTOMER = Link(("orders", "customer_id"), ("customers", "customer_id"), "orders->customers")
RETURN_ITEM = Link(("returns", "order_item_id"), ("order_items", "order_item_id"), "returns->order_items")
KNOWN = links_of(SHOP, NAMES)
SHOP_EDGES = edges((), KNOWN)
# The declared type of each column a link here names, as the snapshot records one: every key an integer and a price an
# exact numeric, so every link compares its columns exactly unless a test says otherwise.
TYPES = {**{(table, column): "integer" for table, columns in {
    "orders": ("order_id", "customer_id", "payer_id"),
    "order_items": ("order_item_id", "order_id", "product_id", "shipment_id"),
    "products": ("product_id",), "customers": ("customer_id", "account_id"),
    "returns": ("order_item_id", "customer_id", "order_id", "product_id"),
    "shipments": ("shipment_id", "customer_id"), "employees": ("employee_id", "department_id"),
    "departments": ("department_id", "manager_id"), "Order Items": ("Order Id",)}.items() for column in columns},
    ("order_items", "unit_price"): "numeric(10,2)", ("products", "unit_price"): "numeric(10,2)"}


def _refused(result: object, *named: str) -> str:
    assert isinstance(result, Refusal), result
    for text in named:
        assert text in result.reason, (text, result.reason)
    return result.reason


def test_links_are_the_visible_relationships_by_table_name():
    assert KNOWN == (ITEM_ORDER, ITEM_PRODUCT, ORDER_CUSTOMER, RETURN_ITEM)
    # A relationship names its tables by the snapshot's object ids; a link names them as the planner does, and a
    # relationship touching a table the builder cannot see is no link. One of two key columns is one link each.
    by_id = [_relationship("obj-items", "obj-orders", ("order_id", "order_id"), ("region", "region")),
             _relationship("obj-orders", "obj-hidden", ("plan_id", "plan_id"))]
    assert links_of(by_id, {"obj-items": "Order Items", "obj-orders": "orders"}) == (
        Link(("Order Items", "order_id"), ("orders", "order_id"), "obj-items->obj-orders"),
        Link(("Order Items", "region"), ("orders", "region"), "obj-items->obj-orders"))


def test_a_measures_base_is_the_table_inside_its_aggregate():
    assert base_of("SUM(order_items.revenue)", None) == "order_items"
    # A CASE's condition chooses the rows; its value is what is added up.
    assert aggregated("AVG(CASE WHEN orders.channel = 'web' THEN order_items.revenue END)") == {"order_items"}
    assert base_of("AVG(CASE WHEN orders.channel = 'web' THEN order_items.revenue END)", None) == "order_items"
    assert base_of("SUM(order_items.revenue) FILTER (WHERE orders.channel = 'web')", None) == "order_items"
    assert base_of("PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY order_items.revenue)", None) == "order_items"
    assert base_of("COUNT(DISTINCT order_items.order_id)", "orders") == "order_items"
    assert base_of("COUNT(*)", "orders") == "orders"
    assert _refused(base_of("COUNT(*)", None)) == ADDS_UP_NO_COLUMN
    _refused(base_of("SUM(order_items.revenue) + SUM(returns.amount)", None), "order_items", "returns")
    _refused(base_of("SUM(revenue)", "order_items"), "'revenue'", "without its table")


# Why a measure whose aggregate adds up no column is refused without its `table`: what to give, whatever its condition
# names.
ADDS_UP_NO_COLUMN = ('its aggregate adds up no column, whatever its condition names, so it does not say whose rows it '
                     'counts: give the measure "table", the table whose rows it counts, or count one of that table\'s '
                     "columns, as COUNT(<table>.<key>)")


@pytest.mark.parametrize("expression", [
    "COUNT(*)",
    "SUM(CASE WHEN orders.status = 'open' THEN 1 ELSE 0 END)",
    "COUNT(CASE WHEN orders.status = 'open' THEN 1 END)",
    "AVG(CASE WHEN orders.channel = 'web' THEN 1.0 ELSE 0 END)",
    "SUM(CASE WHEN orders.status = 'won' THEN 1 ELSE 0 END)::double precision"
    " / NULLIF(SUM(CASE WHEN orders.status IN ('won', 'lost') THEN 1 ELSE 0 END), 0)"])
def test_an_aggregate_of_no_column_takes_its_table_whatever_its_condition_names(expression):
    """A count or sum of 1s, or a share of 1.0 and 0, adds up no column: its condition says which rows it covers, never
    whose rows they are. Its `table` says that, and without one it goes back saying to give it."""
    assert aggregated(expression) == set()
    assert base_of(expression, "order_items") == "order_items"
    assert _refused(base_of(expression, None)) == ADDS_UP_NO_COLUMN


NO_AGGREGATE = ("it is no aggregate: it gives one value per row, not one per group; write the measure with an "
                "aggregate, as SUM(<table>.<column>) or COUNT(*)")
WINDOWED = ("its aggregate is computed over a window (OVER), which gives one value per row, not one per group, so it "
            "is no measure either: write the aggregate without OVER")
UNMODELLED = ("mnemiq cannot tell whether it aggregates: it holds a function mnemiq does not model and no aggregate "
              "mnemiq does; write the measure with SQL's own aggregates, as SUM(<table>.<column>) or COUNT(*)")


@pytest.mark.parametrize(("expression", "why"), [
    ("order_items.revenue", NO_AGGREGATE),
    ("order_items.revenue * order_items.quantity", NO_AGGREGATE),
    ("CASE WHEN orders.channel = 'web' THEN 1 ELSE 0 END", NO_AGGREGATE),
    ("1", NO_AGGREGATE),
    ("SUM(order_items.revenue) OVER ()", WINDOWED),
    ("COUNT(*) OVER (PARTITION BY orders.channel)", WINDOWED),
    ("SUM(order_items.revenue) FILTER (WHERE orders.channel = 'web') OVER ()", WINDOWED),
    ("MY_TOTAL(order_items.revenue)", UNMODELLED),
], ids=["a column", "a product", "a case", "a constant", "a window's sum", "a window's count", "a filtered window",
        "an unmodelled function"])
@pytest.mark.parametrize("table", [None, "order_items"], ids=["no table", "a table"])
def test_an_expression_that_does_not_aggregate_is_no_measure_whatever_its_table(expression, why, table):
    """A measure gives one value per group. An expression with no aggregate, or whose every aggregate is a window's,
    gives one per row: it is refused saying so, with or without a `table`, never told to give one, which would pass
    it here only for the plan check to refuse it, spending the one repair. One whose only function mnemiq does not
    model is refused as the check refuses it: mnemiq cannot tell whether it aggregates."""
    assert _refused(base_of(expression, table)) == why
    # A window over an aggregate aggregates.
    assert base_of("SUM(SUM(order_items.revenue)) OVER ()", table) == "order_items"


def test_qualified_names_each_column_with_its_table_or_says_why_not():
    assert qualified("AVG(CASE WHEN orders.channel = 'web' THEN order_items.revenue END)") == [
        ("orders", "channel"), ("order_items", "revenue")]
    assert qualified("customers.region") == [("customers", "region")]
    assert qualified("SUM(order_items.revenue) - SUM(order_items.revenue)") == [("order_items", "revenue")]
    assert "'region'" in qualified("region")
    assert "public.orders.channel" in qualified("public.orders.channel")
    assert "orders.*" in qualified("COUNT(orders.*)")
    assert "subquery" in qualified("(SELECT MAX(orders.order_date) FROM orders)")
    for text in ("SUM(", "SUM(orders.x) FROM orders", "orders.order_id, orders.channel", "DISTINCT orders.channel",
                 ""):
        assert "not one SQL expression" in qualified(text), text


def test_a_chain_takes_the_shortest_path_over_unique_far_sides():
    assert chain("order_items", {"customers", "products", "order_items"}, SHOP_EDGES, UNIQUE, types=TYPES) == (
        Step("orders", (ITEM_ORDER,)), Step("products", (ITEM_PRODUCT,)), Step("customers", (ORDER_CUSTOMER,)))
    assert chain("returns", {"customers"}, SHOP_EDGES, UNIQUE, types=TYPES) == (
        Step("order_items", (RETURN_ITEM,)), Step("orders", (ITEM_ORDER,)), Step("customers", (ORDER_CUSTOMER,)))
    # A link straight to customers is one join where the way through order_items and orders is three.
    direct = Link(("returns", "customer_id"), ("customers", "customer_id"))
    assert chain("returns", {"customers"}, edges((direct,), KNOWN), UNIQUE, types=TYPES) == (
        Step("customers", (direct,)),)
    # Walking a link from its far end joins the table it starts from: an item is returned at most once.
    assert chain("order_items", {"returns"}, SHOP_EDGES, UNIQUE, types=TYPES) == (
        Step("returns", (Link(("order_items", "order_item_id"), ("returns", "order_item_id"),
                              "returns->order_items"),)),)
    assert chain("orders", {"orders"}, SHOP_EDGES, UNIQUE, types=TYPES) == ()


def test_a_far_side_known_to_repeat_is_never_joined():
    assert far_side("orders", ["customer_id"], UNIQUE) is False
    # Each customer has many orders, so customers' rows cannot take an order's columns.
    _refused(chain("customers", {"orders"}, SHOP_EDGES, UNIQUE, types=TYPES), "orders",
             "customers.customer_id = orders.customer_id", "orders.customer_id holds a value more than once")
    repeating = {**UNIQUE, ("orders", "order_id"): False}
    _refused(chain("order_items", {"customers"}, SHOP_EDGES, repeating, types=TYPES), "customers")


def test_an_unknown_far_side_is_usable():
    unknown = {key: fact for key, fact in UNIQUE.items() if key != ("orders", "order_id")}
    assert far_side("orders", ["order_id"], unknown) is None
    assert chain("order_items", {"orders"}, SHOP_EDGES, unknown, types=TYPES) == (Step("orders", (ITEM_ORDER,)),)


@pytest.mark.parametrize(("left", "right"), [
    ("integer", "bigint"), ("numeric", "integer"), ("numeric(10,2)", "integer"), ("int8", "smallint"),
    ("decimal(5)", "int4"), ("text", "varchar(20)"), ("character varying(10)", "character varying(20)"),
    ("TEXT", "character varying"), ("double precision", "float8"), ("real", "real"), ("char(3)", "bpchar"),
    ("date", "DATE"), ("timestamp(3) without time zone", "timestamp"), ("timestamp with time zone", "timestamptz"),
    ("boolean", "bool"), ("uuid", "uuid"), ("citext", "citext"), ("integer[]", "int4[]"),
])
def test_columns_of_one_kind_compare_exactly(left, right):
    """An integer with an exact numeric, text with varchar, and any other type with itself, its length and precision
    aside: an equality of the two reads no two values of either as one."""
    assert comparable(left, right) and comparable(right, left)


@pytest.mark.parametrize(("left", "right"), [
    ("double precision", "bigint"), ("double precision", "numeric"), ("real", "integer"), ("real", "double precision"),
    ("char(3)", "varchar(3)"), ("character", "text"), ("timestamp", "timestamp with time zone"), ("date", "timestamp"),
    ("text", "integer"), ("uuid", "text"), ("citext", "text"), ("integer[]", "text[]"),
    ("bigint", None), (None, None), ("bigint", ""), ("unknown", "unknown"), ("USER-DEFINED", "USER-DEFINED"),
    ("ARRAY", "ARRAY"),
])
def test_columns_of_two_kinds_or_of_no_known_type_do_not_compare_exactly(left, right):
    """Equality across any other two types converts one side, and a conversion that rounds (a bigint read as a double
    precision) reads two of its values as one; a column with no type recorded, or one that names none, could be
    either."""
    assert not comparable(left, right) and not comparable(right, left)


def test_a_link_between_columns_of_two_kinds_is_never_walked():
    """A double precision equal to a bigint compares both as double precision, which reads two ids past 2^53 as one: a
    customer key that holds each value once may still meet one order twice, and LEFT JOIN would repeat the order. The
    link is never joined by, whatever the key facts say, and the table it alone reaches is refused naming both columns
    and both types."""
    typed = {**TYPES, ("orders", "customer_id"): "double precision", ("customers", "customer_id"): "bigint"}
    reason = _refused(chain("orders", {"customers"}, SHOP_EDGES, UNIQUE, types=typed), "'customers'")
    assert ("orders.customer_id is double precision and customers.customer_id is bigint: a link joins columns whose "
            "values compare exactly") in reason, reason
    _refused(base_among({"orders", "customers"}, SHOP_EDGES, UNIQUE, types=typed), "'orders'", "'customers'")
    # Where another way of one kind reaches the table, it is joined by: shipments' link to customers is an integer's.
    assert chain("order_items", {"customers"}, edges((), SHIPPED), SHIPPED_UNIQUE, types=typed) == (
        Step("shipments", (SHIPPED[-2],)), Step("customers", (SHIPPED[-1],)))
    # A stated link of two kinds is no more joined by than a known one.
    stated = Link(("orders", "customer_id"), ("customers", "customer_id"), "orders->customers")
    _refused(chain("orders", {"customers"}, edges((stated,), KNOWN), UNIQUE, (stated,), types=typed),
             "orders.customer_id is double precision and customers.customer_id is bigint")


def test_a_link_with_a_column_of_no_recorded_type_is_never_walked():
    untyped = {key: data_type for key, data_type in TYPES.items() if key != ("customers", "customer_id")}
    reason = _refused(chain("orders", {"customers"}, SHOP_EDGES, UNIQUE, types=untyped), "'customers'")
    assert ("orders.customer_id is integer and customers.customer_id has no recorded type: a link joins columns whose "
            "values compare exactly, which mnemiq cannot tell without both columns' types") in reason, reason


def test_a_two_column_far_side_is_unique_when_either_column_is():
    assert far_side("order_items", ["order_id", "order_item_id"], UNIQUE) is True
    assert far_side("order_items", ["order_item_id", "order_id"], UNIQUE) is True
    # Each column repeats, and the two together may still hold each value once: unknown, never a repeat.
    assert far_side("order_items", ["order_id", "product_id"], UNIQUE) is None
    assert far_side("order_items", ["order_id", "order_id"], UNIQUE) is False
    two = (Link(("returns", "order_id"), ("order_items", "order_id")),
           Link(("returns", "product_id"), ("order_items", "product_id")))
    assert chain("returns", {"order_items"}, edges(two, KNOWN), UNIQUE, types=TYPES) == (Step("order_items", two),)


def test_two_shortest_paths_or_two_links_to_a_table_are_ambiguous():
    # Items ship in shipments, each to a customer: customers is two links from order_items both ways.
    shipped = KNOWN + (Link(("order_items", "shipment_id"), ("shipments", "shipment_id")),
                       Link(("shipments", "customer_id"), ("customers", "customer_id")))
    unique = {**UNIQUE, ("shipments", "shipment_id"): True}
    _refused(chain("order_items", {"customers"}, edges((), shipped), unique, types=TYPES), "customers",
             "order_items.order_id = orders.order_id then orders.customer_id = customers.customer_id",
             "order_items.shipment_id = shipments.shipment_id then shipments.customer_id = customers.customer_id",
             "state the links of the way you mean")
    _refused(chain("returns", {"customers"}, edges((), shipped), unique, types=TYPES), "customers")
    # A table reached two ways that no part needs is never joined, so it is no ambiguity.
    assert chain("order_items", {"orders", "shipments"}, edges((), shipped), unique, types=TYPES) == (
        Step("orders", (ITEM_ORDER,)), Step("shipments", (shipped[-2],)))
    # An order's buyer and its payer are both customers: two links from orders to one table.
    payer = Link(("orders", "payer_id"), ("customers", "customer_id"))
    _refused(chain("orders", {"customers"}, edges((), (*KNOWN, payer)), UNIQUE, types=TYPES),
             "orders.customer_id = customers.customer_id", "orders.payer_id = customers.customer_id")
    _refused(chain("order_items", {"customers"}, edges((), (*KNOWN, payer)), UNIQUE, types=TYPES), "orders.payer_id")
    # Stated as one link from orders to customers, the two are still two: one link uses each column once.
    both = (Link(("orders", "customer_id"), ("customers", "customer_id"), "orders->customers"),
            Link(("orders", "payer_id"), ("customers", "customer_id"), "orders->customers"))
    _refused(chain("orders", {"customers"}, edges(both, KNOWN), UNIQUE, types=TYPES),
             "orders.customer_id = customers.customer_id or orders.payer_id = customers.customer_id")


SHIPPED = KNOWN + (Link(("order_items", "shipment_id"), ("shipments", "shipment_id"), "shipped"),
                   Link(("shipments", "customer_id"), ("customers", "customer_id"), "ships_to"))
SHIPPED_UNIQUE = {**UNIQUE, ("shipments", "shipment_id"): True}


def test_a_tie_between_shortest_ways_goes_to_the_one_with_the_most_stated_links():
    # Stating the link to shipments chooses the way through it: one stated link against none.
    ship = Link(("order_items", "shipment_id"), ("shipments", "shipment_id"), "order_items->shipments")
    assert chain("order_items", {"customers"}, edges((ship,), SHIPPED), SHIPPED_UNIQUE, (ship,), types=TYPES) == (
        Step("shipments", (ship,)), Step("customers", (SHIPPED[-1],)))
    # Stating a link the snapshot also knows chooses its way the same.
    buyer = Link(("orders", "customer_id"), ("customers", "customer_id"), "orders->customers")
    assert chain("order_items", {"customers"}, edges((buyer,), SHIPPED), SHIPPED_UNIQUE, (buyer,), types=TYPES) == (
        Step("orders", (ITEM_ORDER,)), Step("customers", (buyer,)))
    # Both links of a way against one of the other's.
    ships_to = Link(("shipments", "customer_id"), ("customers", "customer_id"), "shipments->customers")
    stated = (ship, ships_to, buyer)
    assert chain("order_items", {"customers"}, edges(stated, SHIPPED), SHIPPED_UNIQUE, stated, types=TYPES) == (
        Step("shipments", (ship,)), Step("customers", (ships_to,)))
    # A tie further from the base is broken as well: returns reach customers through order_items either way.
    assert chain("returns", {"customers"}, edges((ship,), SHIPPED), SHIPPED_UNIQUE, (ship,), types=TYPES) == (
        Step("order_items", (RETURN_ITEM,)), Step("shipments", (ship,)), Step("customers", (SHIPPED[-1],)))


def test_shortest_ways_with_as_many_stated_links_are_still_ambiguous():
    ship = Link(("order_items", "shipment_id"), ("shipments", "shipment_id"), "order_items->shipments")
    buyer = Link(("orders", "customer_id"), ("customers", "customer_id"), "orders->customers")
    _refused(chain("order_items", {"customers"}, edges((ship, buyer), SHIPPED), SHIPPED_UNIQUE, (ship, buyer),
                   types=TYPES),
             "'customers' is reached from the rows of 'order_items' two ways of one length",
             "state the links of the way you mean")
    # A stated link on neither way breaks no tie.
    returned = Link(("returns", "order_item_id"), ("order_items", "order_item_id"), "returns->order_items")
    _refused(chain("order_items", {"customers"}, edges((returned,), SHIPPED), SHIPPED_UNIQUE, (returned,), types=TYPES),
             "two ways of one length")


def test_a_part_with_no_measure_takes_as_its_base_the_one_table_reaching_the_others():
    assert base_among({"orders", "customers"}, SHOP_EDGES, UNIQUE, types=TYPES) == "orders"
    assert base_among({"order_items", "customers", "products"}, SHOP_EDGES, UNIQUE, types=TYPES) == "order_items"
    assert base_among({"orders"}, SHOP_EDGES, UNIQUE, types=TYPES) == "orders"
    # Neither reaches the other: a customer has many orders, and an order is no product's.
    _refused(base_among({"customers", "products"}, SHOP_EDGES, UNIQUE, types=TYPES), "'customers'", "'products'")
    # An item is returned at most once, so each reaches the other: which rows are meant is not said.
    _refused(base_among({"order_items", "returns"}, SHOP_EDGES, UNIQUE, types=TYPES), "'order_items'", "'returns'")
    _refused(base_among(set(), SHOP_EDGES, UNIQUE, types=TYPES), "no column")


def test_a_chart_column_is_one_column_named_with_its_table():
    assert column_of("orders.order_date") == ("orders", "order_date")
    assert column_of('"Order Items".Revenue') == ("Order Items", "revenue")
    assert "without its table" in column_of("order_date")
    for text in ("SUM(orders.order_id)", "orders.order_date AS day", "orders.channel || orders.channel"):
        assert "not one column" in column_of(text), text
    assert "not one SQL expression" in column_of("orders.")


def test_column_text_reads_back_as_the_column_it_names():
    for table, column in (("orders", "order_date"), ("Order Items", "Revenue"), ("ORDERS", "id"), ("order", "x"),
                          ("Ärger", "wert"), ("select", "from")):
        assert column_of(column_text(table, column)) == (table, column), (table, column)
    assert column_text("orders", "order_date") == "orders.order_date"
    assert column_text("Order Items", "Revenue") == '"Order Items"."Revenue"'


def test_a_join_text_reads_back_as_its_base_and_steps():
    steps = chain("order_items", {"customers", "products"}, SHOP_EDGES, UNIQUE, types=TYPES)
    two = (Link(("returns", "order_id"), ("order_items", "order_id")),
           Link(("returns", "product_id"), ("order_items", "product_id")))
    for base, made in (("order_items", steps), ("returns", (Step("order_items", two),)), ("orders", ())):
        base_read, steps_read = read_join(join_text(base, made))
        assert base_read == base
        assert [(step.table, [(link.left, link.right) for link in step.on]) for step in steps_read] == [
            (step.table, [(link.left, link.right) for link in step.on]) for step in made]
    # A join written another way is read as far as its tables and its equalities go.
    base, steps = read_join("policy p JOIN claim c ON c.policy_id = p.policy_id AND c.status = 'open'")
    assert base == "policy" and [(step.table, [(link.left, link.right) for link in step.on]) for step in steps] == [
        ("claim", [(("policy", "policy_id"), ("claim", "policy_id"))])]
    for text in ("policy JOIN", "(SELECT 1) AS t", "policy, LATERAL (SELECT 1) AS t"):
        assert read_join(text) is None, text


def test_qualify_names_each_dataset_column_with_its_table_again():
    names = joined_names([(table, COLUMNS[table]) for table in ("order_items", "orders", "products")])
    owners = {name: key for key, name in names.items()}
    for written in ("SUM(order_items.revenue)", "AVG(CASE WHEN orders.channel = 'web' THEN order_items.unit_price END)",
                    "SUM(order_items.quantity * products.unit_price) / COUNT(DISTINCT orders.order_id)"):
        shown = rewrite(written, names)
        assert qualify(shown, owners) == written
        assert rewrite(qualify(shown, owners), names) == shown
    # The model's own spelling of a column whose name the dataset keeps is kept.
    assert qualify("SUM(Revenue)", owners) == "SUM(order_items.Revenue)"
    # What the dataset does not name is left as written; text that does not parse is returned as it is.
    assert qualify("SUM(bogus) + SUM(t.revenue)", owners) == "SUM(bogus) + SUM(t.revenue)"
    assert qualify("SUM(", owners) == "SUM("


def test_two_relationships_between_one_pair_are_two_links():
    # An employee works in a department, and a department is managed by an employee.
    staff = links_of([_relationship("employees", "departments", ("department_id", "department_id")),
                      _relationship("departments", "employees", ("manager_id", "employee_id"))],
                     {"employees": "employees", "departments": "departments"})
    unique = {("departments", "department_id"): True, ("employees", "employee_id"): True,
              ("employees", "department_id"): False}
    _refused(chain("employees", {"departments"}, edges((), staff), unique, types=TYPES),
             "employees.department_id = departments.department_id or employees.employee_id = departments.manager_id")
    # From a department only the manager's link is one to join by: a department has many employees.
    managed = Link(("departments", "manager_id"), ("employees", "employee_id"), "departments->employees")
    assert chain("departments", {"employees"}, edges((), staff), unique, types=TYPES) == (
        Step("employees", (managed,)),)
    # The same columns as one link join by both together.
    one = (Link(("employees", "department_id"), ("departments", "department_id"), "employees->departments"),
           Link(("departments", "manager_id"), ("employees", "employee_id"), "employees->departments"))
    assert chain("employees", {"departments"}, edges(one, ()), unique, types=TYPES) == (Step("departments", (
        one[0], Link(("employees", "employee_id"), ("departments", "manager_id"), "employees->departments"))),)
    # An order's customer and its payer's account: two relationships to one table, no column used twice.
    paid = links_of([*SHOP, _relationship("orders", "customers", ("payer_id", "account_id"), name="payer")], NAMES)
    _refused(chain("order_items", {"customers"}, edges((), paid), {**UNIQUE, ("customers", "account_id"): True},
                   types=TYPES),
             "orders.customer_id = customers.customer_id or orders.payer_id = customers.account_id")
    # A relationship of the same column and one more is another link, each named with all its columns.
    priced = links_of([*SHOP, _relationship("order_items", "products", ("product_id", "product_id"),
                                            ("unit_price", "unit_price"), name="priced")], NAMES)
    _refused(chain("order_items", {"products"}, edges((), priced), UNIQUE, types=TYPES),
             "order_items.product_id = products.product_id or order_items.product_id = products.product_id and "
             "order_items.unit_price = products.unit_price")
    # Two relationships of the same columns are one link.
    twice = links_of([*SHOP, _relationship("orders", "customers", ("customer_id", "customer_id"), name="declared")],
                     NAMES)
    assert chain("orders", {"customers"}, edges((), twice), UNIQUE, types=TYPES) == (
        Step("customers", (ORDER_CUSTOMER,)),)


def test_an_unreachable_table_is_refused_by_name():
    reason = _refused(chain("order_items", {"customers", "suppliers", "orders"}, SHOP_EDGES, UNIQUE, types=TYPES),
                      "'suppliers'")
    assert "'customers'" not in reason and "'orders'" not in reason
    _refused(chain("products", {"customers"}, SHOP_EDGES, UNIQUE, types=TYPES), "'customers'", "'products'")


def test_a_table_no_link_reaches_asks_the_model_to_state_the_link():
    reason = _refused(chain("products", {"customers"}, SHOP_EDGES, UNIQUE, types=TYPES), "'customers'")
    assert 'state it in "links"' in reason
    assert '{"from": "<table>.<column>", "to": "<table>.<column>"}' in reason
    assert "referencing table's column" in reason and "referenced table's key" in reason


def test_a_stated_link_decides_its_pair():
    payer = Link(("orders", "payer_id"), ("customers", "customer_id"))
    decided = edges((payer,), KNOWN)
    assert decided[frozenset({"orders", "customers"})] == (payer,)
    assert decided[frozenset({"order_items", "orders"})] == (ITEM_ORDER,)
    assert len(decided) == len(SHOP_EDGES)
    # Stated from either end, and more than once, it is one link.
    reverse = Link(("customers", "customer_id"), ("orders", "payer_id"))
    assert edges((reverse, payer), KNOWN)[frozenset({"orders", "customers"})] == (reverse,)
    # Stating the payer's link chooses it over the buyer's the snapshot also knows.
    assert chain("orders", {"customers"}, edges((payer,), (*KNOWN, payer)), UNIQUE, types=TYPES) == (
        Step("customers", (payer,)),)
    # A link a table states to itself joins nothing.
    assert edges((Link(("orders", "order_id"), ("orders", "order_id")),), ()) == {}


def test_join_text_is_a_left_join_chain_from_the_base():
    steps = chain("order_items", {"customers", "products"}, SHOP_EDGES, UNIQUE, types=TYPES)
    text = join_text("order_items", steps)
    assert text == ('"order_items" LEFT JOIN "orders" ON "order_items"."order_id" = "orders"."order_id" '
                    'LEFT JOIN "products" ON "order_items"."product_id" = "products"."product_id" '
                    'LEFT JOIN "customers" ON "orders"."customer_id" = "customers"."customer_id"')
    tree = sqlglot.parse_one(f"SELECT 1 FROM {text}", read=SUPERSET_DIALECT)
    assert [join.side for join in tree.args["joins"]] == ["LEFT", "LEFT", "LEFT"]
    two = (Link(("returns", "order_id"), ("order_items", "order_id")),
           Link(("returns", "product_id"), ("order_items", "product_id")))
    assert join_text("returns", (Step("order_items", two),)) == (
        '"returns" LEFT JOIN "order_items" ON "returns"."order_id" = "order_items"."order_id" '
        'AND "returns"."product_id" = "order_items"."product_id"')
    assert join_text("orders", ()) == '"orders"'


def test_quoted_and_mixed_case_names_resolve_by_dialect():
    # Unquoted names fold as the dialect folds them, and only so far; a quoted name is the name as written.
    assert qualified('SUM("Order Items".revenue)') == [("Order Items", "revenue")]
    assert qualified("SUM(Order_Items.Revenue)") == [("order_items", "revenue")]
    assert qualified('SUM("Order_Items"."Revenue")') == [("Order_Items", "Revenue")]
    assert qualified("SUM(Ärger.Wert)") == [("Ärger", "wert")]
    assert base_of('SUM("Order Items".revenue)', None) == "Order Items"
    assert aggregated('SUM("ORDERS".revenue)') == {"ORDERS"}
    # A column the dataset keeps its name for keeps its spelling, quotes and all; a renamed one is quoted.
    assert rewrite('SUM("Order Items".revenue) / COUNT(Order_Items.Order_Id)',
                   {("Order Items", "revenue"): "revenue", ("order_items", "order_id"): "order_id"}) == (
        'SUM(revenue) / COUNT(Order_Id)')
    assert rewrite('SUM("Order Items"."Revenue")', {("Order Items", "Revenue"): "Revenue"}) == 'SUM("Revenue")'
    assert rewrite('SUM("Order Items"."Revenue")', {("Order Items", "Revenue"): "Order Items_Revenue"}) == (
        'SUM("Order Items_Revenue")')
    mixed = Link(("Order Items", "Order Id"), ("orders", "order_id"))
    steps = chain("Order Items", {"orders"}, edges((mixed,), ()), {("orders", "order_id"): True}, types=TYPES)
    assert join_text("Order Items", steps) == (
        '"Order Items" LEFT JOIN "orders" ON "Order Items"."Order Id" = "orders"."order_id"')


def test_rewrite_names_every_qualified_column_as_the_dataset_does():
    names = joined_names([(table, COLUMNS[table]) for table in ("order_items", "orders", "products")])
    assert rewrite("SUM(order_items.revenue)", names) == "SUM(revenue)"
    assert rewrite("AVG(CASE WHEN orders.channel = 'web' THEN order_items.unit_price END)", names) == (
        """AVG(CASE WHEN channel = 'web' THEN "order_items_unit_price" END)""")
    rewritten = rewrite("SUM(order_items.quantity * products.unit_price) / COUNT(DISTINCT orders.order_id)", names)
    assert rewritten == 'SUM(quantity * "products_unit_price") / COUNT(DISTINCT "orders_order_id")'
    # A measure over one table's columns reads as the same measure written on that table's own dataset, so a saved
    # metric or a kept measure of that definition is still matched.
    assert normalized(rewrite("SUM(order_items.revenue)", names)) == normalized("SUM(revenue)")
    assert normalized(rewrite("SUM(order_items.revenue * order_items.quantity)", names)) == (
        normalized("SUM(revenue * quantity)"))
    assert not [column for column in sqlglot.parse_one(f"SELECT {rewritten}", read=SUPERSET_DIALECT)
                .find_all(exp.Column) if column.table]
