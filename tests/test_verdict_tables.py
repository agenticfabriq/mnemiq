"""`verdict.tables` is not a log line -- it is the ACL that decides which worked examples a
caller may see (`_retrieve_examples`, *"an example must not reveal a forbidden table"*).

NOT RETROACTIVE: `tables` enters the snapshot at enrich time and `build_example_index` copies it
verbatim, so a store enriched before this fix keeps the flat-subtraction value and keeps showing
the leaky example. Re-indexing does not clear it -- the examples phase has to run again.

Both deciders built it as `find_all(exp.Table)` minus a flat set of CTE aliases: the exact
construct M31 filed and `base_tables()` was written to replace, still standing in the consumer
M31 did not reach, because it was not a guard at the time and is one now (M49).
"""
from __future__ import annotations

import sqlglot

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.scope import base_tables
from mnemiq.sql.verdict import Refusal

VISIBLE = {
    "customer": {"id", "ssn"},
    "orders": {"id", "customer_id"},
    "scratch": {"id", "customer_id"},
}


def _tables(sql):
    v = decide(sql, VISIBLE, policy=AccessPolicy())
    assert not isinstance(v, Refusal), f"refused {v.code.value}"
    return v.tables


def test_a_cte_named_after_a_table_does_not_delete_that_table_from_the_acl():
    """The leak: an orders-only caller is shown an example whose SQL reads `customer.ssn`."""
    with_cte = _tables(
        "WITH customer AS (SELECT id, ssn FROM customer) "
        "SELECT o.id FROM orders o JOIN customer c ON c.id = o.id"
    )
    assert "customer" in with_cte, with_cte


def test_the_same_query_without_the_cte_is_the_control():
    assert _tables("SELECT o.id FROM orders o JOIN customer c ON c.id = o.id") == [
        "customer",
        "orders",
    ]


def test_the_acl_withholds_the_example_once_the_tables_are_recorded():
    """The recorder feeds `set(tables) <= allowed`; assert the decision, not just the list."""
    sql = (
        "WITH customer AS (SELECT id, ssn FROM customer) "
        "SELECT o.id FROM orders o JOIN customer c ON c.id = o.id"
    )
    assert not set(_tables(sql)) <= {"orders"}


def test_a_self_named_cte_still_reports_the_table_its_body_reads():
    """M31's own shape, at this fourth consumer: the body reads the real `customer`."""
    assert "customer" in _tables(
        "WITH customer AS (SELECT id FROM customer) SELECT id FROM customer"
    )


def test_a_cte_alias_is_never_reported_as_a_table():
    """The other direction: `x` is not an object and must not be authorized as one."""
    assert _tables("WITH x AS (SELECT id FROM orders) SELECT id FROM x") == ["orders"]


def test_a_query_reading_nothing_real_reports_nothing():
    """`[]` must keep meaning 'reads no real object'. It is only safe for the ACL to treat an
    empty list as universally permitted while nothing else can produce one -- `base_tables`
    over-reports rather than returning [] when it cannot resolve (M48)."""
    assert _tables("WITH x AS (SELECT 1 AS id) SELECT id FROM x") == []


def _write_tables(sql):
    grants = GrantSet(objects=frozenset(VISIBLE), writable=frozenset({"scratch"}))
    v = decide_write(sql, VISIBLE, grants, adapter=None, dialect="duckdb",
                     policy=AccessPolicy(), views={}, writes_enabled=True)
    assert not isinstance(v, Refusal), f"refused {v.code.value}: {v.message}"
    return v.tables


# The write path spells its CTE inside the INSERT: a LEADING `WITH` is refused by shape
# (UNSCOPED_CTE), so this is the only spelling that reaches the recorder at all.
_LEAK = ("INSERT INTO scratch (id, customer_id) "
         "WITH customer AS (SELECT id, ssn FROM customer) "
         "SELECT o.id, o.customer_id FROM orders o JOIN customer c ON c.id = o.customer_id")


def test_the_write_decider_records_a_cte_shadowed_table_too():
    """The same defect on the other decider, and the one the old test missed: it wrote a plain
    INSERT..SELECT with no CTE at all, so it passed identically before and after the fix.

    Measured on the flat-subtraction code this replaces: `['orders', 'scratch']` -- `customer`
    deleted from the record by a CTE that merely shares its name, while the CTE's own body reads
    the real table.
    """
    assert _write_tables(_LEAK) == ["customer", "orders", "scratch"]


def test_the_write_target_is_recorded_even_though_nothing_reads_it():
    """An INSERT target is written and never read, so `base_tables` cannot see it and the audit
    would under-report what the statement touched. Unioned in explicitly for that reason."""
    assert _write_tables("INSERT INTO scratch SELECT id, id FROM orders") == ["orders", "scratch"]


def test_an_update_target_is_a_read_the_resolver_now_names():
    """This test has had three premises, and the churn is the point.

    v1 asserted the target is "recorded once and not twice" -- a set union cannot violate that,
    so it could not fail. v2 asserted `base_tables` does NOT contain the target and only the
    union puts it back, which was true and was the DEFECT: a target absent from the read set is
    a target no guard checks. Modelling it in the resolver closed three findings at once, so the
    honest assertion is now the opposite of v2's."""
    sql = "UPDATE scratch SET customer_id = 0 WHERE id IN (SELECT id FROM orders)"
    assert {t.name for t in base_tables(sqlglot.parse_one(sql, read="duckdb"))} == {"orders", "scratch"}
    assert _write_tables(sql) == ["orders", "scratch"]


def test_a_plain_insert_target_still_reaches_the_record_only_through_the_union():
    """The union is not redundant after the resolver change. A plain INSERT genuinely does not
    read its target, so `_target_read` returns None and `base_tables` correctly omits it -- while
    the audit record must still name what the write touched."""
    sql = "INSERT INTO scratch SELECT id, id FROM orders"
    assert {t.name for t in base_tables(sqlglot.parse_one(sql, read="duckdb"))} == {"orders"}
    assert _write_tables(sql) == ["orders", "scratch"]
