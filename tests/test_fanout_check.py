"""The fan-out check (register M109): refuse an aggregate whose rows a join has multiplied.

The fixtures restate, against a hand-built profile instead of a live database, the controls the
rule was validated against before any accuracy number was read, plus the edges the engine port
adds.
"""
from __future__ import annotations

import pytest
import sqlglot

from mnemiq.contract import Column, Snapshot
from mnemiq.sql.fanout_check import check_fanout, key_facts, value_terms
from mnemiq.sql.verdict import RefusalCode

# The acme_syn shapes, profiled the way the glossary arm found them: 30,000 claims over 23,322
# policies, 125,000 premium rows over 50,000 policies, one dim_policy row per policy.
_PROFILE = {
    # table: {column: (row_count, distinct_count, null_count)}
    "fact_claim": {"claim_id": (30000, 30000, 0), "policy_id": (30000, 23322, 0),
                   "paid_amount_cents": (30000, 29000, 0), "settlement_days": (30000, 400, 0)},
    "fact_premium": {"premium_id": (125000, 125000, 0), "policy_id": (125000, 50000, 0),
                     "earned_premium_cents": (125000, 90000, 0)},
    "dim_policy": {"policy_id": (50000, 50000, 0), "region": (50000, 4, 0)},
    "policy": {"policy_id": (50000, 50000, 0)},
    "claim": {"claim_id": (30000, 30000, 0), "policy_id": (30000, 23322, 0),
              "settlement_days": (30000, 400, 0)},
    "claim_amount": {"claim_id": (90000, 30000, 0), "amount_component_type": (90000, 3, 0),
                     "amount": (90000, 80000, 0)},
}


def _snapshot(profile=_PROFILE) -> Snapshot:
    return Snapshot(
        version="v1", source_id="acme", created_at="t",
        columns=[
            Column(id=f"{t}.{c}", object_id=t, name=c,
                   row_count=r, distinct_count=d, null_count=n)
            for t, cols in profile.items() for c, (r, d, n) in cols.items()
        ],
    )


VISIBLE = {t: set(cols) for t, cols in _PROFILE.items()}
KEYS = key_facts(_snapshot())


def _check(sql: str, keys=KEYS, visible=VISIBLE):
    return check_fanout(sqlglot.parse_one(sql, read="duckdb"), visible, keys)


# A retail star: many line items per order and per product, one products row per product.
_RETAIL = {
    "order_items": {"order_id": (1000, 300, 0), "product_id": (1000, 50, 0),
                    "quantity": (1000, 20, 0)},
    "products": {"product_id": (50, 50, 0), "unit_price": (50, 40, 0), "category": (50, 5, 0)},
}


def _check_profile(sql: str, profile: dict):
    return _check(sql, keys=key_facts(_snapshot(profile)),
                  visible={t: set(c) for t, c in profile.items()})


def _check_retail(sql: str):
    return _check_profile(sql, _RETAIL)


# Tables that join by no equi-key: a rates table beside the acme facts, and price tiers matched to
# line items by a quantity range.
_WITH_FX = {**_PROFILE, "fx": {"ccy": (10, 10, 0), "rate": (10, 8, 0), "valid_from": (10, 10, 0),
                               "valid_to": (10, 10, 0)}}
_WITH_TIERS = {**_RETAIL, "tiers": {"lo": (5, 5, 0), "hi": (5, 5, 0), "discount": (5, 5, 0)}}


# -- key facts ------------------------------------------------------------------------------------


def test_a_key_is_unique_when_every_non_null_value_is_distinct():
    snap = Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id="t.a", object_id="t", name="a", row_count=10, distinct_count=10, null_count=0),
        Column(id="t.b", object_id="t", name="b", row_count=10, distinct_count=7, null_count=3),
        Column(id="t.c", object_id="t", name="c", row_count=10, distinct_count=6, null_count=3),
    ])
    assert key_facts(snap) == {("t", "a"): True, ("t", "b"): True, ("t", "c"): False}


@pytest.mark.parametrize("counts", [
    {"row_count": None, "distinct_count": 10, "null_count": 0},
    {"row_count": 10, "distinct_count": None, "null_count": 0},
    {"row_count": 10, "distinct_count": 10, "null_count": None},
    {},
], ids=["row_count", "distinct_count", "null_count", "none_measured"])
def test_an_unmeasured_column_is_absent_not_guessed(counts):
    """`distinct_count=None` is what profiling writes for a column it could not count (a LOB, a
    user-defined type). Any one count missing leaves uniqueness undecided, and reading that as
    either answer would make the guard guess."""
    snap = Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id="t.a", object_id="t", name="a", **counts),
    ])
    assert key_facts(snap) == {}


# -- fires ------------------------------------------------------------------------------------------


def test_the_14b_loss_ratio_is_refused():
    """The query the local 14B wrote for loss ratio in every glossary arm: 1.034778 against a
    correct 0.247626, approved every time."""
    verdict = _check(
        "SELECT SUM(fact_claim.paid_amount_cents) / SUM(fact_premium.earned_premium_cents) "
        "AS loss_ratio FROM fact_claim JOIN fact_premium "
        "ON fact_claim.policy_id = fact_premium.policy_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT
    assert verdict.repairable


def test_a_duplicating_join_two_hops_away_is_still_found():
    """claim -> policy is many-to-one and harmless; policy -> fact_premium is where the claim rows
    repeat. The walk has to carry on through the harmless hop to find it."""
    verdict = _check(
        "SELECT dim_policy.region, SUM(fact_claim.paid_amount_cents) / "
        "SUM(fact_premium.earned_premium_cents) FROM fact_claim "
        "JOIN policy ON fact_claim.policy_id = policy.policy_id "
        "JOIN dim_policy ON policy.policy_id = dim_policy.policy_id "
        "JOIN fact_premium ON policy.policy_id = fact_premium.policy_id GROUP BY dim_policy.region"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_using_join_is_read_as_an_equi_join():
    verdict = _check(
        "SELECT p.region, SUM(f.paid_amount_cents) / SUM(fp.earned_premium_cents) "
        "FROM fact_claim f JOIN fact_premium fp USING (policy_id) "
        "JOIN dim_policy p USING (policy_id) GROUP BY 1"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_where_clause_join_is_read_as_an_equi_join():
    verdict = _check(
        "SELECT SUM(c.paid_amount_cents) FROM fact_claim c, fact_premium p "
        "WHERE c.policy_id = p.policy_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_conditional_sum_whose_value_comes_from_the_repeated_table_is_refused():
    verdict = _check(
        "SELECT SUM(CASE WHEN a.amount_component_type = 'loss_payment' THEN c.settlement_days "
        "ELSE 0 END) FROM claim c JOIN claim_amount a ON a.claim_id = c.claim_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_dimension_attribute_summed_alone_over_its_facts_is_refused():
    """A term with one table behaves as it did before terms: `p` repeats, and it is all there is."""
    verdict = _check_retail(
        "SELECT SUM(p.unit_price) FROM order_items li JOIN products p "
        "ON li.product_id = p.product_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_product_of_two_repeated_tables_is_refused():
    """Every table in the term repeats, so no grain counts each row once."""
    verdict = _check(
        "SELECT SUM(c.paid_amount_cents * p.earned_premium_cents) FROM fact_claim c "
        "JOIN fact_premium p ON c.policy_id = p.policy_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


@pytest.mark.parametrize("argument", [
    "li.quantity + p.unit_price",
    "li.quantity - p.unit_price",
    "(li.quantity + p.unit_price)",
    "IF(li.quantity > 10, li.quantity, p.unit_price)",
    "IF(li.quantity > 10, 1, 0) * p.unit_price",
    "CASE WHEN li.quantity > 10 THEN li.quantity ELSE p.unit_price END",
    "COALESCE(li.quantity + p.unit_price, 0)",
    "CAST(li.quantity + p.unit_price AS DOUBLE)",
    "-(li.quantity + p.unit_price)",
    "(li.quantity + p.unit_price) * 1.0",
    "(li.quantity + p.unit_price) / 2",
    "(li.quantity + 1) * p.unit_price",
    "(li.quantity - 1) * p.unit_price",
    "COALESCE(li.quantity, 1) * p.unit_price",
    "NULLIF(li.quantity + p.unit_price, 0)",
    "ROUND(li.quantity + p.unit_price, 2)",
])
def test_a_term_that_reads_only_the_repeated_table_is_refused_wherever_it_sits(argument):
    """Added, in a branch, under a function, or multiplied out -- `(li.quantity + 1) *
    p.unit_price` is `li.quantity * p.unit_price + p.unit_price` -- `p.unit_price` is summed on
    its own and repeats. Read as one term with `li`, each of these would pass as a line total."""
    verdict = _check_retail(
        f"SELECT SUM({argument}) FROM order_items li JOIN products p "
        "ON li.product_id = p.product_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


_CHASM = "FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id"


@pytest.mark.parametrize("joined", [
    f"{_CHASM} CROSS JOIN fx",
    "FROM fact_claim c, fact_premium p, fx WHERE c.policy_id = p.policy_id",
    f"{_CHASM} JOIN fx ON c.settlement_days BETWEEN fx.valid_from AND fx.valid_to",
    f"{_CHASM} JOIN fx ON fx.ccy = UPPER(CAST(c.claim_id AS VARCHAR))",
], ids=["cross", "comma", "range", "expression"])
def test_a_cross_joined_table_cannot_excuse_an_inflated_term(joined):
    """No walk marks `fx` repeated because no key reaches it -- which is not the same as `fx`
    appearing once per row. Only a table keyed to every other one, and repeated by none, can carry
    a term at its own grain; read as one, `fx` excused a claim total inflated by premiums."""
    verdict = _check_profile(f"SELECT SUM(c.paid_amount_cents * fx.rate) {joined}", _WITH_FX)
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_scalar_subquery_column_is_not_an_owner():
    """`rate` belongs to the subquery's scope. Read as an owner here it resolved to nothing, and an
    unresolved owner excuses the term."""
    verdict = _check_profile(
        f"SELECT SUM(c.paid_amount_cents * (SELECT MAX(rate) FROM fx)) {_CHASM}", _WITH_FX
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


_LINES = "FROM order_items li JOIN products p ON li.product_id = p.product_id"


@pytest.mark.parametrize("sql, profile", [
    ("SELECT SUM((SELECT MAX(x.rate) FROM fx x WHERE x.valid_from = c.settlement_days)) "
     f"{_CHASM}", _WITH_FX),
    ("SELECT SUM(li.quantity + (SELECT MAX(x.unit_price) FROM products x "
     f"WHERE x.product_id = p.product_id)) {_LINES}", _RETAIL),
    (f"SELECT SUM((SELECT MAX(t.discount) FROM tiers t WHERE t.lo = p.product_id)) {_LINES}",
     _WITH_TIERS),
], ids=["claim_rate", "line_price", "line_tier"])
def test_a_correlated_reference_inside_a_subquery_is_still_an_owner(sql, profile):
    """`c.settlement_days` inside the subquery is a value of the outer row, so the subquery repeats
    with `c`. Dropped with the subquery's own columns, it left the term ownerless and an inflated
    sum approved: 15 against a true 5 in DuckDB for the first case."""
    verdict = _check_profile(sql, profile)
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_shadowed_alias_inside_a_subquery_is_not_the_outer_table():
    """Inside the subquery `p` is `tiers`, not the outer `products`; read as the outer `p`, which
    repeats per line item, it would refuse a sum of one constant subquery value."""
    assert _check_profile(
        f"SELECT SUM((SELECT MAX(p.discount) FROM tiers p)) {_LINES}", _WITH_TIERS
    ) is None


def test_a_range_joined_tier_cannot_excuse_a_repeated_price():
    """`p` repeats per line item, and nothing shows the range-joined tier appears once per line
    item, so the tier cannot carry `p.unit_price` at a grain of its own."""
    verdict = _check_profile(
        "SELECT SUM(p.unit_price * t.discount) FROM order_items li JOIN products p "
        "ON li.product_id = p.product_id JOIN tiers t ON li.quantity BETWEEN t.lo AND t.hi",
        _WITH_TIERS,
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


@pytest.mark.parametrize("argument, refused", [
    ("p.unit_price / COALESCE(li.quantity, 1)", True),
    ("p.unit_price / CASE WHEN li.quantity > 0 THEN li.quantity ELSE 1 END", True),
    ("p.unit_price / NULLIF(li.quantity, 0)", False),
    ("p.unit_price / li.quantity", False),
    ("p.unit_price / (li.quantity + 1)", False),
])
def test_a_conditional_denominator_is_split_like_a_conditional_factor(argument, refused):
    """Where `li.quantity` is null, `p.unit_price / COALESCE(li.quantity, 1)` divides by 1 and sums
    `p.unit_price` alone -- as `p.unit_price * COALESCE(li.quantity, 1)` does, which was refused
    while the quotient was approved. A sum is not split: 1 / (a + b) is not 1/a + 1/b."""
    profile = {**_RETAIL, "order_items": {**_RETAIL["order_items"], "quantity": (1000, 20, 5)}}
    verdict = _check_profile(
        f"SELECT SUM({argument}) FROM order_items li JOIN products p "
        "ON li.product_id = p.product_id", profile,
    )
    assert (verdict is not None and verdict.code == RefusalCode.FAN_OUT) is refused


def test_a_capped_product_keeps_its_constant_term():
    """Seven `(li.quantity + 1)` factors pass the expansion cap. The fallback dropped the constant
    term, so `p.unit_price` alone -- refused with one factor -- merged into `li`'s terms and the
    same sum was approved. It keeps only a constant the product had: with none in any factor,
    every term still reads `li`."""
    def check(factor: str, n: int):
        factors = " * ".join([factor] * n)
        return _check_retail(
            f"SELECT SUM({factors} * p.unit_price) FROM order_items li "
            "JOIN products p ON li.product_id = p.product_id"
        )

    for n in (1, 7):
        verdict = check("(li.quantity + 1)", n)
        assert verdict is not None and verdict.code == RefusalCode.FAN_OUT, n
    assert check("(li.quantity + li.order_id)", 7) is None


def test_a_join_key_repeated_in_where_is_still_one_key():
    """Stated twice, the pair read as a two-column composite the profile cannot settle, and the
    check went silent."""
    verdict = _check(
        "SELECT SUM(c.paid_amount_cents) FROM fact_claim c JOIN fact_premium p "
        "ON c.policy_id = p.policy_id WHERE c.policy_id = p.policy_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_sum_over_the_asof_right_side_is_still_refused():
    """ASOF never repeats a left row, but one right row can meet many left rows: this premium is
    counted once per claim that matched it."""
    verdict = _check(
        "SELECT SUM(p.earned_premium_cents) FROM fact_claim c ASOF JOIN fact_premium p "
        "ON c.policy_id = p.policy_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_semi_join_leaves_the_count_over_a_chasm_refused():
    """The SEMI-joined table brings no row into the result. Kept as a table of the join, it could
    never be repeated, and a COUNT refused only when every table is could never fire past it."""
    verdict = _check(
        "SELECT COUNT(*) FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id "
        "SEMI JOIN dim_policy d ON d.policy_id = c.policy_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


_SCD = {
    "orders": {"order_id": (1000, 1000, 0), "customer_id": (1000, 200, 0),
               "amount": (1000, 900, 0), "order_date": (1000, 300, 0)},
    "dim_customer": {"customer_id": (600, 200, 0), "is_current": (600, 2, 0),
                     "valid_from": (600, 500, 0), "valid_to": (600, 400, 200),
                     "segment": (600, 4, 0)},
}


def _scd(sql):
    return _check(sql, keys=key_facts(_snapshot(_SCD)),
                  visible={t: set(c) for t, c in _SCD.items()})


_SCD_JOIN = "SELECT d.segment, SUM(o.amount) FROM orders o JOIN dim_customer d ON d.customer_id = o.customer_id"


_AS_OF = " AND o.order_date BETWEEN d.valid_from AND d.valid_to GROUP BY d.segment"


@pytest.mark.parametrize("narrowing", [
    _AS_OF,
    " WHERE o.amount > 0 AND (o.order_date BETWEEN d.valid_from AND d.valid_to) GROUP BY d.segment",
])
def test_an_as_of_join_to_a_versioned_table_makes_its_key_unknown(narrowing):
    """A type-2 dimension joined as of the fact's date is correct: one version per customer per
    order. Uniqueness is read table-wide and the profile cannot see the dates, so this was
    refused, and every repair that kept the join was refused again until the question deferred.
    A BETWEEN on two of the table's own columns, tested against another table's value, MAY leave
    one row per key; unknown, and unknown never fires."""
    assert _scd(_SCD_JOIN + narrowing) is None


@pytest.mark.parametrize("current_row", [
    " AND d.is_current GROUP BY d.segment",
    " AND d.is_current = 1 GROUP BY d.segment",
    " AND d.is_current IS TRUE GROUP BY d.segment",
    " AND d.valid_to IS NULL GROUP BY d.segment",
])
def test_a_current_row_filter_still_refuses_known_limit(current_row):
    """Correct on a type-2 dimension, and still refused. Syntax cannot tell `is_current` from an
    ordinary binary filter (`is_returned = 'N'`), nor `valid_to IS NULL` from `cancel_date IS
    NULL`; reading them as narrowing let an ordinary filter switch the guard off for an inflated
    sum -- a silent wrong number, where this is a visible deferral. The data tells them apart --
    key uniqueness per partition, profiled at enrichment (test_partition_uniqueness) -- but these
    facts carry no partitions, as an older snapshot does not: then it refuses. Pinned."""
    assert _scd(_SCD_JOIN + current_row) is not None


def test_a_two_valued_filter_on_the_only_repeating_table_still_refuses():
    """Line items repeat each product, and `is_returned = 'N'` keeps many of them per product:
    summing the product's price per kept line item is still inflated."""
    profile = {**_RETAIL, "order_items": {**_RETAIL["order_items"], "is_returned": (1000, 2, 0)}}
    assert _check_profile("SELECT SUM(p.unit_price) FROM order_items li JOIN products p "
                          "ON li.product_id = p.product_id WHERE li.is_returned = 'N'",
                          profile) is not None


def test_a_between_tested_against_the_same_table_does_not_narrow():
    """Only the as-of shape counts: bounds from the versioned table, value from another."""
    assert _scd(_SCD_JOIN + " WHERE d.valid_from BETWEEN d.valid_from AND d.valid_to "
                "GROUP BY d.segment") is not None


def test_a_filter_on_the_join_key_itself_does_not_narrow_it():
    """`d.customer_id = 7` pins the key, not the version: customer 7's history still repeats."""
    assert _scd(_SCD_JOIN + " WHERE d.customer_id = 7 GROUP BY d.segment") is not None


def test_a_range_filter_against_a_literal_does_not_narrow_a_key():
    """A period filter keeps a slice of rows per key, not one row per key: the loss ratio for
    premiums above a threshold still sums each claim once per matching premium."""
    assert _check(_LOSS_RATIO + " WHERE p.earned_premium_cents > 0") is not None


def test_a_filter_on_the_other_table_does_not_excuse_the_repeating_one():
    assert _scd(_SCD_JOIN + " WHERE o.amount = 5 GROUP BY d.segment") is not None


def test_a_count_over_a_chasm_is_refused():
    """Every table repeated: the count is of combinations, not of claims or of premiums."""
    verdict = _check(
        "SELECT COUNT(*) FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT
    assert "COUNT(DISTINCT" in verdict.message


def test_the_table_name_is_matched_ignoring_case():
    verdict = _check(
        "SELECT SUM(C.PAID_AMOUNT_CENTS) FROM FACT_CLAIM C JOIN FACT_PREMIUM P "
        "ON C.POLICY_ID = P.POLICY_ID"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_federated_catalog_qualified_name_resolves_to_its_object_id():
    """A federated deployment's object ids are `catalog.table`, and the model writes them that
    way. `object_key` is the one place a table node becomes an object id."""
    profile = {f"pg.{t}": cols for t, cols in _PROFILE.items()}
    visible = {t: set(cols) for t, cols in profile.items()}
    verdict = _check(
        "SELECT SUM(c.paid_amount_cents) FROM pg.fact_claim c JOIN pg.fact_premium p "
        "ON c.policy_id = p.policy_id",
        keys=key_facts(_snapshot(profile)), visible=visible,
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


# -- does not fire ----------------------------------------------------------------------------------


def test_the_frontier_loss_ratio_with_a_cte_per_fact_is_approved():
    """What the hosted model wrote with the same definition, and exactly the rewrite the refusal
    asks for. A derived source is opaque and treated as unique, so this has one base table per
    scope and nothing to walk."""
    assert _check(
        "WITH claim_totals AS (SELECT SUM(paid_amount_cents) AS paid FROM fact_claim), "
        "premium_totals AS (SELECT SUM(earned_premium_cents) AS earned FROM fact_premium) "
        "SELECT paid / NULLIF(earned, 0) FROM claim_totals CROSS JOIN premium_totals"
    ) is None


def test_a_per_key_cte_joined_back_is_approved():
    assert _check(
        "WITH prem AS (SELECT policy_id, SUM(earned_premium_cents) AS earned FROM fact_premium "
        "GROUP BY policy_id) SELECT SUM(c.paid_amount_cents) / SUM(prem.earned) "
        "FROM fact_claim c JOIN prem ON c.policy_id = prem.policy_id"
    ) is None


def test_a_plain_many_to_one_average_is_approved():
    """The negative control the first prototype failed: the policy row repeated per claim is
    policy's repetition, not claim's."""
    assert _check(
        "SELECT d.region, AVG(c.settlement_days) FROM claim c "
        "JOIN policy p ON c.policy_id = p.policy_id "
        "JOIN dim_policy d ON d.policy_id = p.policy_id GROUP BY 1"
    ) is None


def test_a_line_total_over_a_many_to_one_join_is_approved():
    """The canonical revenue query. Each product row meets many line items, so `p` repeats -- but
    the product is computed per line item, line items do not repeat, and the sum is right."""
    revenue = ("SELECT {}SUM(li.quantity * p.unit_price) FROM order_items li "
               "JOIN products p ON li.product_id = p.product_id{}")
    assert _check_retail(revenue.format("", "")) is None
    assert _check_retail(revenue.format("p.category, ", " GROUP BY p.category")) is None


@pytest.mark.parametrize("argument", [
    "li.quantity * (p.unit_price - 1)",
    "p.unit_price / (li.quantity + 1)",
    "ROUND(li.quantity * p.unit_price, 2)",
    "CAST(li.quantity AS DOUBLE) * p.unit_price",
    "COALESCE(li.quantity, 0) * p.unit_price",
    "CASE WHEN li.quantity > 10 THEN li.quantity ELSE 0 END * p.unit_price",
    "CASE WHEN li.quantity > 10 THEN li.quantity ELSE NULL END * p.unit_price",
])
def test_a_factor_that_does_not_repeat_reaches_every_term_it_multiplies(argument):
    """Every term here has `li` in it, so each is right at the line-item grain: a discount
    multiplied out, a denominator (which does not split), a function over a product, and a
    parameter, a zero or a NULL that is no summand of its own."""
    assert _check_retail(
        f"SELECT SUM({argument}) FROM order_items li "
        "JOIN products p ON li.product_id = p.product_id"
    ) is None


def test_a_deeply_nested_product_is_not_multiplied_out():
    """Sixteen binomials multiplied out are 65,536 terms; past a bound, a product falls back to
    one term per column -- which refuses what the rule did before terms."""
    argument = " * ".join(["(li.quantity + p.unit_price)"] * 16)
    assert len(value_terms(sqlglot.parse_one(argument, read="duckdb"))) < 1000
    verdict = _check_retail(
        f"SELECT SUM({argument}) FROM order_items li "
        "JOIN products p ON li.product_id = p.product_id"
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_term_with_an_owner_it_cannot_resolve_does_not_fire_known_limit():
    """Accepted limit: a column the check cannot place on a base table -- here a CTE's -- might be
    the grain that makes the term right, and unknown never fires. So a real fan-out scaled by a
    CTE column is approved. Pinned so a change here is a decision, not a drift."""
    assert _check(
        "WITH fx AS (SELECT 2 AS rate) SELECT SUM(c.paid_amount_cents * fx.rate) "
        "FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id CROSS JOIN fx"
    ) is None


def test_semi_anti_and_asof_joins_never_repeat_rows():
    """SEMI and ANTI keep or drop each left row once; ASOF matches at most one right row. Whatever
    the key, none of them repeats a left row."""
    for join in ("SEMI JOIN", "ANTI JOIN", "ASOF JOIN"):
        assert _check(
            f"SELECT SUM(c.paid_amount_cents) FROM fact_claim c {join} fact_premium p "
            "ON c.policy_id = p.policy_id"
        ) is None, join


def test_a_count_over_a_plain_one_to_many_is_approved():
    """It counts the finer table, which may be what was asked."""
    assert _check(
        "SELECT COUNT(*) FROM claim c JOIN claim_amount a ON a.claim_id = c.claim_id"
    ) is None


def test_a_conditional_sum_whose_condition_is_on_the_repeated_side_is_approved():
    """The shape that made the second prototype fire on 14.6% of BIRD's gold."""
    assert _check(
        "SELECT SUM(CASE WHEN p.region = 'east' THEN c.paid_amount_cents ELSE 0 END) "
        "FROM fact_claim c JOIN dim_policy p ON p.policy_id = c.policy_id"
    ) is None


def test_a_sqlite_iif_whose_condition_is_on_the_repeated_side_is_approved():
    """SQLite spells the conditional `IIF`, and it reaches the rule as the same IF node: its
    condition is never summed."""
    ast = sqlglot.parse_one(
        "SELECT SUM(IIF(p.region = 'east', c.paid_amount_cents, 0)) FROM fact_claim c "
        "JOIN dim_policy p ON p.policy_id = c.policy_id", read="sqlite",
    )
    assert check_fanout(ast, VISIBLE, KEYS) is None


def test_nullif_never_returns_its_second_argument():
    """`NULLIF(li.quantity, p.unit_price)` is `li.quantity` or NULL: the price is compared, and
    never summed."""
    assert _check_retail(
        "SELECT SUM(NULLIF(li.quantity, p.unit_price)) FROM order_items li "
        "JOIN products p ON li.product_id = p.product_id"
    ) is None


def test_a_deeply_nested_quotient_computes_each_denominator_once():
    """Recomputing the denominator for every numerator term cost width ** depth: this shape took
    seconds. Each denominator's alternatives are now computed once."""
    total = "(li.a0 + li.a1 + li.a2 + li.a3)"
    nested = total
    for _ in range(10):
        nested = f"{total} / ({nested})"
    assert _check_retail(
        f"SELECT SUM({nested}) FROM order_items li JOIN products p ON li.product_id = p.product_id"
    ) is None


def test_a_simple_case_operand_is_a_condition():
    """`CASE p.region WHEN ...` compares `p.region`; it never adds it up."""
    assert _check(
        "SELECT SUM(CASE p.region WHEN 'east' THEN c.paid_amount_cents ELSE 0 END) "
        "FROM fact_claim c JOIN dim_policy p ON p.policy_id = c.policy_id"
    ) is None


def test_a_sum_of_constants_over_a_one_to_many_is_a_count_and_is_approved():
    assert _check(
        "SELECT SUM(CASE WHEN a.amount_component_type = 'loss_payment' THEN 1 ELSE 0 END) "
        "FROM claim c JOIN claim_amount a ON a.claim_id = c.claim_id"
    ) is None


def test_min_max_and_distinct_aggregates_are_immune_to_repetition():
    for agg in ("MIN(c.paid_amount_cents)", "MAX(c.paid_amount_cents)",
                "COUNT(DISTINCT c.claim_id)", "SUM(DISTINCT c.paid_amount_cents)"):
        assert _check(
            f"SELECT {agg} FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id"
        ) is None, agg


def test_a_join_whose_key_was_never_profiled_does_not_fire():
    """Unknown is not non-unique. A snapshot without profiles must leave every query alone."""
    assert _check(
        "SELECT SUM(fact_claim.paid_amount_cents) FROM fact_claim JOIN fact_premium "
        "ON fact_claim.policy_id = fact_premium.policy_id",
        keys={},
    ) is None


def test_a_composite_key_the_profile_cannot_settle_does_not_fire():
    """Each column repeats, but the pair may be unique -- per-column profiles cannot tell."""
    profile = {"a": {"k1": (100, 50, 0), "k2": (100, 20, 0), "v": (100, 90, 0)},
               "b": {"k1": (100, 50, 0), "k2": (100, 20, 0), "w": (100, 90, 0)}}
    visible = {t: set(c) for t, c in profile.items()}
    assert _check(
        "SELECT SUM(a.v) FROM a JOIN b ON a.k1 = b.k1 AND a.k2 = b.k2",
        keys=key_facts(_snapshot(profile)), visible=visible,
    ) is None


def test_an_equality_inside_a_subquery_is_not_a_join_of_the_outer_scope():
    assert _check(
        "SELECT SUM(c.paid_amount_cents) FROM fact_claim c JOIN dim_policy d "
        "ON c.policy_id = d.policy_id WHERE EXISTS (SELECT 1 FROM fact_premium p "
        "WHERE p.policy_id = c.policy_id)"
    ) is None


def test_an_aggregate_inside_a_scalar_subquery_belongs_to_the_subquery():
    """The outer SELECT aggregates nothing; the SUM is over `fact_claim` alone, in its own
    scope. Charged to the outer scope, where `fact_claim` repeats per premium row, it would
    refuse a per-claim share of the total."""
    assert _check(
        "SELECT c.claim_id, c.paid_amount_cents * 1.0 / (SELECT SUM(paid_amount_cents) "
        "FROM fact_claim) FROM fact_claim c JOIN dim_policy d ON d.policy_id = c.policy_id "
        "JOIN fact_premium p ON p.policy_id = c.policy_id"
    ) is None


def test_a_subquery_alias_that_shadows_an_outer_table_adds_no_join():
    """Inside the EXISTS, `b` is `x`, not the outer `b`. Reading `b.k = a.k` as a join of the outer
    scope would invent a repeating a -> b edge and refuse a query that multiplies nothing."""
    profile = {"a": {"id": (100, 100, 0), "k": (100, 10, 0), "v": (100, 90, 0)},
               "b": {"id": (100, 100, 0), "k": (100, 10, 0)},
               "c": {"id": (100, 100, 0)},
               "x": {"k": (100, 10, 0)}}
    visible = {t: set(c) for t, c in profile.items()}
    assert _check(
        "SELECT SUM(a.v) FROM a JOIN c ON a.id = c.id JOIN b ON b.id = c.id "
        "WHERE EXISTS (SELECT 1 FROM x b WHERE b.k = a.k)",
        keys=key_facts(_snapshot(profile)), visible=visible,
    ) is None


# -- the message ------------------------------------------------------------------------------------


def test_the_message_names_the_tables_the_key_and_the_restructure():
    verdict = _check(
        "SELECT SUM(fact_claim.paid_amount_cents) / SUM(fact_premium.earned_premium_cents) "
        "FROM fact_claim JOIN fact_premium ON fact_claim.policy_id = fact_premium.policy_id"
    )
    for needed in ("fact_claim", "fact_premium", "policy_id", "CTE"):
        assert needed in verdict.message, needed


_LOSS_RATIO = (
    "SELECT SUM(c.paid_amount_cents) / SUM(p.earned_premium_cents) "
    "FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id"
)


def test_an_overall_total_is_told_to_total_each_table_and_cross_join():
    """The first message said "grouped by the key you join or group on". The local 14B did exactly
    that: one CTE per fact grouped by policy_id, inner-joined on policy_id -- which drops every
    policy with premium but no claim, and scored 0 of 8 repairs. An overall answer has no grain
    but the whole table, so the message has to say so."""
    message = _check(_LOSS_RATIO).message
    assert "no GROUP BY" in message
    assert "CROSS JOIN" in message
    assert "not by the join key (policy_id)" in message


_BY_REGION = (_LOSS_RATIO.replace("FROM", ", d.region FROM", 1)
              + " JOIN dim_policy d ON d.policy_id = c.policy_id GROUP BY ")


def test_a_grouped_answer_is_told_to_aggregate_by_its_own_group_columns():
    """And to combine the groups with a FULL OUTER JOIN: an inner join of per-region aggregates
    drops a region with premium but no claims -- the same loss, one grain up."""
    message = _check(_BY_REGION + "d.region").message
    assert "(d.region)" in message
    assert "FULL OUTER JOIN" in message
    assert "CROSS JOIN" not in message
    assert "not by the join key (policy_id)" in message


def test_grouped_results_are_joined_null_safely():
    """GROUP BY puts NULL keys in one group; `=` never matches NULL to NULL, so a FULL OUTER JOIN
    on plain equality returns a NULL region as two half-rows, each with half the ratio missing."""
    message = _check(_BY_REGION + "d.region").message
    assert "IS NOT DISTINCT FROM" in message
    assert "COALESCE" in message


@pytest.mark.parametrize("group_by", ["ALL", "ROLLUP (d.region)", "CUBE (d.region)",
                                      "GROUPING SETS ((d.region), ())"])
def test_every_grouping_form_gets_grouped_advice(group_by):
    """Reading only plain GROUP BY expressions sent these to the one-overall-total advice, and a
    repair that follows it returns one figure where the question asked for one per region --
    a wrong answer that no longer fans out and so passes this check."""
    message = _check(_BY_REGION + group_by).message
    assert "no GROUP BY" not in message and "CROSS JOIN" not in message
    assert "d.region" in message


@pytest.mark.parametrize("form", ["ROLLUP (d.region)", "CUBE (d.region)",
                                  "GROUPING SETS ((d.region), ())"])
def test_a_subtotal_grouping_is_kept_whole_and_matched_by_level(form):
    """Flattening ROLLUP to its columns told the model to group by d.region alone, and a repair
    that follows it drops the grand-total row -- a changed answer that no longer fans out. Keep the
    query's own grouping in each part, and match rows by level as well as by value: a subtotal row
    and a real NULL region are both NULL in d.region, and only GROUPING() tells them apart."""
    message = _check(_BY_REGION + form).message
    assert form.split(" (")[0] in message
    assert "GROUPING(d.region)" in message


def test_an_answer_grouped_by_the_join_key_is_not_told_to_avoid_it():
    """Per policy, the key IS the answer's grain; "not by the join key" would contradict the
    group it just named. What still holds is how to combine them."""
    message = _check("SELECT c.policy_id, SUM(c.paid_amount_cents) / SUM(p.earned_premium_cents) "
                     "FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id "
                     "GROUP BY c.policy_id").message
    assert "not by the join key" not in message
    assert "FULL OUTER JOIN" in message


def test_a_positional_group_by_is_named_by_its_column():
    message = _check("SELECT d.region, SUM(c.paid_amount_cents) / SUM(p.earned_premium_cents) "
                     "FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id "
                     "JOIN dim_policy d ON d.policy_id = c.policy_id GROUP BY 1").message
    assert "(d.region)" in message


def test_the_message_warns_what_an_inner_join_of_per_key_aggregates_loses():
    assert "appears in only one table" in _check(_LOSS_RATIO).message


def test_the_message_keeps_a_cohort_only_when_the_question_asks_for_one():
    """An inner join can mean "only products that had returns". Whole-table totals, CROSS JOINed,
    silently widen that cohort; the model is told to keep the restriction, with EXISTS, only
    when the question itself sets it -- the loss ratio's inner join was an accident, not a filter."""
    message = _check(_LOSS_RATIO).message
    assert "EXISTS" in message and "only if the question itself" in message


def test_a_window_aggregate_is_told_to_keep_its_partition():
    """SUM() OVER (PARTITION BY ...) got the one-overall-total advice, which discards the
    partition and the row-level output the query asked for."""
    message = _check("SELECT c.claim_id, SUM(c.paid_amount_cents) OVER (PARTITION BY c.policy_id) "
                     "FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id").message
    assert "PARTITION BY" in message
    assert "no GROUP BY" not in message and "CROSS JOIN" not in message


def test_the_message_quotes_no_profile_number():
    """The profile is taken with no row filter, and this runs before the RLS rewrite -- the same
    position `check_values` is in (M5). The counts are never quoted. The refusal does disclose one
    bit: whether the key repeats table-wide, which for a row-filtered identity includes rows
    outside its slice. That is the posture `check_values` holds under M5, accepted because the
    key's name is already in the identity's own query."""
    verdict = _check(
        "SELECT SUM(fact_claim.paid_amount_cents) / SUM(fact_premium.earned_premium_cents) "
        "FROM fact_claim JOIN fact_premium ON fact_claim.policy_id = fact_premium.policy_id"
    )
    for number in ("30000", "23322", "125000", "50000"):
        assert number not in verdict.message, number


# -- a derived table that only projects a join ---------------------------------------------------

_JOINED_ROWS = ("SELECT c.paid_amount_cents AS paid, p.earned_premium_cents AS earned "
                "FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id")


@pytest.mark.parametrize("sql", [
    f"SELECT SUM(t.paid) / SUM(t.earned) FROM ({_JOINED_ROWS}) t",
    f"WITH j AS ({_JOINED_ROWS}) SELECT SUM(paid) / SUM(earned) FROM j",
])
def test_a_fan_out_wrapped_in_a_projecting_subquery_is_still_refused(sql):
    """Derived sources are opaque, so wrapping the fan-out join in a subquery that only passes its
    rows through got an inflated sum approved -- a bypass, and the same opacity hid BIRD gold that
    computes the inflation that way. A derived table with no aggregate, DISTINCT, window or LIMIT
    is merged into the outer query (sqlglot's merge_subqueries) and the flat form is checked."""
    verdict = _check(sql)
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_distinct_subquery_is_not_merged():
    """DISTINCT restores the claim grain, so the sum over it is not inflated."""
    assert _check("SELECT SUM(t.paid) FROM (SELECT DISTINCT c.claim_id, c.paid_amount_cents AS paid "
                  "FROM fact_claim c JOIN fact_premium p ON c.policy_id = p.policy_id) t") is None


def test_a_projecting_subquery_over_a_harmless_join_is_approved():
    assert _check("SELECT t.region, SUM(t.paid) FROM (SELECT d.region, c.paid_amount_cents AS paid "
                  "FROM fact_claim c JOIN dim_policy d ON d.policy_id = c.policy_id) t "
                  "GROUP BY t.region") is None


def test_a_federated_projecting_subquery_is_merged_too():
    profile = {f"pg.{t}": cols for t, cols in _PROFILE.items()}
    visible = {t: set(cols) for t, cols in profile.items()}
    verdict = _check(
        "SELECT SUM(t.paid) FROM (SELECT c.paid_amount_cents AS paid FROM pg.fact_claim c "
        "JOIN pg.fact_premium p ON c.policy_id = p.policy_id) t",
        keys=key_facts(_snapshot(profile)), visible=visible,
    )
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_mixed_depth_catalog_still_merges():
    """A federated catalog can mix bare and qualified object ids; sqlglot's schema rejects mixed
    depths, and that turned the merge off for every query against the catalog."""
    visible = {**VISIBLE, "pg.other": {"x"}}
    verdict = _check(f"SELECT SUM(t.paid) / SUM(t.earned) FROM ({_JOINED_ROWS}) t", visible=visible)
    assert verdict is not None and verdict.code == RefusalCode.FAN_OUT


def test_a_cte_read_twice_is_not_merged_known_limit():
    """sqlglot merges a CTE only when it is read once, so a pass-through CTE read twice stays
    opaque and its inflated sum is approved. Narrowed, not closed; pinned so a change is a choice."""
    assert _check(f"WITH j AS ({_JOINED_ROWS}) SELECT SUM(paid) FROM j "
                  "UNION ALL SELECT SUM(paid) FROM j") is None
