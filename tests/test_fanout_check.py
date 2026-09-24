"""The fan-out check (register M109): refuse an aggregate whose rows a join has multiplied.

Every fixture below is one of the eleven controls the rule was validated against before any
number was read (mnemiq-internal `evals/fanout-guard/validate.py`), restated against a hand-built
profile instead of a live database, plus the edges the engine port adds.
"""
from __future__ import annotations

import sqlglot

from mnemiq.contract import Column, Snapshot
from mnemiq.sql.fanout_check import check_fanout, key_facts
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


# -- key facts ------------------------------------------------------------------------------------


def test_a_key_is_unique_when_every_non_null_value_is_distinct():
    snap = Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id="t.a", object_id="t", name="a", row_count=10, distinct_count=10, null_count=0),
        Column(id="t.b", object_id="t", name="b", row_count=10, distinct_count=7, null_count=3),
        Column(id="t.c", object_id="t", name="c", row_count=10, distinct_count=6, null_count=3),
    ])
    assert key_facts(snap) == {("t", "a"): True, ("t", "b"): True, ("t", "c"): False}


def test_an_unmeasured_column_is_absent_not_guessed():
    """`distinct_count=None` is what profiling writes for a column it could not count (a LOB, a
    user-defined type). Reading that as either answer would make the guard guess."""
    snap = Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id="t.a", object_id="t", name="a", row_count=10, distinct_count=None,
               null_count=None),
        Column(id="t.b", object_id="t", name="b"),
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


# -- the message ------------------------------------------------------------------------------------


def test_the_message_names_the_tables_the_key_and_the_restructure():
    verdict = _check(
        "SELECT SUM(fact_claim.paid_amount_cents) / SUM(fact_premium.earned_premium_cents) "
        "FROM fact_claim JOIN fact_premium ON fact_claim.policy_id = fact_premium.policy_id"
    )
    for needed in ("fact_claim", "fact_premium", "policy_id", "CTE"):
        assert needed in verdict.message, needed


def test_the_message_quotes_no_profile_number():
    """The profile is taken with no row filter, and this runs before the RLS rewrite -- the same
    position `check_values` is in (M5). Uniqueness is disclosed as a yes/no about a key the
    identity can already see; the counts behind it are not."""
    verdict = _check(
        "SELECT SUM(fact_claim.paid_amount_cents) / SUM(fact_premium.earned_premium_cents) "
        "FROM fact_claim JOIN fact_premium ON fact_claim.policy_id = fact_premium.policy_id"
    )
    for number in ("30000", "23322", "125000", "50000"):
        assert number not in verdict.message, number
