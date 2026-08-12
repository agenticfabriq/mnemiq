"""M32: the questions the empty screen offers must come from the data in front of you.

They also have to obey the same boundary the cards do. A starter names a table and a
column, so an unscoped one is a metadata leak wearing an invitation -- the M4 channel,
reopened at the friendliest point in the product.
"""

from __future__ import annotations

from mnemiq.authz.grants import GrantSet
from mnemiq.contract.semantic import CodedValue, Column, Example, Snapshot
from mnemiq.semantic.starters import compose_starters
from mnemiq.sql.policy import AccessPolicy


def col(object_id, name, semantic_type=None, rows=100, codes=0, pii=None):
    return Column(
        id=f"{object_id}.{name}",
        object_id=object_id,
        name=name,
        semantic_type=semantic_type,
        pii_level=pii,
        row_count=rows,
        coded_values=[CodedValue(code=str(i), meaning=str(i)) for i in range(codes)],
    )


def snap(columns, examples=()):
    return Snapshot(
        version="v1",
        source_id="s",
        created_at="2026-01-01T00:00:00Z",
        columns=list(columns),
        examples=list(examples),
    )


def grants(*objects):
    return GrantSet(objects=frozenset(objects))


AMOUNT = col("payment", "amount", "amount", rows=16049)
RATING = col("film", "rating", "code", rows=1000, codes=5)


def test_it_composes_an_aggregate_and_a_grouping_from_the_real_columns():
    out = compose_starters(snap([AMOUNT, RATING]), grants("payment", "film"), AccessPolicy())
    assert any("amount" in q and "payment" in q for q in out), out
    assert any("rating" in q and "film" in q for q in out), out


def test_a_source_with_nothing_to_offer_gets_no_starters():
    """Three plausible questions about the wrong database is worse than none."""
    only_ids = [col("t", "t_id", "identifier")]
    assert compose_starters(snap(only_ids), grants("t"), AccessPolicy()) == []


def test_it_is_deterministic():
    s, g = snap([AMOUNT, RATING]), grants("payment", "film")
    assert compose_starters(s, g, AccessPolicy()) == compose_starters(s, g, AccessPolicy())


def test_the_biggest_table_wins_so_the_starter_is_about_the_real_subject():
    small = col("draft", "amount", "amount", rows=3)
    out = compose_starters(snap([small, AMOUNT]), grants("draft", "payment"), AccessPolicy())
    assert any("payment" in q for q in out) and not any("draft" in q for q in out), out


# --- the boundary ---------------------------------------------------------------


def test_an_ungranted_table_is_never_named():
    out = compose_starters(snap([AMOUNT, RATING]), grants("film"), AccessPolicy())
    assert not any("payment" in q for q in out), out


def test_a_denied_column_is_never_named():
    salary = col("staff", "salary", "amount", rows=99999)
    policy = AccessPolicy(denied={("staff", "salary")})
    out = compose_starters(snap([salary, AMOUNT]), grants("staff", "payment"), policy)
    assert not any("salary" in q for q in out), out


def test_a_masked_column_is_never_named():
    """A masked column reads as NULL, so offering it would demo an empty answer."""
    salary = col("staff", "salary", "amount", rows=99999)
    policy = AccessPolicy(masked={("staff", "salary")})
    out = compose_starters(snap([salary, AMOUNT]), grants("staff", "payment"), policy)
    assert not any("salary" in q for q in out), out


# --- what counts as a dimension -------------------------------------------------


def test_a_single_valued_coded_column_is_not_a_grouping():
    one = col("t", "last_update", "timestamp", rows=500, codes=1)
    assert compose_starters(snap([one]), grants("t"), AccessPolicy()) == []


def test_a_timestamp_that_happens_to_be_low_cardinality_is_not_a_grouping():
    """`last_update` gets harvested as codes because the rows share a value, not because
    it is a dimension."""
    stamp = col("t", "last_update", "timestamp", rows=500, codes=3)
    assert compose_starters(snap([stamp]), grants("t"), AccessPolicy()) == []


def test_a_high_cardinality_coded_column_is_an_identifier_in_disguise():
    many = col("t", "sku", "code", rows=500, codes=400)
    assert compose_starters(snap([many]), grants("t"), AccessPolicy()) == []


# --- curated examples -----------------------------------------------------------


def test_a_curated_example_is_preferred_over_a_composed_one():
    ex = Example(question="Which film category earns the most?", sql="SELECT 1",
                 tables=["film", "payment"], object_id="film")
    out = compose_starters(snap([AMOUNT, RATING], [ex]), grants("payment", "film"), AccessPolicy())
    assert out[0] == "Which film category earns the most?"


def test_a_curated_example_naming_an_ungranted_table_is_dropped():
    ex = Example(question="How much did each staff member take?", sql="SELECT 1",
                 tables=["payment", "staff"], object_id="payment")
    out = compose_starters(snap([AMOUNT], [ex]), grants("payment"), AccessPolicy())
    assert ex.question not in out, out


# --- the two starters should show two sides of the source ------------------------


def test_the_grouping_prefers_a_table_the_aggregate_did_not_already_use():
    """Two questions about one table show the source once."""
    same = col("payment", "method", "code", rows=16049, codes=4)
    other = col("film", "rating", "code", rows=1000, codes=5)
    out = compose_starters(
        snap([AMOUNT, same, other]), grants("payment", "film"), AccessPolicy()
    )
    assert any("film" in q for q in out), out


def test_but_it_falls_back_rather_than_drop_the_starter():
    """One table with both is still worth two questions."""
    same = col("payment", "method", "code", rows=16049, codes=4)
    out = compose_starters(snap([AMOUNT, same]), grants("payment"), AccessPolicy())
    assert len(out) == 2 and any("method" in q for q in out), out
