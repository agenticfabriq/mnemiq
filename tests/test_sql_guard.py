import pytest
from sqlglot import exp

from mnemiq.sql.guard import MAX_ROWS, check_shape
from mnemiq.sql.verdict import Refusal, RefusalCode


def _ok(sql, **kw):
    result = check_shape(sql, **kw)
    assert not isinstance(result, Refusal), result
    return result


def _refused(sql) -> Refusal:
    result = check_shape(sql)
    assert isinstance(result, Refusal), f"should have been refused: {sql}"
    return result


def test_a_plain_select_passes():
    ast = _ok("SELECT claim_identifier FROM claim")
    assert isinstance(ast, exp.Select)


def test_a_missing_limit_is_injected_not_requested():
    ast = _ok("SELECT a FROM claim")
    assert ast.sql(dialect="duckdb").endswith(f"LIMIT {MAX_ROWS}")


def test_an_existing_smaller_limit_is_kept():
    ast = _ok("SELECT a FROM claim LIMIT 5")
    assert ast.sql(dialect="duckdb").endswith("LIMIT 5")


def test_an_oversized_limit_is_clamped():
    ast = _ok("SELECT a FROM claim LIMIT 999999")
    assert ast.sql(dialect="duckdb").endswith(f"LIMIT {MAX_ROWS}")


def test_ctes_and_unions_are_allowed():
    _ok("WITH x AS (SELECT a FROM claim) SELECT a FROM x")
    _ok("SELECT a FROM claim UNION SELECT b FROM policy")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE claim",
        "INSERT INTO claim VALUES (1)",
        "UPDATE claim SET a = 1",
        "DELETE FROM claim",
        "CREATE TABLE t (a INT)",
        "ATTACH 'evil.db'",
    ],
)
def test_anything_that_is_not_a_select_is_refused(sql):
    assert _refused(sql).code == RefusalCode.NOT_SELECT_ONLY


def test_a_trailing_statement_cannot_smuggle_a_drop():
    # sqlglot.parse_one does NOT raise here -- it returns a Block. "It parsed" is not safety.
    refusal = _refused("SELECT a FROM claim; DROP TABLE claim")
    assert refusal.code == RefusalCode.NOT_A_SINGLE_STATEMENT


def test_select_star_is_refused():
    # a star yields no Column nodes at all, so column-level authorization would silently
    # pass over it. The engine must know exactly which columns it returns.
    assert _refused("SELECT * FROM claim").code == RefusalCode.SELECT_STAR
    assert _refused("SELECT c.* FROM claim c").code == RefusalCode.SELECT_STAR


def test_an_inner_star_is_allowed_because_the_output_columns_are_still_explicit():
    # what must be knowable is the set of columns leaving the engine. An EXISTS discards its
    # projection, and a star inside a derived table is bounded by the explicit projection
    # that wraps it. Rejecting these would refuse 22 of the 99 TPC-DS queries.
    _ok("SELECT a FROM claim WHERE EXISTS (SELECT * FROM policy WHERE policy.a = claim.a)")
    _ok("SELECT a FROM (SELECT * FROM claim)")
    _ok("WITH x AS (SELECT * FROM claim) SELECT a FROM x")


def test_a_star_over_a_derived_table_is_allowed_its_columns_are_known():
    # the inner projection is explicit, so the output columns are fully determined
    _ok("SELECT * FROM (SELECT claim_identifier, status FROM claim)")
    _ok("WITH x AS (SELECT a FROM claim) SELECT * FROM x")


def test_a_star_over_a_base_table_is_refused_even_behind_a_join():
    # `claim` is a real table: its column set is unbounded, and we would not know what we
    # are returning -- nor could column-level authorization check it
    assert _refused("SELECT c.* FROM claim c JOIN (SELECT a FROM policy) x ON TRUE").code == (
        RefusalCode.SELECT_STAR
    )


def test_a_star_in_a_union_branch_is_still_refused():
    # every branch of a union returns columns to the caller
    assert _refused("SELECT a FROM claim UNION SELECT * FROM policy").code == (
        RefusalCode.SELECT_STAR
    )


def test_count_star_is_not_a_select_star():
    # COUNT(*) contains an exp.Star. A naive find_all(exp.Star) would reject the most common
    # analytics query there is, and the engine could not count anything.
    _ok("SELECT count(*) FROM claim")
    _ok("SELECT status, count(*) AS n FROM claim GROUP BY status")
    _ok("SELECT count(*) FROM claim WHERE status IN (SELECT status FROM policy)")


def test_unparseable_sql_is_refused_not_raised():
    assert _refused("this is not sql !!").code == RefusalCode.PARSE_ERROR


def test_a_refusal_message_tells_the_model_how_to_repair():
    assert "explicit" in _refused("SELECT * FROM claim").message.lower()
