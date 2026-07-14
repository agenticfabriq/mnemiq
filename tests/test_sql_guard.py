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
