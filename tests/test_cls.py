import sqlglot

from mnemiq.sql.cls import check_cls
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal, RefusalCode


def _ast(sql):
    return sqlglot.parse_one(sql, read="duckdb")


_POL = AccessPolicy(denied={("claim", "secret")}, masked={("claim", "ssn")})


def test_denied_column_referenced_is_refused():
    r = check_cls(_ast("SELECT id, secret FROM claim"), _POL)
    assert isinstance(r, Refusal) and r.code == RefusalCode.UNAUTHORIZED_COLUMN


def test_masked_column_as_bare_projection_is_allowed():
    assert check_cls(_ast("SELECT id, ssn FROM claim"), _POL) is None
    assert check_cls(_ast("SELECT ssn AS s FROM claim"), _POL) is None


def test_masked_column_in_predicate_is_refused():
    for sql in (
        "SELECT id FROM claim WHERE ssn = 'x'",
        "SELECT count(ssn) FROM claim",
        "SELECT id FROM claim ORDER BY ssn",
        "SELECT id FROM claim GROUP BY ssn",
    ):
        r = check_cls(_ast(sql), _POL)
        assert isinstance(r, Refusal) and r.code == RefusalCode.MASKED_COLUMN_IN_PREDICATE, sql


def test_empty_policy_allows_everything():
    assert check_cls(_ast("SELECT ssn FROM claim WHERE secret = 1"), AccessPolicy()) is None
