import sqlglot

from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import apply_row_and_mask
from mnemiq.sql.verdict import Refusal, RefusalCode

_VISIBLE = {"claim": {"id", "amount", "ssn"}}


def _sql(ast):
    return ast.sql(dialect="duckdb")


def test_row_filter_is_injected_at_the_source():
    ast = sqlglot.parse_one("SELECT id, amount FROM claim", read="duckdb")
    out = apply_row_and_mask(ast, AccessPolicy(row_filters={"claim": "amount > 0"}),
                             _VISIBLE, dialect="duckdb")
    text = _sql(out).lower()
    assert "amount > 0" in text and "from claim" in text
    assert "select *" not in text  # explicit column list, never a star


def test_masked_column_becomes_null_at_the_source():
    ast = sqlglot.parse_one("SELECT id, ssn FROM claim", read="duckdb")
    out = apply_row_and_mask(ast, AccessPolicy(masked={("claim", "ssn")}), _VISIBLE, "duckdb")
    text = _sql(out).lower()
    assert "null as ssn" in text  # masked at the source, raw value never fetched


def test_invalid_filter_is_refused():
    ast = sqlglot.parse_one("SELECT id FROM claim", read="duckdb")
    out = apply_row_and_mask(ast, AccessPolicy(row_filters={"claim": "nonexistent > 0"}),
                             _VISIBLE, "duckdb")
    assert isinstance(out, Refusal) and out.code == RefusalCode.INVALID_ROW_FILTER


def test_untouched_when_no_filter_or_mask():
    ast = sqlglot.parse_one("SELECT id FROM claim", read="duckdb")
    before = _sql(ast)
    out = apply_row_and_mask(ast, AccessPolicy(), _VISIBLE, "duckdb")
    assert _sql(out) == before
