import sqlglot

from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.verdict import RefusalCode

VISIBLE = {
    "claim": {"claim_identifier", "status", "party_identifier"},
    "policy": {"policy_identifier", "premium"},
}


def _check(sql):
    return check_access(sqlglot.parse_one(sql, read="duckdb"), VISIBLE)


def test_an_authorized_query_passes():
    assert _check("SELECT claim_identifier FROM claim") is None


def test_a_join_across_authorized_tables_passes():
    sql = (
        "SELECT c.status, p.premium FROM claim c "
        "JOIN policy p ON c.party_identifier = p.policy_identifier"
    )
    assert _check(sql) is None


def test_a_table_that_was_never_retrieved_is_rejected():
    # the model knows `person` from training, not from us. Retrieval scoping did not stop it;
    # this must.
    refusal = _check("SELECT last_name FROM person")
    assert refusal.code == RefusalCode.UNAUTHORIZED_TABLE
    assert refusal.subject == "person"


def test_an_unauthorized_table_hidden_inside_a_join_is_rejected():
    sql = "SELECT c.status FROM claim c JOIN person p ON c.party_identifier = p.person_identifier"
    assert _check(sql).code == RefusalCode.UNAUTHORIZED_TABLE


def test_an_unauthorized_table_hidden_in_a_subquery_is_rejected():
    sql = (
        "SELECT status FROM claim "
        "WHERE party_identifier IN (SELECT person_identifier FROM person)"
    )
    assert _check(sql).code == RefusalCode.UNAUTHORIZED_TABLE


def test_a_cte_name_is_not_mistaken_for_a_table():
    # sqlglot reports the CTE alias in find_all(exp.Table): naive checking rejects valid SQL
    sql = "WITH recent AS (SELECT status FROM claim) SELECT status FROM recent"
    assert _check(sql) is None


def test_a_cte_cannot_launder_an_unauthorized_table():
    sql = "WITH sneaky AS (SELECT last_name FROM person) SELECT last_name FROM sneaky"
    assert _check(sql).code == RefusalCode.UNAUTHORIZED_TABLE


def test_a_hallucinated_column_is_rejected():
    refusal = _check("SELECT claim_total FROM claim")
    assert refusal.code == RefusalCode.UNKNOWN_COLUMN
    assert refusal.subject == "claim_total"


def test_a_column_qualified_by_an_alias_resolves():
    assert _check("SELECT c.status FROM claim c") is None
    assert _check("SELECT c.nope FROM claim c").code == RefusalCode.UNKNOWN_COLUMN


def test_an_unqualified_column_must_exist_in_some_referenced_table():
    assert _check("SELECT premium FROM claim JOIN policy ON TRUE") is None
    assert (
        _check("SELECT salary FROM claim JOIN policy ON TRUE").code == RefusalCode.UNKNOWN_COLUMN
    )
