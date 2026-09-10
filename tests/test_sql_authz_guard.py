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


def test_a_group_by_on_a_select_alias_is_valid_sql():
    # the most natural analytics query there is. GROUP BY / ORDER BY / HAVING may reference a
    # SELECT alias -- rejecting it would leave the engine unable to answer "count X by year".
    sql = (
        "SELECT status AS s, count(*) AS n FROM claim "
        "GROUP BY s HAVING n > 1 ORDER BY n DESC"
    )
    assert _check(sql) is None


def test_an_alias_cannot_launder_an_unknown_column():
    # the alias is fine; the column it is built from is still checked
    assert _check("SELECT nonexistent AS s FROM claim GROUP BY s").code == (
        RefusalCode.UNKNOWN_COLUMN
    )


def test_an_unqualified_column_must_exist_in_some_referenced_table():
    assert _check("SELECT premium FROM claim JOIN policy ON TRUE") is None
    assert (
        _check("SELECT salary FROM claim JOIN policy ON TRUE").code == RefusalCode.UNKNOWN_COLUMN
    )


# --- M88: the decider already knows the schema, so the guard stops guessing ---------------------


def test_decide_hands_the_guard_the_schema_it_already_has():
    """`decide` takes `visible` -- table to columns -- and passes it to `check_access` two lines
    below. The shape check was guessing about the same names at the same moment.

    ACME has two tables whose amount column shares the table's name, so `WHERE claim_amount > 10`
    was refused as a whole-row reference. Nothing about that is ambiguous once the columns are
    known: DuckDB resolves the name to the COLUMN when one exists.
    """
    from mnemiq.sql.decide import decide
    from mnemiq.sql.verdict import Refusal

    visible = {"claim_amount": {"claim_amount", "id"}}
    verdict = decide("SELECT id FROM claim_amount WHERE claim_amount > 10", visible)
    assert not isinstance(verdict, Refusal), getattr(verdict, "message", verdict)


def test_decide_still_refuses_a_real_whole_row_reference():
    """The same call, on a table with no column of that name, is the row and stays refused."""
    from mnemiq.sql.decide import decide
    from mnemiq.sql.verdict import Refusal, RefusalCode

    visible = {"claim": {"claim_identifier", "ssn"}}
    for sql in (
        "SELECT claim FROM claim",
        "SELECT claim_identifier FROM claim WHERE claim['ssn'] = 'x'",
    ):
        verdict = decide(sql, visible)
        assert isinstance(verdict, Refusal), sql
        assert verdict.code == RefusalCode.SELECT_STAR, sql


def test_a_stale_snapshot_is_refused_before_execution_not_leaked():
    """M90, end to end. A catalog that has drifted costs a refusal, not a denied column.

    The snapshot claims `claim` has a column of that name and the live table does not. The guard
    qualifies the reference on the schema's word, the binder rejects what it cannot resolve, and
    `prove` turns that into a refusal before anything runs. Without the rewrite the bare name bound
    to the row struct and returned `ssn` -- and `prove` could not tell the difference, because a
    query that resolves to a row EXPLAINs perfectly well.
    """
    import duckdb

    from mnemiq.sql.decide import decide
    from mnemiq.sql.policy import AccessPolicy
    from mnemiq.sql.verdict import Refusal, RefusalCode

    class _Adapter:
        """`execute`, because that is what `prove` calls -- `validate` first, else `execute`.

        The first version of this double exposed `explain`, so every query raised AttributeError
        and came back EXPLAIN_FAILED for a reason that had nothing to do with the query. The
        assertion below passed with the rewrite under test DELETED, and the commit message called
        that end-to-end verification. DuckDB was never consulted.
        """

        def __init__(self, con):
            self.con = con

        def execute(self, sql):
            return self.con.execute(sql).fetchall()

    con = duckdb.connect()
    con.execute("CREATE TABLE claim(id INT, ssn TEXT)")
    con.execute("INSERT INTO claim VALUES (1, 'SECRET')")

    verdict = decide("SELECT claim FROM claim", {"claim": {"id", "ssn", "claim"}},
                     adapter=_Adapter(con), target="duckdb",
                     policy=AccessPolicy(denied={("claim", "ssn")}))

    assert isinstance(verdict, Refusal)
    assert verdict.code == RefusalCode.EXPLAIN_FAILED, verdict.code
    # The refusal must be the BINDER refusing the qualified name, not the double falling over.
    assert "does not have a column named" in verdict.repair_text, verdict.repair_text

    # And a plainly valid query through the same adapter is approved, so the double is not simply
    # refusing everything -- which is exactly what the broken one did.
    from mnemiq.sql.verdict import Approved
    assert isinstance(
        decide("SELECT ssn FROM claim", {"claim": {"id", "ssn"}},
               adapter=_Adapter(con), target="duckdb"),
        Approved,
    )
    # And the control: with an accurate snapshot the same shape is the row, refused earlier.
    unstale = decide("SELECT claim FROM claim", {"claim": {"id", "ssn"}},
                     adapter=_Adapter(con), target="duckdb")
    assert isinstance(unstale, Refusal)
    assert unstale.code == RefusalCode.SELECT_STAR, unstale.code
