import pytest
import sqlglot

from mnemiq.sql.authz_guard import check_access
from mnemiq.sql.verdict import Refusal, RefusalCode

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


# --------------------------------------------------------------------------------------------
# M43 -- a function in projection position produces no exp.Table, so both the access check
# and the RLS rewrite walk past it
# --------------------------------------------------------------------------------------------


_M43_VISIBLE = {"claim": {"id", "amount", "created_at", "ssn"}}


# Both, always. The first version of this guard tested `isinstance(node, exp.Anonymous)`,
# which asks whether the ONE dialect being parsed models the name -- and these tests pinned
# `postgres`, the dialect where the control held. Production parses as `duckdb`, where sqlglot
# leaves `now` and `date_part` Anonymous, so the shipped guard refused
# `WHERE created_at < now()` while its own control test said it did not.
_DIALECTS = ("duckdb", "postgres")   # duckdb FIRST: it is the production default


def _decide(sql, dialect):
    from mnemiq.sql.decide import decide

    return decide(sql, _M43_VISIBLE, dialect=dialect, target=dialect)


@pytest.mark.parametrize("dialect", _DIALECTS)
@pytest.mark.parametrize("sql", [
    "SELECT customer_rows() AS x",                                  # a UDF that reads
    "SELECT pg_read_file('/etc/passwd')",                           # not even a table read
    "SELECT public.customer_rows() AS x",                           # schema-qualified
    "SELECT id FROM claim WHERE amount > (SELECT max_amount())",    # buried in a predicate
])
def test_an_opaque_call_cannot_be_decided(sql, dialect):
    """Measured before the fix: every one of these was APPROVED with `tables=[]` for a caller
    granted only `claim`. `check_access` re-checks each `exp.Table`; a call in projection
    position is not one, so nothing looked at it and the RLS rewrite had nothing to wrap.

    The refusal does not claim the function is dangerous -- there is no list of dangerous
    functions here, deliberately. It claims the decider cannot tell what the query reads, which
    makes its own premise false.
    """
    v = _decide(sql, dialect)
    assert isinstance(v, Refusal), f"{sql!r} was approved under {dialect}"
    assert v.code == RefusalCode.UNMODELLED_CALL


@pytest.mark.parametrize("dialect", _DIALECTS)
@pytest.mark.parametrize("sql", [
    "SELECT id FROM claim",
    "SELECT sum(amount), count(*), upper(ssn) FROM claim",
    "SELECT date_trunc('month', created_at) FROM claim",
    "SELECT CAST(id AS TEXT) FROM claim",
    "SELECT id FROM claim WHERE created_at < now()",
    "SELECT current_timestamp FROM claim",
    "SELECT date_part('year', created_at) FROM claim",
])
def test_a_modelled_call_is_untouched(sql, dialect):
    """The degeneracy control, and the reason this is a whitelist rather than a blocklist.

    Two different things keep these approved, and conflating them is how the first version of
    this guard shipped broken:

      * sqlglot models the name IN THE DIALECT BEING PARSED, so it never becomes an
        `exp.Anonymous` and the guard never sees it -- `sum`, `count`, `upper`, `cast`.
      * it DOES become `Anonymous` and the cross-dialect allowlist lets it through by name --
        `now()` and `date_part()` under duckdb, which sqlglot models only under postgres.

    The second case is the one that matters here. The first version had no allowlist and tested
    `isinstance(node, exp.Anonymous)`, so it refused `WHERE created_at < now()` on the
    production dialect -- useless whatever else it caught -- while these tests, pinned to
    postgres, said otherwise. Hence `_DIALECTS`, duckdb first.
    """
    assert not isinstance(_decide(sql, dialect), Refusal), f"{sql!r} refused under {dialect}"


def test_the_refusal_is_repairable_so_a_false_positive_costs_a_retry():
    """The measured cost of this guard, over 459 SELECTs in this suite, under both dialects, was Postgres `age()`:
    a pure scalar function sqlglot does not model. Every other refusal was an attack fixture,
    a case another guard already refused, or SQL that never reaches the decider.

    That one case is why UNMODELLED_CALL is repairable. The loop can rewrite `age(x)` into
    arithmetic the engine does model, so a false positive costs an attempt rather than an
    answer -- which is what keeps a fail-closed guard from being the thing people route around.
    """
    v = _decide("SELECT age(created_at) FROM claim", "duckdb")
    assert isinstance(v, Refusal) and v.code == RefusalCode.UNMODELLED_CALL
    assert v.repairable is True
    assert "age" in v.message, "the model cannot repair what the refusal does not name"


# --------------------------------------------------------------------------------------------
# M43 on the WRITE path. Same premise, worse consequence: the write persists what it read.
# --------------------------------------------------------------------------------------------


def _decide_write(sql, dialect):
    from mnemiq.authz.grants import GrantSet
    from mnemiq.sql.decide import AccessPolicy
    from mnemiq.sql.decide_write import decide_write

    return decide_write(
        sql, {"claim": {"id", "amount"}},
        GrantSet(frozenset({"claim"}), writable=frozenset({"claim"})),
        policy=AccessPolicy(), dialect=dialect, target=dialect, writes_enabled=True,
    )


@pytest.mark.parametrize("dialect", _DIALECTS)
@pytest.mark.parametrize("sql", [
    "UPDATE claim SET amount = 1 WHERE id = 1",
    "UPDATE claim SET amount = amount + 1 WHERE id = 1",
    "INSERT INTO claim (id, amount) VALUES (1, 2)",
])
def test_an_ordinary_write_is_still_approved(sql, dialect):
    """The passing control. Without it the refusals below prove only that everything refuses --
    the first run of that probe used a read-only grant and every case came back
    `unauthorized_write`, which looks exactly like the guard working."""
    assert not isinstance(_decide_write(sql, dialect), Refusal), f"{sql!r} refused under {dialect}"


@pytest.mark.parametrize("dialect", _DIALECTS)
@pytest.mark.parametrize("sql", [
    "UPDATE claim SET amount = pg_read_file('/etc/passwd') WHERE id = 1",
    "UPDATE claim SET amount = 1 WHERE id = (SELECT secret_max_id())",
    "INSERT INTO claim (id, amount) SELECT customer_rows(), 1",
])
def test_an_opaque_call_cannot_be_decided_on_the_write_path_either(sql, dialect):
    """`decide_write` called `check_access` and not this guard, so the read path's fix left the
    write path open. Measured before it was wired: all three returned ApprovedWrite with
    `tables=['claim']` -- the same audit lie, except a write persists what it read into a table
    that a later plain SELECT returns forever (the M30 shape)."""
    v = _decide_write(sql, dialect)
    assert isinstance(v, Refusal), f"{sql!r} was approved under {dialect}"
    assert v.code == RefusalCode.UNMODELLED_CALL


# --------------------------------------------------------------------------------------------
# What the source is asked for, versus what the model wrote. Three earlier attempts at this
# question read the written spelling or the tree and each cleared a UDF that then returned an
# SSN (M56). `called_names` reads the RENDERED statement, because that is what the source gets.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", _DIALECTS)
def test_a_call_is_named_by_what_the_source_receives_not_by_what_was_written(dialect):
    """`len(name)` reaches the source as `LENGTH(name)`, so `length` is the name that binds and
    `len` is not. Reading the written word cleared a macro named `length` and returned an SSN.
    """
    from mnemiq.sql.functions import called_names

    written_len = sqlglot.parse_one("SELECT len(name) FROM claim", read=dialect)
    assert "length" in called_names(written_len, dialect)


@pytest.mark.parametrize("dialect", _DIALECTS)
def test_a_name_the_tree_does_not_carry_is_still_found(dialect):
    """`date_trunc` parses to `exp.TimestampTrunc`, which declares `timestamp_trunc` and
    `trunc` and has forgotten the word by the time anything can walk it. Rendering puts it
    back, which is why the scan is over text rather than over nodes."""
    from mnemiq.sql.functions import called_names

    assert "DATE_TRUNC" not in sqlglot.exp.TimestampTrunc.sql_names()
    ast = sqlglot.parse_one("SELECT date_trunc('day', created_at) FROM claim", read=dialect)
    assert "date_trunc" in called_names(ast, dialect)


def test_a_statement_that_will_not_render_is_not_read_as_call_free():
    """The absence-and-failure collapse, in the place it would fail open: an empty set of names
    intersects nothing, so a caller reading a render failure as "no calls" clears all of them.
    """
    from mnemiq.sql.functions import UnreadableCalls, called_names

    class WillNotRender(sqlglot.exp.Expression):
        def sql(self, *a, **kw):
            raise ValueError("no")

    with pytest.raises(UnreadableCalls):
        called_names(WillNotRender(), "duckdb")


def test_a_statement_that_will_not_render_refuses_rather_than_passes():
    """And the guard takes that the same way it takes shadowing, for the same reason."""
    from mnemiq.sql.authz_guard import check_unmodelled_calls
    from mnemiq.sql.functions import FunctionInventory

    class WillNotRender(sqlglot.exp.Expression):
        def sql(self, *a, **kw):
            raise ValueError("no")

        def find_all(self, *types):
            yield sqlglot.exp.Anonymous(this="count")

    inventory = FunctionInventory.of(["commission_rate"], builtins=["count_star"])
    refusal = check_unmodelled_calls(WillNotRender(), inventory, "duckdb")
    assert refusal.code == RefusalCode.UNRESOLVABLE_CALLS
    # ...and says whose problem it is. This source's catalogue answered both halves, so any of
    # the source-level sentences would send a deployer to fix something that is working.
    # Sharing an owner is not enough to share a sentence.
    assert "could not be rendered" in refusal.message
    assert "builtins" not in refusal.message


def test_without_an_inventory_the_guard_is_the_allowlist_it_was():
    """The default has to be the old behaviour exactly. Every caller that does not pass an
    inventory -- fixtures, the write path, a source with no adapter -- would otherwise change
    verdict, and a conservative default here would refuse every call in the suite."""
    from mnemiq.sql.authz_guard import check_unmodelled_calls

    ast = sqlglot.parse_one("SELECT count(*), median(id) FROM claim", read="duckdb")
    assert check_unmodelled_calls(ast) is None


def test_an_unresolvable_source_is_not_repaired_into_an_answer():
    """UNMODELLED_CALL is repairable because a model can route around one named function.
    UNRESOLVABLE_CALLS cannot be: it condemns every statement against the source, so there is
    nothing for a rewrite to avoid and a loop would spend every attempt where it started."""
    from mnemiq.sql.verdict import REPAIRABLE

    assert RefusalCode.UNMODELLED_CALL in REPAIRABLE
    assert RefusalCode.UNRESOLVABLE_CALLS not in REPAIRABLE


def test_one_code_two_answers_about_whether_to_try_again():
    """`UNRESOLVABLE_CALLS` is reached by arrivals that disagree about retrying, so the CODE
    cannot decide it and `repairable_override` says which (M98).

    Two are worth another attempt. A statement that would not render is about the statement, and
    its message asks for a rewrite. A source whose `user_functions` raised may be having a blip,
    and `decide` re-asks it live on every attempt, so the loop resolves that by itself. The rest
    are properties of the source that no attempt of ours changes -- retrying them spends the
    caller's budget to arrive where the first refusal already was, under a card telling them
    rephrasing will not help.
    """
    from mnemiq.sql.authz_guard import check_unmodelled_calls
    from mnemiq.sql.functions import FunctionInventory

    class WillNotRender(sqlglot.exp.Expression):
        def sql(self, *a, **kw):
            raise ValueError("no")

        def find_all(self, *types):
            yield sqlglot.exp.Anonymous(this="count")

    defines_something = FunctionInventory.of(["commission_rate"], builtins=["count_star"])
    ast = sqlglot.parse_one("SELECT median(id) FROM claim", read="duckdb")

    assert check_unmodelled_calls(WillNotRender(), defines_something, "duckdb").repairable is True
    assert check_unmodelled_calls(
        ast, FunctionInventory.unavailable("RuntimeError"), "duckdb").repairable is True

    # ...and so is the OTHER catalogue call that raised. `builtins_asked` with no builtins is
    # `builtin_functions()` having thrown, which the next attempt makes again -- leaving it out
    # while retrying its sibling was a contradiction, not a nuance.
    raised = FunctionInventory.of(["commission_rate"], builtins_asked=True)
    assert check_unmodelled_calls(ast, raised, "duckdb").repairable is True

    shadowing = FunctionInventory.of(["median"], builtins=["median"])
    no_such_method = FunctionInventory.of(["commission_rate"])   # builtins_asked stays False
    for inventory in (shadowing, no_such_method):
        refusal = check_unmodelled_calls(ast, inventory, "duckdb")
        assert refusal.code is RefusalCode.UNRESOLVABLE_CALLS
        assert refusal.repairable is False, inventory
