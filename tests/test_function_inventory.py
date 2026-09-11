"""What the source says about its own functions, and the three states that answer can be in.

The engine cannot tell a builtin from a same-named UDF by parsing -- sqlglot types `log` as a
builtin before the source binds it. Two controls settle for less because of it: lineage declines
to certify on ANY call (issue #5, so `count(*)` trips it), and the decider allows any name
sqlglot models, so a UDF called `median` passes (M43's residual). Both want the same fact.
"""

import os
import tempfile

import duckdb
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.sql.functions import FunctionInventory


def test_an_empty_answer_is_not_a_failed_lookup():
    """The distinction the type exists for, and another instance of it in this codebase.

    A source that defines nothing and a source that could not be asked both produce an empty
    set. Reading the second as the first is what let `check_views` approve on a failed discovery
    and refuse on a successful one, from the identical call.
    """
    answered_none = FunctionInventory.of(())
    could_not_ask = FunctionInventory.unavailable("permission denied")
    nobody_asked = FunctionInventory.never_asked()

    assert not answered_none.names and not could_not_ask.names and not nobody_asked.names
    assert answered_none.certain is True, "an empty ANSWER licenses treating names as builtins"
    assert could_not_ask.certain is False
    assert nobody_asked.certain is False
    # Distinct BY VALUE, which a frozenset subclass could not manage: the first version compared
    # equal with equal hashes, so a memo keyed on one returned the other, and `a | b` produced a
    # plain frozenset whose absent flags read as available under `getattr(x, "available", True)`.
    assert answered_none != could_not_ask != nobody_asked != answered_none
    assert len({answered_none, could_not_ask, nobody_asked}) == 3
    # ...and the two negative cases are themselves distinct, because they differ in whose fault
    # it is: an adapter with no inventory method is not an outage.
    assert (could_not_ask.available, could_not_ask.asked) == (False, True)
    assert (nobody_asked.available, nobody_asked.asked) == (True, False)


def test_both_constructors_normalise():
    """The plain constructor is handed out whether or not you meant to expose it, and it used to
    build a broken instance: `FunctionInventory({"MEDIAN"}).defines("median")` was False and
    `hash()` raised on the unfrozen set, while `of()` did the right thing. Two constructors with
    different semantics is a bug nobody looks for, and this one failed in the direction that
    matters -- a UDF named like a builtin read as the builtin."""
    assert FunctionInventory({"MEDIAN"}).defines("median")
    assert FunctionInventory(["a"]) == FunctionInventory.of(["a"])
    assert hash(FunctionInventory({"MEDIAN"}))
    # `of("median")` must not splay a bare string into single characters.
    assert FunctionInventory.of("median").names == frozenset({"median"})


def test_the_view_body_licence_is_separate_and_defaults_to_denied():
    """Two questions, two answers, and forgetting must give the conservative one.

    `duckdb_functions()` answers completely for a generated query and not at all for a Postgres
    view body, whose calls run server-side. The first version of this field defaulted True, so
    `FunctionInventory.of(adapter.user_functions())` -- the wiring anyone would write --
    certified view bodies on a source where that is false. The guard failed open on arrival.
    """
    plain = FunctionInventory.of(["my_udf"])
    assert plain.certain is True, "the top-level question is answered"
    assert plain.certain_for_view_bodies is False, "and the view-body one is not, unless said"

    covering = FunctionInventory.of(["my_udf"], covers_view_bodies=True)
    assert covering.certain_for_view_bodies is True

    # The two licences are independent: a source that could not be asked grants neither, and
    # `covers_view_bodies` alone must not resurrect the first.
    blind = FunctionInventory.unavailable("permission denied")
    assert blind.certain is False and blind.certain_for_view_bodies is False
    assert FunctionInventory(covers_view_bodies=True, available=False).certain_for_view_bodies is False
    # ...and the `asked` half too, or a property reading `available and covers_view_bodies`
    # would grant the licence to an inventory nobody ever asked for.
    assert FunctionInventory(covers_view_bodies=True, asked=False).certain_for_view_bodies is False


def test_names_match_regardless_of_case():
    """SQL folds case and the catalogue does not agree with itself across engines, so a set of
    raw strings would answer `defines('MEDIAN')` differently from `defines('median')`."""
    inv = FunctionInventory.of({"My_UDF", "MEDIAN"})
    assert inv.defines("my_udf") and inv.defines("MEDIAN") and inv.defines("Median")
    assert not inv.defines("count")


@pytest.fixture()
def duckdb_source():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "src.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT, amount INT)")
    con.execute("CREATE MACRO all_ssns() AS 1")     # the exfiltration fixture's shape
    con.execute("CREATE MACRO median(x) AS 42")     # a UDF wearing a name sqlglot MODELS
    con.close()
    return path


def test_duckdb_reports_what_it_defines_and_not_its_builtins(duckdb_source):
    """`internal` is the discriminator. DuckDB marks its own 945 builtins internal and anything
    a deployment adds comes back false, which is the fact parsing cannot supply."""
    fns = DuckDBAdapter.duckdb(duckdb_source).user_functions()

    assert "all_ssns" in fns
    # The case this whole file exists for: a source macro named exactly like something sqlglot
    # models. Measured -- `SELECT median(1)` on this source returns the macro's 42, not the
    # aggregate. A fixture using `median_` would pass while an implementation that filtered out
    # names colliding with builtins silently dropped the one name that matters.
    assert "median" in fns
    for builtin in ("count", "sum", "upper", "date_trunc"):
        assert builtin not in fns, f"{builtin} is a builtin and must not be reported as defined here"


def test_a_lookup_failure_raises_rather_than_reporting_none(duckdb_source):
    """Same contract as `view_definitions`, for the same reason: returning `[]` on failure turns
    "could not look" into "there are none", and that single choice is what defeated the view
    availability signal -- the job recorded `done`, the inventory read COMPLETE over nothing, and
    a granted view over a filtered table was approved unfiltered.
    """
    adapter = DuckDBAdapter.duckdb(duckdb_source)
    adapter._con.close()

    with pytest.raises(Exception):
        adapter.user_functions()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no ACME Postgres configured")
def test_a_postgres_udf_is_unreachable_from_a_query_and_absent_from_the_answer():
    """The claim this design rests on, checked rather than asserted.

    An earlier version asserted `"median" not in fns` against a database defining no `median`,
    so it could not fail. This one creates the function and checks the behaviour.

    Both halves are exercised, because they disagree. A generated query cannot reach a Postgres
    UDF, since DuckDB's binder resolves against its own catalogue. A Postgres VIEW BODY calling
    the same UDF runs it server-side and returns its value, which is why this answer must not be
    read as covering view bodies.

    EVERYTHING IT CREATES LIVES IN A THROWAWAY SCHEMA WITH A UNIQUE NAME, dropped with one
    CASCADE. The first version put `fn_probe` and `v_fn_probe` in `public` and opened with
    unconditional `DROP ... IF EXISTS` on both, against whatever `MNEMIQ_PG_DSN` names -- so a
    database that happened to hold those objects would have lost them, and the test's own
    fixtures were visible to anything else reading that schema meanwhile. Its DDL also sat
    outside the `try`, so a failure between the two CREATEs orphaned the function.

    The adapter under test is read-only. So is the engine's by default, though not when
    `MNEMIQ_WRITE_ENABLED` is set. Setup goes through a separate connection on purpose: a
    fixture needs write access that the thing being tested must not have.

    A run killed before the `finally` (SIGKILL, a timeout) leaves its schema behind. They are
    invisible to the engine, which introspects `public` only, and the unique names make them
    identifiable, but nothing sweeps them.
    """
    import uuid

    import duckdb as _duckdb

    dsn = os.environ["MNEMIQ_PG_DSN"]
    schema = f"mnemiq_probe_{uuid.uuid4().hex[:12]}"

    # Not wrapped in a skip. A DSN that is set but wrong, refused or pointing at a stopped
    # server is a broken configuration, and the siblings fail on it rather than reporting a
    # pass-shaped SKIPPED. The module-level skipif above covers the only case that is not an
    # error: nobody configured a database at all.
    admin = _duckdb.connect()
    admin.execute("INSTALL postgres; LOAD postgres")
    admin.execute(f"ATTACH '{dsn}' AS pg (TYPE POSTGRES)")

    def pg(stmt: str) -> None:
        admin.execute(f"CALL postgres_execute('pg', '{stmt}')")

    try:
        # Created before the adapter connects, so resolving the view cannot depend on the
        # postgres extension re-querying a schema cache the adapter's own `USE` had populated.
        pg(f"CREATE SCHEMA {schema}")
        pg(f"CREATE FUNCTION {schema}.fn_probe(int) RETURNS int AS $x$ SELECT 99 $x$ LANGUAGE sql")
        pg(f"CREATE VIEW {schema}.v_fn_probe AS SELECT {schema}.fn_probe(1) AS m")

        adapter = DuckDBAdapter.postgres(dsn, schema="src", read_only=True)
        fns = adapter.user_functions()

        assert "fn_probe" not in fns, "a Postgres UDF is not callable here and must not be listed"

        # ...and the reason: the query cannot reach it. QUALIFIED, which matters. The function
        # lives in the throwaway schema and the adapter's search path is `src.public`, so an
        # UNqualified call raises CatalogException for being off the path whether or not a
        # Postgres function is reachable -- the assertion would hold against a future extension
        # whose binder does resolve them, which is precisely the case that would make
        # `user_functions()` under-report. Qualified is also the form measured in the
        # `user_functions` docstring.
        with pytest.raises(_duckdb.CatalogException):
            adapter.execute(f"SELECT src.{schema}.fn_probe(1)")

        # The other half, and why the two licences are separate: through a view the same
        # function DOES run, server-side, and returns its value.
        #
        # No assertion on `FunctionInventory.of(fns)` here. It would read as Postgres coverage
        # and prove nothing -- the answer comes from the field's default, which the unit test
        # already pins. The guarantee that matters is that a PRODUCER never sets
        # `covers_view_bodies=True` for this attachment, and no producer exists yet.
        assert adapter.execute(f"SELECT m FROM src.{schema}.v_fn_probe")[0][0] == 99
    finally:
        # One statement, and it can only reach what this test made: the schema name is unique
        # per run, so a leftover from a crashed run cannot be confused with a real object.
        try:
            pg(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        finally:
            admin.close()
