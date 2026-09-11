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


# --------------------------------------------------------------------------------------------
# The wiring. Every piece below was added by the same change and had no test: `decide` passing
# the inventory, the view-body licence, and the adapter that grants it.
# --------------------------------------------------------------------------------------------


@pytest.fixture()
def source_with_a_shadowing_macro():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "src.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE customer(id INT, name TEXT)")
    con.execute("CREATE TABLE secret(ssn TEXT)")
    con.execute("INSERT INTO secret VALUES ('123-45-6789')")
    # Named `replace` on purpose: sqlglot gives it its own keyword token, so it yields no name
    # from the tokens while the parser sees a call.
    con.execute("CREATE MACRO replace(a, b, c) AS (SELECT max(ssn) FROM secret)")
    # `length`, not `len`: sqlglot renders `len(x)` as `LENGTH(x)`, so this is the name the
    # source binds when the query says `len`.
    con.execute("CREATE MACRO length(a) AS (SELECT max(ssn) FROM secret)")
    con.close()
    return path


@pytest.fixture()
def source_defining_nothing():
    """The ordinary case, and the one issue #5 is about: a database with no functions of its
    own. Every call in a query against it is a builtin, whatever it is spelled."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "plain.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE customer(id INT, name TEXT)")
    con.close()
    return path


def _lineage_through_decide(sql, adapter):
    from mnemiq.sql.decide import decide

    return decide(sql, {"customer": {"id", "name"}}, adapter=adapter,
                  dialect="duckdb", target="duckdb").lineage


def test_decide_actually_asks_the_adapter(source_defining_nothing):
    """Dropping the `functions=` argument in `decide` makes the issue #5 fix disappear in
    production while every lineage unit test keeps passing, because those build the inventory
    themselves. This is the one that notices.

    A source defining nothing, because that is what makes the clearance observable: the marker
    is absent only if something asked and got an empty answer."""
    adapter = DuckDBAdapter.duckdb(source_defining_nothing)

    plain = _lineage_through_decide("SELECT count(*) AS n FROM customer", adapter)
    assert "unconfirmed-function-identity" not in plain.reasons, (
        "count(*) is a builtin on this source and the inventory says so -- issue #5"
    )
    # ...and the same call with no adapter to ask still downgrades, so the clearance comes from
    # asking rather than from the code having stopped caring.
    blind = _lineage_through_decide("SELECT count(*) AS n FROM customer", None)
    assert "unconfirmed-function-identity" in blind.reasons


@pytest.mark.parametrize("sql", [
    # A column list is `ident(`, so counting tokens against the parser's call count inflated the
    # tally and hid one unnamed call. Both of these returned COMPLETE while executing a macro
    # that read another table.
    "SELECT replace(name, 'a', 'b') FROM customer AS c(id, name)",
    "WITH x(name) AS (SELECT name FROM customer) SELECT replace(name, 'a', 'b') FROM x",
    # Written `len`, executed `LENGTH`. The source binds what executes, and no source defines
    # `len`, so reading only the written spelling cleared a macro named `length`.
    "SELECT len(name) FROM customer",
])
def test_a_shadowing_macro_is_caught_however_the_call_is_spelled(sql, source_with_a_shadowing_macro):
    """Three shapes that each returned `complete` while reading an SSN from another table.

    All three are M56, and all three were introduced by successive attempts to name the call:
    the written spelling, then the rendered one, then a count cross-check that read a column
    list as a call. They are kept as a regression set rather than as a spelling test -- nothing
    is looked up by name now, so what they demonstrate is that a source defining anything
    downgrades whatever the query says. Renaming the macros would leave them green, which is
    the point: the rule no longer depends on the spelling that defeated three versions.
    """
    lineage = _lineage_through_decide(sql, DuckDBAdapter.duckdb(source_with_a_shadowing_macro))
    assert "unconfirmed-function-identity" in lineage.reasons


@pytest.mark.parametrize("sql", [
    "SELECT count(*) AS n FROM customer",
    "SELECT CASE WHEN id > 1 THEN 'a' ELSE 'b' END FROM customer",
    "SELECT id::VARCHAR FROM customer",
])
def test_syntax_and_real_builtins_stay_clean(sql, source_defining_nothing):
    """The other half of issue #5. Against a source that defines nothing, every one of these is
    a builtin and the marker stays quiet -- including the shapes an earlier per-call version
    wrongly flagged, `CASE` and `::VARCHAR` among them."""
    lineage = _lineage_through_decide(sql, DuckDBAdapter.duckdb(source_defining_nothing))
    assert "unconfirmed-function-identity" not in lineage.reasons


def test_a_view_bodys_calls_are_judged_by_the_view_licence_not_the_query_one():
    """The two licences reach different code, and only this exercises the body one.

    A Postgres view body runs server-side, where DuckDB's catalogue never looked, so an
    inventory built from that catalogue must not clear a call found inside a body. Asserting
    the adapter's flag is not enough: passing `in_view_body=False` in lineage, or flipping the
    adapter to claim the licence, both leave that assertion green.
    """
    import sqlglot

    from mnemiq.contract.semantic import ViewDefinition
    from mnemiq.sql.functions import FunctionInventory
    from mnemiq.sql.lineage import lineage_for
    from mnemiq.sql.views import ViewInventory

    views = ViewInventory({"v_totals": ViewDefinition(
        object_id="v_totals", definition="SELECT count(*) AS n FROM customer", dialect="duckdb")})
    sql = "SELECT n FROM v_totals"
    ast = sqlglot.parse_one(sql, read="duckdb")

    # The body calls only `count`, which the source does not define. Whether that clears depends
    # entirely on whether this inventory speaks for view bodies.
    covers = FunctionInventory.of([], covers_view_bodies=True)
    does_not = FunctionInventory.of([], covers_view_bodies=False)

    # `v_totals`, not `customer`: the loop walks the tables the statement RESOLVED to and asks
    # which of them are views. Passing the base table means it never reaches a body at all --
    # which is how the first version of this test passed against a bug it was written to catch.
    cleared = lineage_for(ast, ["v_totals"], views, functions=covers)
    withheld = lineage_for(ast, ["v_totals"], views, functions=does_not)

    assert "unconfirmed-function-identity" not in cleared.reasons
    assert "unconfirmed-function-identity" in withheld.reasons, (
        "a body's calls must be judged by the view licence, which this inventory does not grant"
    )


def test_inventory_from_carries_the_failure_and_the_licence_through():
    """The handoff, which nothing exercised. Two mutations of it went unnoticed.

    A lookup that RAISES must not become "defines nothing": that reading approves every call,
    and it is the absence-versus-failure collapse this whole type exists for. And the licence
    has to be read from the adapter rather than assumed, or a Postgres view body gets certified
    from DuckDB's catalogue -- the thing the licence is for.
    """
    from mnemiq.sql.functions import inventory_from

    class Raises:
        functions_cover_view_bodies = True

        def user_functions(self):
            raise RuntimeError("dsn=postgresql://u:p@h/db refused")

    class Denies:
        functions_cover_view_bodies = False

        def user_functions(self):
            return []

    failed = inventory_from(Raises())
    assert failed.available is False and failed.certain is False, (
        "a failed lookup must not read as an empty answer"
    )
    assert "postgresql://" not in failed.reason, "the reason must not carry the source's words (M94)"

    class Grants:
        functions_cover_view_bodies = True

        def user_functions(self):
            return []

    # BOTH directions. Checking only the denying adapter leaves `getattr(...)` replaceable by a
    # literal `False`, which fails closed and silently: a DuckDB file's view bodies would take
    # the issue #5 downgrade again with nothing to notice.
    assert inventory_from(Grants()).certain_for_view_bodies is True

    denied = inventory_from(Denies())
    assert denied.certain is True and denied.certain_for_view_bodies is False

    # The DEFAULT, which `Denies` cannot pin because it sets the same value. An adapter that
    # implements `user_functions` and omits the property is the normal case per the protocol,
    # and defaulting that to True would fail open for every one of them.
    class Silent:
        def user_functions(self):
            return []

    # Paired, like `Denies`. `certain_for_view_bodies is False` alone is also what a
    # fail-closed read produces: move the attribute inside the `try` without a default and
    # `Silent` raises AttributeError, becomes `unavailable`, and this still passes -- taking
    # the downgrade with a false `function-inventory-unavailable` for every adapter that
    # omits the property, which the protocol says is most of them.
    silent = inventory_from(Silent())
    assert silent.certain is True and silent.certain_for_view_bodies is False

    assert inventory_from(object()).asked is False, "an adapter without the method was never asked"


def test_only_a_source_whose_catalogue_governs_its_views_grants_the_view_licence():
    """`functions_cover_view_bodies` is what stops a Postgres view body being certified from
    DuckDB's catalogue. Flipping it, or passing `in_view_body=False`, would do exactly that and
    nothing else in the suite would notice."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "f.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE t(x INT)")
    con.close()

    assert DuckDBAdapter.duckdb(path).functions_cover_view_bodies is True, (
        "a DuckDB file's views are DuckDB views and run against this catalogue"
    )
    assert DuckDBAdapter.sqlite(path).functions_cover_view_bodies is False, (
        "SQLite got the licence from the old FK-flag derivation, harmlessly and by accident"
    )
    # Bare instances, so the real derivation runs. Restating it here would pass whatever
    # `__init__` did, which is what let a mutation flipping it go unnoticed. Every attachment
    # kind except DUCKDB must be denied, not just Postgres: the licence used to key on the
    # foreign-key flag, which gave it to SQLite and would give it to any future kind.
    for attach_type in ("POSTGRES", "SQLITE", "MYSQL"):
        other = DuckDBAdapter.__new__(DuckDBAdapter)
        other._attach_type = attach_type
        assert other.functions_cover_view_bodies is False, attach_type


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

    created = False
    try:
        # Created before the adapter connects, so resolving the view cannot depend on the
        # postgres extension re-querying a schema cache the adapter's own `USE` had populated.
        pg(f"CREATE SCHEMA {schema}")
        created = True
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
        # `covers_view_bodies=True` for this attachment; `inventory_from` is that producer and reads
        # the adapter's own licence, which the unit tests pin.
        assert adapter.execute(f"SELECT m FROM src.{schema}.v_fn_probe")[0][0] == 99
    finally:
        # Only what this test actually created. `DROP ... IF EXISTS` ran unconditionally before,
        # including when `CREATE SCHEMA` had FAILED. Several failures land here having created
        # nothing -- no CREATE privilege on the database, a dropped connection -- and they are
        # harmless. The dangerous one is a name collision: the schema exists, this test does not
        # own it, and CASCADE takes it and everything in it. The uuid makes that improbable; the
        # code should not be relying on improbable.
        #
        # No `IF EXISTS` either: past this flag the schema is known to exist, so its absence is
        # something to surface rather than swallow.
        try:
            if created:
                pg(f"DROP SCHEMA {schema} CASCADE")
        finally:
            admin.close()
