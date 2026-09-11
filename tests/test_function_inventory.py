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
    stops the call being cleared, whatever the query says. Renaming the macros would leave them
    green, which is the point: the rule no longer depends on the spelling that defeated three
    versions.

    They used to be APPROVED and carry a lineage marker, which reported the doubt and then ran
    the query anyway. Both macros here are named after builtins, so the decider now refuses.
    """
    from mnemiq.sql.decide import decide
    from mnemiq.sql.verdict import RefusalCode

    verdict = decide(sql, {"customer": {"id", "name"}},
                     adapter=DuckDBAdapter.duckdb(source_with_a_shadowing_macro),
                     dialect="duckdb", target="duckdb")
    assert getattr(verdict, "code", None) is RefusalCode.UNRESOLVABLE_CALLS, verdict


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


# --------------------------------------------------------------------------------------------
# The DECIDER half. Lineage reports doubt and the query still runs; this is the gate. Everything
# below was measured as a leak first: on a source with a macro reading an ungranted table, both
# `SELECT count(*) FROM claim` and `SELECT median(id) FROM claim` were APPROVED with
# `tables=['claim']` and returned an SSN.
# --------------------------------------------------------------------------------------------


def _source(*macros, name="src.duckdb", read_only=True):
    path = os.path.join(tempfile.mkdtemp(), name)
    con = duckdb.connect(path)
    con.execute("CREATE TABLE secret(ssn VARCHAR)")
    con.execute("INSERT INTO secret VALUES ('123-45-6789')")
    con.execute("CREATE TABLE claim(id INTEGER, name VARCHAR, ts TIMESTAMP)")
    con.execute("INSERT INTO claim VALUES (1, 'a', TIMESTAMP '2020-01-01'), (2, 'b', NULL)")
    for macro in macros:
        con.execute(macro)
    con.close()
    return DuckDBAdapter.duckdb(path, read_only=read_only)


def _verdict(sql, adapter):
    from mnemiq.sql.decide import decide

    return decide(sql, {"claim": {"id", "name", "ts"}}, adapter=adapter,
                  dialect="duckdb", target="duckdb")


def _code(verdict):
    return getattr(getattr(verdict, "code", None), "value", None)


def test_a_macro_named_after_a_builtin_stops_the_source_being_decided_at_all():
    """`count_star` is DuckDB's binder name for `COUNT(*)`, and it appears in no spelling of
    `SELECT count(*) FROM claim` under any dialect. Measured before this: that query was
    approved and returned an SSN from `secret`.

    The refusal covers the whole SOURCE, not the calls anything can find in the statement. The
    first version stopped at `find_all(exp.Func)`, which reads a plain `SELECT id FROM claim`
    as call-free and is right, and reads `SELECT id + 1 FROM claim` as call-free and is wrong:
    `exp.Add` is not an `exp.Func`, DuckDB lists `+` as an internal function, and with a `"+"`
    macro that query was APPROVED and returned the SSN. Enumerating the node types that reach
    the catalogue is the blocklist this codebase refuses to write, so the coarse answer is the
    honest one -- a source whose names cannot be trusted answers nothing.
    """
    adapter = _source("CREATE MACRO count_star() AS (SELECT max(ssn) FROM secret)")

    leak = _verdict("SELECT count(*) AS n FROM claim", adapter)
    assert _code(leak) == "unresolvable_calls", leak
    assert "count_star" in leak.message

    assert _code(_verdict("SELECT median(id) AS m FROM claim", adapter)) == "unresolvable_calls"
    assert _code(_verdict("SELECT id FROM claim", adapter)) == "unresolvable_calls"


@pytest.mark.parametrize("sql, macro", [
    ("SELECT id + 1 AS x FROM claim", 'CREATE MACRO "+"(a, b) AS (SELECT max(ssn) FROM secret)'),
    ("SELECT name || 'x' AS x FROM claim", 'CREATE MACRO "||"(a, b) AS (SELECT max(ssn) FROM secret)'),
    ("SELECT ts AT TIME ZONE 'UTC' AS t FROM claim",
     "CREATE MACRO timezone(a, b) AS (SELECT max(ssn) FROM secret)"),
])
def test_an_operator_is_a_catalogue_call_too(sql, macro):
    """Three shapes with no `exp.Func` node anywhere, each APPROVED and each returning an SSN.

    `exp.Add`, `exp.DPipe` and `exp.AtTimeZone` are not function nodes, and `+`, `||` and
    `timezone` are all `internal` in `duckdb_functions()`, so all three are shadowable. They
    are the measurement behind refusing per-source rather than per-call: three node types were
    found by looking, which says nothing about how many are left.
    """
    adapter = _source(macro, name="ops.duckdb")
    assert _code(_verdict(sql, adapter)) == "unresolvable_calls", sql


def test_a_macro_the_engine_actually_binds_is_refused_by_the_name_it_binds():
    """The other half, and the one that keeps the first from swallowing it.

    `add_days` is a name sqlglot models -- so the cross-dialect allowlist passes it -- and NOT a
    DuckDB builtin, so nothing shadows and the coarse rule stays quiet. It survives rendering,
    which is what makes it reachable: measured, `SELECT add_days(id) FROM claim` returned an SSN
    on this source. 440 names sqlglot knows are reachable this way.

    `count(*)` against the same source is approved, which is the whole reason both rules exist
    instead of the coarse one alone.
    """
    adapter = _source("CREATE MACRO add_days(x) AS (SELECT max(ssn) FROM secret)")

    refusal = _verdict("SELECT add_days(id) AS n FROM claim", adapter)
    assert _code(refusal) == "unmodelled_call", refusal
    assert "defined by this source itself" in refusal.message
    assert refusal.subject == "add_days"

    assert not hasattr(_verdict("SELECT count(*) AS n FROM claim", adapter), "code")


@pytest.mark.parametrize("sql", [
    "SELECT count(*) AS n FROM claim",
    "SELECT median(id) AS m FROM claim",
    "SELECT date_trunc('day', CURRENT_DATE) AS d",
    "SELECT CAST(id AS VARCHAR) AS s FROM claim",
])
def test_a_helper_that_shadows_nothing_costs_nothing(sql):
    """The degeneracy control. A source with a macro of its own, named after nothing, keeps
    answering every ordinary question.

    This is what fails if `builtin_functions` reads the wrong half of `duckdb_functions()`:
    the user names would then intersect themselves, every source with any macro would look
    like a shadowing one, and all four of these would be refused.
    """
    adapter = _source("CREATE MACRO commission_rate(x) AS (x * 0.05)")
    assert not hasattr(_verdict(sql, adapter), "code"), sql


def test_the_two_catalogue_halves_are_one_query_split_on_internal():
    """`builtin_functions` is `user_functions` with the predicate flipped, and the intersection
    of the two is the whole question. Asserting the flag on the adapter is not enough -- what
    matters is that the macro lands on one side and DuckDB's own names on the other."""
    adapter = _source("CREATE MACRO commission_rate(x) AS (x * 0.05)")
    user, builtin = set(adapter.user_functions()), set(adapter.builtin_functions())

    assert user == {"commission_rate"}
    assert {"count_star", "median", "length"} <= builtin
    assert not user & builtin

    shadowing = _source("CREATE MACRO median(x) AS (SELECT max(ssn) FROM secret)")
    assert set(shadowing.user_functions()) & set(shadowing.builtin_functions()) == {"median"}


def test_an_adapter_that_cannot_say_what_a_builtin_is_gets_the_conservative_answer():
    """`builtin_functions` may be omitted, and omitting it must cost answers rather than
    soundness. An adapter offering the other half alone has every statement against such a
    source declined, because a binder rename it cannot rule out is exactly the leak this is
    about -- expensive, and in the safe direction.

    A source that defines NOTHING is untouched either way, which is what keeps the conservative
    default off every ordinary database.
    """
    from mnemiq.sql.functions import inventory_from

    class HalfAnswering:
        def user_functions(self):
            return ["commission_rate"]

    class Angry(HalfAnswering):
        def builtin_functions(self):
            raise RuntimeError("no catalogue for you")

    class Plain:
        def user_functions(self):
            return []

    assert inventory_from(HalfAnswering()).builtins is None
    assert inventory_from(HalfAnswering()).may_shadow_a_builtin is True
    assert inventory_from(Angry()).may_shadow_a_builtin is True
    assert inventory_from(Plain()).may_shadow_a_builtin is False

    # ...and the two are told apart, because the refusal names a different owner for each.
    # Both leave `builtins` None, so without this flag an adapter whose catalogue call RAISED
    # is told to implement the method it already implemented.
    assert inventory_from(HalfAnswering()).builtins_asked is False
    assert inventory_from(Angry()).builtins_asked is True

    # An answer implies the question, whoever built the instance. Nothing in the product can
    # reach the contradictory state today, which is exactly why it would survive unnoticed
    # until something could.
    assert FunctionInventory.of(["x"], builtins=[]).builtins_asked is True
    assert FunctionInventory.of(["x"], builtins=[], builtins_asked=False).builtins_asked is True

    from mnemiq.sql.authz_guard import check_unmodelled_calls

    ast = __import__("sqlglot").parse_one("SELECT count(*) FROM claim", read="duckdb")
    assert "never asked" in check_unmodelled_calls(
        ast, inventory_from(HalfAnswering()), "duckdb").message
    assert "could not say which names" in check_unmodelled_calls(
        ast, inventory_from(Angry()), "duckdb").message


def test_a_source_that_could_not_be_asked_is_not_a_source_that_defines_nothing():
    """`unavailable` carries empty `names`, which every test in the guard below it would read
    as "defines nothing" and clear. Measured on the first version: with a raising
    `user_functions()`, `SELECT median(id) FROM claim` came back approved on a source holding a
    `median` macro -- the collapse `FunctionInventory` exists to prevent, in the guard written
    to use it.

    `never_asked` is the other empty answer and must stay permissive: no adapter, or an adapter
    without the method, is every fixture in this suite and the state the engine shipped in.
    """
    import sqlglot

    from mnemiq.sql.authz_guard import check_unmodelled_calls

    ast = sqlglot.parse_one("SELECT median(id) FROM claim", read="duckdb")
    refused = check_unmodelled_calls(ast, FunctionInventory.unavailable("RuntimeError"), "duckdb")
    assert refused is not None and refused.code.value == "unresolvable_calls"
    assert "RuntimeError" not in refused.message, "the reason is for the trace, not the model"

    # Distinguished from the OTHER source failure, not merely non-empty. Both sentences begin
    # "could not say", so asserting that substring alone let the two be swapped: a source that
    # would not list its builtins and one that would not list its own functions are found by
    # looking in different places, and the sentence is the only thing that says which.
    could_not_name_builtins = check_unmodelled_calls(
        ast, FunctionInventory.of(["median"], builtins_asked=True), "duckdb")
    assert "which functions it defines" in refused.message
    assert "which names are its builtins" in could_not_name_builtins.message
    assert refused.message != could_not_name_builtins.message

    assert check_unmodelled_calls(ast, FunctionInventory.never_asked(), "duckdb") is None


def test_the_write_path_asks_the_same_source_the_same_question():
    """`decide_write` called the guard without an inventory, so the two paths disagreed about
    the same source in the same deployment: a read of `median(id)` was refused while an UPDATE
    computing the same value was approved, and a write persists what it read into a granted
    table that every later plain SELECT returns.

    The ordinary write beside it is the control. Without one this asserts only that the write
    path refuses everything, which is what a wrong grant set looks like.
    """
    from mnemiq.authz.grants import GrantSet
    from mnemiq.sql.decide_write import decide_write
    from mnemiq.sql.policy import AccessPolicy

    def write(sql, adapter):
        return decide_write(
            sql, {"claim": {"id", "name", "ts"}},
            GrantSet(frozenset({"claim"}), writable=frozenset({"claim"})),
            policy=AccessPolicy(), adapter=adapter, dialect="duckdb", target="duckdb",
            writes_enabled=True,
        )

    shadowing = _source("CREATE MACRO count_star() AS (SELECT max(ssn) FROM secret)")
    assert _code(write("UPDATE claim SET id = 2 WHERE id = 1", shadowing)) == "unresolvable_calls"

    # Writable, because the control has to reach `prove`, and EXPLAIN on a write is refused by
    # a read-only attach -- which would make it pass for the wrong reason.
    named = _source("CREATE MACRO add_days(x) AS (SELECT max(ssn) FROM secret)",
                    name="w.duckdb", read_only=False)
    leak = write("UPDATE claim SET name = add_days(id) WHERE id = 1", named)
    assert _code(leak) == "unmodelled_call", leak
    assert not hasattr(write("UPDATE claim SET id = 2 WHERE id = 1", named), "code")


# --------------------------------------------------------------------------------------------
# Reachability. `duckdb_functions()` carries `database_name` and `schema_name` and the first
# version threw both away, so a macro in a schema nothing can reach counted as a shadowed
# builtin and the source answered NOTHING. Found by adversarial review, reproduced before fixing.
# --------------------------------------------------------------------------------------------


def _multi_schema_source(name="multi.duckdb"):
    """A macro named after a builtin, in a schema that is not on the search path."""
    path = os.path.join(tempfile.mkdtemp(), name)
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INTEGER)")
    con.execute("INSERT INTO claim VALUES (1), (2)")
    con.execute("CREATE SCHEMA other")
    con.execute("CREATE MACRO other.median(x) AS (SELECT 999)")
    con.close()
    return DuckDBAdapter.duckdb(path)


def test_a_macro_nothing_can_reach_does_not_condemn_the_source():
    """Measured on the source below: `SELECT median(id) FROM claim` returns **1.5**, DuckDB's
    builtin, because `other` is not on the search path -- and `other.median(1)` returns 999. So
    the macro exists and no unqualified call can collect it.

    The unscoped version read `median` as a shadowed builtin and refused everything, `SELECT id
    FROM claim` included. One unrelated macro in one unused schema made a legitimate source
    answer nothing, which is a worse failure than the leak the rule was added for, because it is
    silent about being wrong.
    """
    adapter = _multi_schema_source()
    assert adapter.user_functions() == ["median"]
    assert adapter.reachable_user_functions() == [], "nothing on the search path defines it"

    assert not hasattr(_verdict("SELECT id FROM claim", adapter), "code")
    assert not hasattr(_verdict("SELECT count(*) AS n FROM claim", adapter), "code")


def test_but_a_qualified_call_to_that_same_macro_is_still_refused():
    """The other half, and the reason `names` stays whole while only the COARSE rule is scoped.
    A query may qualify, and `other.median(1)` reaches the macro -- so the bare name read off the
    rendered statement has to keep matching it."""
    adapter = _multi_schema_source(name="qualified.duckdb")
    refused = _verdict("SELECT other.median(1) AS m", adapter)
    assert _code(refused) == "unmodelled_call", refused


def test_the_same_name_on_the_search_path_still_condemns_the_source():
    """The contrast that makes the scoping meaningful rather than a way to switch the rule off.
    `main` IS the search path, so this macro can collect an unqualified call and the engine
    cannot say which call is which."""
    path = os.path.join(tempfile.mkdtemp(), "onpath.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INTEGER)")
    con.execute("CREATE MACRO main.median(x) AS (SELECT 999)")
    con.close()
    adapter = DuckDBAdapter.duckdb(path)

    assert adapter.reachable_user_functions() == ["median"]
    assert _code(_verdict("SELECT id FROM claim", adapter)) == "unresolvable_calls"


def test_an_adapter_that_cannot_scope_its_catalogue_counts_every_name_as_reachable():
    """`reachable` is optional in the same direction as `builtins`: omitting it costs answers,
    never soundness. An adapter offering only `user_functions` gets the behaviour that shipped
    before scoping existed."""
    from mnemiq.sql.functions import inventory_from

    class Unscoped:
        def user_functions(self):
            return ["median"]

        def builtin_functions(self):
            return ["median", "count_star"]

    class Scoped(Unscoped):
        def reachable_user_functions(self):
            return []

    assert inventory_from(Unscoped()).reachable is None
    assert inventory_from(Unscoped()).may_shadow_a_builtin is True
    assert inventory_from(Scoped()).may_shadow_a_builtin is False
    # ...and the full list is untouched either way, so a qualified call is still caught.
    assert inventory_from(Scoped()).names == frozenset({"median"})


def test_every_adapter_the_resolver_hands_out_can_answer_the_inventory():
    """M99 was that one adapter could and the others could not, and nothing said so.

    `resolve.py` returns `OracleAdapter` or the DuckDB family, and `runtime.py` builds a
    `FederatedAdapter` for a multi-source deployment. All three are checked here because the gap
    was invisible exactly where it mattered: federation IS DuckDB, so every argument written for
    the DuckDB adapter read as though it already applied.

    The unused `pg`/`sqlite` classes are deliberately not asserted -- nothing hands them out, and
    requiring the methods there would be a test defending code no deployment reaches.
    """
    from mnemiq.adapters.duckdb import DuckDBAdapter, DuckDBPostgresAdapter
    from mnemiq.adapters.federated import FederatedAdapter
    from mnemiq.adapters.oracle import OracleAdapter

    for cls in (DuckDBAdapter, DuckDBPostgresAdapter, FederatedAdapter):
        assert hasattr(cls, "user_functions"), cls
        assert hasattr(cls, "builtin_functions"), cls
        assert hasattr(cls, "reachable_user_functions"), cls
        assert getattr(cls, "binder_prefers_builtins", False) is False, (
            f"{cls.__name__} binds a macro over a builtin -- `count(*)` reaches `count_star`"
        )

    # Oracle answers the same question with a different shape, and the difference is measured
    # in `tests/test_oracle_adapter.py`: its binder settles an unqualified call on its own
    # builtin, so it needs no builtin catalogue and never takes the whole-source refusal.
    assert hasattr(OracleAdapter, "user_functions")
    assert OracleAdapter.binder_prefers_builtins is True


# --------------------------------------------------------------------------------------------
# Oracle's synonym walk, driven through a fake `_rows` so it needs no database. The shapes below
# cannot be built on the test container at all -- `CREATE PUBLIC SYNONYM` is ORA-01031 for the
# app user -- and the one that matters is a collision whose outcome USED to depend on the hash
# seed, which is not something an integration test would have caught reliably either.
# --------------------------------------------------------------------------------------------


def _oracle_walk(synonyms, functions, oracle_owned=("sys",), schema="APP", asked=None):
    """`_synonyms_reaching_functions` over a stubbed data dictionary.

    `asked` collects the owners the object read binds, because WHICH owners it asks about is
    where the cost lives and no stub can time it.
    """
    from mnemiq.adapters.oracle import OracleAdapter

    asked = set() if asked is None else asked
    adapter = OracleAdapter.__new__(OracleAdapter)
    adapter._schema = schema

    def rows(sql, **binds):
        if "all_objects" in sql:
            asked.update(binds.values())
        if "all_synonyms" in sql:
            # The WHOLE graph, asserted here because a stub cannot honour a WHERE clause and
            # would silently pass a version that narrowed the read before walking it -- which is
            # the bug that stopped a chain at the first hop into another schema.
            assert "where" not in sql.lower(), "the synonym graph must be read unfiltered"
            return list(synonyms)
        if "oracle_maintained" in sql:
            return [(o,) for o in oracle_owned]
        wanted = set(binds.values())
        return [(o, n) for o, n in functions if o in wanted]

    adapter._rows = rows
    return adapter._synonyms_reaching_functions()


def test_a_name_shared_by_a_private_and_a_public_synonym_resolves_the_same_way_every_time():
    """The walk keyed on the target's bare NAME, so this resolved differently under different
    hash seeds -- reached under PYTHONHASHSEED 0, 1, 4 and 7, missed under 2, 3, 5 and 6.

    A control whose verdict moves with the hash seed is not a control. Both links are keyed by
    (owner, name) and BOTH are followed, so nothing "wins" and nothing depends on iteration
    order -- an alias counts if any object its chain can name is a function.
    """
    synonyms = [
        ("app", "add_days", "app", "x"),     # the alias a query would spell
        ("app", "x", "app", "udf"),          # private: reaches a function
        ("public", "x", "app2", "some_tbl"),  # public: a red herring of the same name
    ]
    functions = [("app", "udf")]
    assert _oracle_walk(synonyms, functions) == {"add_days", "x"}


def test_a_chain_through_a_schema_this_connection_does_not_own_is_still_followed():
    """The edge set used to be narrowed to this schema before the walk, so a hop through another
    owner's synonym ended it. The whole graph is read and only the REPORTING is filtered."""
    synonyms = [
        ("app", "add_days", "shared", "mid"),
        ("shared", "mid", "app", "udf"),
    ]
    assert "add_days" in _oracle_walk(synonyms, [("app", "udf")])


def test_a_cycle_names_nothing_and_does_not_hang():
    """`loop_a -> loop_b -> loop_a` raised ORA-32044 from the recursive SQL form, so it is a real
    shape a deployment can hold. Termination is `seen`, and a cycle reaches no function, so
    neither alias may be reported as one."""
    synonyms = [("app", "loop_a", "app", "loop_b"), ("app", "loop_b", "app", "loop_a")]
    assert _oracle_walk(synonyms, [("app", "udf")]) == set()


def test_a_chain_longer_than_any_fixed_cap_still_resolves():
    """A hop count on top of `seen` adds nothing to termination and silently drops a real chain:
    with a cap of 8, a nine-hop chain left its first two aliases out of the inventory."""
    synonyms = [("app", f"s{i}", "app", f"s{i+1}") for i in range(12)]
    synonyms.append(("app", "s12", "app", "udf"))
    assert "s0" in _oracle_walk(synonyms, [("app", "udf")])


def test_oracles_own_public_synonyms_stay_out_so_lineage_survives():
    """368 PUBLIC synonyms reach a function on the test container, every one Oracle-maintained.
    Reporting them turns `calls_are_confirmable` false for every Oracle source -- issue #5 on a
    schema that defines nothing of its own -- and they protect nothing, since none is in
    sqlglot's vocabulary and the allowlist already refuses each as an unmodelled call.

    The filter is on the ULTIMATE TARGET's owner, not the synonym's, which is what lets a PUBLIC
    synonym over a deployment's own function back in.
    """
    oracle_side = [("public", "dbms_output", "sys", "dbms_output")]
    deployment_side = [("public", "add_days", "app2", "udf")]
    functions = [("sys", "dbms_output"), ("app2", "udf")]

    assert _oracle_walk(oracle_side + deployment_side, functions,
                        oracle_owned=("sys",)) == {"add_days"}


def test_an_alias_and_its_target_may_share_a_name():
    """`CREATE PUBLIC SYNONYM add_days FOR app2.add_days` is the commonest synonym of all, and a
    by-name fallback to PUBLIC read it as a link from the synonym back to itself -- the walk then
    ended on `seen` having found nothing, and the alias was dropped.

    A target carries its owner, so there is no search order to model: the lookup is exact.
    """
    assert _oracle_walk([("public", "add_days", "app2", "add_days")],
                        [("app2", "add_days")], oracle_owned=("sys",)) == {"add_days"}
    # ...and the same shape privately owned.
    assert _oracle_walk([("app", "udf", "app2", "udf")],
                        [("app2", "udf")]) == {"udf"}


def test_the_public_link_is_followed_when_the_named_schema_has_no_such_object():
    """Oracle falls back to PUBLIC when the qualified target does not exist. Measured on the
    container: `CREATE SYNONYM zz FOR appuser.dbms_random` with no `APPUSER.DBMS_RANDOM`, and
    `SELECT zz.value FROM dual` returned a number through `PUBLIC.DBMS_RANDOM`.

    An exact-only lookup dropped the alias, which is an approval. Which link applies depends on
    whether the named object exists -- the very list this walk feeds -- so the order is not
    modelled: both are followed, and over-inclusion costs a refusal rather than an approval.
    """
    synonyms = [
        ("app", "add_days", "app", "calc"),   # app.calc does not exist...
        ("public", "calc", "app2", "calc"),   # ...so Oracle takes the PUBLIC link
    ]
    assert "add_days" in _oracle_walk(synonyms, [("app2", "calc")])


def test_the_synonym_walk_is_linear_in_the_graph():
    """It was quadratic: a reachable set stored per alias, measured on a synthetic chain at
    3.6 ms for 200 synonyms, 13.9 for 400, 63.2 for 800 and 239.3 for 1,600. A dictionary of the
    size M101 is about would not have finished.

    Walking BACKWARDS from the functions visits each node once. The assertion is a ratio rather
    than a wall-clock number, because a timing test pinned to a machine is a flaky test: doubling
    the graph must not quadruple the work.
    """
    import time

    def elapsed(n):
        synonyms = [("app", f"c{i}", "app", f"c{i + 1}") for i in range(n)]
        synonyms += [("app", f"a{j}", "app", "c0") for j in range(n)]
        start = time.perf_counter()
        _oracle_walk(synonyms, [("app", f"c{n}")])
        return time.perf_counter() - start

    small = min(elapsed(500) for _ in range(3))
    large = min(elapsed(2000) for _ in range(3))
    # Quadratic would be ~16x for a 4x graph. Linear is ~4x; the ceiling leaves room for noise
    # without leaving room for the defect.
    assert large < small * 9, f"{small * 1000:.1f} ms -> {large * 1000:.1f} ms looks superlinear"


def test_a_public_chain_that_leaves_oracles_schemas_on_a_later_hop_still_counts():
    """The candidate prefilter skipped a PUBLIC alias whose IMMEDIATE target was
    Oracle-maintained, purely to bound a quadratic walk. Dropping it is a behaviour change: a
    fuzz over random graphs found 68 differences and every one was this gain.

    `public.later -> sys.hop -> app2.udf` ends outside Oracle's schemas, so the alias is a route
    a deployment created and belongs in the inventory -- the first hop passing through SYS says
    nothing about where it ends.
    """
    synonyms = [("public", "later", "sys", "hop"), ("sys", "hop", "app2", "udf")]
    assert _oracle_walk(synonyms, [("app2", "udf")], oracle_owned=("sys",)) == {"later"}

    # ...and the mirror: a chain that ENDS in an Oracle schema stays out however it got there.
    ends_inside = [("public", "inside", "app2", "hop2"), ("app2", "hop2", "sys", "dbms_output")]
    assert _oracle_walk(ends_inside, [("sys", "dbms_output")], oracle_owned=("sys",)) == set()


def test_the_object_read_does_not_ask_about_oracles_own_schemas():
    """Which owners the object read binds is where the cost lives, and no stub can time it.

    Taking every owner named by an edge pulled SYS in, and `all_objects` for SYS alone is 107 ms
    against 3.5 ms for the app schema -- `user_functions` went from 30 ms to 157 ms that way.
    Only two sets can decide anything: what THIS schema's aliases reach, and every owner Oracle
    does not maintain, because a PUBLIC alias counts only where its chain ends outside Oracle's
    schemas.
    """
    asked: set[str] = set()
    synonyms = [
        ("public", "dbms_output", "sys", "dbms_output"),   # Oracle's own: cannot decide anything
        ("app", "mine", "app2", "udf"),                    # this schema's: must be asked about
    ]
    _oracle_walk(synonyms, [("app2", "udf")], oracle_owned=("sys",), asked=asked)

    assert "app2" in asked, "this schema's chain endpoints have to be resolved"
    assert "sys" not in asked, "SYS cannot decide either arm, and asking costs 107 ms"


def test_a_synonym_with_no_target_owner_does_not_take_the_source_down():
    """An unqualified DB-link synonym has a NULL `TABLE_OWNER`, and it used to reach `sorted()`
    and raise TypeError. `inventory_from` turns a raise into `unavailable`, which the guard turns
    into refusing EVERY statement on the source -- so one such row anywhere in the dictionary
    would have stopped the deployment, including queries that call nothing.

    Skipped, not resolved: a DB-link target has no local `all_objects` row to resolve against,
    which the adapter's docstring already names as the limit.
    """
    synonyms = [("other", "remote_s", None, "emp"), ("app", "x", "app", "udf")]
    assert _oracle_walk(synonyms, [("app", "udf")]) == {"x"}


def test_the_owner_read_is_chunked_under_oracles_in_list_cap():
    """An Oracle IN list is capped at 1,000 items (ORA-01795), and this one grows with every
    non-Oracle owner any synonym targets -- a schema-per-tenant instance passes that mark.

    The stub records how many owners each READ binds and asserts on the largest, because the cap
    is per statement rather than per call.
    """
    synonyms = [("app", f"a{i}", f"own{i}", "udf") for i in range(2500)]
    reads: list[int] = []

    from mnemiq.adapters.oracle import OracleAdapter

    adapter = OracleAdapter.__new__(OracleAdapter)
    adapter._schema = "APP"

    def rows(sql, **binds):
        if "all_synonyms" in sql:
            return list(synonyms)
        if "oracle_maintained" in sql:
            return [("sys",)]
        reads.append(len(binds))
        return [(o, "udf") for o in binds.values()]

    adapter._rows = rows
    got = adapter._synonyms_reaching_functions()

    assert len(got) == 2500
    assert reads and max(reads) <= 900, f"one read bound {max(reads)} owners"


# --------------------------------------------------------------------------------------------
# M100. A virtual column runs its stored expression on read, so the statement never names the
# call. Third shape of that here, after `count(*)` reaching a macro named `count_star` and
# `SELECT id + 1` reaching one named `+`.
# --------------------------------------------------------------------------------------------


def _opaque(virtual, defines, certain=True):
    from mnemiq.sql.functions import FunctionInventory, opaque_columns

    class Source:
        def virtual_columns(self):
            return list(virtual)

    inventory = (FunctionInventory.of(defines) if certain
                 else FunctionInventory.unavailable("RuntimeError"))
    return opaque_columns(Source(), inventory)


def test_only_a_virtual_column_that_names_a_user_function_is_opaque():
    """Arithmetic calls nothing. Refusing every virtual column would cost a legitimate modelling
    feature to close a route that needs a function to be a route at all -- measured on the
    container, `total AS (qty * price)` stores `"QTY"*"PRICE"` and `leaked AS (vc_udf(id))`
    stores `"APPUSER"."VC_UDF"("ID")`.
    """
    virtual = [("vc_t", "total", '"QTY"*"PRICE"'),
               ("vc_t", "leaked", '"APPUSER"."VC_UDF"("ID")')]
    assert _opaque(virtual, ["vc_udf"]) == frozenset({("vc_t", "leaked")})


def test_a_virtual_column_calling_something_this_source_does_not_define_is_left_alone():
    """A builtin in a virtual column is still a builtin. `UPPER("NAME")` names nothing this
    source defines, and the name-based question is the same one the call guard asks."""
    assert _opaque([("t", "shouty", 'UPPER("NAME")')], ["vc_udf"]) == frozenset()


def test_an_uncertain_inventory_makes_every_virtual_column_opaque():
    """If no name in an expression can be cleared, none of them is cleared. The empty `names` an
    `unavailable` inventory carries would otherwise read as "this source defines nothing", which
    is the collapse `FunctionInventory` exists to prevent."""
    virtual = [("t", "a", '"X"*"Y"'), ("t", "b", 'F("X")')]
    assert _opaque(virtual, [], certain=False) == frozenset({("t", "a"), ("t", "b")})


def test_an_adapter_that_cannot_report_virtual_columns_changes_nothing():
    """Every adapter shipped in that state, so it must stay the permissive one -- the same
    posture `never_asked` takes for functions."""
    from mnemiq.sql.functions import FunctionInventory, opaque_columns

    class Silent:
        pass

    class Angry:
        def virtual_columns(self):
            raise RuntimeError("no dictionary for you")

    assert opaque_columns(Silent(), FunctionInventory.of(["f"])) == frozenset()
    assert opaque_columns(Angry(), FunctionInventory.of(["f"])) == frozenset()
    assert opaque_columns(None, FunctionInventory.of(["f"])) == frozenset()


def test_the_guard_catches_the_column_qualified_or_not():
    """`check_access` sees a column the snapshot lists and passes it, so this is the only thing
    between the caller and the expression. A qualifier must not be a way around it."""
    import sqlglot

    from mnemiq.sql.authz_guard import check_opaque_columns

    opaque = frozenset({("vc_t", "leaked")})
    for sql in ("SELECT id, leaked FROM vc_t",
                "SELECT vc_t.leaked FROM vc_t",
                "SELECT t.leaked FROM vc_t AS t",
                "SELECT id FROM vc_t WHERE leaked = 'x'"):
        refusal = check_opaque_columns(sqlglot.parse_one(sql, read="oracle"), opaque, "oracle")
        assert refusal is not None, sql
        assert refusal.subject == "leaked"
        assert refusal.repairable is True, "selecting another column is a real rewrite"

    clean = check_opaque_columns(
        sqlglot.parse_one("SELECT id, total FROM vc_t", read="oracle"), opaque, "oracle")
    assert clean is None


@pytest.mark.parametrize("sql, why", [
    ("SELECT vc_t.id FROM vc_t JOIN other USING (leaked)", "USING names it as a bare identifier"),
    ("SELECT id FROM vc_t NATURAL JOIN other", "NATURAL names no column at all"),
])
def test_a_join_key_reads_the_column_without_being_a_column_node(sql, why):
    """Both passed a guard that walked only `exp.Column`, while `WHERE leaked = 'x'` was refused
    -- the same channel one syntax over. A join evaluates the expression per row, and whether
    rows match leaks its value a bit at a time.

    NATURAL has nothing to check against, since the keys are whatever the two tables share, so
    any opaque column on a table in the statement is one it could be joining on.
    """
    import sqlglot

    from mnemiq.sql.authz_guard import check_opaque_columns

    refusal = check_opaque_columns(sqlglot.parse_one(sql, read="oracle"),
                                   frozenset({("vc_t", "leaked")}), "oracle")
    assert refusal is not None, why
    assert refusal.subject == "leaked"


def test_a_join_that_cannot_touch_the_opaque_column_still_answers():
    """The control, and it has to cover BOTH arms: the pair above is satisfied by refusing every
    join, and the NATURAL arm specifically by refusing every natural join on any source that has
    an opaque column anywhere."""
    import sqlglot

    from mnemiq.sql.authz_guard import check_opaque_columns

    opaque = frozenset({("vc_t", "leaked")})
    for sql in ("SELECT vc_t.id FROM vc_t JOIN other USING (id)",
                # NATURAL over tables that do NOT include the opaque column's table.
                "SELECT id FROM other NATURAL JOIN third"):
        assert check_opaque_columns(sqlglot.parse_one(sql, read="oracle"),
                                    opaque, "oracle") is None, sql

    # ...and the message tells the caller something they can act on. "Answer without that
    # column" is useless for a NATURAL join, which never names one.
    natural = check_opaque_columns(
        sqlglot.parse_one("SELECT id FROM vc_t NATURAL JOIN other", read="oracle"),
        opaque, "oracle")
    assert "explicit ON or USING" in natural.message
