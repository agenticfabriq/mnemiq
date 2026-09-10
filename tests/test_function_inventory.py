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
def test_a_postgres_attachment_reports_duckdbs_catalogue_and_that_is_correct():
    """Measured, because it is not obvious and an earlier version raised here instead.

    Queries run on the DuckDB connection and its binder resolves against its own catalogue. On
    a Postgres source defining `median(int) RETURNS 99`, `SELECT median(1)` returns DuckDB's
    aggregate and the qualified forms do not bind at all, so a Postgres UDF is unreachable from
    a generated query. Reporting DuckDB's non-internal names is therefore the right answer, and
    the Postgres function must NOT appear.

    A Postgres view body is the other question. Reading a view whose body calls that function
    returns 99, because it runs server-side where DuckDB's binder never looked. That wants
    `pg_proc` and is not what this method answers.
    """
    import os

    dsn = os.getenv("MNEMIQ_PG_DSN", "postgresql://mnemiq:mnemiq@localhost:5433/acme")
    fns = DuckDBAdapter.postgres(dsn, read_only=True).user_functions()

    assert "median" not in fns, "a Postgres UDF is not callable here and must not be reported"
    assert all(isinstance(f, str) for f in fns)
