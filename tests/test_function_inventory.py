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
    """The distinction the type exists for, and the twelfth instance of it in this codebase.

    A source that defines nothing and a source that could not be asked both produce an empty
    set. Reading the second as the first is what let `check_views` approve on a failed discovery
    and refuse on a successful one, from the identical call.
    """
    answered_none = FunctionInventory(())
    could_not_ask = FunctionInventory.unavailable("permission denied")
    nobody_asked = FunctionInventory.never_asked()

    assert len(answered_none) == len(could_not_ask) == len(nobody_asked) == 0
    assert answered_none.certain is True, "an empty ANSWER licenses treating names as builtins"
    assert could_not_ask.certain is False
    assert nobody_asked.certain is False
    # ...and the two negative cases are themselves distinct, because they differ in whose fault
    # it is: an adapter with no inventory method is not an outage.
    assert (could_not_ask.available, could_not_ask.asked) == (False, True)
    assert (nobody_asked.available, nobody_asked.asked) == (True, False)


def test_names_match_regardless_of_case():
    """SQL folds case and the catalogue does not agree with itself across engines, so a set of
    raw strings would answer `defines('MEDIAN')` differently from `defines('median')`."""
    inv = FunctionInventory({"My_UDF", "MEDIAN"})
    assert inv.defines("my_udf") and inv.defines("MEDIAN") and inv.defines("Median")
    assert not inv.defines("count")


@pytest.fixture()
def duckdb_source():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "src.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT, amount INT)")
    con.execute("CREATE MACRO all_ssns() AS 1")     # the exfiltration fixture's shape
    con.execute("CREATE MACRO median_(x) AS x")     # a UDF wearing a builtin-ish name
    con.close()
    return path


def test_duckdb_reports_what_it_defines_and_not_its_builtins(duckdb_source):
    """`internal` is the discriminator. DuckDB marks its own 945 builtins internal and anything
    a deployment adds comes back false, which is the fact parsing cannot supply."""
    fns = DuckDBAdapter.duckdb(duckdb_source).user_functions()

    assert "all_ssns" in fns and "median_" in fns
    for builtin in ("count", "sum", "median", "upper", "date_trunc"):
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
