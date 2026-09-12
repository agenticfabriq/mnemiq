"""The pure functions in `scripts/run_native_slm.py` decide a reported accuracy.

Every case here is a defect that actually shipped in that script and was found by
review rather than by use, which is why they are pinned rather than described.
"""

from __future__ import annotations

import datetime
import decimal
import importlib.util
import pathlib
import sqlite3
import time

import pyarrow as pa
import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_native_slm.py"
_spec = importlib.util.spec_from_file_location("run_native_slm", _PATH)
slm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(slm)


def _table(names, rows):
    """Build by position, the way the harness must, so duplicate names survive."""
    cols = list(zip(*rows)) if rows else [() for _ in names]
    return pa.Table.from_arrays([pa.array(list(c)) for c in cols], names=names)


class TestExtractSql:
    def test_a_fenced_block_wins(self):
        assert slm.extract_sql("thinking\n```sql\nSELECT 1\n```") == "SELECT 1"

    def test_the_last_fence_wins_because_these_models_answer_last(self):
        text = "```sql\nSELECT 1\n```\nwait, better:\n```sql\nSELECT 2\n```"
        assert slm.extract_sql(text) == "SELECT 2"

    def test_unfenced_takes_the_LAST_select_not_the_first(self):
        # `re.search` returns the leftmost match, so this used to hand back the discarded
        # draft -- unrunnable SQL, graded as a model error rather than an extraction bug.
        text = ("SELECT count(*) FROM t but that is wrong.\n"
                "Actually the answer is:\nSELECT name FROM t WHERE x=1")
        assert slm.extract_sql(text) == "SELECT name FROM t WHERE x=1"

    def test_no_sql_at_all_is_empty_not_an_exception(self):
        assert slm.extract_sql("I cannot answer that.") == ""
        assert slm.extract_sql("") == ""
        assert slm.extract_sql(None) == ""


class TestRunSqlConstruction:
    """Through `run_sql_sqlite`, so the construction that held the defect is what runs.

    The previous version of this test built both tables with the helper below, which
    already calls `from_arrays` -- so reverting the fix in `run_sql_sqlite` left every
    test green. This one fails if it is reverted.
    """

    def test_duplicate_output_names_survive_execution(self):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE t (name text, location text)")
        con.execute("INSERT INTO t VALUES ('b', 'c')")
        # `cur.description` is ['name', 'name', 'location'] -- 3 via from_arrays, 2 via a dict.
        got = slm.run_sql_sqlite(con, "SELECT 'a' AS name, name, location FROM t")
        assert got is not None
        assert got.num_columns == 3, "duplicate output names collapsed"
        assert got.column_names == ["name", "name", "location"]

    def test_a_query_that_cannot_run_is_None_not_an_empty_table(self):
        con = sqlite3.connect(":memory:")
        assert slm.run_sql_sqlite(con, "SELECT * FROM nope") is None

    def test_the_execution_cap_aborts_a_runaway(self):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE t (x integer)")
        con.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(2000)])
        t0 = time.time()
        got = slm.run_sql_sqlite(con, "SELECT COUNT(*) FROM t a, t b, t c", timeout_s=2.0)
        assert got is None, "the cap did not fire"
        assert time.time() - t0 < 15, "the cap fired far too late"


class TestResultKey:
    def test_duplicate_column_names_do_not_collapse(self):
        # `SELECT T2.name, T1.name, T1.location` is real gold in this corpus. Built through
        # a dict, the 3-column gold became 2 -- and so did a candidate missing a column,
        # which then graded correct.
        gold = _table(["name", "name", "location"], [("a", "b", "c")])
        assert gold.num_columns == 3
        candidate = _table(["name", "location"], [("b", "c")])
        assert slm._result_key(gold) != slm._result_key(candidate)

    def test_row_order_is_irrelevant(self):
        a = _table(["x"], [(1,), (2,)])
        b = _table(["x"], [(2,), (1,)])
        assert slm._result_key(a) == slm._result_key(b)

    def test_a_query_that_did_not_run_has_no_key(self):
        assert slm._result_key(None) is None


class TestVote:
    def test_the_majority_result_wins(self):
        t1, t2 = _table(["x"], [(1,)]), _table(["x"], [(2,)])
        sql, tbl, votes, groups = slm.vote(
            [("A", t1), ("B", t2), ("C", t1)])
        assert (sql, votes, groups) == ("A", 2, 2)
        assert tbl is t1

    def test_a_query_that_did_not_run_is_not_evidence(self):
        # Three non-runners must not out-vote one runner: they are absent, not agreeing.
        t = _table(["x"], [(1,)])
        sql, _, votes, groups = slm.vote(
            [("bad1", None), ("bad2", None), ("bad3", None), ("good", t)])
        assert (sql, votes, groups) == ("good", 1, 1)

    def test_all_candidates_failing_yields_no_table_and_no_votes(self):
        sql, tbl, votes, groups = slm.vote([("a", None), ("b", None)])
        assert (sql, tbl, votes, groups) == ("a", None, 0, 0)

    def test_semantically_equal_results_vote_together(self):
        # Different SQL, same rows: one vote, which is the point of execution-based voting.
        t1 = _table(["x"], [(1,)])
        t2 = _table(["x"], [(1,)])
        _, _, votes, groups = slm.vote([("q1", t1), ("q2", t2)])
        assert (votes, groups) == (2, 1)


class TestCell:
    def test_decimals_stay_numbers(self):
        # `default=str` stringified these, and an independent grader then disagreed on 11
        # answers that were correct.
        assert slm._cell(decimal.Decimal("109398.548387096774")) == pytest.approx(
            109398.548387096774)

    def test_dates_render_as_iso_not_python_repr(self):
        assert slm._cell(datetime.date(2012, 8, 26)) == "2012-08-26"

    def test_plain_values_pass_through(self):
        assert (slm._cell(1), slm._cell("a"), slm._cell(None)) == (1, "a", None)
