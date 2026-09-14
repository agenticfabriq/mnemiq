"""The pure functions in `scripts/run_native_slm.py` decide a reported accuracy.

Every case here is a defect that actually shipped in that script and was found by
review rather than by use, which is why they are pinned rather than described.
"""

from __future__ import annotations

import datetime
import decimal
import importlib.util
import os
import pathlib
import sqlite3
import sys
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


class TestBuildMessages:
    """The arctic style is only worth running if it is THEIR prompt, byte for byte.

    Verified once against their own `bird_eval/infer.py` by executing their source and
    diffing the result; these pin the strings so a later edit cannot quietly paraphrase
    what the model was RL-trained against.
    """

    def test_omnisql_is_one_user_turn_and_no_transport_extras(self):
        msgs, extra = slm.build_messages("omnisql", "SQLite", "CREATE TABLE t (a int);", "q?")
        assert [m["role"] for m in msgs] == ["user"]
        assert extra == {}
        assert "Task Overview:" in msgs[0]["content"]

    def test_arctic_moves_the_task_text_to_a_system_turn(self):
        msgs, _ = slm.build_messages("arctic", "SQLite", "CREATE TABLE t (a int);", "q?")
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[0]["content"].startswith("You are a data science expert.")
        # the two things their prompt DROPS -- their absence is the change being measured
        assert "Task Overview:" not in msgs[1]["content"]
        assert "deep breath" not in msgs[1]["content"]

    def test_the_prefill_is_exact(self):
        # The space before the newline is theirs. `.strip()` anywhere near this string, or a
        # "tidy" reflow, silently changes the tokens the assistant turn opens with.
        msgs, _ = slm.build_messages("arctic", "SQLite", "s", "q")
        assert msgs[2]["content"] == "Let me solve this step by step. \n<think>"

    def test_arctic_asks_the_server_to_continue_the_turn_not_start_one(self):
        # Without BOTH flags the server closes the assistant turn and the prefill becomes a
        # stray user-visible line instead of the opening of the model's own reasoning.
        _, extra = slm.build_messages("arctic", "SQLite", "s", "q")
        assert extra == {"continue_final_message": True, "add_generation_prompt": False}

    def test_the_schema_and_question_actually_reach_the_prompt(self):
        """Pinning the literal strings proves nothing about the two values that vary.

        `str.format` ignores an unused kwarg, so deleting `{schema}` from the template, or
        swapping the two kwargs, left every other test here green -- and the run would send
        a schema-less or role-swapped prompt, execute cleanly, and report the near-zero EX
        as a model result.
        """
        for style in ("omnisql", "arctic"):
            msgs, _ = slm.build_messages(style, "SQLite", "CREATE TABLE zz (q int);", "HOWMANY?")
            user = next(m["content"] for m in msgs if m["role"] == "user")
            assert "CREATE TABLE zz (q int);" in user, f"{style}: schema missing"
            assert "HOWMANY?" in user, f"{style}: question missing"
            # ...and not transposed: the schema must precede the question, under its header
            assert user.index("CREATE TABLE zz") < user.index("HOWMANY?"), f"{style}: swapped"
            assert "Database Schema:\nCREATE TABLE zz" in user, f"{style}: schema misplaced"
            assert "Question:\nHOWMANY?" in user, f"{style}: question misplaced"

    def test_the_engine_line_is_the_one_asked_for(self):
        # Both templates: `{engine}` is interpolated separately in each, and a test that
        # reads only one leaves the other's kwarg deletable -- `str.format` ignores it.
        for style in ("omnisql", "arctic"):
            msgs, _ = slm.build_messages(style, "PostgreSQL", "s", "q")
            user = next(m["content"] for m in msgs if m["role"] == "user")
            assert "Database Engine:\nPostgreSQL" in user, f"{style}: engine line wrong"

    def test_the_envelope_states_both_budgets(self):
        msgs, _ = slm.build_messages("arctic", "SQLite", "s", "q")
        assert "[Limited by 4K tokens]" in msgs[1]["content"]
        assert "[Limited by 1K tokens]" in msgs[1]["content"]

    def test_an_unknown_style_raises_rather_than_falling_back(self):
        # A typo must not silently select the other prompt: the two score ~15 points apart
        # on a DDL-trained model, and the report would name neither.
        with pytest.raises(ValueError):
            slm.build_messages("Arctic", "SQLite", "s", "q")


class TestGenerateReportsWhyItStopped:
    """`generate` must hand back the finish reasons, not just the text.

    A generation cut off at the token cap is an unfinished answer; `extract_sql`'s
    last-SELECT fallback turns one into a plausible query that grades as a model error.
    This covers `generate` handing the reasons back. It does NOT cover main()'s consumer:
    flipping `if why == "length"` to `"stop"`, or deleting that block, still leaves every
    test here green, because nothing in this suite runs `gen_one`. That half is untested
    and is recorded as untested rather than implied to be covered.
    """

    def _stub(self, monkeypatch, choices):
        import contextlib
        import io
        import json as _json

        @contextlib.contextmanager
        def fake_urlopen(req, timeout=None):
            yield io.BytesIO(_json.dumps({"choices": choices}).encode())

        monkeypatch.setattr(slm.urllib.request, "urlopen", fake_urlopen)

    def test_finish_reasons_come_back_alongside_the_text(self, monkeypatch):
        self._stub(monkeypatch, [
            {"message": {"content": "a"}, "finish_reason": "stop"},
            {"message": {"content": "b"}, "finish_reason": "length"},
        ])
        texts, reasons = slm.generate("http://x/v1", "m", [{"role": "user", "content": "q"}],
                                      timeout=1, max_tokens=10, n=2)
        assert texts == ["a", "b"]
        assert reasons == ["stop", "length"]

    def test_a_missing_finish_reason_is_empty_not_None(self, monkeypatch):
        # `None` would compare unequal to "length" too, but it also reads as "not truncated"
        # when the truth is "the server did not say" -- the absence/failure collapse again.
        self._stub(monkeypatch, [{"message": {"content": "a"}}])
        _, reasons = slm.generate("http://x/v1", "m", [{"role": "user", "content": "q"}],
                                  timeout=1, max_tokens=10)
        assert reasons == [""]

    def test_the_extras_reach_the_request_body(self, monkeypatch):
        # Without these the server closes the assistant turn and the prefilled `<think>`
        # becomes a completed message, so the arctic arm silently measures another prompt.
        seen = {}

        import contextlib
        import io
        import json as _json

        @contextlib.contextmanager
        def fake_urlopen(req, timeout=None):
            seen["body"] = _json.loads(req.data)
            yield io.BytesIO(b'{"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}')

        monkeypatch.setattr(slm.urllib.request, "urlopen", fake_urlopen)
        slm.generate("http://x/v1", "m", [{"role": "user", "content": "q"}], timeout=1,
                     max_tokens=10, extra={"continue_final_message": True})
        assert seen["body"]["continue_final_message"] is True


class TestUsableCandidates:
    """A truncated generation must not be able to grade CORRECT.

    It could, and did: the flag was metadata only, the unfinished text was executed like
    any other, and 30 of 88 truncations graded correct on the 2026-09-13 envelope run
    because `extract_sql` falls back to the last SELECT in a cut-off chain of thought.
    """

    def test_a_truncated_candidate_is_dropped(self):
        assert slm.usable_candidates(["SELECT 1", "SELECT 2"], ["length", "stop"]) == ["SELECT 2"]

    def test_all_truncated_yields_no_sql_not_no_candidates(self):
        # `[]` would be indistinguishable from a transport fault, which `None` already means.
        assert slm.usable_candidates(["SELECT 1"], ["length"]) == [""]

    def test_a_finished_candidate_survives_untouched(self):
        assert slm.usable_candidates(["SELECT 1"], ["stop"]) == ["SELECT 1"]

    def test_no_reasons_changes_nothing(self):
        # The error paths return no reasons; they must not be reinterpreted as truncation.
        assert slm.usable_candidates(["SELECT 1"], []) == ["SELECT 1"]

    def test_a_truncated_candidate_cannot_win_a_vote(self):
        # Voting grades the majority RESULT, so an unfinished candidate that agrees with
        # another could carry the case. Two truncated against one finished must not.
        kept = slm.usable_candidates(["BAD", "BAD", "SELECT 1"], ["length", "length", "stop"])
        assert kept == ["SELECT 1"]


class TestRunStatesEndToEnd:
    """The states the harness must not publish a number for, exercised through the script.

    Every one of these was unasserted: the state distinctions live in `main()`, which no
    unit test reaches, so inverting the rejected/transport ternary or swapping `all` for
    `any` on the truncation check left the whole suite green. The earlier version of this
    class needed the real BIRD tree and therefore SKIPPED in CI -- `.github/workflows/ci.yml`
    fetches no dataset -- so the mutation it was written for landed green on every PR. This
    one builds its own two-row corpus and runs anywhere.
    """

    def _corpus(self, tmp_path, questions=None):
        """A mini-dev tree with one table, one row, and (by default) two questions."""
        import json as _json
        import sqlite3 as _sq

        root = tmp_path / "MINIDEV"
        (root / "dev_databases" / "toy").mkdir(parents=True)
        con = _sq.connect(root / "dev_databases" / "toy" / "toy.sqlite")
        con.execute("CREATE TABLE t (a integer, b text)")
        con.execute("INSERT INTO t VALUES (1, 'x')")
        con.commit()
        con.close()
        (root / "dev_tables.json").write_text(_json.dumps(
            [{"db_id": "toy", "table_names_original": ["t"]}]))
        (root / "mini_dev_sqlite.json").write_text(_json.dumps(questions if questions is not None else [
            {"question_id": 1, "db_id": "toy", "question": "how many rows?",
             "SQL": "SELECT count(*) FROM t", "difficulty": "simple", "evidence": ""},
            {"question_id": 2, "db_id": "toy", "question": "what is b?",
             "SQL": "SELECT b FROM t", "difficulty": "simple", "evidence": ""},
        ]))
        return root

    def _serve(self, handler_body):
        import http.server
        import threading

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                handler_body(self)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)  # port 0: no collisions
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, srv.server_address[1]

    @staticmethod
    def _completion(sql, finish_reason):
        import json as _json

        def body(h):
            payload = _json.dumps({"choices": [
                {"message": {"content": f"```sql\n{sql}\n```"},
                 "finish_reason": finish_reason}]}).encode()
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(payload)))
            h.end_headers()
            h.wfile.write(payload)
        return body

    @staticmethod
    def _refuse(h):
        h.send_response(400)
        h.send_header("Content-Length", "2")
        h.end_headers()
        h.wfile.write(b"{}")

    def _run(self, handler_body, tmp_path, extra=(), questions=None):
        import json as _json
        import subprocess

        root = self._corpus(tmp_path, questions)
        srv, port = self._serve(handler_body)
        out = tmp_path / "run.jsonl"
        env = {**os.environ, "MNEMIQ_MINIDEV_DIR": str(root)}
        try:
            proc = subprocess.run(
                [sys.executable, str(_PATH), "--base-url", f"http://127.0.0.1:{port}/v1",
                 "--model", "m", "--backend", "sqlite", "--engine", "SQLite",
                 "--prompt-style", "arctic", "--concurrency", "1", "--timeout", "20",
                 "--out", str(out), *extra],
                capture_output=True, text=True, timeout=240, check=False, env=env)
        finally:
            srv.shutdown()
        combined = proc.stdout + proc.stderr
        rows = ([_json.loads(x) for x in out.read_text().splitlines() if x.strip()]
                if out.exists() else [])
        # The captured output is the diagnosis; asserting before returning it threw it away.
        return proc.returncode, combined, rows

    def test_a_finished_gold_answer_scores_and_exits_clean(self, tmp_path):
        """The positive control. Without it every assertion below could pass because the
        harness cannot score this corpus at all."""
        code, outp, rows = self._run(self._completion("SELECT count(*) FROM t", "stop"),
                                     tmp_path)
        assert code == 0, outp
        assert [r["outcome"] for r in rows] == ["correct", "wrong"], outp
        assert "NATIVE_CONTROL_EXIT=0" in outp

    def test_an_all_truncated_run_publishes_no_number(self, tmp_path):
        code, outp, rows = self._run(self._completion("SELECT count(*) FROM t", "length"),
                                     tmp_path)
        assert all(r["truncated"] for r in rows), outp
        assert all(r["outcome"] != "correct" for r in rows), "unfinished graded as right"
        assert "RUN VOID" in outp and "no-finished-candidate=2" in outp, outp
        assert code == 1 and "NATIVE_CONTROL_EXIT=1" in outp, outp
        # The live meter has to say so too, in its own words: the summary below says
        # TRUNCATED whatever the meter does, so a bare substring would not pin it.
        assert "2 TRUNCATED" in outp, "the progress meter gave no in-flight signal"
        # ...and the cause list must not name a fault that did not occur. Hoisting the
        # gold-failure counter out of its branch made this run blame the corpus, which is
        # the same misattribution as before pointing the other way.
        assert "produced no FINISHED candidate" in outp, outp   # the cause's own phrase
        assert "raise --max-tokens" in outp, outp                  # ...and its remedy
        assert "GOLD query that did not run" not in outp, outp
        assert "returned no extractable SQL" not in outp, outp

    def test_a_partly_truncated_case_is_not_reported_as_truncated(self, tmp_path):
        """`all` not `any`: the two counters send the operator to different fixes.

        Needs MIXED finish reasons to distinguish, which needs more than one candidate --
        with every candidate alike the two are the same expression. Here one candidate
        finished but carried no SQL and the other was cut off: the case produced no SQL,
        but not BECAUSE of the cap, so `--max-tokens` is the wrong advice.
        """
        import json as _json

        def body(h):
            payload = _json.dumps({"choices": [
                {"message": {"content": "I cannot answer that."}, "finish_reason": "stop"},
                {"message": {"content": "```sql\nSELECT 1\n```"}, "finish_reason": "length"},
            ]}).encode()
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(payload)))
            h.end_headers()
            h.wfile.write(payload)

        code, outp, _rows = self._run(body, tmp_path, extra=("--candidates", "2",
                                                             "--temperature", "0.8"))
        assert "empty-sql=2" in outp, outp
        assert "no-finished-candidate=0" in outp, outp
        # The fourth cause and its remedy. This state was reached by a test that read the
        # summary counters only, so deleting the whole `if n_empty_sql:` branch printed
        # "Cause: unknown" here with the suite green.
        assert "returned no extractable SQL" in outp, outp
        assert "check the prompt and the extractor" in outp, outp
        # NOT `"raise --max-tokens" not in outp`: the SUMMARY prints "Raise --max-tokens"
        # for this run, so that assertion passed on letter case alone and a copy-edit
        # lowercasing the summary would have failed it for an unrelated reason.
        assert "produced no FINISHED candidate" not in outp, outp
        assert code == 1 and "RUN VOID" in outp, outp

    def test_a_refused_run_voids_and_says_the_request_was_refused(self, tmp_path):
        code, outp, rows = self._run(self._refuse, tmp_path)
        assert [r.get("error") for r in rows] == ["rejected", "rejected"], outp
        assert "REFUSED" in outp and "RUN VOID" in outp, outp
        assert code == 1, outp


    def test_a_run_whose_gold_never_executes_publishes_no_number(self, tmp_path):
        """The third state the void claims to cover, and the only one that was unreached.

        Dropping `gold is not None` from the `n_scored` condition left every other test
        green, and under that mutation a corpus whose gold SQL cannot run reports EX=0.00%
        and exits 0 -- a number about the harness's own fixture, not the model.
        """
        broken = [{"question_id": 1, "db_id": "toy", "question": "q",
                   "SQL": "SELECT * FROM no_such_table", "difficulty": "simple",
                   "evidence": ""}]
        code, outp, rows = self._run(self._completion("SELECT count(*) FROM t", "stop"),
                                     tmp_path, questions=broken)
        assert [r["outcome"] for r in rows] == ["error"], outp
        assert "RUN VOID" in outp, outp
        # The cause must name OUR fixture. The old message said the cases "returned no SQL",
        # which was false here: the model answered with runnable SQL and the gold is what
        # failed, so a reader was sent to the prompt instead of to the corpus.
        assert "GOLD query that did not run" in outp, outp
        assert "check the corpus and the dialect" in outp, outp   # the actionable half
        assert "returned no extractable SQL" not in outp, outp
        assert "produced no FINISHED candidate" not in outp, outp
        assert code == 1 and "NATIVE_CONTROL_EXIT=1" in outp, outp

    def test_a_filter_that_selects_no_cases_publishes_no_number(self, tmp_path):
        """`and n` exempted this: a `--db` typo measured nothing and printed EX over all 0."""
        code, outp, _rows = self._run(self._completion("SELECT count(*) FROM t", "stop"),
                                      tmp_path, extra=("--db", "nosuchdb"))
        assert "EX=0.00% over all 0" not in outp, outp
        # Exit 1 alone is also what a crash or a silent bail gives, so assert the void's own
        # output the way the sibling tests do -- `if not cases: return 1` inserted after
        # `load_bird` exits 1 with neither string and used to pass this.
        assert "RUN VOID" in outp and "NATIVE_CONTROL_EXIT=1" in outp, outp
        assert "no cases were loaded at all" in outp, outp
        # The REMEDY as well as the phrase. No coverage claim is written here: three
        # successive versions of this comment asserted one -- two siblings when one was
        # pinned, three causes when there are four, then all four when the truncated
        # cause's phrase was still unpinned. The property is checkable by swapping each
        # cause's phrase for another's; the comment was not the place to record it.
        assert "check --db" in outp, outp
        assert "GOLD query that did not run" not in outp, outp
