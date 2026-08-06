"""Spider 2.0-lite loader and CSV-gold grading. No LLM, no real dataset: everything runs
against a fixture tree, because the loader's job is layout and the grader's job is
best-of-alternatives -- both checkable without spending a token."""

import json

import pyarrow as pa

from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import IdentityContext, Trace
from mnemiq.eval.harness import Outcome
from mnemiq.eval.spider2 import (
    gold_alternatives,
    grade_alternatives,
    load_spider2_local,
    run_case_csv,
    spider2_db_path,
)


def _fixture(tmp_path):
    repo = tmp_path / "repo" / "spider2-lite"
    (repo / "resource" / "documents").mkdir(parents=True)
    (repo / "evaluation_suite" / "gold" / "sql").mkdir(parents=True)
    (repo / "evaluation_suite" / "gold" / "exec_result").mkdir(parents=True)

    records = [
        {"instance_id": "local001", "db": "alpha", "question": "how many widgets?",
         "external_knowledge": "widgets.md"},
        {"instance_id": "local002", "db": "beta", "question": "total sales?",
         "external_knowledge": None},
        {"instance_id": "bq001", "db": "cloudy", "question": "not local",
         "external_knowledge": None},
    ]
    with open(repo / "spider2-lite.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    (repo / "resource" / "documents" / "widgets.md").write_text("A widget is a thing.")
    (repo / "evaluation_suite" / "gold" / "sql" / "local001.sql").write_text(
        "SELECT count(*) FROM widget"
    )
    er = repo / "evaluation_suite" / "gold" / "exec_result"
    (er / "local001_a.csv").write_text("n\n3\n")
    (er / "local001_b.csv").write_text("n\n4\n")  # an alternative acceptable answer
    (er / "local002.csv").write_text("total\n100.0\n")
    return str(tmp_path)


def test_loader_keeps_local_cases_and_appends_the_knowledge_doc(tmp_path):
    cases = load_spider2_local(_fixture(tmp_path))

    assert [c.id for c in cases] == ["local001", "local002"], "cloud instances are filtered"
    assert "how many widgets?" in cases[0].question
    assert "A widget is a thing." in cases[0].question, "the reference doc rides as a hint"
    assert "Hint" not in cases[1].question


def test_loader_reads_gold_sql_where_published(tmp_path):
    cases = load_spider2_local(_fixture(tmp_path))
    assert cases[0].gold_sql == "SELECT count(*) FROM widget"
    assert cases[1].gold_sql == ""  # most local cases publish only result CSVs


def test_loader_filters_and_limits(tmp_path):
    root = _fixture(tmp_path)
    assert [c.id for c in load_spider2_local(root, db_ids=["beta"])] == ["local002"]
    assert len(load_spider2_local(root, limit=1)) == 1


def test_knowledge_can_be_left_out(tmp_path):
    cases = load_spider2_local(_fixture(tmp_path), with_knowledge=False)
    assert "A widget is a thing." not in cases[0].question


def test_gold_alternatives_finds_bare_and_suffixed_files(tmp_path):
    root = _fixture(tmp_path)
    assert len(gold_alternatives(root, "local001")) == 2
    assert len(gold_alternatives(root, "local002")) == 1
    assert gold_alternatives(root, "local999") == []


def test_db_path_is_flat(tmp_path):
    assert spider2_db_path("/data", "alpha") == "/data/databases/alpha.sqlite"


# --- grading ---------------------------------------------------------------------------


def test_any_alternative_can_make_a_case_correct():
    alts = [pa.table({"n": [3]}), pa.table({"n": [4]})]
    assert grade_alternatives(pa.table({"count": [4]}), alts) == Outcome.CORRECT
    assert grade_alternatives(pa.table({"count": [3]}), alts) == Outcome.CORRECT


def test_extra_columns_are_facts_not_correct():
    alts = [pa.table({"n": [3]})]
    candidate = pa.table({"n": [3], "context": ["x"]})
    assert grade_alternatives(candidate, alts) == Outcome.CORRECT_FACTS


def test_matching_no_alternative_is_wrong():
    alts = [pa.table({"n": [3]}), pa.table({"n": [4]})]
    assert grade_alternatives(pa.table({"n": [5]}), alts) == Outcome.WRONG


# --- the case runner's terminal states -------------------------------------------------


def _trace(sql: str) -> Trace:
    return Trace(
        question="q", plan_sql=sql, target_sql=sql, result_shape="scalar", timing={},
        enrichment_version="v1",
        identity=IdentityContext(tenant_id="t", principal_id="u", roles=[]),
        tables_used=["widget"],
    )


class _Adapter:
    dialect = "sqlite"

    def __init__(self, table: pa.Table):
        self._table = table

    def execute_arrow(self, sql, timeout_s=30):
        return self._table


def _case():
    from mnemiq.contract import EvaluationCase

    return EvaluationCase(id="local001", question="q", gold_sql="", db_id="alpha")


def test_an_answer_is_graded_against_the_csv_gold():
    engine = lambda q: AgentAnswer(answer="3", trace=_trace("SELECT 3"))  # noqa: E731
    result = run_case_csv(_case(), engine, _Adapter(pa.table({"n": [3]})),
                          [pa.table({"n": [3]})])
    assert result.outcome == Outcome.CORRECT
    assert result.executed and result.engine_row_count == 1


def test_a_deferral_on_an_answerable_case_is_deferred_wrongly():
    engine = lambda q: AgentAnswer(answer="cannot", deferred=True)  # noqa: E731
    result = run_case_csv(_case(), engine, _Adapter(pa.table({"n": [3]})),
                          [pa.table({"n": [3]})])
    assert result.outcome == Outcome.DEFERRED_WRONGLY


def test_an_engine_failure_is_an_error_never_a_deferral():
    engine = lambda q: AgentAnswer(answer="outage", failed=True)  # noqa: E731
    result = run_case_csv(_case(), engine, _Adapter(pa.table({"n": [3]})),
                          [pa.table({"n": [3]})])
    assert result.outcome == Outcome.ERROR


def test_running_on_the_benchmarks_own_sqlite_is_portable_by_construction():
    # Left as None this reads "unverified", and a leaderboard-comparable column shows n/a
    # for a run that is comparable -- understating it exactly where it matters most.
    engine = lambda q: AgentAnswer(answer="3", trace=_trace("SELECT 3"))  # noqa: E731
    result = run_case_csv(_case(), engine, _Adapter(pa.table({"n": [3]})),
                          [pa.table({"n": [3]})])

    assert result.portable_to_gold_engine is True


def test_another_executor_does_not_inherit_that_claim():
    # Run this path through a DuckDB attachment and the SQL is no longer known to run on
    # the benchmark's engine; the flag has to be earned again, not assumed.
    class _Attached(_Adapter):
        dialect = "duckdb"

    engine = lambda q: AgentAnswer(answer="3", trace=_trace("SELECT 3"))  # noqa: E731
    result = run_case_csv(_case(), engine, _Attached(pa.table({"n": [3]})),
                          [pa.table({"n": [3]})])

    assert result.portable_to_gold_engine is False


def test_a_case_that_never_executed_claims_nothing():
    engine = lambda q: AgentAnswer(answer="cannot", deferred=True)  # noqa: E731
    result = run_case_csv(_case(), engine, _Adapter(pa.table({"n": [3]})),
                          [pa.table({"n": [3]})])

    assert result.portable_to_gold_engine is None, "a declined case ran no SQL to be portable"


def test_a_case_with_no_published_gold_is_an_error_not_a_silent_pass():
    engine = lambda q: AgentAnswer(answer="3", trace=_trace("SELECT 3"))  # noqa: E731
    result = run_case_csv(_case(), engine, _Adapter(pa.table({"n": [3]})), [])
    assert result.outcome == Outcome.ERROR
    assert "no gold result" in result.answer
