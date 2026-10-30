"""The BEAVER runner: loading, annotation text, grants, and the run loop with resume.

Every fixture here is made up: BEAVER's questions and gold SQL are gated and never enter this
repository. The run loop runs on DuckDB, which stands in for BEAVER's MySQL; the MySQL source has
its own tests.
"""

import json

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import IdentityContext, Trace
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.eval import beaver
from mnemiq.eval.beaver import (
    GoldControl,
    Question,
    RunConfig,
    annotated,
    annotated_grants,
    control_failures,
    load_questions,
    run,
    summarize,
)


def _row(qid, question, sql, *, db="shop", tables=(), join_keys=(), mapping=None, knowledge=(),
         subs=(), sub_sqls=()):
    return {"id": qid, "question": question, "sql": sql, "db": db,
            "tables": json.dumps(list(tables)), "join_keys": json.dumps([list(p) for p in join_keys]),
            "column_mapping": json.dumps(mapping or {}), "domain_knowledge": json.dumps(list(knowledge)),
            "sub_questions": json.dumps(list(subs)), "sub_sqls": json.dumps(list(sub_sqls))}


def _write_set(root, name, rows):
    path = beaver.parquet_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


SHOP = [
    _row("shop_0", "How many orders are there?", "SELECT count(*) AS n FROM orders",
         tables=["ORDERS"]),
    _row("shop_1", "Total spend per customer?",
         "SELECT c.name, sum(o.amount) AS total FROM customers c JOIN orders o "
         "ON o.customer_id = c.id GROUP BY c.name ORDER BY c.name",
         tables=["customers", "Orders", "returns"], join_keys=[["orders.customer_id", "customers.id"]]),
    _row("shop_2", "Largest single order?", "SELECT max(amount) AS m FROM orders",
         tables=["orders"]),
]


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "beaver"
    _write_set(root, "shop", SHOP)
    return root


@pytest.fixture
def source(tmp_path):
    path = str(tmp_path / "shop.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name VARCHAR)")
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DOUBLE)")
    con.execute("INSERT INTO customers VALUES (1, 'ada'), (2, 'bo')")
    con.execute("INSERT INTO orders VALUES (1, 1, 10.0), (2, 1, 5.0), (3, 2, 7.5)")
    con.close()
    return path


def _control_cfg(dataset, tmp_path, **kw):
    return RunConfig(beaver_dir=dataset, sets=["shop"], out=tmp_path / "out" / "r.jsonl",
                     control=True, **kw)


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# -- loading ---------------------------------------------------------------------------------


def test_a_set_loads_in_file_order_with_its_annotations_decoded(dataset):
    qs = load_questions(dataset, "shop")
    assert [q.id for q in qs] == ["shop_0", "shop_1", "shop_2"]
    q = qs[1]
    assert q.db == "shop" and q.set == "shop" and q.gold_sql.startswith("SELECT c.name")
    assert q.tables == ["customers", "Orders", "returns"]
    assert q.join_keys == [["orders.customer_id", "customers.id"]]
    assert q.column_mapping == {} and q.sub_questions == []


def test_a_set_without_gold_loads_with_none_and_takes_its_database_from_the_map(tmp_path):
    # A held-back set: id and question only. `dw_real` runs on `dw`.
    _write_set(tmp_path, "dw_real", [{"id": "x_0", "question": "q?"}])
    (q,) = load_questions(tmp_path, "dw_real")
    assert q.gold_sql is None and q.db == "dw" and q.tables == [] and q.column_mapping == {}


def test_an_unknown_set_without_a_db_column_is_refused(tmp_path):
    _write_set(tmp_path, "mystery", [{"id": "x_0", "question": "q?"}])
    with pytest.raises(ValueError, match="no `db` column"):
        load_questions(tmp_path, "mystery")


def test_a_missing_set_names_the_file_it_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope-00000-of-00001.parquet"):
        load_questions(tmp_path, "nope")


# -- annotations -----------------------------------------------------------------------------


def _annotated_question():
    return Question(
        id="a_0", set="s", db="d", question="Which rooms are free?",
        tables=["rooms", "bookings"], join_keys=[["rooms.id", "bookings.room_id"]],
        column_mapping={"free": ["rooms.status", "rooms.status"], "room": ["rooms.id"]},
        domain_knowledge=["free means status = 'F'"],
        sub_questions=["List rooms", "Drop booked ones"],
        sub_sqls=["SELECT id FROM rooms", "SELECT room_id FROM bookings"])


def test_no_annotations_asks_the_question_alone():
    assert annotated(_annotated_question(), "none") == "Which rooms are free?"


def test_all_annotations_render_in_the_measured_wording():
    # The exact text matters: a rerun must ask what the measured runs asked.
    assert annotated(_annotated_question(), "all") == "\n".join([
        "Which rooms are free?",
        "",
        "Annotations for this question:",
        "Tables: rooms, bookings",
        "Join keys: rooms.id = bookings.room_id",
        'Column mapping: "free" -> rooms.status; "room" -> rooms.id',
        "Domain knowledge: free means status = 'F'",
        "Sub-questions and their SQL:",
        "1. List rooms\n   SQL: SELECT id FROM rooms",
        "2. Drop booked ones\n   SQL: SELECT room_id FROM bookings",
    ])


def test_no_sub_sql_drops_the_sub_questions_and_their_sql_and_keeps_the_rest():
    text = annotated(_annotated_question(), "no-sub-sql")
    assert text == annotated(_annotated_question(), "all").split("\nSub-questions")[0]
    assert "List rooms" not in text and "SELECT" not in text


def test_empty_annotation_fields_add_no_line_and_sub_questions_without_sql_add_no_sql_line():
    q = Question(id="b", set="s", db="d", question="q?", tables=["t"], sub_questions=["first"])
    assert annotated(q, "all") == "q?\n\nAnnotations for this question:\nTables: t\n" \
                                  "Sub-questions and their SQL:\n1. first"


def test_an_unknown_annotation_mode_is_refused():
    with pytest.raises(ValueError, match="annotations must be one of"):
        annotated(_annotated_question(), "some")


def test_annotated_grants_match_tables_without_regard_to_case_and_report_the_rest(source):
    snapshot = enrich_structural(DuckDBAdapter.duckdb(source), "shop")
    q = load_questions_from(SHOP[1])
    grants, unmatched = annotated_grants(snapshot, q)
    assert grants.objects == frozenset({"customers", "orders"})
    assert unmatched == ["returns"]


def load_questions_from(row):
    return Question(id=row["id"], set="shop", db="shop", question=row["question"],
                    gold_sql=row["sql"], tables=json.loads(row["tables"]))


# -- the run loop ----------------------------------------------------------------------------


def test_the_control_grades_every_question_correct_and_writes_one_line_each(dataset, source,
                                                                             tmp_path):
    cfg = _control_cfg(dataset, tmp_path)
    rows = run(cfg, source_for=lambda db: DuckDBAdapter.duckdb(source))
    assert [r["outcome"] for r in rows] == ["correct"] * 3
    assert _lines(cfg.out) == rows
    first = rows[0]
    assert first["sql"] == SHOP[0]["sql"] and first["db"] == "shop" and first["set"] == "shop"
    assert first["control"] is True and first["graded"] is True and first["model"] is None
    assert first["gold_gradable"] is True and first["reason_code"] is None
    assert control_failures(rows) == []
    assert summarize(rows) == {"shop": {"correct": 3}}


def test_a_resumed_run_asks_only_what_the_file_lacks_and_the_limit_applies_first(
        dataset, source, tmp_path):
    opened = []

    def source_for(db):
        opened.append(db)
        return DuckDBAdapter.duckdb(source)

    run(_control_cfg(dataset, tmp_path, limit=1), source_for=source_for)
    again = run(_control_cfg(dataset, tmp_path, limit=2), source_for=source_for)
    assert [r["id"] for r in again] == ["shop_1"]
    assert [r["id"] for r in _lines(tmp_path / "out" / "r.jsonl")] == ["shop_0", "shop_1"]
    # A third run with nothing left opens no source at all.
    run(_control_cfg(dataset, tmp_path, limit=2), source_for=source_for)
    assert len(opened) == 2


def test_a_file_from_another_configuration_is_refused(dataset, source, tmp_path):
    out = tmp_path / "r.jsonl"
    run(RunConfig(beaver_dir=dataset, sets=["shop"], out=out, control=True, limit=1),
        source_for=lambda db: DuckDBAdapter.duckdb(source))
    with pytest.raises(ValueError, match="another run .control"):
        run(RunConfig(beaver_dir=dataset, sets=["shop"], out=out, grade=False),
            source_for=lambda db: DuckDBAdapter.duckdb(source),
            engine_for=lambda adapter, db: (_answering("SELECT 1"), None))


def test_a_last_line_cut_short_is_asked_again(dataset, source, tmp_path):
    cfg = _control_cfg(dataset, tmp_path, limit=2)
    run(cfg, source_for=lambda db: DuckDBAdapter.duckdb(source))
    text = cfg.out.read_text()
    cfg.out.write_text(text[: len(text) - 20])  # the second row, cut by a killed run
    again = run(cfg, source_for=lambda db: DuckDBAdapter.duckdb(source))
    assert [r["id"] for r in again] == ["shop_1"]
    lines = cfg.out.read_text().splitlines()
    assert json.loads(lines[-1])["id"] == "shop_1" and len(lines) == 3  # the cut line stays put


def test_ids_select_questions_and_dw_real_shares_its_database_source(tmp_path, source):
    root = tmp_path / "beaver"
    _write_set(root, "dw", [_row("dw_0", "q", "SELECT 1 AS x", db="dw")])
    _write_set(root, "dw_real", [_row("dw_real_0", "q", "SELECT 2 AS x", db="dw"),
                                 _row("dw_real_1", "q", "SELECT 3 AS x", db="dw")])
    opened = []

    def source_for(db):
        opened.append(db)
        return DuckDBAdapter.duckdb(source)

    rows = run(RunConfig(beaver_dir=root, sets=["dw", "dw_real"], out=tmp_path / "r.jsonl",
                         control=True, ids=frozenset({"dw_0", "dw_real_1"})),
               source_for=source_for)
    assert [r["id"] for r in rows] == ["dw_0", "dw_real_1"]
    assert opened == ["dw"]


def test_grading_a_set_without_gold_is_refused_before_anything_is_asked(tmp_path):
    _write_set(tmp_path, "dw", [{"id": "x_0", "question": "q?"}])
    with pytest.raises(ValueError, match="pass --no-grade"):
        run(RunConfig(beaver_dir=tmp_path, sets=["dw"], out=tmp_path / "r.jsonl", control=True),
            source_for=lambda db: pytest.fail("no source should open"))


def test_the_control_needs_gold():
    with pytest.raises(ValueError, match="needs gold"):
        run(RunConfig(beaver_dir="x", sets=["dw"], out="r.jsonl", control=True, grade=False))


def _trace(sql):
    return Trace(question="q", plan_sql=sql, target_sql=sql, result_shape="table", timing={},
                 enrichment_version="v1", identity=IdentityContext(tenant_id="t", principal_id="u"))


def _answering(sql):
    def ask(question, grants=None):
        return AgentAnswer(answer="an answer", trace=_trace(sql))
    return ask


def test_without_grading_a_row_records_the_sql_or_null_with_the_reason(tmp_path):
    _write_set(tmp_path, "dw", [{"id": "x_0", "question": "answer me"},
                                {"id": "x_1", "question": "decline me"},
                                {"id": "x_2", "question": "break me"}])

    def ask(question, grants=None):
        if question == "decline me":
            from mnemiq.contract.seams import DeferralReason
            return AgentAnswer(answer="no", deferred=True, trace=_trace("SELECT vetoed"),
                               reason_code=DeferralReason.SOURCE_UNAVAILABLE)
        if question == "break me":
            raise RuntimeError("engine fell over")
        return AgentAnswer(answer="yes", trace=_trace("SELECT 42"))

    rows = run(RunConfig(beaver_dir=tmp_path, sets=["dw"], out=tmp_path / "r.jsonl", grade=False),
               source_for=lambda db: object(), engine_for=lambda adapter, db: (ask, None))
    answered, declined, broke = rows
    assert answered["sql"] == "SELECT 42" and "outcome" not in answered
    assert declined["sql"] is None and declined["reason_code"] == "source_unavailable"
    assert broke["sql"] is None and broke["error"] == "engine fell over"
    assert summarize(rows) == {"dw": {"answered": 1, "declined": 1, "error": 1}}


def test_with_annotations_the_engine_reads_them_and_sees_only_the_annotated_tables(
        dataset, source, tmp_path):
    adapter = DuckDBAdapter.duckdb(source)
    snapshot = enrich_structural(adapter, "shop")
    seen = []

    def ask(question, grants=None):
        seen.append((question, grants))
        return AgentAnswer(answer="a", trace=_trace("SELECT count(*) AS n FROM orders"))

    rows = run(RunConfig(beaver_dir=dataset, sets=["shop"], out=tmp_path / "r.jsonl",
                         annotations="all", limit=2),
               source_for=lambda db: adapter, engine_for=lambda a, db: (ask, snapshot))
    (q0, g0), (q1, g1) = seen
    assert q0.startswith("How many orders are there?\n\nAnnotations for this question:")
    assert g0.objects == frozenset({"orders"})
    assert g1.objects == frozenset({"customers", "orders"})
    assert rows[1]["granted"] == ["customers", "orders"]
    assert rows[1]["grant_unmatched"] == ["returns"]
    assert [r["outcome"] for r in rows] == ["correct", "wrong"]


def test_a_control_row_whose_gold_cannot_be_graded_is_not_a_control_failure():
    rows = [{"outcome": "correct", "gold_gradable": True},
            {"outcome": "error", "gold_gradable": False},
            {"outcome": "wrong", "gold_gradable": True}]
    assert control_failures(rows) == [rows[2]]


def test_the_control_answers_with_the_gold_it_was_given():
    control = GoldControl()
    control.gold = "SELECT 7"
    answer = control("q?")
    assert answer.trace.target_sql == "SELECT 7" and not answer.deferred and not answer.failed
