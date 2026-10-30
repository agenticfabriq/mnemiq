"""BEAVER through mnemiq: load the questions, render the annotations, run and checkpoint.

BEAVER is MIT's enterprise text-to-SQL benchmark. Its questions and gold SQL are gated, so
nothing from it is kept in this repository: this module reads the Hugging Face parquet files
(`query/data/<set>-00000-of-00001.parquet`) from a local copy, and answers on BEAVER's own MySQL
through `mnemiq.eval.mysql_source`, since BEAVER's gold depends on MySQL's type rules.

A run writes one JSON line per question and resumes from its own file. With gold present it grades
each answer with the harness every benchmark here uses (`run_case`); with `grade=False` it only
records the SQL, for a set whose gold is held back.

Annotations follow BEAVER's Setting 2: the question, then its tables, join keys, column mapping,
domain knowledge and sub-questions with their SQL. With annotations the question is also granted
only the tables its annotation names, as in the runs behind our published BEAVER numbers.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from mnemiq.agent.loop import AgentAnswer
from mnemiq.authz.grants import GrantSet, classified_levels
from mnemiq.contract import EvaluationCase, Snapshot, Trace
from mnemiq.eval.engine import IDENTITY
from mnemiq.eval.harness import Outcome, run_case
from mnemiq.sql.guard import MAX_ROWS

# Each public set and the database it runs on: `dw_real` asks real users' questions of `dw`.
# A set not listed here takes its database from the parquet's own `db` column.
SETS = {"dw": "dw", "dw_real": "dw", "nova": "nova", "neutron": "neutron"}

# `no-sub-sql` leaves out the sub-questions together with their SQL, as the measured arm did.
ANNOTATIONS = ("none", "all", "no-sub-sql")

SUB_QUESTIONS = "Sub-questions and their SQL:"


@dataclass
class Question:
    id: str
    set: str
    db: str
    question: str
    gold_sql: str | None = None
    tables: list[str] = field(default_factory=list)
    join_keys: list[list[str]] = field(default_factory=list)
    column_mapping: dict[str, list[str]] = field(default_factory=dict)
    domain_knowledge: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    sub_sqls: list[str] = field(default_factory=list)


def parquet_path(beaver_dir: str | Path, set_name: str) -> Path:
    return Path(beaver_dir) / "query" / "data" / f"{set_name}-00000-of-00001.parquet"


def _decoded(value, empty):
    """An annotation field: BEAVER stores each one as JSON text, and an empty one as ''."""
    return json.loads(value) if value else empty


def load_questions(beaver_dir: str | Path, set_name: str) -> list[Question]:
    """Every question in one set, in the file's order. `id` and `question` are required; gold SQL
    and the annotations are read when the file has them, so a set whose gold is held back loads
    with `gold_sql` None."""
    path = parquet_path(beaver_dir, set_name)
    if not path.exists():
        raise FileNotFoundError(f"no BEAVER set {set_name!r}: {path} does not exist")
    rows = pq.read_table(path).to_pylist()
    if rows and not {"id", "question"} <= rows[0].keys():
        raise ValueError(f"{path} has no `id` and `question` columns")
    questions = []
    for row in rows:
        db = row.get("db") or SETS.get(set_name)
        if not db:
            raise ValueError(f"{path}: no `db` column, and {set_name!r} is not a known set")
        questions.append(Question(
            id=row["id"], set=set_name, db=db, question=row["question"],
            gold_sql=row.get("sql") or None,
            tables=_decoded(row.get("tables"), []),
            join_keys=_decoded(row.get("join_keys"), []),
            column_mapping=_decoded(row.get("column_mapping"), {}),
            domain_knowledge=_decoded(row.get("domain_knowledge"), []),
            sub_questions=_decoded(row.get("sub_questions"), []),
            sub_sqls=_decoded(row.get("sub_sqls"), []),
        ))
    return questions


def annotated(q: Question, mode: str) -> str:
    """The text mnemiq is asked: the question alone (`none`), or followed by BEAVER's annotations
    for it. The wording is the measured runs' own, so a rerun asks the same text."""
    if mode not in ANNOTATIONS:
        raise ValueError(f"annotations must be one of {ANNOTATIONS}, not {mode!r}")
    if mode == "none":
        return q.question
    lines = [q.question, "", "Annotations for this question:", "Tables: " + ", ".join(q.tables)]
    if q.join_keys:
        lines.append("Join keys: " + "; ".join(" = ".join(pair) for pair in q.join_keys))
    if q.column_mapping:
        lines.append("Column mapping: " + "; ".join(
            f'"{phrase}" -> {", ".join(dict.fromkeys(cols))}'
            for phrase, cols in q.column_mapping.items()))
    if q.domain_knowledge:
        lines.append("Domain knowledge: " + "; ".join(q.domain_knowledge))
    if q.sub_questions and mode == "all":
        lines.append(SUB_QUESTIONS)
        sqls = q.sub_sqls or [""] * len(q.sub_questions)
        for i, (sub, sql) in enumerate(zip(q.sub_questions, sqls, strict=False), 1):
            lines.append(f"{i}. {sub}" + (f"\n   SQL: {sql}" if sql else ""))
    return "\n".join(lines)


def annotated_grants(snapshot: Snapshot, q: Question) -> tuple[GrantSet, list[str]]:
    """The tables the question's annotation names, matched to the snapshot without regard to case
    (BEAVER's gold and its dumps disagree on case), with every classified level cleared, as every
    benchmark arm here clears them. Returns the grants and the annotated names that matched no
    table, which a run records rather than hides."""
    ids = {b.object_id.lower(): b.object_id for b in snapshot.source_bindings}
    matched = frozenset(ids[t.lower()] for t in q.tables if t.lower() in ids)
    unmatched = sorted(t for t in q.tables if t.lower() not in ids)
    return GrantSet(matched, pii_clearance=classified_levels(snapshot)), unmatched


class GoldControl:
    """An engine that answers every question with its own gold SQL. Every row it grades must
    come out correct, which proves the grading path (gold and answer through the same source, the
    same grader) before any model runs."""

    def __init__(self) -> None:
        self.gold = ""

    def __call__(self, question: str, grants: GrantSet | None = None) -> AgentAnswer:
        trace = Trace(question=question, plan_sql=self.gold, target_sql=self.gold,
                      result_shape="table", timing={}, enrichment_version="control",
                      identity=IDENTITY)
        return AgentAnswer(answer="(control: the gold SQL)", trace=trace)


Ask = Callable[..., AgentAnswer]


def structural_engine(adapter, db: str, settings, cache_dir: str | Path,
                      engine: str) -> tuple[Ask, Snapshot]:
    """mnemiq over one database: structural enrichment (no model writes descriptions), then the
    engine every benchmark here builds (`build_engine`). The snapshot is cached per database and
    engine commit; a tree with uncommitted changes is never cached, since its commit does not say
    what code wrote it."""
    from mnemiq.enrichment.pipeline import enrich_structural
    from mnemiq.eval.engine import build_engine

    cacheable = bool(engine) and not engine.endswith("-dirty")
    path = Path(cache_dir) / f"snapshot-{db}-{engine.replace(' ', '_')}.json"
    if cacheable and path.exists():
        snapshot = Snapshot.model_validate_json(path.read_text())
    else:
        snapshot = enrich_structural(adapter, db)
        if cacheable:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(snapshot.model_dump_json())
    ask, _ = build_engine(snapshot, adapter, settings)
    return ask, snapshot


@dataclass
class RunConfig:
    beaver_dir: Path
    sets: list[str]
    out: Path
    annotations: str = "none"
    grade: bool = True
    control: bool = False
    limit: int | None = None  # per set, applied before resume skips, so a rerun asks the same
    ids: frozenset[str] | None = None
    cache_dir: Path = Path("eval-reports/beaver-cache")


# The fields a resumed file must agree on: a file mixing two of any is two measurements.
_RUN_KEYS = ("annotations", "control", "graded", "model", "engine")


def _resume(out: Path, header: dict) -> set[tuple[str, str]]:
    """The (set, id) pairs `out` already holds, after checking its rows were written by this
    configuration. A last line cut short by a killed run is dropped and that question asked again;
    a malformed line anywhere else stops the run, since appending would bury it."""
    if not out.exists():
        return set()
    lines = out.read_text().splitlines()
    done = set()
    for n, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if n == len(lines):
                break
            raise ValueError(f"{out}: line {n} is not JSON; fix or move the file") from None
        for key in _RUN_KEYS:
            if row.get(key) != header[key]:
                raise ValueError(
                    f"{out} holds rows from another run ({key}: {row.get(key)!r}, this run "
                    f"{header[key]!r}); write to a new --out")
        done.add((row["set"], row["id"]))
    if lines and not out.read_text().endswith("\n"):
        with out.open("a") as fh:
            fh.write("\n")
    return done


def _code(value) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


def _gold_gradable(gold_row_count: int | None) -> bool:
    """Whether the gold ran and fits the row limit every answer faces, so the row can be graded."""
    return gold_row_count is not None and gold_row_count <= MAX_ROWS


def run(cfg: RunConfig, *, settings=None, source_for: Callable[[str], object] | None = None,
        engine_for: Callable[[object, str], tuple[Ask, Snapshot | None]] | None = None,
        on_row: Callable[[dict], None] | None = None) -> list[dict]:
    """Ask every selected question once and append its row to `cfg.out`; returns the new rows.

    `source_for(db)` opens the database a set runs on (BEAVER's MySQL by default) and
    `engine_for(adapter, db)` builds mnemiq over it; both are seams for tests and for a caller
    with its own source.
    """
    from mnemiq.eval.bird_runner import source_rev

    if cfg.annotations not in ANNOTATIONS:
        raise ValueError(f"annotations must be one of {ANNOTATIONS}, not {cfg.annotations!r}")
    if cfg.control and not cfg.grade:
        raise ValueError("the control answers with the gold SQL, so it needs gold: drop --no-grade")
    if not cfg.control and settings is None and engine_for is None:
        from mnemiq.config import Settings

        settings = Settings.from_env()
        settings.assert_local_only()
    if source_for is None:
        from mnemiq.eval.mysql_source import MySQLSource

        local_only = bool(getattr(settings, "local_only", False))

        def source_for(db):
            return MySQLSource(db, local_only=local_only)
    engine = source_rev()
    header = {"annotations": cfg.annotations, "control": cfg.control, "graded": cfg.grade,
              "model": None if cfg.control else getattr(settings, "llm_model", None),
              "engine": engine}
    done = _resume(cfg.out, header)
    control = GoldControl() if cfg.control else None
    if engine_for is None:
        def engine_for(adapter, db):
            if control is not None:
                return control, None
            return structural_engine(adapter, db, settings, cfg.cache_dir, engine)

    opened: dict[str, tuple[object, Ask, Snapshot | None]] = {}  # dw and dw_real share `dw`
    written: list[dict] = []
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cfg.out.open("a") as fh:
            for set_name in cfg.sets:
                questions = load_questions(cfg.beaver_dir, set_name)
                if cfg.ids is not None:
                    questions = [q for q in questions if q.id in cfg.ids]
                questions = [q for q in questions[: cfg.limit] if (q.set, q.id) not in done]
                if cfg.grade and any(q.gold_sql is None for q in questions):
                    raise ValueError(f"set {set_name!r} has no gold SQL to grade against; "
                                     "pass --no-grade to record the SQL only")
                for q in questions:
                    if q.db not in opened:
                        adapter = source_for(q.db)
                        opened[q.db] = (adapter, *engine_for(adapter, q.db))
                    adapter, ask, snapshot = opened[q.db]
                    row = _ask_one(q, cfg, header, adapter, ask, snapshot, control)
                    fh.write(json.dumps(row, default=str) + "\n")
                    fh.flush()
                    written.append(row)
                    if on_row is not None:
                        on_row(row)
    finally:
        for adapter, _, _ in opened.values():
            close = getattr(adapter, "close", None)
            if close is not None:
                close()
    return written


def _ask_one(q: Question, cfg: RunConfig, header: dict, adapter, ask: Ask,
             snapshot: Snapshot | None, control: GoldControl | None) -> dict:
    text = annotated(q, cfg.annotations)
    grants, unmatched = (None, [])
    if cfg.annotations != "none" and snapshot is not None:
        grants, unmatched = annotated_grants(snapshot, q)
    if control is not None:
        control.gold = q.gold_sql or ""
    answers: list[AgentAnswer] = []

    def asked(question: str) -> AgentAnswer:
        answer = ask(question) if grants is None else ask(question, grants=grants)
        answers.append(answer)
        return answer

    row = {"id": q.id, "set": q.set, "db": q.db, **header}
    started = time.perf_counter()
    if cfg.grade:
        case = EvaluationCase(id=q.id, question=text, gold_sql=q.gold_sql, answerable=True,
                              db_id=q.db)
        result = run_case(case, asked, adapter)
        row.update(outcome=str(result.outcome), gold_row_count=result.gold_row_count,
                   engine_row_count=result.engine_row_count,
                   gold_gradable=_gold_gradable(result.gold_row_count),
                   verify_layer=result.verify_layer, verify_confidence=result.verify_confidence)
        error = result.answer if not answers else None  # the engine raised before answering
        if (answers and result.outcome is Outcome.ERROR
                and not answers[-1].failed and not answers[-1].deferred):
            # Answered, then not graded: the gold failed or is over the row limit, the answer
            # failed when the grader re-ran it, or the grader could not decide. The harness puts
            # the reason in its answer text.
            row["grade_note"] = result.answer
    else:
        error = None
        try:
            asked(text)
        except Exception as exc:  # noqa: BLE001 -- one question's failure is that row's record
            error = str(exc)
    answer = answers[-1] if answers else None
    answered = answer is not None and not answer.deferred and not answer.failed
    row.update(
        sql=answer.trace.target_sql if answered and answer.trace else None,
        reason_code=_code(answer.reason_code) if answer is not None else None,
        answer=answer.answer if answer is not None else None,
        error=error,
        s=round(time.perf_counter() - started, 1),
    )
    if grants is not None:
        row.update(granted=sorted(grants.objects), grant_unmatched=unmatched)
    return row


def summarize(rows: Iterable[dict]) -> dict[str, Counter]:
    """Per set: outcomes when graded; answered, declined and errors when not."""
    by_set: dict[str, Counter] = {}
    for row in rows:
        counts = by_set.setdefault(row["set"], Counter())
        if row.get("graded"):
            counts[row["outcome"]] += 1
        elif row.get("error"):
            counts["error"] += 1
        else:
            counts["answered" if row.get("sql") else "declined"] += 1
    return by_set


def control_failures(rows: Iterable[dict]) -> list[dict]:
    """Control rows that should have graded correct and did not. A row whose gold failed or is
    over the row limit is not graded, so it is not a failure of the grading path."""
    return [r for r in rows
            if r.get("gold_gradable") and r.get("outcome") != Outcome.CORRECT.value]
