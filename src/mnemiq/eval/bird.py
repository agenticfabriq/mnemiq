from __future__ import annotations

import json
import os

from mnemiq.contract import EvaluationCase


def bird_db_path(minidev_dir: str, db_id: str) -> str:
    return os.path.join(minidev_dir, "dev_databases", db_id, f"{db_id}.sqlite")


_DIALECT_FILE = {
    "sqlite": "mini_dev_sqlite.json",
    "postgresql": "mini_dev_postgresql.json",
    "mysql": "mini_dev_mysql.json",
}


def load_bird(
    minidev_dir: str,
    *,
    dialect: str = "sqlite",
    limit: int | None = None,
    db_ids: list[str] | None = None,
    difficulty: str | None = None,
    with_evidence: bool = True,
) -> list[EvaluationCase]:
    """BIRD mini-dev as EvaluationCases, routed by db_id.

    `dialect` picks the question file: the gold SQL differs by dialect (postgresql/mysql are
    transpiled+refined from the SQLite gold); question/evidence/db_id are shared.

    Evidence (BIRD's per-question external knowledge) is appended to the question by
    default -- the leaderboard convention, and the per-question analogue of our glossary.
    """
    with open(os.path.join(minidev_dir, _DIALECT_FILE[dialect])) as fh:
        records = json.load(fh)

    allowed = set(db_ids) if db_ids else None
    cases: list[EvaluationCase] = []
    for rec in records:
        if allowed is not None and rec["db_id"] not in allowed:
            continue
        if difficulty is not None and rec.get("difficulty") != difficulty:
            continue

        question = rec["question"]
        evidence = (rec.get("evidence") or "").strip()
        if with_evidence and evidence:
            question = f"{question}\n\nHint: {evidence}"

        cases.append(
            EvaluationCase(
                id=f"bird-{rec['question_id']}",
                question=question,
                gold_sql=rec["SQL"],
                db_id=rec["db_id"],
                answerable=True,
                tags=[rec["difficulty"]] if rec.get("difficulty") else [],
            )
        )
        if limit is not None and len(cases) >= limit:
            break

    return cases
