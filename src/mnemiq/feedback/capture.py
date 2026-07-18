from __future__ import annotations

import hashlib
import json
import os

from mnemiq.contract import EvaluationCase, Example


def _case_id(question: str, sql: str) -> str:
    return "fb-" + hashlib.sha256(f"{question}\x00{sql}".encode()).hexdigest()[:12]


def _load(path: str) -> list[dict]:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return []


def _dump(path: str, rows: list[dict]) -> None:
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=2)


def capture_fix(question: str, sql: str, tables: list[str], source_id: str,
                golden_path: str, examples_path: str) -> str:
    """Turn a fixed failure into durable assets: a golden EvaluationCase (protected by the eval
    gate) and a captured Example (indexed at the next build). Idempotent on the derived id."""
    cid = _case_id(question, sql)
    case = EvaluationCase(id=cid, question=question, gold_sql=sql, answerable=True,
                          db_id=source_id, tags=["feedback"])
    example = Example(question=question, sql=sql, tables=list(tables),
                      object_id=tables[0] if tables else "")

    golden = _load(golden_path)
    if not any(r.get("id") == cid for r in golden):
        golden.append(case.model_dump(by_alias=True))
        _dump(golden_path, golden)

    examples = _load(examples_path)
    if not any(r.get("question") == question and r.get("sql") == sql for r in examples):
        examples.append(example.model_dump(by_alias=True))
        _dump(examples_path, examples)
    return cid
