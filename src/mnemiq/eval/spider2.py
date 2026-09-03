"""Spider 2.0-lite (local subset) as EvaluationCases, plus its official grading rule.

Spider 2.0-lite differs from Spider 1.0 in three ways that matter here:

  * questions are real-world analytics over enterprise-shaped schemas, not toy joins;
  * there is no gold SQL for most instances -- correctness is defined by stored gold *result*
    CSVs, several of which may be acceptable for one question;
  * grading is column containment, not row equality: every gold column must appear somewhere
    in the prediction, with extra prediction columns allowed.

That last rule is mnemiq's got-the-facts, which is why Spider's headline metric is got-facts
rather than exact match. `compare_tables` below is a port of Spider 2.0's own
`compare_pandas_table` so the numbers line up with the official evaluator.
"""

from __future__ import annotations

import glob
import json
import math
import os

from mnemiq.contract import EvaluationCase


def load_spider2_local(
    jsonl_path: str,
    *,
    limit: int | None = None,
    db_ids: list[str] | None = None,
    documents_dir: str | None = None,
    with_knowledge: bool = True,
) -> list[EvaluationCase]:
    """The 135 `local*` instances -- the ones backed by downloadable SQLite databases.

    13 of them name an `external_knowledge` document -- a definition the question leans on
    ("overtake label", "RFM segment") that cannot be inferred from the schema. Spider 2.0
    expects the solver to have read it, so with_knowledge=True appends it to the question,
    which is the benchmark's own protocol. with_knowledge=False is the no-extra-context
    baseline: defensible, but a strictly harder task than the published one, and the two
    conditions must not be compared to each other.
    """
    allowed = set(db_ids) if db_ids else None
    if documents_dir is None:
        documents_dir = os.path.join(os.path.dirname(jsonl_path), "resource", "documents")

    cases: list[EvaluationCase] = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec["instance_id"].startswith("local"):
                continue
            if allowed is not None and rec["db"] not in allowed:
                continue

            question = rec["question"]
            document = rec.get("external_knowledge")
            tags: list[str] = []
            if document:
                tags.append("external_knowledge")
                if with_knowledge:
                    path = os.path.join(documents_dir, document)
                    try:
                        with open(path) as doc:
                            question = (
                                f"{question}\n\n"
                                f"Reference documentation ({document}):\n{doc.read().strip()}"
                            )
                    except OSError:
                        # A named-but-absent document is worth seeing, not silently dropping:
                        # the question is then being asked without context it depends on.
                        tags.append("knowledge_missing")

            cases.append(
                EvaluationCase(
                    id=rec["instance_id"],
                    question=question,
                    gold_sql="",  # Spider 2.0-lite grades against result CSVs, not gold SQL
                    db_id=rec["db"],
                    answerable=True,
                    tags=tags,
                )
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


def load_eval_meta(eval_jsonl: str) -> dict[str, dict]:
    """{instance_id: {"condition_cols": [...], "ignore_order": bool}}"""
    meta: dict[str, dict] = {}
    with open(eval_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta[rec["instance_id"]] = rec
    return meta


def gold_paths(exec_result_dir: str, instance_id: str) -> list[str]:
    """Every accepted gold for one question: local002_a.csv, local002_b.csv, ..."""
    exact = os.path.join(exec_result_dir, f"{instance_id}.csv")
    variants = sorted(glob.glob(os.path.join(exec_result_dir, f"{instance_id}_*.csv")))
    return ([exact] if os.path.exists(exact) else []) + variants


def _vectors_match(a: list, b: list, *, tolerance: float = 1e-2, ignore_order: bool = False) -> bool:
    """One gold column against one predicted column, Spider 2.0's comparison verbatim."""
    def sort_key(x):
        return (x is None, str(x), isinstance(x, (int, float)))

    if ignore_order:
        a, b = sorted(a, key=sort_key), sorted(b, key=sort_key)
    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        x_null = x is None or (isinstance(x, float) and math.isnan(x))
        y_null = y is None or (isinstance(y, float) and math.isnan(y))
        if x_null and y_null:
            continue
        if x_null or y_null:
            return False
        if isinstance(x, bool) or isinstance(y, bool):
            if bool(x) != bool(y):
                return False
            continue
        try:
            if math.isclose(float(x), float(y), abs_tol=tolerance):
                continue
            return False
        except (TypeError, ValueError):
            pass
        if str(x).strip() != str(y).strip():
            return False
    return True


def compare_tables(
    predicted: list[list],
    gold: list[list],
    *,
    condition_cols: list[int] | None = None,
    ignore_order: bool = False,
) -> bool:
    """Spider 2.0's `compare_pandas_table`, on column-major data.

    Every gold column must be matched by some predicted column. Column order is irrelevant and
    extra predicted columns are allowed -- the rule mnemiq calls got-the-facts.
    """
    columns = list(gold)
    if condition_cols:
        try:
            columns = [gold[i] for i in condition_cols]
        except IndexError:
            columns = list(gold)

    for column in columns:
        if not any(
            _vectors_match(column, candidate, ignore_order=ignore_order)
            for candidate in predicted
        ):
            return False
    return True
