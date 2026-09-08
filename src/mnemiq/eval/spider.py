"""Spider dev as EvaluationCases, plus the key metadata its tables.json declares.

Spider differs from BIRD in two ways that matter for benchmarking a semantic layer:
questions carry no external-knowledge hint, and the schema's primary/foreign keys live in
`tables.json` rather than in the SQLite files. The keys are what a semantic layer needs to
infer joins, so they are parsed here alongside the questions.
"""

from __future__ import annotations

import json
import os

from mnemiq.contract import EvaluationCase


def spider_db_path(spider_dir: str, db_id: str, subdir: str = "database") -> str:
    return os.path.join(spider_dir, subdir, db_id, f"{db_id}.sqlite")


def load_spider(
    spider_dir: str,
    *,
    split: str = "dev",
    limit: int | None = None,
    db_ids: list[str] | None = None,
) -> list[EvaluationCase]:
    """Spider questions as EvaluationCases, routed by db_id.

    Spider has no per-question evidence and no difficulty label, so neither is set -- the
    harness's difficulty slicing simply comes out empty for a Spider run.
    """
    with open(os.path.join(spider_dir, f"{split}.json")) as fh:
        records = json.load(fh)

    allowed = set(db_ids) if db_ids else None
    cases: list[EvaluationCase] = []
    for index, rec in enumerate(records):
        if allowed is not None and rec["db_id"] not in allowed:
            continue
        cases.append(
            EvaluationCase(
                id=f"spider-{split}-{index}",
                question=rec["question"],
                gold_sql=rec["query"],
                db_id=rec["db_id"],
                answerable=True,
                tags=[],
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def load_keys(spider_dir: str, tables_file: str = "tables.json") -> dict[str, dict]:
    """{db_id: {"primary": {table: [cols]}, "foreign": [(t, c, ref_t, ref_c)]}}

    tables.json indexes columns globally as [table_index, column_name] pairs, with primary and
    foreign keys given as indices into that list; this resolves them to plain names.
    """
    with open(os.path.join(spider_dir, tables_file)) as fh:
        schemas = json.load(fh)

    out: dict[str, dict] = {}
    for schema in schemas:
        tables = schema["table_names_original"]
        columns = schema["column_names_original"]  # [[table_idx, col_name], ...], [-1, "*"] first

        def resolve(index: int) -> tuple[str, str] | None:
            table_index, column = columns[index]
            if table_index < 0:
                return None
            return tables[table_index], column

        primary: dict[str, list[str]] = {}
        for index in schema.get("primary_keys", []):
            pair = resolve(index)
            if pair:
                primary.setdefault(pair[0], []).append(pair[1])

        foreign: list[tuple[str, str, str, str]] = []
        for source, target in schema.get("foreign_keys", []):
            a, b = resolve(source), resolve(target)
            if a and b:
                foreign.append((a[0], a[1], b[0], b[1]))

        out[schema["db_id"]] = {"primary": primary, "foreign": foreign}
    return out
