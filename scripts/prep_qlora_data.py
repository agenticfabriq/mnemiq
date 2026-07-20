"""Build QLoRA training data for the local-model spike, in mnemiq's EXACT inference format.

For each bird23-train-filtered example (db_id, question, evidence, gold SQL) we reconstruct a
per-db Snapshot from train_column_meaning.json, render it with mnemiq's own build_cards +
system_prompt + user_prompt, transpile the gold SQL SQLite->DuckDB (the dialect our engine
generates), and emit a chat record whose assistant turn is the JSON envelope the engine parses.
Train == inference, so the tuned model drops straight into the pipeline.

  uv run python scripts/prep_qlora_data.py --out /tmp/qlora_train.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import sqlglot

from mnemiq.contract import Column, Snapshot
from mnemiq.generate.prompts import system_prompt, user_prompt
from mnemiq.semantic.cards import build_cards
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

_DIR = os.environ.get("MNEMIQ_BIRD23_DIR", os.path.expanduser("~/src/dataset/bird23"))


def _snapshots(meaning: dict) -> dict[str, Snapshot]:
    """train_column_meaning.json keys are 'db|table|column' -> description."""
    by_db: dict[str, list[Column]] = collections.defaultdict(list)
    for key, desc in meaning.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        db, table, col = parts
        by_db[db].append(Column(id=f"{table}.{col}", object_id=table, name=col, description=desc))
    return {db: Snapshot(version="train", source_id=db, created_at="t", columns=cols)
            for db, cols in by_db.items()}


def _to_duckdb(sql: str) -> str | None:
    try:
        return sqlglot.transpile(sql, read="sqlite", write="duckdb")[0]
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/qlora_train.jsonl")
    p.add_argument("--dir", default=_DIR)
    args = p.parse_args()

    meaning = json.load(open(f"{args.dir}/train_column_meaning.json"))
    snaps = _snapshots(meaning)
    rows = [json.loads(x) for x in open(f"{args.dir}/train.jsonl")]

    sys_prompt = system_prompt(dialect="duckdb")
    written = skipped = 0
    with open(args.out, "w") as fh:
        for r in rows:
            snap = snaps.get(r["db_id"])
            duck = _to_duckdb(r["SQL"])
            if snap is None or duck is None:
                skipped += 1
                continue
            cards = [RetrievedCard(object_id=c.object_id, card=c.text, score=1.0)
                     for c in build_cards(snap)]
            ev = (r.get("evidence") or "").strip()
            q = f"{r['question']}\n\nHint: {ev}" if ev else r["question"]
            packet = ContextPacket(question=q, cards=cards, grant_fingerprint="", enrichment_version=None)
            target = json.dumps({"sql": duck, "reason": ""})
            fh.write(json.dumps({"messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt(packet)},
                {"role": "assistant", "content": target},
            ]}) + "\n")
            written += 1
    print(f"wrote {written} training records, skipped {skipped} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
