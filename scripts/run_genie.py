"""Ask Databricks AI/BI Genie every BIRD mini-dev question, one space per db_id.

Genie has no batch interface either: one conversation per question, polled until the message
completes. A *new* conversation per question is deliberate -- reusing one leaks context between
questions and inflates the score.

Space ids come from a JSON map (db_id -> space id), written by hand or by --print-spaces after
creating the spaces in the UI:

  {"financial": "01ef...", "superhero": "01ef..."}

  uv run python scripts/run_genie.py --spaces eval-reports/genie-spaces.json --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mnemiq.eval.bird import load_bird  # noqa: E402
from mnemiq.eval.spider import load_spider  # noqa: E402
from mnemiq.eval.spider2 import load_spider2_local  # noqa: E402
from mnemiq.eval.warehouse import (  # noqa: E402
    VendorResult,
    append_result,
    databricks_workspace,
    load_results,
)

_DEFAULT_MINIDEV = os.environ.get(
    "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/minidev/MINIDEV")
)


def ask_genie(workspace, space_id: str, question: str, *, timeout: int = 180) -> tuple[dict, int]:
    """Start a conversation, poll to completion, return the finished message."""
    started = time.monotonic()
    conversation = workspace.genie.start_conversation(space_id=space_id, content=question)
    message_id = conversation.message_id
    conversation_id = conversation.conversation_id

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = workspace.genie.get_message(
            space_id=space_id, conversation_id=conversation_id, message_id=message_id
        )
        status = str(message.status)
        if "COMPLETED" in status or "FAILED" in status or "CANCELLED" in status:
            return message.as_dict(), int((time.monotonic() - started) * 1000)
        time.sleep(2)
    raise TimeoutError(f"Genie did not finish within {timeout}s")


def extract(payload: dict) -> tuple[str | None, str | None]:
    """(sql, clarification). A query attachment is an answer; bare text is a deferral."""
    for attachment in payload.get("attachments") or []:
        query = attachment.get("query") or {}
        if query.get("query"):
            return query["query"], None
    text = " ".join(
        (a.get("text") or {}).get("content", "") for a in payload.get("attachments") or []
    ).strip()
    return None, text or payload.get("error") or "empty response"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minidev", default=_DEFAULT_MINIDEV)
    p.add_argument("--spaces", default="eval-reports/genie-spaces.json")
    p.add_argument("--profile", default=os.environ.get("DATABRICKS_PROFILE", "DEFAULT"))
    p.add_argument(
        "--host",
        default=os.environ.get("DATABRICKS_HOST"),
        help="workspace URL; triggers browser OAuth instead of a stored token",
    )
    p.add_argument(
        "--benchmark",
        default="bird",
        choices=["bird", "spider", "spider2"],
        help="which question set to ask; spider reads --spider-dir and has no evidence hints, "
        "spider2 reads --spider2-jsonl and has no gold SQL at all",
    )
    p.add_argument(
        "--spider-dir",
        default=os.environ.get(
            "MNEMIQ_SPIDER_DIR", os.path.expanduser("~/src/dataset/spider/spider_data")
        ),
    )
    p.add_argument("--split", default="dev", help="spider only")
    p.add_argument(
        "--spider2-jsonl",
        default=os.environ.get(
            "MNEMIQ_SPIDER2_JSONL",
            os.path.expanduser("~/src/dataset/spider2-lite/Spider2/spider2-lite/spider2-lite.jsonl"),
        ),
        help="spider2 only: the spider2-lite instance file (local* instances are used)",
    )
    p.add_argument(
        "--no-knowledge",
        action="store_true",
        help="spider2 only: drop the external_knowledge document 13 questions depend on. "
        "Default is to include it, which is Spider 2.0's own protocol.",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument("--no-evidence", action="store_true")
    p.add_argument("--results", default="eval-reports/genie-results.jsonl")
    p.add_argument("--refresh", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="questions asked concurrently. Each question is its own conversation, so there is "
        "no context to leak between them; the ceiling is the warehouse and Genie rate limits.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="seconds to wait for one Genie message before recording it as an error",
    )
    p.add_argument(
        "--print-spaces",
        action="store_true",
        help="list every Genie space this workspace has, with its id, and exit",
    )
    args = p.parse_args()

    workspace = databricks_workspace(host=args.host, profile=args.profile)

    if args.print_spaces:
        for space in (workspace.genie.list_spaces().spaces or []):
            print(f"{space.space_id}  {space.title}")
        return 0

    if not os.path.exists(args.spaces):
        print(f"no space map at {args.spaces} -- create the Genie spaces, then run "
              f"--print-spaces and write a db_id -> space_id JSON map", file=sys.stderr)
        return 1
    with open(args.spaces) as fh:
        spaces = json.load(fh)

    if args.benchmark == "spider2":
        cases = load_spider2_local(
            os.path.expanduser(args.spider2_jsonl),
            limit=args.limit,
            db_ids=args.dbs,
            with_knowledge=not args.no_knowledge,
        )
    elif args.benchmark == "spider":
        cases = load_spider(
            os.path.expanduser(args.spider_dir),
            split=args.split,
            limit=args.limit,
            db_ids=args.dbs,
        )
    else:
        cases = load_bird(
            os.path.expanduser(args.minidev),
            limit=args.limit,
            db_ids=args.dbs,
            with_evidence=not args.no_evidence,
        )
    done = {} if args.refresh else load_results(args.results)
    remaining = [c for c in cases if c.id not in done and c.db_id in spaces]
    missing = sorted({c.db_id for c in cases} - set(spaces))
    if missing:
        print(f"no Genie space for: {', '.join(missing)} -- those questions are skipped")
    print(f"{len(cases)} cases, {len(done)} answered, {len(remaining)} to ask", flush=True)

    counts = {"sql": 0, "deferred": 0, "error": 0}
    # One lock for both the checkpoint file and the counters: a torn line in the JSONL would
    # lose a paid-for answer, and a lost count would misreport the run.
    write_lock = threading.Lock()
    done_count = 0

    def ask_one(case) -> VendorResult:
        result = VendorResult(
            case_id=case.id,
            db_id=case.db_id,
            question=case.question,
            gold_sql=case.gold_sql,
            difficulty=(case.tags or [""])[0],
        )
        try:
            payload, elapsed = ask_genie(
                workspace, spaces[case.db_id], case.question, timeout=args.timeout
            )
            result.latency_ms = elapsed
            result.sql, result.deferral = extract(payload)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"[:400]
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(ask_one, case): case for case in remaining}
        for future in as_completed(futures):
            case = futures[future]
            result = future.result()
            with write_lock:
                done_count += 1
                counts[result.outcome] += 1
                append_result(args.results, result)
                print(
                    f"  [{done_count:4}/{len(remaining)}] {result.outcome:9} "
                    f"{case.id:16} {case.db_id}",
                    flush=True,
                )

    print(
        f"\nasked {len(remaining)}: {counts['sql']} sql, "
        f"{counts['deferred']} deferred, {counts['error']} error"
    )
    print(f"results -> {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
