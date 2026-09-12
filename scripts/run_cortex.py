"""Ask Snowflake Cortex Analyst every BIRD mini-dev question, one semantic model per db_id.

Cortex Analyst has no batch interface -- this is one REST call per question, authenticated
with the key-pair JWT. Results are checkpointed to JSONL, so an interrupted run resumes
without re-paying for the questions already answered.

  uv run python scripts/run_cortex.py --limit 20 --db financial
  uv run python scripts/run_cortex.py                      # all 500

Each db_id needs its own semantic object. Snowflake is deprecating stage-hosted YAML
semantic *models* in favour of native semantic *views*, so views are the default here:
BENCH.<DB_ID>.BIRD_SV. Pass --semantic-model-file to use a YAML model instead.
"""

from __future__ import annotations

import argparse
import os
import time
import tomllib

import requests
from cryptography.hazmat.primitives import serialization
from snowflake.connector.auth import AuthByKeyPair

from mnemiq.eval.bird import load_bird
from mnemiq.eval.spider import load_spider
from mnemiq.eval.spider2 import load_spider2_local
from mnemiq.eval.warehouse import VendorResult, append_result, load_results

_DEFAULT_MINIDEV = os.environ.get(
    "MNEMIQ_MINIDEV_DIR", os.path.expanduser("~/src/dataset/bird-minidev/minidev/MINIDEV")
)


def read_connection(name: str) -> dict:
    """One named connection out of ~/.snowflake/config.toml."""
    path = os.path.expanduser("~/.snowflake/config.toml")
    with open(path, "rb") as fh:
        config = tomllib.load(fh)
    try:
        return config["connections"][name]
    except KeyError:
        raise SystemExit(f"no [connections.{name}] in {path}")


class CortexClient:
    """Cortex Analyst REST, with a JWT that is refreshed before it expires."""

    def __init__(self, account: str, user: str, private_key_file: str, *, timeout: int = 120):
        self.account = account
        self.user = user
        self.timeout = timeout
        # AuthByKeyPair wants a parsed key or DER bytes -- handing it the PEM file's bytes
        # fails with an ASN.1 parsing error, so deserialize here.
        with open(os.path.expanduser(private_key_file), "rb") as fh:
            self._key = serialization.load_pem_private_key(fh.read(), password=None)
        self.url = f"https://{account}.snowflakecomputing.com/api/v2/cortex/analyst/message"
        self._token = ""
        self._token_expires = 0.0
        self._session = requests.Session()

    def _jwt(self) -> str:
        # AuthByKeyPair mints a short-lived JWT; regenerate a minute before it lapses so a
        # long run never fails halfway on an expired token.
        if time.time() < self._token_expires:
            return self._token
        lifetime = 3600
        auth = AuthByKeyPair(private_key=self._key, lifetime_in_seconds=lifetime)
        self._token = auth.prepare(account=self.account.split(".")[0], user=self.user)
        self._token_expires = time.time() + lifetime - 60
        return self._token

    def ask(self, question: str, semantic: str, *, is_view: bool) -> tuple[dict, int]:
        # A semantic view is referenced by qualified name; a legacy model by its stage path.
        target = {"semantic_view": semantic} if is_view else {"semantic_model_file": semantic}
        started = time.monotonic()
        response = self._session.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self._jwt()}",
                "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                **target,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": question}]}
                ],
            },
            timeout=self.timeout,
        )
        elapsed = int((time.monotonic() - started) * 1000)
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:400]}")
        return response.json(), elapsed


def extract(payload: dict) -> tuple[str | None, str | None]:
    """(sql, clarification) out of a Cortex Analyst message.

    A `statement` block is an answer. Text with no statement alongside it is the model saying
    it cannot answer -- a deferral, which is a different outcome from a wrong query.
    """
    blocks = (payload.get("message") or {}).get("content") or []
    sql = next((b.get("statement") for b in blocks if b.get("type") == "sql"), None)
    if sql:
        return sql, None
    text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
    return None, text or "empty response"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minidev", default=_DEFAULT_MINIDEV)
    p.add_argument(
        "--benchmark",
        default="bird",
        choices=["bird", "spider", "spider2"],
        help="which question set to ask; spider reads --spider-dir and has no evidence hints, "
        "spider2 reads --spider2-dir (the 135 local* instances)",
    )
    # A DIRECTORY, not the jsonl. `load_spider2_local` appends `repo/spider2-lite/
    # spider2-lite.jsonl` itself, so passing the file made it open a path UNDER that file and
    # raise before the first question -- `--benchmark spider2` could never have run. Same
    # variable and default as `run_spider2.py` and `keys_spider2_snowflake.py`.
    p.add_argument(
        "--spider2-dir",
        default=os.environ.get(
            "MNEMIQ_SPIDER2_DIR", os.path.expanduser("~/src/dataset/spider2-lite")
        ),
    )
    p.add_argument(
        "--no-knowledge",
        action="store_true",
        help="spider2 only: drop the external_knowledge document the benchmark supplies",
    )
    p.add_argument(
        "--spider-dir",
        default=os.environ.get(
            "MNEMIQ_SPIDER_DIR", os.path.expanduser("~/src/dataset/spider/spider_data")
        ),
    )
    p.add_argument("--split", default="dev", help="spider only")
    p.add_argument("--connection", default=os.environ.get("SNOWFLAKE_CONNECTION", "bench_key"))
    p.add_argument("--database", default="BENCH")
    p.add_argument(
        "--semantic-pattern",
        default="{database}.{schema}.BIRD_SV",
        help="semantic object, formatted with database/schema/db_id. Default is a semantic "
        "view's qualified name; with --semantic-model-file give a stage path instead, "
        "e.g. @{database}.{schema}.BIRD_STG/{db_id}.yaml",
    )
    p.add_argument(
        "--semantic-model-file",
        action="store_true",
        help="treat --semantic-pattern as a legacy stage-hosted YAML model, not a view",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--db", action="append", dest="dbs")
    p.add_argument("--difficulty", choices=["simple", "moderate", "challenging"])
    p.add_argument("--no-evidence", action="store_true", help="drop BIRD's per-question hint")
    p.add_argument("--results", default="eval-reports/cortex-results.jsonl")
    p.add_argument("--refresh", action="store_true", help="ignore the checkpoint and re-ask")
    args = p.parse_args()

    if args.benchmark == "spider2":
        cases = load_spider2_local(
            os.path.expanduser(args.spider2_dir),
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
            difficulty=args.difficulty,
            with_evidence=not args.no_evidence,
        )
    done = {} if args.refresh else load_results(args.results)
    remaining = [c for c in cases if c.id not in done]
    print(
        f"{len(cases)} cases, {len(done)} already answered, {len(remaining)} to ask",
        flush=True,
    )
    if not remaining:
        return 0

    conn = read_connection(args.connection)
    client = CortexClient(conn["account"], conn["user"], conn["private_key_file"])

    counts = {"sql": 0, "deferred": 0, "error": 0}
    for index, case in enumerate(remaining, 1):
        schema = case.db_id.upper()
        # Spider 2.0-lite has hyphenated db_ids (DB-IMDB, SQLITE-SAKILA); an unquoted
        # hyphen is a minus sign to the SQL parser, so quote anything not [A-Z0-9_].
        if not schema.replace("_", "").isalnum():
            schema = f'"{schema}"'
        semantic = args.semantic_pattern.format(
            database=args.database, schema=schema, db_id=case.db_id
        )
        result = VendorResult(
            case_id=case.id,
            db_id=case.db_id,
            question=case.question,
            gold_sql=case.gold_sql,
            difficulty=(case.tags or [""])[0],
        )
        try:
            payload, elapsed = client.ask(
                case.question, semantic, is_view=not args.semantic_model_file
            )
            result.latency_ms = elapsed
            result.sql, result.deferral = extract(payload)
            result.raw = {"request_id": payload.get("request_id")}
        except Exception as exc:  # network, auth, throttling -- recorded, not fatal
            result.error = f"{type(exc).__name__}: {exc}"

        counts[result.outcome] += 1
        append_result(args.results, result)
        print(
            f"  [{index:3}/{len(remaining)}] {result.outcome:9} {case.id:14} {case.db_id}",
            flush=True,
        )

    print(
        f"\nasked {len(remaining)}: {counts['sql']} sql, "
        f"{counts['deferred']} deferred, {counts['error']} error"
    )
    print(f"results -> {args.results}")
    return 1 if counts["error"] == len(remaining) else 0


if __name__ == "__main__":
    raise SystemExit(main())
