"""The `mnemiq` command: enrich, build, ask, serve, eval."""

from __future__ import annotations

import argparse
import sys

from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.runtime import SnapshotMissing, build_runtime


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mnemiq", description="Open-source data-agent engine.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("enrich", help="profile + describe the source; save a snapshot")
    sub.add_parser("build", help="index the current snapshot for retrieval")

    a = sub.add_parser("ask", help="ask a question in natural language")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="emit the machine record")
    a.add_argument("--principal", default="local")
    a.add_argument("--roles", default="", help="comma-separated")

    sub.add_parser("serve", help="run the MCP server on stdio")

    e = sub.add_parser("eval", help="run the ACME golden set")
    e.add_argument("--golden", default="evals/acme.json")
    return p


def _identity(args) -> IdentityContext:
    return IdentityContext(
        tenant_id="local",
        principal_id=getattr(args, "principal", "local"),
        roles=[r for r in getattr(args, "roles", "").split(",") if r],
    )


def _cmd_enrich(settings: Settings) -> int:
    from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
    from mnemiq.enrichment.enricher import LLMEnricher
    from mnemiq.enrichment.pipeline import enrich_structural
    from mnemiq.enrichment.semantic import enrich_semantic
    from mnemiq.llm.client import LLMClient
    from mnemiq.semantic.values import build_value_index
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import save_snapshot

    if not settings.pg_dsn:
        print("set MNEMIQ_PG_DSN", file=sys.stderr)
        return 1
    adapter = DuckDBPostgresAdapter(settings.pg_dsn)
    snap = enrich_structural(adapter, settings.source_id)
    snap = enrich_semantic(snap, LLMEnricher(LLMClient(settings)))
    con = init_store(settings.store_path)
    save_snapshot(con, snap)
    n_values = build_value_index(adapter, snap, con)
    print(
        f"snapshot {snap.version} ({len(snap.source_bindings)} tables, "
        f"{n_values} indexed values) -> {settings.store_path}"
    )
    return 0


def _cmd_build(settings: Settings) -> int:
    from mnemiq.llm.embeddings import LLMEmbedder
    from mnemiq.semantic.store import build_index
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import current_version, load_snapshot

    con = init_store(settings.store_path)
    version = current_version(con, settings.source_id)
    if version is None:
        print(
            f"no snapshot for {settings.source_id!r} -- run `mnemiq enrich` first", file=sys.stderr
        )
        return 1
    n = build_index(con, load_snapshot(con, version), LLMEmbedder(settings))
    print(f"indexed {n} cards -> {settings.store_path}")
    return 0


def _cmd_ask(settings: Settings, args) -> int:
    try:
        rt = build_runtime(settings)
    except SnapshotMissing as exc:
        print(str(exc), file=sys.stderr)
        return 1
    ans = rt.ask(args.question, _identity(args))
    if args.json:
        import json

        print(
            json.dumps(
                {
                    "answer": ans.answer,
                    "deferred": ans.deferred,
                    "sql": ans.trace.target_sql if ans.trace else None,
                },
                default=str,
            )
        )
        return 0
    print(ans.answer)
    if ans.trace is not None:
        print(f"\nSQL:\n{ans.trace.target_sql}")
        print(f"\ntables: {ans.trace.tables_used}  ({ans.trace.timing.get('total_ms', 0):.0f} ms)")
    return 0


def _cmd_serve(settings: Settings) -> int:
    from mnemiq.mcp.server import serve

    serve(settings)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.command == "enrich":
        return _cmd_enrich(settings)
    if args.command == "build":
        return _cmd_build(settings)
    if args.command == "ask":
        return _cmd_ask(settings, args)
    if args.command == "serve":
        return _cmd_serve(settings)
    if args.command == "eval":
        from mnemiq.eval.run import run_acme

        return run_acme(settings, golden=args.golden)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
