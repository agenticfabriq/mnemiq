"""The `mnemiq` command: enrich, build, ask, serve, eval."""

from __future__ import annotations

import argparse
import sys

from mnemiq.agent.modes import MODES
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.runtime import SnapshotMissing, build_runtime


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mnemiq", description="Open-source data-agent engine.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("enrich", help="profile + describe the source; save a snapshot")
    sub.add_parser("build", help="index the current snapshot for retrieval")
    sub.add_parser(
        "refresh", help="re-crawl the source; re-enrich + publish if the catalog changed"
    )

    a = sub.add_parser("ask", help="ask a question in natural language")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="emit the machine record")
    a.add_argument("--principal", default="local")
    a.add_argument("--roles", default="", help="comma-separated")
    a.add_argument("--mode", choices=sorted(MODES), default=None,
                   help="instant (cheapest) | thinking (default) | deep (highest precision)")

    w = sub.add_parser("write", help="execute a single INSERT/UPDATE/DELETE (governed)")
    w.add_argument("sql")
    w.add_argument("--json", action="store_true", help="emit the WriteResult")
    w.add_argument("--principal", default="local")
    w.add_argument("--roles", default="", help="comma-separated")

    f = sub.add_parser("feedback", help="record a fixed failure as a golden case + example")
    f.add_argument("--question", required=True)
    f.add_argument("--sql", required=True)
    f.add_argument("--tables", default="", help="comma-separated")
    f.add_argument("--golden", default="evals/acme.json")
    f.add_argument("--examples", default="evals/captured_examples.json")

    sub.add_parser("serve", help="run the MCP server on stdio")

    e = sub.add_parser("eval", help="run the ACME golden set")
    e.add_argument("--golden", default="evals/acme.json")
    e.add_argument("--gate", action="store_true", help="exit non-zero on accuracy regression")
    e.add_argument("--record", action="store_true", help="record this run in the accuracy trend")
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
    from mnemiq.enrichment.examples import LLMExampleGenerator, enrich_examples
    from mnemiq.enrichment.facts import LLMFactsEnricher, enrich_table_facts
    from mnemiq.enrichment.pipeline import enrich_structural
    from mnemiq.enrichment.semantic import enrich_semantic
    from mnemiq.eval.bird_runner import _flag
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
    if _flag("MNEMIQ_ENRICH_FACTS"):
        snap = enrich_table_facts(snap, LLMFactsEnricher(LLMClient(settings)))
    if _flag("MNEMIQ_ENRICH_EXAMPLES"):
        snap = enrich_examples(
            snap, LLMExampleGenerator(LLMClient(settings)), adapter, dialect=adapter.dialect
        )
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
    from mnemiq.semantic.store import build_example_index, build_index
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import current_version, load_snapshot

    con = init_store(settings.store_path)
    embedder = LLMEmbedder(settings)
    if len(settings.source_specs()) > 1:
        from mnemiq.store.federated_build import build_federated

        n, n_ex = build_federated(settings, con, embedder)
        print(f"indexed {n} cards, {n_ex} examples (federated) -> {settings.store_path}")
        if settings.control_dsn:
            from mnemiq.store.control import publish_version

            for spec in settings.source_specs():
                v = current_version(con, spec.id)
                if v is not None:
                    publish_version(settings.control_dsn, spec.id, v)
        return 0
    version = current_version(con, settings.source_id)
    if version is None:
        print(
            f"no snapshot for {settings.source_id!r} -- run `mnemiq enrich` first", file=sys.stderr
        )
        return 1
    snap = load_snapshot(con, version)
    n = build_index(con, snap, embedder)
    n_ex = build_example_index(con, snap, embedder)
    print(f"indexed {n} cards, {n_ex} examples -> {settings.store_path}")
    if settings.control_dsn:
        from mnemiq.store.control import publish_version

        publish_version(settings.control_dsn, settings.source_id, version)
    return 0


def _cmd_refresh(settings: Settings) -> int:
    from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
    from mnemiq.enrichment.refresh import catalog_diff
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import current_version, load_snapshot

    if not settings.pg_dsn:
        print("set MNEMIQ_PG_DSN", file=sys.stderr)
        return 1
    con = init_store(settings.store_path)
    version = current_version(con, settings.source_id)
    if version is None:
        print("no snapshot -- run `mnemiq enrich` first", file=sys.stderr)
        return 1
    diff = catalog_diff(DuckDBPostgresAdapter(settings.pg_dsn), load_snapshot(con, version))
    print(f"added={diff.added} changed={diff.changed} dropped={diff.dropped}")
    if not diff.has_changes:
        print("catalog unchanged -- nothing to refresh")
        return 0
    rc = _cmd_enrich(settings)
    if rc == 0:
        rc = _cmd_build(settings)  # build publishes the new version when a control DSN is set
    return rc


def _cmd_ask(settings: Settings, args) -> int:
    try:
        rt = build_runtime(settings)
    except SnapshotMissing as exc:
        print(str(exc), file=sys.stderr)
        return 1
    ans = rt.ask(args.question, _identity(args), mode=args.mode)
    if args.json:
        import json

        print(
            json.dumps(
                {
                    "answer": ans.answer,
                    "deferred": ans.deferred,
                    "mode": ans.mode,
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


def _cmd_write(settings: Settings, args) -> int:
    try:
        rt = build_runtime(settings)
    except SnapshotMissing as exc:
        print(str(exc), file=sys.stderr)
        return 1
    res = rt.write(args.sql, _identity(args))
    if args.json:
        import json

        print(json.dumps(vars(res), default=str))
        return 0
    if res.approved:
        print(f"OK: wrote to {res.target} (rows affected: {res.rows_affected})")
    else:
        print(f"refused: {res.refusal}")
    return 0


def _cmd_feedback(args) -> int:
    from mnemiq.feedback.capture import capture_fix

    tables = [t for t in args.tables.split(",") if t]
    cid = capture_fix(args.question, args.sql, tables, "acme", args.golden, args.examples)
    print(f"captured {cid} -> {args.golden}, {args.examples}")
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
    if args.command == "refresh":
        return _cmd_refresh(settings)
    if args.command == "ask":
        return _cmd_ask(settings, args)
    if args.command == "write":
        return _cmd_write(settings, args)
    if args.command == "feedback":
        return _cmd_feedback(args)
    if args.command == "serve":
        return _cmd_serve(settings)
    if args.command == "eval":
        from mnemiq.eval.run import run_acme

        return run_acme(settings, golden=args.golden, gate=args.gate, record=args.record)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
