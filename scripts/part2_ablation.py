#!/usr/bin/env python3
"""Drive Part 2's ablation over the fs payments corpus and print the answer, cut three ways.

Invoked by the cross-repo wide run once Verity has a live records endpoint. Kept as a script rather
than a `mnemiq` subcommand because it is an EXPERIMENT driver: it decides which arms exist, where
their stores go and how the bands are tagged, none of which is product behaviour.

The seam to Verity is deliberately a process boundary. This reads a golden set Verity exported and
posts traces back over HTTP; it never reads Verity's bundle formats. That is the same relationship
the product has, so the run exercises it rather than a shortcut around it.

Usage:
    part2_ablation.py --golden <exported.json> --control <control.json> \
        --records-url <url> [--traces-url <url>] --store-dir <dir> [--out <report.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mnemiq.config import Settings
from mnemiq.eval.ablation import AblationReport, load_golden, run_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", required=True,
                        help="golden set exported by `semantic_cli export-golden`")
    parser.add_argument("--control", required=True,
                        help="the schema-recoverable control band")
    parser.add_argument("--records-url", required=True,
                        help="Verity GET /api/semantic/records, for the GROUNDED arm only")
    parser.add_argument("--traces-url", default=None,
                        help="Verity POST /api/traces/batch; unset emits no traces")
    parser.add_argument("--authz", default="evals/fs_payments_authz.json")
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--out", default=None, help="write the report as JSON here")
    args = parser.parse_args()

    # The band each case belongs to. Assigned HERE and not by the exporter, because it is a claim
    # about the question and Verity has no way to know it: every case it exports is built on a
    # certified metric, which makes it a candidate for the meaning band but does not prove one.
    # These six were written to turn on rules the schema cannot state, and that is the claim being
    # recorded by this line.
    meaning = load_golden(args.golden)
    for case in meaning:
        if "meaning" not in case.tags:
            case.tags = [*case.tags, "meaning"]

    control = load_golden(args.control)
    for case in control:
        if "control" not in case.tags:
            case.tags = [*case.tags, "control"]

    cases = [*meaning, *control]
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate case ids across the two bands: {sorted(ids)}")

    store_dir = Path(args.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    base = Settings.from_env()
    print(f"cases:    {len(meaning)} meaning + {len(control)} control", flush=True)
    print(f"source:   {base.source_id}", flush=True)
    print(f"records:  {args.records_url}", flush=True)
    print(f"traces:   {args.traces_url or 'not emitted'}", flush=True)
    print(flush=True)

    report = run_ablation(
        base, cases,
        store_dir=store_dir,
        records_url=args.records_url,
        traces_url=args.traces_url,
        authz_path=args.authz,
    )
    print(report.render(), flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(_as_json(report), indent=1))
        print(f"\nreport: {args.out}", flush=True)

    # A failed GATE is a failed run: it means the experiment could not show its own instrument was
    # connected. A wrong ANSWER is not -- that is a finding, and the report above is where it is
    # read. Exiting non-zero on a low score would make the run a test of the model rather than of
    # the pipeline.
    return 0 if report.gate.passed else 1


def _as_json(report: AblationReport) -> dict:
    return {
        "gate": {
            "passed": report.gate.passed,
            "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in report.gate.checks],
        },
        "arms": {
            arm.name: {
                "snapshot_version": arm.snapshot_version,
                "certified_refs": arm.certified_refs,
                "metrics": arm.metrics,
                "dimensions": arm.dimensions,
                "results": [
                    # The bounded fields, not the whole CaseResult: the row previews it carries are
                    # for a human debugging one case, and dumping them here would put the seed's
                    # contents into every run's artifact.
                    {
                        "case_id": r.case_id,
                        "outcome": str(r.outcome),
                        "tags": report.cases[r.case_id].tags if r.case_id in report.cases else [],
                        "sql": r.sql,
                        "gold_sql": r.gold_sql,
                        "answer": r.answer[:400],
                        "ms": round(r.ms, 1),
                    }
                    for r in arm.results
                ],
            }
            for arm in (report.bare, report.grounded)
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
