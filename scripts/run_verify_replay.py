"""Replay the deterministic verifier over a saved mini_dev run and print the wrong-caught vs
correct-lost trade. Judge sweep is added in Task 7.

  .venv/bin/python scripts/run_verify_replay.py eval-reports/minidev-pg-14b-guided-fixed.jsonl
"""

from __future__ import annotations

import sys

from mnemiq.eval.verify_replay import load_records, replay
from mnemiq.verify.verifier import Verifier


def main() -> int:
    path = sys.argv[1]
    records = load_records(path)
    print(f"loaded {len(records)} records from {path}")
    for name, vf in (("sanity+grounding", Verifier()),
                     ("sanity-only", Verifier(grounding=False)),
                     ("grounding-only", Verifier(sanity=False))):
        o = replay(records, vf)
        print(f"\n== {name} ==")
        print(f"  wrong-caught {o['wrong_caught']}/{o['wrong_before']}  "
              f"correct-lost {o['correct_lost']}/{o['correct_before']}")
        print(f"  EX {o['ex_before']:.1%} -> {o['ex_after']:.1%}   "
              f"wrong {o['wrong_before_rate']:.1%} -> {o['wrong_after_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
