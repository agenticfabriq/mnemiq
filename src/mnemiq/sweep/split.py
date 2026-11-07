"""The questions a sweep tunes on, and the ones it keeps back to confirm its pick on questions the search never saw.

A case's side is decided by a hash of the split seed and the case's id, so the split is the same on every machine and
under every Python version, and it does not move when the question file is reordered. Within each `answerable` group the
held-out part takes `floor(held_out * size of the group)` cases, so both parts keep questions to decline.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from fractions import Fraction

from mnemiq.contract import EvaluationCase
from mnemiq.sweep.config import SweepConfig


def _rank(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{case_id}".encode()).hexdigest()


def split(cfg: SweepConfig, cases: Sequence[EvaluationCase]) -> dict[str, list[EvaluationCase]]:
    """`tuning` and `held_out`, each in the question file's order. Check mode has no search to hold questions back
    from, so every case is in `tuning`."""
    if cfg.search.strategy == "check":
        return {"tuning": list(cases), "held_out": []}
    # Read as written: 0.29 * 100 is 28.999999999999996 in floating point, and would floor to 28.
    share = Fraction(repr(cfg.dataset.held_out))
    held: set[str] = set()
    for answerable in (True, False):
        group = sorted((case.id for case in cases if case.answerable is answerable),
                       key=lambda case_id: _rank(cfg.dataset.split_seed, case_id))
        held.update(group[:math.floor(share * len(group))])
    return {"tuning": [case for case in cases if case.id not in held],
            "held_out": [case for case in cases if case.id in held]}
