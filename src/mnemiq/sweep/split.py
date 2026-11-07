"""The questions a sweep tunes on, and the ones it keeps back to confirm its pick on questions the search never saw.

A case's side is decided by a hash of the split seed and the case's id, so the split is the same on every machine and
under every Python version, and it does not move when the question file is reordered. Within each `answerable` group the
held-out part takes `floor(held_out * size of the group)` cases, so both parts keep questions to decline.

A sweep draws its split once, saves it as `split.json` and records its digest in `sweep.json`. Every later reader, the
run resuming and the report alike, reads that saved split through `saved_split`, which verifies it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

from mnemiq.contract import EvaluationCase
from mnemiq.sweep.config import SPLIT, SWEEP, SweepConfig, SweepConfigError, load_questions


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


PARTS = ("tuning", "held_out")


def split_digest(ids: Mapping[str, Sequence[str]]) -> str:
    """The sha256 of a split's case ids, each part's in its order, as `split.json` holds them."""
    body = json.dumps({part: list(ids[part]) for part in PARTS}, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def saved_split(cfg: SweepConfig) -> dict[str, list[EvaluationCase]] | None:
    """The split a sweep saved in its out folder, read back and verified, or None where no sweep has started (there is
    no `sweep.json`).

    `split.json` must hold each part as a list of case ids, each id once and in one part only, the two together exactly
    the question file's ids, and it must hash to the digest `sweep.json` recorded when the split was drawn: another
    split, however well formed, would read the arms' rows against other questions than the ones they answered.
    """
    out = Path(cfg.sweep.out)
    record, path = out / SWEEP, out / SPLIT
    if not record.exists():
        return None
    try:
        ids = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SweepConfigError(f"{path} cannot be read ({exc.strerror}), and this sweep's cases are read from "
                               "it") from None
    except ValueError:
        raise SweepConfigError(f"{path} is not JSON, and this sweep's cases are read from it") from None
    if not (isinstance(ids, dict) and set(ids) == set(PARTS) and all(
            isinstance(ids[part], list) and all(isinstance(case_id, str) for case_id in ids[part]) for part in PARTS)):
        raise SweepConfigError(f"{path} does not hold the tuning and held-out parts as lists of case ids")
    for part in PARTS:
        if twice := sorted(case_id for case_id, count in Counter(ids[part]).items() if count > 1):
            raise SweepConfigError(f"{path} names {', '.join(twice)} twice in its {part} part")
    if both := sorted(set(ids["tuning"]) & set(ids["held_out"])):
        raise SweepConfigError(f"{path} puts {', '.join(both)} in both parts")
    known = {case.id: case for case in load_questions(cfg)}
    named = set(ids["tuning"]) | set(ids["held_out"])
    if named != known.keys():
        why = [f"does not name {', '.join(sorted(known.keys() - named))}, which the question file holds"
               if known.keys() - named else "",
               f"names {', '.join(sorted(named - known.keys()))}, which the question file does not hold"
               if named - known.keys() else ""]
        raise SweepConfigError(f"{path} {'; and '.join(reason for reason in why if reason)}")
    if split_digest(ids) != json.loads(record.read_text(encoding="utf-8")).get("split_digest"):
        raise SweepConfigError(f"{path} is not the split this sweep drew: its digest is not the one {record} recorded. "
                               "Put back the split.json the sweep wrote, or start a new sweep in another out folder.")
    return {part: [known[case_id] for case_id in ids[part]] for part in PARTS}
