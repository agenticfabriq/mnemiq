from __future__ import annotations

import json

from mnemiq.contract import EvaluationCase


def load_cases(path: str) -> list[EvaluationCase]:
    with open(path) as fh:
        raw = json.load(fh)
    return [EvaluationCase.model_validate(case) for case in raw]
