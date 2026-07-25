from __future__ import annotations

import json
import re
from collections.abc import Sequence

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Definition


def load_definitions(path: str) -> list[Definition]:
    with open(path) as fh:
        raw = json.load(fh)
    return [Definition.model_validate(d) for d in raw]


def _term_pattern(term: str) -> re.Pattern[str]:
    # Each word of the term on a word boundary, in order; the last word may inflect
    # ("premium" matches "premiums", "loss ratio" matches "loss ratios").
    words = [re.escape(w) for w in term.split()]
    body = r"\s+".join(words[:-1] + [words[-1] + r"\w*"]) if words else ""
    return re.compile(rf"\b{body}", re.IGNORECASE)


def select_definitions(
    question: str, definitions: Sequence[Definition], grants: GrantSet
) -> list[Definition]:
    """The definitions this question needs AND this identity may see.

    A definition binding any ungranted table is never shown: it would leak that the
    table exists, and it would steer the model into SQL the decider must reject.

    Visibility is fail-closed: a definition is shown only if it is `public` (a standard whose
    text names no table) or it is bound to objects the identity is granted. An unbound,
    non-public definition -- a confidential internal taxonomy -- is visible to no one.
    """
    selected = []
    for definition in definitions:
        visible = definition.public or (
            bool(definition.bound_objects)
            and all(grants.allows(obj) for obj in definition.bound_objects)
        )
        if not visible:
            continue
        if _term_pattern(definition.term).search(question):
            selected.append(definition)
    return selected
