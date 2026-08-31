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


def term_pattern(term: str) -> re.Pattern[str]:
    """Each word of the term on a word boundary, in order; the last word may inflect
    ("premium" matches "premiums", "loss ratio" matches "loss ratios").

    Public because it is the tolerance rule for the whole engine, not this module's private
    convenience: `generate.undefined_terms` grounds a model's declared term against it. Two
    matchers over one corpus is what let retrieval offer a definition while the M35 guard called
    the same words ungrounded, so there is one rule and it lives here, where retrieval defines it.
    """
    words = [re.escape(w) for w in term.split()]
    body = r"\s+".join(words[:-1] + [words[-1] + r"\w*"]) if words else ""
    return re.compile(rf"\b{body}", re.IGNORECASE)


def select_definitions(
    question: str,
    definitions: Sequence[Definition],
    grants: GrantSet,
    table_ids: Sequence[str] = (),
) -> list[Definition]:
    """The definitions this question needs AND this identity may see.

    A definition binding any ungranted table is never shown: it would leak that the
    table exists, and it would steer the model into SQL the decider must reject.

    Visibility is fail-closed: a definition is shown only if it is `public` (a standard whose
    text names no table) or it is bound to objects the identity is granted. An unbound,
    non-public definition -- a confidential internal taxonomy -- is visible to no one.

    **Two selection rules, because there are two kinds of definition.** A definition BOUND to a
    table rides with it: if the table is in the retrieved context, the definition is offered
    whatever the question said. A policy governing a table -- "the business date is this column,
    not the other five" -- matters most precisely when the asker did not know to ask for it, and
    drafting the fs corpus measured the cost of the other rule: three of five policy definitions
    matched no realistic question, so under term-matching alone they could never be retrieved
    however true they were.

    An UNBOUND definition keeps the term-match rule. A public standard belongs to no table, so it
    has no table to ride with, and matching the asker's words is the right rule for something
    looked up by name.
    """
    in_context = set(table_ids)
    selected = []
    for definition in definitions:
        visible = definition.public or (
            bool(definition.bound_objects)
            and all(grants.allows(obj) for obj in definition.bound_objects)
        )
        if not visible:
            continue
        rides_with_a_table = any(obj in in_context for obj in definition.bound_objects)
        if rides_with_a_table or term_pattern(definition.term).search(question):
            selected.append(definition)
    return selected
