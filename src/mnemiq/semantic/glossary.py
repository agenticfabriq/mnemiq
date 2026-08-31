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


# How far the last word may run past the term. RETRIEVAL's default: anything, because a wider
# match there only offers the model an extra definition it can ignore.
INFLECTION_ANY = r"\w*"
# GROUNDING's: a plural and nothing else. The two callers need different widths because the cost
# of a wrong match points in opposite directions -- see `term_pattern`.
INFLECTION_PLURAL = r"(?:e?s)?"


def _last_word(word: str, inflection: str) -> str:
    """The last word plus its inflections, including the `-y`/`-ies` one no suffix rule reaches.

    A suffix appended to the literal cannot turn `policy` into `policies` -- the stem loses a
    character -- so neither width covered it and both callers were wrong in their own direction.
    Retrieval missed the definition on a question saying "policies"; grounding called "policies"
    ungrounded with `policy` certified, which is a refusal of an answerable question. Business
    vocabularies are full of these: policy, entity, category, party.

    Consonant + `y` only. `day` -> `days` is the ordinary rule and already matched, and rewriting
    it to `daies` would match nothing a corpus contains.
    """
    forms = [re.escape(word) + inflection]
    if len(word) > 2 and word[-1].lower() == "y" and word[-2].lower() not in "aeiou":
        forms.append(re.escape(word[:-1]) + "ies")
    return "(?:" + "|".join(forms) + ")"


def term_pattern(term: str, inflection: str = INFLECTION_ANY) -> re.Pattern[str]:
    r"""Each word of the term on a word boundary, in order; the last word may inflect
    ("premium" matches "premiums", "loss ratio" matches "loss ratios").

    Public because it is the tolerance rule for the whole engine, not this module's private
    convenience: `generate.undefined_terms` grounds a model's declared term against it. Two
    matchers over one corpus is what let retrieval offer a definition while the M35 guard called
    the same words ungrounded, so there is one rule and it lives here, where retrieval defines it.

    **One rule, two widths, because the failure directions are opposite.** Here a loose match
    widens RECALL: a spurious extra definition in the packet is noise the model can ignore. In the
    M35 grounding check the same looseness widens GROUNDING and SUPPRESSES a refusal, so `\w*`
    there is not tolerance but a hole -- it makes `policy` ground `policyholder` and `claim` ground
    `claimant`, handing back the confident answer that guard exists to refuse. The width is a
    parameter rather than a second function so that the words-in-order rule stays single: it was
    two independent implementations of THAT which caused the defect.
    """
    words = term.split()
    if not words:
        # Matches NOTHING. The previous expression collapsed to a bare `\b`, which matches every
        # question, so a definition with `term=""` was offered on every packet -- and that shape is
        # in this corpus, since a record may carry its name in `id` alone and `Definition.term`
        # has no non-empty constraint. A definition with nothing to match on has one honest way to
        # be retrieved, which is to ride with a table it is bound to.
        return re.compile(r"(?!)")
    head = [re.escape(w) for w in words[:-1]]
    body = r"\s+".join(head + [_last_word(words[-1], inflection)])
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
