from __future__ import annotations

import json
import re
from collections.abc import Sequence

from mnemiq.authz.grants import GrantSet
from dataclasses import dataclass
from typing import Any

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


def normalized(spelling: str) -> str:
    """One spelling in the form every matcher compares in: lowercase, underscores as spaces,
    whitespace collapsed."""
    return " ".join(spelling.replace("_", " ").split()).lower()


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
        # Matches NOTHING. The previous expression collapsed to a bare `\b`, which is true at the
        # start of any word, so a definition with `term=""` was offered on every packet whatever
        # was asked. `Definition.term` carries no non-empty constraint, so the shape is
        # constructible; no corpus IN THIS REPO contains it, and whether a shipped bundle does is
        # not checkable from here. The bare `\b` is wrong on its own terms either way.
        return re.compile(r"(?!)")
    head = [re.escape(w) for w in words[:-1]]
    body = r"\s+".join(head + [_last_word(words[-1], inflection)])
    # Boundaries at BOTH ends. Only the leading one was there, so a match could end mid-word: at
    # the plural width the tail `a` matched inside "average", putting that definition in every
    # packet. `\w*` hid it by consuming to the end of whatever word it landed in, which is why the
    # missing boundary surfaced only when a narrower width was introduced.
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


@dataclass(frozen=True)
class Name:
    """A spelling a definition answers to, and whether it is prose.

    The pair travels together because the two callers need different halves of it and deriving
    either one separately is how this module keeps breaking: `select_definitions` picks a matching
    width from `prose`, `ungrounded_terms` compares on `spelling`, and neither re-walks the
    precedence.
    """

    spelling: str
    prose: bool  # a term/name/label a person writes, as opposed to the tail of an id


def spellings(definition: Any) -> list[Name]:
    """The names this definition answers to, normalized -- THE one implementation.

    `term`, or `name`/`label` for a record that spells it either of those ways. Failing all three,
    the bare tail of `id`, because a record may carry its name only there. The tail only: a
    namespaced id is not something anyone writes into a sentence.

    **The tail only when there is no term.** It was briefly a spelling alongside the term, and that
    is the M35 suppressing direction: a definition with `term="net revenue"` and `id="...:revenue"`
    certifies net revenue, so a question saying "revenue" that its TERM does not match must not
    pull it into the packet -- doing so grounds a model declaring "revenue" against a definition
    that does not define it, and the deferral disappears. A term, where one exists, is the
    certified name; the id is an implementation detail that happens to be legible.

    Underscores become spaces because one record spells a term two ways -- `term="total payment"`
    beside `id="fspay:policy:total_payment"` -- and neither a question nor a model writes the
    underscored form.

    Used by BOTH `select_definitions` here and `generate.undefined_terms.ungrounded_terms`, by
    import rather than by resemblance. Two implementations of this list is the defect this module
    has now been fixed for four times in one branch, each time in the same direction.
    """
    for attr in ("term", "name", "label"):
        value = getattr(definition, attr, None)
        if isinstance(value, str) and value.strip():
            return [Name(normalized(value), prose=True)]
    identifier = getattr(definition, "id", None)
    if isinstance(identifier, str) and identifier:
        tail = normalized(identifier.rsplit(":", 1)[-1])
        if tail:
            return [Name(tail, prose=False)]
    return []


def _asked_for(definition: Definition, question: str) -> bool:
    r"""Whether the question names this definition.

    Prose gets prose's tolerance: `premium` should retrieve on "premiums". An id TAIL is not prose
    -- it is an identifier that happens to be legible -- and giving it `\w*` made a short one match
    nearly everything: a definition whose tail is `a` was selected by "what is our average
    revenue", and `re`, `rev` likewise, so an unrelated definition appeared in every packet. The
    tail is matched at the plural width instead. Nothing certified is lost by that; a definition
    wanting prose tolerance has a name, which is what a name is for.

    `Name.prose` rather than a second reading of `definition.term`: `spellings` also answers to
    `name` and `label`, which ARE prose, and re-deriving only the `term` case here gave those the
    identifier width.
    """
    return any(
        term_pattern(n.spelling, INFLECTION_ANY if n.prose else INFLECTION_PLURAL).search(question)
        for n in spellings(definition)
    )


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

    Everything else keeps the name-match rule -- its `term`, or the tail of its `id` when it has no
    term (`spellings`). That is two classes and not three: a bound definition whose table missed
    the retrieval cut, and a public one. NOT "an unbound definition": one fails
    `bool(bound_objects)` in the visibility test above and is seen by no matcher at all. The name is normalized before matching, so an underscored
    `term="total_payment"` is retrieved by a question saying "total payment" and no longer by one
    saying `total_payment`; nobody writes the underscored form into a sentence, and the same
    normalization is what lets the M35 guard ground against the same corpus. A public standard belongs to no table, so it
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
        if rides_with_a_table or _asked_for(definition, question):
            selected.append(definition)
    return selected
