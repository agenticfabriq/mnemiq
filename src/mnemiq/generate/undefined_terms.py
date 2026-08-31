"""Which terms an answer leaned on that nobody certified a meaning for.

M35: the engine answered *"What is the lifetime value of our average customer?"* by inventing a
derivation. Measured on the fs_payments corpus, `policy_records.json` certifies five definitions --
`payment_date`, `payment_identity`, `revenue`, `total_payment`, `payment_revisions` -- and none for
lifetime value. So the model has `revenue` and `total_payment` to hand and composes something
plausible from them.

The four refusal items that DO refuse reliably each name a COLUMN that does not exist. The guard
that holds is "no such column"; there was no guard for "no such definition", and that is a *meaning*
failure rather than a structural one.

**The model supplies candidates; this decides.** That split is the point. Asking the model to
refuse when it is unsure is asking the thing that invented the derivation to notice it invented it
-- and it refuses 2 times in 6, which is a coin rather than a guard. Asking it only to DECLARE what
it assumed is a question it can answer reliably, and leaves the refusal to code that cannot be
persuaded.

Deliberately not a term list. Enumerating business terms is the shape `views.py` documents as
unfixable: each round of review finds another, because an unlisted term passes. A list also cannot
survive a new corpus, which is every deployment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mnemiq.semantic.glossary import INFLECTION_PLURAL, term_pattern


def _normalized(spelling: str) -> str:
    """One spelling, in the form both matchers can be compared in.

    Underscores become spaces because a corpus spells one term two ways in one record --
    `term="total payment"` beside `id="fspay:policy:total_payment"` -- and a model writes neither
    form reliably.
    """
    return " ".join(spelling.replace("_", " ").split()).lower()


def _same_term(one: str, other: str) -> bool:
    r"""Whether two spellings name the same term, under retrieval's words-in-order RULE at a
    narrower WIDTH.

    `term_pattern` is `select_definitions`' rule: each word on a boundary, in order, the last one
    free to inflect. Sharing it is the point -- the defect this closes was two matchers reading one
    corpus, where retrieval put `total payment` in the packet BECAUSE the model said "total
    payments" and this check then called "total payments" ungrounded. A guard that refuses a term
    on the strength of a definition sitting in its own input is not measuring meaning.

    What is NOT shared is how far the last word may run, and the paragraph on `INFLECTION_PLURAL`
    below is the reason: one rule, two widths.

    `fullmatch`, not `search`: the tolerance is on the term's ENDING, not on what surrounds it.
    "revenue per customer" is a derivation over a defined term and must stay ungrounded -- that
    composition-of-certified-parts shape is the M35 finding itself, and a substring match would
    silence the guard on the case it exists for.

    Both directions, because inflection is not the corpus's alone: a record saying `payment
    revisions` has to ground a model that declared `payment revision`.

    `INFLECTION_PLURAL`, not retrieval's `\w*`. Two limits are needed and only one of them is
    about spaces. A longer PHRASE cannot win either direction because the extension never crosses a
    space -- but an unbounded extension of one WORD is the same failure inside a word: `\w*` makes
    `policy` ground `policyholder`, so a model declaring "policyholder" against a corpus certifying
    only "policy" is handed the confident answer instead of the deferral. Grounding is the
    suppressing direction, so its tolerance has to be the narrow one.
    """
    return bool(
        term_pattern(one, INFLECTION_PLURAL).fullmatch(other)
        or term_pattern(other, INFLECTION_PLURAL).fullmatch(one)
    )


def _spoken(definition: Any) -> set[str]:
    """The spellings a model might use for this definition.

    Both `term` and the bare tail of `id`, because a corpus spells the term in either place: the
    fs_payments records carry `term="revenue"` beside `id="fspay:policy:revenue"`, while others
    carry `id="loss_ratio"` with no term at all. The tail only -- a namespaced id is not something
    a model writes, so matching the whole of it would leave those definitions unmatched.
    """
    out: set[str] = set()
    for attr in ("term", "name", "label"):
        value = getattr(definition, attr, None)
        if isinstance(value, str) and value.strip():
            out.add(value.strip().lower())
    # `id`, which is what `Definition` actually calls it. The first version read `object_id` --
    # a field the model does not have -- so this whole branch was dead in production and the test
    # that "proved" it passed only because the fake invented the field. A fixture that shapes
    # itself to the code cannot falsify the code, which is why the tests now use the real type.
    identifier = getattr(definition, "id", None)
    if isinstance(identifier, str) and identifier:
        # A bare id like `loss_ratio` IS how some corpora spell the term; a namespaced one is not.
        tail = identifier.rsplit(":", 1)[-1]
        out.add(tail.replace("_", " ").strip().lower())
    return {s for s in out if s}


def ungrounded_terms(assumed: Sequence[str], definitions: Sequence[Any]) -> list[str]:
    """The declared terms with no certified definition behind them, in the order declared.

    Every one of them, not the first: the refusal is a repair instruction, and a caller told about
    one missing definition fixes one and comes back.

    An empty definition set grounds nothing, which is why an ablated deployment refuses rather than
    skipping the check -- the register measured that arm refusing 0 of 4, and "we certified nothing"
    is a reason to trust the engine less, not more.
    """
    known: set[str] = set()
    for definition in definitions or ():
        known |= {_normalized(s) for s in _spoken(definition)}
    out: list[str] = []
    for term in assumed or ():
        if not isinstance(term, str):
            continue
        cleaned = _normalized(term)
        if not cleaned:
            continue
        if any(_same_term(cleaned, spelling) for spelling in known):
            continue
        if any(_same_term(cleaned, _normalized(seen)) for seen in out):
            continue
        out.append(term.strip())
    return out
