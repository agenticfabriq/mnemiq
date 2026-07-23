from __future__ import annotations

import re


def trigrams(text: str) -> set[str]:
    padded = f"  {text.lower()} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def similarity(a: str, b: str) -> float:
    """Jaccard over character trigrams. Symmetric, and therefore the right choice when both
    sides are the same kind of thing (a question literal against a stored value)."""
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    union = len(ta | tb)
    return len(ta & tb) / union if union else 0.0


_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Words that appear in nearly every code-scheme label and say nothing about WHICH scheme.
# Dropped from both sides before comparison: without this a column literally named `code`
# scores 0.800 against "Product Claim Code" -- a match on the one word that carries no signal.
_GENERIC_TOKENS = frozenset({
    "cd", "class", "classification", "classifications", "code", "codes", "id", "list", "lists",
    "scheme", "schemes", "system", "systems", "type", "types", "value", "values", "vocab",
    "vocabulary",
})

# Below this many discriminative characters the overlap coefficient inflates on noise: a
# one-character column name has so few trigrams that almost any label "contains" it.
_MIN_DISCRIMINATIVE = 3


def _tokens(text: str) -> list[str]:
    return [t for t in _NON_ALNUM.split(text.lower()) if t]


def _squash(text: str) -> str:
    return _NON_ALNUM.sub("", text.lower())


def _discriminative(text: str) -> str:
    """Drop tokens shared by nearly every scheme label. Falls back to the full token set when
    stripping would leave nothing, so a scheme legitimately named just "Codes" still matches."""
    tokens = _tokens(text)
    kept = [t for t in tokens if t not in _GENERIC_TOKENS]
    return " ".join(kept or tokens)


def _overlap(a: str, b: str) -> float:
    """Trigram overlap coefficient: shared trigrams over the SMALLER set.

    Deliberately not Jaccard. Column names are short and scheme labels are long, so Jaccard's
    union denominator punishes the length asymmetry that is inherent to the comparison -- every
    trigram of `origin` appears in "Sample Origin Classification", yet Jaccard scores that 0.200.
    The question here is containment ("does the column name appear in the label"), not sameness.
    """
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def name_affinity(column: str, label: str) -> float:
    """How much a column name looks like it names a code scheme.

    Guards the binder's value-set containment check: a scheme whose notations are `1/2/3` would
    "contain" the values of half a database, so gate 2 alone proves the values FIT while this
    proves the fit MEANS something.

    Compares on two surface forms and takes the better. Token-joined handles the common case;
    separator-squashed rescues abbreviations whose punctuation breaks trigrams (`icd10_cd` vs
    `ICD-10-CM` scores 0.667 tokenised but 0.833 squashed). Jaccard is not consulted: the
    overlap coefficient dominates it for every input, since min(|a|,|b|) <= |a union b|.
    """
    col, lab = _discriminative(column), _discriminative(label)
    if len(col.replace(" ", "")) < _MIN_DISCRIMINATIVE:
        return 0.0
    return max(_overlap(col, lab), _overlap(_squash(col), _squash(lab)))
