from __future__ import annotations

import re


def trigrams(text: str) -> set[str]:
    padded = f"  {text.lower()} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def similarity(a: str, b: str) -> float:
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    union = len(ta | tb)
    return len(ta & tb) / union if union else 0.0


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _squash(text: str) -> str:
    return _NON_ALNUM.sub("", text.lower())


def name_affinity(column: str, label: str) -> float:
    """How much a column name looks like it names a code scheme. Best of two surface forms.

    Raw similarity alone fails the abbreviation-with-punctuation case that motivates ontology
    grounding in the first place: `icd10_cd` vs `ICD-10-CM` scores 0.188 raw but 0.600 once
    separators are dropped -- the shared token `icd10` is real, the punctuation is noise.
    Squashing alone would instead penalise multi-word labels, where the extra words dilute the
    trigram set (`rating` vs `MPAA Film Rating` falls 0.333 -> 0.294). Neither form dominates,
    so take the better evidence rather than committing to one. True negatives score 0.0 under
    both (`zzz`, `sample_pk`), so this widens recall without widening false positives.
    """
    return max(similarity(column, label), similarity(_squash(column), _squash(label)))
