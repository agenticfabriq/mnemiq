"""Affinity is the binder's third gate. These cases are the record of what it must accept and
what it must reject -- drawn from the Pagila and USDA validation runs, not invented."""
import pytest

from mnemiq.ontology.binder import AFFINITY_MIN
from mnemiq.semantic.textmatch import name_affinity, similarity, trigrams

# Real column/scheme pairs that MUST bind. `origin` and `claim` are the USDA columns the
# original Jaccard metric rejected at 0.200 and 0.250 against a 0.30 gate.
POSITIVES = [
    ("rating", "MPAA Film Rating"),
    ("origin", "Sample Origin Classification"),
    ("claim", "Product Claim Code"),
    ("icd10_cd", "ICD-10-CM"),
    ("colour_code", "Colour Codes"),
    ("naics_code", "NAICS"),
    ("food_code", "USDA Food Codes"),
    ("country", "ISO 3166 Country Codes"),
    ("currency_cd", "ISO 4217 Currency Codes"),
]

# Must NOT bind. The last four are adversarial: short or non-discriminative names that the
# overlap coefficient alone would wave through (`code` vs "Product Claim Code" scores 0.800
# before generic tokens are stripped).
NEGATIVES = [
    ("zzz", "Colour Codes"),
    ("notes", "Colour Codes"),
    ("mean", "Product Claim Code"),
    ("origin", "Product Claim Code"),
    ("claim", "Sample Origin Classification"),
    ("description", "MPAA Film Rating"),
    ("email", "ISO 3166 Country Codes"),
    ("quantity", "Colour Codes"),
    ("code", "Product Claim Code"),
    ("c", "Colour Codes"),
    ("cd", "ICD-10-CM"),
    ("status", "Product Claim Code"),
]


@pytest.mark.parametrize("column,label", POSITIVES)
def test_genuine_pairs_clear_the_gate(column, label):
    assert name_affinity(column, label) >= AFFINITY_MIN


@pytest.mark.parametrize("column,label", NEGATIVES)
def test_unrelated_or_generic_pairs_are_rejected(column, label):
    assert name_affinity(column, label) < AFFINITY_MIN


def test_the_gate_sits_in_an_empty_band():
    """The threshold must not be wedged between adjacent scores. Jaccard's positives bottomed
    at 0.200 while its negatives topped 0.136, leaving a 0.064-wide band -- too tight to
    survive an unseen column name. This asserts the replacement keeps real separation."""
    worst_positive = min(name_affinity(c, s) for c, s in POSITIVES)
    best_negative = max(name_affinity(c, s) for c, s in NEGATIVES)
    assert best_negative < AFFINITY_MIN <= worst_positive
    assert worst_positive - best_negative >= 0.25


def test_generic_tokens_carry_no_signal():
    """`code` matches "Product Claim Code" on the one word that says nothing about which
    scheme it is. Stripping generic tokens is what makes that a non-match."""
    assert name_affinity("code", "Product Claim Code") == 0.0
    assert name_affinity("claim", "Product Claim Code") > 0.8


def test_a_scheme_named_only_generically_still_matches():
    """Stripping must not empty a label outright: falling back to the full token set keeps a
    scheme legitimately named "Codes" reachable."""
    assert name_affinity("colour_code", "Colour Codes") > 0.9


def test_short_names_score_zero():
    """Overlap inflates when the shorter string has few trigrams -- `c` scored 0.500 against
    "Colour Codes" before the length guard."""
    assert name_affinity("c", "Colour Codes") == 0.0
    assert name_affinity("ab", "Abbreviation Codes") == 0.0


def test_similarity_is_unchanged_and_symmetric():
    """The value index still uses plain Jaccard: both its sides are the same kind of thing."""
    assert similarity("abc", "abc") == 1.0
    assert similarity("type 2 diabetes", "diabetes type 2") == similarity(
        "diabetes type 2", "type 2 diabetes")
    assert similarity("", "abc") == 0.0
    assert trigrams("ab") == {"  a", " ab", "ab "}
