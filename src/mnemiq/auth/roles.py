"""The one role normalizer, for roles from settings, from a token's claim and from the identity map alike."""

from __future__ import annotations

from collections.abc import Iterable


def normalize_roles(values: Iterable[str]) -> list[str]:
    """Strip each role and drop the empty ones. Case is kept: the policy's role names are matched exactly, so
    folding here would let `Analyst` in a token grant `analyst`'s tables."""
    return [stripped for value in values if (stripped := value.strip())]
