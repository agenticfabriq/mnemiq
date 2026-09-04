"""Grants for a GOVERNED measurement arm — one that actually narrows an answer.

Every other eval arm grants everything on purpose (`build_engine`'s own comment says so: grade SQL
capability with full data access, because governance has its own tests). The consequence is that
nothing the benchmark measures has ever carried a row filter or a mask through the whole engine,
so the disclosure path has no end-to-end evidence behind it at all -- and the four defects review
found in that path were exactly the kind an arm like this catches mechanically.

This builds grants that narrow SOMETHING REACHED. A policy that filters a table the questions never
query, or masks a level no column carries, produces a governed arm that is indistinguishable from
the ungoverned one and would pass any test that merely asserts "the arm ran".
"""

from __future__ import annotations

from dataclasses import dataclass

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Snapshot


@dataclass(frozen=True)
class GovernedPlan:
    """What the arm will narrow, resolved against a real snapshot before anything runs."""

    grants: GrantSet
    filtered_table: str | None
    masked_level: str | None
    masked_columns: tuple[str, ...]

    @property
    def narrows_something(self) -> bool:
        """False when the policy cannot narrow this corpus, whatever it says.

        The kill criterion reads a disclosure RATE, and a rate over an arm that never narrows is
        zero for the wrong reason -- indistinguishable from a disclosure path that is broken.

        A resolved table is NOT sufficient: `1 = 1` names a real table and withholds no row, and
        it was this module's own default predicate, so the check written to refuse arms that
        cannot narrow certified one by construction. `filtered_table` is set only when the
        predicate can exclude something.
        """
        return bool(self.filtered_table) or bool(self.masked_columns)


_TAUTOLOGIES = {"1 = 1", "1=1", "true", "1", "1 is not null"}


def _can_exclude(predicate: str) -> bool:
    """Whether a row filter can withhold ANY row.

    Syntactic and deliberately conservative: it rejects the handful of spellings that are
    unambiguously total, and accepts everything else rather than pretending to decide satisfiability.
    A predicate that is subtly total on the actual data -- `amount > -1` on non-negative amounts --
    still passes here, and the honest guard for that is the arm's own measured disclosure count,
    not this function.

    It exists because `1 = 1` was this module's DEFAULT, so the plainest possible non-narrowing
    filter was the one a caller got for free.
    """
    return bool(predicate) and predicate.strip().lower().rstrip(";") not in _TAUTOLOGIES


def governed_grants(
    snapshot: Snapshot,
    tables: list[str],
    *,
    filter_table: str | None = None,
    filter_predicate: str | None = None,
    mask_level: str | None = None,
) -> GovernedPlan:
    """Grants that filter one queried table and mask one reached column level.

    `mask_level` goes to `pii_mask` and is deliberately withheld from `pii_clearance`: a level in
    both is seen raw, so putting it in both would produce a policy that reads as governed and masks
    nothing. Every other level stays cleared, so the arm measures the effect of ONE mask rather
    than the effect of denying most of the schema.
    """
    # No default predicate. The one that reads most natural is `1 = 1`, which withholds no row --
    # it WAS the default here, and it certified an arm that could not narrow. A mask-only arm needs
    # no predicate at all, so this is required only when a table is actually named.
    if filter_table and not filter_predicate:
        raise ValueError(
            "filter_table needs a filter_predicate that can exclude rows; there is no safe "
            "default, because the obvious one (`1 = 1`) withholds nothing")

    all_levels = {c.pii_level for c in snapshot.columns if c.pii_level and c.pii_level != "none"}

    chosen_level = mask_level if mask_level in all_levels else None
    masked = tuple(sorted(
        c.object_id + "." + c.name for c in snapshot.columns
        if chosen_level and c.pii_level == chosen_level))

    known = {t.lower() for t in tables}
    in_corpus = (filter_table or "").lower() in known
    chosen_table = filter_table if (in_corpus and _can_exclude(filter_predicate or "")) else None

    return GovernedPlan(
        grants=GrantSet(
            frozenset(tables),
            row_filters={chosen_table: filter_predicate} if chosen_table and filter_predicate else {},
            # cleared: everything EXCEPT the level under test
            pii_clearance=frozenset(all_levels - ({chosen_level} if chosen_level else set())),
            pii_mask=frozenset({chosen_level} if chosen_level else set()),
        ),
        filtered_table=chosen_table,
        masked_level=chosen_level,
        masked_columns=masked,
    )
