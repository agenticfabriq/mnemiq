from __future__ import annotations

from mnemiq.contract import Snapshot

# (object_id, column name) -> True when the column's non-null values are all distinct, False when
# some repeat. A column the profile never measured is ABSENT, and absence is never read as either.
KeyFacts = dict[tuple[str, str], bool]


def key_facts(snapshot: Snapshot) -> KeyFacts:
    """Which columns hold each value at most once, from the profile enrichment already took.

    Declared foreign keys are not enough and are not used: `fact_claim` and `fact_premium` share
    `policy_id` with no relationship declared between them, which is exactly how a chasm trap
    escapes schema metadata. The profile's `count(DISTINCT col)` is exact, so this is a fact about
    the data at profiling time, not an estimate.
    """
    facts: KeyFacts = {}
    for column in snapshot.columns:
        if column.row_count is None or column.distinct_count is None or column.null_count is None:
            continue
        facts[(column.object_id, column.name)] = (
            column.distinct_count == column.row_count - column.null_count
        )
    return facts
