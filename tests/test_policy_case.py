"""Object ids are compared case-insensitively in the deciders' policy lookups.

NOT at literally every enforcement point: `values_check` still compares
`table in row_filtered` exactly. That one is not a live disclosure -- `ValueIndex` runs its own
exact-case `WHERE object_id = ?`, so a case mismatch makes the whole grounding check silently
no-op rather than leak -- but it does mean value grounding goes inert on a case-mismatched query,
and it is left alone here rather than claimed closed.

Found by Codex's stop-time review, which said the PII bypass "remains exploitable" after a fix
whose test asserted only that `policy.denied` CONTAINED the entry. It did. Nothing asserted that
a query was refused, and it was not: `check_cls` compared the snapshot's spelling against the
spelling the query resolved to, exactly.

Measured before this, on a PLAIN TABLE READ with no view involved:

  snapshot keys CLAIM, query says claim  -> APPROVED, a `direct` PII column returned
  snapshot keys claim, query says CLAIM  -> APPROVED
  the same two shapes for a row filter   -> APPROVED UNFILTERED

Unquoted identifiers are case-insensitive in all three engines, so every one of those spells the
same object. The fold lives on `AccessPolicy` so there is ONE normalisation rather than one per
guard -- the keys stay as the snapshot wrote them, so refusal subjects and `verdict.tables` still
report the source's own names; only how a lookup is ANSWERED changed.
"""
from __future__ import annotations

import itertools

import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.contract.semantic import Column, Job, Snapshot
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy, build_access_policy
from mnemiq.sql.verdict import Refusal, RefusalCode

CASES = ["claim", "Claim", "CLAIM"]
JOBS = [Job(id="discover:views", source_id="s", kind="discover", status="done")]


def _col(obj, name, pii=None):
    return Column(id=f"{obj}.{name}", object_id=obj, name=name, data_type="text", pii_level=pii)


@pytest.mark.parametrize("snapshot_key,query_ref", list(itertools.product(CASES, CASES)))
def test_a_denied_column_is_refused_however_the_table_is_spelled(snapshot_key, query_ref):
    """The ENFORCEMENT, not the policy's contents. The test this replaces asserted the denial was
    recorded and stopped there -- a control that agrees with itself."""
    snapshot = Snapshot(version="v", source_id="s", created_at="t", views=[], jobs=JOBS,
                        columns=[_col(snapshot_key, "id"), _col(snapshot_key, "ssn", "direct")])
    policy = build_access_policy(snapshot, GrantSet(objects=frozenset({snapshot_key})))
    v = decide(f"SELECT ssn FROM {query_ref}",
               {snapshot_key: {"id", "ssn"}, query_ref: {"id", "ssn"}}, policy=policy)
    assert isinstance(v, Refusal) and v.code is RefusalCode.UNAUTHORIZED_COLUMN, v


@pytest.mark.parametrize("snapshot_key,query_ref", list(itertools.product(CASES, CASES)))
def test_a_row_filter_binds_however_the_table_is_spelled(snapshot_key, query_ref):
    snapshot = Snapshot(version="v", source_id="s", created_at="t", views=[], jobs=JOBS,
                        columns=[_col(snapshot_key, "id"), _col(snapshot_key, "region")])
    policy = build_access_policy(
        snapshot, GrantSet(objects=frozenset({snapshot_key}),
                           row_filters={snapshot_key: "region = 'west'"}))
    v = decide(f"SELECT id FROM {query_ref}",
               {snapshot_key: {"id", "region"}, query_ref: {"id", "region"}}, policy=policy)
    assert not isinstance(v, Refusal), v
    assert "region = 'west'" in v.plan_sql, v.plan_sql


@pytest.mark.parametrize("column_case", ["ssn", "SSN", "Ssn"])
def test_the_column_name_folds_too(column_case):
    """The tuple has two halves and only one of them was ever the reported bug."""
    policy = AccessPolicy(denied={("claim", "ssn")})
    assert policy.denies("CLAIM", column_case)


def test_a_policy_free_caller_is_untouched():
    """The control for all of the above: folding must answer lookups differently, never invent
    a disposition where the policy holds none."""
    snapshot = Snapshot(version="v", source_id="s", created_at="t", views=[], jobs=JOBS,
                        columns=[_col("claim", "id"), _col("claim", "ssn")])
    policy = build_access_policy(snapshot, GrantSet(objects=frozenset({"claim"})))
    v = decide("SELECT ssn FROM claim", {"claim": {"id", "ssn"}}, policy=policy)
    assert not isinstance(v, Refusal), v


def test_the_keys_stay_as_the_snapshot_wrote_them():
    """The fold answers lookups; it does not rewrite what the engine reports. A refusal subject
    and `verdict.tables` must still carry the source's own spelling."""
    policy = AccessPolicy(denied={("CLAIM", "ssn")}, row_filters={"CLAIM": "region = 'west'"})
    assert ("CLAIM", "ssn") in policy.denied
    assert list(policy.row_filters) == ["CLAIM"]
    assert policy.denies("claim", "SSN") and policy.row_filter_for("claim") == "region = 'west'"


# -- masking, which the first version of this fold broke in a new way ---------------------------
#
# Folding the mask dict's KEYS while leaving one consumer reading it with the query's own-case
# spelling left `needs_mask` correctly true and the column set handed to the projection EMPTY, so
# the masked column was emitted as its real value. Measured on that version:
#
#   SELECT id, ssn FROM CLAIM, masked={('claim','ssn')}
#     -> SELECT id, ssn FROM (SELECT amount, id, ssn FROM CLAIM) AS CLAIM     <- no NULL
#   the same query in matching case
#     -> SELECT id, ssn FROM (SELECT amount, id, NULL AS ssn FROM claim) AS claim
#
# A mask that silently does nothing is worse than one that refuses: the answer looks complete.

@pytest.mark.parametrize("query_case", CASES)
@pytest.mark.parametrize("policy_table", CASES)
@pytest.mark.parametrize("policy_column", ["ssn", "Ssn", "SSN"])
def test_a_masked_column_is_nulled_however_either_name_is_spelled(
    query_case, policy_table, policy_column
):
    v = decide(f"SELECT id, ssn FROM {query_case}", {query_case: {"id", "amount", "ssn"}},
               policy=AccessPolicy(masked={(policy_table, policy_column)}))
    assert not isinstance(v, Refusal), v
    assert "NULL AS ssn" in v.target_sql, v.target_sql


def test_an_unmasked_column_is_not_nulled():
    """The control: folding must not start masking columns no policy names."""
    v = decide("SELECT id, ssn FROM claim", {"claim": {"id", "ssn"}}, policy=AccessPolicy())
    assert "NULL" not in v.target_sql, v.target_sql


@pytest.mark.parametrize("snapshot_column", ["ssn", "Ssn", "SSN"])
def test_the_mask_folds_the_snapshot_s_column_spelling_too(snapshot_column):
    """The half the matrix above missed, and the mutation caught: those cases vary the POLICY's
    column case, while the projection compares against the SNAPSHOT's. `masked_by_table` already
    lowercases what it stores, so only a differently-cased VISIBLE column exercises the fold in
    `_derived_table` -- which is why unfolding it left every test green."""
    v = decide(f"SELECT id, {snapshot_column} FROM claim", {"claim": {"id", snapshot_column}},
               policy=AccessPolicy(masked={("claim", "ssn")}))
    assert not isinstance(v, Refusal), v
    assert f"NULL AS {snapshot_column}" in v.target_sql, v.target_sql



# -- the write path, because M30 is what happens when the two implementations differ -------------

def test_a_denied_column_is_refused_on_the_write_path_too(query_case="CLAIM"):
    """`decide_write` feeds `check_cls` an `AccessPolicy(denied=policy.denied | policy.masked)`,
    so it inherits the fold -- asserted rather than assumed, because this codebase's own M30 is
    two RLS implementations that diverged exactly where nobody was looking."""
    from mnemiq.authz.grants import GrantSet
    from mnemiq.sql.decide_write import decide_write
    visible = {query_case: {"id", "ssn"}, "claim": {"id", "ssn"}}
    v = decide_write(f"UPDATE {query_case} SET ssn = 'x' WHERE id = 1", visible,
                     GrantSet(objects=frozenset(visible), writable=frozenset({query_case})),
                     adapter=None, dialect="duckdb",
                     policy=AccessPolicy(denied={("claim", "ssn")}), views={},
                     writes_enabled=True)
    assert isinstance(v, Refusal) and v.code is RefusalCode.UNAUTHORIZED_COLUMN, v


@pytest.mark.parametrize("query_case", CASES)
def test_a_row_filter_binds_on_the_write_path_however_the_table_is_spelled(query_case):
    from mnemiq.authz.grants import GrantSet
    from mnemiq.sql.decide_write import decide_write
    visible = {query_case: {"id", "region", "amount"}, "claim": {"id", "region", "amount"}}
    v = decide_write(f"UPDATE {query_case} SET amount = 0 WHERE id = 1", visible,
                     GrantSet(objects=frozenset(visible), writable=frozenset({query_case})),
                     adapter=None, dialect="duckdb",
                     policy=AccessPolicy(row_filters={"claim": "region = 'west'"}), views={},
                     writes_enabled=True)
    assert not isinstance(v, Refusal), v
    assert "region = 'west'" in v.plan_sql, v.plan_sql


def test_the_applied_row_policy_does_not_depend_on_how_the_caller_spells_the_table():
    """The defect this replaces an earlier assertion for. That one asserted an exact-case key
    WINS over a folded one -- which means a caller picks its own row policy by changing case.
    Measured on that version, one identity, one policy holding both `claim` and `CLAIM`:

      SELECT id FROM claim  -> WHERE region = 'west'
      SELECT id FROM CLAIM  -> WHERE region = 'east'

    Picking a winner in any order still leaves the caller choosing. Both apply now, OR-combined,
    which is the combinator `grants.py` already uses for per-role filters."""
    policy = AccessPolicy(row_filters={"claim": "region = 'west'", "CLAIM": "region = 'east'"})
    plans = {}
    for q in CASES:
        v = decide(f"SELECT id FROM {q}", {q: {"id", "region"}}, policy=policy)
        assert "west" in v.plan_sql and "east" in v.plan_sql, (q, v.plan_sql)
        plans[q] = v.plan_sql.replace(q, "T")
    assert len(set(plans.values())) == 1, plans


def test_two_roles_naming_one_table_in_different_cases_or_combine_at_the_merge():
    """The root, in `grants_for`. Its own comment says "more roles = more visible rows:
    OR-combine", and the exact `in` check never did that across cases -- so the two grants stayed
    separate keys and the query's spelling chose between them."""
    import json
    import os
    import tempfile

    from mnemiq.authz.grants import FileAuthzProvider
    from mnemiq.contract import IdentityContext

    policy = {"roles": {"a": {"read": ["claim"], "row_filters": {"claim": "region = 'west'"}},
                        "b": {"read": ["claim"], "row_filters": {"CLAIM": "region = 'east'"}}}}
    fd, path = tempfile.mkstemp(suffix=".json")
    os.write(fd, json.dumps(policy).encode())
    os.close(fd)
    try:
        grants = FileAuthzProvider(path).grants_for(
            IdentityContext(subject="u", tenant_id="t", principal_id="p", roles=["a", "b"]))
    finally:
        os.unlink(path)
    assert len(grants.row_filters) == 1, dict(grants.row_filters)
    combined = next(iter(grants.row_filters.values()))
    assert "west" in combined and "east" in combined and "OR" in combined, combined


@pytest.mark.parametrize("query_case", CASES)
def test_a_masked_column_in_a_predicate_is_refused_however_spelled(query_case):
    """The `MASKED_COLUMN_IN_PREDICATE` branch, which is the only caller of `AccessPolicy.masks`
    -- every other masking test goes through `_derived_table`'s nulling path, which never calls
    it. Masking a filtered column would silently change the answer, so it refuses instead."""
    v = decide(f"SELECT id FROM {query_case} WHERE ssn = 'x'",
               {query_case: {"id", "ssn"}}, policy=AccessPolicy(masked={("claim", "ssn")}))
    assert isinstance(v, Refusal) and v.code is RefusalCode.MASKED_COLUMN_IN_PREDICATE, v


def test_a_masked_column_in_a_bare_projection_is_still_allowed():
    """The control: only predicate use refuses. Selecting it is what the NULL projection is for."""
    v = decide("SELECT ssn FROM CLAIM", {"CLAIM": {"id", "ssn"}},
               policy=AccessPolicy(masked={("claim", "ssn")}))
    assert not isinstance(v, Refusal), v
