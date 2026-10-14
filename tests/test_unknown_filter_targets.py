"""M50: a row filter whose key names no object in the snapshot refuses, rather than vanishing.

`build_access_policy` keeps only the filters whose key it can reach, which is right for a key
naming a real table this identity cannot read -- nothing it could filter is readable. A key naming
NO table cannot be told apart from a typo for one that is readable, and was dropped the same way.
Measured on `1257c11`: with a snapshot keyed `claim`, the keys `public.claim`, `claims` and
`pg.claim` each approved `SELECT id FROM claim LIMIT 1000` -- no predicate, no refusal, no log
line. The owner's decision (2026-10-13): refuse every read for that identity until the operator
fixes the key; the server log names it, the caller is told only that the policy needs fixing.
"""

from __future__ import annotations

import logging

import pytest

from mnemiq.authz.coverage import warn_unknown_filter_targets
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, DeferralReason, IdentityContext, Snapshot, ViewDefinition
from mnemiq.generate.generator import FakeGenerator
from mnemiq.generate.plan_query import Deferred, plan_query
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql import policy as policy_module
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

_FILTER = "region = 'west'"


def _col(obj: str, name: str) -> Column:
    return Column(id=f"{obj}.{name}", object_id=obj, name=name)


def _snapshot(prefix: str = "", views=()) -> Snapshot:
    return Snapshot(
        version="v1", source_id="s", created_at="t",
        columns=[_col(f"{prefix}claim", "id"), _col(f"{prefix}claim", "region"),
                 _col(f"{prefix}scratch", "id"), _col(f"{prefix}entitlement", "who")],
        views=list(views),
    )


_VISIBLE = {"claim": {"id", "region"}}


@pytest.fixture(autouse=True)
def _fresh_log_memory(monkeypatch):
    """The log memory, emptied per test so each sees its own first warning."""
    monkeypatch.setattr(policy_module, "_reported", set())


def _read(key: str, snapshot: Snapshot | None = None):
    grants = GrantSet(frozenset({"claim"}), row_filters={key: _FILTER})
    return decide("SELECT id FROM claim", _VISIBLE,
                  policy=build_access_policy(snapshot or _snapshot(), grants), target="duckdb")


@pytest.mark.parametrize("key", ["public.claim", "claims", "pg.claim"])
def test_a_filter_key_naming_no_table_refuses_the_read_without_naming_the_key(key):
    verdict = _read(key)
    assert isinstance(verdict, Refusal), verdict
    assert (verdict.code, verdict.subject) == (RefusalCode.INVALID_ROW_FILTER, None)
    assert key not in verdict.message


@pytest.mark.parametrize("key", ["claim", "Claim", "CLAIM"])
def test_a_key_the_snapshot_holds_still_binds_in_any_case(key):
    """The control, and the fold: unquoted identifiers are case-insensitive in every engine
    mnemiq runs on, so these name the table and the filter applies."""
    verdict = _read(key)
    assert isinstance(verdict, Approved), verdict
    assert _FILTER in verdict.target_sql


def test_a_filter_on_a_real_table_the_role_cannot_read_changes_nothing():
    """The other half of the decision: a key naming an object that exists is not suspicious.
    `entitlement` is in the snapshot and not granted, so its filter is irrelevant and the read of
    `claim` goes ahead as the policy wrote it."""
    grants = GrantSet(frozenset({"claim"}), row_filters={"entitlement": "who = 'u'"})
    policy = build_access_policy(_snapshot(), grants)
    assert policy.unknown_filter_targets == frozenset()
    assert isinstance(decide("SELECT id FROM claim", _VISIBLE, policy=policy, target="duckdb"),
                      Approved)


def test_a_key_naming_a_view_the_snapshot_holds_is_not_unknown():
    """A view is an object the snapshot holds even where none of its columns were recorded."""
    view = ViewDefinition(object_id="claim_v", definition="SELECT id FROM claim", dialect="duckdb")
    grants = GrantSet(frozenset({"claim_v"}), row_filters={"claim_v": "id > 0"})
    assert build_access_policy(_snapshot(views=[view]), grants).unknown_filter_targets == frozenset()


@pytest.mark.parametrize("key,refused", [("claim", True), ("pg.claim", False)])
def test_on_a_federated_snapshot_the_catalog_is_part_of_the_name(key, refused):
    """Federation qualifies every object id with its catalog, so there a bare key is the one
    that names nothing."""
    grants = GrantSet(frozenset({"pg.claim"}), row_filters={key: _FILTER})
    verdict = decide("SELECT id FROM pg.claim", {"pg.claim": {"id", "region"}},
                     policy=build_access_policy(_snapshot(prefix="pg."), grants), target="duckdb")
    assert isinstance(verdict, Refusal) is refused, verdict


def test_a_write_is_refused_too():
    grants = GrantSet(frozenset({"claim", "scratch"}), writable=frozenset({"scratch"}),
                      row_filters={"public.claim": _FILTER})
    verdict = decide_write("INSERT INTO scratch (id) SELECT id FROM claim",
                           {"claim": {"id", "region"}, "scratch": {"id"}}, grants,
                           policy=build_access_policy(_snapshot(), grants), dialect="duckdb",
                           writes_enabled=True)
    assert isinstance(verdict, Refusal), verdict
    assert verdict.code is RefusalCode.INVALID_ROW_FILTER


def test_the_question_is_deferred_before_any_model_call():
    """Every proposal would be refused identically, so none is asked for."""
    generator = FakeGenerator(['{"sql": "SELECT id FROM claim"}'])
    packet = ContextPacket(question="how many claims?",
                           cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
                           grant_fingerprint="fp", enrichment_version="v1")
    grants = GrantSet(frozenset({"claim"}), row_filters={"claims": _FILTER})
    outcome = plan_query(packet, _snapshot(), grants, generator, target="duckdb")
    assert isinstance(outcome, Deferred), outcome
    assert outcome.code is DeferralReason.UNGOVERNABLE
    assert "claims" not in outcome.reason
    assert generator.calls == []


def test_the_key_reaches_the_server_log_once(caplog):
    grants = GrantSet(frozenset({"claim"}), row_filters={"public.claim": _FILTER})
    with caplog.at_level(logging.WARNING, logger="mnemiq.sql.policy"):
        build_access_policy(_snapshot(), grants)
        build_access_policy(_snapshot(), grants)
    named = [r for r in caplog.records if "'public.claim'" in r.getMessage()]
    assert len(named) == 1, [r.getMessage() for r in caplog.records]


def test_a_snapshot_swap_logs_the_key_again_under_the_new_version(caplog):
    grants = GrantSet(frozenset({"claim"}), row_filters={"public.claim": _FILTER})
    swapped = _snapshot().model_copy(update={"version": "v2"})
    with caplog.at_level(logging.WARNING, logger="mnemiq.sql.policy"):
        build_access_policy(_snapshot(), grants)
        build_access_policy(swapped, grants)
    versions = [r.getMessage() for r in caplog.records if "'public.claim'" in r.getMessage()]
    assert len(versions) == 2 and "v2" in versions[1], versions


def test_two_snapshots_in_use_at_once_log_once_each_and_the_memory_stays_bounded(caplog):
    """A request in flight across a swap builds against the old version while new ones build
    against the new: alternating must not log on every call, and many swaps must not grow the
    memory without limit."""
    grants = GrantSet(frozenset({"claim"}), row_filters={"public.claim": _FILTER})
    v1, v2 = _snapshot(), _snapshot().model_copy(update={"version": "v2"})
    with caplog.at_level(logging.WARNING, logger="mnemiq.sql.policy"):
        for snapshot in (v1, v2, v1, v2, v1):
            build_access_policy(snapshot, grants)
    assert len([r for r in caplog.records if "'public.claim'" in r.getMessage()]) == 2
    for n in range(3 * policy_module._REPORTED_BOUND):
        build_access_policy(_snapshot().model_copy(update={"version": f"s{n}"}), grants)
    assert len(policy_module._reported) <= policy_module._REPORTED_BOUND


def test_the_boot_advisory_names_the_key(caplog):
    grants = GrantSet(frozenset({"claim"}), row_filters={"claims": _FILTER})
    with caplog.at_level(logging.WARNING, logger="mnemiq.authz.coverage"):
        warn_unknown_filter_targets(grants, _snapshot(), role="manager")
    assert any("'claims'" in r.getMessage() and "manager" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_the_boot_advisory_is_wired_into_the_runtime(caplog):
    """The join, not the end: the advisory is only useful if boot and a snapshot swap call it."""
    from mnemiq.runtime import _warn_policy_advisories

    class _Provider:
        def policy_roles(self):
            return ["manager"]

        def grants_for(self, identity: IdentityContext) -> GrantSet:
            return GrantSet(frozenset({"claim"}), row_filters={"claims": _FILTER})

    with caplog.at_level(logging.WARNING, logger="mnemiq.authz.coverage"):
        _warn_policy_advisories(_Provider(), _snapshot())
    assert any("'claims'" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]
