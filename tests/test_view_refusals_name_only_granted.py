"""M53 (M145 merged into it): a view refusal names only what the caller is granted.

A view refusal is repairable, so its message goes to the model as repair text and -- once the
retries run out -- to the caller as the deferral reason, word for word. It named the base table
and said it was row-filtered for the caller, to a caller granted only the view; and a nested view,
an unknown object or a cycle's whole chain the same way. It now names the view the query read,
and a dependency only where the caller reads it; the operator's log keeps the detail.
"""

from __future__ import annotations

import logging

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot, ViewDefinition
from mnemiq.generate.generator import FakeGenerator
from mnemiq.generate.plan_query import Deferred, plan_query
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal, RefusalCode

D = "duckdb"
SCHEMA = {"customer": {"id", "store_id"}, "v": {"id"}, "inner_v": {"id"}, "scratch": {"id"}}
POLICY = AccessPolicy(row_filters={"customer": "store_id = 1"}, policy_schema=SCHEMA)
ONLY_V = {"v": {"id"}}


def _views(**bodies: str) -> dict[str, ViewDefinition]:
    return {name: ViewDefinition(object_id=name, dialect=D, definition=body)
            for name, body in bodies.items()}


def _read(views, visible=ONLY_V) -> Refusal:
    verdict = decide("SELECT id FROM v", visible, dialect=D, target=D, policy=POLICY, views=views)
    assert isinstance(verdict, Refusal), verdict
    return verdict


def test_a_nested_view_and_its_base_table_are_not_named():
    verdict = _read(_views(v="SELECT id FROM inner_v", inner_v="SELECT id FROM customer"))
    assert verdict.code is RefusalCode.UNGOVERNED_VIEW
    assert verdict.subject == "v"
    assert "inner_v" not in verdict.message and "customer" not in verdict.message
    assert "'v'" in verdict.message


def test_an_object_the_body_reads_that_the_source_lacks_is_not_named():
    verdict = _read(_views(v="SELECT id FROM secret_layoffs"))
    assert verdict.code is RefusalCode.UNRESOLVABLE_VIEW
    assert "secret_layoffs" not in verdict.message and "'v'" in verdict.message


def test_a_cycle_is_named_by_the_view_the_query_read_only():
    verdict = _read(_views(v="SELECT id FROM inner_v", inner_v="SELECT id FROM v"))
    assert verdict.code is RefusalCode.UNRESOLVABLE_VIEW
    assert "inner_v" not in verdict.message and "->" not in verdict.message


def test_the_operator_log_keeps_what_the_caller_is_not_told(caplog):
    with caplog.at_level(logging.INFO, logger="mnemiq.sql.views"):
        _read(_views(v="SELECT id FROM inner_v", inner_v="SELECT id FROM customer"))
    assert any("inner_v" in r.getMessage() and "customer" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_the_write_decider_names_only_the_view_too():
    visible = {"v": {"id"}, "scratch": {"id"}}
    grants = GrantSet(frozenset(visible), writable=frozenset({"scratch"}))
    verdict = decide_write("INSERT INTO scratch (id) SELECT id FROM v", visible, grants,
                           policy=POLICY, dialect=D, writes_enabled=True,
                           views=_views(v="SELECT id FROM customer"))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW
    assert "customer" not in verdict.message


def test_the_write_decider_names_the_table_to_a_caller_who_reads_it():
    """The control for the write decider: it passes the caller's grants through."""
    visible = {"v": {"id"}, "scratch": {"id"}, "customer": {"id", "store_id"}}
    grants = GrantSet(frozenset(visible), writable=frozenset({"scratch"}))
    verdict = decide_write("INSERT INTO scratch (id) SELECT id FROM v", visible, grants,
                           policy=POLICY, dialect=D, writes_enabled=True,
                           views=_views(v="SELECT id FROM customer"))
    assert isinstance(verdict, Refusal) and "customer" in verdict.message


def test_the_caller_is_not_told_once_the_retries_run_out():
    """The join: `plan_query` hands the last refusal's message to the caller as the deferral
    reason, so the message is caller-facing however many attempts the model makes first."""
    snapshot = Snapshot(version="v1", source_id="s", created_at="t",
                        columns=[Column(id=f"{o}.{c}", object_id=o, name=c)
                                 for o, cols in SCHEMA.items() for c in cols],
                        views=list(_views(v="SELECT id FROM customer").values()))
    packet = ContextPacket(question="ids?", cards=[RetrievedCard(object_id="v", card="TABLE v",
                                                                 score=1.0)],
                           grant_fingerprint="fp", enrichment_version="v1")
    generator = FakeGenerator(['{"sql": "SELECT id FROM v"}'] * 3)
    grants = GrantSet(frozenset({"v"}), row_filters={"customer": "store_id = 1"})
    outcome = plan_query(packet, snapshot, grants, generator, target=D)
    assert isinstance(outcome, Deferred), outcome
    assert "row-filtered" in outcome.reason, outcome.reason  # the view refusal, not another exit
    assert "customer" not in outcome.reason
    assert not any("customer" in (c or "") for c in generator.calls)


def test_a_grant_on_a_same_named_table_elsewhere_does_not_name_it():
    """Review gate: matching on bare last segments let a grant on `sales.customer` name the
    filtered `customer` the view reads -- a table this caller does not hold."""
    verdict = _read(_views(v="SELECT id FROM customer"),
                    visible={"v": {"id"}, "sales.customer": {"id"}})
    assert "customer" not in verdict.message and "directly" not in verdict.message
