"""M156: `Approved.asked_plan` is the statement as asked, in the plan's dialect, for model calls.

The judge, the selector and the synthesizer were shown `plan_sql`, which carries the row or
column policy's rewrite. Measured by the ui lane on the acme_syn demo as west_region (row filter
`region = 'west'`): the judge withdrew 9 of 9 answers on policy questions; at each verifier call,
with only the SQL swapped, the rewritten plan scored a median 0.05 (24 of 24 under the 0.5
threshold) and the statement as asked 0.94 (0 of 24 under). `asked_plan` is what they read
instead (the consumer change is the ui lane's); execution, the cache key and the trace keep
`plan_sql` and `target_sql`.
"""

from __future__ import annotations

import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved

VISIBLE = {"policy": {"id", "region", "premium"}}
SQL = "SELECT region, count(*) AS n FROM policy GROUP BY region"


def _approved(policy=None, target="duckdb"):
    verdict = decide(SQL, VISIBLE, dialect="duckdb", target=target, policy=policy)
    assert isinstance(verdict, Approved), verdict
    return verdict


def test_with_nothing_narrowed_it_is_the_plan_byte_for_byte():
    """What keeps every prompt with no policy identical to before."""
    approved = _approved()
    assert approved.asked_plan == approved.plan_sql


def test_under_a_row_filter_it_carries_no_policy_predicate():
    approved = _approved(AccessPolicy(row_filters={"policy": "region = 'west'"},
                                      policy_schema=VISIBLE))
    assert "'west'" in approved.plan_sql
    assert "'west'" not in approved.asked_plan and "GROUP BY" in approved.asked_plan


def test_it_is_in_the_plan_dialect_not_the_targets():
    """`asked_sql` is the target's rendering -- Oracle's FETCH FIRST here -- and would change
    every prompt on a source whose target differs from the plan. Under a mask: a row filter is
    applied only to a statement decided as the engine that runs it (M218)."""
    approved = _approved(AccessPolicy(masked={("policy", "premium")},
                                      policy_schema=VISIBLE), target="oracle")
    assert "FETCH FIRST" in approved.asked_sql
    assert "LIMIT" in approved.asked_plan and "FETCH" not in approved.asked_plan


def test_model_sql_is_the_statement_as_asked_under_a_policy():
    approved = _approved(AccessPolicy(row_filters={"policy": "region = 'west'"},
                                      policy_schema=VISIBLE))
    assert approved.model_sql() == approved.asked_plan


def test_model_sql_falls_back_to_the_plan_only_where_no_policy_was_evaluated():
    """`verify_replay` builds `Approved` itself, with neither field set."""
    assert Approved(plan_sql="SELECT 1", target_sql="SELECT 1").model_sql() == "SELECT 1"
    evaluated = Approved(plan_sql="SELECT 1 WHERE region = 'west'", target_sql="x", narrowed=[])
    with pytest.raises(ValueError):
        evaluated.model_sql()
