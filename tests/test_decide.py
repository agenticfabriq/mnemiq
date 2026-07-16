import os

import pytest

from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

VISIBLE = {"claim": {"claim_identifier", "claim_open_date"}, "policy": {"policy_identifier"}}

_DSN = "postgresql://mnemiq:mnemiq@localhost:5433/acme"


def test_an_approved_query_carries_both_dialects_and_its_references():
    verdict = decide("SELECT claim_identifier FROM claim", VISIBLE)
    assert isinstance(verdict, Approved)

    assert "LIMIT 1000" in verdict.plan_sql  # the limit was imposed
    assert "LIMIT 1000" in verdict.target_sql
    assert verdict.tables == ["claim"]
    assert verdict.columns == ["claim_identifier"]


def test_the_shape_guard_runs_before_the_access_guard():
    # a DROP naming an unauthorized table is refused as a DROP -- we never get far enough to
    # discuss what it names
    verdict = decide("DROP TABLE person", VISIBLE)
    assert isinstance(verdict, Refusal)
    assert verdict.code == RefusalCode.NOT_SELECT_ONLY


def test_an_unauthorized_table_is_refused():
    verdict = decide("SELECT last_name FROM person", VISIBLE)
    assert isinstance(verdict, Refusal)
    assert verdict.code == RefusalCode.UNAUTHORIZED_TABLE
    assert not verdict.repairable  # never repaired into existence; deferred instead


def test_transpiling_targets_the_source_dialect():
    verdict = decide(
        "SELECT claim_identifier FROM claim WHERE claim_open_date > '2020-01-01'", VISIBLE
    )
    assert isinstance(verdict, Approved)
    assert "SELECT" in verdict.target_sql


@pytest.mark.integration
def test_explain_proves_the_query_against_the_real_source():
    from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter

    adapter = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))
    visible = {"claim": {"claim_identifier", "claim_open_date"}}

    verdict = decide(
        "SELECT claim_identifier FROM claim", visible, adapter=adapter, target="duckdb"
    )
    assert isinstance(verdict, Approved), verdict


@pytest.mark.integration
def test_a_query_the_snapshot_believes_but_the_source_denies_is_refused():
    # the snapshot says claim.ghost_column exists; the source disagrees. Only EXPLAIN knows.
    from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter

    adapter = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))
    stale = {"claim": {"ghost_column"}}

    verdict = decide("SELECT ghost_column FROM claim", stale, adapter=adapter, target="duckdb")
    assert isinstance(verdict, Refusal)
    assert verdict.code == RefusalCode.EXPLAIN_FAILED
    assert verdict.repairable


def test_decide_flags_a_logic_lint_and_refuses():
    from mnemiq.sql.decide import decide
    from mnemiq.sql.verdict import Refusal, RefusalCode

    visible = {"t": {"name", "score"}}
    verdict = decide("SELECT name FROM t ORDER BY score LIMIT 1", visible, adapter=None)
    assert isinstance(verdict, Refusal)
    assert verdict.code == RefusalCode.LOGIC_LINT
    assert verdict.repairable


def test_decide_approves_the_guarded_form():
    from mnemiq.sql.decide import decide
    from mnemiq.sql.verdict import Approved

    visible = {"t": {"name", "score"}}
    verdict = decide(
        "SELECT name FROM t WHERE score IS NOT NULL ORDER BY score LIMIT 1", visible, adapter=None
    )
    assert isinstance(verdict, Approved)
