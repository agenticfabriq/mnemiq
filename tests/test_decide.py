import os

import pytest

from acme_dsn import acme_dsn, requires_acme

from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode

VISIBLE = {"claim": {"claim_identifier", "claim_open_date"}, "policy": {"policy_identifier"}}

_DSN = acme_dsn()


def test_an_approved_query_carries_both_dialects_and_its_references():
    verdict = decide("SELECT claim_identifier FROM claim", VISIBLE)
    assert isinstance(verdict, Approved)

    assert "LIMIT 1000" in verdict.plan_sql  # the limit was imposed
    assert "LIMIT 1000" in verdict.target_sql
    assert verdict.tables == ["claim"]
    assert verdict.columns == ["claim_identifier"]


def test_a_value_grounding_miss_is_refused_when_an_index_is_supplied():
    from tests.test_values_check import IDX  # reuse the fake index

    v = decide(
        "SELECT Country FROM gasstations WHERE Country = 'CZE'",
        {"gasstations": {"Country"}},
        values=IDX,
    )
    assert isinstance(v, Refusal) and v.code == RefusalCode.VALUE_GROUNDING


def test_without_an_index_the_value_check_is_skipped():
    v = decide(
        "SELECT Country FROM gasstations WHERE Country = 'CZE'",
        {"gasstations": {"Country"}},
    )
    assert isinstance(v, Approved)  # values=None -> today's behavior, no value check


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


def test_a_capped_rollup_reaches_the_target_dialect_instead_of_crashing():
    # M137: sqlglot parses `GROUP BY ... WITH ROLLUP` and the row cap appends `LIMIT 1000`, but sqlglot cannot
    # parse its own `... WITH ROLLUP LIMIT 1000` back, so re-reading the capped plan raised out of `decide`.
    # The plan is re-read without its cap and the cap put back (`_transpile_under_the_cap`).
    verdict = decide(
        "SELECT claim_identifier, COUNT(claim_open_date) AS n FROM claim GROUP BY claim_identifier WITH ROLLUP",
        VISIBLE, target="mysql",
    )
    assert isinstance(verdict, Approved)
    assert verdict.target_sql.endswith("GROUP BY claim_identifier WITH ROLLUP LIMIT 1000")


def test_a_parseable_plan_still_reaches_the_target_through_the_dialect_round_trip():
    # The round trip normalizes dialect types: BigQuery's FLOAT is 64-bit, and the plan text re-read as BigQuery
    # becomes DuckDB's DOUBLE. Rendering the tree straight to DuckDB gives REAL, which loses precision at
    # 16777217 and still passes EXPLAIN. So the target is never rendered straight from the tree, not even in
    # M137's fallback, which re-reads the plan without its cap instead.
    verdict = decide("SELECT CAST(a AS FLOAT) AS x FROM t", {"t": {"a"}}, dialect="bigquery", target="duckdb")
    assert isinstance(verdict, Approved)
    assert "CAST(a AS DOUBLE)" in verdict.target_sql


def test_a_capped_grouping_set_keeps_the_dialect_round_trip_and_its_precision():
    # sqlglot 30.12 cannot parse a LIMIT straight after ROLLUP, CUBE or GROUPING SETS, and the cap puts it there.
    # Rendering that plan straight from the tree skips the round trip's type normalization: BigQuery's 64-bit
    # FLOAT becomes DuckDB's REAL, and 16777217 comes back as 16777216 after EXPLAIN has passed.
    import duckdb

    verdict = decide(
        "SELECT CAST(a AS FLOAT) AS x, COUNT(*) AS n FROM t GROUP BY ROLLUP(a)", {"t": {"a"}},
        dialect="bigquery", target="duckdb",
    )
    assert isinstance(verdict, Approved)
    assert verdict.target_sql.endswith("LIMIT 1000")
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 16777217::BIGINT AS a")
    assert 16777217.0 in {row[0] for row in con.execute(verdict.target_sql).fetchall()}


@pytest.mark.integration
@requires_acme
def test_explain_proves_the_query_against_the_real_source():
    from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter

    adapter = DuckDBPostgresAdapter(os.getenv("MNEMIQ_PG_DSN", _DSN))
    visible = {"claim": {"claim_identifier", "claim_open_date"}}

    verdict = decide(
        "SELECT claim_identifier FROM claim", visible, adapter=adapter, target="duckdb"
    )
    assert isinstance(verdict, Approved), verdict


@pytest.mark.integration
@requires_acme
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


def test_decide_with_empty_policy_is_unchanged():
    from mnemiq.sql.decide import decide
    from mnemiq.sql.policy import AccessPolicy
    from mnemiq.sql.verdict import Approved

    visible = {"claim": {"id", "amount"}}
    a = decide("SELECT id, amount FROM claim", visible, dialect="duckdb", target="duckdb")
    b = decide("SELECT id, amount FROM claim", visible, dialect="duckdb", target="duckdb",
               policy=AccessPolicy())
    assert isinstance(a, Approved) and isinstance(b, Approved)
    assert a.target_sql == b.target_sql


def test_decide_injects_row_filter_and_masks():
    from mnemiq.sql.decide import decide
    from mnemiq.sql.policy import AccessPolicy
    from mnemiq.sql.verdict import Approved

    visible = {"claim": {"id", "amount", "ssn"}}
    pol = AccessPolicy(row_filters={"claim": "amount > 0"}, masked={("claim", "ssn")})
    v = decide("SELECT id, ssn FROM claim", visible, dialect="duckdb", target="duckdb", policy=pol)
    assert isinstance(v, Approved)
    low = v.target_sql.lower()
    assert "amount > 0" in low and "null as ssn" in low


def test_decide_refuses_denied_column():
    from mnemiq.sql.decide import decide
    from mnemiq.sql.policy import AccessPolicy
    from mnemiq.sql.verdict import Refusal, RefusalCode

    visible = {"claim": {"id", "ssn"}}
    v = decide("SELECT ssn FROM claim", visible, dialect="duckdb", target="duckdb",
               policy=AccessPolicy(denied={("claim", "ssn")}))
    assert isinstance(v, Refusal) and v.code == RefusalCode.UNAUTHORIZED_COLUMN
