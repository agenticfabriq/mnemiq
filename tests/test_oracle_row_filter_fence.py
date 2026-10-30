"""M160: on Oracle a row filter is applied before the statement's own predicates.

Oracle may merge a filtered derived table into the outer query, or push the outer predicates into
it, and then evaluate the caller's conditions on rows the filter removes -- measured on Oracle Free
23 (register M160). The owner's decision (2026-10-15): fence the reads, refuse the writes that
cannot be fenced. These pin the rewrite's shape on each engine; the live suite runs it on Oracle.
"""

from __future__ import annotations

import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode

SCHEMA = {"claim": {"id", "amount"}, "entitlement": {"id", "who"}, "scratch": {"id", "amount"}}
FILTER = "id IN (SELECT id FROM entitlement WHERE who = 'r')"
POLICY = AccessPolicy(row_filters={"claim": FILTER}, policy_schema=SCHEMA)
VISIBLE = {"claim": SCHEMA["claim"], "scratch": SCHEMA["scratch"]}
GRANTS = GrantSet(frozenset({"claim", "scratch"}), writable=frozenset({"claim", "scratch"}),
                  row_filters={"claim": FILTER})


def _read(sql, engine, policy=POLICY):
    return decide(sql, VISIBLE, dialect=engine, target=engine, policy=policy)


def _write(sql, engine):
    return decide_write(sql, VISIBLE, GRANTS, policy=POLICY, dialect=engine, target=engine,
                        writes_enabled=True)


def test_an_oracle_read_fences_the_filtered_table_inside_its_derived_table():
    verdict = _read("SELECT id FROM claim WHERE amount > 5", "oracle")
    assert isinstance(verdict, Approved), verdict
    # Inside the derived table, beside the policy: outside it, the fence would fence nothing.
    assert "WHERE (id IN (SELECT id FROM entitlement WHERE who = 'r')\n) AND ROWNUM > 0) claim" in (
        verdict.target_sql), verdict.target_sql
    assert "ROWNUM" not in verdict.asked_sql  # what the caller is shown is what they asked


def test_every_filtered_reference_is_fenced():
    verdict = _read("SELECT a.id FROM claim a JOIN claim b ON a.id = b.id "
                    "WHERE a.amount IN (SELECT amount FROM claim)", "oracle")
    assert isinstance(verdict, Approved), verdict
    assert verdict.target_sql.count("ROWNUM > 0") == 3, verdict.target_sql


def test_a_duckdb_read_is_not_fenced():
    """DuckDB applies the policy first (pinned by the repair-feedback suite's filter-order matrix),
    and ROWNUM is not DuckDB's."""
    verdict = _read("SELECT id FROM claim WHERE amount > 5", "duckdb")
    assert isinstance(verdict, Approved) and "ROWNUM" not in verdict.target_sql.upper()


def test_a_mask_alone_is_not_fenced():
    """A mask removes no row, and a merged mask still reads NULL."""
    masked = AccessPolicy(masked={("claim", "amount")}, policy_schema=SCHEMA)
    verdict = _read("SELECT id, amount FROM claim", "oracle", masked)
    assert isinstance(verdict, Approved) and "ROWNUM" not in verdict.target_sql


@pytest.mark.parametrize("sql", ["UPDATE claim SET amount = 1 WHERE id = 2",
                                 "DELETE FROM claim WHERE id = 2"])
def test_an_oracle_write_to_a_filtered_target_is_refused(sql):
    """The target is filtered by conjunction, which no fence reaches."""
    verdict = _write(sql, "oracle")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.POLICY_TARGET_UNSUPPORTED
    assert not verdict.repairable  # the deployment's limit: no rewrite of the statement passes it
    assert "not of your grants" in verdict.message


@pytest.mark.parametrize("sql", ["INSERT INTO scratch (id, amount) SELECT id, amount FROM claim",
                                 "UPDATE scratch SET amount = 1 WHERE id IN (SELECT id FROM claim)"])
def test_an_oracle_write_that_only_reads_a_filtered_table_is_fenced_not_refused(sql):
    verdict = _write(sql, "oracle")
    assert isinstance(verdict, ApprovedWrite), verdict
    assert "AND ROWNUM > 0" in verdict.target_sql, verdict.target_sql


@pytest.mark.parametrize("sql", ["UPDATE claim SET amount = 1 WHERE id = 2",
                                 "DELETE FROM claim WHERE id = 2"])
def test_the_same_writes_on_duckdb_are_approved_as_before(sql):
    verdict = _write(sql, "duckdb")
    assert isinstance(verdict, ApprovedWrite), verdict
    assert "ROWNUM" not in verdict.target_sql.upper()
