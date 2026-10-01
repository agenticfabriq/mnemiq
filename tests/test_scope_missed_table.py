"""M121: a write whose scope model misses a table it reads fails closed, whatever sqlglot thinks.

sqlglot 30.17 began scoping UPDATE, and its scope lists none of the FROM tables. `base_tables`
trusted the scope walk, so `UPDATE scratch ... FROM claim` reported `scratch` alone: `claim` was
neither grant-checked nor row-filtered, and an identity with no grant on it could copy it into a
table it could write. The suite runs on the locked sqlglot, whose scope gives up on UPDATE and so
never showed it -- these tests stand in a scope that behaves as 30.17's does.
"""

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope

import mnemiq.sql.scope as scope_module
from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.scope import base_tables
from mnemiq.sql.verdict import ApprovedWrite, Refusal, RefusalCode

_SQL = "UPDATE scratch SET amount = claim.amount FROM claim WHERE scratch.id = claim.id"
_SCHEMA = {"claim": {"id", "amount", "region"}, "scratch": {"id", "amount"}}


def _blind(monkeypatch):
    """Stand in sqlglot 30.17's UPDATE scope as mnemiq walks it: a real scope that lists no tables
    and no sources -- the scope of `SELECT 1`."""
    empty = build_scope(sqlglot.parse_one("SELECT 1"))
    assert not empty.tables and not empty.sources
    real = scope_module._root_scope  # every other statement -- the row filter's too -- as before
    monkeypatch.setattr(scope_module, "_root_scope",
                        lambda ast: empty if isinstance(ast, exp.Update) else real(ast))


class _Ok:
    def execute(self, sql):
        return []


def _write(granted: set[str]):
    visible = {t: c for t, c in _SCHEMA.items() if t in granted}
    return decide_write(_SQL, visible, GrantSet(frozenset(granted), writable=frozenset({"scratch"})),
                        adapter=_Ok(), dialect="duckdb", writes_enabled=True,
                        policy=AccessPolicy(row_filters={"claim": "region = 'west'"},
                                            policy_schema=_SCHEMA))


def test_a_table_the_scope_missed_is_still_reported(monkeypatch):
    _blind(monkeypatch)
    names = sorted(t.name for t in base_tables(sqlglot.parse_one(_SQL, read="duckdb")))
    assert names == ["claim", "scratch"]


def test_a_write_reading_an_ungranted_table_the_scope_missed_is_refused(monkeypatch):
    _blind(monkeypatch)
    verdict = _write({"scratch"})
    assert isinstance(verdict, Refusal), verdict
    assert verdict.code == RefusalCode.UNAUTHORIZED_TABLE and verdict.subject == "claim"


def test_a_granted_table_the_scope_missed_is_row_filtered(monkeypatch):
    _blind(monkeypatch)
    verdict = _write({"claim", "scratch"})
    assert isinstance(verdict, ApprovedWrite), verdict
    assert "region" in verdict.plan_sql, "the read of claim carries its row filter"


def test_an_insert_target_is_not_mistaken_for_a_missed_read(monkeypatch):
    # INSERT writes its target without reading it: excused, so a scope that sees the SELECT's
    # tables correctly does not trip the fallback.
    ast = sqlglot.parse_one("INSERT INTO scratch (id, amount) SELECT id, amount FROM claim", read="duckdb")
    assert sorted(t.name for t in base_tables(ast)) == ["claim"]
    assert not any(isinstance(t, exp.Table) and t.name == "scratch" for t in base_tables(ast))


_SHADOW = ("UPDATE scratch SET amount = (WITH claim AS (SELECT 1 AS amount) SELECT amount FROM claim) "
           "FROM claim WHERE scratch.id = claim.id")


def test_a_cte_in_a_subquery_does_not_hide_the_real_table_of_its_name():
    """On the LOCKED sqlglot, no stand-in: a CTE named `claim` inside a scalar subquery hid the
    outer `FROM claim` from every guard, and the write was approved with `claim` ungranted (M122)."""
    visible = {"scratch": _SCHEMA["scratch"]}
    verdict = decide_write(_SHADOW, visible, GrantSet(frozenset({"scratch"}), writable=frozenset({"scratch"})),
                           adapter=_Ok(), dialect="duckdb", writes_enabled=True,
                           policy=AccessPolicy(policy_schema=_SCHEMA))
    assert isinstance(verdict, Refusal), verdict
    assert verdict.code == RefusalCode.UNAUTHORIZED_TABLE and verdict.subject == "claim"


def test_a_cte_on_an_enclosing_query_still_stands_in_front_of_its_name():
    ast = sqlglot.parse_one("INSERT INTO scratch (id, amount) WITH c AS (SELECT id, amount FROM claim) "
                            "SELECT id, amount FROM c", read="duckdb")
    assert sorted(t.name for t in base_tables(ast)) == ["claim"]


def test_a_merge_is_checked_and_its_target_is_not_excused(monkeypatch):
    """MERGE reads its target and nothing models that, so a scope that misses its tables must
    fall back to all of them -- the check runs on MERGE even though no target is excused."""
    empty = build_scope(sqlglot.parse_one("SELECT 1"))
    real = scope_module._root_scope
    monkeypatch.setattr(scope_module, "_root_scope",
                        lambda ast: empty if isinstance(ast, exp.Merge) else real(ast))
    ast = sqlglot.parse_one("MERGE INTO scratch USING claim ON scratch.id = claim.id "
                            "WHEN MATCHED THEN UPDATE SET amount = claim.amount", read="duckdb")
    assert sorted(t.name for t in base_tables(ast)) == ["claim", "scratch"]



def _ungranted(sql: str, dialect: str = "duckdb"):
    return decide_write(sql, {"scratch": _SCHEMA["scratch"]},
                        GrantSet(frozenset({"scratch"}), writable=frozenset({"scratch"})),
                        adapter=_Ok(), dialect=dialect, writes_enabled=True,
                        policy=AccessPolicy(policy_schema=_SCHEMA))


def test_a_qualified_table_is_never_excused_as_a_cte():
    """Review of the scope-aware rule, on the locked sqlglot, no stand-in: `main.claim` was taken
    for the CTE `claim`, and the write approved with `claim` ungranted. A CTE has no schema."""
    verdict = _ungranted("UPDATE scratch SET amount = x.amount FROM (WITH claim AS (SELECT 1 AS amount) "
                         "SELECT amount FROM main.claim) x WHERE scratch.id = 1")
    assert isinstance(verdict, Refusal) and verdict.code == RefusalCode.UNAUTHORIZED_TABLE, verdict


def test_a_quoted_cte_does_not_excuse_a_table_the_engine_names_differently():
    """Postgres: `"Claim"` and `claim` are different names; lowercasing both excused the table."""
    verdict = _ungranted('UPDATE scratch SET amount = x.amount FROM (WITH "Claim" AS (SELECT 1 AS amount) '
                         "SELECT amount FROM claim) x WHERE scratch.id = 1", dialect="postgres")
    assert isinstance(verdict, Refusal) and verdict.code == RefusalCode.UNAUTHORIZED_TABLE, verdict
