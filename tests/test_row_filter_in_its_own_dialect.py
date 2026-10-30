"""M218: a row filter is applied only to a statement decided as the engine that runs it.

A row filter is SQL in the source's own dialect. Decided in another dialect, the decider parsed the
filter as one engine's SQL and printed it as another's, with no equivalence behind the translation:
DuckDB's `greatest` and `concat` skip a NULL argument, and sqlglot prints them for SQLite as `MAX(id,
NULL)` and `region || NULL`, which give NULL, so a filter excluding a row on DuckDB admitted it on
SQLite, and a write copied it on (Codex's fiftieth review of #160, executed on SQLite). A function
can also keep its spelling and change its meaning (MySQL's GREATEST and CONCAT give NULL where
DuckDB's skip it), which no comparison of the two renderings could see, so the fix is the
precondition and not a list of safe translations. Every shipped door decides as the adapter's own
dialect; a direct call to `decide` with its defaults, duckdb for postgres, is the one reached.
"""

from __future__ import annotations

import sqlite3

import duckdb
import pytest
import sqlglot

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide as _decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode
from tests.stored_names import as_stored, keyed, policy_keyed, stored

SCHEMA = {"claim": {"id", "region"}, "scratch": {"id", "region"}}


def decide(sql, visible, *, dialect="duckdb", target="postgres", policy=None, **kwargs):
    """`decide`, each table spelled as the target stores it unquoted (M223): on Oracle a grant
    keys CLAIM, and a statement reads it as CLAIM."""
    policy = policy_keyed(policy, target, *SCHEMA) if policy is not None else None
    return _decide(as_stored(sql, target, *SCHEMA), keyed(visible, target, *SCHEMA),
                   dialect=dialect, target=target, policy=policy, **kwargs)
DRIFTING = ["greatest(id, NULL) IS NULL", "concat(region, NULL) IS NULL"]
ROWS = "CREATE TABLE claim (id INT, region VARCHAR); INSERT INTO claim VALUES (1, 'west')"


def _filtered(filt: str) -> AccessPolicy:
    return AccessPolicy(row_filters={"claim": filt}, policy_schema=SCHEMA)


@pytest.mark.parametrize("filt", DRIFTING)
def test_the_translation_changes_which_rows_the_filter_keeps(filt):
    # The defect, measured on both engines: DuckDB keeps no row, SQLite's rendering keeps it.
    duck = duckdb.connect()
    duck.execute(ROWS)
    assert duck.execute(f"SELECT id FROM claim WHERE {filt}").fetchall() == []
    lite = sqlite3.connect(":memory:")
    lite.executescript(ROWS + ";")
    printed = sqlglot.transpile(filt, read="duckdb", write="sqlite")[0]
    assert lite.execute(f"SELECT id FROM claim WHERE {printed}").fetchall() == [(1,)], printed


@pytest.mark.parametrize("filt", DRIFTING)
@pytest.mark.parametrize("target", ["sqlite", "postgres", "mysql", "oracle"])
def test_a_filter_decided_in_another_dialect_is_refused_as_the_deployment_s(filt, target):
    verdict = decide("SELECT id FROM claim", SCHEMA, dialect="duckdb", target=target, policy=_filtered(filt))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.POLICY_TARGET_UNSUPPORTED, verdict
    assert not verdict.repairable  # the model cannot write its way past a deployment setting
    assert repr(stored("claim", target)) in verdict.message and repr(target) in verdict.message, verdict.message


@pytest.mark.parametrize("filt", DRIFTING)
def test_decided_as_the_engine_that_runs_it_the_filter_applies(filt):
    verdict = decide("SELECT id FROM claim", SCHEMA, dialect="duckdb", target="duckdb", policy=_filtered(filt))
    assert isinstance(verdict, Approved), verdict
    duck = duckdb.connect()
    duck.execute(ROWS)
    assert duck.execute(verdict.target_sql).fetchall() == []  # the row the filter excludes stays out


@pytest.mark.parametrize("engine", ["duckdb", "postgres", "sqlite", "oracle"])
def test_every_engine_decided_as_itself_applies_its_filter(engine):
    # The shipped doors' shape: the adapter's dialect for both.
    verdict = decide("SELECT id FROM claim", SCHEMA, dialect=engine, target=engine,
                     policy=_filtered("region = 'west'"))
    assert isinstance(verdict, Approved), (engine, verdict)
    assert "region = 'west'" in verdict.target_sql, verdict.target_sql


def test_decide_s_default_pair_is_another_dialect():
    # `decide` decides as duckdb for postgres unless told otherwise: a direct caller under a row
    # filter is refused until it names the engine it runs on.
    verdict = decide("SELECT id FROM claim", SCHEMA, policy=_filtered("region = 'west'"))
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.POLICY_TARGET_UNSUPPORTED, verdict


def test_a_mask_is_not_translated_text_and_still_applies_across_dialects():
    policy = AccessPolicy(masked={("claim", "region")}, policy_schema=SCHEMA)
    verdict = decide("SELECT id, region FROM claim", SCHEMA, dialect="duckdb", target="postgres", policy=policy)
    assert isinstance(verdict, Approved), verdict
    assert "NULL AS region" in verdict.target_sql.replace('"', ""), verdict.target_sql


def test_a_filter_on_a_table_the_statement_does_not_read_refuses_nothing():
    policy = AccessPolicy(row_filters={"scratch": DRIFTING[0]}, policy_schema=SCHEMA)
    verdict = decide("SELECT id FROM claim", SCHEMA, dialect="duckdb", target="sqlite", policy=policy)
    assert isinstance(verdict, Approved), verdict


@pytest.mark.parametrize("sql", [
    "INSERT INTO scratch (id, region) SELECT id, region FROM claim",  # the filter on a read
    "UPDATE claim SET region = 'east' WHERE id = 1",  # the filter conjoined to the target
    "DELETE FROM claim WHERE id = 1",
])
def test_the_write_decider_applies_the_same_precondition(sql):
    grants = GrantSet(frozenset(SCHEMA), writable=frozenset(SCHEMA), row_filters={"claim": DRIFTING[0]})
    elsewhere = decide_write(sql, SCHEMA, grants, dialect="duckdb", target="sqlite",
                             policy=_filtered(DRIFTING[0]), writes_enabled=True)
    assert isinstance(elsewhere, Refusal) and elsewhere.code is RefusalCode.POLICY_TARGET_UNSUPPORTED, elsewhere
    itself = decide_write(sql, SCHEMA, grants, dialect="duckdb", target="duckdb",
                          policy=_filtered(DRIFTING[0]), writes_enabled=True)
    assert getattr(itself, "code", None) is not RefusalCode.POLICY_TARGET_UNSUPPORTED, itself


def test_with_no_engine_named_the_parse_dialect_runs_it():
    # `apply_row_and_mask` reads an absent `executes_as` as the parse dialect throughout (its
    # resolver binds `executes_as or dialect`), so the filter applies.
    from mnemiq.sql.rls import apply_row_and_mask

    ast = sqlglot.parse_one("SELECT id FROM claim", read="duckdb")
    out, narrowed = apply_row_and_mask(ast, _filtered("region = 'west'"), SCHEMA, dialect="duckdb")
    assert not isinstance(out, Refusal), out
    assert "region = 'west'" in out.sql(dialect="duckdb") and [n.object for n in narrowed] == ["claim"]
