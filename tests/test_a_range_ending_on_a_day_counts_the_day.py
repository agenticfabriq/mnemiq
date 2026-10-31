"""A range written to end on a bare date against a timestamp column ends at the next day's start.

`ts <= '2024-03-09'` compares with midnight at the START of the 9th, so every row later that day
falls out, where the range a person means by "as of", "through" or "up to" the 9th ends before the
10th. A local 32B model wrote this bound on most runs of the questions that turn on it, and every
such answer was wrong; ending the range at the next day turned them right and turned no right
answer wrong. The decider moves the bound (`close_day_bounds`): `<=` becomes `<` the next day,
in the literal's own spelling, so the statement stays valid in every dialect, and the caller, the
judge and the source all read the moved bound. Only a bare column placed on a column declared a
naive timestamp (or an Oracle DATE, which holds a time of day) is read this way: on a DATE the two
bounds are the same rows, and on text, a zone-aware timestamp or an expression nothing is moved.
"""
from __future__ import annotations

import json

import duckdb
import pytest
import sqlglot

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.modes import MODES, build_agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.generate.generator import FakeGenerator
from mnemiq.runtime import _build_mode_verifiers
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.server.serialize import answer_payload
from mnemiq.sql.day_bounds import close_day_bounds
from mnemiq.sql.decide import decide
from mnemiq.sql.verdict import Approved

VISIBLE = {"claim": {"id", "ts", "filed", "tz", "note", "region"}}
TYPES = {"claim": {"id": "INTEGER", "ts": "TIMESTAMP", "filed": "DATE", "tz": "TIMESTAMPTZ",
                   "note": "VARCHAR", "region": "VARCHAR"}}


def _closed(sql: str, dialect: str = "duckdb", types=TYPES) -> tuple[str, int]:
    tree = sqlglot.parse_one(sql, read=dialect)
    moved = close_day_bounds(tree, VISIBLE, types, dialect)
    return tree.sql(dialect=dialect), moved


@pytest.mark.parametrize("asked, closed", [
    ("SELECT id FROM claim WHERE ts <= '2024-03-09'",
     "SELECT id FROM claim WHERE ts < '2024-03-10'"),
    ("SELECT id FROM claim WHERE ts <= CAST('2024-03-09' AS TIMESTAMP)",
     "SELECT id FROM claim WHERE ts < CAST('2024-03-10' AS TIMESTAMP)"),
    ("SELECT id FROM claim WHERE ts <= DATE '2024-03-09'",
     "SELECT id FROM claim WHERE ts < CAST('2024-03-10' AS DATE)"),
    ("SELECT id FROM claim WHERE ts <= TRY_CAST('2024-03-09' AS TIMESTAMP)",
     "SELECT id FROM claim WHERE ts < TRY_CAST('2024-03-10' AS TIMESTAMP)"),
    # The literal on the left is the same bound.
    ("SELECT id FROM claim WHERE '2024-03-09' >= ts",
     "SELECT id FROM claim WHERE '2024-03-10' > ts"),
    # A qualified column, through its alias.
    ("SELECT c.id FROM claim AS c WHERE c.ts <= '2024-03-09'",
     "SELECT c.id FROM claim AS c WHERE c.ts < '2024-03-10'"),
    # The month, the year and a leap day roll over as a calendar does.
    ("SELECT id FROM claim WHERE ts <= '2024-12-31'",
     "SELECT id FROM claim WHERE ts < '2025-01-01'"),
    ("SELECT id FROM claim WHERE ts <= '2024-02-28'",
     "SELECT id FROM claim WHERE ts < '2024-02-29'"),
])
def test_an_inclusive_end_on_a_bare_date_moves_to_the_next_days_start(asked, closed):
    assert _closed(asked) == (closed, 1)


def test_a_between_ending_on_a_bare_date_keeps_its_start_and_ends_before_the_next_day():
    assert _closed("SELECT id FROM claim WHERE ts BETWEEN '2024-03-01' AND '2024-03-09'") == (
        "SELECT id FROM claim WHERE (ts >= '2024-03-01' AND ts < '2024-03-10')", 1)
    # Negated, the whole range is negated: the rows outside the days the range names.
    assert _closed("SELECT id FROM claim WHERE NOT ts BETWEEN '2024-03-01' AND '2024-03-09'") == (
        "SELECT id FROM claim WHERE NOT (ts >= '2024-03-01' AND ts < '2024-03-10')", 1)


def test_every_such_bound_in_the_statement_moves_wherever_it_sits():
    sql, moved = _closed(
        "WITH recent AS (SELECT id, ts FROM claim WHERE ts <= '2024-03-09') "
        "SELECT id, CASE WHEN ts <= '2024-03-05' THEN 'early' ELSE 'late' END AS half FROM recent")
    assert moved == 2
    assert "ts < '2024-03-10'" in sql and "ts < '2024-03-06'" in sql and "<=" not in sql


@pytest.mark.parametrize("asked", [
    # A DATE holds whole days, where `<=` the day and `<` the next are the same rows.
    "SELECT id FROM claim WHERE filed <= '2024-03-09'",
    # A zone-aware timestamp's midnight is the session's zone's.
    "SELECT id FROM claim WHERE tz <= '2024-03-09'",
    # Text compares as text.
    "SELECT id FROM claim WHERE note <= '2024-03-09'",
    # A column of no declared type, and one the statement does not resolve.
    "SELECT id FROM claim WHERE missing <= '2024-03-09'",
    # An expression is not the column.
    "SELECT id FROM claim WHERE CAST(ts AS DATE) <= '2024-03-09'",
    # A bound with its time of day says where it ends.
    "SELECT id FROM claim WHERE ts <= '2024-03-09 00:00:00'",
    "SELECT id FROM claim WHERE ts <= '2024-03-09 23:59:59'",
    # A string that names no day.
    "SELECT id FROM claim WHERE ts <= '2024-02-30'",
    # The exclusive end and the inclusive start already mean the day boundary they name.
    "SELECT id FROM claim WHERE ts < '2024-03-10'",
    "SELECT id FROM claim WHERE ts >= '2024-03-01'",
    # Not measured, so not moved: a strict start and a single day by equality.
    "SELECT id FROM claim WHERE ts > '2024-03-09'",
    "SELECT id FROM claim WHERE ts = '2024-03-09'",
    # A BETWEEN whose end has a time of day.
    "SELECT id FROM claim WHERE ts BETWEEN '2024-03-01' AND '2024-03-09 12:00:00'",
    # Text, in the mirrored and the BETWEEN spellings too.
    "SELECT id FROM claim WHERE '2024-03-09' >= note",
    "SELECT id FROM claim WHERE note BETWEEN '2024-03-01' AND '2024-03-09'",
])
def test_nothing_moves_where_the_bound_is_not_a_bare_day_ending_a_range_on_a_naive_timestamp(asked):
    sql, moved = _closed(asked)
    assert moved == 0
    assert sql == sqlglot.parse_one(asked, read="duckdb").sql(dialect="duckdb")


def test_a_symmetric_between_is_left_as_written():
    asked = "SELECT id FROM claim WHERE ts BETWEEN SYMMETRIC '2024-03-09' AND '2024-03-01'"
    assert _closed(asked, dialect="postgres")[1] == 0


def test_without_declared_types_nothing_moves():
    assert _closed("SELECT id FROM claim WHERE ts <= '2024-03-09'", types=None)[1] == 0


def test_an_oracle_date_holds_a_time_of_day_and_its_bound_moves_in_every_spelling():
    types = {"claim": {"id": "NUMBER", "ts": "DATE", "filed": "DATE", "tz": "TIMESTAMP WITH TIME ZONE",
                       "note": "VARCHAR2(20)", "region": "VARCHAR2(20)"}}
    for asked, closed in [
        ("SELECT id FROM claim WHERE ts <= DATE '2024-03-09'",
         "SELECT id FROM claim WHERE ts < TO_DATE('2024-03-10', 'YYYY-MM-DD')"),
        ("SELECT id FROM claim WHERE ts <= TO_DATE('2024-03-09', 'YYYY-MM-DD')",
         "SELECT id FROM claim WHERE ts < TO_DATE('2024-03-10', 'YYYY-MM-DD')"),
        ("SELECT id FROM claim WHERE ts <= '2024-03-09'",
         "SELECT id FROM claim WHERE ts < '2024-03-10'"),
    ]:
        assert _closed(asked, dialect="oracle", types=types) == (closed, 1), asked
    # A TO_DATE with a time of day says where it ends; one whose format swaps day and month reads
    # '2024-03-09' as 3 September, so the string's next day is not the date's.
    for asked in ("SELECT id FROM claim WHERE ts <= TO_DATE('2024-03-09 10:00', 'YYYY-MM-DD HH24:MI')",
                  "SELECT id FROM claim WHERE ts <= TO_DATE('2024-03-09', 'YYYY-DD-MM')"):
        assert _closed(asked, dialect="oracle", types=types)[1] == 0, asked
    # Elsewhere a DATE is whole days.
    assert _closed("SELECT id FROM claim WHERE ts <= '2024-03-09'", dialect="duckdb", types=types)[1] == 0


def test_the_moved_bound_counts_the_day_whole_on_the_engine():
    """The premise, measured: on DuckDB `<=` the bare day drops the day's later rows, and the moved
    bound keeps them and nothing past the day."""
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, ts TIMESTAMP)")
    con.execute("INSERT INTO claim VALUES (1, '2024-03-08 23:00'), (2, '2024-03-09 00:00'), "
                "(3, '2024-03-09 15:30'), (4, '2024-03-10 00:00')")
    asked = "SELECT id FROM claim WHERE ts <= '2024-03-09' ORDER BY id"
    closed, moved = _closed(asked)
    assert moved == 1
    assert [r[0] for r in con.execute(asked).fetchall()] == [1, 2]
    assert [r[0] for r in con.execute(closed).fetchall()] == [1, 2, 3]


def test_the_decider_approves_the_moved_bound_and_counts_it():
    verdict = decide("SELECT id FROM claim WHERE ts <= '2024-03-09'", VISIBLE, types=TYPES)
    assert isinstance(verdict, Approved), verdict
    assert verdict.day_bounds == 1
    # What runs, what the caller is shown and what the model calls read all carry the moved bound.
    for sql in (verdict.target_sql, verdict.asked_sql, verdict.plan_sql, verdict.model_sql()):
        assert "< '2024-03-10'" in sql and "<=" not in sql, sql
    untouched = decide("SELECT id FROM claim WHERE ts < '2024-03-10'", VISIBLE, types=TYPES)
    assert isinstance(untouched, Approved) and untouched.day_bounds == 0


def test_a_move_that_raises_leaves_the_statement_as_written_and_costs_no_decision(monkeypatch):
    """Half done and then raising, the move is discarded whole: the statement runs as written."""
    import mnemiq.sql.decide as decide_module

    def half_then_raise(tree, *a, **k):
        close_day_bounds(tree, *a, **k)
        raise RuntimeError("a shape the walk does not read")

    monkeypatch.setattr(decide_module, "close_day_bounds", half_then_raise)
    verdict = decide("SELECT id FROM claim WHERE ts <= '2024-03-09'", VISIBLE, types=TYPES)
    assert isinstance(verdict, Approved), verdict
    assert verdict.day_bounds is None
    assert "<= '2024-03-09'" in verdict.target_sql and "2024-03-10" not in verdict.target_sql


def test_a_row_policys_own_bound_is_never_moved():
    """The policy grants rows up to the 9th's midnight. Moving its bound would widen what the
    caller reads, so the move runs on the statement as asked, before the policy's predicate is
    added: the model's bound moves, the policy's stays as written."""
    from mnemiq.sql.policy import AccessPolicy

    policy = AccessPolicy(row_filters={"claim": "ts <= '2024-03-09'"})
    verdict = decide("SELECT id FROM claim WHERE ts <= '2024-03-20'", VISIBLE, dialect="duckdb",
                     target="duckdb", policy=policy, types=TYPES)
    assert isinstance(verdict, Approved), verdict
    assert verdict.day_bounds == 1
    assert "ts <= '2024-03-09'" in verdict.target_sql and "'2024-03-10'" not in verdict.target_sql
    assert "ts < '2024-03-21'" in verdict.target_sql
    assert "ts < '2024-03-21'" in verdict.asked_sql and "2024-03-09" not in verdict.asked_sql


# --- through the doors a caller uses -------------------------------------------------------------

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_CARD = "TABLE claim: id INTEGER, region VARCHAR (north|south|east|west), ts TIMESTAMP"
BOUND_SQL = "SELECT count(*) AS n FROM claim WHERE ts <= '2024-03-09'"


class _ApprovingJudgeClient:
    def __init__(self):
        self.calls: list[str] = []

    def complete(self, system, user, max_tokens=None, temperature=None, **_):
        self.calls.append(user)
        return '{"confidence": 0.95}'


@pytest.fixture
def source(tmp_path, monkeypatch):
    for var in ("MNEMIQ_VERIFY", "MNEMIQ_VERIFY_SANITY", "MNEMIQ_VERIFY_BASE_URL",
                "MNEMIQ_VERIFY_MODEL", "MNEMIQ_VERIFY_API_KEY", "MNEMIQ_VERIFY_GROUNDING",
                "MNEMIQ_VERIFY_THRESHOLD", "MNEMIQ_VERIFY_FAIL_CLOSED"):
        monkeypatch.delenv(var, raising=False)
    path = str(tmp_path / "syn.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim (id INTEGER, region VARCHAR, ts TIMESTAMP)")
    # Ten claims a day from the 1st to the 12th, each a quarter past an hour from 0 to 9. Through the
    # 9th is 90; `<=` the bare 9th stops at its midnight, before any of its ten, at 80.
    rows = [(d * 10 + h, ["north", "south", "east", "west"][h % 4], f"2024-03-{d:02d} {h:02d}:15:00")
            for d in range(1, 13) for h in range(10)]
    con.executemany("INSERT INTO claim VALUES (?, ?, ?)", rows)
    con.close()
    adapter = DuckDBAdapter.duckdb(path)
    assert adapter.execute("SELECT count(*) FROM claim WHERE ts <= '2024-03-09'")[0][0] == 80
    assert adapter.execute("SELECT count(*) FROM claim WHERE ts < '2024-03-10'")[0][0] == 90
    return adapter, enrich_structural(adapter, "syn")


def test_every_mode_answers_through_the_whole_day_and_says_it_moved_the_bound(source):
    adapter, snap = source
    judge = _ApprovingJudgeClient()
    verifiers, _ = _build_mode_verifiers(Settings(), client=judge)
    for name, mode in MODES.items():
        agent = build_agent(mode, generator=FakeGenerator([json.dumps({"sql": BOUND_SQL})] * 6),
                            synthesizer=FakeSynthesizer("SYNTHESIZED"), adapter=adapter,
                            cache=TwoTierCache(L1Cache()), corrector=None, values=None,
                            selector=None, verifier=verifiers[name])
        packet = ContextPacket(question="How many claims were filed as of 2024-03-09?",
                               cards=[RetrievedCard(object_id="claim", card=_CARD, score=1.0)],
                               grant_fingerprint=_GRANTS.fingerprint, enrichment_version="v1")
        ans = agent.answer(packet, snap, _GRANTS, _IDENTITY)
        assert ans.deferred is False and ans.failed is False, (name, ans.answer)
        payload = answer_payload(ans)
        assert payload["day_bounds_closed"] == 1, name
        assert "< '2024-03-10'" in payload["sql"] and "<=" not in payload["sql"], (name, payload["sql"])
        assert ans.preview.rows == [[90]], (name, ans.preview)
    # The judge read the moved bound, never the one the model wrote.
    assert judge.calls and all("< '2024-03-10'" in c and "<= '2024-03-09'" not in c for c in judge.calls)
