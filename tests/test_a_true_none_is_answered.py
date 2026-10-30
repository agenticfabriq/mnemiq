"""M146: a question whose true answer is "none" is answered, through the doors a caller uses.

The no-rows rule declined every empty result, so a correct "none" was unanswerable in every mode,
and the judge never saw one. An empty result is answered now when every string value it filters
on is grounded in the value index; one resting on a value nothing could check is still declined,
and the decline names that value. Driven here through the real Agent with the product's own
per-mode verifiers (`_build_mode_verifiers`, `build_agent`) over a synthetic DuckDB source that the
real structural enrichment profiles and the real value index covers, and through `Runtime.ask` to
the payload `/v1/ask` serves. Only the model calls are scripted.
"""

from __future__ import annotations

import json

import duckdb
import pytest

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
from mnemiq.semantic.values import ValueIndex, build_value_index
from mnemiq.server.serialize import answer_payload
from mnemiq.store.bootstrap import init_store

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_CARD = ("TABLE claim: id INTEGER, region VARCHAR (north|south|east|west), status VARCHAR "
         "(open|closed), amount DOUBLE, adjuster VARCHAR, filed DATE")

# The true answer is "none": no claim is anywhere near a million (asserted below, not assumed).
NONE_SQL = "SELECT id, amount FROM claim WHERE status = 'open' AND amount > 1000000"
NONE_NOTE = "No rows matched: status = 'open' AND amount > 1000000."
# A misspelled value on a column with 300 distinct values, past the index's 200: nothing checks it
# before the run, so it returns nothing, and "none" would be a wrong answer. Asked once the result
# is empty, the source says the column does not hold it.
MISSPELLED_SQL = "SELECT id, amount FROM claim WHERE adjuster = 'agnet_0042'"
MISSPELLED_DECLINE = "No rows matched, and 'agnet_0042' does not appear in claim.adjuster"
# The same column, a value it holds, and a total no claim reaches: a true "none" the index cannot
# ground and the source can.
UNINDEXED_NONE_SQL = "SELECT id, amount FROM claim WHERE adjuster = 'agent_0042' AND amount > 1000000"


class _CountingJudgeClient:
    """The endpoint behind the real `SemanticJudge`: approves, and records what it was shown."""

    def __init__(self, reply: str = '{"confidence": 0.95}'):
        self.reply = reply
        self.calls: list[str] = []

    def complete(self, system, user, max_tokens=None, temperature=None, **_):
        self.calls.append(user)
        return self.reply


@pytest.fixture
def source(tmp_path, monkeypatch):
    for var in ("MNEMIQ_VERIFY", "MNEMIQ_VERIFY_SANITY", "MNEMIQ_VERIFY_BASE_URL",
                "MNEMIQ_VERIFY_MODEL", "MNEMIQ_VERIFY_API_KEY", "MNEMIQ_VERIFY_GROUNDING",
                "MNEMIQ_VERIFY_THRESHOLD", "MNEMIQ_VERIFY_FAIL_CLOSED"):
        monkeypatch.delenv(var, raising=False)
    path = str(tmp_path / "syn.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim (id INTEGER, region VARCHAR, status VARCHAR, amount DOUBLE, "
                "adjuster VARCHAR, filed DATE, ts TIMESTAMP, tags VARCHAR[], tz TIMESTAMPTZ, "
                "j JSON)")
    regions = ["north", "south", "east", "west"]
    rows = [(i, regions[i % 4], "open" if i % 3 else "closed", float(100 + (i * 37) % 4900),
             f"agent_{i % 300:04d}", f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}",
             f"2024-{1 + i % 12:02d}-{1 + i % 28:02d} {8 + i % 9:02d}:{i % 60:02d}:00",
             ["open", "urgent"] if i % 2 else ["open"],
             f"2024-{1 + i % 12:02d}-{1 + i % 28:02d} {8 + i % 9:02d}:{i % 60:02d}:00+00",
             "[42]")
            for i in range(1200)]
    con.executemany("INSERT INTO claim VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()
    adapter = DuckDBAdapter.duckdb(path)
    snap = enrich_structural(adapter, "syn")
    store = init_store(str(tmp_path / "store.duckdb"))
    build_value_index(adapter, snap, store)
    values = ValueIndex(store)
    # The premises, measured rather than assumed.
    assert values.has("claim", "status") and values.has("claim", "region")
    assert not values.has("claim", "adjuster")
    assert adapter.execute("SELECT max(amount) FROM claim")[0][0] < 1_000_000
    assert adapter.execute("SELECT count(*) FROM claim WHERE adjuster = 'agent_0042'")[0][0] > 0
    return adapter, snap, values


def _agents(adapter, values, sql: str, judge_client, settings: Settings | None = None):
    verifiers, _ = _build_mode_verifiers(settings or Settings(), client=judge_client)
    reply = json.dumps({"sql": sql})
    return {name: build_agent(mode, generator=FakeGenerator([reply] * 6),
                              synthesizer=FakeSynthesizer("SYNTHESIZED"), adapter=adapter,
                              cache=TwoTierCache(L1Cache()), corrector=None, values=values,
                              selector=None, verifier=verifiers[name])
            for name, mode in MODES.items()}


def _packet(question: str, grants: GrantSet = _GRANTS) -> ContextPacket:
    return ContextPacket(question=question,
                         cards=[RetrievedCard(object_id="claim", card=_CARD, score=1.0)],
                         grant_fingerprint=grants.fingerprint, enrichment_version="v1")


def test_a_true_none_is_answered_in_every_mode_and_the_judge_reads_it_where_one_is_wired(source):
    adapter, snap, values = source
    judge = _CountingJudgeClient()
    agents = _agents(adapter, values, NONE_SQL, judge)
    for name in ("instant", "thinking", "deep"):
        before = len(judge.calls)
        ans = agents[name].answer(_packet("Which open claims are over 1,000,000?"), snap,
                                  _GRANTS, _IDENTITY)
        assert ans.deferred is False and ans.failed is False, (name, ans.answer)
        assert ans.preview is not None and ans.preview.row_count == 0, name
        # The note is the engine's, appended after the synthesizer and never through it.
        assert ans.answer.startswith("SYNTHESIZED"), name
        assert NONE_NOTE in ans.answer, (name, ans.answer)
        judged = len(judge.calls) - before
        if name == "instant":
            assert judged == 0 and ans.verify_layer == "pass"  # the grounding is the whole decision
        else:
            # A grounded empty result goes on to the judge, which reads the zero rows.
            assert judged == 1 and ans.verify_layer == "judge", name
            assert "(0 rows)" in judge.calls[-1]


def test_a_value_the_source_does_not_hold_keeps_the_empty_result_declined_and_is_named(source):
    adapter, snap, values = source
    judge = _CountingJudgeClient()
    agents = _agents(adapter, values, MISSPELLED_SQL, judge)
    for name in ("instant", "thinking", "deep"):
        ans = agents[name].answer(_packet("List the claims handled by adjuster agent_0042."), snap,
                                  _GRANTS, _IDENTITY)
        assert ans.deferred is True and ans.failed is False, (name, ans.answer)
        assert ans.verify_layer == "sanity" and str(ans.reason_code) == "verification", name
        assert ans.answer.startswith(MISSPELLED_DECLINE), (name, ans.answer)
    assert judge.calls == []  # declined before the judge: no model call spent on it



class _Recording:
    """The adapter, recording each statement it runs, and failing the probes when told to."""

    def __init__(self, inner, fail_probes: bool = False):
        self.inner, self.fail_probes, self.statements = inner, fail_probes, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def execute_arrow(self, sql, timeout_s=None):
        self.statements.append(sql)
        if self.fail_probes and _is_probe(sql):
            raise TimeoutError("the probe ran past its bound")
        return self.inner.execute_arrow(sql, timeout_s=timeout_s)

    def probes(self):
        return [s for s in self.statements if _is_probe(s)]


def _is_probe(sql: str) -> bool:
    return sql.startswith('SELECT 1 FROM "claim" WHERE "adjuster" = ')


def test_a_true_none_on_a_column_the_index_cannot_hold_is_answered_in_every_mode(source):
    """The index holds no value of a 300-value column, so the source is asked: it holds the value,
    the empty result rests on it, and the "none" is answered, the source asked once a turn."""
    adapter, snap, values = source
    assert adapter.execute(UNINDEXED_NONE_SQL.replace("SELECT id, amount", "SELECT count(*)"))[0][0] == 0
    recording = _Recording(adapter)
    agents = _agents(recording, values, UNINDEXED_NONE_SQL, _CountingJudgeClient())
    for name in ("instant", "thinking", "deep"):
        before = len(recording.probes())
        ans = agents[name].answer(_packet("Which claims of adjuster agent_0042 are over 1,000,000?"),
                                  snap, _GRANTS, _IDENTITY)
        assert ans.deferred is False and ans.failed is False, (name, ans.answer)
        assert "No rows matched: adjuster = 'agent_0042' AND amount > 1000000." in ans.answer, name
        assert recording.probes()[before:] == [
            'SELECT 1 FROM "claim" WHERE "adjuster" = \'agent_0042\' LIMIT 1'], name


def test_a_probe_the_source_does_not_answer_keeps_the_empty_result_declined(source):
    adapter, snap, values = source
    recording = _Recording(adapter, fail_probes=True)
    ans = _ask_instant(recording, snap, values, UNINDEXED_NONE_SQL, "q")
    assert ans.deferred is True and ans.verify_layer == "sanity", ans.answer
    assert ans.answer.startswith("No rows matched, and 'agent_0042' could not be checked against "
                                 "claim.adjuster's known values"), ans.answer
    assert len(recording.probes()) == 1



def test_the_turns_probes_stop_at_their_budget(source, monkeypatch):
    import mnemiq.agent.loop as loop_mod

    adapter, snap, values = source
    monkeypatch.setattr(loop_mod, "PROBE_BUDGET_S", 0.0)
    recording = _Recording(adapter)
    ans = _ask_instant(recording, snap, values, UNINDEXED_NONE_SQL, "q")
    assert ans.deferred is True and "could not be checked against claim.adjuster" in ans.answer
    assert recording.probes() == []

def test_a_result_with_rows_asks_the_source_nothing(source):
    adapter, snap, values = source
    recording = _Recording(adapter)
    ans = _ask_instant(recording, snap, values,
                       "SELECT id, amount FROM claim WHERE adjuster = 'agent_0042'", "q")
    assert ans.deferred is False and ans.preview.row_count > 0, ans.answer
    assert recording.probes() == []


def test_a_row_filtered_callers_value_is_never_asked_of_the_source(source):
    """Whether a value exists in rows the caller cannot read is what the row filter withholds."""
    adapter, snap, values = source
    recording = _Recording(adapter)
    ans = _ask_instant(recording, snap, values, UNINDEXED_NONE_SQL, "q", grants=_NORTH_ONLY)
    assert ans.deferred is True and "'agent_0042'" in ans.answer, ans.answer
    assert recording.probes() == []


def test_a_none_asked_as_a_total_is_answered_as_one(source):
    """`SUM` over no rows is one NULL, which the null-scalar sibling rule declined (M146)."""
    adapter, snap, values = source
    sql = "SELECT sum(amount) AS total FROM claim WHERE status = 'open' AND amount > 1000000"
    agents = _agents(adapter, values, sql, _CountingJudgeClient())
    ans = agents["instant"].answer(_packet("What is the total of open claims over 1,000,000?"),
                                   snap, _GRANTS, _IDENTITY)
    assert ans.deferred is False, ans.answer
    assert ("The result holds no value: no rows matched (status = 'open' AND amount > 1000000), "
            "or every row that matched holds none.") in ans.answer


def test_a_misspelled_total_stays_declined(source):
    adapter, snap, values = source
    sql = "SELECT sum(amount) AS total FROM claim WHERE adjuster = 'agnet_0042'"
    agents = _agents(adapter, values, sql, _CountingJudgeClient())
    ans = agents["instant"].answer(_packet("What is the total for adjuster agent_0042?"),
                                   snap, _GRANTS, _IDENTITY)
    assert ans.deferred is True and ans.verify_layer == "sanity"
    assert ans.answer.startswith("The query produced an empty value, and 'agnet_0042' does not "
                                 "appear in claim.adjuster"), ans.answer


def test_a_date_range_needs_no_grounding(source):
    """A well-formed date against a DATE column is a bound, as a number is: no spelling to check."""
    adapter, snap, values = source
    sql = "SELECT id FROM claim WHERE region = 'west' AND filed >= '2031-01-01'"
    agents = _agents(adapter, values, sql, _CountingJudgeClient())
    ans = agents["instant"].answer(_packet("Which west claims were filed in 2031 or later?"),
                                   snap, _GRANTS, _IDENTITY)
    assert ans.deferred is False, ans.answer
    assert "No rows matched: region = 'west' AND filed >= '2031-01-01'." in ans.answer


_NORTH_ONLY = GrantSet(frozenset({"claim"}), row_filters={"claim": "region = 'north'"})


def _ask_instant(adapter, snap, values, sql, question, grants=_GRANTS):
    agents = _agents(adapter, values, sql, _CountingJudgeClient())
    return agents["instant"].answer(_packet(question, grants), snap, grants, _IDENTITY)


def test_a_row_filtered_callers_empty_result_is_declined_naming_the_value(source):
    """The value index is built without the row filter (M5), so on a row-filtered table it cannot
    say whether a value is in the rows this caller may read; answering on it would answer for rows
    the caller cannot see. The cost, accepted: a true "none" there is declined, naming the value."""
    adapter, snap, values = source
    ans = _ask_instant(adapter, snap, values, "SELECT id, amount FROM claim WHERE region = 'south'",
                       "List the south-region claims.", _NORTH_ONLY)
    assert ans.deferred is True and ans.verify_layer == "sanity", ans.answer
    assert ans.answer.startswith("No rows matched, and 'south' could not be checked"), ans.answer
    assert ans.narrowed and ans.narrowed[0].rows is True
    assert "north" not in ans.answer  # never the policy's own predicate


@pytest.mark.parametrize("template", [
    "SELECT id FROM claim WHERE region <> 'north' AND region <> '{v}'",
    "SELECT id FROM claim WHERE region <> 'north' AND region IS DISTINCT FROM '{v}'",
])
def test_on_a_row_filtered_table_a_hidden_value_and_an_absent_guess_get_one_answer(source,
                                                                                   template):
    """Codex's review of ui-36: with the unfiltered index, a value present only in rows the filter
    hides read GROUNDED and the empty result was answered, while an absent guess was declined with
    "does not appear" -- a membership test on hidden rows that main never had, since the value
    check before the run reads no `<>`."""
    adapter, snap, values = source
    hidden = _ask_instant(adapter, snap, values, template.format(v="south"), "q", _NORTH_ONLY)
    guess = _ask_instant(adapter, snap, values, template.format(v="sooth"), "q", _NORTH_ONLY)
    assert hidden.deferred is True and guess.deferred is True, (hidden.answer, guess.answer)
    assert hidden.reason_code == guess.reason_code
    assert hidden.answer.replace("south", "?") == guess.answer.replace("sooth", "?")
    assert "does not appear" not in guess.answer


@pytest.mark.parametrize("sql", [
    "SELECT id FROM claim WHERE region = 'south'",
    "SELECT id FROM claim WHERE region IN ('east', 'south')",
    "SELECT id FROM claim WHERE region NOT IN ('north', 'south')",
])
def test_on_a_row_filtered_table_a_hidden_value_is_never_answered(source, sql):
    """For `=`, IN and NOT IN the absent guess never reaches the run: the value check refuses it
    before, as main does (M5 keeps that refusal and withholds the real values). The hidden value
    runs, comes back empty, and is declined like any value nothing could check."""
    adapter, snap, values = source
    ans = _ask_instant(adapter, snap, values, sql, "q", _NORTH_ONLY)
    assert ans.deferred is True and ans.verify_layer == "sanity", ans.answer
    assert "'south' could not be checked against claim.region's known values" in ans.answer


def test_a_misspelled_value_inside_a_set_returning_projection_is_declined(source):
    """Measured: `unnest(list_filter(tags, x -> x = 'opne'))` returns no rows where 'open' returns
    every claim. The projection removes rows, so its value is a filter's."""
    adapter, snap, values = source
    assert adapter.execute("SELECT count(*) FROM (SELECT unnest(list_filter(tags, x -> x = 'open')) "
                           "FROM claim) q")[0][0] == 1200
    ans = _ask_instant(adapter, snap, values,
                       "SELECT unnest(list_filter(tags, x -> x = 'opne')) AS t FROM claim",
                       "Which claims are tagged open?")
    assert ans.deferred is True and "'opne'" in ans.answer, ans.answer


def test_a_misspelled_value_in_a_projection_the_where_clause_reads_is_declined(source):
    """DuckDB reads a select alias in WHERE: the projection is the filter."""
    adapter, snap, values = source
    sql = "SELECT id, upper(status) = 'OPNE' AS flag FROM claim WHERE flag"
    assert adapter.execute(sql.replace("OPNE", "OPEN").replace("SELECT id,", "SELECT count(*),")
                           .replace("WHERE flag", "WHERE flag GROUP BY flag"))[0][0] > 0
    ans = _ask_instant(adapter, snap, values, sql, "Which claims are open?")
    assert ans.deferred is True and "'OPNE'" in ans.answer, ans.answer


def test_a_date_only_string_against_a_timestamp_is_declined(source):
    """Measured: `ts = '2024-03-03'` is midnight, so it returns no rows while claims fall on that
    day. A date-only string is a bound against a DATE column alone."""
    adapter, snap, values = source
    assert adapter.execute("SELECT count(*) FROM claim WHERE CAST(ts AS DATE) = DATE '2024-03-03'"
                           )[0][0] > 0
    ans = _ask_instant(adapter, snap, values, "SELECT id FROM claim WHERE ts = '2024-03-03'",
                       "Which claims were stamped on 3 March 2024?")
    assert ans.deferred is True and "'2024-03-03'" in ans.answer, ans.answer


def test_a_half_open_day_on_a_timestamp_is_answered(source):
    """`ts >= day AND ts < next day` counts the day whole, so its "none" is answered; the same day
    ended with `<=` is declined."""
    adapter, snap, values = source
    half_open = "SELECT id FROM claim WHERE region = 'west' AND ts >= '2031-01-01' AND ts < '2031-01-02'"
    assert adapter.execute(half_open.replace("SELECT id", "SELECT count(*)"))[0][0] == 0
    ans = _ask_instant(adapter, snap, values, half_open, "Which west claims were stamped on 1 January 2031?")
    assert ans.deferred is False, ans.answer
    assert "No rows matched: region = 'west' AND ts >= '2031-01-01' AND ts < '2031-01-02'." in ans.answer
    inclusive = half_open.replace("ts < '2031-01-02'", "ts <= '2031-01-01'")
    ans = _ask_instant(adapter, snap, values, inclusive, "Which west claims were stamped on 1 January 2031?")
    assert ans.deferred is True and "'2031-01-01'" in ans.answer, ans.answer


# Fix round 2: the reviews of fix round 1 found each exemption bypassed one level deeper -- a
# projected value carried into WHERE, QUALIFY or a set-returning call through another item's alias,
# and a DATE literal compared with an expression or a column pair of timestamp type. Each returns
# no rows while its intended spelling returns some, and each was answered; on main each declined.
_ROUND_2 = [
    ("SELECT id, upper(status) = 'OPNE' AS f, f AS g FROM claim WHERE g", "'OPNE'",
     "SELECT count(*) FROM claim WHERE upper(status) = 'OPEN'"),
    ("SELECT id, adjuster = 'agnet_0042' AS f, f AS g FROM claim WHERE g", "'agnet_0042'",
     "SELECT count(*) FROM claim WHERE adjuster = 'agent_0042'"),
    ("SELECT list_filter(tags, x -> x = 'opne') AS l, unnest(l) AS t FROM claim", "'opne'",
     "SELECT count(*) FROM (SELECT unnest(list_filter(tags, x -> x = 'open')) FROM claim) q"),
    ("SELECT id, adjuster = 'agnet_0042' AS f, f AS g, row_number() OVER () AS rn FROM claim "
     "QUALIFY g", "'agnet_0042'", "SELECT count(*) FROM claim WHERE adjuster = 'agent_0042'"),
    ("SELECT adjuster = 'agnet_0042' AS x, unnest(CASE WHEN x THEN [1] ELSE [] END) AS u "
     "FROM claim", "'agnet_0042'", "SELECT count(*) FROM claim WHERE adjuster = 'agent_0042'"),
    ("SELECT id FROM claim WHERE CAST(ts AS TIMESTAMP) = DATE '2024-03-03'", "'2024-03-03'",
     "SELECT count(*) FROM claim WHERE CAST(ts AS DATE) = DATE '2024-03-03'"),
    ("SELECT id FROM claim WHERE greatest(ts, ts) = DATE '2024-03-03'", "'2024-03-03'",
     "SELECT count(*) FROM claim WHERE CAST(ts AS DATE) = DATE '2024-03-03'"),
    ("SELECT id FROM claim WHERE DATE '2024-03-03' BETWEEN ts AND ts", "'2024-03-03'",
     "SELECT count(*) FROM claim WHERE CAST(ts AS DATE) = DATE '2024-03-03'"),
]


@pytest.mark.parametrize("sql,named,intended", _ROUND_2, ids=[f"r2-{i}" for i in range(len(_ROUND_2))])
def test_a_value_reaching_the_rows_one_level_deeper_keeps_the_empty_result_declined(
        source, sql, named, intended):
    adapter, snap, values = source
    assert adapter.execute(intended)[0][0] > 0  # the intended spelling has rows
    ans = _ask_instant(adapter, snap, values, sql, "q")
    assert ans.deferred is True and ans.verify_layer == "sanity", (sql, ans.answer)
    assert named in ans.answer, ans.answer


def test_a_week_written_to_the_second_on_a_timestamp_column_is_answered(source):
    """A full timestamp against a bare TIMESTAMP column is a bound of the column's own precision,
    as safe as a number (the controller's addition to fix round 2)."""
    adapter, snap, values = source
    ans = _ask_instant(adapter, snap, values,
                       "SELECT id FROM claim WHERE ts >= '2031-01-01 00:00:00' "
                       "AND ts < '2031-01-08 00:00:00'", "Which claims were stamped in 2031's first week?")
    assert ans.deferred is False, ans.answer
    assert ("No rows matched: ts >= '2031-01-01 00:00:00' AND ts < '2031-01-08 00:00:00'."
            in ans.answer)


# Fix round 3: Codex's pass on fix round 2, each reproduced through decide, DuckDB and sanity.
# A string written other than as a plain literal (E'', N'', $$...$$) was invisible to a scan that
# read only plain literals, so a misspelled value slipped past both checks with nothing to name;
# a naive timestamp against a TIMESTAMPTZ column reads in the session's zone; and a full timestamp
# compared for equality is a disguised day filter. Each was answered.
_ROUND_3 = [
    ("SELECT id FROM claim WHERE status = E'opne'", "opne"),
    ("SELECT id FROM claim WHERE status = N'opne'", "opne"),
    ("SELECT id FROM claim WHERE status = $$opne$$", "opne"),
    ("SELECT id FROM claim WHERE tz >= '2031-01-01 00:00:00' AND tz < '2031-01-08 00:00:00'",
     "'2031-01-01 00:00:00'"),
    ("SELECT id FROM claim WHERE ts = '2024-03-03 08:02:00'", "'2024-03-03 08:02:00'"),
    ("SELECT id FROM claim WHERE ts IN ('2024-03-03 08:02:00', '2024-03-04 08:02:00')",
     "'2024-03-03 08:02:00'"),
]


@pytest.mark.parametrize("sql,named", _ROUND_3, ids=[f"r3-{i}" for i in range(len(_ROUND_3))])
def test_a_value_the_scan_cannot_read_as_a_bound_keeps_the_empty_result_declined(source, sql,
                                                                                 named):
    adapter, snap, values = source
    assert adapter.execute("SELECT count(*) FROM claim WHERE status = 'open'")[0][0] > 0
    # The day the equality cases name has rows, at other times of day.
    assert adapter.execute("SELECT count(*) FROM claim WHERE CAST(ts AS DATE) = DATE '2024-03-03'"
                           )[0][0] > 0
    assert adapter.execute(f"SELECT count(*) FROM ({sql}) q")[0][0] == 0  # and the result is empty
    ans = _ask_instant(adapter, snap, values, sql, "q")
    assert ans.deferred is True and ans.verify_layer == "sanity", (sql, ans.answer)
    assert named in ans.answer, ans.answer


# Fix round 4: a JSON path is a constant too, whatever its leaves. Codex's pass on fix round 3:
# `(j->'$[9]') IS NOT NULL` over `[42]` was approved with no value to check, ran empty and passed,
# since a root, an integer subscript and a wildcard carry no text the constant scan read.
#
# A union path (`'$[1,2]'`) never reaches the run on main or here: the decider's niladic-name guard
# raises printing it (sqlglot 30.12.0 has no DuckDB rendering for JSONPathUnion), a pre-existing
# fault outside M146, so its classification is pinned in the unit file.
@pytest.mark.parametrize("sql", [
    "SELECT id FROM claim WHERE (j->'$[9]') IS NOT NULL",
    "SELECT id FROM claim WHERE (j->'$') IS NULL",
    "SELECT id FROM claim WHERE (j->'$[*]') IS NULL",
    "SELECT id FROM claim WHERE (j->'$[1:3]') IS NULL",
], ids=["index", "root", "wildcard", "slice"])
def test_an_empty_result_resting_on_a_json_path_is_declined(source, sql):
    from mnemiq.sql.decide import decide
    from mnemiq.sql.schema import visible_schema

    adapter, snap, values = source
    # The statement the engine runs (sqlglot reprints the path) is approved, runs, and is empty.
    executed = decide(sql, visible_schema(snap, _GRANTS), target="duckdb").target_sql
    assert adapter.execute(f"SELECT count(*) FROM ({executed}) q")[0][0] == 0
    ans = _ask_instant(adapter, snap, values, sql, "q")
    assert ans.deferred is True and ans.verify_layer == "sanity", ans.answer
    assert "a JSON path" in ans.answer, ans.answer


def test_an_offset_timestamp_range_on_a_timestamptz_column_is_answered(source):
    adapter, snap, values = source
    ans = _ask_instant(adapter, snap, values,
                       "SELECT id FROM claim WHERE tz >= '2031-01-01 00:00:00+00' "
                       "AND tz < '2031-01-08 00:00:00+00'", "q")
    assert ans.deferred is False, ans.answer


def test_a_true_none_with_a_string_constant_in_its_projection_is_declined(source):
    """The cost of the allow-list without an output exemption, accepted (fix round 2): a string the
    projection writes is checked like any other, and a constant matches no column."""
    adapter, snap, values = source
    ans = _ask_instant(adapter, snap, values,
                       "SELECT id, 'big claim' AS kind FROM claim WHERE status = 'open' "
                       "AND amount > 1000000", "Which open claims are over 1,000,000?")
    assert ans.deferred is True and "'big claim'" in ans.answer, ans.answer
    grounded = _ask_instant(adapter, snap, values,
                            "SELECT id, CASE WHEN status = 'open' THEN 1 END AS o FROM claim "
                            "WHERE amount > 1000000", "Which claims are over 1,000,000?")
    assert grounded.deferred is False, grounded.answer  # a grounded projected value is fine


def test_with_the_layer_off_instant_runs_no_verifier_and_says_none_ran(source):
    """`MNEMIQ_VERIFY_SANITY=0` leaves `instant` with no layer at all, and a verifier with nothing
    to run stamped `verified: "basic"`, which the serializer defines as the deterministic net."""
    adapter, snap, values = source
    agents = _agents(adapter, values, NONE_SQL, _CountingJudgeClient(),
                     Settings(verify_sanity=False))
    assert agents["instant"].verifier is None
    payload = answer_payload(agents["instant"].answer(
        _packet("Which open claims are over 1,000,000?"), snap, _GRANTS, _IDENTITY))
    assert payload["deferred"] is False and payload["verified"] is None


def test_the_deep_vote_rests_an_empty_cluster_on_a_grounded_member(source):
    """Empty results with the same columns cluster together, so a cluster can hold a candidate
    resting on a misspelled value beside ones resting on grounded values; which one represented it
    decided the answer by candidate order."""
    adapter, snap, values = source
    verifiers, _ = _build_mode_verifiers(Settings(), client=_CountingJudgeClient())
    unchecked = json.dumps({"sql": MISSPELLED_SQL})
    grounded = json.dumps({"sql": NONE_SQL})

    def deep(replies):
        return build_agent(MODES["deep"], generator=FakeGenerator(replies),
                           synthesizer=FakeSynthesizer("SYNTHESIZED"), adapter=adapter,
                           cache=TwoTierCache(L1Cache()), corrector=None, values=values,
                           selector=None, verifier=verifiers["deep"])

    ans = deep([unchecked] + [grounded] * 4).answer(_packet("q"), snap, _GRANTS, _IDENTITY)
    assert ans.deferred is False and NONE_NOTE in ans.answer, ans.answer
    ans = deep([unchecked] * 5).answer(_packet("q"), snap, _GRANTS, _IDENTITY)
    assert ans.deferred is True and ans.answer.startswith(MISSPELLED_DECLINE), ans.answer


# --- through Runtime.ask, to the payload /v1/ask serves -------------------------------------------

def _runtime(monkeypatch, adapter, snap, values, sql: str, grants: GrantSet = _GRANTS):
    import mnemiq.runtime as rt_mod
    from mnemiq.runtime import Runtime

    class _Authz:
        def grants_for(self, _identity):
            return grants

    def _retrieve(con, question, *a, **k):
        return _packet(question, grants)

    monkeypatch.setattr(rt_mod, "retrieve", _retrieve)
    agents = _agents(adapter, values, sql, _CountingJudgeClient())
    return Runtime(con=None, snapshot=snap, adapter=adapter, agent=agents["thinking"],
                   embedder=None, authz=_Authz(), settings=None, agents=agents)


def test_runtime_ask_serves_a_true_none_as_an_answer(source, monkeypatch):
    adapter, snap, values = source
    rt = _runtime(monkeypatch, adapter, snap, values, NONE_SQL)
    for mode in ("instant", "thinking"):
        payload = answer_payload(rt.ask("Which open claims are over 1,000,000?", _IDENTITY,
                                        mode=mode))
        assert payload["deferred"] is False and payload["failed"] is False, payload["answer"]
        assert payload["preview"]["row_count"] == 0 and payload["preview"]["rows"] == []
        assert NONE_NOTE in payload["answer"]
        assert payload["verified"] == ("basic" if mode == "instant" else "judged")


def test_runtime_ask_declines_a_misspelled_none_and_names_the_value(source, monkeypatch):
    adapter, snap, values = source
    rt = _runtime(monkeypatch, adapter, snap, values, MISSPELLED_SQL)
    payload = answer_payload(rt.ask("List the claims handled by adjuster agent_0042.", _IDENTITY,
                                    mode="instant"))
    assert payload["deferred"] is True and payload["reason_code"] == "verification"
    assert payload["verified"] == "basic" and payload["preview"] is None
    assert payload["answer"].startswith(MISSPELLED_DECLINE)


def test_runtime_ask_declines_a_row_filtered_empty_result_naming_the_value(source, monkeypatch):
    adapter, snap, values = source
    rt = _runtime(monkeypatch, adapter, snap, values,
                  "SELECT id, amount FROM claim WHERE region = 'south'", _NORTH_ONLY)
    payload = answer_payload(rt.ask("List the south-region claims.", _IDENTITY, mode="instant"))
    assert payload["deferred"] is True and payload["reason_code"] == "verification"
    assert payload["answer"].startswith("No rows matched, and 'south' could not be checked")
    assert "north" not in payload["answer"]
