"""Under an access policy, a source's error reaches the repair prompt as its error class only.

Codex's review of #152: the judge, selector and synthesizer were moved off the policy's rewrite, and
the repair loop was not. The model repairs against the source's error text, and the source raised
that error running the REWRITTEN statement. Measured on DuckDB with
`id IN (SELECT id FROM entitlement WHERE who = 'secret_role_42')` on `claim`:

  * a conversion error in the caller's own WHERE echoed `... WHERE who = 'secret_role_42')) AS claim`
    into the generator's feedback;
  * a policy naming a column the source no longer has failed EXPLAIN, and the refusal fed back the
    whole predicate plus DuckDB's `Candidate bindings: "who"`.

#153 cut the echo and withheld text naming a word only the policy wrote. Codex's review of #153 broke
that: a policy that raises on data, its column wrapped in an expression (`CAST(UPPER(region) AS
INTEGER) > 0`), gets an error quoting the hidden row's value without naming any column. DuckDB also
prints values unquoted ("with value 31337"), and Oracle runs the caller's own predicate on rows the
policy removes and quotes a character of them (M160). No reading of the text tells data from the rest.
So under a policy the model reads the engine's error class, a fixed vocabulary, and nothing else of the
source's. With nothing narrowed, the source's words reach the model exactly as before, which keeps
every benchmark prompt byte-identical.
"""
import logging

import duckdb
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.generate.generator import FakeGenerator
from mnemiq.redact import policy_safe_detail
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

ENTITLED = "id IN (SELECT id FROM entitlement WHERE who = 'secret_role_42')"
# The same policy after a rename the snapshot has not caught up with: `whom` is in the snapshot, the
# source has `who`, so the policy's own predicate is what EXPLAIN refuses. The stale snapshot also
# holds `claim.status`, which the source lacks, so a statement can fail on its own too: a refusal of
# the policy's part alone goes to the operator, not the model (M158).
STALE = "id IN (SELECT id FROM entitlement WHERE whom = 'secret_role_42')"
# Fails at run time, in the caller's own WHERE -- past the policy's derived table in the statement,
# which is where DuckDB's echo window then sits.
BAD_CAST = "SELECT id FROM claim WHERE amount > 0 AND CAST(region AS INTEGER) > 0"
WITHHELD = " The rest of the source's message is not shown."


def _class_only(label):
    return f"The source rejected this query. Error class: {label}.{WITHHELD}"


def _source(tmp_path, claims):
    path = str(tmp_path / "s.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, amount INT);"
                "CREATE TABLE entitlement(id INT, who VARCHAR);"
                f"INSERT INTO claim VALUES {claims};"
                "INSERT INTO entitlement VALUES (1, 'secret_role_42')")
    con.close()
    return DuckDBAdapter.duckdb(path)


@pytest.fixture
def adapter(tmp_path):
    return _source(tmp_path, "(1, 'west', 10), (2, 'east', 20)")


def _snapshot(stale=False):
    names = [("claim", "id"), ("claim", "region"), ("claim", "amount"),
             ("entitlement", "id"), ("entitlement", "who")] + (
                 [("entitlement", "whom"), ("claim", "status")] if stale else [])
    return Snapshot(version="v1", source_id="s", created_at="2026-10-15T00:00:00Z",
                    columns=[Column(id=f"{t}.{c}", object_id=t, name=c) for t, c in names])


def _reply(sql):
    return f'{{"sql": "{sql}", "reason": "ok"}}'


GIVES_UP = '{"sql": null, "reason": "Region holds names, not numbers."}'


def _ask(adapter, *, sql=None, row_filter=None, replies=None, stale=False):
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": row_filter} if row_filter else {})
    generator = FakeGenerator(replies or [_reply(sql)] * 3)
    agent = Agent(generator=generator, synthesizer=FakeSynthesizer("x"), adapter=adapter,
                  cache=TwoTierCache(L1Cache()))
    packet = ContextPacket(question="which claims?", grant_fingerprint=grants.fingerprint,
                           enrichment_version="v1",
                           cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)])
    answer = agent.answer(packet, _snapshot(stale), grants,
                          IdentityContext(tenant_id="t", principal_id="u", roles=["r"]))
    return answer, generator.calls


def test_the_control_the_source_s_own_error_quotes_the_policy(adapter, caplog):
    # Without this, the tests below could pass on an error that never quoted the policy at all. The
    # operator's log keeps the whole text: it is the one place the full error is meant to be.
    with caplog.at_level(logging.WARNING):
        _ask(adapter, sql=BAD_CAST, row_filter=ENTITLED)
    logged = "\n".join(r.getMessage() for r in caplog.records if "the source said" in r.getMessage())
    # DuckDB's echo is a window around the error, here opening at `WHERE who =`.
    assert "who = 'secret_role_42'" in logged


def test_an_execution_error_reaches_the_model_as_its_class_only(adapter):
    _, calls = _ask(adapter, sql=BAD_CAST, row_filter=ENTITLED)
    assert calls[1] == _class_only("Conversion Error")


def test_an_explain_refusal_reaches_the_model_as_its_class_only(adapter):
    # `status` fails as asked too, so the refusal is the model's to repair; the source's words are
    # still about the rewrite, which names `whom`.
    _, calls = _ask(adapter, sql="SELECT status FROM claim", row_filter=STALE, stale=True)
    assert calls[1] == _class_only("Binder Error")


# Codex's two reproductions against #153 were raising conversions (`CAST(UPPER(region) AS INTEGER)`,
# `CAST(region || '' AS INTEGER)`), which M159 now refuses before they run
# (test_row_filter_raising_conversion.py). A raising expression that is not a conversion still
# runs, so the class is still all the model may read: `error()` quotes the hidden value whole.
@pytest.mark.parametrize("policy", ["CASE WHEN id = 2 THEN error(region) END IS NULL"])
def test_a_raising_policy_s_hidden_value_never_reaches_the_model(tmp_path, policy, caplog):
    # Row 1 passes the policy; row 2 is outside the caller's slice, and the policy raises on it
    # instead of removing it, its message naming no column, only the hidden row's value.
    source = _source(tmp_path, "(1, '7', 10), (2, 'secret_hidden_value', 20)")
    with caplog.at_level(logging.WARNING):
        _, calls = _ask(source, sql="SELECT id FROM claim", row_filter=policy)
    assert "secret_hidden_value" in caplog.text.lower(), "control: the source did quote it"
    assert calls[1] == _class_only("Invalid Input Error")


def test_with_nothing_narrowed_the_source_s_words_reach_the_model_whole(adapter):
    _, calls = _ask(adapter, sql=BAD_CAST)
    assert calls[1].startswith("The source rejected this query. Conversion Error: Could not convert")
    assert "LINE 1: SELECT id FROM claim WHERE amount > 0" in calls[1], "untouched without a policy"


def test_an_error_class_counts_as_the_source_s_words(adapter):
    # Shown even the class, the model's own reason does not go to the caller, as on any turn it was
    # shown the source's words: the class is fixed vocabulary, and the rule stays one rule.
    shown, _ = _ask(adapter, row_filter=ENTITLED, replies=[_reply(BAD_CAST), GIVES_UP])
    assert shown.deferred and shown.answer == "The model could not answer from these tables."


class _UnlabelledSource:
    """Approves at EXPLAIN, then fails at run time with text no pattern reads as a class."""

    def execute(self, sql):
        return []

    def execute_arrow(self, sql, timeout_s=None):
        raise RuntimeError("no such column: secret_hidden_value")


def test_a_model_shown_none_of_the_source_s_words_may_give_its_own_reason():
    # `Feedback.from_source` follows what the model was actually shown. With no class to read, the
    # model saw only our sentence, so its reason has nothing of the source's to carry out.
    answer, calls = _ask(_UnlabelledSource(), row_filter=ENTITLED,
                         replies=[_reply("SELECT id FROM claim"), GIVES_UP])
    assert calls[1] == "The source rejected this query."
    assert answer.deferred and answer.answer == "Region holds names, not numbers."


ASKED = "SELECT id FROM claim LIMIT 1000"
EXECUTED = ("SELECT id FROM (SELECT amount, id, region FROM claim WHERE id IN (SELECT id FROM "
            "entitlement WHERE who = 'secret_role_42')) AS claim LIMIT 1000")


@pytest.mark.parametrize("detail, dialect, label", [
    # DuckDB: `<type> Error:` leads every message, whatever follows -- quoted, unquoted, echoed.
    ("Conversion Error: Could not convert string 'SECRET_HIDDEN_VALUE' to INT32", "duckdb",
     "Conversion Error"),
    ("Conversion Error: Type INT32 with value 31337 can't be cast because the value is out of range "
     "for the destination type INT8 when casting from source column amount", "duckdb", "Conversion Error"),
    ("Out of Range Error: Overflow in multiplication of INT16 (31337 * 2)!", "duckdb", "Out of Range Error"),
    ('Binder Error: Referenced column "whom" not found in FROM clause!\nCandidate bindings: "who"\n\n'
     "LINE 1: ... WHERE whom = 'secret_role_42')) AS claim\n         ^", "duckdb", "Binder Error"),
    # Oracle 23 quotes one character of the value it could not convert (M160), and links help.
    ("ORA-01722: unable to convert string value containing 'h' to a number: REGION\n"
     "Help: https://docs.oracle.com/error-help/db/ora-01722/", "oracle", "ORA-01722"),
    ("DPY-4011: the database or network closed the connection", "oracle", "DPY-4011"),
])
def test_under_a_policy_only_the_error_class_is_kept(detail, dialect, label):
    assert (policy_safe_detail(detail, asked=ASKED, executed=EXECUTED, dialect=dialect)
            == f"Error class: {label}.{WITHHELD}")


@pytest.mark.parametrize("detail", [
    "no such column: secret_hidden_value",  # SQLite: no class at all
    "secret_hidden_value Error: x",  # not a class: DuckDB's never holds anything but letters
    "the source said ORA-01722: invalid number",  # a code is read only where the message begins
    "Conversion Error",  # no colon, so not DuckDB's `<type> Error: <message>` shape
])
def test_text_with_no_recognised_class_is_withheld_whole(detail):
    for dialect in ("duckdb", "oracle"):
        assert policy_safe_detail(detail, asked=ASKED, executed=EXECUTED, dialect=dialect) is None


@pytest.mark.parametrize("detail, dialect", [
    # A class is read in the shape its OWN engine writes, never another's: text from an engine whose
    # messages are not known to lead with a fixed class is withheld whole, whatever it looks like.
    ("Conversion Error: x", "oracle"), ("ORA-01722: x", "duckdb"),
    ("Conversion Error: x", "sqlite"), ("Conversion Error: x", "postgres"), ("ORA-01722: x", "mysql"),
])
def test_a_class_is_read_only_in_its_own_engine_s_shape(detail, dialect):
    assert policy_safe_detail(detail, asked=ASKED, executed=EXECUTED, dialect=dialect) is None


def test_with_nothing_narrowed_the_detail_is_returned_untouched():
    detail = "Binder Error: x\n\nLINE 1: SELECT id FROM claim\n               ^"
    assert policy_safe_detail(detail, asked=ASKED, executed=ASKED, dialect="duckdb") == detail


def test_an_unknown_asked_statement_counts_as_narrowed():
    # No record of what was asked means no way to say the policy changed nothing.
    assert (policy_safe_detail("Binder Error: x", asked=None, executed=EXECUTED, dialect="duckdb")
            == f"Error class: Binder Error.{WITHHELD}")


def test_no_detail_stays_none():
    assert policy_safe_detail(None, asked=ASKED, executed=EXECUTED, dialect="duckdb") is None


# M160's channel on DuckDB: whether an error fires at all says whether a removed row exists, if the
# caller's predicate ever runs on one. Oracle does (measured by the e2e lane); DuckDB keeps no promise
# not to, and this holds that it filters by the policy first. Every row but one is hidden by each
# policy below, and every hidden row would fail the caller's cast, so an engine upgrade that
# reorders the two turns this red rather than opening the channel quietly.
HIDING = ["region = 'west'", "region LIKE 'w%'", "length(region) = 4", "upper(region) = 'WEST'",
          "id = 1", "id % 20000 = 1", "region IN ('west', 'nowhere')", "starts_with(region, 'we')",
          # Subquery policies: the entitlements IN, which DuckDB plans as a semi-join, and a correlated
          # EXISTS in the unqualified spelling the policy validator admits (it refuses `claim.region`).
          ENTITLED, "EXISTS (SELECT 1 FROM entitlement e WHERE e.who = 'secret_role_42' AND region = 'west')"]
CASTS = ["SELECT id FROM claim WHERE CAST(amount AS INTEGER) > 0",
         "SELECT id FROM claim WHERE TRY_CAST(region AS INTEGER) IS NULL AND CAST(amount AS INTEGER) > 0",
         "SELECT id FROM claim WHERE CAST(amount AS INTEGER) > 0 AND id > 0"]


@pytest.fixture(scope="module")
def hiding_adapter(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("hiding") / "s.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, amount VARCHAR);"
                "CREATE TABLE entitlement(id INT, who VARCHAR);"
                "INSERT INTO claim SELECT i, CASE WHEN i = 1 THEN 'west' ELSE 'hidden_' || i END, "
                "CASE WHEN i = 1 THEN '10' ELSE 'hidden_' || i END FROM range(1, 20000) t(i);"
                "INSERT INTO entitlement VALUES (1, 'secret_role_42')")
    con.close()
    return DuckDBAdapter.duckdb(path)


@pytest.mark.parametrize("cast", CASTS)
@pytest.mark.parametrize("policy", HIDING)
def test_a_caller_s_predicate_never_raises_on_a_row_the_policy_removes(hiding_adapter, policy, cast):
    answer, calls = _ask(hiding_adapter, sql=cast, row_filter=policy)
    assert not answer.failed and not answer.deferred, (answer.answer, calls)
    assert calls == [None], "the one visible row casts cleanly, so nothing was ever repaired"
