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
# source has `who`, so the policy's own predicate is what EXPLAIN refuses.
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
             ("entitlement", "id"), ("entitlement", "who")] + ([("entitlement", "whom")] if stale else [])
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
    _, calls = _ask(adapter, sql="SELECT id FROM claim", row_filter=STALE, stale=True)
    assert calls[1] == _class_only("Binder Error")


@pytest.mark.parametrize("policy", ["CAST(UPPER(region) AS INTEGER) > 0",
                                    "CAST(region || '' AS INTEGER) > 0"])
def test_a_raising_policy_s_hidden_value_never_reaches_the_model(tmp_path, policy, caplog):
    # Codex's two reproductions against #153: the column is wrapped, so DuckDB's message names no
    # column, only the hidden row's value. Row 1 passes the policy; row 2 is outside the caller's
    # slice, and the policy raises on it instead of removing it.
    source = _source(tmp_path, "(1, '7', 10), (2, 'secret_hidden_value', 20)")
    with caplog.at_level(logging.WARNING):
        _, calls = _ask(source, sql="SELECT id FROM claim", row_filter=policy)
    assert "secret_hidden_value" in caplog.text.lower(), "control: the source did quote it"
    assert calls[1] == _class_only("Conversion Error")


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


@pytest.mark.parametrize("detail, label", [
    # DuckDB: `<type> Error:` leads every message, whatever follows -- quoted, unquoted, echoed.
    ("Conversion Error: Could not convert string 'SECRET_HIDDEN_VALUE' to INT32", "Conversion Error"),
    ("Conversion Error: Type INT32 with value 31337 can't be cast because the value is out of range "
     "for the destination type INT8 when casting from source column amount", "Conversion Error"),
    ("Out of Range Error: Overflow in multiplication of INT16 (31337 * 2)!", "Out of Range Error"),
    ('Binder Error: Referenced column "whom" not found in FROM clause!\nCandidate bindings: "who"\n\n'
     "LINE 1: ... WHERE whom = 'secret_role_42')) AS claim\n         ^", "Binder Error"),
    # Oracle 23 quotes one character of the value it could not convert (M160), and links help.
    ("ORA-01722: unable to convert string value containing 'h' to a number: REGION\n"
     "Help: https://docs.oracle.com/error-help/db/ora-01722/", "ORA-01722"),
    ("DPY-4011: the database or network closed the connection", "DPY-4011"),
])
def test_under_a_policy_only_the_error_class_is_kept(detail, label):
    assert (policy_safe_detail(detail, asked=ASKED, executed=EXECUTED)
            == f"Error class: {label}.{WITHHELD}")


@pytest.mark.parametrize("detail", [
    "no such column: secret_hidden_value",  # SQLite: no class at all
    "secret_hidden_value Error: x",  # not a class: DuckDB's never holds anything but letters
    "the source said ORA-01722: invalid number",  # a code is read only where the message begins
    "Conversion Error",  # no colon, so not DuckDB's `<type> Error: <message>` shape
])
def test_text_with_no_recognised_class_is_withheld_whole(detail):
    assert policy_safe_detail(detail, asked=ASKED, executed=EXECUTED) is None


def test_with_nothing_narrowed_the_detail_is_returned_untouched():
    detail = "Binder Error: x\n\nLINE 1: SELECT id FROM claim\n               ^"
    assert policy_safe_detail(detail, asked=ASKED, executed=ASKED) == detail


def test_an_unknown_asked_statement_counts_as_narrowed():
    # No record of what was asked means no way to say the policy changed nothing.
    assert (policy_safe_detail("Binder Error: x", asked=None, executed=EXECUTED)
            == f"Error class: Binder Error.{WITHHELD}")


def test_no_detail_stays_none():
    assert policy_safe_detail(None, asked=ASKED, executed=EXECUTED) is None
