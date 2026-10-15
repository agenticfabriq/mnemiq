"""A source's error reaches the repair prompt without the access policy's own words in it.

Codex's review of #152: the judge, selector and synthesizer were moved off the policy's rewrite, and
the repair loop was not. The model repairs against the source's error text, and the source raised
that error running the REWRITTEN statement, so its echo of the statement can quote the policy.
Measured on DuckDB with `id IN (SELECT id FROM entitlement WHERE who = 'secret_role_42')` on `claim`:

  * a conversion error in the caller's own WHERE, raised at run time, echoed
    `... WHERE who = 'secret_role_42')) AS claim WHERE ...` into the generator's feedback;
  * a policy naming a column the source no longer has failed EXPLAIN, and the refusal fed back the
    whole predicate plus DuckDB's `Candidate bindings: "who"`.

Neither table nor literal is the caller's to read, and both were sent to the model server. With a
policy narrowing the statement, the echo is cut, and what remains is withheld if it still names a
word only the policy wrote. With nothing narrowed, the source's words reach the model exactly as
before, which keeps every benchmark prompt byte-identical.
"""
import logging
import re

import duckdb
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.generate.generator import FakeGenerator
from mnemiq.redact import withhold_policy_words
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

ENTITLED = "id IN (SELECT id FROM entitlement WHERE who = 'secret_role_42')"
# The same policy after a rename the snapshot has not caught up with: `whom` is in the snapshot, the
# source has `who`, so the policy's own predicate is what EXPLAIN refuses.
STALE = "id IN (SELECT id FROM entitlement WHERE whom = 'secret_role_42')"
# Fails at run time, in the caller's own WHERE -- past the policy's derived table in the statement,
# which is where DuckDB's echo window then sits.
BAD_CAST = "SELECT id FROM claim WHERE amount > 0 AND CAST(region AS INTEGER) > 0"
POLICY_WORDS = ("entitlement", "secret_role_42", "who", "whom")


@pytest.fixture
def adapter(tmp_path):
    path = str(tmp_path / "s.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim(id INT, region VARCHAR, amount INT);"
                "CREATE TABLE entitlement(id INT, who VARCHAR);"
                "INSERT INTO claim VALUES (1, 'west', 10), (2, 'east', 20);"
                "INSERT INTO entitlement VALUES (1, 'secret_role_42')")
    con.close()
    return DuckDBAdapter.duckdb(path)


def _snapshot(stale=False):
    names = [("claim", "id"), ("claim", "region"), ("claim", "amount"),
             ("entitlement", "id"), ("entitlement", "who")] + ([("entitlement", "whom")] if stale else [])
    return Snapshot(version="v1", source_id="s", created_at="2026-10-15T00:00:00Z",
                    columns=[Column(id=f"{t}.{c}", object_id=t, name=c) for t, c in names])


def _reply(sql):
    return f'{{"sql": "{sql}", "reason": "ok"}}'


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


def _named(text, word):
    return re.search(rf"(?i)(?<![\w]){re.escape(word)}(?![\w])", text) is not None


def test_the_control_the_source_s_own_error_quotes_the_policy(adapter, caplog):
    # Without this, the tests below could pass on an error that never quoted the policy at all. The
    # operator's log keeps the whole text: it is the one place the full error is meant to be.
    with caplog.at_level(logging.WARNING):
        _ask(adapter, sql=BAD_CAST, row_filter=ENTITLED)
    logged = "\n".join(r.getMessage() for r in caplog.records if "the source said" in r.getMessage())
    # DuckDB's echo is a window around the error, here opening at `WHERE who =`: the literal is in
    # it, the table name is not.
    assert "who = 'secret_role_42'" in logged


def test_an_execution_error_reaches_the_model_without_the_policy(adapter):
    _, calls = _ask(adapter, sql=BAD_CAST, row_filter=ENTITLED)
    repair = calls[1]
    assert repair is not None
    assert not [w for w in POLICY_WORDS if _named(repair, w)], repair
    # ...and the part that repairs the query survives the cut.
    assert "Could not convert string 'west' to INT32" in repair


def test_an_explain_refusal_reaches_the_model_without_the_policy(adapter):
    _, calls = _ask(adapter, sql="SELECT id FROM claim", row_filter=STALE, stale=True)
    # What is left after the echo still names the policy's column (`whom`, and DuckDB's own
    # candidate `who`), so none of the source's words go: only ours.
    assert calls[1] == "The source rejected this query."


def test_with_nothing_narrowed_the_source_s_words_reach_the_model_whole(adapter):
    _, calls = _ask(adapter, sql=BAD_CAST)
    assert calls[1].startswith("The source rejected this query. Conversion Error: Could not convert")
    assert "LINE 1: SELECT id FROM claim WHERE amount > 0" in calls[1], "the echo is cut only under a policy"


def test_a_model_shown_none_of_the_source_s_words_may_give_its_own_reason(adapter):
    # `Feedback.from_source` follows what the model was actually shown. Withheld whole, the model saw
    # only our sentence, so its reason has nothing of the source's to carry out to the caller.
    gives_up = '{"sql": null, "reason": "Region holds names, not numbers."}'
    withheld, _ = _ask(adapter, row_filter=STALE, stale=True,
                       replies=[_reply("SELECT id FROM claim"), gives_up])
    assert withheld.deferred and withheld.answer == "Region holds names, not numbers."
    # The execution path's flag too. This policy passes EXPLAIN and fails at run time, and its error
    # names `region`, which only the policy's predicate reads here.
    withheld, calls = _ask(adapter, row_filter="CAST(region AS INTEGER) > 0",
                           replies=[_reply("SELECT id FROM claim"), gives_up])
    assert calls[1] == "The source rejected this query."
    assert withheld.deferred and withheld.answer == "Region holds names, not numbers."
    # Shown the source's words (the cut remainder is clean here), its reason stays withheld as before.
    shown, _ = _ask(adapter, row_filter=ENTITLED, replies=[_reply(BAD_CAST), gives_up])
    assert shown.deferred and shown.answer == "The model could not answer from these tables."


ASKED = "SELECT id FROM claim WHERE amount > 0 LIMIT 1000"
EXECUTED = ("SELECT id FROM (SELECT amount, id, region FROM claim WHERE id IN (SELECT id FROM "
            "entitlement WHERE who = 'secret_role_42')) AS claim WHERE amount > 0 LIMIT 1000")


def test_the_echo_is_cut_and_a_clean_remainder_kept():
    detail = ("Conversion Error: Could not convert string 'abc' to INT32\n\n"
              "LINE 1: ... WHERE who = 'secret_role_42')) AS claim WHERE amount = 'abc'\n"
              "                                                         ^")
    assert (withhold_policy_words(detail, asked=ASKED, executed=EXECUTED, dialect="duckdb")
            == "Conversion Error: Could not convert string 'abc' to INT32")


@pytest.mark.parametrize("word", ["entitlement", "ENTITLEMENT", "who", "secret_role_42"])
def test_a_remainder_naming_a_policy_word_is_withheld(word):
    detail = f'Binder Error: something about "{word}" here'
    assert withhold_policy_words(detail, asked=ASKED, executed=EXECUTED, dialect="duckdb") is None


def test_a_word_inside_a_longer_one_is_not_a_mention():
    # `who` inside `whole` or `anywho` is not the policy's column; withholding on it would cost the
    # repair. One word for each side the boundary guards.
    detail = "Binder Error: the whole statement is ambiguous near anywho"
    assert withhold_policy_words(detail, asked=ASKED, executed=EXECUTED, dialect="duckdb") == detail


def test_a_word_the_caller_wrote_too_is_not_the_policy_s():
    # `amount` and `claim` are in the statement as asked, so naming them tells the model nothing.
    detail = 'Binder Error: column "amount" of "claim" is ambiguous'
    assert withhold_policy_words(detail, asked=ASKED, executed=EXECUTED, dialect="duckdb") == detail


def test_a_granted_column_only_the_rewrite_names_is_withheld_too():
    # `region` is granted, but only the policy's derived table names it here. An error naming it
    # came from the policy's part, which reads rows the filter removes (see the docstring). The
    # predicate-reads-it case runs end to end in the `CAST(region AS INTEGER) > 0` policy above.
    detail = 'Binder Error: Referenced column "regoin" not found\nCandidate bindings: "region"'
    assert withhold_policy_words(detail, asked=ASKED, executed=EXECUTED, dialect="duckdb") is None


def test_with_nothing_narrowed_the_detail_is_returned_untouched():
    detail = "Error\n\nLINE 1: SELECT id FROM claim\n               ^"
    assert withhold_policy_words(detail, asked=ASKED, executed=ASKED, dialect="duckdb") == detail


def test_an_unknown_asked_statement_withholds():
    # No record of what was asked means no way to tell the policy's words from the caller's.
    assert withhold_policy_words("Error: x", asked=None, executed=EXECUTED, dialect="duckdb") is None


def test_an_executed_statement_that_will_not_parse_withholds():
    assert withhold_policy_words("Error: x", asked=ASKED, executed="SELECT (((", dialect="duckdb") is None


def test_no_detail_stays_none():
    assert withhold_policy_words(None, asked=ASKED, executed=EXECUTED, dialect="duckdb") is None


# Every row but one is hidden by each policy below, and every hidden row would fail the caller's cast
# with a value spelled `hidden_<n>`. DuckDB keeps no promise to filter by the policy first; this holds
# that it does, because `withhold_policy_words` keeps a cut remainder naming only the caller's words,
# which is safe only while the caller's predicate never sees a removed row.
HIDING = ["region = 'west'", "region LIKE 'w%'", "length(region) = 4", "upper(region) = 'WEST'",
          "id = 1", "id % 20000 = 1", "region IN ('west', 'nowhere')", "starts_with(region, 'we')"]
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
                "CASE WHEN i = 1 THEN '10' ELSE 'hidden_' || i END FROM range(1, 20000) t(i)")
    con.close()
    return DuckDBAdapter.duckdb(path)


@pytest.mark.parametrize("cast", CASTS)
@pytest.mark.parametrize("policy", HIDING)
def test_a_caller_s_predicate_never_raises_on_a_row_the_policy_removes(hiding_adapter, policy, cast):
    answer, calls = _ask(hiding_adapter, sql=cast, row_filter=policy)
    assert not answer.failed and not answer.deferred, (answer.answer, calls)
    assert calls == [None], "the one visible row casts cleanly, so nothing was ever repaired"
