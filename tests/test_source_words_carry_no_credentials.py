"""M96: a source's error reaches the model without the credentials in it.

The model is shown the source's own words when a query is refused or fails -- they are the
repair's whole input -- and it can copy them into a string literal of its SQL, which reaches the
caller: measured, `SELECT id, 'postgresql://svc:pw@10.2.0.7' AS note FROM claim` was approved
verbatim. The owner's decision (2026-10-14): never show credentials. A DSN has no repair value,
and text the model never sees it cannot launder.
"""

from __future__ import annotations

import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.execute.runner import ExecutionError, run
from mnemiq.generate.generator import FakeGenerator
from mnemiq.generate.plan_query import plan_query
from mnemiq.redact import scrub_credentials
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.prove import prove

_DUCKDB_PG = ('IO Error: Unable to connect to Postgres at "host=10.0.0.5 user=svc '
              'password=hunter2 dbname=acme": connection refused')


@pytest.mark.parametrize("text,secret,kept", [
    (_DUCKDB_PG, "hunter2", "host=10.0.0.5"),
    ("could not reach postgresql://svc:pw@10.2.0.7/acme", "svc:pw", "10.2.0.7/acme"),
    ("mysql://root:s3cr@t@db:3306/x refused", "s3cr@t", "mysql://***@db:3306/x refused"),
    ("jdbc:postgresql://h/db?user=u&password=y9&ssl=true", "y9", "ssl=true"),
    ("Driver={x};UID=u;PWD=topsecret;Server=s", "topsecret", "Server=s"),
    ("password='a b c' host=h", "a b c", "host=h"),
    ('Password = "q w"', "q w", "Password"),
    # The review gate's shapes on the first version, each of which came back unchanged:
    ("PGPASSWORD=hunter2 psql -h db", "hunter2", "psql -h db"),
    ("password: hunter2\nhost: db", "hunter2", "host: db"),
    ('{"password": "x9", "host": "db"}', "x9", '"host": "db"'),
    ("postgresql://svc:p/w@h/db", "p/w", "postgresql://***@h/db"),
    ("Driver={x};PWD={a;b};Server=s", "a;b", "Server=s"),
    ("password=abc123== host=h", "abc123", "password=*** host=h"),  # a base64 tail of `=`
    # The gate's second round: a bare value holds `,` and braces; braces escape as `}}`; and
    # libpq skips whitespace after `=` (an empty value is `''`), so these are all password.
    ("password=a,b host=h", "a,b", "password=*** host=h"),
    ("password=ab{cd host=h", "ab{cd", "password=*** host=h"),
    ("Driver={x};PWD={a}}b};Server=s", "b}", "Driver={x};PWD=***;Server=s"),
    ("password= se=cret host=h", "se=cret", "password= *** host=h"),
    ("password= abc123== host=h", "abc123", "password= *** host=h"),
    ("password=\n  secret", "secret", "password=\n  ***"),  # any whitespace, as libpq skips it
    ("password=\r\nsecret host=h", "secret", "password=\r\n*** host=h"),
])
def test_a_credential_in_a_source_error_is_taken_out(text, secret, kept):
    scrubbed = scrub_credentials(text)
    assert secret not in scrubbed and kept in scrubbed, scrubbed


@pytest.mark.parametrize("text", [
    "passwordless=true",
    "contact user@example.com",
    "column \"password\" does not exist",
    "the password column is not granted",
    "SELECT 'a@b' FROM t",
    "Binder Error: Referenced column x not found",
])
def test_ordinary_words_are_left_as_they_are(text):
    assert scrub_credentials(text) == text


class _Refusing:
    dialect = "duckdb"

    def __init__(self, times: int = 1):
        self.left = times

    def execute(self, sql):
        if sql.startswith("EXPLAIN") and self.left:
            self.left -= 1
            raise RuntimeError(_DUCKDB_PG)
        return []

    def execute_arrow(self, sql, timeout_s=None):
        raise RuntimeError(_DUCKDB_PG)


def test_a_refused_plan_repairs_against_no_credential():
    refusal = prove(_Refusing(), "SELECT 1")
    assert "hunter2" not in refusal.repair_text
    assert "host=10.0.0.5" in refusal.repair_text  # the repair input survives


def test_a_failed_execution_repairs_against_no_credential():
    with pytest.raises(ExecutionError) as caught:
        run(_Refusing(), "SELECT 1")
    assert "hunter2" not in caught.value.repair_text
    assert "connection refused" in caught.value.repair_text


def test_the_model_is_never_handed_the_credential():
    """The join: `plan_query` feeds a refused plan's repair text to the generator. The second
    proposal is the one made with feedback, so the feedback it was handed is what the model read."""
    snapshot = Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id="claim.id", object_id="claim", name="id")])
    packet = ContextPacket(question="how many claims?",
                           cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
                           grant_fingerprint="fp", enrichment_version="v1")
    generator = FakeGenerator(['{"sql": "SELECT id FROM claim"}'] * 2)
    plan_query(packet, snapshot, GrantSet(frozenset({"claim"})), generator,
               adapter=_Refusing(times=1), target="duckdb")
    assert len(generator.calls) == 2, generator.calls
    feedback = generator.calls[1]
    assert feedback and "hunter2" not in feedback and "host=10.0.0.5" in feedback
