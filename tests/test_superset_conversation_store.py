"""The conversation store on the control Postgres: the tests of `test_superset_conversations.py`, run here on a
`ConversationStore` in a schema of each test's own, plus what only a server shows, two connections appending to one
conversation and claiming one proposal at once.

Gated as `test_superset_build_store.py` is, on MNEMIQ_TEST_BUILD_STORE_DSN, and run with a server by the
superset-build-store workflow, which MNEMIQ_TEST_BUILD_STORE_REQUIRED keeps from passing by skipping them."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from mnemiq.superset.build_store import BuildStoreMissing
from mnemiq.superset.conversations import CONVERSATION_TABLE, ConversationStore
from tests.test_superset_build_store import WORKFLOW, _database_dsn
from tests.test_superset_conversations import (  # noqa: F401  (collected here, on this module's fixture)
    ME,
    SOMEONE_ELSE,
    TENANT,
    _ask,
    _proposal,
    test_a_conversation_holds_at_most_max_messages,
    test_a_conversation_is_its_owners_alone,
    test_a_list_limit_out_of_range_is_refused,
    test_a_malformed_cursor_is_refused,
    test_a_refused_message_is_not_kept,
    test_a_settled_proposal_refuses_a_second_settle,
    test_an_outcome_is_held_to_the_shape,
    test_an_unknown_conversation_is_not_found,
    test_list_pages_through_conversations_updated_at_the_same_moment,
    test_messages_append_in_order_and_concurrent_appends_both_land,
    test_the_first_persons_message_titles_the_conversation,
    test_the_sentence_and_sql_are_kept,
    test_the_store_stamps_each_message_with_its_own_time,
    test_two_settles_at_once_settle_once,
)


class _PostgresClock:
    """The store's clock is Postgres's: a test that moves the clock waits for it instead."""

    def advance(self, seconds: float) -> None:
        threading.Event().wait(min(seconds, 0.01))


@pytest.fixture
def store_dsn() -> Iterator[str]:
    database = _database_dsn()
    schema = f"conversation_store_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(database, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield make_conninfo(database, options=f"-c search_path={schema}")
    finally:
        with psycopg.connect(database, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def stores(store_dsn) -> Iterator:
    made: list[ConversationStore] = []

    def make() -> ConversationStore:
        made.append(ConversationStore(store_dsn))
        made[-1].clock = _PostgresClock()
        return made[-1]

    try:
        yield make
    finally:
        for store in made:
            store.close()


@pytest.fixture
def conversations(stores) -> ConversationStore:
    return stores()


def test_list_is_newest_first_and_pages_by_before(conversations):
    from tests.test_superset_conversations import test_list_is_newest_first_and_pages_by_before as on_memory

    on_memory(conversations)


def test_two_connections_appending_at_once_keep_both(stores):
    first, second = stores(), stores()
    talk = first.create(TENANT, ME, "")
    start = threading.Barrier(2)
    indexes: list[int] = []

    def send(store, n: int) -> None:
        start.wait()
        for k in range(10):
            indexes.append(store.append(TENANT, talk.conversation_id, ME, _ask(f"{n}-{k}")))

    threads = [threading.Thread(target=send, args=(store, n)) for n, store in enumerate((first, second))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    kept = second.load(TENANT, talk.conversation_id, ME)
    assert sorted(indexes) == list(range(20))
    assert sorted(message.text for message in kept.messages) == sorted(f"{n}-{k}" for n in (0, 1) for k in range(10))


def test_two_connections_claiming_one_proposal_claim_it_once(stores):
    first, second = stores(), stores()
    talk = first.create(TENANT, ME, "")
    index = first.append(TENANT, talk.conversation_id, ME, _proposal())
    start = threading.Barrier(2)
    won: list[bool] = []

    def claim(store) -> None:
        start.wait()
        won.append(store.settle(TENANT, talk.conversation_id, ME, index, {"state": "applying"}, expect="proposed"))

    threads = [threading.Thread(target=claim, args=(store,)) for store in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(won) == [False, True]


def test_the_table_holds_no_row_of_results(conversations, store_dsn):
    """Read back from Postgres itself: every key at every depth of every kept message, and none is one a value, a row
    or a sample is kept under."""
    from mnemiq.superset.build_store import VALUE_SHAPED_KEYS
    from tests.test_superset_conversations import _answer

    talk = conversations.create(TENANT, ME, "")
    for message in (_ask(), _answer(), _proposal()):
        conversations.append(TENANT, talk.conversation_id, ME, message)
    with psycopg.connect(store_dsn, autocommit=True) as con:
        keys = {row[0] for row in con.execute(
            f"WITH RECURSIVE walk(doc) AS (SELECT jsonb_array_elements(messages) FROM {CONVERSATION_TABLE} "
            "UNION ALL SELECT e.value FROM walk, LATERAL (SELECT value FROM jsonb_each(walk.doc) "
            "WHERE jsonb_typeof(walk.doc) = 'object' UNION ALL SELECT value FROM jsonb_array_elements(walk.doc) "
            "WHERE jsonb_typeof(walk.doc) = 'array') e) "
            "SELECT DISTINCT k FROM walk, LATERAL jsonb_object_keys(walk.doc) k "
            "WHERE jsonb_typeof(walk.doc) = 'object'").fetchall()}
    assert keys  # a positive control: the walk reads the messages' keys
    assert {"answer", "sentence", "sql", "proposal", "charts"} <= keys
    assert not keys & VALUE_SHAPED_KEYS


def test_another_persons_rows_are_untouched_by_my_writes(stores):
    store = stores()
    theirs = store.create(TENANT, SOMEONE_ELSE, "")
    index = store.append(TENANT, theirs.conversation_id, SOMEONE_ELSE, _proposal())
    assert store.settle(TENANT, theirs.conversation_id, ME, index, {"state": "applying"}, expect="proposed") is False
    assert store.delete(TENANT, theirs.conversation_id, ME) is False
    kept = store.load(TENANT, theirs.conversation_id, SOMEONE_ELSE)
    assert kept.messages[index].doc["outcome"] == {"state": "proposed"}


@pytest.mark.parametrize("dsn", [None, "", "   "])
def test_missing_store_is_named(dsn):
    with pytest.raises(BuildStoreMissing):
        ConversationStore(dsn)


def test_the_ci_workflow_runs_these_tests_with_a_server():
    """The workflow's one store step runs this module beside the build store's, so the env that gives those a server
    and refuses a skip gives these one too."""
    text = WORKFLOW.read_text()
    runs = [line.split() for line in text.splitlines() if line.lstrip().startswith("- run:") and "pytest" in line]
    module = f"tests/{Path(__file__).name}"
    assert [words for words in runs if module in words] == [
        words for words in runs if "tests/test_superset_build_store.py" in words], (
        f"{module} is not run in the step that runs the build store's tests in {WORKFLOW.name}")
