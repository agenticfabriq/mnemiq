"""Superset mode on Oracle uses a value enrichment harvested from a table's rows only in a request of a person no
Superset row rule applies to on that table.

A snapshot enriched directly from Oracle for Superset mode (`mnemiq enrich --for-superset`, whose `superset:oracle` job
says `values: harvested`) holds what the enrichment login read past every row rule: each column's code list and the
meanings grounded for it, the code scheme its values bound it to, the descriptions and table facts the model wrote from
them, the examples written and run on them, and the value index. `resolve_grants` asks Superset, as the person, whether
a row rule applies to them on each granted table (the extension's `table_rules`, through the request's adapter, which
answers from what the request's own grant check was told), and every table it does not answer "no" for is withheld
(`GrantSet.values_withheld`); one predicate, `values_shown`, says whether a request may use a table's values. The
snapshot a request reads is the served one with the withheld tables' harvested content taken out (`values_scoped`), and
the value index it reads answers only for tables whose values are shown (`ShownValues`). On PostgreSQL, and for a
snapshot whose values were not harvested, nothing is asked and nothing changes.

Superset is `FakeSuperset`, a real MCP server on loopback, answering as a Superset whose database is Oracle's and where
a row rule applies to the person on the tables a test names.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest
import sqlglot
from sqlglot import exp

import mnemiq
from mnemiq.agent.loop import Agent
from mnemiq.authz.grants import GrantSet, values_shown
from mnemiq.contract import CertifiedRecord, CodedValue, Column, IdentityContext, RecordEnvelope, Snapshot
from mnemiq.contract.semantic import (
    CodeScheme,
    ColumnEstimate,
    Example,
    JoinKey,
    Relationship,
    SourceBinding,
    TableFacts,
)
from mnemiq.llm.embeddings import LLMEmbedder
from mnemiq.runtime import build_runtime
from mnemiq.semantic.store import build_index
from mnemiq.semantic.values import NoValueIndex, ValueIndex
from mnemiq.sql.decide import decide
from mnemiq.sql.schema import visible_schema
from mnemiq.sql.verdict import Refusal
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import save_snapshot
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.client import DelegatedToken, SupersetUnavailable
from mnemiq.superset.extension import TOOL_PREFIX
from mnemiq.superset.grants import resolve_grants
from mnemiq.enrichment.certified import CertifiedSet, apply_certified
from mnemiq.superset.harvested import ShownValues, snapshot_objects, values_harvested, values_scoped
from tests.superset_fake import (
    FakeSuperset,
    RecordingClient,
    dry_run_as_sent,
    tables_read_as_mnemiq,
)
from tests.superset_shared import (
    DATABASE_ID,
    ORACLE_SCHEMA_SETTING,
    SOURCE_ID,
    _oracle_job,
    _source_job,
    superset_deployment,
)
from tests.sweep.stub_model import StubModel

TOKEN = "tok-asker"
ASKER = IdentityContext(tenant_id="t", principal_id="asker")
BINDING = TOOL_PREFIX + "session_binding"
TABLE_RULES = TOOL_PREFIX + "table_rules"

# The two tables, by the object ids an Oracle snapshot gives them (as Oracle stores them), and as Superset shows them.
ORDERS, ITEMS = "ORDERS", "ITEMS"
SHOWN = {ORDERS: "orders", ITEMS: "items"}

# Words only one table's harvested content holds: a prompt holding one shows that table's values.
HARVESTED = {
    ORDERS: {"code": "HOLD_FOR_AUDIT", "meaning": "kept back for the audit desk", "scheme": "Audit desk codes",
             "description": "Where the order stands as the audit desk records it",
             "fact": "an order the audit desk holds books no revenue", "question": "How many orders await the desk?"},
    ITEMS: {"code": "SHIPPED_TO_DEPOT", "meaning": "sent on to the depot", "scheme": "Depot route codes",
            "description": "Where the item stands on its way to the depot",
            "fact": "an item sent to the depot is counted once", "question": "How many items went to the depot?"},
}
# What enrichment read from the database's dictionary or classified, never from a row: kept whatever the rules.
COMMENTS = {ORDERS: "lifecycle status of the order", ITEMS: "lifecycle status of the item"}
# A value another source's enrichment left in the store's value index under the same table.
OTHER_SOURCE = "acme"
OTHER_SOURCES_VALUE = "LEFT_BY_ANOTHER_SOURCE"


def _status(table: str, *, source: str = "correlated") -> Column:
    words = HARVESTED[table]
    return Column(id=f"{table}.STATUS", object_id=table, name="STATUS", data_type="VARCHAR2(20)", semantic_type="code",
                  pii_level="none", description=words["description"], comment=COMMENTS[table],
                  coded_values=[CodedValue(code=words["code"], meaning=words["meaning"], source=source)],
                  code_scheme=CodeScheme(id=f"scheme-{table.lower()}", label=words["scheme"]),
                  row_count=10, distinct_count=1, null_count=0)


def _key(table: str) -> Column:
    return Column(id=f"{table}.ID", object_id=table, name="ID", data_type="NUMBER", semantic_type="identifier",
                  pii_level="none", row_count=10, distinct_count=10, null_count=0)


def _harvested_snapshot(*, jobs=None, version: str = "v1") -> Snapshot:
    """A snapshot enriched directly from Oracle for Superset mode, with values harvested, of two tables: each with its
    key and a coded status column carrying a meaning, a code scheme, a description and the database's comment, a table
    fact, and an example."""
    tables = (ORDERS, ITEMS)
    return Snapshot(
        version=version, source_id=SOURCE_ID, created_at="t", jobs=[_oracle_job()] if jobs is None else jobs,
        columns=[column for table in tables for column in (_key(table), _status(table))],
        source_bindings=[SourceBinding(id=f"sb:{t}", source_id=SOURCE_ID, object_id=t, source_object=t,
                                       binding_type="table") for t in tables],
        table_facts=[TableFacts(object_id=t, gotchas=[HARVESTED[t]["fact"]], reads={HARVESTED[t]["fact"]: [t]})
                     for t in tables],
        examples=[Example(question=HARVESTED[t]["question"],
                          sql=f"SELECT COUNT(*) AS n FROM {t} WHERE STATUS = '{HARVESTED[t]['code']}'",
                          tables=[t], object_id=t) for t in tables])


# --- Superset, answering as the person ------------------------------------------------------------------------------


def _wrapped(sql: str, ruled: set[str]) -> str:
    """`sql` as rc2's executor would run it for a person a row rule applies to on each table in `ruled` (in Superset's
    spelling): each such table read through a subquery carrying the rule, under the table's own alias."""
    tree = sqlglot.parse_one(sql, read="oracle")
    for table in list(tree.find_all(exp.Table)):
        if table.name in ruled:
            rule = sqlglot.parse_one(f"SELECT * FROM {table.name} WHERE region = 'west'", read="oracle")
            alias = exp.TableAlias(this=exp.to_identifier(table.alias or table.name, quoted=True))
            table.replace(exp.Subquery(this=rule, alias=alias))
    return tree.sql(dialect="oracle", pretty=True)


class RuledSuperset:
    """What `FakeSuperset` answers as a Superset whose database is Oracle's, where a row rule applies to the person on
    each table named in `ruled` (in Superset's spelling): the person's session bound, each dry run as Superset would
    run the statement under those rules, every catalogue statement with no rows and every other with one count, the
    extension's `tables_read` as mnemiq reads a statement, and `table_rules` answered by `rules`, which a test may
    change, or replace with `answer`, a function of the names asked."""

    def __init__(self, fake: FakeSuperset, ruled: set[str]) -> None:
        self.ruled = set(ruled)
        self.answer = None
        fake.reply(BINDING, lambda request, authorization: {"bound": True})
        fake.reply("execute_sql", self._execute)
        fake.reply(TOOL_PREFIX + "tables_read", lambda request, authorization: tables_read_as_mnemiq(request))
        fake.reply(TABLE_RULES, self._rules)

    def _rules(self, request, authorization):
        if "tables" not in request:
            return {"refused": "the request names no tables"}
        if self.answer is not None:
            return self.answer(request["tables"])
        return {"tables": [{"table": name, "applies": name in self.ruled} for name in request["tables"]]}

    def _execute(self, request, authorization):
        if request.get("dry_run"):
            answer = dry_run_as_sent(request)
            answer["statements"][0]["executed_sql"] = _wrapped(answer["statements"][0]["executed_sql"], self.ruled)
            return answer
        if request.get("force_refresh"):
            return {"success": True, "columns": [], "rows": [], "error": None, "error_type": None}
        return {"success": True, "columns": [{"name": "N", "type": "int64"}], "rows": [{"N": 3}], "error": None,
                "error_type": None}


def _rules_asked(fake: FakeSuperset) -> list[list[str]]:
    """The tables each `table_rules` call asked, in order."""
    return [call.request["tables"] for call in fake.calls if call.tool == TABLE_RULES and "tables" in call.request]


@pytest.fixture(autouse=True)
def _no_mnemiq_environment(monkeypatch):
    """`Settings()` reads the environment, matching each name in any case: a variable set in the shell running the
    tests must not decide them."""
    for name in [n for n in os.environ if n.upper().startswith("MNEMIQ_") and n.upper() != "MNEMIQ_STORE_PATH"]:
        monkeypatch.delenv(name)


@pytest.fixture
def served():
    with FakeSuperset() as fake, StubModel(sql=f"SELECT COUNT(*) AS n FROM {ITEMS}") as model:
        yield fake, model


def _deployment(tmp_path, served, snapshot: Snapshot, *, schema: str = ORACLE_SCHEMA_SETTING):
    """A runtime in Superset mode on Oracle serving `snapshot`, its store indexed as `mnemiq build` indexes it and its
    value index holding what enrichment harvested, beside a value another source's enrichment left there."""
    fake, model = served
    deployment = superset_deployment(tmp_path, model.url, fake.url, snapshot=False, superset_schema=schema)
    con = init_store(deployment.store_path)
    save_snapshot(con, snapshot)
    build_index(con, snapshot, LLMEmbedder(deployment))
    ValueIndex(con)  # the index's table, as enrichment creates it
    rows = [(SOURCE_ID, t, "STATUS", HARVESTED[t]["code"]) for t in (ORDERS, ITEMS)]
    rows.append((OTHER_SOURCE, ITEMS, "STATUS", OTHER_SOURCES_VALUE))
    con.executemany("INSERT INTO value_index (source_id, object_id, column_name, value_text) VALUES (?, ?, ?, ?)", rows)
    con.close()
    return build_runtime(deployment)


@pytest.fixture
def runtime(tmp_path, served):
    return _deployment(tmp_path, served, _harvested_snapshot())


@pytest.fixture
def answered(monkeypatch):
    """Each agent that answers, as it answers: its value index, and the snapshot and grants it answers under."""
    seen: list[dict] = []
    real = Agent.answer

    def recording(self, packet, snapshot, grants, *args, **kwargs):
        seen.append({"values": self.values, "snapshot": snapshot, "grants": grants, "packet": packet})
        return real(self, packet, snapshot, grants, *args, **kwargs)

    monkeypatch.setattr(Agent, "answer", recording)
    return seen


def _prompts(model) -> str:
    return "\n".join(json.dumps(request) for request in model.requests)


def _ask(runtime, question: str = "how many items went out?"):
    return runtime.ask(question, ASKER, delegation=DelegatedToken(TOKEN))


def _scope(runtime) -> dict:
    return runtime.scope(ASKER, delegation=DelegatedToken(TOKEN))


# --- the predicate ----------------------------------------------------------------------------------------------------


def test_values_are_shown_unless_withheld_or_read_through_a_row_filter():
    """`values_shown` is the one predicate: no for a table the request withholds, as for one its grants read through a
    row filter, matched whatever the case (withholding more is never a disclosure), yes for any other."""
    grants = GrantSet(frozenset({ORDERS, ITEMS, "CLAIM"}), values_withheld=frozenset({ORDERS}),
                      row_filters={"claim": "region = 'west'"})
    assert values_shown(grants, ITEMS)
    assert not values_shown(grants, ORDERS) and not values_shown(grants, "orders")
    assert not values_shown(grants, "CLAIM")
    assert values_shown(GrantSet(frozenset({ORDERS})), ORDERS)


def test_the_fingerprint_is_what_it_was_unless_values_are_withheld():
    """A grant set's fingerprint is what it was where nothing is withheld (pinned as it was before values could be),
    so direct mode's and PostgreSQL's are unchanged, and differs where values are withheld: a turn answered while a
    table's values were shown is not carried into a request that withholds them (`scope_history`)."""
    pinned = GrantSet(frozenset({"a", "b"}), row_filters={"a": "x = 1"}, pii_clearance=frozenset({"pii"}))
    assert pinned.fingerprint == "5c8c44d7aa5a"
    tables = GrantSet(frozenset({ORDERS, ITEMS}))
    assert tables.fingerprint == "bd3ba07eb386"
    withheld = GrantSet(frozenset({ORDERS, ITEMS}), values_withheld=frozenset({ORDERS}))
    assert withheld.fingerprint != tables.fingerprint
    assert withheld.fingerprint != GrantSet(frozenset({ORDERS, ITEMS}), values_withheld=frozenset({ITEMS})).fingerprint


# --- which tables the grants withhold ---------------------------------------------------------------------------------


def _oracle_client(wrapped=(), *, rules=None) -> RecordingClient:
    """A recording stand-in for Superset on Oracle: every dry run as Superset runs it under a rule on each table in
    `wrapped` (Superset's spelling), `tables_read` as mnemiq reads a statement, and `table_rules` answered by `rules`, a
    function of the names asked returning the answer (or an exception to raise), else with a rule on `wrapped`."""
    wrapped = set(wrapped)

    def answer(request: dict):
        if request.get("dry_run"):
            answered = dry_run_as_sent(request)
            answered["statements"][0]["executed_sql"] = _wrapped(answered["statements"][0]["executed_sql"], wrapped)
            return answered
        if "sql" in request:
            return tables_read_as_mnemiq(request)
        if "tables" in request:
            if rules is not None:
                return rules(request["tables"])
            return {"tables": [{"table": n, "applies": n in wrapped} for n in request["tables"]]}
        return {"refused": "the request names no tables"}

    return RecordingClient(answer)


def _adapter(client, *, dialect: str = "oracle") -> SupersetSqlAdapter:
    return SupersetSqlAdapter(client, database_id=DATABASE_ID, schema=ORACLE_SCHEMA_SETTING, dialect=dialect)


def _table_rule_calls(client: RecordingClient) -> list[dict]:
    return [sent.request for sent in client.calls if sent.tool == TABLE_RULES and "tables" in sent.request]


def _unreadable_naming(table: str):
    """`table_rules` answering a call that names `table` with nothing mnemiq can read, and any other as it is ruled."""
    def rules(names):
        if table in names:
            return {"tables": "unreadable"}
        return {"tables": [{"table": n, "applies": False} for n in names]}
    return rules


@pytest.mark.parametrize("told", [True, None, False], ids=["a rule applies", "cannot tell", "no rule"])
def test_a_harvested_snapshots_values_are_withheld_unless_superset_says_no_rule_applies(superset_oracle, told):
    """With values harvested, the grants withhold every table Superset does not answer "no rule" for: a rule that
    applies, and an answer mnemiq cannot read, which cannot tell of any table its call asked about. A table with no
    rule is shown."""
    if told is None:  # Superset wraps both tables, and its rule lookup cannot be read
        client = _oracle_client({SHOWN[ORDERS], SHOWN[ITEMS]}, rules=_unreadable_naming(SHOWN[ORDERS]))
    else:
        client = _oracle_client({SHOWN[ORDERS]} if told else ())
    grants = resolve_grants(_adapter(client), _harvested_snapshot())
    assert grants.available and grants.objects == {ORDERS, ITEMS}
    expected = {True: {ORDERS}, None: {ORDERS, ITEMS}, False: set()}[told]
    assert grants.values_withheld == expected
    assert values_shown(grants, ITEMS) is (told is not None)


def test_the_rules_are_asked_as_the_person_in_supersets_spelling_and_no_more_than_the_grant_check_did(superset_oracle):
    """The tables are asked in Superset's spelling, in the configured schema and database, through the request's
    adapter, which answers from what the grant check's own dry run was told: harvested values cost no call of their
    own."""
    client = _oracle_client({SHOWN[ORDERS]})
    grants = resolve_grants(_adapter(client), _harvested_snapshot())
    assert grants.values_withheld == {ORDERS}
    [call] = _table_rule_calls(client)
    assert (call["database_id"], call["schema"], sorted(call["tables"])) == (
        DATABASE_ID, ORACLE_SCHEMA_SETTING, ["items", "orders"])


def test_the_adapter_asks_only_what_this_requests_statements_were_not_told(superset_oracle):
    """`SupersetSqlAdapter.table_rules` answers a table a statement of this request was checked against from that
    check, and asks Superset once, `TABLE_RULES_BATCH` names a call, for the rest."""
    client = _oracle_client({SHOWN[ORDERS]})
    adapter = _adapter(client)
    adapter.dry_run(f"SELECT 1 FROM {ITEMS}")
    assert adapter.table_rules(["items", "orders"]) == {"items": False, "orders": True}
    assert adapter.table_rules(["orders", "items"]) == {"orders": True, "items": False}
    assert [call["tables"] for call in _table_rule_calls(client)] == [["items"], ["orders"]]
    adapter.table_rules([f"t{i}" for i in range(51)])
    assert [len(call["tables"]) for call in _table_rule_calls(client)[2:]] == [50, 1]


def test_an_answer_mnemiq_cannot_read_is_never_kept(superset_oracle):
    answers = iter([{"tables": "unreadable"}, {"tables": [{"table": "orders", "applies": False}]}])
    adapter = _adapter(_oracle_client(rules=lambda names: next(answers)))
    assert adapter.table_rules(["orders"]) == {"orders": None}
    assert adapter.table_rules(["orders"]) == {"orders": False}


def test_superset_not_answering_the_rules_answers_nothing(superset_oracle, monkeypatch):
    """Superset not answering the rule lookup is Superset unavailable for the request, never values shown: the lookup
    raises it, and the grants are unavailable, as for any question about them Superset does not answer."""
    from mnemiq.superset.grants import UnreadableAnswer, _values_withheld

    gone = SupersetUnavailable("Superset did not answer", detail="connection refused")
    adapter = _adapter(_oracle_client(rules=lambda names: gone))
    with pytest.raises(SupersetUnavailable):
        adapter.table_rules(["orders"])
    with pytest.raises(UnreadableAnswer):
        _values_withheld(adapter, _harvested_snapshot(), frozenset({ORDERS}))
    answering = _adapter(_oracle_client())

    def unanswered(names):
        raise gone

    monkeypatch.setattr(answering, "table_rules", unanswered)
    assert not resolve_grants(answering, _harvested_snapshot()).available


def test_a_snapshot_binding_no_table_withholds_everything_it_names(superset_oracle):
    """With no table to grant, nothing is asked and every object the harvested snapshot names is withheld, as anywhere
    the grants were not told a table is free."""
    snapshot = _harvested_snapshot().model_copy(update={"source_bindings": []})
    client = _oracle_client()
    grants = resolve_grants(_adapter(client), snapshot)
    assert grants.objects == frozenset() and grants.values_withheld == {ORDERS, ITEMS}
    assert client.calls == []


def test_a_table_not_granted_is_withheld_and_never_asked(superset_oracle):
    """Only the tables the person may read are asked about; every other table of the snapshot is withheld."""
    from mnemiq.superset.grants import _values_withheld

    client = _oracle_client()
    assert _values_withheld(_adapter(client), _harvested_snapshot(), frozenset({ITEMS})) == {ORDERS}
    assert [call["tables"] for call in _table_rule_calls(client)] == [["items"]]


def test_a_table_stored_under_a_plain_lower_case_name_is_not_asked_and_is_withheld(superset_oracle):
    """A table stored under a plain lower-case name is never asked about, and its values are withheld."""
    from mnemiq.superset.grants import _values_withheld

    snapshot = _harvested_snapshot()
    lower = snapshot.model_copy(update={"source_bindings": [*snapshot.source_bindings, SourceBinding(
        id="sb:orders", source_id=SOURCE_ID, object_id="orders", source_object="orders", binding_type="table")]})
    client = _oracle_client()
    assert _values_withheld(_adapter(client), lower, frozenset({"orders", ITEMS})) == {"orders", ORDERS}
    assert [call["tables"] for call in _table_rule_calls(client)] == [["items"]]


def test_an_object_bound_under_several_names_is_shown_only_when_every_name_has_no_rule(superset_oracle):
    """An object the snapshot binds under two stored names is shown only when Superset says no rule applies under each
    of them: a rule under one withholds it."""
    from mnemiq.superset.grants import _values_withheld

    snapshot = _harvested_snapshot()
    twice = snapshot.model_copy(update={"source_bindings": [*snapshot.source_bindings, SourceBinding(
        id="sb:ORDERS_ARCHIVE", source_id=SOURCE_ID, object_id=ORDERS, source_object="ORDERS_ARCHIVE",
        binding_type="table")]})
    client = _oracle_client({"orders_archive"})
    assert _values_withheld(_adapter(client), twice, frozenset({ORDERS, ITEMS})) == {ORDERS}
    free = _oracle_client()
    assert _values_withheld(_adapter(free), twice, frozenset({ORDERS, ITEMS})) == frozenset()
    viewed = snapshot.model_copy(update={"source_bindings": [*snapshot.source_bindings, SourceBinding(
        id="sb:ORDERS_V", source_id=SOURCE_ID, object_id=ORDERS, source_object="ORDERS_V", binding_type="view")]})
    assert _values_withheld(_adapter(_oracle_client()), viewed, frozenset({ORDERS, ITEMS})) == {ORDERS}


def test_values_not_harvested_still_withhold_what_was_read_of_a_ruled_tables_rows(superset_oracle):
    """An Oracle snapshot whose values were not harvested still holds the counts profiling read from every row: the
    grants withhold a ruled table as for a harvested one, and ask no more than the grant check's own statement asks."""
    snapshot = _harvested_snapshot(jobs=[_oracle_job(values="not harvested")])
    assert not values_harvested(snapshot)
    client = _oracle_client({SHOWN[ORDERS]})
    grants = resolve_grants(_adapter(client), snapshot)
    assert grants.values_withheld == {ORDERS}
    assert len(_table_rule_calls(client)) == 1  # the grant check's own


def test_postgresql_asks_nothing_and_withholds_nothing():
    """A snapshot enriched through Superset, on PostgreSQL, harvested no values: no rule is asked, and the grants are
    what they always were."""
    snapshot = _harvested_snapshot(jobs=[_source_job()])
    assert not values_harvested(snapshot)
    client = RecordingClient({"success": True, "rows": None, "columns": None, "error": None, "error_type": None,
                              "statements": []})
    grants = resolve_grants(_adapter(client, dialect="postgres"), snapshot)
    assert grants == GrantSet(frozenset({ORDERS, ITEMS}), pii_clearance=frozenset())
    assert [sent.tool for sent in client.calls] == ["execute_sql"]


def test_a_snapshot_whose_record_cannot_be_read_is_read_as_harvested():
    """An Oracle job whose record cannot be read cannot say no values were harvested: read as harvested."""
    assert values_harvested(_harvested_snapshot(jobs=[_oracle_job(detail="{not json")]))
    assert values_harvested(_harvested_snapshot())


# --- the snapshot a request reads -------------------------------------------------------------------------------------


def _withholding(*tables: str) -> GrantSet:
    return GrantSet(frozenset({ORDERS, ITEMS}), values_withheld=frozenset(tables))


def _column(snapshot: Snapshot, table: str, name: str = "STATUS") -> Column:
    return next(c for c in snapshot.columns if c.object_id == table and c.name == name)


def test_a_withheld_table_loses_every_field_written_from_its_values_and_a_shown_one_keeps_them():
    """One ruled table and one free one. The ruled table's code list with its meanings, the code scheme
    its values bound it to, its descriptions (which the model wrote from its values, so may quote one), its table facts
    and every example reading it are taken out; its name, types, classifications, counts and the database's comments,
    none read from a row, are kept. The free table keeps everything."""
    snapshot = _harvested_snapshot()
    scoped = values_scoped(snapshot, _withholding(ORDERS))
    held, free = _column(scoped, ORDERS), _column(scoped, ITEMS)
    assert held.coded_values == [] and held.code_scheme is None and held.description is None
    assert (held.comment, held.data_type, held.semantic_type, held.pii_level, held.row_count, held.distinct_count) == (
        COMMENTS[ORDERS], "VARCHAR2(20)", "code", "none", 10, 1)
    assert free == _column(snapshot, ITEMS)
    assert [tf.object_id for tf in scoped.table_facts] == [ITEMS]
    assert [e.object_id for e in scoped.examples] == [ITEMS]
    assert snapshot.columns[1].coded_values  # the served snapshot itself is never changed


def test_nothing_withheld_is_the_snapshot_itself():
    """With nothing withheld, as on PostgreSQL and for a snapshot whose values were not harvested, the snapshot a
    request reads is the served one, unchanged."""
    snapshot = _harvested_snapshot()
    assert values_scoped(snapshot, GrantSet(frozenset({ORDERS, ITEMS}))) is snapshot


def test_a_column_grounded_from_a_withheld_lookup_table_withholds_its_own_table():
    """A code list grounded from a lookup table carries that table's labels: a table holding one is shown only while
    every table it joins that column to is shown, and withheld when no join of that column is recorded."""
    lookup = Column(id="ITEMS.ORDER_ID", object_id=ITEMS, name="ORDER_ID", data_type="NUMBER",
                    coded_values=[CodedValue(code="7", meaning="the audit desk's order", source="lookup")])
    joined = Relationship(id="r1", **{"from": ITEMS}, to=ORDERS, cardinality="many_to_one",
                          join_keys=[JoinKey(left="ORDER_ID", right="ID")])
    snapshot = _harvested_snapshot()
    with_lookup = snapshot.model_copy(update={"columns": [*snapshot.columns, lookup], "relationships": [joined]})
    scoped = values_scoped(with_lookup, _withholding(ORDERS))
    assert _column(scoped, ITEMS).coded_values == [] and _column(scoped, ITEMS, "ORDER_ID").coded_values == []
    assert values_scoped(with_lookup, _withholding()) is with_lookup
    unjoined = with_lookup.model_copy(update={"relationships": []})
    assert _column(values_scoped(unjoined, _withholding(ORDERS)), ITEMS, "ORDER_ID").coded_values == []


def _with_supplier_lookup(snapshot: Snapshot, *, joined: bool = True) -> Snapshot:
    """`snapshot` with a lookup-grounded column on ITEMS whose labels were read from SUPPLIERS, a table the snapshot
    binds too, joined to it where `joined`."""
    lookup = Column(id="ITEMS.SUPPLIER_ID", object_id=ITEMS, name="SUPPLIER_ID", data_type="NUMBER",
                    coded_values=[CodedValue(code="9", meaning=SUPPLIER_LABEL, source="lookup")])
    suppliers = Column(id="SUPPLIERS.ID", object_id="SUPPLIERS", name="ID", data_type="NUMBER")
    join = Relationship(id="r-suppliers", **{"from": ITEMS}, to="SUPPLIERS", cardinality="many_to_one",
                        join_keys=[JoinKey(left="SUPPLIER_ID", right="ID")])
    return snapshot.model_copy(update={
        "version": f"{snapshot.version}-suppliers",
        "columns": [*snapshot.columns, lookup, suppliers],
        "source_bindings": [*snapshot.source_bindings, SourceBinding(
            id="sb:SUPPLIERS", source_id=SOURCE_ID, object_id="SUPPLIERS", source_object="SUPPLIERS",
            binding_type="table")],
        "relationships": [*snapshot.relationships, *([join] if joined else [])]})


# A label read from SUPPLIERS, carried on ITEMS' lookup-grounded column.
SUPPLIER_LABEL = "DEPOT_SUPPLIER_LABEL"


def test_values_are_shown_only_for_a_granted_table():
    """A table the grants do not give the person is shown nothing, whatever the withheld set names: what is shown is a
    granted table Superset was asked about, never one the grants know nothing of."""
    assert not values_shown(GrantSet(frozenset({ITEMS})), ORDERS)
    assert values_shown(GrantSet(frozenset({ITEMS})), ITEMS)


def test_a_lookup_from_a_table_the_grants_were_not_decided_for_withholds_its_table():
    """Grants decided for one snapshot, scoping another that records a new lookup join from ITEMS to SUPPLIERS: the
    grants never asked of SUPPLIERS, so ITEMS' code lists, which carry SUPPLIERS' labels, are withheld, and so is
    SUPPLIERS; ORDERS, ruled, stays withheld."""
    decided = _withholding(ORDERS)  # ITEMS and ORDERS granted, only ITEMS asked and free
    scoped = values_scoped(_with_supplier_lookup(_harvested_snapshot()), decided)
    assert _column(scoped, ITEMS, "SUPPLIER_ID").coded_values == [] and _column(scoped, ITEMS).coded_values == []
    assert _column(scoped, ORDERS).coded_values == []
    values = ShownValues(NoValueIndex(), decided, frozenset({ORDERS, ITEMS, "SUPPLIERS"}))
    assert not values._shown("SUPPLIERS")


def test_a_lookup_column_with_no_join_recorded_withholds_its_table_with_nothing_withheld():
    """Where the grants withhold nothing, a lookup-grounded column whose join is not recorded still withholds its table:
    which table its labels came from cannot be told."""
    granted = GrantSet(frozenset({ORDERS, ITEMS, "SUPPLIERS"}))
    scoped = values_scoped(_with_supplier_lookup(_harvested_snapshot(), joined=False), granted)
    assert _column(scoped, ITEMS, "SUPPLIER_ID").coded_values == [] and _column(scoped, ITEMS).coded_values == []
    assert _column(scoped, ORDERS).coded_values
    joined = _with_supplier_lookup(_harvested_snapshot())
    assert values_scoped(joined, granted) is joined
    for jobs in ([_oracle_job(values="not harvested")], [_source_job()]):  # nothing harvested: nothing taken out
        unharvested = _with_supplier_lookup(_harvested_snapshot(jobs=jobs), joined=False)
        assert values_scoped(unharvested, GrantSet(frozenset({ITEMS}))) is unharvested


def test_an_example_reading_a_table_the_snapshot_does_not_name_is_withheld():
    """An example is kept only where every table it reads is one the snapshot names and shows: a spelling the snapshot
    does not hold cannot be told to be a shown table."""
    snapshot = _harvested_snapshot()
    odd = Example(question="odd", sql="SELECT 1 FROM SHOP.ITEMS", tables=["SHOP.ITEMS"], object_id=ITEMS)
    unnamed = Example(question="unnamed", sql="SELECT 1", tables=[], object_id="")
    scoped = values_scoped(snapshot.model_copy(update={"examples": [*snapshot.examples, odd, unnamed]}),
                           _withholding(ORDERS))
    assert [e.question for e in scoped.examples] == [HARVESTED[ITEMS]["question"]]


def test_snapshot_objects_names_every_object_the_snapshot_holds():
    snapshot = _harvested_snapshot()
    extra = snapshot.model_copy(update={"examples": [*snapshot.examples, Example(question="q", sql="s",
                                                                                tables=["ELSEWHERE"], object_id="")]})
    assert snapshot_objects(extra) == {ORDERS, ITEMS, "ELSEWHERE"}


# --- the value index a request reads ----------------------------------------------------------------------------------


@pytest.fixture
def index(tmp_path):
    con = init_store(str(tmp_path / "index.duckdb"))
    ValueIndex(con)
    rows = [(SOURCE_ID, t, "STATUS", HARVESTED[t]["code"]) for t in (ORDERS, ITEMS)]
    rows.append((OTHER_SOURCE, ITEMS, "STATUS", OTHER_SOURCES_VALUE))
    con.executemany("INSERT INTO value_index (source_id, object_id, column_name, value_text) VALUES (?, ?, ?, ?)", rows)
    yield con
    con.close()


def test_the_value_index_answers_no_literal_of_a_withheld_table(index):
    """The index a request reads answers a withheld table's columns as not indexed, every way it is
    read, and a shown table's as the index holds them, for the snapshot's own source only."""
    values = ShownValues(ValueIndex(index, source_id=SOURCE_ID), _withholding(ORDERS), frozenset({ORDERS, ITEMS}))
    code = HARVESTED[ORDERS]["code"]
    assert not values.has(ORDERS, "STATUS") and not values.contains(ORDERS, "STATUS", code)
    assert values.nearest(ORDERS, "STATUS", code) == [] and values.values(ORDERS, "STATUS") == []
    assert values.holders(code) == set()
    shown_code = HARVESTED[ITEMS]["code"]
    assert values.has(ITEMS, "STATUS") and values.contains(ITEMS, "STATUS", shown_code)
    assert values.values(ITEMS, "STATUS") == [shown_code] and values.holders(shown_code) == {(ITEMS, "STATUS")}
    assert not values.contains(ITEMS, "STATUS", OTHER_SOURCES_VALUE) and values.holders(OTHER_SOURCES_VALUE) == set()


def test_the_value_index_answers_only_for_objects_the_snapshot_names(index):
    values = ShownValues(ValueIndex(index, source_id=SOURCE_ID), _withholding(), frozenset({ORDERS}))
    assert values.has(ORDERS, "STATUS") and not values.has(ITEMS, "STATUS")
    assert values.holders(HARVESTED[ITEMS]["code"]) == set()


def test_a_request_reads_the_index_only_for_a_harvested_snapshot_of_its_own_source(index):
    """`served_values` hands a request the index, as its grants let it read it, only for a snapshot that harvested
    values, of the source the index is held to; none for any other, and none where no index was opened."""
    from mnemiq.superset.harvested import served_values

    snapshot, grants = _harvested_snapshot(), _withholding(ORDERS)
    held = ValueIndex(index, source_id=SOURCE_ID)
    served = served_values(held, snapshot, grants)
    assert isinstance(served, ShownValues) and served.has(ITEMS, "STATUS") and not served.has(ORDERS, "STATUS")
    assert isinstance(served_values(ValueIndex(index, source_id=OTHER_SOURCE), snapshot, grants), NoValueIndex)
    assert isinstance(served_values(ValueIndex(index), snapshot, grants), NoValueIndex)
    assert isinstance(served_values(None, snapshot, grants), NoValueIndex)
    not_harvested = _harvested_snapshot(jobs=[_oracle_job(values="not harvested")])
    assert isinstance(served_values(held, not_harvested, grants), NoValueIndex)
    assert isinstance(served_values(held, _harvested_snapshot(jobs=[_source_job()]), grants), NoValueIndex)


def test_an_index_given_a_source_reads_that_sources_rows(index):
    """With a source id the index reads that source's rows; without one, every row it holds."""
    assert ValueIndex(index).contains(ITEMS, "STATUS", OTHER_SOURCES_VALUE)
    assert not ValueIndex(index, source_id=SOURCE_ID).contains(ITEMS, "STATUS", OTHER_SOURCES_VALUE)
    assert ValueIndex(index, source_id=SOURCE_ID).values(ITEMS, "STATUS") == [HARVESTED[ITEMS]["code"]]


@pytest.mark.parametrize("table", [ITEMS, ORDERS])
def test_the_value_check_quotes_a_shown_tables_values_and_never_checks_a_withheld_one(index, table):
    """Through the decider: a literal the shown table's column does not hold is refused naming its real values; the
    same on the withheld table is not checked at all, so neither its values nor whether one exists is told."""
    snapshot = _harvested_snapshot()
    grants = _withholding(ORDERS)
    values = ShownValues(ValueIndex(index, source_id=SOURCE_ID), grants, snapshot_objects(snapshot))
    verdict = decide(f"SELECT COUNT(*) AS n FROM {table} WHERE STATUS = 'NOT_A_CODE'", visible_schema(snapshot, grants),
                     dialect="oracle", target="oracle", values=values)
    if table == ITEMS:
        assert isinstance(verdict, Refusal) and HARVESTED[ITEMS]["code"] in verdict.message
    else:
        assert not isinstance(verdict, Refusal), verdict.message


# --- through the runtime ----------------------------------------------------------------------------------------------


def test_an_answer_over_a_ruled_and_a_free_table_shows_only_the_free_tables_values(runtime, served, answered):
    """End to end: with a row rule on ORDERS for the person and none on ITEMS, every prompt the model is
    sent describes both tables, carries ITEMS' code list, meaning, scheme, description and fact, and none of ORDERS';
    no example reaches it. The agent answers under grants withholding ORDERS, from the snapshot with ORDERS' harvested
    content taken out, and its value index answers ITEMS' literals and none of ORDERS'."""
    fake, model = served
    RuledSuperset(fake, {SHOWN[ORDERS]})
    answer = _ask(runtime)
    assert not answer.failed, answer.answer
    prompts = _prompts(model)
    for word in HARVESTED[ITEMS].values():
        if word != HARVESTED[ITEMS]["question"]:
            assert word in prompts, word
    for word in HARVESTED[ORDERS].values():
        assert word not in prompts, word
    assert f"TABLE {ORDERS}" in prompts and COMMENTS[ORDERS] in prompts
    assert "WORKED EXAMPLES" not in prompts
    [seen] = answered
    assert seen["grants"].values_withheld == {ORDERS}
    assert runtime.value_index.source_id == SOURCE_ID
    assert _column(seen["snapshot"], ORDERS).coded_values == [] and _column(seen["snapshot"], ITEMS).coded_values
    values = seen["values"]
    assert values.contains(ITEMS, "STATUS", HARVESTED[ITEMS]["code"])
    assert not values.has(ORDERS, "STATUS") and values.holders(HARVESTED[ORDERS]["code"]) == set()
    assert not values.contains(ITEMS, "STATUS", OTHER_SOURCES_VALUE)
    assert {call.authorization for call in fake.calls} == {f"Bearer {TOKEN}"}
    assert fake.calls[0].tool == BINDING
    # The grant check's own dry run asked the rule of both tables once; harvested values asked nothing more.
    asked = [sorted(names) for names in _rules_asked(fake)]
    assert asked[0] == ["items", "orders"] and ["items", "orders"] not in asked[1:]


def test_the_fit_and_the_history_read_the_scoped_snapshot(runtime, served, monkeypatch):
    """The prompt fit, which renders the cards again when it trims them, and the history, which decides the newest
    turn's SQL again, are handed the snapshot this request reads, not the served one whole: the fit the one it shows,
    the history the one it decides with."""
    import mnemiq.runtime as runtime_module

    fake, _ = served
    RuledSuperset(fake, {SHOWN[ORDERS]})
    handed: dict[str, Snapshot] = {}

    class Fitter:
        def fit(self, packet, snapshot, grants, card_style=None):
            handed["fit"] = snapshot
            return packet

    real_history = runtime_module.superset_history

    def history(turns, **kwargs):
        handed["history"] = kwargs["snapshot"]
        return real_history(turns, **kwargs)

    monkeypatch.setattr(runtime_module, "superset_history", history)
    runtime.profiles[runtime.default_profile].fitter = Fitter()
    _ask(runtime)
    for reader in ("fit", "history"):
        assert _column(handed[reader], ORDERS).coded_values == [], reader
        assert _column(handed[reader], ITEMS).coded_values, reader
    # The fit shows the cards, so ORDERS' counts are out of its snapshot; the history decides, so they are in its.
    assert _column(handed["fit"], ORDERS).row_count is None and _column(handed["history"], ORDERS).row_count == 10


def _swapping_after_the_grants(monkeypatch, runtime, swapped: Snapshot) -> None:
    """Make the served snapshot `swapped` the moment a request's grants are resolved, as a reload on another thread
    would while this request waits on Superset."""
    import mnemiq.runtime as runtime_module

    real = runtime_module.resolve_grants

    def resolve_then_swap(adapter, snapshot):
        grants = real(adapter, snapshot)
        runtime.snapshot = swapped
        return grants

    monkeypatch.setattr(runtime_module, "resolve_grants", resolve_then_swap)


def test_a_request_reads_one_snapshot_though_another_is_swapped_in_while_its_grants_are_resolved(
        runtime, served, answered, monkeypatch):
    """A snapshot swapped in while the request's grants are asked of Superset is not read by that request: its prompt,
    its value index and every reader are the snapshot its grants were decided for. The swapped one records a lookup on
    ITEMS carrying SUPPLIERS' labels, which no part of this request asked about."""
    fake, model = served
    RuledSuperset(fake, {SHOWN[ORDERS]})
    first = runtime.snapshot
    _swapping_after_the_grants(monkeypatch, runtime, _with_supplier_lookup(first))
    _ask(runtime)
    [seen] = answered
    assert seen["snapshot"].version == first.version
    assert SUPPLIER_LABEL not in _prompts(model) and "SUPPLIER_ID" not in _prompts(model)
    assert seen["values"].has(ITEMS, "STATUS") and not seen["values"].has("SUPPLIERS", "ID")


def test_the_schema_reads_one_snapshot_though_another_is_swapped_in_while_its_grants_are_resolved(
        runtime, served, monkeypatch):
    fake, _ = served
    RuledSuperset(fake, {SHOWN[ORDERS]})
    _swapping_after_the_grants(monkeypatch, runtime, _with_supplier_lookup(runtime.snapshot))
    scope = _scope(runtime)
    assert SUPPLIER_LABEL not in json.dumps(scope) and "SUPPLIER_ID" not in json.dumps(scope)
    assert HARVESTED[ITEMS]["code"] in json.dumps(scope)


def test_a_wrong_literal_on_the_free_table_is_repaired_with_its_values_and_one_on_the_ruled_table_is_not(
        runtime, served, monkeypatch):
    """Through an answer: the writer's filter on a value ITEMS' column does not hold is refused naming the values it
    does hold, and the writer is told them; the same on ORDERS is not checked against the index at all. (The source's
    catalogue is not what this test reads: the adapter answers it as a schema defining no function and no computed
    column.)"""
    for part in ("user_functions", "reachable_user_functions", "virtual_columns"):
        monkeypatch.setattr(SupersetSqlAdapter, part, lambda self: [])
    monkeypatch.setattr(SupersetSqlAdapter, "builtin_functions", lambda self: ["count"])
    fake, model = served
    RuledSuperset(fake, {SHOWN[ORDERS]})
    model.sql = f"SELECT COUNT(*) AS n FROM {ITEMS} WHERE STATUS = 'SHIPPED'"
    _ask(runtime)
    assert f"the real values include: {HARVESTED[ITEMS]['code']}" in _prompts(model)
    model.requests.clear()
    model.sql = f"SELECT COUNT(*) AS n FROM {ORDERS} WHERE STATUS = 'HOLD'"
    _ask(runtime, "how many orders await the desk?")
    prompts = _prompts(model)
    assert "does not appear in" not in prompts and HARVESTED[ORDERS]["code"] not in prompts


def test_the_schema_and_its_starters_show_only_the_free_tables_values(runtime, served):
    """The opening screen's cards and starters: ITEMS' card carries its code list and description, ORDERS' carries
    neither; a starter built from ORDERS' example or code list is not offered."""
    fake, _ = served
    RuledSuperset(fake, {SHOWN[ORDERS]})
    scope = _scope(runtime)
    cards = {card["object_id"]: card["card"] for card in scope["tables"]}
    assert set(cards) == {ORDERS, ITEMS}
    assert HARVESTED[ITEMS]["code"] in cards[ITEMS] and HARVESTED[ITEMS]["description"] in cards[ITEMS]
    for word in HARVESTED[ORDERS].values():
        assert word not in json.dumps(scope), word
    assert HARVESTED[ITEMS]["question"] in scope["starters"]


def test_a_rule_added_between_two_requests_withholds_from_the_next(runtime, served):
    """The rules are asked on each request and kept by none: ITEMS free on the first request, ruled on the second."""
    fake, model = served
    superset = RuledSuperset(fake, set())
    _ask(runtime)
    assert HARVESTED[ORDERS]["code"] in _prompts(model) and HARVESTED[ITEMS]["code"] in _prompts(model)
    model.requests.clear()
    superset.ruled = {SHOWN[ORDERS], SHOWN[ITEMS]}
    _ask(runtime)
    prompts = _prompts(model)
    assert not any(HARVESTED[t]["code"] in prompts for t in (ORDERS, ITEMS))


def test_a_turn_answered_while_a_tables_values_were_shown_is_not_carried_once_they_are_withheld(runtime, served):
    """A follow-up keeps an earlier turn only under the grants it was answered under: one answered while ORDERS' values
    were shown, its SQL quoting one, reaches the next prompt while that holds, and not once a rule applies."""
    from mnemiq.contract import HistoryTurn

    fake, model = served
    superset = RuledSuperset(fake, set())
    first = _ask(runtime)
    turn = HistoryTurn(question="which orders await the desk?",
                       sql=f"SELECT COUNT(*) AS n FROM {ORDERS} WHERE STATUS = '{HARVESTED[ORDERS]['code']}'",
                       grant_fingerprint=first.grant_fingerprint)
    model.requests.clear()
    runtime.ask("and items?", ASKER, history=[turn], delegation=DelegatedToken(TOKEN))
    assert "which orders await the desk?" in _prompts(model)
    model.requests.clear()
    superset.ruled = {SHOWN[ORDERS]}
    runtime.ask("and items?", ASKER, history=[turn], delegation=DelegatedToken(TOKEN))
    prompts = _prompts(model)
    assert "which orders await the desk?" not in prompts and HARVESTED[ORDERS]["code"] not in prompts


def test_a_rule_superset_cannot_tell_of_withholds(runtime, served):
    """Superset wraps both tables, and its rule lookup answers what mnemiq cannot read: both are withheld."""
    fake, model = served
    superset = RuledSuperset(fake, {SHOWN[ORDERS], SHOWN[ITEMS]})
    superset.answer = lambda names: {"tables": [{"table": n, "applies": None} for n in names]}
    _scope(runtime)
    _ask(runtime)
    assert not any(HARVESTED[t]["code"] in _prompts(model) for t in (ORDERS, ITEMS))


def test_values_not_harvested_serve_no_value_index_and_decide_with_the_snapshot_whole(tmp_path, served, answered):
    """An Oracle snapshot whose values were not harvested is decided with whole and served no value index; its ruled
    table is withheld, for what the request shows of it."""
    fake, _ = served
    RuledSuperset(fake, {SHOWN[ORDERS]})
    runtime = _deployment(tmp_path, served, _harvested_snapshot(jobs=[_oracle_job(values="not harvested")]))
    _ask(runtime)
    [seen] = answered
    assert isinstance(seen["values"], NoValueIndex) and seen["grants"].values_withheld == {ORDERS}
    assert seen["snapshot"] is runtime.snapshot


def test_postgresql_superset_mode_is_unchanged(tmp_path, answered):
    """On PostgreSQL nothing is asked of the extension, the snapshot a request reads is the served one, and no value
    index is served, whatever the store holds."""
    with FakeSuperset() as fake, StubModel(sql=f"SELECT COUNT(*) AS n FROM {ITEMS}") as model:
        fake.reply("execute_sql", lambda request, authorization: (
            {"success": True, "rows": None, "columns": None, "error": None, "error_type": None, "statements": []}
            if request.get("dry_run") else
            {"success": True, "columns": [], "rows": []} if request.get("force_refresh") else
            {"success": True, "columns": [{"name": "n", "type": "int64"}], "rows": [{"n": 1}]}))
        snapshot = _harvested_snapshot(jobs=[_source_job()]).model_copy(update={"columns": [
            c.model_copy(update={"object_id": c.object_id.lower()}) for c in _harvested_snapshot().columns]})
        runtime = _deployment(tmp_path, (fake, model), snapshot.model_copy(update={"source_bindings": [
            SourceBinding(id=f"sb:{t}", source_id=SOURCE_ID, object_id=t, source_object=t, binding_type="table")
            for t in ("orders", "items")]}), schema="public")
        runtime.ask("how many items?", ASKER, delegation=DelegatedToken(TOKEN))
        assert {call.tool for call in fake.calls} == {"execute_sql"}
    [seen] = answered
    assert seen["snapshot"] is runtime.snapshot and isinstance(seen["values"], NoValueIndex)
    assert seen["grants"].values_withheld == frozenset()
    assert runtime.value_index is None  # no index is even opened


def _profiled(snapshot: Snapshot) -> Snapshot:
    """`snapshot` with, on each table, an amount column of its own size and a column every row of which is NULL."""
    sizes = {ORDERS: 100, ITEMS: 5}
    extra = [column for table, rows in sizes.items() for column in (
        Column(id=f"{table}.AMOUNT", object_id=table, name="AMOUNT", data_type="NUMBER", semantic_type="amount",
               row_count=rows, distinct_count=rows, null_count=0,
               estimate=ColumnEstimate(rows=rows, distinct=rows, nulls=0)),
        Column(id=f"{table}.NOTE", object_id=table, name="NOTE", data_type="VARCHAR2(40)", row_count=rows,
               distinct_count=0, null_count=rows))]
    return snapshot.model_copy(update={"columns": [*snapshot.columns, *extra]})


def test_a_withheld_tables_profile_is_kept_for_decisions_and_taken_out_of_what_is_shown():
    """A withheld table's counts were read past the rule as its values were: the snapshot a request shows (`shown`)
    takes its row, distinct and null counts and its estimates out, and the one it decides with keeps them, for the
    checks that never render them (the fan-out guard's uniqueness). A shown table keeps its counts in both."""
    snapshot = _profiled(_harvested_snapshot())
    shown = values_scoped(snapshot, _withholding(ORDERS), shown=True)
    held, free = _column(shown, ORDERS, "AMOUNT"), _column(shown, ITEMS, "AMOUNT")
    assert (held.row_count, held.distinct_count, held.null_count, held.estimate) == (None, None, None, None)
    assert free == _column(snapshot, ITEMS, "AMOUNT")
    assert _column(shown, ORDERS, "NOTE").null_count is None
    decided = values_scoped(snapshot, _withholding(ORDERS))
    assert _column(decided, ORDERS, "AMOUNT") == _column(snapshot, ORDERS, "AMOUNT")
    assert values_scoped(snapshot, _withholding(), shown=True) is snapshot
    unharvested = _profiled(_harvested_snapshot(jobs=[_source_job()]))
    assert values_scoped(unharvested, _withholding(ORDERS), shown=True) is unharvested


# A description a person certified, in Verity, for one column of each table: its provenance is known.
CERTIFIED = {ORDERS: "The sales region the order was booked in", ITEMS: "The sales region the item ships to"}


def _unharvested(snapshot: Snapshot | None = None) -> Snapshot:
    """`snapshot` (by default the two tables, profiled) as enriched from Oracle with no value harvested: each table's
    status column keeps a code list of one certified code and one grounded from rows, and each table has a REGION
    column whose description and code list were certified."""
    base = snapshot or _profiled(_harvested_snapshot(jobs=[_oracle_job(values="not harvested")]))
    columns = []
    for column in base.columns:
        if column.name == "STATUS":
            column = column.model_copy(update={"coded_values": [
                CodedValue(code="C1", meaning="certified status", source="certified"),
                CodedValue(code=HARVESTED[column.object_id]["code"], meaning=HARVESTED[column.object_id]["meaning"],
                           source="correlated")]})
        columns.append(column)
    regions = [Column(id=f"{t}.REGION", object_id=t, name="REGION", data_type="VARCHAR2(10)") for t in (ORDERS, ITEMS)]
    stated = [_certified(f"{t}.REGION", description=CERTIFIED[t],
                         coded_values=[CodedValue(code="W", meaning="west")]) for t in (ORDERS, ITEMS)]
    return apply_certified(base.model_copy(update={"columns": [*columns, *regions]}), stated)


def _certified(column_id: str, **stated) -> CertifiedRecord:
    """A certified record of the column `column_id` stating exactly the fields given."""
    table, _, name = column_id.partition(".")
    return CertifiedRecord(envelope=RecordEnvelope(object_type="column", object_id=column_id, version="v1",
                                                   source_system="verity"),
                           payload=Column(id=column_id, object_id=table, name=name, **stated))


def test_where_no_value_was_harvested_a_withheld_tables_written_meaning_and_counts_are_not_shown():
    """Profiling read every row, and the model wrote each description, table fact and example from a card carrying the
    counts: for an Oracle snapshot that harvested no values, what a request shows leaves out a withheld table's counts,
    estimates, model-written column descriptions, code schemes, table facts and the examples reading it, and keeps what
    a person certified (a certified column's description, certified codes). A shown table keeps everything, and the
    snapshot the request decides with is the served one."""
    snapshot = _unharvested()
    shown = values_scoped(snapshot, _withholding(ORDERS), shown=True)
    held = _column(shown, ORDERS, "AMOUNT")
    assert (held.row_count, held.distinct_count, held.null_count, held.estimate) == (None, None, None, None)
    status = _column(shown, ORDERS)
    assert status.description is None and status.code_scheme is None
    assert [cv.code for cv in status.coded_values] == ["C1"]  # the certified code; the one read from rows is out
    region = _column(shown, ORDERS, "REGION")
    assert region.description == CERTIFIED[ORDERS] and [cv.code for cv in region.coded_values] == ["W"]
    assert [tf.object_id for tf in shown.table_facts] == [ITEMS]
    assert [e.object_id for e in shown.examples] == [ITEMS]
    for name in ("AMOUNT", "NOTE", "STATUS", "REGION"):
        assert _column(shown, ITEMS, name) == _column(snapshot, ITEMS, name), name
    assert values_scoped(snapshot, _withholding(ORDERS)) is snapshot
    assert values_scoped(snapshot, _withholding(), shown=True) is snapshot


def test_a_certified_record_marks_the_fields_it_states():
    """`apply_certified` records on the column which meaning fields its record stated, and only those."""
    snapshot = apply_certified(_unharvested(), [_certified(f"{ORDERS}.STATUS", semantic_type="code")])
    assert _column(snapshot, ORDERS).certified_fields == ["semantic_type"]
    assert _column(snapshot, ORDERS, "REGION").certified_fields == ["coded_values", "description"]
    assert _column(snapshot, ITEMS).certified_fields is None
    # A level stated as null is no statement (`apply_certified` keeps the column's own): not marked.
    unset = apply_certified(_unharvested(), [_certified(f"{ITEMS}.STATUS", semantic_type="code", pii_level=None)])
    assert _column(unset, ITEMS).certified_fields == ["semantic_type"]


def test_a_description_the_model_writes_over_a_marked_column_is_no_longer_marked():
    """The mark names only what a certified record stated and the column still holds: a description, type and level
    the model writes afterwards are not marked as the record's, and on a snapshot that harvested no values a ruled
    table's model-written description is withheld."""
    from mnemiq.enrichment.enricher import FakeEnricher
    from mnemiq.enrichment.semantic import enrich_semantic

    laid = apply_certified(_unharvested(), [_certified(
        f"{ORDERS}.STATUS", description="Certified status", semantic_type="code", pii_level="none",
        code_scheme=CodeScheme(id="scheme-cert", label="Certified scheme"))])
    assert _column(laid, ORDERS).certified_fields == ["code_scheme", "description", "pii_level", "semantic_type"]
    reply = json.dumps({"columns": [{"name": "STATUS", "description": "Written by the model", "semantic_type": "code",
                                     "pii_level": "none"}]})
    described = enrich_semantic(laid, FakeEnricher({ORDERS: reply}))
    status = _column(described, ORDERS)
    assert status.description == "Written by the model" and status.certified_fields == ["code_scheme"]
    shown = values_scoped(described, _withholding(ORDERS), shown=True)
    assert _column(shown, ORDERS).description is None


def test_codes_the_model_drops_are_no_longer_marked():
    """A column the model reads as a measure loses its code list; the mark no longer names the codes."""
    from mnemiq.enrichment.enricher import FakeEnricher
    from mnemiq.enrichment.semantic import enrich_semantic

    laid = apply_certified(_unharvested(), [_certified(f"{ORDERS}.STATUS", coded_values=[CodedValue(code="C1")])])
    assert _column(laid, ORDERS).certified_fields == ["coded_values"]
    reply = json.dumps({"columns": [{"name": "STATUS", "description": "A measure", "semantic_type": "amount"}]})
    status = _column(enrich_semantic(laid, FakeEnricher({ORDERS: reply})), ORDERS)
    assert status.coded_values == [] and status.certified_fields == []


@pytest.mark.parametrize("own, stated, laid, marked", [
    ("pii", "phi", "phi", True),
    ("pii", "none", "pii", False),  # an unattested record may not lower a level: the column's own is kept
    ("none", "personal", "pii", False),  # a level outside the vocabulary is read as the most sensitive
], ids=["laid as stated", "a refused downgrade", "an unrecognised level"])
def test_the_mark_names_a_level_only_where_the_records_level_was_laid(own, stated, laid, marked):
    snapshot = _unharvested()
    held = snapshot.model_copy(update={"columns": [
        c.model_copy(update={"pii_level": own}) if c.id == f"{ITEMS}.STATUS" else c for c in snapshot.columns]})
    out = apply_certified(held, [_certified(f"{ITEMS}.STATUS", pii_level=stated)])
    column = _column(out, ITEMS)
    assert column.pii_level == laid
    assert ("pii_level" in column.certified_fields) is marked


def test_a_description_a_certified_record_did_not_state_is_withheld_with_the_rest():
    """A record certifying a ruled column's type alone vouches for no description: the one the model wrote stays out
    of what is shown, beside the counts. The same record on a free table leaves its description shown."""
    snapshot = apply_certified(_unharvested(), [_certified(f"{t}.STATUS", semantic_type="code")
                                                for t in (ORDERS, ITEMS)])
    shown = values_scoped(snapshot, _withholding(ORDERS), shown=True)
    assert _column(shown, ORDERS).description is None and _column(shown, ORDERS).row_count is None
    assert _column(shown, ITEMS).description == HARVESTED[ITEMS]["description"]


def test_a_snapshot_without_the_marker_withholds_a_certified_columns_description():
    """A snapshot written before the marker records no field a record stated: a ruled column's description is withheld
    though a certified reference names the column."""
    marked = _unharvested()
    unmarked = marked.model_copy(update={"columns": [column.model_copy(update={"certified_fields": None})
                                                      for column in marked.columns]})
    assert any(ref.object_id == f"{ORDERS}.REGION" for ref in unmarked.certified_refs)
    shown = values_scoped(unmarked, _withholding(ORDERS), shown=True)
    assert _column(shown, ORDERS, "REGION").description is None
    assert _column(shown, ITEMS, "REGION").description == CERTIFIED[ITEMS]


def test_a_later_run_laying_a_record_without_a_description_withholds_the_earlier_runs():
    """`enrich --add`: an earlier run's model-written description is kept by the merge, and a record certified since,
    stating the column's type alone, is laid over the merged snapshot again: the description is still the model's, so
    a ruled table's stays out of what is shown."""
    from mnemiq.enrichment.accumulate import merge_run, reapply_overlays

    earlier = _unharvested()
    run = earlier.model_copy(update={
        "columns": [c for c in earlier.columns if c.object_id == ITEMS],
        "source_bindings": [b for b in earlier.source_bindings if b.object_id == ITEMS],
        "table_facts": [], "examples": []})
    merged, _ = merge_run(earlier, run)
    laid = reapply_overlays(merged, CertifiedSet(records=[_certified(f"{ORDERS}.STATUS", semantic_type="code")]),
                            None)
    assert _column(laid, ORDERS).description == HARVESTED[ORDERS]["description"]  # kept, and the model's
    assert any(ref.object_id == f"{ORDERS}.STATUS" for ref in laid.certified_refs)
    shown = values_scoped(laid, _withholding(ORDERS), shown=True)
    assert _column(shown, ORDERS).description is None


def test_a_snapshot_written_without_the_marker_keeps_its_version():
    """The marker is hashed only where set: a snapshot no record marked keeps the version it had."""
    from mnemiq.enrichment.pipeline import content_version

    assert content_version(_harvested_snapshot()) == "adac94bece37"
    marked = _harvested_snapshot().model_copy(update={"columns": [
        c.model_copy(update={"certified_fields": ["description"]}) if c.id == f"{ORDERS}.STATUS" else c
        for c in _harvested_snapshot().columns]})
    assert content_version(marked) != "adac94bece37"


def test_a_ruled_columns_description_no_record_stated_reaches_no_card_or_prompt(tmp_path, served, answered):
    """End to end, values not harvested, a rule on ORDERS, and a record certifying each table's status type alone:
    neither the schema nor the writer's prompt carries ORDERS' model-written description; ITEMS' carry its own."""
    fake, model = served
    RuledSuperset(fake, {SHOWN[ORDERS]})
    snapshot = apply_certified(_unharvested(), [_certified(f"{t}.STATUS", semantic_type="code")
                                                for t in (ORDERS, ITEMS)])
    runtime = _deployment(tmp_path, served, snapshot)
    cards = {card["object_id"]: card["card"] for card in _scope(runtime)["tables"]}
    _ask(runtime)
    [request] = [r for r in model.requests if "QUESTION:" in json.dumps(r)][:1]
    prompt = request["messages"][-1]["content"]
    for orders, items in ((cards[ORDERS], cards[ITEMS]), (_card_of(prompt, ORDERS), _card_of(prompt, ITEMS))):
        assert HARVESTED[ORDERS]["description"] not in orders and CERTIFIED[ORDERS] in orders
        assert HARVESTED[ITEMS]["description"] in items


def test_where_values_were_harvested_a_withheld_tables_certified_meaning_is_withheld_too():
    """With values harvested the rule is as it was: a withheld table's code lists, schemes and descriptions are taken
    out, certified ones included."""
    snapshot = _unharvested(_profiled(_harvested_snapshot()))
    shown = values_scoped(snapshot, _withholding(ORDERS), shown=True)
    region = _column(shown, ORDERS, "REGION")
    assert region.description is None and region.coded_values == []
    assert _column(shown, ITEMS, "REGION") == _column(snapshot, ITEMS, "REGION")


def test_a_ruled_tables_written_meaning_reaches_no_card_where_no_value_was_harvested(tmp_path, served, answered):
    """End to end, with values not harvested and a rule on ORDERS: neither the schema nor the writer's prompt carries
    ORDERS' model-written description or table fact, both carry its certified description, and ITEMS' cards carry its
    own, its table fact on the prompt's; the agent decides with the served snapshot."""
    fake, model = served
    RuledSuperset(fake, {SHOWN[ORDERS]})
    runtime = _deployment(tmp_path, served, _unharvested())
    cards = {card["object_id"]: card["card"] for card in _scope(runtime)["tables"]}
    _ask(runtime)
    [request] = [r for r in model.requests if "QUESTION:" in json.dumps(r)][:1]
    prompt = request["messages"][-1]["content"]
    for orders, items in ((cards[ORDERS], cards[ITEMS]), (_card_of(prompt, ORDERS), _card_of(prompt, ITEMS))):
        assert HARVESTED[ORDERS]["description"] not in orders and HARVESTED[ORDERS]["fact"] not in orders
        assert CERTIFIED[ORDERS] in orders
        assert HARVESTED[ITEMS]["description"] in items and CERTIFIED[ITEMS] in items
    # Table facts ride on a retrieved card, not on the schema's.
    assert HARVESTED[ITEMS]["fact"] in _card_of(prompt, ITEMS)
    [seen] = answered
    assert seen["snapshot"] is runtime.snapshot


def _card_of(text: str, table: str) -> str:
    """The part of a prompt or schema text that is `table`'s card."""
    return next(part for part in text.split("TABLE ")[1:] if part.startswith(f"{table}\n"))


@pytest.mark.parametrize("values", ["harvested", "not harvested"])
def test_a_ruled_tables_profile_reaches_no_card_starter_or_prompt_and_still_decides(tmp_path, served, answered, values):
    """With a rule on ORDERS: ORDERS' card says nothing its counts say (no ENTIRELY NULL note), ITEMS' card does; the
    opening screen's starter about the biggest table's amount is ITEMS', since ORDERS' size is not told; and the agent
    decides with ORDERS' counts still there. With no rule, the starter is about ORDERS."""
    fake, model = served
    superset = RuledSuperset(fake, {SHOWN[ORDERS]})
    runtime = _deployment(tmp_path, served, _profiled(_harvested_snapshot(jobs=[_oracle_job(values=values)])))
    for listed in (_scope(runtime)["tables"], runtime.schema(ASKER, delegation=DelegatedToken(TOKEN))):
        cards = {card["object_id"]: card["card"] for card in listed}
        assert "ENTIRELY NULL" not in cards[ORDERS] and "ENTIRELY NULL" in cards[ITEMS]
    assert "What is the total AMOUNT in ITEMS?" in _scope(runtime)["starters"]
    _ask(runtime)
    [request] = [r for r in model.requests if "QUESTION:" in json.dumps(r)][:1]
    prompt = request["messages"][-1]["content"]
    assert "ENTIRELY NULL" not in _card_of(prompt, ORDERS) and "ENTIRELY NULL" in _card_of(prompt, ITEMS)
    [seen] = answered
    assert _column(seen["snapshot"], ORDERS, "AMOUNT").row_count == 100
    superset.ruled = set()
    assert "What is the total AMOUNT in ORDERS?" in _scope(runtime)["starters"]


def test_only_superset_mode_reads_a_scoped_snapshot():
    """Direct mode's snapshot is returned as it is; Superset mode reads the scoped one."""
    from types import SimpleNamespace

    from mnemiq.runtime import Runtime

    snapshot = _harvested_snapshot()
    grants = GrantSet(frozenset({ORDERS, ITEMS}), row_filters={ORDERS: "1 = 1"}, values_withheld=frozenset({ORDERS}))
    fields = {"con": None, "snapshot": snapshot, "adapter": None, "agent": None, "embedder": None, "authz": None}
    assert Runtime(**fields, settings=None)._served(grants, snapshot) is snapshot
    assert Runtime(**fields, settings=SimpleNamespace(superset_mode=False))._served(grants, snapshot) is snapshot
    scoped = Runtime(**fields, settings=SimpleNamespace(superset_mode=True))._served(grants, snapshot)
    assert _column(scoped, ORDERS).coded_values == [] and _column(scoped, ITEMS).coded_values


# --- every reader ----------------------------------------------------------------------------------------------------

SOURCE = Path(mnemiq.__file__).parent


def _modules_reading(attribute: str) -> set[str]:
    return {path.relative_to(SOURCE).as_posix() for path in SOURCE.rglob("*.py")
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Attribute) and node.attr == attribute and isinstance(node.ctx, ast.Load)}


# The modules that read each field `values_scoped` takes out of a withheld table, by its attribute's name. A new reader
# fails `test_every_reader_of_a_stripped_field_is_known` until it is shown to read the snapshot a request reads (or to
# read the field of something else: `description` is also a measure's, a setting's and a cursor's), and is added here.
FIELD_READERS = {
    "coded_values": {
        # enrichment and the ontology binder, offline, over the source
        "enrichment/accumulate.py", "enrichment/certified.py", "enrichment/grounding.py", "enrichment/semantic.py",
        "enrichment/statistics.py", "ontology/binder.py",
        # at answer time: the renderers the runtime hands the request's snapshot, and the scoping that makes it
        "semantic/cards.py", "semantic/starters.py", "superset/harvested.py",
    },
    "code_scheme": {
        "enrichment/grounding.py", "ontology/binder.py", "semantic/ontology_index.py",  # offline
        "semantic/cards.py", "semantic/retrieval.py",  # handed the request's snapshot (and its columns)
        "superset/harvested.py",  # the scoping
    },
    "description": {
        "enrichment/facts.py", "enrichment/grounding.py", "enrichment/semantic.py", "ontology/digest.py",  # offline
        "semantic/cards.py", "semantic/fit.py",  # a column's, from the request's snapshot
        "superset/harvested.py",  # the scoping
        # another object's: a cursor's (`adapters`), a setting's (`config.py`), a Superset measure's
        "adapters/oracle.py", "adapters/pg.py", "adapters/sqlite.py", "config.py", "server/superset_build.py",
        "superset/build_store.py", "superset/measure_actions.py", "superset/measures.py", "superset/planner.py",
    },
    "table_facts": {
        "eval/engine.py", "semantic/federation.py",  # eval and the federated merge, never a Superset-mode request
        "llm/window.py",  # the prompt-window advisory, an operator's log
        "runtime.py", "semantic/fit.py", "superset/harvested.py",  # the request's snapshot
    },
    "examples": {
        "cli.py", "enrichment/accumulate.py", "semantic/federation.py", "semantic/store.py", "store/federated_build.py",
        "generate/prompts.py",  # a packet's, which Superset mode empties
        "semantic/starters.py", "superset/harvested.py",  # the request's snapshot
    },
    # A column's counts and estimates: profiling and enrichment offline; the fan-out guard, which decides and never
    # shows; the card and the starters, handed the snapshot a request shows. `row_count` is also a result's (an answer's
    # preview, as the MCP door, a trace and the server serialise it).
    "row_count": {
        "enrichment/joins.py", "enrichment/pipeline.py", "enrichment/profiling.py", "enrichment/prompts.py",
        "enrichment/semantic.py", "sql/fanout_check.py", "semantic/cards.py", "semantic/starters.py",
        "mcp/server.py", "observability/trace_sink.py", "server/serialize.py", "server/superset_conversations.py",
    },
    "distinct_count": {
        "enrichment/grounding.py", "enrichment/joins.py", "enrichment/pipeline.py", "enrichment/profiling.py",
        "enrichment/prompts.py", "enrichment/semantic.py", "ontology/binder.py", "semantic/values.py",
        "sql/fanout_check.py",
    },
    "null_count": {
        "enrichment/joins.py", "enrichment/pipeline.py", "enrichment/profiling.py", "enrichment/prompts.py",
        "enrichment/semantic.py", "sql/fanout_check.py", "semantic/cards.py",
    },
    "estimate": {
        "enrichment/pipeline.py", "enrichment/semantic.py", "enrichment/statistics.py", "sql/fanout_check.py",
    },
}


@pytest.mark.parametrize("field", sorted(FIELD_READERS))
def test_every_reader_of_a_stripped_field_is_known(field):
    assert _modules_reading(field) == FIELD_READERS[field]


def _functions(tree: ast.Module):
    """The module's functions and its classes' methods."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
        elif isinstance(node, ast.ClassDef):
            yield from (item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))


def _is_snapshot_of(node, holder: str) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "snapshot" and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name) and node.value.id == holder)


def _snapshot_reads(source: str) -> dict[str, int]:
    """For each function or method of `source`, how many times it reads `self.snapshot`, however the read is used: an
    argument, a part of one (`self.snapshot.table_facts if self.snapshot else ()`), an alias, a condition."""
    reads: dict[str, int] = {}
    for function in _functions(ast.parse(source)):
        if count := sum(_is_snapshot_of(node, "self") for node in ast.walk(function)):
            reads[function.name] = reads.get(function.name, 0) + count
    return reads


def _callee(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "?")


def _snapshot_hand_offs(source: str) -> set[tuple[str, str]]:
    """Each read of another object's snapshot (`runtime.snapshot`, `asking.snapshot`, ...) in `source`, as the object's
    name and the function of the nearest call holding it as an argument, or part of one; "<not an argument>" for a read
    no call takes (an alias, a return, a comparison)."""
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr == "snapshot" and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name) and node.value.id != "self"):
            continue
        taker, child, parent = "<not an argument>", node, parents.get(node)
        while parent is not None and not isinstance(parent, ast.stmt):
            if isinstance(parent, ast.Call) and child is not parent.func:
                taker = _callee(parent)
                break
            child, parent = parent, parents.get(parent)
        found.add((node.value.id, taker))
    return found


@pytest.mark.parametrize(("source", "reads"), [
    ("def ask(self):\n    retrieve(table_facts=self.snapshot.table_facts if self.snapshot else ())\n", {"ask": 2}),
    ("def _cards(self, x):\n    build_cards(self.snapshot if x else None)\n", {"_cards": 1}),
    ("def scope(self):\n    snap = self.snapshot\n    compose_starters(snap)\n", {"scope": 1}),
    ("class R:\n    def fit(self, p):\n        return fitter.fit(p, self.snapshot)\n", {"fit": 1}),
    ("def swap(self, s):\n    self.snapshot = s\n", {}),
    ("def scope(self, snapshot):\n    compose_starters(snapshot, other.snapshot)\n", {}),
], ids=["inside a conditional argument", "a conditional argument", "an alias", "a method", "a store", "not self's"])
def test_the_read_counter_counts_every_way_the_snapshot_is_read(source, reads):
    assert _snapshot_reads(source) == reads


@pytest.mark.parametrize(("source", "found"), [
    ("run_decided(sql, snapshot=runtime.snapshot)\n", {("runtime", "run_decided")}),
    ("plan(x if y else runtime.snapshot)\n", {("runtime", "plan")}),
    ("plan(only(runtime.snapshot.columns))\n", {("runtime", "only")}),
    ("s = runtime.snapshot\nplan(s)\n", {("runtime", "<not an argument>")}),
    ("def f():\n    return asking.snapshot\n", {("asking", "<not an argument>")}),
    ("runtime.snapshot.model_copy()\n", {("runtime", "<not an argument>")}),
    ("plan(self.snapshot, snapshot)\n", set()),
], ids=["a keyword", "a conditional", "a part of one", "an alias", "a return", "its own method", "not another's"])
def test_the_hand_off_finder_finds_every_way_a_snapshot_is_handed_on(source, found):
    assert _snapshot_hand_offs(source) == found


# Where `runtime.py` reads the snapshot it serves, and how often. Each request's door reads it once, after the reload
# (`_reloaded`, and in direct mode `_grants_for` and `write`, which do not reload), and hands that one reference to
# its grants, its value index and its scoping; the rest is bookkeeping read by no person: the source id answers are
# recorded under, the certified versions a trace names, and the binding check's own record when it is asked with no
# snapshot of a request (`bound`).
SNAPSHOT_READS = {"_reloaded": 1, "_grants_for": 1, "write": 1, "bound": 1, "_source_id": 3, "_record": 2}


def test_the_runtime_reads_its_served_snapshot_only_where_known():
    """A new read of `self.snapshot` in `runtime.py`, or a second one in a door, fails this: a request reads one
    snapshot, so a swap while it runs never gives its grants one snapshot and its readers another."""
    assert _snapshot_reads((SOURCE / "runtime.py").read_text()) == SNAPSHOT_READS


# What the server and the builder hand another object's snapshot to. The server reads none of the runtime's: its build
# routes take the snapshot `builder_grants` returns, the one their grants were resolved against. The builder hands the
# snapshot it was given (a run's, a measure change's) to the chart and statement checks, none of which reads a field
# `values_scoped` takes out (pinned above, by module). A request in Superset mode is answered from the snapshot
# `Runtime._served` scopes; a build reads the request's whole, its names, links and key facts.
SNAPSHOT_HAND_OFFS = {
    # The build routes plan, step, check and run again from the snapshot `builder_grants` resolved their grants against,
    # never from a second read of the runtime's.
    "server": set(),
    "superset": {("run", "check_chart"), ("asking", "visible_schema"), ("asking", "check_chart"),
                 ("asking", "check_measure"), ("asking", "run_decided")},
}


@pytest.mark.parametrize("package", sorted(SNAPSHOT_HAND_OFFS))
def test_the_server_and_the_builder_hand_the_snapshot_only_to_known_non_readers(package):
    found = set().union(*(_snapshot_hand_offs(path.read_text()) for path in (SOURCE / package).glob("*.py")))
    assert found == SNAPSHOT_HAND_OFFS[package]


# Where a value index is opened: direct mode's components, and the runtime, for its advisories and, in Superset mode,
# once at boot, the reader `served_values` hands each request.
INDEX_OPENERS = {"assembly.py", "runtime.py"}


def test_the_value_index_is_opened_only_where_known():
    opened = {path.relative_to(SOURCE).as_posix() for path in SOURCE.rglob("*.py")
              for node in ast.walk(ast.parse(path.read_text()))
              if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ValueIndex"}
    assert opened == INDEX_OPENERS


def test_the_scoping_and_the_index_ask_the_one_predicate():
    """`values_scoped` and every read of `ShownValues` decide by `values_shown`, the one predicate."""
    tree = ast.parse((SOURCE / "superset/harvested.py").read_text())
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    def calls(function, name: str) -> bool:
        return any(isinstance(n, ast.Call) and _callee(n) == name for n in ast.walk(functions[function]))

    assert calls("_shown_tables", "values_shown")
    assert calls("_shown", "values_shown")
    for read in ("has", "contains", "nearest", "values", "holders"):
        assert calls(read, "_shown"), read
