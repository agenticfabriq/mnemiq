"""M144: certified metrics, worked examples and code concepts reach the prompt by COLUMN, not table.

Each was selected by table grant alone. The card for `orders` withheld `secret_code` (phi) from an
identity without phi clearance, and then the prompt named it anyway: a metric's `count(secret_code)`,
an example's `SELECT secret_code`, the column's code vocabulary. The decider would still refuse
SQL reading it, so the leak was metadata -- the column's existence, its use and its codes -- to a
model and a caller the card had already decided should not see them.

The rule is the one a certified dimension already meets (M136): offered only where the identity
reads every column it reads raw.
"""

from __future__ import annotations

import duckdb

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import CodeScheme, Column, Example, IdentityContext, Metric, Snapshot
from mnemiq.contract.values import MeasureExpr
from mnemiq.llm.embeddings import FakeEmbedder
from mnemiq.semantic.measures import select_metrics
from mnemiq.semantic.ontology_index import OntologyIndex
from mnemiq.semantic.retrieval import _resolve_concepts, _retrieve_examples, retrieve
from mnemiq.semantic.store import build_example_index, build_index
from mnemiq.store.bootstrap import init_store

_SCHEME = CodeScheme(id="urn:icd10", label="ICD-10-CM")
_COLUMNS = [
    Column(id="orders.id", object_id="orders", name="id", data_type="int"),
    Column(id="orders.secret_code", object_id="orders", name="secret_code", data_type="text",
           pii_level="phi", code_scheme=_SCHEME),
]
_UNCLEARED = GrantSet(frozenset({"orders"}))
_CLEARED = GrantSet(frozenset({"orders"}), pii_clearance=frozenset({"phi"}))


def _metric(expr: str) -> Metric:
    return Metric(id=f"m:{expr}", label=expr, status="certified", owner="o", grain="order",
                  time_dimension="orders.id", measure=MeasureExpr(expr=expr, source="orders"))


_SECRET_METRIC, _PLAIN_METRIC = _metric("count(secret_code)"), _metric("count(id)")
_EXAMPLES = [
    Example(question="how many orders", sql="SELECT count(id) FROM orders",
            tables=["orders"], object_id="orders"),
    Example(question="how many orders by code", sql="SELECT o.secret_code, count(*) FROM orders o "
            "GROUP BY o.secret_code", tables=["orders"], object_id="orders"),
]


def _snapshot() -> Snapshot:
    return Snapshot(version="v1", source_id="s", created_at="t", columns=_COLUMNS,
                    examples=_EXAMPLES)


def test_a_metric_reading_a_column_the_identity_cannot_read_raw_is_not_offered():
    offered = select_metrics(["orders"], [_SECRET_METRIC, _PLAIN_METRIC], _UNCLEARED,
                             columns=_COLUMNS)
    assert offered == [_PLAIN_METRIC]
    assert select_metrics(["orders"], [_SECRET_METRIC, _PLAIN_METRIC], _CLEARED,
                          columns=_COLUMNS) == [_SECRET_METRIC, _PLAIN_METRIC]


def _examples_store(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    embedder = FakeEmbedder()
    build_example_index(con, _snapshot(), embedder)
    (qvec,) = embedder.embed(["how many orders"])
    return con, qvec


def test_an_example_reading_a_withheld_column_is_not_shown_through_its_alias(tmp_path):
    """The column is read through a table alias, which the decider's own resolution follows."""
    con, qvec = _examples_store(tmp_path)
    shown = _retrieve_examples(con, qvec, {"orders"}, k=5,
                               not_raw=frozenset({("orders", "secret_code")}))
    assert [e.sql for e in shown] == ["SELECT count(id) FROM orders"]
    assert len(_retrieve_examples(con, qvec, {"orders"}, k=5, not_raw=frozenset())) == 2


def _concept_index() -> OntologyIndex:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE ontology_concept (source_id TEXT, scheme_id TEXT, notation TEXT, "
                "label TEXT, definition TEXT)")
    con.execute("INSERT INTO ontology_concept VALUES "
                "('s','urn:icd10','E11','Type 2 diabetes mellitus','')")
    return OntologyIndex(con)


def test_a_code_vocabulary_on_a_withheld_column_is_not_resolved():
    question = "how many with type 2 diabetes"
    assert _resolve_concepts(question, _COLUMNS, {"orders"}, _concept_index(), _UNCLEARED) == []
    assert _resolve_concepts(question, _COLUMNS, {"orders"}, _concept_index(), _CLEARED)


class _Authz:
    def __init__(self, grants: GrantSet):
        self._grants = grants

    def grants_for(self, identity):
        return self._grants


def test_retrieve_scopes_all_three_to_the_identity(tmp_path):
    """The join: `retrieve` is what a deployment calls, and each selector must be handed what
    it needs to apply the rule -- a selector that is right in isolation and never given the
    identity's columns leaves the prompt as it was."""
    con = init_store(str(tmp_path / "s.duckdb"))
    embedder = FakeEmbedder()
    build_index(con, _snapshot(), embedder)
    build_example_index(con, _snapshot(), embedder)
    con.execute("CREATE TABLE ontology_concept (source_id TEXT, scheme_id TEXT, notation TEXT, "
                "label TEXT, definition TEXT)")
    con.execute("INSERT INTO ontology_concept VALUES "
                "('s','urn:icd10','E11','Type 2 diabetes mellitus','')")

    def ask(grants: GrantSet):
        return retrieve(con, "how many orders with type 2 diabetes", IdentityContext(
                            tenant_id="t", principal_id="p", roles=["r"]),
                        _Authz(grants), embedder, metrics=[_SECRET_METRIC, _PLAIN_METRIC],
                        columns=_COLUMNS, ontology_index=OntologyIndex(con), snapshot=_snapshot())

    uncleared, cleared = ask(_UNCLEARED), ask(_CLEARED)
    assert [m.id for m in uncleared.metrics] == [_PLAIN_METRIC.id]
    assert all("secret_code" not in e.sql for e in uncleared.examples)
    assert uncleared.concepts == []
    # The control: a cleared identity is shown all three, so the above is the rule, not an outage.
    assert len(cleared.metrics) == 2
    assert any("secret_code" in e.sql for e in cleared.examples)
    assert cleared.concepts


def test_an_example_whose_columns_cannot_be_read_is_not_shown_and_says_so(caplog):
    """Fail closed: SQL neither dialect parses cannot be checked, so it is not shown -- but only
    when something is withheld; with nothing to withhold there is nothing to check. The log line
    is how an operator learns why an example is missing from some identities' prompts."""
    import logging

    from mnemiq.semantic.retrieval import _reads_only_raw

    broken = "SELEC secret_code FRM orders"
    with caplog.at_level(logging.INFO, logger="mnemiq.semantic.retrieval"):
        assert _reads_only_raw(broken, frozenset({("orders", "secret_code")})) is False
    assert any("withheld" in r.getMessage() and r.levelno >= logging.INFO
               for r in caplog.records), [r.getMessage() for r in caplog.records]
    assert _reads_only_raw(broken, frozenset()) is True
