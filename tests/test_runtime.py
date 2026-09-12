import logging

import pytest

from mnemiq.authz.grants import DenyAll, GrantSet
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext, JoinKey, Relationship, Snapshot
from mnemiq.runtime import Runtime, SnapshotMissing, _warn_policy_advisories, build_runtime


def _identity():
    return IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


def test_build_runtime_without_a_snapshot_is_a_clear_error(tmp_path):
    # empty store, no snapshot -> actionable error, not a crash (and no LLM client built)
    s = Settings(
        llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="x", acme_data_dir=None,
        store_path=str(tmp_path / "empty.duckdb"),
    )
    with pytest.raises(SnapshotMissing):
        build_runtime(s)


class _StaticAuthz:
    def __init__(self, *objects):
        self._g = GrantSet(frozenset(objects))

    def grants_for(self, _identity):
        return self._g


class _RoledAuthz:
    """Enumerates one role (like FileAuthzProvider) and returns a fixed grant for it, so the
    boot advisory has a role to walk."""

    def __init__(self, role, grant):
        self._role, self._grant = role, grant

    def policy_roles(self):
        return [self._role]

    def grants_for(self, _identity):
        return self._grant


def _snapshot_with(relationships):
    return Snapshot(version="v1", source_id="s", created_at="2026-01-01T00:00:00Z",
                    relationships=relationships)


def test_boot_advisory_warns_about_an_off_vocabulary_clearance_without_a_snapshot(caplog):
    # The wiring restructure must run the clearance check even when no snapshot is available:
    # a clearance value's validity does not depend on the snapshot.
    authz = _RoledAuthz(
        "hq_analyst", GrantSet(frozenset({"customer"}), pii_clearance=frozenset({"high"}))
    )

    with caplog.at_level(logging.WARNING):
        _warn_policy_advisories(authz, None)

    assert "hq_analyst" in caplog.text and "high" in caplog.text


def test_boot_advisory_still_reports_row_filter_holes_when_a_snapshot_is_present(caplog):
    # And the row-filter advisory the restructure sits beside must keep firing when it can.
    authz = _RoledAuthz(
        "store1", GrantSet(frozenset({"customer", "payment"}),
                           row_filters={"customer": "store_id = 1"})
    )
    snap = _snapshot_with([
        Relationship(id="payment->customer", **{"from": "payment"}, to="customer",
                     cardinality="many_to_one", join_keys=[JoinKey(left="k", right="k")]),
    ])

    with caplog.at_level(logging.WARNING):
        _warn_policy_advisories(authz, snap)

    assert "payment" in caplog.text  # the unfiltered dependent is still named


class _Con:
    """Minimal store stand-in: schema() reads semantic_object."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        self._last = (sql, params)
        return self

    def fetchall(self):
        return self._rows


def test_schema_returns_only_granted_objects():
    con = _Con([("claim", "TABLE claim ..."), ("party", "TABLE party ...")])
    rt = Runtime(con=con, snapshot=None, adapter=None, agent=None, embedder=None,
                 authz=_StaticAuthz("claim"), settings=None)
    got = rt.schema(_identity())
    assert got == [{"object_id": "claim", "card": "TABLE claim ..."}]


def test_schema_is_empty_under_denyall():
    con = _Con([("claim", "c")])
    rt = Runtime(con=con, snapshot=None, adapter=None, agent=None, embedder=None,
                 authz=DenyAll(), settings=None)
    assert rt.schema(_identity()) == []


def test_ask_retrieves_scoped_and_delegates_to_the_agent(monkeypatch):
    # ask wires retrieve -> agent.answer; verify with fakes, no store/LLM
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.contract import Snapshot
    from mnemiq.semantic.retrieval import ContextPacket

    calls = {}

    def fake_retrieve(con, question, identity, authz, embedder, k=5, table_facts=(), **kwargs):
        calls["retrieve"] = (question, k, list(table_facts))
        return ContextPacket(question=question, cards=[], grant_fingerprint="fp",
                             enrichment_version="v1")

    class _Agent:
        def answer(self, packet, snapshot, grants, identity, emit=None):
            calls["answer"] = (packet, snapshot, grants.objects)
            return AgentAnswer(answer="ANSWER")

    snap = Snapshot(version="v1", source_id="acme", created_at="t")
    monkeypatch.setattr(rt_mod, "retrieve", fake_retrieve)
    rt = Runtime(con=None, snapshot=snap, adapter=None, agent=_Agent(), embedder=None,
                 authz=_StaticAuthz("claim"), settings=None)
    got = rt.ask("how many claims?", _identity())
    assert got.answer == "ANSWER"
    assert got.mode == "thinking"  # the resolved default, stamped by Runtime
    assert calls["retrieve"] == ("how many claims?", 24, [])  # k=24 default; facts threaded in
    assert calls["answer"][0].question == "how many claims?"
    assert calls["answer"][1] is snap


def test_ask_dispatches_to_the_mode_agent_and_stamps_the_mode(monkeypatch):
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.loop import AgentAnswer

    from mnemiq.semantic.retrieval import ContextPacket

    monkeypatch.setattr(
        rt_mod, "retrieve",
        lambda *a, **k: ContextPacket(question="q", cards=[], grant_fingerprint="fp",
                                      enrichment_version="v1"),
    )

    class _A:
        def __init__(self, tag):
            self.tag = tag

        def answer(self, packet, snapshot, grants, identity, emit=None):
            return AgentAnswer(answer=self.tag)

    agents = {"instant": _A("i"), "thinking": _A("t"), "deep": _A("d")}
    rt = Runtime(con=None, snapshot=None, adapter=None, agent=agents["thinking"],
                 embedder=None, authz=_StaticAuthz("claim"), settings=None, agents=agents)
    deep = rt.ask("q", _identity(), mode="deep")
    assert deep.answer == "d" and deep.mode == "deep"
    default = rt.ask("q", _identity())
    assert default.answer == "t" and default.mode == "thinking"


def test_ask_with_an_unknown_mode_fails_closed_before_any_work(monkeypatch):
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.route import UnknownMode

    def _no_retrieve(*a, **k):
        raise AssertionError("retrieval must not run for an unknown mode")

    monkeypatch.setattr(rt_mod, "retrieve", _no_retrieve)
    rt = Runtime(con=None, snapshot=None, adapter=None, agent=None, embedder=None,
                 authz=DenyAll(), settings=None)
    with pytest.raises(UnknownMode):
        rt.ask("q", _identity(), mode="fastest")


def test_build_runtime_rejects_an_unknown_default_mode(tmp_path):
    from mnemiq.agent.route import UnknownMode

    s = Settings(
        llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="x", acme_data_dir=None,
        store_path=str(tmp_path / "empty.duckdb"), default_mode="fastest",
    )
    with pytest.raises(UnknownMode):  # validated BEFORE the snapshot check -- boot fails fast
        build_runtime(s)


class _WriteAuthz:
    def __init__(self, *tables):
        self._g = GrantSet(frozenset(tables), writable=frozenset(tables))

    def grants_for(self, _identity):
        return self._g


def test_write_refuses_under_denyall():
    from mnemiq.contract import Snapshot
    from mnemiq.runtime import Runtime

    snap = Snapshot(version="v1", source_id="acme", created_at="t")
    rt = Runtime(con=None, snapshot=snap, adapter=None, agent=None, embedder=None,
                 authz=DenyAll(), settings=None)
    res = rt.write("INSERT INTO claim (id) VALUES (1)", _identity())
    assert res.approved is False and res.refusal


def test_write_executes_on_approval():
    from mnemiq.contract import Column, Snapshot
    from mnemiq.runtime import Runtime

    snap = Snapshot(version="v1", source_id="acme", created_at="t",
                    columns=[Column(id="claim.id", object_id="claim", name="id")])

    class _RWAdapter:
        dialect = "duckdb"

        def __init__(self):
            self.ran = []

        def execute(self, sql):
            self.ran.append(sql)
            return [] if sql.startswith("EXPLAIN") else [(1,)]

    adapter = _RWAdapter()

    class _WritesEnabled:
        # M3: the deployment switch now reaches the decider and defaults CLOSED, so a test that
        # exercises the execution path has to say which deployment it is testing. `settings=None`
        # used to mean "unconfigured", which quietly meant "writes allowed".
        write_enabled = True
        source_id = "acme"

    rt = Runtime(con=None, snapshot=snap, adapter=adapter, agent=None, embedder=None,
                 authz=_WriteAuthz("claim"), settings=_WritesEnabled())  # write grant on claim
    res = rt.write("INSERT INTO claim (id) VALUES (1)", _identity())
    assert res.approved is True and res.target == "claim" and res.rows_affected == 1
    assert any(not s.startswith("EXPLAIN") for s in adapter.ran)  # the write actually ran


def test_a_source_that_refuses_a_write_does_not_say_why_to_the_caller(caplog):
    """`WriteResult.refusal` is handed to an MCP agent verbatim by `db_write`.

    Any refusal from the source reaches here -- a permission error, a failed connection, a
    read-only deployment rejecting the statement -- and those name the table they refused and
    carry the DSN they refused it on. The read path withholds those words; this surface is the
    one an external agent actually reads.

    Both halves are asserted. Withholding from the caller is only defensible because the
    operator still gets it, so a test that checks the caller alone would go green on a change
    that disclosed the words to nobody at all.
    """
    from mnemiq.contract import Column, Snapshot
    from mnemiq.runtime import Runtime

    snap = Snapshot(version="v1", source_id="acme", created_at="t",
                    columns=[Column(id="claim.id", object_id="claim", name="id")])
    leaky = ('permission denied for table hr_prod.payroll_salary; '
             'connection postgresql://svc_mnemiq@10.2.0.7:5432/hr_prod')

    class _RefusingAdapter:
        dialect = "duckdb"

        def execute(self, sql):
            if sql.startswith("EXPLAIN"):
                return []
            raise RuntimeError(leaky)

    class _WritesEnabled:
        write_enabled = True
        source_id = "acme"

    rt = Runtime(con=None, snapshot=snap, adapter=_RefusingAdapter(), agent=None, embedder=None,
                 authz=_WriteAuthz("claim"), settings=_WritesEnabled())
    with caplog.at_level(logging.WARNING, logger="mnemiq.runtime"):
        res = rt.write("INSERT INTO claim (id) VALUES (1)", _identity())

    assert res.approved is False
    assert res.refusal, "the caller must still be told it was refused"
    for secret in ("payroll_salary", "hr_prod", "svc_mnemiq", "10.2.0.7", "postgresql://"):
        assert secret not in res.refusal, f"the write refusal disclosed {secret!r}"
    assert leaky in caplog.text, "withheld from the caller AND from the operator is not the trade"


def test_ask_threads_ontology_index_columns_and_definitions(monkeypatch):
    """Regression: question-time code resolution and the glossary reached eval's build_engine
    but NOT the product path, so `mnemiq ask` and the MCP server saw neither. The absence of a
    test over Runtime.ask is exactly why that gap survived review."""
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.contract import (
        CodeScheme, Column, Definition, Dimension, MeasureExpr, Metric, Snapshot,
    )

    seen = {}

    def fake_retrieve(con, question, identity, authz, embedder, k=5, table_facts=(),
                      definitions=(), metrics=(), dimensions=(), columns=(), ontology_index=None,
                      snapshot=None, card_style="cards"):
        seen["definitions"] = list(definitions)
        # Certified metrics and dimensions were the next pair to reach the snapshot and stop
        # there -- five of the fs corpus's records, carrying the certified SQL for settled, gross
        # and net volume, appended by `apply_certified` and read by nothing. Same gap this test
        # was written for, one generation later.
        seen["metrics"] = [m.id for m in metrics]
        seen["dimensions"] = [d.id for d in dimensions]
        seen["columns"] = [c.id for c in columns]
        seen["ontology_index"] = ontology_index
        # M4 added a fourth thing that has to reach the product path: without the snapshot,
        # retrieve cannot re-render a card against the caller's column policy and silently
        # serves the unscoped one. That is the same gap this test was written for.
        seen["snapshot"] = snapshot
        seen["card_style"] = card_style
        from mnemiq.semantic.retrieval import ContextPacket

        return ContextPacket(question=question, cards=[], grant_fingerprint="f",
                             enrichment_version=None)

    monkeypatch.setattr(rt_mod, "retrieve", fake_retrieve)

    class _Agent:
        def answer(self, packet, snapshot, grants, identity, emit=None):
            return AgentAnswer(answer="ANSWER")

    snap = Snapshot(
        version="v", source_id="s", created_at="t",
        columns=[Column(id="patient.icd10_cd", object_id="patient", name="icd10_cd",
                        code_scheme=CodeScheme(id="urn:icd10", label="ICD-10-CM"))],
        definitions=[Definition(id="d1", term="ICD-10-CM", domain="ontology",
                                definition="A diagnosis coding system.")],
        metrics=[Metric(id="patient_count", label="Patient Count", status="certified", owner="o",
                        grain="day",
                        measure=MeasureExpr(expr="count(distinct patient_id)", source="patient"),
                        time_dimension="day")],
        dimensions=[Dimension(id="patient.icd10_cd", label="Diagnosis", source="patient")],
    )
    sentinel = object()
    rt = Runtime(con=None, snapshot=snap, adapter=None, agent=_Agent(), embedder=None,
                 authz=_StaticAuthz("patient"), settings=None, ontology=sentinel)
    rt.ask("how many with type 2 diabetes", _identity())

    assert seen["ontology_index"] is sentinel        # the index reaches retrieval
    assert seen["snapshot"] is snap                 # ...and so does the snapshot (M4)
    assert seen["columns"] == ["patient.icd10_cd"]   # bound columns are visible to it
    assert [d.term for d in seen["definitions"]] == ["ICD-10-CM"]  # glossary seam fed
    assert seen["metrics"] == ["patient_count"]      # ...and the certified measures
    assert seen["dimensions"] == ["patient.icd10_cd"]
    # The card FORM is the same kind of thing: settable for eval, and the product path has
    # to be able to reach the same value or the flag measures a prompt `ask` cannot emit.
    assert seen["card_style"] == "cards"


def test_the_eval_door_and_the_product_door_ground_identically():
    """The eval door and the product door must NAME the same grounding arguments.

    **This is the weaker half of two guards, and it is worth knowing which half.** It compares
    argument NAMES; `test_undefined_term_guard`'s `REQUIRED`/`_is_hardcoded` scan compares VALUES,
    at every `retrieve` site in the repo, and is the one that catches the real regression. Verified:
    replacing `metrics=snapshot.metrics` with `metrics=()` leaves both name sets identical and this
    test green, while the derivation scan reports "hardcodes metrics instead of deriving it".

    What this adds that the scan cannot: the scan checks a PINNED list of owed arguments, so a NEW
    grounding argument threaded into one door and not the other is invisible to it until someone
    remembers to add it to `REQUIRED`. This notices the asymmetry itself.

    The gap it exists for has happened in both directions. The ontology index, the glossary and the
    bound columns once reached eval's `build_engine` and not `Runtime.ask` -- the test above was
    written for that. Then `apply_certified`'s metrics and dimensions, plus M4's snapshot, reached
    `Runtime.ask` and not `build_engine`, so every `mnemiq eval` measured a less-grounded engine
    than production and an ablation through it reported a false null (register M81).

    Only these two doors, deliberately. `retrieve` has four call sites -- `scripts/answer.py` and
    `scripts/ask.py` are the others -- but those are developer scripts, and it is the EVAL claiming
    to measure the product that has to match it. The scan covers all four.

    Compared as an AST, so reformatting cannot break it and an added argument cannot hide behind a
    line break. A deliberate divergence is allowed; it just has to be made here, with a reason.
    """
    import ast
    import inspect
    import textwrap

    import mnemiq.runtime as runtime_module
    from mnemiq.eval.engine import build_engine

    def retrieve_kwargs(function) -> set[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "retrieve":
                return {keyword.arg for keyword in node.keywords}
        raise AssertionError(f"{function.__qualname__} no longer calls retrieve()")

    eval_door = retrieve_kwargs(build_engine)
    product_door = retrieve_kwargs(runtime_module.Runtime.ask)

    assert eval_door == product_door, (
        "the eval and product doors ground differently, so an eval no longer measures the engine "
        f"the product runs. Only the product passes {sorted(product_door - eval_door)}; only the "
        f"eval passes {sorted(eval_door - product_door)}"
    )
    # Named explicitly as well as compared, because two doors that BOTH stopped passing the
    # certified measures would agree with each other and ground nothing -- the equality above
    # cannot tell that from two doors that are both right.
    assert {
        "metrics", "dimensions", "snapshot", "definitions", "ontology_index", "card_style",
    } <= eval_door


def test_ask_hands_retrieval_the_configured_card_style(monkeypatch, tmp_path):
    """The equality above only proves both doors pass the keyword, not that a value arrives.

    `card_style` is the settable one: an eval measuring `ddl` is measuring a prompt form the
    product must be able to emit, and `Runtime.ask` reads it off `Settings`. Passing a
    hardcoded `"cards"` at the product door satisfies the scan and grounds nothing.
    """
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.contract import Snapshot
    from mnemiq.semantic.retrieval import ContextPacket

    seen = {}

    def fake_retrieve(con, question, identity, authz, embedder, **kwargs):
        seen["card_style"] = kwargs.get("card_style")
        return ContextPacket(question=question, cards=[], grant_fingerprint="fp",
                             enrichment_version=None)

    class _Agent:
        def answer(self, packet, snapshot, grants, identity, emit=None):
            return AgentAnswer(answer="A")

    monkeypatch.setattr(rt_mod, "retrieve", fake_retrieve)
    settings = Settings(
        llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="x", acme_data_dir=None,
        store_path=str(tmp_path / "s.duckdb"), card_style="ddl",
    )
    rt = Runtime(con=None, snapshot=Snapshot(version="v", source_id="s", created_at="t"),
                 adapter=None, agent=_Agent(), embedder=None, authz=_StaticAuthz("claim"),
                 settings=settings)
    rt.ask("q", _identity())

    assert seen["card_style"] == "ddl"


def test_allow_all_clears_the_pii_levels_the_snapshot_tags():
    """`_AllowAll` must mean what it says once the snapshot reaches retrieval.

    These demo scripts print "every table is visible" and then hand `retrieve` the snapshot, which
    makes the column policy live. `GrantSet.pii_clearance` defaults to EMPTY, so an `_AllowAll` that
    forgets it DENIES every enrichment-tagged column: the card is served with those columns stripped
    and nothing on screen says why. Measured on the ACME snapshot -- 11 of 51 cards change.

    Asserted here because nothing else can see it: `_AllowAll` lives in two scripts, and the
    derivation scan reads `retrieve`'s keywords, not `GrantSet` construction. Reverting the
    clearance leaves the whole suite green without this.
    """
    import importlib.util
    import sys
    from pathlib import Path

    from mnemiq.contract import Column, Snapshot
    from mnemiq.semantic.cards import build_cards
    from mnemiq.sql.policy import build_access_policy

    snapshot = Snapshot(
        version="v", source_id="s", created_at="t",
        columns=[Column(id="patient.name", object_id="patient", name="name", pii_level="pii"),
                 Column(id="patient.id", object_id="patient", name="id")],
    )
    levels = {c.pii_level for c in snapshot.columns if c.pii_level and c.pii_level != "none"}

    for script in ("ask", "answer"):
        path = Path(__file__).resolve().parents[1] / "scripts" / f"{script}.py"
        spec = importlib.util.spec_from_file_location(f"_script_{script}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        grants = module._AllowAll(["patient"], levels).grants_for(None)
        policy = build_access_policy(snapshot, grants)
        assert not policy.denied, (
            f"scripts/{script}.py: _AllowAll denies {sorted(policy.denied)} -- a provider whose own "
            "docstring says every table is visible must clear the levels the snapshot tags, or the "
            "cards it serves silently lose those columns"
        )
        # The card must still carry the tagged column, which is the observable consequence.
        card = next(c for c in build_cards(snapshot, policy=policy) if c.object_id == "patient")
        assert "name" in card.text
