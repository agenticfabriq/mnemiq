"""A bare schema form, and settings each mode can take on its own.

The best prompt depends on the model and the dataset, so a mode's knobs are settings an operator can
benchmark, not constants. The bare form is the smallest schema a writer can use: names, types and
declared keys, spelled so that what the model copies runs on this engine.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest
import sqlglot
from pydantic import ValidationError

from mnemiq.agent.modes import MODES, PROFILES, card_styles, modes_for
from mnemiq.agent.route import UnknownMode
from mnemiq.authz.grants import GrantSet
from mnemiq.config import Settings
from mnemiq.contract import CodedValue, Column, JoinKey, Relationship, Snapshot, SourceBinding
from mnemiq.semantic.cards import build_cards
from mnemiq.semantic.fit import PromptFitter
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard, _attach_facts
from mnemiq.sql.policy import build_access_policy


def _column(table: str, name: str, data_type: str = "INTEGER", **fields) -> Column:
    return Column(id=f"{table}.{name}", object_id=table, name=name, data_type=data_type, **fields)


def _snapshot(*columns: Column, relationships=(), tables=None) -> Snapshot:
    names = tables or sorted({c.object_id for c in columns})
    return Snapshot(version="v", source_id="s", created_at="t", columns=list(columns),
                    relationships=list(relationships),
                    source_bindings=[SourceBinding(id=t, source_id="s", object_id=t, source_object=t, binding_type="table") for t in names])


def _card(snapshot: Snapshot, table: str, dialect: str | None = "duckdb", policy=None) -> str:
    return next(c.text for c in build_cards(snapshot, policy=policy, style="bare", dialect=dialect)
                if c.object_id == table)


def _fk(child: str, parent: str, *pairs: tuple[str, str]) -> Relationship:
    return Relationship(id=f"{child}->{parent}", **{"from": child}, to=parent, cardinality="many_to_one",
                        join_keys=[JoinKey(left=a, right=b) for a, b in pairs])


# --- the bare form -------------------------------------------------------------------------------


def test_bare_holds_names_and_types_and_nothing_derived_from_the_data():
    snapshot = _snapshot(
        _column("claim", "id"),
        _column("claim", "status", "VARCHAR", semantic_type="category", pii_level="pii",
                description="WORDS OF DESCRIPTION", row_count=9, null_count=1,
                coded_values=[CodedValue(code="A", meaning="MEANING OF A")]),
        _column("claim", "gone", "VARCHAR", row_count=9, null_count=9),
    )
    assert _card(snapshot, "claim") == ("CREATE TABLE claim (\n    id INTEGER,\n    status VARCHAR,\n"
                                        "    gone VARCHAR\n);")


def test_the_other_forms_still_carry_what_bare_leaves_out():
    """The control: bare is a new form, and the forms beside it do not change."""
    snapshot = _snapshot(_column("claim", "status", "VARCHAR", description="WORDS OF DESCRIPTION"))
    for style in ("cards", "ddl"):
        assert "WORDS OF DESCRIPTION" in build_cards(snapshot, style=style)[0].text


def test_bare_keeps_declared_keys_inside_the_statement():
    """A keyed database joins on them. The form that dropped them left the writer guessing a join
    the schema states outright."""
    snapshot = _snapshot(
        _column("orders", "id"), _column("orders", "account_id"), _column("orders", "region"),
        _column("orders", "branch"), _column("account", "id"),
        _column("site", "region"), _column("site", "code"),
        relationships=[_fk("orders", "account", ("account_id", "id")),
                       _fk("orders", "site", ("region", "region"), ("branch", "code"))],
    )
    card = _card(snapshot, "orders")
    assert "FOREIGN KEY (account_id) REFERENCES account (id)" in card
    assert "FOREIGN KEY (region, branch) REFERENCES site (region, code)" in card
    parsed = sqlglot.parse_one(card, read="duckdb")
    assert len(list(parsed.find_all(sqlglot.exp.ForeignKey))) == 2


def test_a_join_with_no_keys_is_still_named():
    """The relationship is known and its keys are not: it cannot be a FOREIGN KEY, and dropping it
    would hide that the tables join at all."""
    snapshot = _snapshot(_column("orders", "id"), _column("account", "id"),
                         relationships=[_fk("orders", "account")])
    card = _card(snapshot, "orders")
    assert "FOREIGN KEY" not in card
    assert card.endswith(");\n-- joins account")
    sqlglot.parse_one(card, read="duckdb")


@pytest.mark.parametrize("name, dialect, spelled", [
    ("plain_name", "duckdb", "plain_name"),
    ("MixedCase", "duckdb", "MixedCase"),          # DuckDB finds it either way
    ("MixedCase", "postgres", '"MixedCase"'),      # unquoted, Postgres folds it to another name
    ("ORDER_ID", "oracle", "ORDER_ID"),            # unquoted, Oracle folds to this very spelling
    ("order_id", "oracle", '"order_id"'),          # unquoted, Oracle folds it to another name
    ("Enrollment (K-12)", "duckdb", '"Enrollment (K-12)"'),
    ("has space", "mysql", "`has space`"),
    ("order", "duckdb", '"order"'),                # a keyword where a name must stand
    ("order", "sqlite", '"order"'),
    ("order", "mysql", "`order`"),                 # an engine with no list of its own: every list
    ("MixedCase", None, '"MixedCase"'),            # no dialect known: the spelling that cannot fold
])
def test_a_name_is_spelled_so_a_copy_of_it_runs_on_this_engine(name, dialect, spelled):
    snapshot = _snapshot(_column("t", name, "INTEGER"), _column("t", "other"))
    assert f"\n    {spelled} INTEGER," in _card(snapshot, "t", dialect=dialect)


def test_a_dotted_column_is_one_name_and_a_dotted_table_may_be_a_path():
    """A column name is never qualified, so its dot belongs to the name: unquoted, the statement would not parse and a
    copy of it would read as table and column. An object id may be either, and the id alone cannot say which."""
    snapshot = _snapshot(_column("main.orders", "a.b"), _column("account", "c.d"),
                         relationships=[_fk("main.orders", "account", ("a.b", "c.d"))])
    card = _card(snapshot, "main.orders")
    assert card.startswith("CREATE TABLE main.orders (")
    assert '\n    "a.b" INTEGER,' in card
    assert 'FOREIGN KEY ("a.b") REFERENCES account ("c.d")' in card
    sqlglot.parse_one(card, read="duckdb")


def test_a_table_and_its_keys_are_spelled_the_same_way():
    snapshot = _snapshot(_column("order", "group"), _column("Big Table", "id"),
                         relationships=[_fk("order", "Big Table", ("group", "id"))])
    card = _card(snapshot, "order")
    assert card.startswith('CREATE TABLE "order" (')
    assert 'FOREIGN KEY ("group") REFERENCES "Big Table" (id)' in card
    sqlglot.parse_one(card, read="duckdb")


def test_a_masked_column_keeps_its_note_and_a_denied_one_is_not_there():
    """Bare strips what helps the writer, never what governs it: the note is why a masked column
    goes unread, and a denied column's name is not this identity's to see."""
    snapshot = _snapshot(_column("p", "id"), _column("p", "secret", "VARCHAR", pii_level="pii"),
                         _column("p", "hidden", "VARCHAR", pii_level="phi"))
    grants = GrantSet(frozenset({"p"}), pii_mask=frozenset({"pii"}))
    card = _card(snapshot, "p", policy=build_access_policy(snapshot, grants))
    assert "MASKED for this identity" in card
    assert "secret" in card
    assert "hidden" not in card
    sqlglot.parse_one(card, read="duckdb")


def test_a_key_on_a_denied_column_is_not_shown():
    snapshot = _snapshot(_column("orders", "id"), _column("orders", "hidden", pii_level="phi"),
                         _column("account", "id"),
                         relationships=[_fk("orders", "account", ("hidden", "id"))])
    grants = GrantSet(frozenset({"orders", "account"}))
    assert "FOREIGN KEY" not in _card(snapshot, "orders", policy=build_access_policy(snapshot, grants))


def test_bare_takes_no_facts_block():
    class Facts:
        object_id = "claim"
    card = RetrievedCard(object_id="claim", card="CREATE TABLE claim (\n    id INTEGER\n);", score=0.0)
    before = card.card
    _attach_facts([card], [Facts()], "bare", grants=GrantSet(frozenset({"claim"})), columns=[],
                  relationships=[])
    assert card.card == before


# --- a mode's settings ---------------------------------------------------------------------------


def _settings(**fields) -> Settings:
    return Settings(llm_base_url=None, llm_api_key=None, llm_model=None, acme_data_dir=None, **fields)


def test_no_settings_leaves_every_mode_as_built():
    assert modes_for(_settings()) == MODES


def test_a_setting_changes_that_mode_and_only_that_knob():
    modes = modes_for(_settings(mode_settings={"instant": {"card_style": "bare", "max_attempts": 2}}))
    assert modes["instant"].card_style == "bare"
    assert modes["instant"].max_attempts == 2
    assert modes["instant"].budget_s == MODES["instant"].budget_s
    assert modes["thinking"] == MODES["thinking"]
    assert modes["deep"] == MODES["deep"]


def test_settings_come_from_the_environment_as_json(monkeypatch):
    monkeypatch.setenv("MNEMIQ_MODE_SETTINGS", '{"thinking": {"verify": "sanity"}}')
    assert modes_for(_settings())["thinking"].verify == "sanity"


def test_an_explicit_null_clears_the_agreement_gate():
    """Unset keeps the mode's value, and null turns off the one knob for which null means off."""
    assert modes_for(_settings(mode_settings={"deep": {"judge": False}}))["deep"].min_agreement == 0.6
    assert modes_for(_settings(mode_settings={"deep": {"min_agreement": None}}))["deep"].min_agreement is None


def test_an_unknown_mode_is_refused_naming_the_real_ones():
    with pytest.raises(UnknownMode, match="instant"):
        modes_for(_settings(mode_settings={"fast": {"max_attempts": 2}}))


@pytest.mark.parametrize("knobs", [
    {"card_size": "bare"},         # no such knob: a typo must not read as "no effect"
    {"card_style": "BARE"},
    {"verify": "sometimes"},
    {"max_attempts": 0},
    {"candidates": 0},
    {"budget_s": 0},
    {"min_agreement": 1.5},
    {"max_attempts": None},        # null means off for only one knob
])
def test_a_knob_that_cannot_be_read_is_refused(knobs):
    with pytest.raises(ValidationError):
        _settings(mode_settings={"instant": knobs})


@pytest.mark.parametrize("knobs", [{"judge": True}, {"min_agreement": 0.5}])
def test_a_knob_that_would_change_nothing_is_refused(knobs):
    """An arm that silently changes nothing scores cleanly and reads as "no effect", so a benchmark
    would report a finding about a setting that never ran. Both read only several candidates."""
    with pytest.raises(ValueError, match="candidates"):
        modes_for(_settings(mode_settings={"instant": knobs}))
    assert modes_for(_settings(mode_settings={"instant": {**knobs, "candidates": 3}}))["instant"].candidates == 3


def test_every_form_a_mode_sends_is_named_once():
    assert card_styles(_settings()) == ["cards"]
    assert card_styles(_settings(card_style="bare")) == ["bare"]
    assert card_styles(_settings(card_style="bare", mode_settings={"deep": {"card_style": "ddl"}})) == ["bare", "ddl"]
    assert card_styles(_settings(mode_settings={"instant": {"card_style": "bare"}})) == ["bare", "cards"]


def test_a_copied_env_example_line_means_no_settings(monkeypatch):
    """`.env.example` says "Copy to .env" and renders this line empty. A JSON setting that refused an
    empty value, or rendered a placeholder, would stop every deployment that followed it at boot."""
    monkeypatch.setenv("MNEMIQ_MODE_SETTINGS", "")
    assert modes_for(_settings()) == MODES
    line = next(ln for ln in Settings.env_example().splitlines() if ln.startswith("MNEMIQ_MODE_SETTINGS="))
    assert line.startswith("MNEMIQ_MODE_SETTINGS=  #")


def test_malformed_json_is_refused(monkeypatch):
    monkeypatch.setenv("MNEMIQ_MODE_SETTINGS", "{instant: bare}")
    with pytest.raises(ValidationError):
        _settings()


# --- profiles: one column of the table per kind of model ----------------------------------------


def test_the_frontier_profile_is_todays_table_and_the_default():
    assert _settings().profile == "frontier"
    assert PROFILES["frontier"] == {}
    assert modes_for(_settings(profile="frontier")) == MODES


def test_the_qwen_profile_changes_what_was_measured_and_nothing_else():
    modes = modes_for(_settings(profile="qwen-32b"))
    assert modes["instant"].card_style == "bare"
    assert replace(modes["instant"], card_style=None) == MODES["instant"]
    assert modes["thinking"] == MODES["thinking"]
    assert modes["deep"] == MODES["deep"]


def test_a_deployments_settings_win_over_its_profile():
    modes = modes_for(_settings(profile="qwen-32b",
                                mode_settings={"instant": {"card_style": "cards", "max_attempts": 2}}))
    assert modes["instant"].card_style == "cards"
    assert modes["instant"].max_attempts == 2
    assert modes_for(_settings(profile="qwen-32b", mode_settings={"instant": {"max_attempts": 2}}))[
        "instant"].card_style == "bare"


def test_an_unknown_profile_is_refused_naming_the_real_ones():
    with pytest.raises(ValueError, match="frontier"):
        modes_for(_settings(profile="fast"))


def test_a_profile_reaches_the_window_check():
    assert card_styles(_settings(profile="qwen-32b")) == ["bare", "cards"]
    assert card_styles(_settings(profile="qwen-32b", card_style="bare")) == ["bare"]


# --- the join: the mode's form reaches the prompt ---------------------------------------------------


def test_each_mode_asks_retrieval_and_the_fit_for_its_own_form(monkeypatch, tmp_path):
    """Per-component tests would stay green with the setting never reaching `ask`."""
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.runtime import Runtime

    retrieved: list[tuple] = []
    fitted: list = []

    def fake_retrieve(con, question, identity, authz, embedder, **kwargs):
        retrieved.append((kwargs["card_style"], kwargs["dialect"]))
        return ContextPacket(question=question, cards=[], grant_fingerprint="fp", enrichment_version=None)

    class _Fitter:
        def fit(self, packet, snapshot, grants, card_style=None):
            fitted.append(card_style)
            return packet

    class _Agent:
        def decline_if_too_long(self, question):
            return None

        def answer(self, packet, snapshot, grants, identity, emit=None):
            return AgentAnswer(answer="A")

    class _Authz:
        def grants_for(self, identity):
            return GrantSet(frozenset({"claim"}))

    class _Adapter:
        dialect = "postgres"

    monkeypatch.setattr(rt_mod, "retrieve", fake_retrieve)
    settings = _settings(pg_dsn="x", store_path=str(tmp_path / "s.duckdb"),
                         mode_settings={"instant": {"card_style": "bare"}})
    rt = Runtime(con=None, snapshot=Snapshot(version="v", source_id="s", created_at="t"), adapter=_Adapter(),
                 agent=_Agent(), embedder=None, authz=_Authz(), settings=settings, fitter=_Fitter(),
                 agents={name: _Agent() for name in MODES}, modes=modes_for(settings))
    from mnemiq.contract import IdentityContext
    identity = IdentityContext(tenant_id="t", principal_id="u")
    rt.ask("q", identity, mode="instant")
    rt.ask("q", identity, mode="thinking")

    assert retrieved == [("bare", "postgres"), ("cards", "postgres")]
    assert fitted == ["bare", "cards"]


def test_a_mode_with_its_own_knobs_is_built_with_them():
    from mnemiq.agent.modes import build_agent

    settings = _settings(mode_settings={"instant": {"max_attempts": 2, "corrector": True, "budget_s": 45.0}})
    corrector = object()
    agent = build_agent(modes_for(settings)["instant"], generator=None, synthesizer=None, adapter=None, cache=None,
                        corrector=corrector, values=None, selector=None)
    assert agent.budget.max_attempts == 2
    assert agent.budget.wall_clock_s == 45.0
    assert agent.corrector is corrector


def test_verifiers_are_built_from_the_modes_settings_not_the_table():
    from mnemiq.runtime import _build_mode_verifiers

    class _Client:
        pass

    settings = _settings(mode_settings={"instant": {"verify": "off"}})
    verifiers, _ = _build_mode_verifiers(settings, client=_Client(), modes=modes_for(settings))
    assert verifiers["instant"] is None
    assert verifiers["thinking"] is not None


# --- the fit ---------------------------------------------------------------------------------------


def _fitter(window: int) -> PromptFitter:
    return PromptFitter(dialect="duckdb", card_style="cards", reply_tokens=10, window=window)


def _described() -> Snapshot:
    return _snapshot(*[_column("wide", f"c{i}", "VARCHAR", description="WORDS " * 40) for i in range(40)])


def _packet(snapshot: Snapshot, style: str) -> ContextPacket:
    return ContextPacket(
        question="q", grant_fingerprint="fp", enrichment_version=None,
        cards=[RetrievedCard(object_id=c.object_id, card=c.text, score=1.0)
               for c in build_cards(snapshot, style=style, dialect="duckdb")])


def test_the_fit_renders_the_form_it_is_asked_for_not_its_own_default():
    snapshot = _described()
    grants = GrantSet(frozenset({"wide"}))
    fitted = _fitter(window=3000).fit(_packet(snapshot, "ddl"), snapshot, grants, card_style="ddl")
    assert fitted.cards[0].card.startswith("CREATE TABLE")


def test_a_bare_prompt_over_the_window_has_nothing_to_withhold(caplog):
    """The fit trims descriptions, and bare carries none. Reporting "0 of N sent" would disclose a
    withholding that did not happen, so the prompt goes as it is, with the warning it deserves."""
    snapshot = _described()
    grants = GrantSet(frozenset({"wide"}))
    packet = _packet(snapshot, "bare")
    with caplog.at_level(logging.WARNING, logger="mnemiq.semantic.fit"):
        fitted = _fitter(window=20).fit(packet, snapshot, grants, card_style="bare")
    assert fitted is packet
    assert fitted.descriptions is None
    assert any("no column descriptions to withhold" in r.getMessage() for r in caplog.records)


def _report(prompt_tokens: int):
    from mnemiq.llm.window import WindowReport

    return WindowReport(prompt_tokens=prompt_tokens, reply_tokens=10, window=100, counted_by_server=True, cards=1)


def _window_warnings(monkeypatch, caplog, sizes: dict, **fields) -> tuple[list[str], str]:
    import mnemiq.llm.window as window_mod
    from mnemiq.runtime import _warn_prompt_window

    seen: list[str] = []

    def fake_worst(con, snapshot, settings, dialect, authz):
        seen.append(settings.card_style)
        return _report(sizes[settings.card_style])

    monkeypatch.setattr(window_mod, "worst_window", fake_worst)
    settings = _settings(llm_window_check=True, **fields).model_copy(update={"llm_base_url": "http://x/v1"})
    with caplog.at_level(logging.INFO, logger="mnemiq.runtime"):
        _warn_prompt_window(settings, None, Snapshot(version="v", source_id="s", created_at="t"), None, None,
                            frozenset())
    return seen, caplog.text


@pytest.mark.parametrize("sizes", [{"bare": 500, "cards": 50}, {"bare": 50, "cards": 500}])
def test_the_window_is_checked_for_every_form_and_the_worst_is_reported(monkeypatch, caplog, sizes):
    """No form is always the largest: on a schema with little to describe, the card can be the smaller. So each form
    in use is measured, and a mode that cannot fit is reported whichever form it is on."""
    seen, text = _window_warnings(monkeypatch, caplog, sizes, mode_settings={"instant": {"card_style": "bare"}})
    assert seen == ["bare", "cards"]
    assert "510" in text          # the larger one's tokens, prompt and reply
    assert "prompt window fits" not in text


def test_one_form_is_measured_once(monkeypatch, caplog):
    seen, _ = _window_warnings(monkeypatch, caplog, {"cards": 50})
    assert seen == ["cards"]


@pytest.mark.parametrize("fields", [{"profile": "qwen-32b"}, {"mode_settings": {"instant": {"card_style": "bare"}}},
                                    {"profile": "nobody"}])
def test_the_eval_door_refuses_settings_it_would_not_apply(fields):
    """It builds its own agent and takes no mode, so these would never run, and a benchmark through it would report
    settings that did not. `MNEMIQ_CARD_STYLE` and its own arguments are what it reads."""
    from mnemiq.eval.engine import build_engine

    with pytest.raises(ValueError, match="eval"):
        build_engine(Snapshot(version="v", source_id="s", created_at="t"), None, _settings(**fields))
