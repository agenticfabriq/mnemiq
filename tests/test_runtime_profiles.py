"""One runtime, several profiles: a request can pick one it is allowed, with no restart.

Each profile has its own model, settings and agents, over one shared source, store, cache and grants. These run the real
`build_runtime` over a real store and source, with only the model's reply faked, so the join is what is tested.
"""

from __future__ import annotations

import json
import logging

import duckdb
import pytest

import mnemiq.llm.window as window
import mnemiq.runtime as rt_mod
from mnemiq.agent.route import UnknownMode
from mnemiq.config import Settings
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.llm.client import LLMClient
from mnemiq.profiles import UnknownProfile, offered_to
from mnemiq.runtime import SnapshotMissing, build_runtime
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import save_snapshot

LOCAL = "http://localhost:8000/v1"
HOSTED = "https://models.example.com/v1"

FILE = f"""
default = "qwen"

[profiles.qwen]
llm_model = "qwen-model"
retrieval_k = 11
[profiles.qwen.instant]
card_style = "bare"

[profiles.gpt]
llm_base_url = "{HOSTED}"
llm_model = "gpt-model"
api_key_env = "TEST_GPT_KEY"
retrieval_k = 3
roles = ["analyst"]

[profiles.open]
llm_model = "open-model"
roles = ["*"]
"""


def _snapshot() -> Snapshot:
    return Snapshot(version="v1", source_id="only", created_at="t",
                    columns=[Column(id="wide.col_0", object_id="wide", name="col_0", data_type="VARCHAR")])


@pytest.fixture
def base(tmp_path, monkeypatch) -> Settings:
    """A deployment with a real store and source, and a profile file."""
    monkeypatch.setenv("TEST_GPT_KEY", "gpt-key")
    source = tmp_path / "src.duckdb"
    duckdb.connect(str(source)).execute("CREATE TABLE wide (col_0 TEXT)").close()
    store = tmp_path / "store.duckdb"
    con = init_store(str(store))
    save_snapshot(con, _snapshot())
    con.close()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "duckdb", "target": str(source),
                                     "catalog": "src", "schema": "main"}]))
    authz = tmp_path / "authz.json"
    authz.write_text(json.dumps({"roles": {"analyst": ["wide"], "viewer": ["wide"]}}))
    profiles = tmp_path / "profiles.toml"
    profiles.write_text(FILE)
    return Settings(acme_data_dir=None, llm_base_url=LOCAL, llm_api_key="local-key", llm_model="base-model",
                    sources_path=str(manifest), store_path=str(store), authz_path=str(authz),
                    profiles_path=str(profiles), llm_window_check=False)


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    """The models actually called. The reply is a deferral, which ends the turn after the writer's first call."""
    seen: list[str] = []

    def complete(self, system, user, max_tokens=512, extra_body=None, temperature=None):
        seen.append(self._model)
        return '{"sql": null, "reason": "no"}'

    monkeypatch.setattr(LLMClient, "complete", complete)
    return seen


@pytest.fixture
def retrieved(monkeypatch) -> list[dict]:
    """What retrieval was asked for, returning one card. The real one needs an embedding endpoint."""
    seen: list[dict] = []

    def fake(con, question, identity, authz, embedder, **kwargs):
        seen.append(kwargs)
        return ContextPacket(question=question, cards=[RetrievedCard(object_id="wide", card="C", score=1.0)],
                             grant_fingerprint="fp", enrichment_version=None)

    monkeypatch.setattr(rt_mod, "retrieve", fake)
    return seen


def _who(*roles, groups=()) -> IdentityContext:
    return IdentityContext(tenant_id="t", principal_id="u", roles=list(roles), groups=list(groups))


def test_a_request_reaches_the_named_profile(base, calls, retrieved):
    rt = build_runtime(base)
    rt.ask("q", _who("analyst"), mode="instant", profile="gpt")
    assert calls == ["gpt-model"]


def test_a_request_naming_none_reaches_the_default(base, calls, retrieved):
    rt = build_runtime(base)
    rt.ask("q", _who("analyst"), mode="instant")
    assert calls == ["qwen-model"]


def test_switching_needs_no_restart(base, calls, retrieved):
    rt = build_runtime(base)
    who = _who("analyst")
    for profile in ("gpt", "qwen", "open", "gpt"):
        rt.ask("q", who, mode="instant", profile=profile)
    assert calls == ["gpt-model", "qwen-model", "open-model", "gpt-model"]


def test_retrieval_and_the_form_come_from_the_picked_profile(base, calls, retrieved):
    rt = build_runtime(base)
    rt.ask("q", _who("analyst"), mode="instant", profile="gpt")
    rt.ask("q", _who("analyst"), mode="instant", profile="qwen")
    assert [(k["k"], k["card_style"]) for k in retrieved] == [(3, "cards"), (11, "bare")]


def test_the_fit_comes_from_the_picked_profile(base, calls, retrieved, monkeypatch):
    rt = build_runtime(base)
    fitted: list = []
    for name, loaded in rt.profiles.items():
        original = loaded.fitter.fit
        loaded.fitter.fit = (lambda orig, label: lambda packet, snapshot, grants, card_style=None: (
            fitted.append((label, card_style)), orig(packet, snapshot, grants, card_style=card_style))[1])(original, name)
    rt.ask("q", _who("analyst"), mode="instant", profile="gpt")
    rt.ask("q", _who("analyst"), mode="instant", profile="qwen")
    assert fitted == [("gpt", "cards"), ("qwen", "bare")]


def test_profiles_share_the_source_cache_store_and_grants(base, calls, retrieved):
    rt = build_runtime(base)
    agents = [loaded.agents["instant"] for loaded in rt.profiles.values()]
    assert all(a.adapter is rt.adapter for a in agents)
    assert all(a.cache is agents[0].cache for a in agents)
    assert all(a.values.__class__ is agents[0].values.__class__ for a in agents)
    assert len({id(a.generator) for a in agents}) == len(agents)


def test_the_embedder_stays_on_the_deployment(base, calls, retrieved, monkeypatch):
    built: list[Settings] = []
    real = rt_mod.LLMEmbedder
    monkeypatch.setattr(rt_mod, "LLMEmbedder", lambda settings: (built.append(settings), real(settings))[1])
    rt = build_runtime(base.model_copy(update={"embed_base_url": "http://localhost:9000/v1", "embed_api_key": "e"}))
    assert len(built) == 1
    assert built[0].llm_model == "base-model"
    assert rt.embedder is not None


@pytest.mark.parametrize("who, name", [(_who("viewer"), "gpt"), (_who(), "gpt"), (_who("analyst"), "nobody"),
                                       (_who("viewer"), "nobody")])
def test_unknown_and_forbidden_read_alike_and_do_no_work(base, calls, retrieved, who, name):
    rt = build_runtime(base)
    with pytest.raises(UnknownProfile) as refused:
        rt.ask("q", who, mode="instant", profile=name)
    assert name not in str(refused.value)
    assert calls == [] and retrieved == []


def test_what_is_offered_is_only_what_this_caller_may_pick(base, calls, retrieved):
    rt = build_runtime(base)
    with pytest.raises(UnknownProfile) as refused:
        rt.ask("q", _who("viewer"), profile="gpt")
    assert refused.value.offered == ["open", "qwen"]
    with pytest.raises(UnknownProfile) as refused:
        rt.ask("q", _who("analyst"), profile="nobody")
    assert refused.value.offered == ["gpt", "open", "qwen"]


def test_roles_read_as_the_access_policy_reads_them(base, calls, retrieved):
    """`FileAuthzProvider.grants_for` reads `[*roles, *groups]`, and picking a profile must not read them differently."""
    rt = build_runtime(base)
    rt.ask("q", _who(groups=["analyst"]), mode="instant", profile="gpt")
    assert calls == ["gpt-model"]
    assert offered_to(_who(), {n: lp.profile for n, lp in rt.profiles.items()}) == ["open", "qwen"]


def test_anyone_may_name_the_default_and_a_star(base, calls, retrieved):
    """Reached, not refused. With no grant the turn then ends before any model call, which is the access policy's doing."""
    rt = build_runtime(base)
    rt.ask("q", _who(), mode="instant", profile="qwen")
    rt.ask("q", _who(), mode="instant", profile="open")
    assert len(retrieved) == 2


def test_an_unknown_mode_still_fails_before_any_work(base, calls, retrieved):
    rt = build_runtime(base)
    with pytest.raises(UnknownMode):
        rt.ask("q", _who("analyst"), mode="fast", profile="gpt")
    assert calls == [] and retrieved == []


def test_a_bad_file_stops_boot_with_a_sentence(base, tmp_path):
    (tmp_path / "profiles.toml").write_text('default = "nobody"\n[profiles.only]\nllm_model = "m"')
    with pytest.raises(SnapshotMissing, match="nobody"):
        build_runtime(base)


def test_boot_checks_the_window_for_each_profile(base, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(rt_mod, "_warn_prompt_window", lambda settings, *a, **k: seen.append(settings.llm_model))
    build_runtime(base)
    assert sorted(seen) == ["gpt-model", "open-model", "qwen-model"]


def test_a_snapshot_swap_warns_for_each_profile(base, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(rt_mod, "_warn_prompt_window", lambda settings, *a, **k: seen.append(settings.llm_model))
    rt = build_runtime(base)
    seen.clear()
    rt.settings = rt.settings.model_copy(update={"control_dsn": "x"})
    monkeypatch.setattr(rt_mod, "load_current_snapshot", lambda settings, con: (_snapshot(), {"only": "v2"}))
    monkeypatch.setattr(rt_mod, "_warn_view_inventory", lambda *a, **k: None)
    monkeypatch.setattr(rt_mod, "_warn_policy_advisories", lambda *a, **k: None)
    rt.reload_if_stale()
    assert sorted(seen) == ["gpt-model", "open-model", "qwen-model"]


def test_no_file_is_one_default_profile_over_the_runtimes_own_agents(base, calls, retrieved):
    rt = build_runtime(base.model_copy(update={"profiles_path": None}))
    assert list(rt.profiles) == ["default"]
    assert rt.profiles["default"].agents is rt.agents
    assert rt.profiles["default"].settings is rt.settings
    rt.ask("q", _who("analyst"), mode="instant")
    assert calls == ["base-model"]


def test_no_file_refuses_a_named_profile(base, calls, retrieved):
    rt = build_runtime(base.model_copy(update={"profiles_path": None}))
    with pytest.raises(UnknownProfile) as refused:
        rt.ask("q", _who("analyst"), profile="gpt")
    assert refused.value.offered == ["default"]


def test_a_runtime_built_by_hand_has_no_profiles_and_refuses_a_name():
    """Tests and the eval door build a `Runtime` directly, with no profiles to pick from."""
    rt = rt_mod.Runtime(con=None, snapshot=None, adapter=None, agent=None, embedder=None, authz=None, settings=None)
    with pytest.raises(UnknownProfile) as refused:
        rt.ask("q", _who("analyst"), profile="gpt")
    assert refused.value.offered == []


def test_each_profile_has_a_version_of_its_own(base):
    rt = build_runtime(base)
    assert len({loaded.version for loaded in rt.profiles.values()}) == 3


def test_the_window_measures_each_profiles_own_server(base, monkeypatch, caplog):
    """The join under `_warn_prompt_window`: each profile's endpoint and model reach the count."""
    measured: list[tuple] = []

    def server(base_url, model, system, user, http=None, api_key=None):
        measured.append((base_url, model))
        return (10, 28_672)

    monkeypatch.setattr(window, "count_on_server", server)
    with caplog.at_level(logging.INFO):
        build_runtime(base.model_copy(update={"llm_window_check": True}))
    assert {m for m in measured} >= {(HOSTED, "gpt-model"), (LOCAL, "qwen-model")}


# --- a restricted profile's endpoint sees only what its own roles could send it -------------------------------------


@pytest.fixture
def two_views(base, tmp_path, monkeypatch) -> Settings:
    """Two roles over two tables, one holding text the other role may not see, and a count that records what it was sent."""
    store = tmp_path / "views.duckdb"
    con = init_store(str(store))
    save_snapshot(con, Snapshot(version="v1", source_id="only", created_at="t", columns=[
        Column(id="wide.col_0", object_id="wide", name="col_0", data_type="VARCHAR"),
        Column(id="private_table.planted_private_column", object_id="private_table", name="planted_private_column",
               data_type="VARCHAR", description="planted private description")]))
    con.close()
    authz = tmp_path / "views.json"
    authz.write_text(json.dumps({"roles": {"analyst": ["wide"], "viewer": ["private_table"]}}))
    return base.model_copy(update={"store_path": str(store), "authz_path": str(authz), "llm_window_check": True})


def _counted(monkeypatch) -> list[tuple[str, str]]:
    sent: list[tuple[str, str]] = []

    def server(base_url, model, system, user, http=None, api_key=None):
        sent.append((base_url, user))
        return (10, 28_672)

    monkeypatch.setattr(window, "count_on_server", server)
    return sent


def test_a_restricted_profiles_endpoint_never_counts_another_roles_view(two_views, monkeypatch):
    """Sending it is the disclosure, whatever the endpoint answers."""
    sent = _counted(monkeypatch)
    build_runtime(two_views)
    hosted = [user for url, user in sent if url == HOSTED]
    assert hosted, "the positive control: the hosted profile is counted, on its own role's view"
    assert not any("planted_private" in user for user in hosted)
    assert any("col_0" in user for user in hosted)


def test_the_default_profile_still_counts_every_role(two_views, monkeypatch):
    sent = _counted(monkeypatch)
    build_runtime(two_views)
    assert any("planted_private" in user for url, user in sent if url == LOCAL)


def test_the_configured_identity_is_held_to_the_same_rule(two_views, monkeypatch):
    """Its roles merge into one view, which is measured beside the policy's."""
    sent = _counted(monkeypatch)
    build_runtime(two_views.model_copy(update={"roles": "viewer", "principal": "operator"}))
    assert not any("planted_private" in user for url, user in sent if url == HOSTED)


def test_a_snapshot_swap_holds_a_restricted_profile_to_the_same_rule(two_views, monkeypatch):
    rt = build_runtime(two_views.model_copy(update={"llm_window_check": True}))
    sent = _counted(monkeypatch)
    rt.settings = rt.settings.model_copy(update={"control_dsn": "x"})
    snapshot = rt.snapshot
    monkeypatch.setattr(rt_mod, "load_current_snapshot", lambda settings, con: (snapshot, {"only": "v2"}))
    monkeypatch.setattr(rt_mod, "_warn_view_inventory", lambda *a, **k: None)
    monkeypatch.setattr(rt_mod, "_warn_policy_advisories", lambda *a, **k: None)
    rt.reload_if_stale()
    hosted = [user for url, user in sent if url == HOSTED]
    assert hosted
    assert not any("planted_private" in user for user in hosted)
