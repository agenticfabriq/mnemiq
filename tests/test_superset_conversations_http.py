"""The Superset panel's conversations, over HTTP: each person's own, each message routed by the intent step and
recorded, and every write waiting for the person's confirm.

The deployment, Superset (`FakeSuperset` and `World`) and the build store are the build routes' own test fixtures;
the conversation store is the in-memory one. The Ask path is replaced by a stand-in stream at the one name the
conversation routes call it by, so these tests read what the routes do with its frames, never what it answers."""

from __future__ import annotations

import importlib
import json
import threading

import pytest

from mnemiq.server.sse import frame
from mnemiq.superset.chart_change import ChartChange, FakeInterpreter, Interpretation
from mnemiq.superset.intent import FakeIntentReader, Intent, Routed
from mnemiq.superset.planner import BuildPlan, FakePlanner, PlanRefused, RefusedPart
from tests.superset_build_world import Hook, MemoryConversationStore
from tests import test_superset_build_http as build_http
from tests.superset_shared import ONE_BAR
from tests.test_superset_build_http import (
    _a_built_dashboard,
    _bearer,
    _dashboard,
    _planned,
    _serve,
    _settings,
    _stored,
    _token,
    _whoami,
    _writes,
)

# The build routes' fixtures, collected here by name.
_no_mnemiq_environment = build_http._no_mnemiq_environment
clock, fake, jwks, model = build_http.clock, build_http.fake, build_http.jwks, build_http.model
no_direct_source, store, world = build_http.no_direct_source, build_http.store, build_http.world

PREFIX = "/v1/superset/conversations"


def _kept(talks) -> str:
    """Everything the conversation store holds, as text."""
    return json.dumps(list(talks._rows.values()), default=str)
SENTENCE = "West leads with 12 policies."
SQL = "SELECT region, COUNT(*) AS policies FROM policy GROUP BY region"


def _module():
    return importlib.import_module("mnemiq.server.superset_conversations")


@pytest.fixture
def talks(clock) -> MemoryConversationStore:
    return MemoryConversationStore(clock)


@pytest.fixture
def served(tmp_path, model, fake, world, store, talks, jwks, monkeypatch, no_direct_source):
    """The deployment, its build store and conversation store the memory ones, its default profile planning
    `ONE_BAR`, and the Ask path a stand-in (`_answering`)."""
    monkeypatch.setattr(_module(), "_open_conversations", lambda _settings: talks)
    client, runtime = _serve(_settings(tmp_path, model, fake, jwks), monkeypatch, store)
    _planned(runtime, FakePlanner(ONE_BAR))
    return client, runtime


def _routing(runtime, *routed) -> FakeIntentReader:
    reader = FakeIntentReader(list(routed) if len(routed) > 1 else routed[0])
    runtime.profiles[runtime.default_profile].intent = reader
    return reader


def _answering(monkeypatch, seen: list | None = None, *, fingerprint: str = "g1"):
    """The Ask path, standing in at the name the conversation routes call: what `chat_stream` yields for one answer,
    its rows ahead of the prose, and each call recorded."""

    async def chat_stream(runtime, identity, question, mode, heartbeat_s=15.0, history=None, profile=None,
                          delegation=None):
        if seen is not None:
            seen.append({"question": question, "history": history, "mode": mode, "profile": profile})
        yield frame({"type": "RUN_STARTED", "runId": "r1"})
        yield frame({"type": "CUSTOM", "name": "mnemiq.result", "value": {
            "sql": SQL, "tables_used": ["policy"], "preview": {"columns": ["region", "policies"],
                                                               "rows": [["West", 12]], "row_count": 1}}})
        yield frame({"type": "TEXT_MESSAGE_START", "messageId": "m1", "role": "assistant"})
        yield frame({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": SENTENCE})
        yield frame({"type": "TEXT_MESSAGE_END", "messageId": "m1"})
        yield frame({"type": "CUSTOM", "name": "mnemiq.answer", "value": {
            "answer": SENTENCE, "sql": SQL, "grant_fingerprint": fingerprint, "tables_used": ["policy"],
            "deferred": False, "failed": False}})
        yield frame({"type": "RUN_FINISHED", "runId": "r1"})

    monkeypatch.setattr(_module(), "chat_stream", chat_stream)


def _other() -> str:
    return _token(sub="east", preferred_username="east")


def _new(client, token: str | None = None) -> str:
    r = client.post(PREFIX, headers=_bearer(token or _token()))
    assert r.status_code == 201, r.text
    return r.json()["conversation"]["id"]


def _send(client, talk: str, text: str, token: str | None = None, **body):
    r = client.post(f"{PREFIX}/{talk}/messages", json={"text": text, **body}, headers=_bearer(token or _token()))
    assert r.status_code == 200, r.text
    return [json.loads(chunk[6:]) for chunk in r.text.split("\n\n") if chunk.startswith("data: ")]


def _custom(frames, name: str) -> dict:
    [found] = [f["value"] for f in frames if f.get("type") == "CUSTOM" and f.get("name") == name]
    return found


def _read(client, talk: str, token: str | None = None) -> dict:
    r = client.get(f"{PREFIX}/{talk}", headers=_bearer(token or _token()))
    assert r.status_code == 200, r.text
    return r.json()["conversation"]


def _confirm(client, talk: str, index: int, token: str | None = None, **body):
    return client.post(f"{PREFIX}/{talk}/messages/{index}/confirm", json=body, headers=_bearer(token or _token()))


def _dashboard_with_charts(client, world) -> tuple[int, int, str]:
    dashboard, chart = _a_built_dashboard(client)
    return dashboard, chart, world.dashboards[dashboard]["dashboard_title"]


# --- whose a conversation is


def test_another_persons_conversation_is_404_on_every_route(served, fake, world, store, talks, monkeypatch):
    """Every route, asked of a conversation that is not the asker's, finds nothing, says nothing of it, and changes
    nothing in it."""
    client, runtime = served
    _routing(runtime, Routed(Intent.ASK, "how many policies?"))
    _answering(monkeypatch)
    mine = _new(client)
    _send(client, mine, "how many policies?")
    before = _read(client, mine)
    for method, path, body in (("GET", f"{PREFIX}/{mine}", None),
                               ("POST", f"{PREFIX}/{mine}/delete", {}),
                               ("POST", f"{PREFIX}/{mine}/messages", {"text": "hello"}),
                               ("POST", f"{PREFIX}/{mine}/messages/1/confirm", {}),
                               ("POST", f"{PREFIX}/{mine}/messages/1/decline", {}),
                               ("POST", f"{PREFIX}/{mine}/messages/1/results", {})):
        r = client.request(method, path, json=body, headers=_bearer(_other()))
        assert r.status_code == 404, (path, r.text)
        assert r.json()["detail"] == "This person has no conversation of that id." and "how many" not in r.text
    assert client.get(PREFIX, headers=_bearer(_other())).json() == {"conversations": [], "next": None}
    assert _read(client, mine) == before


def test_a_conversation_lists_newest_first_and_deletes(served, talks, clock):
    client, _ = served
    first = _new(client)
    clock.advance(1)
    second = _new(client)
    listed = client.get(PREFIX, headers=_bearer(_token())).json()
    assert [c["id"] for c in listed["conversations"]] == [second, first] and listed["next"] is None
    page = client.get(PREFIX, params={"limit": 1}, headers=_bearer(_token())).json()
    assert [c["id"] for c in page["conversations"]] == [second] and page["next"]
    rest = client.get(PREFIX, params={"limit": 1, "before": page["next"]}, headers=_bearer(_token())).json()
    assert [c["id"] for c in rest["conversations"]] == [first]
    assert client.post(f"{PREFIX}/{first}/delete", headers=_bearer(_token())).status_code == 204
    assert client.get(f"{PREFIX}/{first}", headers=_bearer(_token())).status_code == 404


def test_the_first_message_titles_the_conversation(served, monkeypatch):
    client, runtime = served
    _routing(runtime, Routed(Intent.ASK, "how many policies by region?"))
    _answering(monkeypatch)
    talk = _new(client)
    _send(client, talk, "how many policies by region?")
    assert _read(client, talk)["title"] == "how many policies by region?"
    [listed] = client.get(PREFIX, headers=_bearer(_token())).json()["conversations"]
    assert listed["title"] == "how many policies by region?"


# --- asking


def test_an_ask_streams_as_chat_does_and_records_the_sentence_and_sql_not_the_rows(served, talks, monkeypatch):
    """An ask is routed, then the Ask path's frames pass on unchanged, its rows among them; the conversation keeps the
    question, the sentence and the SQL, and no row."""
    client, runtime = served
    seen: list = []
    _routing(runtime, Routed(Intent.ASK, "how many policies by region?"))
    _answering(monkeypatch, seen)
    talk = _new(client)
    frames = _send(client, talk, "how many policies by region?", dashboard_id=7, dashboard_title="Sales",
                   dashboard_has_charts=True)
    assert _custom(frames, "mnemiq.routed") == {"intent": "ask", "text": "how many policies by region?",
                                                "index": 0}
    assert _custom(frames, "mnemiq.result")["preview"]["rows"] == [["West", 12]]
    assert [f["delta"] for f in frames if f.get("type") == "TEXT_MESSAGE_CONTENT"] == [SENTENCE]
    assert _custom(frames, "mnemiq.recorded") == {"index": 1}
    assert frames[-1]["type"] == "RUN_FINISHED"
    assert seen[0]["question"] == "how many policies by region?" and seen[0]["history"] == []
    question, answer = _read(client, talk)["messages"]
    assert (question["role"], question["kind"], question["text"], question["dashboard"]) == (
        "person", "ask", "how many policies by region?", {"id": 7, "title": "Sales"})
    assert answer["answer"] == {"sentence": SENTENCE, "sql": SQL, "grant_fingerprint": "g1", "tables_used": ["policy"]}
    assert "West" not in _kept(talks).replace(SENTENCE, "")


def test_an_ask_carries_the_earlier_answers_as_history_with_their_fingerprints_and_no_rows(served, monkeypatch):
    """A follow-up's history is the conversation's earlier answers: each question with its SQL and the grant
    fingerprint it was answered under, which the Ask path replays only under the same boundary; never a row."""
    client, runtime = served
    seen: list = []
    _routing(runtime, Routed(Intent.ASK, "x"))
    _answering(monkeypatch, seen, fingerprint="g1")
    talk = _new(client)
    _send(client, talk, "how many policies by region?")
    _send(client, talk, "and by channel?")
    [turn] = seen[1]["history"]
    assert (turn.question, turn.sql, turn.grant_fingerprint, turn.rows, turn.columns) == (
        "how many policies by region?", SQL, "g1", [], [])


# --- building


def test_a_build_message_proposes_and_writes_nothing(served, fake, world, store):
    """A message routed to a build onto the dashboard on screen plans it and keeps it waiting for "Build it": the
    proposal frame and its record name the plan, and nothing is written to Superset."""
    client, runtime = served
    _dashboard(world, 900, title="Sales")
    _routing(runtime, Routed(Intent.BUILD_ONTO, "policies by region as a bar chart"))
    talk = _new(client)
    frames = _send(client, talk, "add that here", dashboard_id=900, dashboard_title="Sales",
                   dashboard_has_charts=False)
    proposal = _custom(frames, "mnemiq.proposal")
    assert proposal["index"] == 1 and proposal["dashboard_id"] == 900
    assert proposal["goal"] == "policies by region as a bar chart"
    assert [chart["title"] for chart in proposal["plan"]["charts"]] == [c.title for c in ONE_BAR.charts]
    assert _writes(fake) == []
    asked, proposed = _read(client, talk)["messages"]
    assert (asked["kind"], asked["intent"], asked["goal"]) == ("build", "build_onto",
                                                               "policies by region as a bar chart")
    assert proposed["outcome"] == {"state": "proposed"} and proposed["proposal"]["build_id"] == proposal["build_id"]
    tenant, principal = _whoami(client, _token())
    assert store.load_build(tenant, proposal["build_id"], principal).status == "proposed"


def test_confirm_runs_the_proposal_once(served, fake, world, store):
    client, runtime = served
    _routing(runtime, Routed(Intent.BUILD_NEW, "policies by region"))
    talk = _new(client)
    proposal = _custom(_send(client, talk, "build a dashboard of policies by region"), "mnemiq.proposal")
    first = _confirm(client, talk, proposal["index"], dashboard_id=None)
    assert first.status_code == 200, first.text
    assert first.json()["build"]["status"] == "building"
    again = _confirm(client, talk, proposal["index"], dashboard_id=None)
    assert again.status_code == 409 and again.json()["detail"] == "This was already done."
    assert _read(client, talk)["messages"][proposal["index"]]["outcome"]["state"] == "done"
    assert len([b for b in store._builds.values() if b["status"] == "building"]) == 1


def test_confirm_needs_the_proposals_dashboard_on_screen(served, fake, world, store):
    """A proposal for a dashboard is confirmed only from that dashboard: from anywhere else it is a 409 that leaves it
    waiting, and from the right one it then goes ahead. A new dashboard's proposal takes any page, none included."""
    client, runtime = served
    _dashboard(world, 900, title="Sales")
    _routing(runtime, Routed(Intent.BUILD_ONTO, "policies by region"))
    talk = _new(client)
    proposal = _custom(_send(client, talk, "add it", dashboard_id=900, dashboard_has_charts=True), "mnemiq.proposal")
    for elsewhere in (None, 901):
        wrong = _confirm(client, talk, proposal["index"], dashboard_id=elsewhere)
        assert wrong.status_code == 409 and wrong.json()["detail"] == "Open the dashboard this was for, then confirm."
        assert _read(client, talk)["messages"][proposal["index"]]["outcome"] == {"state": "proposed"}
    assert _confirm(client, talk, proposal["index"], dashboard_id=900).status_code == 200
    _routing(runtime, Routed(Intent.BUILD_NEW, "claims by region"))
    fresh = _custom(_send(client, talk, "a new dashboard of claims"), "mnemiq.proposal")
    assert _confirm(client, talk, fresh["index"], dashboard_id=None).status_code == 200


@pytest.mark.parametrize("meanwhile", ["busy", "changed"])
def test_a_busy_or_changed_dashboard_plans_again_and_withdraws_the_old_proposal(served, fake, world, store, clock,
                                                                                meanwhile):
    """At "Build it", a dashboard another build is building, or one changed since the plan, is said, the old proposal
    is withdrawn, and the goal is planned again as a new message that waits for its own confirm."""
    client, runtime = served
    _dashboard(world, 900, title="Sales")
    _routing(runtime, Routed(Intent.BUILD_ONTO, "policies by region"))
    talk = _new(client)
    proposal = _custom(_send(client, talk, "add it", dashboard_id=900, dashboard_has_charts=True), "mnemiq.proposal")
    if meanwhile == "busy":
        _stored(store, client, dashboard_id=900)
        said = "The dashboard is being built; this was planned again."
    else:
        clock.advance(5)
        world.dashboards[900]["changed_on"] = world.superset_now()
        said = "The dashboard changed since this plan; it was planned again."
    r = _confirm(client, talk, proposal["index"], dashboard_id=900)
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == said
    again = r.json()["message"]
    assert again["index"] == proposal["index"] + 1 and again["proposal"]["goal"] == "policies by region"
    messages = _read(client, talk)["messages"]
    assert messages[proposal["index"]]["outcome"]["state"] == "replanned"
    assert messages[again["index"]]["outcome"] == {"state": "proposed"}
    tenant, principal = _whoami(client, _token())
    assert store.load_build(tenant, proposal["build_id"], principal) is None
    assert store.load_build(tenant, again["proposal"]["build_id"], principal).status == "proposed"
    assert _writes(fake) == []


def test_a_build_message_offered_a_dashboard_records_the_offer_and_a_choice_proposes(served, fake, world, store):
    import dataclasses

    client, runtime = served
    _dashboard(world, 900, title="Policies at a glance")
    offer = dataclasses.replace(ONE_BAR, reuse_candidates=(900,), reuse_reason="It already shows policies.")
    _planned(runtime, FakePlanner([offer, ONE_BAR]))
    _routing(runtime, Routed(Intent.BUILD_NEW, "policies by region"))
    talk = _new(client)
    offered = _custom(_send(client, talk, "build me policies by region"), "mnemiq.offer")
    assert (offered["dashboard_id"], offered["title"], offered["choices"]) == (
        900, "Policies at a glance", ["open", "add", "new"])
    assert _read(client, talk)["messages"][offered["index"]]["outcome"] == {"state": "offered"}
    chose = _confirm(client, talk, offered["index"], dashboard_id=None, choice="new")
    assert chose.status_code == 200, chose.text
    proposed = chose.json()["message"]
    assert proposed["proposal"]["dashboard_id"] is None and proposed["outcome"] == {"state": "proposed"}
    assert _read(client, talk)["messages"][offered["index"]]["outcome"]["state"] == "chosen"
    assert _writes(fake) == []


def test_a_refused_or_already_shown_build_is_a_note_and_writes_nothing(served, fake, world, store):
    client, runtime = served
    _planned(runtime, FakePlanner(PlanRefused(reason="nothing here measures that",
                                              parts=(RefusedPart(part="churn", reason="no table holds it"),))))
    _routing(runtime, Routed(Intent.BUILD_NEW, "churn by plan"))
    talk = _new(client)
    refused = _custom(_send(client, talk, "build churn by plan"), "mnemiq.refused")
    assert refused["parts"] == [{"part": "churn", "reason": "no table holds it"}]
    note = _read(client, talk)["messages"][refused["index"]]
    assert note["kind"] == "note" and "outcome" not in note
    assert _writes(fake) == [] and store._builds == {}


# --- changing


def test_a_change_message_previews_and_writes_nothing(served, fake, world, store):
    client, runtime = served
    dashboard, chart, title = _dashboard_with_charts(client, world)
    runtime.profiles[runtime.default_profile].interpreter = FakeInterpreter(Interpretation(
        changes=(ChartChange(chart_id=chart, set={"row_limit": 2}),)))
    _routing(runtime, Routed(Intent.CHANGE, "only the top two regions on the policies chart"))
    talk = _new(client)
    before = len(world.calls("patch_chart"))
    preview = _custom(_send(client, talk, "top two", dashboard_id=dashboard, dashboard_title=title,
                            dashboard_has_charts=True), "mnemiq.preview")
    assert [edit["chart_id"] for edit in preview["edits"]] == [chart] and preview["index"] == 1
    assert len(world.calls("patch_chart")) == before
    applied = _confirm(client, talk, preview["index"], dashboard_id=dashboard)
    assert applied.status_code == 200, applied.text
    assert applied.json()["change"]["changed"] == [chart] and applied.json()["state"] == "done"
    assert world.charts[chart]["params"]["row_limit"] == 2


def test_two_confirms_at_once_apply_once(served, fake, world, store):
    client, runtime = served
    dashboard, chart, title = _dashboard_with_charts(client, world)
    runtime.profiles[runtime.default_profile].interpreter = FakeInterpreter(Interpretation(
        changes=(ChartChange(chart_id=chart, set={"row_limit": 2}),)))
    _routing(runtime, Routed(Intent.CHANGE, "top two"))
    talk = _new(client)
    preview = _custom(_send(client, talk, "top two", dashboard_id=dashboard, dashboard_has_charts=True),
                      "mnemiq.preview")
    before = len(world.calls("patch_chart"))
    start = threading.Barrier(2)
    answers: list = []

    def confirm() -> None:
        start.wait()
        answers.append(_confirm(client, talk, preview["index"], dashboard_id=dashboard).status_code)

    threads = [threading.Thread(target=confirm) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(answers) == [200, 409]
    assert len(world.calls("patch_chart")) == before + 1


def test_a_change_stopped_partway_is_recorded_partial_with_both_lists(served, fake, world, store):
    """Two charts previewed; the second edited by hand between the check and its write: the first is changed, the
    second left, and the record says partial, naming both."""
    from mnemiq.superset.chart_change import STALE_SINCE_PREVIEW
    from tests.superset_shared import POLICIES, _chart
    from mnemiq.superset.charts import Family, WrittenRef

    client, runtime = served
    by_channel = _chart("policies_by_channel", Family.BAR, "Policies by channel", 12, (WrittenRef(POLICIES),),
                        dimensions=("channel",))
    from tests.superset_shared import BY_REGION

    _planned(runtime, FakePlanner(BuildPlan(charts=(BY_REGION, by_channel), measures=(POLICIES,))))
    dashboard, first = _a_built_dashboard(client)
    second = next(i for i, c in world.charts.items() if c["slice_name"] == "Policies by channel")
    runtime.profiles[runtime.default_profile].interpreter = FakeInterpreter(Interpretation(changes=(
        ChartChange(chart_id=first, set={"row_limit": 2}), ChartChange(chart_id=second, set={"row_limit": 3}))))
    _routing(runtime, Routed(Intent.CHANGE, "fewer rows on both"))
    talk = _new(client)
    preview = _custom(_send(client, talk, "fewer rows", dashboard_id=dashboard, dashboard_has_charts=True),
                      "mnemiq.preview")

    def hand_edit(request, _world):
        if request["chart_id"] == second:
            world.charts[second]["params"] = {**world.charts[second]["params"], "color_scheme": "by hand"}
            world.charts[second]["changed_on"] = world._stamp()

    world.hooks["patch_chart"] = Hook(before=hand_edit)
    r = _confirm(client, talk, preview["index"], dashboard_id=dashboard)
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "partial"
    outcome = _read(client, talk)["messages"][preview["index"]]["outcome"]
    assert outcome["state"] == "partial" and outcome["changed"] == [first]
    assert outcome["left"] == [{"chart_id": second, "reason": STALE_SINCE_PREVIEW}]


# --- a question back


def test_an_unreadable_intent_asks_the_person_and_writes_nothing(served, fake, world, store):
    client, runtime = served
    _dashboard(world, 900)
    _routing(runtime, None)
    talk = _new(client)
    frames = _send(client, talk, "fix it", dashboard_id=900, dashboard_has_charts=True)
    clarify = _custom(frames, "mnemiq.clarify")
    assert clarify["choices"] == ["ask", "build", "change"] and clarify["question"]
    assert _writes(fake) == [] and store._builds == {}
    asked, back = _read(client, talk)["messages"]
    assert asked["intent"] == "unclear" and back["kind"] == "clarify"


def test_choose_after_a_clarify_routes_as_chosen(served, fake, world, store, monkeypatch):
    """Choosing after a question back skips the intent step: the message goes where the person chose, as typed."""
    client, runtime = served
    _dashboard(world, 900, title="Sales")
    reader = _routing(runtime, None)
    _answering(monkeypatch)
    talk = _new(client)
    assert _custom(_send(client, talk, "policies by region", choose="ask"), "mnemiq.routed")["intent"] == "ask"
    built = _send(client, talk, "policies by region", choose="build", dashboard_id=900, dashboard_has_charts=True)
    assert _custom(built, "mnemiq.routed") == {"intent": "build_onto", "text": "policies by region", "index": 2}
    assert _custom(built, "mnemiq.proposal")["goal"] == "policies by region"
    assert reader.reads == []
    off = _send(client, talk, "make it a bar", choose="change")
    assert _custom(off, "mnemiq.clarify")["question"] == "Which dashboard? Open it, then ask again."


# --- a reopened answer


def test_results_rerun_as_the_person_now_and_store_nothing(served, fake, world, talks, monkeypatch):
    client, runtime = served
    _routing(runtime, Routed(Intent.ASK, "x"))
    _answering(monkeypatch)
    talk = _new(client)
    _send(client, talk, "how many policies by region?")
    kept = _kept(talks)
    r = client.post(f"{PREFIX}/{talk}/messages/1/results", headers=_bearer(_token()))
    assert r.status_code == 200, r.text
    shown = r.json()
    assert shown["columns"] == ["region", "policies"] and shown["rows"]
    assert any(call.tool == "execute_sql" and "GROUP BY region" in call.request.get("sql", "")
               for call in fake.calls)
    assert _kept(talks) == kept
    assert client.post(f"{PREFIX}/{talk}/messages/0/results", headers=_bearer(_token())).status_code == 404


def test_results_after_access_narrowed_answer_the_ask_paths_refusal(served, fake, world, monkeypatch):
    """The person can no longer read the answer's table: the Ask path's own refusal sentence, and no row."""
    client, runtime = served
    _routing(runtime, Routed(Intent.ASK, "x"))
    _answering(monkeypatch)
    talk = _new(client)
    _send(client, talk, "how many policies by region?")
    world.hooks["execute_sql"] = Hook(instead=lambda request, _world: (
        {"success": False, "error_type": "TABLE_SECURITY_ACCESS_ERROR", "error": "no"}
        if "policy" in request.get("sql", "") else None))
    r = client.post(f"{PREFIX}/{talk}/messages/1/results", headers=_bearer(_token()))
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"refused"}
    assert r.json()["refused"] == "Answering this would require access to 'policy', which you do not have."


def test_decline_withdraws_the_proposal(served, fake, world, store):
    client, runtime = served
    _routing(runtime, Routed(Intent.BUILD_NEW, "policies by region"))
    talk = _new(client)
    proposal = _custom(_send(client, talk, "build policies by region"), "mnemiq.proposal")
    r = client.post(f"{PREFIX}/{talk}/messages/{proposal['index']}/decline", headers=_bearer(_token()))
    assert r.status_code == 200 and r.json() == {"state": "declined"}
    tenant, principal = _whoami(client, _token())
    assert store.load_build(tenant, proposal["build_id"], principal) is None
    assert _confirm(client, talk, proposal["index"], dashboard_id=None).status_code == 409


def test_deleting_a_conversation_withdraws_its_waiting_proposals(served, fake, world, store):
    client, runtime = served
    _routing(runtime, Routed(Intent.BUILD_NEW, "policies by region"))
    talk = _new(client)
    proposal = _custom(_send(client, talk, "build policies by region"), "mnemiq.proposal")
    assert client.post(f"{PREFIX}/{talk}/delete", headers=_bearer(_token())).status_code == 204
    tenant, principal = _whoami(client, _token())
    assert store.load_build(tenant, proposal["build_id"], principal) is None


# --- the review's fixes: removals held to the preview, failures that strand nothing, answers paired with questions


def _previewing_a_removal(client, runtime, world, *, with_edit: bool):
    dashboard, chart, title = _dashboard_with_charts(client, world)
    changes = (ChartChange(chart_id=chart, set={"row_limit": 2}),) if with_edit else ()
    runtime.profiles[runtime.default_profile].interpreter = FakeInterpreter(Interpretation(
        changes=changes, remove_chart_ids=(chart,)))
    _routing(runtime, Routed(Intent.CHANGE, "drop the policies chart"))
    talk = _new(client)
    preview = _custom(_send(client, talk, "drop it", dashboard_id=dashboard, dashboard_has_charts=True),
                      "mnemiq.preview")
    return talk, preview, dashboard, chart


def test_a_confirmed_removal_from_a_fresh_preview_removes_the_chart(served, fake, world, store):
    client, runtime = served
    talk, preview, dashboard, chart = _previewing_a_removal(client, runtime, world, with_edit=False)
    r = _confirm(client, talk, preview["index"], dashboard_id=dashboard, remove_chart_ids=[chart])
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "done" and r.json()["change"]["removed"] == [chart]
    assert chart in world.trashed


def test_a_stale_preview_removes_nothing_through_the_confirm(served, fake, world, store):
    from mnemiq.superset.chart_change import STALE_SINCE_PREVIEW

    client, runtime = served
    talk, preview, dashboard, chart = _previewing_a_removal(client, runtime, world, with_edit=False)
    world.charts[chart]["params"] = {**world.charts[chart]["params"], "color_scheme": "by hand"}
    world.charts[chart]["changed_on"] = world._stamp()
    r = _confirm(client, talk, preview["index"], dashboard_id=dashboard, remove_chart_ids=[chart])
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "stale" and chart not in world.trashed
    assert "delete_chart" not in [call.tool for call in fake.calls]
    outcome = _read(client, talk)["messages"][preview["index"]]["outcome"]
    assert {"chart_id": chart, "reason": STALE_SINCE_PREVIEW} in outcome["left"]


def test_only_a_chart_the_preview_would_remove_is_confirmed_for_removal(served, fake, world, store):
    client, runtime = served
    talk, preview, dashboard, chart = _previewing_a_removal(client, runtime, world, with_edit=False)
    r = _confirm(client, talk, preview["index"], dashboard_id=dashboard, remove_chart_ids=[chart + 1000])
    assert r.status_code == 422, r.text
    assert _read(client, talk)["messages"][preview["index"]]["outcome"] == {"state": "proposed"}


def test_a_change_that_fails_before_writing_can_be_confirmed_again(served, fake, world, store):
    """Superset failing on the confirm's first read (the dashboard's layout): nothing was written, so the preview
    waits again. (A chart that cannot be read is no failure of the confirm: it reads as changed, and nothing is
    written.)"""
    client, runtime = served
    dashboard, chart, _ = _dashboard_with_charts(client, world)
    runtime.profiles[runtime.default_profile].interpreter = FakeInterpreter(Interpretation(
        changes=(ChartChange(chart_id=chart, set={"row_limit": 2}),)))
    _routing(runtime, Routed(Intent.CHANGE, "top two"))
    talk = _new(client)
    preview = _custom(_send(client, talk, "top two", dashboard_id=dashboard, dashboard_has_charts=True),
                      "mnemiq.preview")
    world.hooks["get_layout"] = Hook(before=lambda request, _w: (_ for _ in ()).throw(RuntimeError("down")))
    failed = _confirm(client, talk, preview["index"], dashboard_id=dashboard)
    assert failed.status_code >= 400, failed.text
    assert _read(client, talk)["messages"][preview["index"]]["outcome"] == {"state": "proposed"}
    world.hooks.pop("get_layout")
    again = _confirm(client, talk, preview["index"], dashboard_id=dashboard)
    assert again.status_code == 200 and again.json()["state"] == "done", again.text


def test_a_change_interrupted_mid_write_is_recorded_partial(served, fake, world, store):
    from mnemiq.superset.chart_change import NOT_ANSWERED

    client, runtime = served
    dashboard, chart, _ = _dashboard_with_charts(client, world)
    runtime.profiles[runtime.default_profile].interpreter = FakeInterpreter(Interpretation(
        changes=(ChartChange(chart_id=chart, set={"row_limit": 2}),)))
    _routing(runtime, Routed(Intent.CHANGE, "top two"))
    talk = _new(client)
    preview = _custom(_send(client, talk, "top two", dashboard_id=dashboard, dashboard_has_charts=True),
                      "mnemiq.preview")
    world.hooks["patch_chart"] = Hook(before=lambda request, _w: (_ for _ in ()).throw(RuntimeError("down")))
    _confirm(client, talk, preview["index"], dashboard_id=dashboard)
    outcome = _read(client, talk)["messages"][preview["index"]]["outcome"]
    assert outcome["state"] == "partial"
    assert outcome["left"] == [{"chart_id": chart, "reason": NOT_ANSWERED}]
    again = _confirm(client, talk, preview["index"], dashboard_id=dashboard)
    assert again.status_code == 409 and again.json()["detail"] == "This was already done."


def test_a_build_left_applying_is_confirmed_from_the_builds_own_state(served, fake, world, store, talks):
    """A server that stopped between claiming a "Build it" and recording it: the build store says whether it was
    confirmed, so the message is set right rather than left saying it was done."""
    client, runtime = served
    _routing(runtime, Routed(Intent.BUILD_NEW, "policies by region"))
    talk = _new(client)
    proposal = _custom(_send(client, talk, "build policies by region"), "mnemiq.proposal")
    tenant, principal = _whoami(client, _token())
    assert talks.settle(tenant, talk, principal, proposal["index"], {"state": "applying"}, expect="proposed")
    r = _confirm(client, talk, proposal["index"], dashboard_id=None)
    assert r.status_code == 200, r.text
    assert r.json()["build"]["status"] == "building"
    assert _read(client, talk)["messages"][proposal["index"]]["outcome"]["state"] == "done"


def test_a_change_left_applying_says_it_may_or_may_not_be_done(served, fake, world, store, talks):
    client, runtime = served
    dashboard, chart, _ = _dashboard_with_charts(client, world)
    runtime.profiles[runtime.default_profile].interpreter = FakeInterpreter(Interpretation(
        changes=(ChartChange(chart_id=chart, set={"row_limit": 2}),)))
    _routing(runtime, Routed(Intent.CHANGE, "top two"))
    talk = _new(client)
    preview = _custom(_send(client, talk, "top two", dashboard_id=dashboard, dashboard_has_charts=True),
                      "mnemiq.preview")
    tenant, principal = _whoami(client, _token())
    talks.settle(tenant, talk, principal, preview["index"], {"state": "applying"}, expect="proposed")
    r = _confirm(client, talk, preview["index"], dashboard_id=dashboard)
    assert r.status_code == 409
    assert r.json()["detail"] == "This may or may not have been done; check the dashboard."


def test_a_worker_that_times_out_ends_the_stream(served, fake, world, store):
    """A plan whose own work raises TimeoutError ends the stream with its error, never an endless keep-alive."""
    client, runtime = served

    class Late:
        def plan(self, goal, context):
            raise TimeoutError("the model took too long")

    _planned(runtime, Late())
    _routing(runtime, Routed(Intent.BUILD_NEW, "policies by region"))
    talk = _new(client)
    done: list = []
    sender = threading.Thread(target=lambda: done.append(_send(client, talk, "build policies by region")))
    sender.start()
    sender.join(30)
    assert not sender.is_alive(), "the stream did not end"
    assert done[0][-1]["type"] == "RUN_ERROR"


def test_each_answer_is_kept_with_the_question_it_answers(served, monkeypatch):
    """Two tabs asking at once: each answer names the question it replies to, and a follow-up's history pairs them by
    that, never by their order."""
    client, runtime = served
    _routing(runtime, Routed(Intent.ASK, "x"))
    _answering(monkeypatch)
    talk = _new(client)
    _send(client, talk, "how many policies by region?")
    [question, answer] = _read(client, talk)["messages"]
    assert answer["replies_to"] == question["index"]
    mod = _module()
    from mnemiq.superset.conversations import Conversation, Message

    docs = [{"role": "person", "kind": "ask", "text": "Q1"}, {"role": "person", "kind": "ask", "text": "Q2"},
            {"role": "mnemiq", "kind": "answer", "text": "", "replies_to": 0,
             "answer": {"sentence": "a1", "sql": "SELECT 1", "grant_fingerprint": "g", "tables_used": []}},
            {"role": "mnemiq", "kind": "answer", "text": "", "replies_to": 1,
             "answer": {"sentence": "a2", "sql": "SELECT 2", "grant_fingerprint": "g", "tables_used": []}}]
    talk_ = Conversation(conversation_id="c", title="", created_at="t", updated_at="t", messages=tuple(
        Message(i, d["role"], d["kind"], d["text"], "t", d) for i, d in enumerate(docs)))
    assert [(turn.question, turn.sql) for turn in mod._history(talk_)] == [("Q1", "SELECT 1"), ("Q2", "SELECT 2")]


def test_an_ask_whose_reader_left_keeps_its_answer():
    """The panel closed or reloaded mid-answer: the Ask path's stream is read to its end all the same, and its answer
    kept, so the question is never left with nothing after it."""
    import asyncio

    mod = _module()
    kept: list = []

    async def answer():
        yield frame({"type": "RUN_STARTED", "runId": "r"})
        await asyncio.sleep(0.05)
        yield frame({"type": "CUSTOM", "name": "mnemiq.answer", "value": {"answer": SENTENCE, "sql": SQL}})
        yield frame({"type": "RUN_FINISHED", "runId": "r"})

    async def run():
        out: asyncio.Queue = asyncio.Queue()
        task = asyncio.ensure_future(mod._ask_and_keep(answer(), keep_answer=lambda value: kept.append(value) or 1,
                                                       keep_note=lambda text: kept.append(text) or 1, out=out))
        await out.get()  # the reader reads one frame, then leaves
        await task
        return task

    asyncio.run(run())
    assert kept == [{"answer": SENTENCE, "sql": SQL}]


def test_a_kept_offer_is_in_the_conversations_words(served, fake, world, store):
    import dataclasses

    client, runtime = served
    _dashboard(world, 900, title="Policies at a glance")
    offer = dataclasses.replace(ONE_BAR, reuse_candidates=(900,), reuse_reason="It already shows policies.")
    _planned(runtime, FakePlanner([offer, ONE_BAR]))
    _routing(runtime, Routed(Intent.BUILD_NEW, "policies by region"))
    talk = _new(client)
    offered = _custom(_send(client, talk, "build me policies by region"), "mnemiq.offer")
    text = _read(client, talk)["messages"][offered["index"]]["text"]
    assert "dashboard_id" not in text and "fresh" not in text and "Policies at a glance" in text
