"""Cards fitted to the model's window once every column is described (M127).

Describing every column (M112) took a design partner's largest prompt to 40,293 tokens against a
28,672-token window, and every question reaching their wide tables was refused. These pin what
the fit promises: nothing changes while a prompt fits; past the window every column keeps its name
and type, and descriptions go first to the columns the question names; and both doors -- the
product's `Runtime.ask` and the eval's `build_engine` -- fit the same way.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import duckdb
import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.config import Settings
from mnemiq.contract import Column, Definition, JoinKey, Relationship, Snapshot, SourceBinding
from mnemiq.generate.prompts import user_prompt
from mnemiq.llm import window
from mnemiq.semantic.cards import build_cards
from mnemiq.llm.window import TEMPLATE_ALLOWANCE
from mnemiq.semantic.fit import FEEDBACK_ALLOWANCE, PromptFitter, rank_columns
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.policy import build_access_policy

REPLY = 4000


def _snapshot(width: int = 60) -> Snapshot:
    """A wide table and a narrow one, every column described at length, joined on lot_id."""
    tables = {"wide_trace": width, "lot": 3}
    columns = []
    for table, n in tables.items():
        names = ["lot_id", "body_weight", "grade_code"] + [f"step_{i:03d}_time" for i in range(n - 3)]
        for name in names[:n]:
            columns.append(Column(id=f"{table}.{name}", object_id=table, name=name, data_type="text",
                                  description=f"The {name.replace('_', ' ')} recorded for each lot "
                                              f"in {table}, kept as written by the line system."))
    return Snapshot(
        version="v1", source_id="s", created_at="2026-10-03T00:00:00Z",
        source_bindings=[SourceBinding(id=f"sb:{t}", source_id="s", object_id=t, source_object=t,
                                       binding_type="table") for t in tables],
        columns=columns,
        relationships=[Relationship(**{"id": "r1", "from": "wide_trace", "to": "lot",
                                       "cardinality": "many_to_one",
                                       "join_keys": [JoinKey(left="lot_id", right="lot_id")]})],
        definitions=[Definition(id="d1", term="defect", domain="plant",
                                definition="A wafer is defective when GRADE_CODE is not A00.",
                                bound_objects=["wide_trace"])],
    )


def _grants(snap: Snapshot) -> GrantSet:
    return window.everything(snap)


def _packet(snap: Snapshot, question: str = "total body weight per lot") -> ContextPacket:
    grants = _grants(snap)
    cards = build_cards(snap, policy=build_access_policy(snap, grants))
    return ContextPacket(question=question, grant_fingerprint="fp", enrichment_version=None,
                         cards=[RetrievedCard(object_id=c.object_id, card=c.text, score=1.0)
                                for c in cards],
                         definitions=list(snap.definitions))


def _fitter(window_tokens: int | None, count=None) -> PromptFitter:
    return PromptFitter(dialect="duckdb", card_style="cards", reply_tokens=REPLY,
                        window=window_tokens, count=count)


def _tokens(fitter: PromptFitter, packet: ContextPacket) -> int:
    """A quarter of the characters: the counting server these tests stand in for."""
    return (len(fitter.system()) + len(user_prompt(packet))) // 4


# --- the renderer -------------------------------------------------------------------------------

def test_describe_withholds_only_the_description():
    snap = _snapshot(5)
    full = {c.object_id: c.text for c in build_cards(snap)}
    bare = {c.object_id: c.text for c in build_cards(snap, describe=lambda t, c: False)}
    for column in snap.columns:
        assert f"- {column.name} (text)" in bare[column.object_id], "name and type stay"
        assert column.description not in bare[column.object_id]
        assert column.description in full[column.object_id]
    assert {c.object_id: c.text for c in build_cards(snap, describe=None)} == full


# --- nothing changes while it fits --------------------------------------------------------------

def test_a_prompt_that_fits_comes_back_as_the_same_packet_without_asking_the_server():
    snap = _snapshot()
    packet = _packet(snap)
    asked = []
    fitter = _fitter(10_000_000, count=lambda s, u: asked.append(1))
    assert fitter.fit(packet, snap, _grants(snap)) is packet
    assert asked == [], "a conservative estimate that fits needs no count"


def test_the_server_count_decides_when_the_estimate_is_over():
    """The estimate errs long (2.5 characters a token); the server's count is the real one."""
    snap = _snapshot()
    packet = _packet(snap)
    fitter = _fitter(None)
    chars = len(fitter.system()) + len(user_prompt(packet))
    window_tokens = math.ceil(chars / 2.5) + REPLY + FEEDBACK_ALLOWANCE - 1  # the estimate misses
    fitter = _fitter(window_tokens, count=lambda s, u: (len(s + u) // 4, window_tokens))
    assert fitter.fit(packet, snap, _grants(snap)) is packet


def test_without_a_known_window_nothing_changes():
    snap = _snapshot()
    packet = _packet(snap)
    assert _fitter(None, count=lambda s, u: None).fit(packet, snap, _grants(snap)) is packet


def test_a_failing_count_or_fit_never_stops_a_question(caplog):
    snap = _snapshot()
    packet = _packet(snap)

    def boom(system, user):
        raise RuntimeError("tokenize exploded")

    with caplog.at_level(logging.WARNING):
        assert _fitter(1_000, count=boom).fit(packet, snap, _grants(snap)) is packet
    assert "sending them as retrieved" in caplog.text


# --- past the window ----------------------------------------------------------------------------

def test_past_the_window_every_column_keeps_its_name_and_the_prompt_fits():
    snap = _snapshot()
    packet = _packet(snap)
    probe = _fitter(None)
    full = _tokens(probe, packet)
    window_tokens = full + REPLY + FEEDBACK_ALLOWANCE - full // 3  # a third too small
    fitter = _fitter(window_tokens, count=lambda s, u: (len(s + u) // 4, window_tokens))

    fitted = fitter.fit(packet, snap, _grants(snap))

    assert fitted is not packet
    sent, described = fitted.descriptions
    assert 0 < sent < described == len(snap.columns)
    text = "\n".join(c.card for c in fitted.cards)
    assert all(f"- {c.name} (text)" in text for c in snap.columns), "no column became unfindable"
    assert _tokens(fitter, fitted) + REPLY + FEEDBACK_ALLOWANCE <= window_tokens


def test_the_columns_the_question_and_definitions_name_keep_their_descriptions():
    snap = _snapshot()
    packet = _packet(snap, question="total BODY_WEIGHT per lot")
    probe = _fitter(None)
    bare = replace(packet, cards=[RetrievedCard(object_id=c.object_id, card=c.text, score=1.0)
                                  for c in build_cards(snap, describe=lambda t, c: False)])
    one = len(snap.columns[0].description) // 4  # a description's tokens, at the server's rate
    # The descriptionless floor plus room for about six descriptions, with the fit's margin.
    window_tokens = math.ceil((_tokens(probe, bare) + 6 * one) * 1.05) + REPLY + FEEDBACK_ALLOWANCE
    fitter = _fitter(window_tokens, count=lambda s, u: (len(s + u) // 4, window_tokens))

    fitted = fitter.fit(packet, snap, _grants(snap))
    wide = next(c.card for c in fitted.cards if c.object_id == "wide_trace")
    described = {c.name for c in snap.columns
                 if c.object_id == "wide_trace" and c.description in wide}
    assert {"body_weight", "grade_code"} <= described, "named by the question and a definition"
    assert len(described) < 20


def test_the_ranking_is_named_then_join_keys_then_shared_words():
    snap = _snapshot(8)
    packet = _packet(snap, question="when did step 004 happen for BODY_WEIGHT")
    ranked = rank_columns(packet, snap, build_access_policy(snap, _grants(snap)))
    position = {key: i for i, key in enumerate(ranked)}
    named = [("wide_trace", "body_weight"), ("wide_trace", "grade_code")]  # question, definition
    joins = [("wide_trace", "lot_id"), ("lot", "lot_id")]
    worded = ("wide_trace", "step_004_time")  # "step" and "004" are question words
    assert max(position[k] for k in named) < min(position[k] for k in joins)
    assert max(position[k] for k in joins) < position[worded]
    assert position[worded] < position[("wide_trace", "step_000_time")]


def test_a_window_too_small_for_names_alone_sends_none_and_says_so(caplog):
    snap = _snapshot()
    packet = _packet(snap)
    fitter = _fitter(REPLY + FEEDBACK_ALLOWANCE + 10,
                     count=lambda s, u: (len(s + u) // 4, REPLY + FEEDBACK_ALLOWANCE + 10))
    with caplog.at_level(logging.WARNING):
        fitted = fitter.fit(packet, snap, _grants(snap))
    assert fitted.descriptions == (0, len(snap.columns))
    assert "even with no column descriptions" in caplog.text


# --- both doors ---------------------------------------------------------------------------------

def test_both_doors_fit_the_packet_before_the_agent_sees_it():
    """The two-doors scan, for the fit: an eval that skipped it would measure prompts `ask` no
    longer sends, and the reverse would leave the product refusing what the eval answered."""
    import ast
    import inspect
    import textwrap

    import mnemiq.runtime as runtime_module
    from mnemiq.eval.engine import build_engine

    def fits_before_answer(function) -> bool:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)]
        fit = [n.lineno for n in calls if n.func.attr == "fit"]
        answer = [n.lineno for n in calls if n.func.attr == "answer"]
        return bool(fit) and bool(answer) and min(fit) < min(answer)

    assert fits_before_answer(runtime_module.Runtime.ask)
    assert fits_before_answer(build_engine)


def test_runtime_ask_hands_the_agent_the_fitted_packet(monkeypatch, tmp_path):
    """The join, not the ends: the product door with a real fitter and a window too small."""
    import mnemiq.runtime as rt_mod
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.runtime import Runtime

    snap = _snapshot()
    packet = _packet(snap)
    probe = _fitter(None)
    full = _tokens(probe, packet)
    window_tokens = full + REPLY + FEEDBACK_ALLOWANCE - full // 3
    seen = {}

    class _Agent:
        def answer(self, packet, snapshot, grants, identity, emit=None):
            seen["packet"] = packet
            return AgentAnswer(answer="A")

    class _Authz:
        def grants_for(self, identity):
            return _grants(snap)

    monkeypatch.setattr(rt_mod, "retrieve", lambda *a, **k: packet)
    settings = Settings(llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="x",
                        acme_data_dir=None, store_path=str(tmp_path / "s.duckdb"))
    from mnemiq.contract import IdentityContext

    rt = Runtime(con=duckdb.connect(), snapshot=snap, adapter=None, agent=_Agent(), embedder=None,
                 authz=_Authz(), settings=settings,
                 fitter=_fitter(window_tokens,
                                count=lambda s, u: (len(s + u) // 4, window_tokens)))
    rt.ask("total body weight per lot", IdentityContext(tenant_id="t", principal_id="p", roles=[]))

    assert seen["packet"] is not packet and seen["packet"].descriptions is not None


# --- the boot advisory --------------------------------------------------------------------------

def _counting_server(window_tokens: int):
    import json

    import httpx

    def handle(request):
        body = json.loads(request.content)["messages"]
        return httpx.Response(200, json={"count": (len(body[0]["content"]) + len(body[1]["content"])) // 4,
                                         "max_model_len": window_tokens})
    return httpx.Client(transport=httpx.MockTransport(handle))


class _All:
    def __init__(self, snap):
        self._snap = snap

    def grants_for(self, identity):
        return window.everything(self._snap)


@pytest.mark.parametrize("room", ["fits", "trims", "too small", "inside the allowance"])
def test_the_advisory_says_whether_the_window_fits_trims_or_is_too_small(caplog, room):
    from mnemiq.runtime import _warn_prompt_window

    snap = _snapshot()
    settings = Settings(llm_base_url="http://127.0.0.1:9/v1", llm_api_key="k", llm_model="m",
                        retrieval_k=2)
    con = duckdb.connect()
    full = window.worst_window(con, snap, settings, "duckdb", _All(snap),
                               http=_counting_server(10**9))
    floor = window.worst_window(con, snap, settings, "duckdb", _All(snap),
                                http=_counting_server(1))  # forces the floor measurement
    window_tokens = {"fits": full.needed,
                     "trims": floor.floor_tokens + REPLY + FEEDBACK_ALLOWANCE,
                     "too small": floor.floor_tokens + REPLY - 1,
                     # the fit keeps FEEDBACK_ALLOWANCE free too: the boot line must agree with it
                     "inside the allowance": floor.floor_tokens + REPLY + FEEDBACK_ALLOWANCE - 1}[room]
    assert floor.floor_tokens + REPLY + FEEDBACK_ALLOWANCE < full.needed, "the case: a gap to trim in"
    report = window.worst_window(con, snap, settings, "duckdb", _All(snap),
                                 http=_counting_server(window_tokens))
    import mnemiq.llm.window as window_module

    with caplog.at_level(logging.INFO):
        original = window_module.worst_window
        window_module.worst_window = lambda *a, **k: report
        try:
            _warn_prompt_window(settings, con, snap, None, _All(snap), frozenset())
        finally:
            window_module.worst_window = original
    text = caplog.text
    if room == "fits":
        assert "prompt window fits:" in text and "WARNING" not in text
    elif room == "trims":
        assert "fits only by withholding column descriptions" in text
        assert "TOO SMALL" not in text
    else:
        assert "prompt window TOO SMALL" in text, room


# --- gate review of the first version -----------------------------------------------------------

_PIECE = __import__("re").compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]")


def _piece_count(system: str, user: str) -> int:
    """Words cost one token, every digit and punctuation mark another: a column name such as
    `step_001_time` is seven tokens in thirteen characters, prose about four characters a token --
    the split that made one average ratio undercount a trimmed prompt (3.96 vs 3.16 measured)."""
    return len(_PIECE.findall(system)) + len(_PIECE.findall(user))


def test_the_fitted_prompt_fits_the_servers_own_count_when_names_and_prose_count_differently():
    snap = _snapshot(120)
    packet = _packet(snap)
    fitter = _fitter(None)
    full = _piece_count(fitter.system(), user_prompt(packet))
    window_tokens = full + REPLY + FEEDBACK_ALLOWANCE - full // 4
    fitter = _fitter(window_tokens, count=lambda s, u: (_piece_count(s, u), window_tokens))

    fitted = fitter.fit(packet, snap, _grants(snap))

    sent, described = fitted.descriptions
    assert 0 < sent < described
    assert (_piece_count(fitter.system(), user_prompt(fitted)) + REPLY + FEEDBACK_ALLOWANCE
            <= window_tokens), "what is sent must fit the server's count, not the estimate's"


def test_a_server_that_gives_no_count_is_not_asked_on_every_question():
    snap = _snapshot()
    asked = []
    fitter = _fitter(None, count=lambda s, u: asked.append(1))
    fitter.fit(_packet(snap), snap, _grants(snap))
    fitter.fit(_packet(snap, "another question"), snap, _grants(snap))
    assert asked == [1]


def test_a_column_a_metric_or_dimension_names_ranks_with_the_named():
    from mnemiq.contract import Dimension, Metric
    from mnemiq.contract.values import MeasureExpr

    snap = _snapshot(8)
    packet = _packet(snap, question="anything at all")
    packet.definitions = []
    packet.metrics = [Metric(id="m1", label="cycle", status="certified", owner="o", grain="lot",
                             measure=MeasureExpr(expr="AVG(step_003_time)", source="wide_trace"),
                             time_dimension="step_000_time")]
    packet.dimensions = [Dimension(id="d1", label="by step", source="wide_trace",
                                   expr="step_004_time")]
    ranked = rank_columns(packet, snap, build_access_policy(snap, _grants(snap)))
    first = set(ranked[:3])
    assert {("wide_trace", "step_003_time"), ("wide_trace", "step_004_time")} <= first


def test_the_advisory_floor_is_the_heaviest_view_without_descriptions(caplog, monkeypatch):
    """Gate review: measured only for the view heaviest WITH descriptions, a view of many columns
    and short descriptions could be the heaviest without them and read as trims, not TOO SMALL."""
    tables = {"long_prose": (6, "x" * 2000), "many_columns": (90, "short")}
    snap = Snapshot(
        version="v1", source_id="s", created_at="2026-10-03T00:00:00Z",
        source_bindings=[SourceBinding(id=f"sb:{t}", source_id="s", object_id=t, source_object=t,
                                       binding_type="table") for t in tables],
        columns=[Column(id=f"{t}.column_number_{i:03d}", object_id=t, name=f"column_number_{i:03d}",
                        data_type="text", description=d)
                 for t, (n, d) in tables.items() for i in range(n)])

    class _TwoRoles:
        def policy_roles(self):
            return ["prose", "wide"]

        def grants_for(self, identity):
            seen = {"prose": "long_prose", "wide": "many_columns"}
            return GrantSet(frozenset(seen[r] for r in identity.roles if r in seen))

    settings = Settings(llm_base_url="http://127.0.0.1:9/v1", llm_api_key="k", llm_model="m",
                        retrieval_k=1)
    con = duckdb.connect()
    by_view = {label: window.largest_prompts(con, snap, settings, "duckdb", grants,
                                             describe=lambda t, c: False)
               for label, grants in window.grant_sets(_TwoRoles(), settings)}
    heaviest_bare = max(len(s) + len(u) for s, u, _r, _c in by_view.values())
    report = window.worst_window(con, snap, settings, "duckdb", _TwoRoles(),
                                 http=_counting_server(1))
    assert report.who == "role prose", "the full prompt's heaviest view is the prose one"
    assert report.floor_tokens >= heaviest_bare // 4, "the floor is the wide view's"
    assert report.floor_who == "role wide", "and the report names the view it measured"

    # ...and so does the boot line, at a window where the floor fits and the full prompt does not.
    from mnemiq.runtime import _warn_prompt_window

    trims = window.worst_window(con, snap, settings, "duckdb", _TwoRoles(),
                                http=_counting_server(report.floor_tokens + REPLY
                                                      + FEEDBACK_ALLOWANCE))
    assert trims.floor_fits and trims.who == "role prose" and trims.floor_who == "role wide"
    monkeypatch.setattr(window, "worst_window", lambda *a, **k: trims)
    with caplog.at_level(logging.INFO):
        _warn_prompt_window(settings, con, snap, None, _TwoRoles(), frozenset())
    assert "(for role wide)" in caplog.text and "(for role prose)" not in caplog.text


# --- Codex review of #78: a characters-per-token ratio is not a bound ---------------------------

def _cjk_snapshot() -> Snapshot:
    """Descriptions in Chinese: three UTF-8 bytes a character, where 2.5 characters a token would
    guess under half a token each."""
    snap = _snapshot(60)
    return snap.model_copy(update={"columns": [
        c.model_copy(update={"description": "每一批次在生产线上记录的数值，按系统原样保存。" * 3})
        for c in snap.columns]})


def _bytes_count(system: str, user: str) -> int:
    """The worst a byte-level tokenizer can do: a token a byte."""
    return len(system.encode()) + len(user.encode())


def _between(fitter, packet) -> int:
    """A window the 2.5-characters-a-token guess says fits and the byte count says does not."""
    user = user_prompt(packet)
    guess = math.ceil((len(fitter.system()) + len(user)) / 2.5)
    bytes_ = _bytes_count(fitter.system(), user)
    assert guess + 1000 < bytes_, "the case: the guess and the bytes are far apart"
    return (guess + bytes_) // 2 + REPLY + FEEDBACK_ALLOWANCE


def test_a_token_dense_prompt_is_counted_not_waved_through_on_a_ratio():
    snap = _cjk_snapshot()
    packet = _packet(snap)
    window_tokens = _between(_fitter(None), packet)
    fitter = _fitter(window_tokens, count=lambda s, u: (_bytes_count(s, u), window_tokens))

    fitted = fitter.fit(packet, snap, _grants(snap))

    assert fitted is not packet, "the ratio said it fits; the server's count says it does not"
    assert (_bytes_count(fitter.system(), user_prompt(fitted)) + REPLY + FEEDBACK_ALLOWANCE
            <= window_tokens)


def test_without_a_count_a_declared_window_is_held_to_the_byte_bound():
    snap = _cjk_snapshot()
    packet = _packet(snap)
    window_tokens = _between(_fitter(None), packet)
    fitter = _fitter(window_tokens, count=lambda s, u: None)

    fitted = fitter.fit(packet, snap, _grants(snap))

    assert fitted is not packet
    assert (_bytes_count(fitter.system(), user_prompt(fitted)) + TEMPLATE_ALLOWANCE + REPLY
            + FEEDBACK_ALLOWANCE <= window_tokens), "trimmed to the byte bound"
