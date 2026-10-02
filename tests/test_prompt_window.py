"""The boot advisory that says when the largest generation prompt cannot fit the window (M119)."""

from __future__ import annotations

import json
import logging
import math

import duckdb
import httpx
import pytest

from mnemiq.config import Settings
from mnemiq.contract import Column, Definition, Snapshot, SourceBinding
from mnemiq.generate.generator import GENERATOR_MAX_TOKENS
from mnemiq.llm import window
from mnemiq.llm.window import WindowReport, check_window, count_on_server, largest_prompt


def _snapshot() -> Snapshot:
    """Three tables of very different widths, a definition bound to each end, one unbound."""
    widths = {"wide": 40, "middle": 12, "narrow": 2}
    return Snapshot(
        version="v1", source_id="s", created_at="2026-10-02T00:00:00Z",
        source_bindings=[
            SourceBinding(id=f"sb:{t}", source_id="s", object_id=t, source_object=t,
                          binding_type="table")
            for t in widths
        ],
        columns=[
            Column(id=f"{t}.col_{i}", object_id=t, name=f"col_{i}", data_type="text",
                   description=f"Column {i} of {t}, described at some length.")
            for t, n in widths.items() for i in range(n)
        ],
        definitions=[
            Definition(id="d:wide", term="wide policy", domain="ops", definition="WIDE-DEFINITION-TEXT",
                       bound_objects=["wide"]),
            Definition(id="d:narrow", term="narrow policy", domain="ops",
                       definition="NARROW-DEFINITION-TEXT",
                       bound_objects=["narrow"]),
        ],
    )


def _settings(**over) -> Settings:
    return Settings(**{"llm_base_url": "http://127.0.0.1:9/v1", "llm_api_key": "k",
                       "llm_model": "m", "retrieval_k": 2, **over})


class _Everything:
    """A provider that lists no roles and grants every table, every PII level cleared."""

    def grants_for(self, identity):
        return window.everything(_snapshot())


class _Roles:
    """A provider that declares roles, each with its own grants."""

    def __init__(self, by_role: dict):
        self._by_role = by_role

    def policy_roles(self):
        return list(self._by_role)

    def grants_for(self, identity):
        from mnemiq.authz.grants import EMPTY

        return next((self._by_role[r] for r in identity.roles if r in self._by_role), EMPTY)


_ALL = _Everything()


def _con_with_examples(examples: list[tuple[str, str]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE example (question TEXT, sql TEXT, tables TEXT, object_id TEXT, "
                "source_id TEXT)")
    for q, s in examples:
        con.execute("INSERT INTO example VALUES (?, ?, ?, ?, ?)", [q, s, json.dumps(["wide"]),
                                                                   "wide", "s"])
    return con


def test_the_largest_prompt_holds_the_k_widest_cards_and_what_rides_with_them():
    con = _con_with_examples([("short?", "SELECT 1"), ("a much longer question?" * 5, "SELECT " + "x, " * 50),
                              ("medium question?", "SELECT a, b")])
    system, user, cards = largest_prompt(con, _snapshot(), _settings(), "duckdb")

    assert cards == 2
    assert "wide" in user and "middle" in user
    assert "narrow" not in user.split("TABLES:")[1], "the narrowest card is past k=2"
    assert "WIDE-DEFINITION-TEXT" in user, "a definition bound to a shown table rides with it"
    assert "NARROW-DEFINITION-TEXT" not in user, "its table was not shown"
    assert "a much longer question?" in user and "medium question?" in user
    assert "short?" not in user, "only the k largest examples"
    assert user.startswith("QUESTION: \n"), "the question is an allowance in tokens, not text"
    assert system


def test_a_narrow_table_with_much_bound_to_it_outweighs_a_wider_bare_one():
    """Ranking by the bare card picked wide and middle and measured a fraction of the prompt real
    retrieval builds when the narrow table is shown: its bound definitions ride in with it (Codex
    review of #73: 5,639 characters measured against 33,875)."""
    snap = _snapshot()
    heavy = [Definition(id=f"d:n{i}", term=f"narrow rule {i}", domain="ops",
                        definition="N" * 500 + f" rule {i}", bound_objects=["narrow"])
             for i in range(50)]
    snap = snap.model_copy(update={"definitions": [*snap.definitions, *heavy]})
    _, user, _ = largest_prompt(duckdb.connect(), snap, _settings(), "duckdb")

    tables = user.split("TABLES:")[1]
    assert "narrow" in tables and "wide" in tables and "middle" not in tables
    assert user.count("N" * 500) == 50


def test_a_definition_shared_with_a_table_outside_the_k_still_counts():
    """Every table is visible when measuring: an identity granted both tables sees a definition
    bound to the shown one and to one past the k, so the largest prompt carries it."""
    snap = _snapshot()
    shared = Definition(id="d:shared", term="shared rule", domain="ops",
                        definition="SHARED-DEFINITION-TEXT", bound_objects=["wide", "narrow"])
    snap = snap.model_copy(update={"definitions": [*snap.definitions, shared]})
    _, user, _ = largest_prompt(duckdb.connect(), snap, _settings(), "duckdb")

    assert "narrow" not in user.split("TABLES:")[1], "narrow is past k=2"
    assert "SHARED-DEFINITION-TEXT" in user


def _abc_shared() -> Snapshot:
    """Codex's case on #73: A and B share fifty long definitions, C has fifty slightly shorter ones
    of its own. Ranked alone, A and B each bring the most, but the real pair A+C is the longest."""
    tables = ("a", "b", "c")
    shared = [Definition(id=f"d:s{i}", term=f"shared {i}", domain="ops",
                         definition="S" * 400 + f" {i}", bound_objects=["a", "b"]) for i in range(50)]
    own = [Definition(id=f"d:c{i}", term=f"own {i}", domain="ops",
                      definition="C" * 390 + f" {i}", bound_objects=["c"]) for i in range(50)]
    return Snapshot(
        version="v1", source_id="s", created_at="2026-10-02T00:00:00Z",
        source_bindings=[SourceBinding(id=f"sb:{t}", source_id="s", object_id=t, source_object=t,
                                       binding_type="table") for t in tables],
        columns=[Column(id=f"{t}.x", object_id=t, name="x", data_type="text") for t in tables],
        definitions=[*shared, *own],
    )


def _real_prompt(snap: Snapshot, ids: list[str]) -> str:
    """The prompt real retrieval builds when exactly these tables are shown, everything granted."""
    from mnemiq.authz.grants import GrantSet
    from mnemiq.generate.prompts import user_prompt
    from mnemiq.semantic.cards import build_cards
    from mnemiq.semantic.glossary import select_definitions
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    cards = [RetrievedCard(object_id=c.object_id, card=c.text, score=0.0)
             for c in build_cards(snap) if c.object_id in ids]
    everything = GrantSet(frozenset(c.object_id for c in build_cards(snap)))
    # The question as largest_prompt renders it: empty, its 500 tokens added beside the count.
    return user_prompt(ContextPacket(
        question="", cards=cards, grant_fingerprint="", enrichment_version=None,
        definitions=select_definitions("", snap.definitions, everything, ids)))


def test_shared_definitions_cannot_hide_a_longer_real_packet():
    snap = _abc_shared()
    _, measured, _ = largest_prompt(duckdb.connect(), snap, _settings(), "duckdb")
    real = {pair: len(_real_prompt(snap, list(pair))) for pair in (("a", "b"), ("a", "c"), ("b", "c"))}
    assert real[("a", "c")] > real[("a", "b")], "the case: the pair the ranking passes over is longer"
    assert len(measured) >= max(real.values())


def test_a_window_too_small_for_the_longest_real_pair_never_reads_fits():
    snap = _abc_shared()
    con = duckdb.connect()
    system, _, _ = largest_prompt(con, snap, _settings(), "duckdb")
    longest = math.ceil((len(system) + len(_real_prompt(snap, ["a", "c"])))
                        / window.CHARS_PER_TOKEN_FLOOR)
    tight = _settings(llm_context_window=longest + GENERATOR_MAX_TOKENS - 1)
    report = check_window(con, snap, tight, "duckdb", http=_server(None, 404))
    assert report.fits is False


def _ab_defined_c_bare(c_description: str) -> Snapshot:
    """Gate review of #73: A and B each a short card plus a short definition, C a longer card and
    nothing else. Ranked with the DEFINITIONS header counted per table, A and B beat C whenever
    C's card is within a header of A's card plus its definition -- and the real A+C is longer."""
    tables = ("a", "b", "c")
    return Snapshot(
        version="v1", source_id="s", created_at="2026-10-02T00:00:00Z",
        source_bindings=[SourceBinding(id=f"sb:{t}", source_id="s", object_id=t, source_object=t,
                                       binding_type="table") for t in tables],
        columns=[Column(id="a.x", object_id="a", name="x", data_type="text"),
                 Column(id="b.x", object_id="b", name="x", data_type="text"),
                 Column(id="c.x", object_id="c", name="x", data_type="text",
                        description=c_description)],
        definitions=[Definition(id=f"d:{t}", term=f"rule {t}", domain="ops",
                                definition="ten chars.", bound_objects=[t]) for t in ("a", "b")],
    )


def test_section_headers_cannot_tip_the_ranking_below_a_real_packet():
    shortfalls = []
    for n in range(0, 120, 8):  # each bug this guards shows across lengths 20-88
        snap = _ab_defined_c_bare("d" * n)
        _, measured, _ = largest_prompt(duckdb.connect(), snap, _settings(), "duckdb")
        longest = max(len(_real_prompt(snap, list(pair)))
                      for pair in (("a", "b"), ("a", "c"), ("b", "c")))
        if len(measured) < longest:
            shortfalls.append((n, longest - len(measured)))
    assert not shortfalls, f"measured below a real pair at C description lengths {shortfalls}"


def _ab_bare_c_defined(b_description: str) -> Snapshot:
    """Gate review of #73: A and B bare cards, C a short card with a definition. When B's card
    outweighs C's content, A and B are chosen and their prompt has no DEFINITIONS section -- yet
    the real A+C carries that header, which a sum of contents leaves out."""
    tables = ("a", "b", "c")
    return Snapshot(
        version="v1", source_id="s", created_at="2026-10-02T00:00:00Z",
        source_bindings=[SourceBinding(id=f"sb:{t}", source_id="s", object_id=t, source_object=t,
                                       binding_type="table") for t in tables],
        columns=[Column(id="a.x", object_id="a", name="x", data_type="text",
                        description="a" * 300),
                 Column(id="b.x", object_id="b", name="x", data_type="text",
                        description=b_description),
                 Column(id="c.x", object_id="c", name="x", data_type="text")],
        definitions=[Definition(id="d:c", term="rule c", domain="ops",
                                definition="ten chars.", bound_objects=["c"])],
    )


def test_a_section_the_chosen_tables_leave_empty_still_counts_its_header():
    shortfalls = []
    for n in range(0, 120, 8):  # each bug this guards shows across lengths 20-88
        snap = _ab_bare_c_defined("b" * n)
        _, measured, _ = largest_prompt(duckdb.connect(), snap, _settings(), "duckdb")
        longest = max(len(_real_prompt(snap, list(pair)))
                      for pair in (("a", "b"), ("a", "c"), ("b", "c")))
        if len(measured) < longest:
            shortfalls.append((n, longest - len(measured)))
    assert not shortfalls, f"measured below a real pair at B description lengths {shortfalls}"


def test_examples_are_ranked_as_rendered_not_as_stored():
    """Codex review of #73: the prompt cuts an example's question to 300 characters, so a
    2,000-character question with `SELECT 1` adds less than a short question with long SQL --
    ranked by stored length it won, and the longer example real retrieval can bring was left out."""
    long_sql = "SELECT " + ", ".join(f"col_{i}" for i in range(150)) + " FROM wide"
    con = _con_with_examples([("q" * 2000, "SELECT 1"), ("short?", long_sql)])
    _, user, _ = largest_prompt(con, _snapshot(), _settings(retrieval_k=1), "duckdb")
    assert long_sql in user


def test_the_question_is_counted_as_its_allowance():
    report = check_window(duckdb.connect(), _snapshot(), _settings(), "duckdb",
                          http=_server({"count": 1_000, "max_model_len": 28_672}))
    assert report.prompt_tokens == 1_000 + window.QUESTION_ALLOWANCE


def _pii_table(described: int, bare: int) -> Snapshot:
    """One table of PII columns: `described` with long descriptions (masking shortens them),
    `bare` with none (masking lengthens them: the mask note is longer than nothing)."""
    cols = [Column(id=f"p.d{i}", object_id="p", name=f"d{i}", data_type="text", pii_level="high",
                   description="A long description of what this column holds. " * 3)
            for i in range(described)]
    cols += [Column(id=f"p.b{i}", object_id="p", name=f"b{i}", data_type="text", pii_level="high")
             for i in range(bare)]
    return Snapshot(version="v1", source_id="s", created_at="2026-10-02T00:00:00Z",
                    source_bindings=[SourceBinding(id="sb:p", source_id="s", object_id="p",
                                                   source_object="p", binding_type="table")],
                    columns=cols)


@pytest.mark.parametrize("style", ["cards", "ddl"])
@pytest.mark.parametrize(("described", "bare"), [(0, 300), (40, 40)])
def test_each_identity_is_measured_as_it_is_shown(style, described, bare):
    """Codex review of #73: a masked column's note can outrun the description it replaces, so a
    masking identity's card can be the longer one -- and the measure is per identity, under its own
    policy, so it covers that card without ever rendering what the identity is not shown."""
    from mnemiq.authz.grants import GrantSet
    from mnemiq.semantic.cards import build_cards
    from mnemiq.sql.policy import build_access_policy

    snap = _pii_table(described, bare)
    for grants in (GrantSet(frozenset({"p"}), pii_clearance=frozenset({"high"})),
                   GrantSet(frozenset({"p"}), pii_mask=frozenset({"high"}))):
        _, user, _ = largest_prompt(duckdb.connect(), snap, _settings(card_style=style), "duckdb",
                                    grants=grants)
        measured = user.split("TABLES:")[1]
        shown = build_cards(snap, policy=build_access_policy(snap, grants), style=style)
        assert len(measured) >= len(shown[0].text)
        if grants.pii_mask and described:
            assert "A long description" not in user, "a masked description was rendered"


def test_what_is_sent_is_only_what_that_role_is_shown():
    """Codex review of #73: the worst case was built from every column's metadata and POSTed to
    the model server -- a denied column's description included. Now each role is measured under its
    own policy, and the server sees only what that role's own questions could send."""
    from mnemiq.authz.grants import GrantSet

    snap = _snapshot().model_copy(update={"columns": [
        *_snapshot().columns,
        Column(id="wide.secret", object_id="wide", name="secret", data_type="text",
               pii_level="direct", description="CONFIDENTIAL-COLUMN-NOTE"),
    ]})
    analyst = GrantSet(frozenset({"wide", "middle"}))  # no clearance: the direct column is denied
    sent: list[str] = []

    def handle(request):
        sent.append(request.content.decode())
        return httpx.Response(200, json={"count": 10, "max_model_len": 28_672})

    con = duckdb.connect()
    con.execute("CREATE TABLE example (question TEXT, sql TEXT, tables TEXT, object_id TEXT, "
                "source_id TEXT)")
    con.execute("INSERT INTO example VALUES ('q?', 'SELECT HIDDEN_EXAMPLE FROM narrow', "
                "'[\"narrow\"]', 'narrow', 's')")
    report = window.worst_window(con, snap, _settings(), "duckdb", _Roles({"analyst": analyst}),
                                 http=httpx.Client(transport=httpx.MockTransport(handle)))
    assert report is not None and report.who == "role analyst" and sent
    assert all("HIDDEN_EXAMPLE" not in body for body in sent), "an example over an ungranted table"
    assert all("CONFIDENTIAL-COLUMN-NOTE" not in body and "secret" not in body for body in sent)
    assert all("narrow" not in body.split("TABLES:")[1] for body in sent), "an ungranted table"


def test_the_heaviest_role_is_the_one_counted_and_named():
    from mnemiq.authz.grants import GrantSet

    roles = _Roles({"narrow_only": GrantSet(frozenset({"narrow"})),
                    "wide_reader": GrantSet(frozenset({"wide", "middle"}))})
    report = window.worst_window(duckdb.connect(), _snapshot(), _settings(), "duckdb", roles,
                                 http=_server(None, 404))
    assert report.who == "role wide_reader"
    assert "for role wide_reader, the longest" in report.sentence()


def test_roles_that_see_nothing_leave_nothing_to_measure(caplog):
    from mnemiq.authz.grants import EMPTY
    from mnemiq.runtime import _warn_prompt_window

    assert window.worst_window(duckdb.connect(), _snapshot(), _settings(), "duckdb",
                               _Roles({"none": EMPTY})) is None
    with caplog.at_level(logging.INFO):
        _warn_prompt_window(_settings(), duckdb.connect(), _snapshot(), None,
                            _Roles({"none": EMPTY}), frozenset())
    assert "neither a role in the access policy nor the configured identity sees a table" in (
        caplog.text)


def test_the_configured_identitys_merged_roles_are_measured_too():
    """An identity holding two roles is shown the union -- more than either role -- so the
    configured identity is measured beside each role, and here it is the heaviest."""
    from mnemiq.authz.grants import GrantSet

    class _Merging(_Roles):
        def grants_for(self, identity):
            granted = [self._by_role[r] for r in identity.roles if r in self._by_role]
            return GrantSet(frozenset().union(*(g.objects for g in granted)))

    roles = _Merging({"a": GrantSet(frozenset({"wide"})), "b": GrantSet(frozenset({"middle"}))})
    report = window.worst_window(duckdb.connect(), _snapshot(), _settings(roles="a,b"), "duckdb",
                                 roles, http=_server(None, 404))
    assert report.who.startswith("identity ") and report.cards == 2


def test_a_provider_that_cannot_list_roles_measures_the_configured_identity():
    report = window.worst_window(duckdb.connect(), _snapshot(), _settings(), "duckdb", _ALL,
                                 http=_server(None, 404))
    assert report is not None and report.who.startswith("identity ")


def test_a_store_without_examples_still_measures():
    _, user, cards = largest_prompt(duckdb.connect(), _snapshot(), _settings(), "duckdb")
    assert cards == 2 and "WORKED EXAMPLES" not in user


def _server(reply: dict | None, status: int = 200, seen: list | None = None) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(status, json=reply or {})
    return httpx.Client(transport=httpx.MockTransport(handle))


def test_vllm_counts_at_its_root_tokenize_with_the_chat_messages():
    seen: list = []
    got = count_on_server("http://h:8002/v1", "m", "SYS", "USER",
                          http=_server({"count": 21813, "max_model_len": 28672}, seen=seen))
    assert got == (21813, 28672)
    path, body = seen[0]
    assert path == "/tokenize", "beside /v1, not under it"
    assert body["messages"] == [{"role": "system", "content": "SYS"},
                                {"role": "user", "content": "USER"}]


def test_the_count_carries_the_chat_calls_key():
    """Codex review of #73: behind an authenticating proxy a keyless /tokenize got 401, and the
    advisory reported no check at all against a server that would have counted."""
    def guarded(request):
        if request.headers.get("Authorization") != "Bearer k":
            return httpx.Response(401)
        return httpx.Response(200, json={"count": 30_000, "max_model_len": 28_672})

    report = check_window(duckdb.connect(), _snapshot(), _settings(), "duckdb",
                          http=httpx.Client(transport=httpx.MockTransport(guarded)))
    assert report.counted_by_server and report.window == 28_672 and report.fits is False


def test_a_server_that_is_down_or_not_json_is_no_count():
    def refused(request):
        raise httpx.ConnectError("refused")

    down = httpx.Client(transport=httpx.MockTransport(refused))
    html = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="<html>catch-all</html>")))
    listed = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[1])))
    for client in (down, html, listed):
        assert count_on_server("http://h/v1", "m", "s", "u", http=client) is None


def test_a_reply_dripped_past_the_deadline_is_no_count(monkeypatch):
    """Codex review of #73: httpx's timeout limits each silence, so a proxy sending a byte at a
    time could hold the boot indefinitely. The whole reply now has a deadline."""
    import time

    monkeypatch.setattr(window, "DEADLINE_S", 0.2)

    def drip():
        for _ in range(1_000):
            time.sleep(0.01)
            yield b" "

    slow = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=drip())))
    started = time.monotonic()
    assert count_on_server("http://h/v1", "m", "s", "u", http=slow) is None
    assert time.monotonic() - started < 1.0, "it waited for the drip instead of the deadline"


def test_a_server_that_never_sends_headers_cannot_hold_the_check(monkeypatch):
    """Before the body there is nothing to check a deadline against -- a DNS lookup has no timeout
    at all -- so the check stops waiting at the deadline instead."""
    import time

    monkeypatch.setattr(window, "DEADLINE_S", 0.2)

    def stall(request):
        time.sleep(3)
        return httpx.Response(200, json={"count": 1, "max_model_len": 2})

    started = time.monotonic()
    report = check_window(duckdb.connect(), _snapshot(), _settings(llm_context_window=100),
                          "duckdb", http=httpx.Client(transport=httpx.MockTransport(stall)))
    assert time.monotonic() - started < 1.5
    assert not report.counted_by_server and report.window == 100, "fell back to the declaration"


def test_a_base_url_httpx_cannot_parse_is_reported_not_read_as_no_count(caplog):
    """The thread hands its exceptions back: an unparseable base URL is a configuration error the
    operator must see, not a server without `/tokenize` -- and with a declared window it would
    otherwise have read as 'fits'."""
    from mnemiq.runtime import _warn_prompt_window

    settings = _settings(llm_base_url="http://[::1", llm_context_window=999_999)
    with caplog.at_level(logging.INFO):
        _warn_prompt_window(settings, duckdb.connect(), _snapshot(), None, _ALL, frozenset())
    assert "prompt window not checked: the measurement failed" in caplog.text
    assert "fits" not in caplog.text


def test_a_reply_past_the_size_cap_is_no_count(monkeypatch):
    monkeypatch.setattr(window, "MAX_REPLY_BYTES", 1_000)
    big = {"count": 5, "max_model_len": 9, "tokens": list(range(1_000))}
    assert count_on_server("http://h/v1", "m", "s", "u", http=_server(big)) is None
    monkeypatch.undo()
    assert count_on_server("http://h/v1", "m", "s", "u", http=_server(big)) == (5, 9)


def test_a_declared_window_is_still_checked_when_the_server_is_down():
    def refused(request):
        raise httpx.ConnectError("refused")

    report = check_window(duckdb.connect(), _snapshot(), _settings(llm_context_window=100),
                          "duckdb", http=httpx.Client(transport=httpx.MockTransport(refused)))
    assert report.window == 100 and report.fits is False


def test_a_prompt_and_reply_exactly_filling_the_window_fit():
    assert WindowReport(24_672, 4_000, 28_672, counted_by_server=True, cards=5).fits is True
    assert WindowReport(24_673, 4_000, 28_672, counted_by_server=True, cards=5).fits is False


@pytest.mark.parametrize(("reply", "status"), [
    (None, 404),                        # no such door: Ollama, a hosted API
    ({"count": 10}, 200),               # a count with no window says nothing to compare
    ({"count": "10", "max_model_len": 5}, 200),
])
def test_a_server_that_cannot_count_and_report_is_not_read_as_one(reply, status):
    assert count_on_server("http://h/v1", "m", "s", "u", http=_server(reply, status)) is None


def test_the_server_count_and_window_decide_when_the_server_gives_them():
    con = duckdb.connect()
    report = check_window(con, _snapshot(), _settings(llm_context_window=999_999), "duckdb",
                          http=_server({"count": 25_000, "max_model_len": 28_672}))
    assert report.counted_by_server and report.window == 28_672, "the server outranks a declaration"
    assert report.reply_tokens == GENERATOR_MAX_TOKENS
    assert report.fits is False  # 25,000 + 4,000 > 28,672


def test_without_a_server_count_the_estimate_errs_long_against_the_declared_window():
    con = duckdb.connect()
    snap, settings = _snapshot(), _settings(llm_context_window=100)
    report = check_window(con, snap, settings, "duckdb", http=_server(None, 404))
    system, user, _ = largest_prompt(con, snap, settings, "duckdb")
    assert not report.counted_by_server and report.window == 100
    assert report.prompt_tokens == (math.ceil((len(system) + len(user)) / window.CHARS_PER_TOKEN_FLOOR)
                                    + window.QUESTION_ALLOWANCE)
    assert report.fits is False


def test_no_count_and_no_declaration_is_not_a_verdict():
    report = check_window(duckdb.connect(), _snapshot(), _settings(), "duckdb",
                          http=_server(None, 404))
    assert report.window is None and report.fits is None


# --- the advisory: how a report reaches the operator --------------------------------------------

def _report(fits: bool, by_server: bool = True) -> WindowReport:
    prompt = 21_000 if fits else 30_000
    return WindowReport(prompt, 4_000, 28_672, counted_by_server=by_server, cards=5,
                        real_tokens=prompt)


@pytest.mark.parametrize(("by_server", "label"), [(True, "TOO SMALL"), (False, "MAY BE TOO SMALL")])
def test_a_prompt_past_the_window_warns_and_says_how_sure(caplog, monkeypatch, by_server, label):
    from mnemiq.runtime import _warn_prompt_window

    monkeypatch.setattr(window, "worst_window", lambda *a, **k: _report(False, by_server))
    with caplog.at_level(logging.INFO):
        _warn_prompt_window(_settings(), None, _snapshot(), None, _ALL, frozenset())
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and f"prompt window {label}:" in warned[0].getMessage()
    assert "5 tables bringing the most into it" in warned[0].getMessage()
    assert "MNEMIQ_RETRIEVAL_K" in warned[0].getMessage()


def _all_share(n_defs: int) -> Snapshot:
    """Codex review of #73: definitions bound to every table. The bound lists each with every
    chosen table; a real packet lists it once."""
    tables = ("a", "b", "c")
    return Snapshot(
        version="v1", source_id="s", created_at="2026-10-02T00:00:00Z",
        source_bindings=[SourceBinding(id=f"sb:{t}", source_id="s", object_id=t, source_object=t,
                                       binding_type="table") for t in tables],
        columns=[Column(id=f"{t}.x", object_id=t, name="x", data_type="text") for t in tables],
        definitions=[Definition(id=f"d:{i}", term=f"rule {i}", domain="ops",
                                definition="R" * 500 + f" {i}", bound_objects=list(tables))
                     for i in range(n_defs)],
    )


def _counting_server(window_tokens: int) -> httpx.Client:
    """Counts a quarter of the user prompt's characters, reports the given window."""
    def handle(request):
        user = json.loads(request.content)["messages"][1]["content"]
        return httpx.Response(200, json={"count": len(user) // 4, "max_model_len": window_tokens})
    return httpx.Client(transport=httpx.MockTransport(handle))


def test_a_bound_inflated_by_shared_definitions_warns_may_not_can(caplog, monkeypatch):
    from mnemiq.runtime import _warn_prompt_window

    snap = _all_share(20)
    con = duckdb.connect()
    _, bound, real, _ = window.largest_prompts(con, snap, _settings(), "duckdb", window.everything(snap))
    assert len(bound) > 1.5 * len(real), "the case: the bound counts each definition twice"
    between = (len(real) + len(bound)) // 8 + window.QUESTION_ALLOWANCE + GENERATOR_MAX_TOKENS
    report = check_window(con, snap, _settings(), "duckdb", http=_counting_server(between))
    assert report.fits is False and not report.certain

    monkeypatch.setattr(window, "worst_window", lambda *a, **k: report)
    with caplog.at_level(logging.WARNING):
        _warn_prompt_window(_settings(), None, snap, None, _ALL, frozenset())
    assert "prompt window MAY BE TOO SMALL" in caplog.text and "tables may fail" in caplog.text
    assert "a real packet of those tables" in caplog.text


def test_a_real_packet_past_the_window_warns_can(caplog, monkeypatch):
    from mnemiq.runtime import _warn_prompt_window

    snap = _all_share(20)
    con = duckdb.connect()
    report = check_window(con, snap, _settings(), "duckdb", http=_counting_server(1_000))
    assert report.certain

    monkeypatch.setattr(window, "worst_window", lambda *a, **k: report)
    with caplog.at_level(logging.WARNING):
        _warn_prompt_window(_settings(), None, snap, None, _ALL, frozenset())
    assert "prompt window TOO SMALL" in caplog.text and "tables can fail" in caplog.text


@pytest.mark.parametrize("second", ["refused", "slow"])
def test_a_second_count_that_fails_keeps_the_first(monkeypatch, second):
    """Gate review of #73: both counts once shared one wait that returned only when both did, so a
    slow second call threw away a first count the server had already given."""
    import time

    monkeypatch.setattr(window, "DEADLINE_S", 0.5)
    calls = []

    def handle(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json={"count": 900, "max_model_len": 1_000})
        if second == "slow":
            time.sleep(3)
        return httpx.Response(404)

    snap = _all_share(5)
    started = time.monotonic()
    report = check_window(duckdb.connect(), snap, _settings(), "duckdb",
                          http=httpx.Client(transport=httpx.MockTransport(handle)))
    assert time.monotonic() - started < 2.0
    assert report.counted_by_server and report.window == 1_000, "the first count was kept"
    assert report.real_tokens is None and not report.certain, "an uncounted real packet is MAY"


def test_an_acknowledged_small_window_drops_to_info(caplog, monkeypatch):
    from mnemiq.runtime import _warn_prompt_window

    monkeypatch.setattr(window, "worst_window", lambda *a, **k: _report(False))
    with caplog.at_level(logging.INFO):
        _warn_prompt_window(_settings(), None, _snapshot(), None, _ALL, frozenset({"window:too-small"}))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "prompt window TOO SMALL" in caplog.text


def test_a_prompt_that_fits_does_not_warn(caplog, monkeypatch):
    from mnemiq.runtime import _warn_prompt_window

    monkeypatch.setattr(window, "worst_window", lambda *a, **k: _report(True))
    with caplog.at_level(logging.INFO):
        _warn_prompt_window(_settings(), None, _snapshot(), None, _ALL, frozenset())
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "prompt window fits" in caplog.text


def test_the_advisory_never_raises_and_says_it_did_not_measure(caplog, monkeypatch):
    from mnemiq.runtime import _warn_prompt_window

    def boom(*a, **k):
        raise ValueError("no cards")

    monkeypatch.setattr(window, "worst_window", boom)
    with caplog.at_level(logging.INFO):
        _warn_prompt_window(_settings(), None, _snapshot(), None, _ALL, frozenset())  # no exception
    assert "prompt window not checked: the measurement failed (no cards)" in caplog.text


def test_one_table_reads_as_one():
    assert "(the table bringing" in WindowReport(1, 1, 9, counted_by_server=True, cards=1).sentence()


def test_no_count_says_the_server_may_simply_be_down():
    report = WindowReport(10, 4_000, None, counted_by_server=False, cards=1)
    assert "did not answer" in report.sentence()


def test_switched_off_it_asks_nothing(monkeypatch):
    from mnemiq.runtime import _warn_prompt_window

    def must_not_run(*a, **k):
        raise AssertionError("checked while switched off")

    monkeypatch.setattr(window, "worst_window", must_not_run)
    _warn_prompt_window(_settings(llm_window_check=False), None, _snapshot(), None, _ALL, frozenset())


def test_build_runtime_runs_it_over_the_store_it_built(caplog, monkeypatch, tmp_path):
    """The join: the boot door, with only the server's reply stubbed. The prompt measured is the
    real one from the store `build_runtime` opened, and the warning reaches the log."""
    from mnemiq.runtime import build_runtime
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import save_snapshot

    source = tmp_path / "src.duckdb"
    duckdb.connect(str(source)).execute("CREATE TABLE wide (col_0 TEXT)").close()
    store = tmp_path / "store.duckdb"
    con = init_store(str(store))
    save_snapshot(con, _snapshot().model_copy(update={"source_id": "only"}))
    con.close()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "duckdb", "target": str(source),
                                     "catalog": "src", "schema": "main"}]))
    authz = tmp_path / "authz.json"
    authz.write_text(json.dumps({"roles": {"analyst": ["wide", "middle", "narrow"]}}))

    measured: list[str] = []

    def server(base_url, model, system, user, http=None, api_key=None):
        measured.append(user)
        return (30_000, 28_672)

    monkeypatch.setattr(window, "count_on_server", server)
    settings = _settings(sources_path=str(manifest), store_path=str(store),
                         authz_path=str(authz))
    with caplog.at_level(logging.WARNING):
        build_runtime(settings)

    assert measured and "WIDE-DEFINITION-TEXT" in measured[0], "measured over the loaded snapshot"
    assert "prompt window TOO SMALL" in caplog.text


def test_a_hot_swapped_snapshot_re_measures(monkeypatch):
    """The cards are the subject, and `reload_if_stale` replaces them: a store rebuilt with wider
    tables must be measured again, not judged by the boot snapshot."""
    from mnemiq.runtime import Runtime

    class _Settings:
        control_dsn = "postgresql://x"
        ack_advisories = ""

    new_snapshot = object()
    seen = []
    rt = Runtime.__new__(Runtime)
    rt.settings, rt.con, rt.authz = _Settings(), object(), None
    rt.snapshot, rt.loaded_versions = object(), {"src": "v1"}
    monkeypatch.setattr("mnemiq.runtime.load_current_snapshot",
                        lambda _s, _c: (new_snapshot, {"src": "v2"}))
    monkeypatch.setattr("mnemiq.runtime._warn_prompt_window",
                        lambda settings, con, snapshot, adapter, authz, ack: seen.append(snapshot))

    rt.reload_if_stale()

    assert seen == [new_snapshot]
