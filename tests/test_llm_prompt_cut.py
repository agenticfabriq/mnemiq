"""M119: a server that silently cuts the prompt is caught, not answered from.

Ollama keeps part of an over-long prompt and returns no error; the only trace is in
`usage.prompt_tokens`. Two signs raise suspicion, because servers cut two ways: a big cut (a recent
Ollama keeps about half its window) pushes the ratio of characters sent to tokens read past
anything mnemiq's own prompts measure (1.4 to 4.5); a modest cut to a full window (Ollama 0.5.4)
leaves the ratio normal but parks the count on the window. Neither is proof -- result rows that
repeat long words compress past 6 with nothing cut -- so a suspicious count is re-sent with
padding, and a count that does not grow is a cut. Only when that check cannot run (the probe
failed, or carried no count) does a ratio decide alone, and only past 12, far above anything
measured (4.50 over 612 real prompts; 6.19 for deliberately repetitive rows).
"""

from types import SimpleNamespace

import pyarrow as pa
import pytest

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer, LLMSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import Column, DeferralReason, IdentityContext, Snapshot
from mnemiq.execute.render import render_result
from mnemiq.generate.generator import LLMGenerator
from mnemiq.llm.client import LLMClient, ModelUnavailable, PromptCut, _on_a_window_edge
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard


def _server_reading(prompt_tokens):
    """A fake OpenAI client whose server reports reading `prompt_tokens` of every prompt."""

    class _Fake:
        def __init__(self, *_, **__):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **_kwargs):
            usage = None if prompt_tokens is None else SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=5)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"sql": "SELECT 1"}'))],
                usage=usage)
    return _Fake


def _client(monkeypatch, prompt_tokens, **overrides) -> LLMClient:
    import mnemiq.llm.client as module

    monkeypatch.setattr(module, "OpenAI", _server_reading(prompt_tokens))
    return LLMClient(Settings(llm_base_url="http://x", llm_api_key="k", llm_model="m",
                              pg_dsn=None, acme_data_dir=None, **overrides))


def test_a_prompt_the_server_cut_is_refused(monkeypatch):
    # 60,000 characters read as 1,000 tokens, and still 1,000 when padded: cut to a fragment.
    client = _client(monkeypatch, prompt_tokens=1_000)
    with pytest.raises(PromptCut) as cut:
        client.complete("s" * 30_000, "u" * 30_000)
    message = str(cut.value)
    assert "1,000" in message and "60,000" in message, "both numbers, so the operator can see the gap"
    assert "context" in message, "and what to raise"


def test_the_worst_real_prompt_measured_still_passes(monkeypatch):
    # 4.50 characters a token: the highest of 612 real prompts, every call site, either tokenizer.
    client = _client(monkeypatch, prompt_tokens=1_000)
    assert client.complete("s" * 2_250, "u" * 2_250) == '{"sql": "SELECT 1"}'


def test_just_under_the_threshold_passes_and_just_over_is_refused(monkeypatch):
    client = _client(monkeypatch, prompt_tokens=1_000)
    client.complete("s" * 2_950, "u" * 3_000)  # 5.95 characters a token
    with pytest.raises(PromptCut):
        client.complete("s" * 3_050, "u" * 3_000)  # 6.05


def test_a_cut_call_is_still_counted(monkeypatch):
    # The server did the work and billed it; the cost report must not lose it.
    client = _client(monkeypatch, prompt_tokens=1_000)
    with pytest.raises(PromptCut):
        client.complete("s" * 30_000, "u" * 30_000)
    assert client.calls == 2, "the call and its probe"
    assert client.prompt_tokens == 2_000 and client.completion_tokens == 10


def test_no_usage_or_zero_tokens_cannot_be_checked_and_is_not_refused(monkeypatch):
    for reported in (None, 0):
        client = _client(monkeypatch, prompt_tokens=reported)
        assert client.complete("s" * 30_000, "u" * 30_000)


def test_the_check_can_be_turned_off_for_a_proxy_that_under_reports(monkeypatch):
    client = _client(monkeypatch, prompt_tokens=1_000, llm_prompt_cut_check=False)
    assert client.complete("s" * 30_000, "u" * 30_000)


def test_a_cut_is_a_kind_of_model_failure():
    # So every existing handler catches it without learning a new type: the agent turns it into a
    # stated failure; the judge records itself unavailable, and the verifier fails closed on that
    # by default (MNEMIQ_VERIFY_FAIL_CLOSED).
    assert issubclass(PromptCut, ModelUnavailable)


def _server_with_a_window(window: int, probe_usage: bool = True, chars_per_token: float = 3):
    """A fake server that reads `chars_per_token` characters a token and keeps at most `window`."""

    class _Fake:
        def __init__(self, *_, **__):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
            self.requests = 0

        def _create(self, messages, **_kwargs):
            self.requests += 1
            sent = sum(len(m["content"]) for m in messages)
            usage = SimpleNamespace(prompt_tokens=min(int(sent / chars_per_token), window),
                                    completion_tokens=5)
            if self.requests > 1 and not probe_usage:
                usage = None
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=f"reply {self.requests}"))],
                usage=usage)
    return _Fake


def test_compressible_text_read_whole_is_answered(monkeypatch):
    # 6.2 characters a token -- over the ratio's threshold -- but the padding is read too: no cut.
    client = _windowed(monkeypatch, window=1_000_000, chars_per_token=6.2)
    assert client.complete("s" * 30_000, "u" * 30_000) == "reply 1"
    assert client.calls == 2


def test_a_real_synthesis_prompt_over_compressible_rows_is_answered(monkeypatch):
    """The review's counterexample in shape: 50 rows of one long company name, rendered and wrapped
    by the real renderer and synthesizer, read whole at 6.19 characters a token under o200k. The
    fake server sets the ratio (6.2), not the prompt; what this pins is that the synthesizer's call
    is probed and answered rather than refused. Refusing would fail an answer no window could fix."""
    rows = pa.table({"company": ["International Business Machines Corporation"] * 50,
                     "n": list(range(50))})
    synth = LLMSynthesizer(_windowed(monkeypatch, window=1_000_000, chars_per_token=6.2))
    reply = synth.answer("Which companies placed orders?", "SELECT company, n FROM orders",
                         render_result(rows), row_count=50)
    assert reply == "reply 1" and synth._client.calls == 2, "probed, then answered"


def _windowed(monkeypatch, window, **kw) -> LLMClient:
    import mnemiq.llm.client as module

    monkeypatch.setattr(module, "OpenAI", _server_with_a_window(window, **kw))
    return LLMClient(Settings(llm_base_url="http://x", llm_api_key="k", llm_model="m",
                              pg_dsn=None, acme_data_dir=None))


def test_a_modest_cut_to_a_full_window_is_caught_by_the_probe(monkeypatch):
    # The measured Ollama 0.5.4 shape: about 3,300 tokens sent, exactly 2,048 read -- 4.4 characters a
    # token, inside the normal range, so only the stalled count gives it away.
    client = _windowed(monkeypatch, window=2_048)
    with pytest.raises(PromptCut) as cut:
        client.complete("s" * 4_475, "u" * 4_475)
    assert "fixed window" in str(cut.value)
    assert client.calls == 2, "the probe is a real request, and it is counted"


def test_a_prompt_that_lands_on_a_window_edge_by_chance_is_answered(monkeypatch):
    client = _windowed(monkeypatch, window=100_000)
    assert client.complete("s" * 3_072, "u" * 3_072) == "reply 1", "the original reply, not the probe's"
    assert client.calls == 2


def test_a_count_off_the_edge_is_not_probed(monkeypatch):
    client = _windowed(monkeypatch, window=100_000)
    client.complete("s" * 3_500, "u" * 3_500)  # 2,333 tokens
    assert client.calls == 1


def test_a_probe_that_fails_keeps_the_reply_in_hand_and_says_so(monkeypatch, caplog):
    import httpx
    from openai import APIConnectionError

    import mnemiq.llm.client as module

    class _ProbeFails(_server_with_a_window(2_048)):
        def _create(self, messages, **kwargs):
            if self.requests:
                raise APIConnectionError(request=httpx.Request("POST", "http://x"))
            return super()._create(messages, **kwargs)

    monkeypatch.setattr(module, "OpenAI", _ProbeFails)
    client = LLMClient(Settings(llm_base_url="http://x", llm_api_key="k", llm_model="m",
                                pg_dsn=None, acme_data_dir=None))
    with caplog.at_level("WARNING", logger="mnemiq.llm.client"):
        assert client.complete("s" * 4_475, "u" * 4_475) == "reply 1"
    assert "could not check for a cut prompt" in caplog.text, "a probe that never ran must be visible"


def test_a_probe_that_reports_no_usage_is_not_called_a_cut_but_says_so(monkeypatch, caplog):
    client = _windowed(monkeypatch, window=2_048, probe_usage=False)
    with caplog.at_level("WARNING", logger="mnemiq.llm.client"):
        assert client.complete("s" * 4_475, "u" * 4_475) == "reply 1"
    assert "no token count" in caplog.text


@pytest.mark.parametrize("chars, refused", [(11_900, False), (12_100, True)])
def test_the_ceiling_for_an_undecided_probe_sits_at_12(monkeypatch, chars, refused):
    client = _windowed(monkeypatch, window=1_000, probe_usage=False)
    if refused:
        with pytest.raises(PromptCut):
            client.complete("s" * (chars // 2), "u" * (chars // 2))
    else:
        assert client.complete("s" * (chars // 2), "u" * (chars // 2)) == "reply 1"


def test_a_failed_probe_still_refuses_a_ratio_no_whole_prompt_reaches(monkeypatch):
    import httpx
    from openai import APIConnectionError

    import mnemiq.llm.client as module

    class _ProbeFails(_server_with_a_window(1_000)):
        def _create(self, messages, **kwargs):
            if self.requests:
                raise APIConnectionError(request=httpx.Request("POST", "http://x"))
            return super()._create(messages, **kwargs)

    monkeypatch.setattr(module, "OpenAI", _ProbeFails)
    client = LLMClient(Settings(llm_base_url="http://x", llm_api_key="k", llm_model="m",
                                pg_dsn=None, acme_data_dir=None))
    with pytest.raises(PromptCut) as cut:
        client.complete("s" * 30_000, "u" * 30_000)
    assert "probe failed" in str(cut.value)


def test_an_undecided_probe_still_refuses_a_ratio_no_whole_prompt_reaches(monkeypatch):
    # 60 characters a token, and the probe carries no count: past doubt, so refused anyway.
    client = _windowed(monkeypatch, window=1_000, probe_usage=False)
    with pytest.raises(PromptCut) as cut:
        client.complete("s" * 30_000, "u" * 30_000)
    assert "could not run" in str(cut.value)


def test_window_edges_are_multiples_of_1024_give_or_take_eight():
    assert _on_a_window_edge(2_048) and _on_a_window_edge(16_386) and _on_a_window_edge(2_056)
    assert _on_a_window_edge(4_090)  # just under 4,096
    assert not _on_a_window_edge(2_057) and not _on_a_window_edge(3_000)
    assert not _on_a_window_edge(1_020), "below the smallest window anyone sets"


class _Adapter:
    dialect = "duckdb"

    def execute_arrow(self, sql, timeout_s=30):
        return pa.table({"n": [1]})


def test_a_cut_reaches_the_user_as_a_configuration_failure_not_an_outage(monkeypatch):
    """The join: the real generator over the real client, the server faked one layer below."""
    big_card = "TABLE claim (" + ", ".join(f"col_{i} INTEGER" for i in range(3_000)) + ")"
    agent = Agent(generator=LLMGenerator(_client(monkeypatch, prompt_tokens=2_048)),
                  synthesizer=FakeSynthesizer(), adapter=_Adapter(), cache=TwoTierCache(L1Cache()),
                  budget=Budget(wall_clock_s=5.0, max_attempts=2))
    packet = ContextPacket(question="how many claims?",
                           cards=[RetrievedCard(object_id="claim", card=big_card, score=1.0)],
                           grant_fingerprint="f", enrichment_version="v1")
    snapshot = Snapshot(version="v1", source_id="acme", created_at="2026-07-13T00:00:00Z",
                        columns=[Column(id="claim.n", object_id="claim", name="n")])
    identity = IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])

    answer = agent.answer(packet, snapshot, GrantSet(frozenset({"claim"})), identity)

    assert answer.failed is True and answer.deferred is False
    assert answer.reason_code == DeferralReason.MODEL_UNAVAILABLE
    assert "cut" in answer.answer and "context" in answer.answer
    assert "outage" not in answer.answer and "try again" not in answer.answer, \
        "retrying cannot fix a window that is too small"
