"""Wide tables are described in chunks, and what falls short says so (M112).

Semantic enrichment asked for a whole table in one completion capped at 4,000 tokens. A 34-column
table's reply closed cleanly; an 84-column table's stopped mid-JSON, so the table got no
descriptions at all, and the job recorded `failed` with no cause and nothing read it -- the run
looked healthy. Measured with a local 7B on a design partner's shape: 34 of 1,527 columns described.
"""

from __future__ import annotations

import json
import re

from mnemiq.contract import Column, Snapshot, SourceBinding
from mnemiq.enrichment.enricher import CHUNK_COLUMNS, LLMEnricher
from mnemiq.enrichment.prompts import ColumnFacts
from mnemiq.enrichment.proposals import diagnose_reply
from mnemiq.enrichment.semantic import enrich_semantic, semantic_warnings

_ASKED = re.compile(r"^- (\S+) \(type=", re.MULTILINE)


class _Model:
    """Documents exactly the columns a prompt's facts list, as a well-behaved model would. A reply
    stops mid-JSON, as an over-long one does at the budget, for the calls numbered in `cut` (from
    1), or whenever a prompt asks about any column in `cut_columns` -- a span that overflows however
    small it is split."""

    def __init__(self, cut: frozenset[int] = frozenset(), cut_columns: frozenset[str] = frozenset(),
                 raises: bool = False, skip: int = 0):
        self.prompts: list[str] = []
        self.calls = 0
        self._cut, self._cut_columns, self._raises, self._skip = cut, cut_columns, raises, skip

    def complete(self, system, user, max_tokens=512, **_):
        self.calls += 1
        if self._raises:
            raise RuntimeError("rate limited at https://provider.example/v1 key=sk-secret")
        self.prompts.append(user)
        names = _ASKED.findall(user.split("The table's other columns")[0])
        documented = names[self._skip:]  # a model that skips some of what it was asked
        reply = json.dumps({"columns": [{"name": n, "description": f"about {n}"}
                                        for n in documented]})
        cut = len(self.prompts) in self._cut or self._cut_columns.intersection(names)
        return reply[: len(reply) // 2] if cut else reply


def _facts(n: int) -> list[ColumnFacts]:
    return [ColumnFacts(name=f"c{i}", data_type="text") for i in range(n)]


def test_a_wide_table_is_described_in_chunks_and_every_column_lands():
    model = _Model()
    annotation = LLMEnricher(model).annotate("wide", _facts(84))

    assert len(annotation.columns) == 84 and not annotation.failures
    assert len(model.prompts) == 3, "84 columns in chunks of 30"
    for prompt in model.prompts:
        asked = _ASKED.findall(prompt.split("The table's other columns")[0])
        assert len(asked) <= CHUNK_COLUMNS
        assert "for context only" in prompt, "each chunk names the rest of the table"


def test_a_chunk_whose_calls_raise_keeps_the_earlier_chunks_and_stops_asking():
    """Codex review of #75: a timeout on the second chunk escaped the loop and the whole table
    lost the first chunk's columns. The failure stays with its chunk, after a retry; and a chunk
    that got no reply at all means the endpoint is failing, so the rest is recorded, not asked."""
    class _Flaky(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            reply = super().complete(system, user, max_tokens, **kw)
            if len(self.prompts) in (2, 3):  # both attempts at the second chunk time out
                raise TimeoutError("read timed out at https://provider.example key=sk-secret")
            return reply

    annotation = LLMEnricher(_Flaky()).annotate("wide", _facts(84))
    assert len(annotation.columns) == 30, "the first chunk's columns survive"
    assert annotation.failures == ["columns 31-60 of 84: the call failed: TimeoutError",
                                   "columns 61-84 of 84: not asked -- the call failed: TimeoutError"]


def test_a_garbled_reply_then_a_timeout_on_a_split_half_is_not_an_outage():
    """Codex review of #75: the retry forgot that the first ask was answered, so a half that got
    non-JSON and then a timeout took the endpoint for dead and skipped the rest of the table."""
    class _GarbledThenTimeout(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            reply = super().complete(system, user, max_tokens, **kw)
            names = _ASKED.findall(user.split("The table's other columns")[0])
            if len(names) == 30 and "c0" in names:
                return reply[: len(reply) // 2]  # the first chunk overflows: split
            if len(self.prompts) == 3:
                return "Sorry, I can't."  # the first half's first answer is not JSON
            if len(self.prompts) == 4:
                raise TimeoutError("slow")  # and its retry times out
            return reply

    annotation = LLMEnricher(_GarbledThenTimeout()).annotate("wide", _facts(84))
    assert len(annotation.columns) == 69, "only the first half's 15 are lost"
    assert annotation.failures == ["columns 1-15 of 84: the call failed: TimeoutError"]


def test_a_malformed_reply_in_a_later_chunk_keeps_the_earlier_ones_and_the_table_goes_on():
    """Codex review of #75: valid JSON with a list where a column name belongs made the parser
    raise, which escaped the chunk and discarded the first chunk's descriptions."""
    class _Malformed(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            reply = super().complete(system, user, max_tokens, **kw)
            if len(self.prompts) in (2, 3):  # both answers to the second chunk are malformed
                return '{"columns": [{"name": ["c30"], "description": "x"}]}'
            return reply

    annotation = LLMEnricher(_Malformed()).annotate("wide", _facts(84))
    assert len(annotation.columns) == 54, "the first and third chunks land"
    assert annotation.failures == [
        "columns 31-60 of 84: the reply described none of the 30 column(s) asked about"]


def test_a_list_for_a_column_name_is_ignored_not_an_error():
    from mnemiq.enrichment.proposals import parse_annotation

    annotation = parse_annotation('{"columns": [{"name": ["a"]}, {"name": "a"}]}', "t",
                                  {"a": set()})
    assert [c.name for c in annotation.columns] == ["a"]


def test_a_dead_endpoint_costs_two_calls_per_table_not_two_per_chunk():
    """Gate review: retrying every chunk of every table against an endpoint that is down paid for
    100+ failing calls on a 1,527-column shape."""
    model = _Model(raises=True)
    annotation = LLMEnricher(model).annotate("wide", _facts(300))
    assert model.calls == 2 and not annotation.columns
    assert annotation.failures[0] == "columns 1-30 of 300: the call failed: RuntimeError"
    assert all("not asked" in f for f in annotation.failures[1:])


def test_a_cut_off_reply_still_splits_when_the_retry_times_out():
    """Gate review: the retry's exception overwrote the first attempt's size diagnosis, so the
    chunk was not halved."""
    class _CutThenTimeout(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            reply = super().complete(system, user, max_tokens, **kw)
            if len(self.prompts) == 1:
                return reply[: len(reply) // 2]
            if len(self.prompts) == 2:
                raise TimeoutError("slow")
            return reply

    annotation = LLMEnricher(_CutThenTimeout()).annotate("wide", _facts(30))
    assert len(annotation.columns) == 30 and not annotation.failures


def test_a_prompt_too_long_for_the_window_splits_and_the_table_goes_on():
    """Gate review: a PromptCut is one chunk's prompt over the server's window -- not an endpoint
    that is down. Not retried (asking again cuts again), halved (which shortens the prompt), and the
    rest of the table is still asked."""
    from mnemiq.llm.client import PromptCut

    class _Window(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            reply = super().complete(system, user, max_tokens, **kw)
            names = _ASKED.findall(user.split("The table's other columns")[0])
            if len(names) == 30 and "c30" in names:  # the second chunk's prompt is too long
                raise PromptCut("the server kept a fixed window")
            return reply

    model = _Window()
    annotation = LLMEnricher(model).annotate("wide", _facts(84))
    assert len(annotation.columns) == 84 and not annotation.failures
    second = [p for p in model.prompts if "- c30 (type=" in p.split("The table's other columns")[0]]
    assert len(second) == 2, "the cut chunk once, not retried, then its first half"


def test_a_passing_fault_on_a_split_half_is_retried_not_taken_for_a_dead_endpoint():
    class _HalfGlitch(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            reply = super().complete(system, user, max_tokens, **kw)
            names = _ASKED.findall(user.split("The table's other columns")[0])
            if len(names) == 30:
                return reply[: len(reply) // 2]  # the chunk overflows: split
            if len(self.prompts) == 3:
                raise ConnectionError("reset")  # the first half's first ask fails once
            return reply

    annotation = LLMEnricher(_HalfGlitch()).annotate("wide", _facts(30))
    assert len(annotation.columns) == 30 and not annotation.failures


def test_a_call_that_raises_once_is_retried():
    class _Once(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            reply = super().complete(system, user, max_tokens, **kw)
            if len(self.prompts) == 1:
                raise ConnectionError("reset")
            return reply

    annotation = LLMEnricher(_Once()).annotate("narrow", _facts(20))
    assert len(annotation.columns) == 20 and not annotation.failures


def test_a_narrow_table_is_one_call_with_no_context_line():
    model = _Model()
    annotation = LLMEnricher(model).annotate("narrow", _facts(20))
    assert len(annotation.columns) == 20 and len(model.prompts) == 1
    assert "for context only" not in model.prompts[0]


def test_a_chunk_cut_off_once_is_halved_and_recovered():
    """A column count does not bound a reply -- code-heavy columns carry their meanings -- so a
    chunk that overflows is split in half and asked again."""
    model = _Model(cut=frozenset({2, 3}))  # both attempts at the second chunk come back cut off
    annotation = LLMEnricher(model).annotate("wide", _facts(84))
    assert len(annotation.columns) == 84 and not annotation.failures


_MIDDLE = frozenset(f"c{i}" for i in range(30, 60))


def test_a_chunk_that_never_fits_costs_at_most_sixteen_calls():
    """Halves are tried once each: a 30-column chunk that never fits is 2 calls, then 2 + 4 + 8 for
    its halves down to 3- and 4-column spans."""
    model = _Model(cut_columns=frozenset(f"c{i}" for i in range(30)))
    LLMEnricher(model).annotate("wide", _facts(30))
    assert len(model.prompts) == 16


def test_an_empty_reply_at_the_cap_is_halved_too():
    """A reasoning model that spends its cap thinking returns an empty string, not a cut-off JSON."""
    class _Thinker(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            reply = super().complete(system, user, max_tokens, **kw)
            names = _ASKED.findall(user.split("The table's other columns")[0])
            return "" if len(names) > 15 else reply

    annotation = LLMEnricher(_Thinker()).annotate("wide", _facts(30))
    assert len(annotation.columns) == 30 and not annotation.failures


def test_a_server_that_always_answers_empty_is_halved_once_not_to_the_floor():
    """Gate review: an empty reply is not always a full budget -- a filter, or an answer put in a
    reasoning channel -- and halving to the floor would ask 16 times for nothing. Once: 2 calls
    for the chunk, then each half asked and retried as any failure is."""
    class _Silent(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            super().complete(system, user, max_tokens, **kw)
            return ""

    model = _Silent()
    annotation = LLMEnricher(model).annotate("wide", _facts(30))
    assert len(model.prompts) == 6 and not annotation.columns
    assert all(f.endswith("the reply was empty") for f in annotation.failures)


def test_a_split_span_still_gets_a_retry_for_a_dropped_reply():
    """Size failures split; a one-off garbled reply on a split span is asked again, as the root's
    would be."""
    class _Glitchy(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            reply = super().complete(system, user, max_tokens, **kw)
            names = _ASKED.findall(user.split("The table's other columns")[0])
            if len(names) == 30:
                return reply[: len(reply) // 2]  # the root overflows: split
            if len(self.prompts) == 3:
                return "Sorry, something went wrong."  # the first half's first answer is garbled
            return reply

    model = _Glitchy()
    annotation = LLMEnricher(model).annotate("wide", _facts(30))
    assert len(annotation.columns) == 30 and not annotation.failures


def test_a_span_that_overflows_however_small_costs_its_columns_and_says_why():
    annotation = LLMEnricher(_Model(cut_columns=_MIDDLE)).annotate("wide", _facts(84))

    assert len(annotation.columns) == 54
    assert annotation.failures, "the lost columns are accounted for"
    for failure in annotation.failures:
        span = re.match(r"columns (\d+)-(\d+) of 84: the reply stopped before its JSON closed",
                        failure)
        assert span and 31 <= int(span[1]) <= int(span[2]) <= 60
        assert int(span[2]) - int(span[1]) + 1 <= 5, "halved down to the smallest span"


def _snapshot(width: int) -> Snapshot:
    return Snapshot(
        version="v1", source_id="s", created_at="2026-10-02T00:00:00Z",
        source_bindings=[SourceBinding(id="sb:wide", source_id="s", object_id="wide",
                                       source_object="wide", binding_type="table")],
        columns=[Column(id=f"wide.c{i}", object_id="wide", name=f"c{i}", data_type="text")
                 for i in range(width)])


def _job(snap: Snapshot):
    return next(j for j in snap.jobs if j.id == "semantic:wide")


def test_a_partly_described_table_is_done_and_says_how_much():
    snap = enrich_semantic(_snapshot(84), LLMEnricher(_Model(cut_columns=_MIDDLE)))
    job = _job(snap)
    assert job.status == "done"
    assert job.detail.startswith("54 of 84 columns described; columns 31-")
    assert "the reply stopped before its JSON closed" in job.detail
    described = [c for c in snap.columns if c.description]
    assert len(described) == 54


def test_a_table_with_nothing_described_is_failed_with_its_cause():
    every = frozenset(f"c{i}" for i in range(20))
    snap = enrich_semantic(_snapshot(20), LLMEnricher(_Model(cut_columns=every)))
    job = _job(snap)
    assert job.status == "failed"
    assert "the reply stopped before its JSON closed" in job.detail


def test_a_call_that_raises_records_its_type_and_never_its_text():
    """A provider's exception text can carry a host and a key, and the job outlives the run."""
    snap = enrich_semantic(_snapshot(20), LLMEnricher(_Model(raises=True)))
    job = _job(snap)
    assert job.status == "failed" and job.detail == "the call failed: RuntimeError"
    assert "sk-secret" not in job.detail and "provider.example" not in job.detail


def test_columns_the_model_skips_are_counted_as_not_described():
    """Gate review: a chunk answered for 3 of its 30 columns was a success, and the table passed as
    healthy. What is counted is what landed on the snapshot."""
    snap = enrich_semantic(_snapshot(30), LLMEnricher(_Model(skip=27)))
    job = _job(snap)
    assert job.status == "done" and job.detail == "3 of 30 columns described"
    (line,) = semantic_warnings(snap)
    assert "PARTLY described: wide (3 of 30 columns described)" in line


def test_a_certified_column_is_not_counted_as_described_or_missing():
    snap = enrich_semantic(_snapshot(30), LLMEnricher(_Model()), protected=frozenset({"wide.c0"}))
    assert _job(snap).detail is None, "29 of 29 unprotected columns landed"


def test_a_fully_described_table_carries_no_detail():
    job = _job(enrich_semantic(_snapshot(84), LLMEnricher(_Model())))
    assert job.status == "done" and job.detail is None


def test_each_way_a_reply_falls_short_is_named():
    allowed = {"a": set(), "b": set()}
    assert diagnose_reply("", allowed) == "the reply was empty"
    assert diagnose_reply("I cannot help with that.", allowed).startswith("the reply held no JSON")
    assert diagnose_reply('{"columns": [{"name": "a"', allowed).startswith(
        "the reply stopped before its JSON closed")
    assert diagnose_reply('{"columns": [}', allowed).startswith("the reply's JSON did not parse")
    assert diagnose_reply('{"answer": 1}', allowed) == "the reply's JSON carried no columns list"
    assert diagnose_reply('{"columns": [{"name": "zz"}]}', allowed) == (
        "the reply described none of the 2 column(s) asked about")


def test_enrich_says_out_loud_what_fell_short():
    ok = enrich_semantic(_snapshot(84), LLMEnricher(_Model()))
    assert semantic_warnings(ok) == []

    partial = enrich_semantic(_snapshot(84), LLMEnricher(_Model(cut_columns=_MIDDLE)))
    (line,) = semantic_warnings(partial)
    assert line.startswith("WARNING: 1 table(s) were only PARTLY described: wide (54 of 84")

    failed = enrich_semantic(_snapshot(20), LLMEnricher(_Model(raises=True)))
    (line,) = semantic_warnings(failed)
    assert line == ("WARNING: 1 table(s) got NO column descriptions -- semantic enrichment "
                    "failed: wide (the call failed: RuntimeError)")


def test_the_enrich_command_prints_them():
    """Structural, because a full `_cmd_enrich` run needs an index, an embedder and a store: what
    it pins is that the command prints `semantic_warnings` to stderr, which is the half M112 said
    was missing (the job existed and nobody was told)."""
    import ast
    import inspect

    from mnemiq import cli

    tree = ast.parse(inspect.getsource(cli._cmd_enrich))
    printed = [node for node in ast.walk(tree)
               if isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
               and getattr(node.iter.func, "id", "") == "semantic_warnings"]
    assert printed, "_cmd_enrich no longer prints semantic_warnings"
    body = ast.unparse(printed[0])
    assert "sys.stderr" in body


def test_any_parse_failure_stays_with_its_chunk(monkeypatch):
    """`parse_annotation` never raises by design; if a reply shape still gets past it, the chunk
    fails and is retried, and the chunks already described survive."""
    import mnemiq.enrichment.enricher as enricher
    from mnemiq.enrichment.proposals import parse_annotation

    calls = []

    def fragile(raw, table, allowed):
        calls.append(1)
        if len(calls) in (2, 3):
            raise TypeError("unhashable type: 'list'")
        return parse_annotation(raw, table, allowed)

    monkeypatch.setattr(enricher, "parse_annotation", fragile)
    annotation = LLMEnricher(_Model()).annotate("wide", _facts(84))
    assert len(annotation.columns) == 54
    assert annotation.failures == ["columns 31-60 of 84: the reply could not be read: TypeError"]



def _cut_then(second):
    """A model whose first answer to the 30-column chunk is cut off and whose second is `second`
    ('raise' in the parser, or a reply with no JSON); halves are answered normally."""
    class _M(_Model):
        def complete(self, system, user, max_tokens=512, **kw):
            reply = super().complete(system, user, max_tokens, **kw)
            names = _ASKED.findall(user.split("The table's other columns")[0])
            if len(names) == 30 and len(self.prompts) == 1:
                return reply[: len(reply) // 2]
            if len(names) == 30 and len(self.prompts) == 2:
                return '{"columns": [{"name": ["c0"]}]}' if second == "raise" else "Sorry."
            return reply
    return _M()


def test_a_cut_off_answer_still_splits_when_the_retry_cannot_be_parsed(monkeypatch):
    """Gate review: a size diagnosis from the first answer outranks whatever the retry does --
    here a reply the parser chokes on -- so the chunk is halved and recovered."""
    import mnemiq.enrichment.enricher as enricher
    from mnemiq.enrichment.proposals import parse_annotation

    def fragile(raw, table, allowed):
        if '["c0"]' in raw:
            raise TypeError("unhashable type: 'list'")
        return parse_annotation(raw, table, allowed)

    monkeypatch.setattr(enricher, "parse_annotation", fragile)
    annotation = LLMEnricher(_cut_then("raise")).annotate("wide", _facts(30))
    assert len(annotation.columns) == 30 and not annotation.failures


def test_a_cut_off_answer_still_splits_when_the_retry_holds_no_json():
    annotation = LLMEnricher(_cut_then("no json")).annotate("wide", _facts(30))
    assert len(annotation.columns) == 30 and not annotation.failures
