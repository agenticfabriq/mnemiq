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
        self._cut, self._cut_columns, self._raises, self._skip = cut, cut_columns, raises, skip

    def complete(self, system, user, max_tokens=512, **_):
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
