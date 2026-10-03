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
    """Documents exactly the columns a prompt's facts list, as a well-behaved model would; the
    calls numbered in `cut` (from 1) stop mid-JSON, as an over-long reply does at the budget."""

    def __init__(self, cut: frozenset[int] = frozenset(), raises: bool = False):
        self.prompts: list[str] = []
        self._cut, self._raises = cut, raises

    def complete(self, system, user, max_tokens=512, **_):
        if self._raises:
            raise RuntimeError("rate limited at https://provider.example/v1 key=sk-secret")
        self.prompts.append(user)
        names = _ASKED.findall(user.split("The table's other columns")[0])
        reply = json.dumps({"columns": [{"name": n, "description": f"about {n}"} for n in names]})
        return reply[: len(reply) // 2] if len(self.prompts) in self._cut else reply


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


def test_a_chunk_that_fails_costs_its_columns_and_says_why():
    model = _Model(cut=frozenset({2, 3}))  # both attempts at the second chunk come back cut off
    annotation = LLMEnricher(model).annotate("wide", _facts(84))

    assert len(annotation.columns) == 54
    (failure,) = annotation.failures
    assert failure.startswith("columns 31-60 of 84: the reply stopped before its JSON closed (")
    assert failure.endswith(" characters; the reply budget is the likely cut)")


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
    snap = enrich_semantic(_snapshot(84), LLMEnricher(_Model(cut=frozenset({2, 3}))))
    job = _job(snap)
    assert job.status == "done"
    assert job.detail.startswith("54 of 84 columns described; columns 31-60 of 84: the reply "
                                 "stopped before its JSON closed")
    described = [c for c in snap.columns if c.description]
    assert len(described) == 54


def test_a_table_with_nothing_described_is_failed_with_its_cause():
    snap = enrich_semantic(_snapshot(20), LLMEnricher(_Model(cut=frozenset({1, 2}))))
    job = _job(snap)
    assert job.status == "failed"
    assert job.detail.startswith("the reply stopped before its JSON closed")


def test_a_call_that_raises_records_its_type_and_never_its_text():
    """A provider's exception text can carry a host and a key, and the job outlives the run."""
    snap = enrich_semantic(_snapshot(20), LLMEnricher(_Model(raises=True)))
    job = _job(snap)
    assert job.status == "failed" and job.detail == "the call failed: RuntimeError"
    assert "sk-secret" not in job.detail and "provider.example" not in job.detail


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
    assert diagnose_reply('{"columns": [{"name": "zz"}]}', allowed) == (
        "the reply named none of the 2 column(s) asked about")


def test_enrich_says_out_loud_what_fell_short():
    ok = enrich_semantic(_snapshot(84), LLMEnricher(_Model()))
    assert semantic_warnings(ok) == []

    partial = enrich_semantic(_snapshot(84), LLMEnricher(_Model(cut=frozenset({2, 3}))))
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
