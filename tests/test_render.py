import pyarrow as pa
import pytest

from mnemiq.execute.render import render_result


def _table(rows: int, value: str = "x") -> pa.Table:
    return pa.table({"n": list(range(rows)), "label": [value] * rows})


def test_a_small_result_renders_every_row():
    text = render_result(_table(3))
    assert "n" in text and "label" in text
    assert text.count("\n") >= 3
    assert "truncated" not in text.lower()


def test_a_large_result_says_it_is_truncated():
    text = render_result(_table(500), max_rows=10)
    assert "500" in text  # the true row count is stated
    assert "10" in text  # and how many are shown
    assert "truncat" in text.lower()  # never let the model summarize a partial view silently


def test_a_wide_cell_is_clipped():
    text = render_result(_table(1, value="y" * 500), max_cell=20)
    assert "y" * 21 not in text


def test_an_empty_result_is_stated_plainly():
    text = render_result(pa.table({"n": pa.array([], type=pa.int64())}))
    assert "0 rows" in text  # "no rows" is a real answer, and must not read as an error


def test_nulls_render_as_null_not_none():
    table = pa.table({"n": pa.array([None, 1], type=pa.int64())})
    assert "NULL" in render_result(table)


def test_a_shortened_cell_is_declared_not_just_marked():
    # The model read a bare `…` as evidence and reported rows truncated when only cells
    # were. Row truncation was always announced; cell truncation was not.
    t = pa.table({"detail": ["x" * 400]})
    out = render_result(t)

    assert "…" in out
    assert "shortened for this prompt" in out
    assert "(1 rows)" in out, "the row count must still be honest"


def test_nothing_is_declared_when_nothing_was_shortened():
    out = render_result(pa.table({"n": [1, 2]}))
    assert "shortened" not in out
    assert "…" not in out


def test_ordinary_values_are_no_longer_cut():
    # 120 halved schema/description columns; a policy number or a sentence must survive.
    value = "a claim description of the kind an adjuster writes, " * 3
    assert len(value) < 240
    assert value in render_result(pa.table({"note": [value]}))


def test_both_bounds_are_declared_together():
    rows = ["y" * 400] * 60
    out = render_result(pa.table({"detail": rows}), max_rows=50)
    assert "showing 50 of 60 rows (truncated)" in out
    assert "shortened for this prompt" in out


def test_a_wide_result_shows_the_first_columns_and_counts_the_rest():
    """M130: a 965-column SELECT * rendered every column into the judge's and the answer-writer's
    prompts. Columns are bounded like rows and cells, and the bound is declared."""
    import pyarrow as pa

    from mnemiq.execute.render import MAX_COLS, render_result

    table = pa.table({f"c{i:03d}": [i, i + 1] for i in range(965)})
    text = render_result(table, max_rows=5)
    header = text.splitlines()[0].split(" | ")
    assert header == [f"c{i:03d}" for i in range(MAX_COLS)]
    assert f"showing the first {MAX_COLS} of 965 columns; the other {965 - MAX_COLS} are not shown" in text
    assert "c964" not in text, "the hidden names are counted, not listed"
    empty = render_result(table.slice(0, 0))
    assert "of 965 columns" in empty and "c964" not in empty


def test_a_result_within_the_column_bound_renders_as_before():
    import pyarrow as pa

    from mnemiq.execute.render import MAX_COLS, render_result

    table = pa.table({f"c{i}": ["x", "y"] for i in range(MAX_COLS)})
    assert render_result(table) == render_result(table, max_cols=None)
    assert "columns;" not in render_result(table)


@pytest.mark.parametrize("protocol", ["read", "score"])
def test_the_judge_reads_a_bounded_preview_of_a_wide_result(protocol):
    """The join, not the ends: the verifier hands its judge a preview of a 965-column result, and
    that preview is bounded -- the path M130 was found on. Both doors: `read` is what the shipped
    judge speaks, `score` what a stub speaks, and each renders the preview on its own line."""
    from types import SimpleNamespace

    from mnemiq.execute.render import MAX_COLS
    from mnemiq.semantic.retrieval import ContextPacket
    from mnemiq.sql.verdict import Approved
    from mnemiq.verify.verifier import Verifier

    seen = {}

    class _Scorer:
        def score(self, question, schema, sql, preview):
            seen["preview"] = preview
            return 0.9

    class _Reader:  # no usable `score`, so this case proves the `read` door was taken
        def score(self, question, schema, sql, preview):
            raise AssertionError("a judge offering `read` must be read, not scored")

        def read(self, question, schema, sql, preview):
            seen["preview"] = preview
            return SimpleNamespace(score=0.9, fell_open=False, reason="ok")

    _Judge = _Reader if protocol == "read" else _Scorer
    table = pa.table({f"c{i:03d}": [i] for i in range(965)})
    packet = ContextPacket(question="everything for lot 7", cards=[], grant_fingerprint="f",
                           enrichment_version=None)
    Verifier(sanity=False, judge=_Judge()).verify(
        packet, Approved(plan_sql="SELECT * FROM t", target_sql="SELECT * FROM t"), table)
    assert f"showing the first {MAX_COLS} of 965 columns" in seen["preview"]
    assert len(seen["preview"]) < 5_000


def _wide(marker: int = 0) -> pa.Table:
    return pa.table({f"c{i:03d}": [marker] for i in range(965)})


class _WideAdapter:
    """Every query returns 965 columns; a SUM in the SQL changes the values, so two candidates
    disagree and the selector is consulted."""

    def execute(self, sql):  # EXPLAIN, inside decide()
        return []

    def execute_arrow(self, sql, timeout_s=None):
        return _wide(5 if "SUM" in sql.upper() else 7)


def test_the_answer_writer_reads_a_bounded_preview_and_the_user_sees_every_column():
    """The answer writer shares the judge's bound, which is safe only because the user's own view
    of the result -- the answer's display preview -- keeps every column. Both halves together."""
    from mnemiq.execute.render import MAX_COLS
    from tests.test_agent import _agent, _answer

    class _Synth:
        rendered = None

        def answer(self, question, sql, rendered, forced=False, row_count=None):
            _Synth.rendered = rendered
            return "Lot 7."

    answer = _answer(_agent(['{"sql": "SELECT n FROM claim"}'], adapter=_WideAdapter(),
                            synth=_Synth()))
    assert f"showing the first {MAX_COLS} of 965 columns" in _Synth.rendered
    assert "c964" not in _Synth.rendered
    assert len(answer.preview.columns) == 965 and answer.preview.columns[-1] == "c964"


def test_the_selector_reads_bounded_previews_of_wide_candidates():
    from mnemiq.agent.loop import Agent
    from mnemiq.agent.synthesize import FakeSynthesizer
    from mnemiq.cache.store import L1Cache, TwoTierCache
    from mnemiq.execute.render import MAX_COLS
    from mnemiq.execute.select import FakeSelector
    from mnemiq.generate.generator import FakeGenerator
    from tests.test_agent import _answer, _sql

    selector = FakeSelector([0])
    agent = Agent(generator=FakeGenerator([_sql("count(*)"), _sql("sum(n)")]),
                  synthesizer=FakeSynthesizer("Lot 7."), adapter=_WideAdapter(),
                  cache=TwoTierCache(L1Cache()), candidates=2, selector=selector)
    _answer(agent)
    assert len(selector.calls) == 1, "two disagreeing clusters must reach the selector"
    previews = [view.preview for view in selector.calls[0][1]]
    assert len(previews) == 2
    for preview in previews:
        assert f"showing the first {MAX_COLS} of 965 columns" in preview and "c964" not in preview


def test_a_zero_column_bound_shows_no_columns_and_says_so():
    text = render_result(pa.table({"a": [1], "b": [2]}), max_cols=0)
    assert "showing the first 0 of 2 columns; the other 2 are not shown" in text
