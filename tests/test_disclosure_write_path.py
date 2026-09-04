"""A governed write that mutated fewer rows than asked says so, on every surface.

The failure this closes was predicted in the spec and then shipped anyway: "an implementer who
builds items 1-4 and stops ships a governed DELETE narrowed from 47 rows to 3 that reports
nothing". `ApprovedWrite` carried the narrowing from the decider; `WriteResult` dropped it at the
runtime boundary, so CLI and MCP saw an ordinary success with a smaller row count and no reason.

`rows_affected` is what makes it dangerous rather than merely incomplete. A caller who asked to
delete 47 rows and is told 3, with nothing else said, has been given a true number and a false
impression.
"""

import sqlglot

from mnemiq.contract.seams import Narrowed, disclosure_sentence
from mnemiq.runtime import WriteResult
from mnemiq.sql.decide_write import _target_node
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import Narrowing, apply_row_filters_to_write

_VISIBLE = {"claim": {"id", "amount", "ssn"}}


def test_the_decider_still_produces_the_narrowing_for_a_governed_delete():
    ast = sqlglot.parse_one("DELETE FROM claim WHERE id > 0", read="duckdb")
    out, narrowed = apply_row_filters_to_write(
        ast, AccessPolicy(row_filters={"claim": "amount > 0"}), _VISIBLE, _target_node(ast),
        "duckdb")
    assert narrowed == [Narrowing(object="claim", rows=True, columns=False)]
    assert "amount > 0" in out.sql(dialect="duckdb").lower()


def test_WriteResult_carries_it_so_the_boundary_cannot_drop_it():
    """The field exists and defaults to None -- "not evaluated" -- rather than to `[]`, which
    would claim on every ungoverned write that the decision ran and narrowed nothing."""
    assert WriteResult(approved=True).narrowed is None
    r = WriteResult(approved=True, rows_affected=3,
                    narrowed=[Narrowed(object="claim", rows=True, columns=False)])
    assert r.narrowed and r.narrowed[0].object == "claim"


def test_the_runtime_threads_it_from_the_verdict_rather_than_defaulting():
    """Guards the boundary itself: the constructor call must read the verdict, not omit the
    argument and inherit the default that says nothing was evaluated."""
    import inspect

    from mnemiq import runtime

    src = inspect.getsource(runtime)
    assert src.count('narrowed=getattr(verdict, "narrowed", None)') >= 2, (
        "the approved and source-refused write paths must both carry the narrowing")


def test_the_write_surfaces_render_the_same_sentence_as_the_read_path():
    """One renderer, so a governed DELETE and a governed SELECT cannot disagree about wording."""
    assert disclosure_sentence([Narrowed(object="claim", rows=True, columns=False)]) == (
        "Some rows were withheld by policy.")


def test_both_write_surfaces_report_it_structurally():
    import inspect

    from mnemiq import cli
    from mnemiq.mcp import server

    assert '"narrowed"' in inspect.getsource(server._db_write), "MCP write drops the narrowing"
    cli_src = inspect.getsource(cli)
    assert 'payload["narrowed"]' in cli_src, "--json write emits stringified dataclasses"
    assert "disclosure_sentence(res.narrowed)" in cli_src, "human write output says nothing"
