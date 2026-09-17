"""The disclosure signal: what reaches the ANSWER, and what is said once at boot instead.

§8b called this the last of the completeness marker's three jobs. It had reached the trace and
the CLI mark since M56 and never the sentence a human reads or an agent consumes.

The split these pin is measured, not chosen. `calls_are_confirmable` is
`licensed and not inventory.names`, so one unrelated helper in the schema flips it and every
statement calling COUNT or SUM then reports lineage `unknown`: BIRD/Spider 6% -> 92% of
answers, the ACME demo 0% -> 88%, the live Oracle container 92%. A per-answer sentence keyed on
that is wallpaper, which is how M35's deferral guard died at a lower rate.
"""

import pytest

from mnemiq.contract.seams import lineage_disclosure_sentence as sentence

SOURCE_LEVEL = [
    "unconfirmed-function-identity",
    "function-inventory-unavailable",
    "function-inventory-never-asked",
    "function-inventory-covers-no-view-bodies",
    "view-inventory-unavailable",
    "view-inventory-never-asked",
]


def test_a_complete_answer_says_nothing():
    assert sentence("complete", [], []) == ""


@pytest.mark.parametrize("reason", SOURCE_LEVEL)
def test_a_source_level_unknown_says_nothing_per_answer(reason):
    """Each of these is true of EVERY answer this deployment will ever give, so none of them
    supports a claim about this one. `_warn_unconfirmable_functions` says it once at boot."""
    assert sentence("unknown", [], [reason]) == ""


def test_the_92_percent_case_is_silent():
    """The shape measured at 88-92% on three corpora: a schema defining one helper, a query
    calling an aggregate. Exactly the answer that must NOT carry a caveat."""
    assert sentence("unknown", [], ["unconfirmed-function-identity"]) == ""


def test_an_object_this_statement_named_is_disclosed():
    """`unresolved` holds ids and function names from the caller's world -- a fact about the
    question that was asked, not about the database."""
    out = sentence("unknown", ["julianday"], ["unconfirmed-function-identity"])
    assert "julianday" in out and "could not be accounted for" in out


def test_a_scope_this_statement_defeated_is_disclosed():
    """`scope-unresolved` is the one reason code that is about the statement."""
    assert sentence("unknown", [], ["scope-unresolved"]) != ""


def test_a_demonstrable_reach_past_the_list_reads_differently():
    """INCOMPLETE is not UNKNOWN: the engine can point at the view. The sentence says the list
    is not all of them, rather than that something was unclear."""
    out = sentence("incomplete", ["v_totals"], [])
    assert "v_totals" in out and "not necessarily all of them" in out


def test_a_source_reason_does_not_suppress_a_statement_fact():
    """Both present is the ordinary case on a source that defines helpers. The statement fact
    still has to get out -- suppressing on the presence of a source reason would hide it."""
    out = sentence("unknown", ["printf"], ["unconfirmed-function-identity", "scope-unresolved"])
    assert "printf" in out


def test_the_list_is_bounded():
    out = sentence("unknown", [f"obj{i}" for i in range(9)], [])
    assert "..." in out and out.count(",") <= 3


def test_the_answer_carries_it_so_every_surface_does():
    """The join, asserted the way `test_disclosure_surfaces` asserts the sibling's.

    Source-level because nothing in the suite drives the loop's assembly, and this is the
    arrangement that makes "we fixed three of the four surfaces" impossible: the sentence is
    appended once where the answer is built, and CLI, HTTP, MCP and `--json` inherit it by
    printing `answer`. The lineage marker is the one that was fixed surface-by-surface before,
    which is why it gets the same guard rather than a promise.
    """
    import inspect

    from mnemiq.agent import loop

    src = inspect.getsource(loop)
    assert "lineage_disclosure_sentence(" in src, (
        "the answer no longer carries the lineage disclosure; each surface would render it")
    i = src.index("lineage_disclosure_sentence(")
    j = src.index("return AgentAnswer(answer=answer")
    assert i < j, "the disclosure must be appended BEFORE the answer is returned"


def test_the_boot_advisory_exists_and_keeps_its_three_states_apart():
    """The other half of the split. A source that DEFINES functions, one that could not be
    ASKED, and one that was asked and FAILED need different actions, and collapsing them is the
    absence-versus-failure defect this register has already paid for twice (M102, M104)."""
    import inspect

    from mnemiq import runtime

    src = inspect.getsource(runtime._warn_unconfirmable_functions)
    assert "inventory.available" in src, "an outage reads as a source that defines helpers"
    assert "inventory.asked" in src, "an adapter that cannot answer reads as one that answered"
    assert "inventory.names" in src
    # The CALL, not the def -- a bare name count matches `def _warn_unconfirmable_functions(
    # adapter)` too and passes on an advisory nothing invokes. Indentation is what separates
    # them, and being wired at one end with a green suite is exactly the M26 shape.
    calls = [ln for ln in inspect.getsource(runtime).splitlines()
             if ln.strip() == "_warn_unconfirmable_functions(adapter)" and ln.startswith(" ")]
    assert calls, "the advisory is defined and never called, which is the M26 shape"


def test_append_notes_is_the_join_and_it_executes():
    """What the source-text guard above cannot reach.

    Asserting `lineage_disclosure_sentence(` appears in `loop.py` proves the call exists, not
    that its result is used: disabling the append left all fifteen tests green, measured. This
    exercises the composition instead.
    """
    from mnemiq.contract.seams import append_notes

    assert append_notes("42") == "42", "an answer with nothing to disclose is unchanged"
    assert append_notes("42", "", "") == "42", "empty notes must not add blank lines"
    assert append_notes("42", "A.") == "42\n\nA."
    assert append_notes("42", "A.", "B.") == "42\n\nA.\n\nB."
    assert append_notes("42", "", "B.") == "42\n\nB.", "a silent first note must not swallow"


def test_policy_comes_before_engine_limits_in_the_answer():
    """Order is a product decision, not an accident of which line was written first: what was
    withheld FROM them is a decision someone made, and outranks what the engine could not
    account for, which is a limit of ours."""
    from mnemiq.contract.seams import Narrowed, append_notes, disclosure_sentence

    out = append_notes(
        "42",
        disclosure_sentence([Narrowed(object="claim", rows=True, columns=False)]),
        sentence("incomplete", ["v_totals"], []),
    )
    assert out.index("withheld by policy") < out.index("v_totals")
