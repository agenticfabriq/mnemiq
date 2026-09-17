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
    assert "v_totals" in out and "reaches past the tables in its trace" in out
    # And it must not call them tables: `unresolved` carries this engine's own tokens beside
    # object names, so `unmodelled-source:lateral` would be announced as a table.
    assert "Unaccounted:" in out


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


class _Inv:
    """Enough of a FunctionInventory for the advisory to read."""

    def __init__(self, available=True, asked=True, names=()):
        self.available, self.asked, self.names = available, asked, frozenset(names)


def _advise(monkeypatch, inv, acknowledged=frozenset()):
    from mnemiq import runtime

    monkeypatch.setattr("mnemiq.sql.functions.inventory_from", lambda _a: inv)
    runtime._warn_unconfirmable_functions(object(), acknowledged)


@pytest.mark.parametrize(
    ("inv", "expected", "forbidden"),
    [
        (_Inv(available=False), "UNAVAILABLE", "does not implement"),
        (_Inv(asked=False), "NEVER ASKED", "could not answer"),
        (_Inv(names=["helper"]), "defines 1 function", "UNAVAILABLE"),
    ],
)
def test_the_boot_advisory_keeps_its_three_states_apart(caplog, monkeypatch, inv, expected,
                                                       forbidden):
    """BEHAVIOURALLY, because asserting the three field names appear in the source passes even
    with the message bodies SWAPPED -- an outage then tells the operator to go fix a working
    adapter, which is the absence-versus-failure collapse M102 and M104 each cost.

    Each case also asserts what must NOT be said, since that is what a swap breaks.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        _advise(monkeypatch, inv)
    assert expected in caplog.text
    assert forbidden not in caplog.text


def test_a_healthy_schema_with_no_helpers_says_nothing_at_boot():
    """No verdict, no line. An advisory that fires on every boot regardless is the same
    wallpaper problem one layer up."""
    import logging

    from mnemiq import runtime

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr("mnemiq.sql.functions.inventory_from", lambda _a: _Inv())
        import logging as _l

        records = []
        handler = _l.Handler()
        handler.emit = records.append
        logger = _l.getLogger("mnemiq.runtime")
        logger.addHandler(handler)
        try:
            runtime._warn_unconfirmable_functions(object())
        finally:
            logger.removeHandler(handler)
        assert not [r for r in records if r.levelno >= logging.WARNING]
    finally:
        mp.undo()


def test_an_acknowledged_verdict_is_still_RECORDED_at_info(caplog, monkeypatch):
    """`MNEMIQ_ACK_ADVISORIES` is defined as "log at INFO instead of WARNING", and the sibling
    advisory does exactly that. Returning silently made an acknowledged verdict identical at
    every level to one that never occurred, so an operator could not confirm their
    acknowledgement had matched anything."""
    import logging

    with caplog.at_level(logging.INFO):
        _advise(monkeypatch, _Inv(names=["helper"]), frozenset({"functions:defines"}))
    assert "defines 1 function" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "an acknowledged verdict still warned")


def test_the_sibling_validator_accepts_these_keys(caplog):
    """THE BLOCKER. `known` was built from one function's local tuple, so `functions:defines`
    was reported as naming no advisory -- false, since it does silence one -- and sent the
    operator hunting a typo they did not make.

    A real typo must still warn, or the fix would have traded one collapse for another.
    """
    import logging

    from mnemiq import runtime

    class _Attached:
        def assert_enforcing(self):
            return "attached", "ok"

    with caplog.at_level(logging.WARNING):
        runtime._warn_source_enforcement(_Attached(), frozenset({"functions:defines"}))
    assert "names no advisory" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        runtime._warn_source_enforcement(_Attached(), frozenset({"bogus:x"}))
    assert "names no advisory" in caplog.text and "functions" in caplog.text


def test_a_key_that_silenced_nothing_says_so(caplog, monkeypatch):
    """The gap the sibling's skip opened. It stopped reporting on `functions:` keys, correctly,
    because it cannot produce their verdicts -- and left them unmentioned by ANYONE, which is
    the same silence an unset key produces.

    Asserted at INFO on purpose: the earlier test's floor was WARNING, so a false diagnosis
    here sat below it and deleting the skip left the suite green.
    """
    import logging

    with caplog.at_level(logging.INFO):
        _advise(monkeypatch, _Inv(), frozenset({"functions:unavailable"}))
    assert "did not apply" in caplog.text

    # And the key that DID match must not be reported as unused.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _advise(monkeypatch, _Inv(names=["helper"]), frozenset({"functions:defines"}))
    assert "did not apply" not in caplog.text


def test_each_verdict_is_silenced_separately(caplog, monkeypatch):
    """Acknowledging a schema's helpers must not also silence an outage. Per verdict, like the
    sibling advisory, because the defines-helpers case is the normal state of a healthy schema
    and the other two are not."""
    import logging

    with caplog.at_level(logging.WARNING):
        _advise(monkeypatch, _Inv(names=["helper"]), frozenset({"functions:defines"}))
    assert not caplog.text

    with caplog.at_level(logging.WARNING):
        _advise(monkeypatch, _Inv(available=False), frozenset({"functions:defines"}))
    assert "UNAVAILABLE" in caplog.text, "acknowledging helpers silenced an outage"


def test_the_advisory_is_actually_called_at_boot():
    """Wired at one end with a green suite is the M26 shape this repo has a row for."""
    import inspect

    from mnemiq import runtime

    calls = [ln.strip() for ln in inspect.getsource(runtime).splitlines()
             if ln.strip().startswith("_warn_unconfirmable_functions(") and ln.startswith(" ")]
    assert calls, "the advisory is defined and never called"
    # The ARGUMENT too. Matching the name alone let the acknowledgement be dropped from the one
    # product line this feature adds, with every acknowledgement test still green -- they call
    # the function directly. M26 reopened one argument to the right.
    assert all("_acknowledged(settings)" in c for c in calls), (
        "the advisory is called without the acknowledgement, so the setting reaches nothing")


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
