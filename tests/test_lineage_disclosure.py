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

    covers_view_bodies = True

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


class _Views(dict):
    def __init__(self, available=True, asked=True):
        super().__init__()
        self.available, self.asked = available, asked


def _advise_views(monkeypatch, views, acknowledged=frozenset()):
    from mnemiq import runtime

    monkeypatch.setattr("mnemiq.sql.views.inventory_for", lambda _s: views)
    runtime._warn_view_inventory(object(), acknowledged)


# The view advisory reads NO function inventory -- that independence is the whole point, and
# it is why parametrising these over function-inventory states was inert: `fn_state` could not
# reach the assertion. The property that the two do not depend on each other is structural and
# belongs to `test_both_advisories_are_called_at_boot_and_NEITHER_is_nested_in_the_other`,
# which is the only thing that detects the nesting. These pin the advisory's own behaviour.


def test_every_source_level_reason_reaches_a_reader():
    """THE CLAIM THE SPLIT MAKES, and it was false for three of the six.

    The per-answer sentence suppresses all six source-level codes, correctly -- each holds for
    every answer the deployment will give. That is only sound if a boot line says each of them
    somewhere. It did not: the advisory branched on `available`, `asked` and `names`, so a
    licence that misses view bodies and either view-inventory state were silent at BOTH readers
    while every answer carrying a call reported `unknown`.

    Enumerated against the frozenset, so a seventh code fails on set equality first rather than
    slipping through with no entry.
    """
    import inspect

    from mnemiq import runtime
    from mnemiq.contract.seams import _SOURCE_LEVEL_LINEAGE_REASONS

    advisories = (inspect.getsource(runtime._warn_unconfirmable_functions)
                  + inspect.getsource(runtime._warn_view_inventory))
    covered = {
        "unconfirmed-function-identity": "inventory.names",
        "function-inventory-unavailable": "inventory.available",
        "function-inventory-never-asked": "inventory.asked",
        "function-inventory-covers-no-view-bodies": "inventory.covers_view_bodies",
        "view-inventory-unavailable": 'views, "available"',
        "view-inventory-never-asked": 'views, "asked"',
    }
    assert set(covered) == set(_SOURCE_LEVEL_LINEAGE_REASONS), (
        "a source-level reason has no entry here; it is suppressed per-answer and may reach "
        "no boot line either")
    for reason, branch in covered.items():
        assert branch in advisories, f"{reason} is suppressed per-answer and said by no advisory"


def test_a_licence_that_misses_view_bodies_is_announced(caplog, monkeypatch):
    """The FOURTH function case. A source defining nothing still cannot certify a view body's
    calls, so none of the defines/asked/available branches covers it."""
    import logging

    class _NoViewBodies(_Inv):
        covers_view_bodies = False

    with caplog.at_level(logging.WARNING):
        _advise(monkeypatch, _NoViewBodies())
    assert "VIEW BODIES" in caplog.text


@pytest.mark.parametrize(
    ("views", "expected"),
    [(_Views(available=False), "UNAVAILABLE"), (_Views(asked=False), "NEVER ASKED")],
)
def test_the_view_inventory_states_are_announced(caplog, monkeypatch, views, expected):
    """A snapshot whose view discovery failed or never ran downgrades every statement reading a
    view, and the function advisory never looked at views at all."""
    import logging

    with caplog.at_level(logging.WARNING):
        _advise_views(monkeypatch, views)
    assert expected in caplog.text


def test_a_views_acknowledgement_is_diagnosable(caplog, monkeypatch):
    """A misspelled `views:` key, a correct one, and an unset one must not be one observable."""
    import logging

    with caplog.at_level(logging.INFO):
        _advise_views(monkeypatch, _Views(), frozenset({"views:typo-half"}))
    assert "did not apply" in caplog.text


def test_a_healthy_view_inventory_is_quiet(caplog, monkeypatch):
    import logging

    with caplog.at_level(logging.WARNING):
        _advise_views(monkeypatch, _Views())
    assert not caplog.text


def test_both_advisories_are_called_at_boot_and_NEITHER_is_nested_in_the_other():
    """The wiring, which no test can reach by calling the advisories itself.

    Two mutations survived a suite that drove both functions directly: nesting
    `_warn_view_inventory` back inside `_warn_unconfirmable_functions`, and deleting its boot
    call outright. The first is the defect this change exists to fix -- nested, it runs only
    for a source with no helpers, no failure and a covering licence, which is every deployment
    except the one that needs it least.

    Structural rather than behavioural because the boot path assembles a Runtime from a live
    store, and structural rather than a substring: the nesting check asks whether the call
    appears INSIDE the other function.
    """
    import inspect

    from mnemiq import runtime

    boot = inspect.getsource(runtime.build_runtime)
    for name in ("_warn_unconfirmable_functions(", "_warn_view_inventory("):
        assert name in boot, f"{name} is never called at boot"

    fn_advisory = inspect.getsource(runtime._warn_unconfirmable_functions)
    assert "_warn_view_inventory(" not in fn_advisory, (
        "the view advisory is nested inside the function advisory, so it runs only when that "
        "one finds nothing to report -- the two degrade independently")


def test_a_missing_snapshot_reports_rather_than_returning_quietly(caplog):
    """`inventory_for` calls a missing snapshot definitively VIEWS_UNAVAILABLE, so an early
    return here would suppress the one state it is sure about. Unreachable at boot today, and
    pinned so the branch cannot come back as a convenience."""
    import logging

    from mnemiq import runtime

    with caplog.at_level(logging.WARNING):
        runtime._warn_view_inventory(None, frozenset())
    assert "UNAVAILABLE" in caplog.text


def test_a_hot_swapped_snapshot_re_runs_the_view_advisory(caplog, monkeypatch):
    """The advisory describes the SNAPSHOT, and `reload_if_stale` is what replaces it.

    Running it only at `build_runtime` meant a replica that booted on a healthy snapshot and
    hot-swapped to one whose view discovery failed reported lineage `unknown` on every answer
    reading a view, under a boot log that said otherwise. The state it describes had changed
    and the reader had not been told.
    """
    import logging

    from mnemiq.runtime import Runtime

    class _Settings:
        control_dsn = "postgresql://x"
        ack_advisories = ""

    rt = Runtime.__new__(Runtime)
    rt.settings, rt.con = _Settings(), object()
    rt.snapshot, rt.loaded_versions = object(), {"src": "v1"}
    rt.authz = None  # `_warn_policy_advisories` returns on a provider with no roles

    monkeypatch.setattr("mnemiq.runtime.load_current_snapshot",
                        lambda _s, _c: (object(), {"src": "v2"}))
    monkeypatch.setattr("mnemiq.sql.views.inventory_for", lambda _s: _Views(available=False))

    with caplog.at_level(logging.WARNING):
        rt.reload_if_stale()
    assert rt.loaded_versions == {"src": "v2"}, "the swap did not happen"
    assert "view inventory UNAVAILABLE" in caplog.text, "the swap was silent"


def test_an_unchanged_version_does_not_re_announce(caplog, monkeypatch):
    """Only on an actual swap. `reload_if_stale` runs per ask, so warning on every call would
    be the wallpaper this whole split exists to avoid."""
    import logging

    from mnemiq.runtime import Runtime

    class _Settings:
        control_dsn = "postgresql://x"
        ack_advisories = ""

    rt = Runtime.__new__(Runtime)
    rt.settings, rt.con = _Settings(), object()
    rt.snapshot, rt.loaded_versions = object(), {"src": "v1"}
    rt.authz = None  # `_warn_policy_advisories` returns on a provider with no roles

    from mnemiq import runtime as _rt

    policy_calls = []
    monkeypatch.setattr("mnemiq.runtime.load_current_snapshot",
                        lambda _s, _c: (object(), {"src": "v1"}))
    monkeypatch.setattr("mnemiq.sql.views.inventory_for", lambda _s: _Views(available=False))
    # OBSERVABLE, not `authz=None`. The real `_warn_policy_advisories` returns immediately on a
    # provider with no roles, so a version of this test that let it run could not tell "guarded
    # by the version check" from "returned before it could log" -- and unindenting the call out
    # of that check passed.
    monkeypatch.setattr(_rt, "_warn_policy_advisories", lambda _a, s: policy_calls.append(s))

    with caplog.at_level(logging.WARNING):
        rt.reload_if_stale()
    assert not caplog.text, "the view advisory re-announced without a version change"
    assert not policy_calls, "the policy advisory ran without a version change"


def test_the_policy_advisory_re_runs_on_a_swap_too(monkeypatch):
    """`warn_unfiltered_dependents` reads `snapshot.relationships`, so it drifts exactly as the
    view one did: a swap bringing new dependent tables off a role's tenancy axis would be
    reported against the boot snapshot. Every advisory whose SUBJECT is the snapshot re-runs."""
    from mnemiq import runtime
    from mnemiq.runtime import Runtime

    class _Settings:
        control_dsn = "postgresql://x"
        ack_advisories = ""

    seen = []
    rt = Runtime.__new__(Runtime)
    rt.settings, rt.con, rt.authz = _Settings(), object(), None
    rt.snapshot, rt.loaded_versions = object(), {"src": "v1"}

    monkeypatch.setattr("mnemiq.runtime.load_current_snapshot",
                        lambda _s, _c: (object(), {"src": "v2"}))
    monkeypatch.setattr("mnemiq.sql.views.inventory_for", lambda _s: _Views())
    monkeypatch.setattr(runtime, "_warn_policy_advisories",
                        lambda _a, snap: seen.append(snap))

    rt.reload_if_stale()
    assert seen, "the policy advisory was not re-run on the swap"
    assert seen[0] is rt.snapshot, "it was re-run against the OLD snapshot"
