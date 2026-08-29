"""M56 + M35 — lineage the audit store can trust, or that says it cannot be trusted.

The artifact is not a table list. A bare list is WORSE than the absent field it replaces: absent
reads as "not recorded", `[]` reads as "nothing was read", and `SELECT all_ssns() AS x` returns
both SSNs with `tables=[]`. So the list ships with a marker saying whether it is the whole story.

Three-valued on purpose, for the reason `rls_tables = 0` had to be: COMPLETE, INCOMPLETE (we know
something reaches past what we resolved) and UNKNOWN (we cannot tell). Collapsing the last two
into `False` would put a fail-open and an absence back on one value, which this codebase has now
recorded eleven times.
"""

import pytest
import sqlglot

from mnemiq.contract.semantic import Job, Snapshot, ViewDefinition
from mnemiq.sql.lineage import COMPLETE, INCOMPLETE, UNKNOWN, lineage_for
from mnemiq.sql.views import inventory_for


def _ast(sql: str):
    return sqlglot.parse_one(sql, read="duckdb")


def _snapshot(views=(), jobs=()):
    return Snapshot(version="v1", source_id="s", created_at="t", views=list(views), jobs=list(jobs))


_DISCOVERED = Job(id="discover:views", source_id="s", kind="discover", status="done")


# -- the control, first: the marker must not be degenerate ---------------------------------------


def test_an_ordinary_read_is_complete():
    """Without this every other assertion is satisfied by returning UNKNOWN always."""
    lineage = lineage_for(_ast("SELECT id, amount FROM claim"), ["claim"],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == COMPLETE
    assert lineage.tables == ["claim"]
    assert lineage.unresolved == []


# -- source 1: a function reaches where lineage cannot follow ------------------------------------


def test_a_function_call_makes_lineage_unknown_and_is_named():
    """M43's shape. Verified live: `SELECT all_ssns() AS x` is APPROVED, returns both SSNs, and
    reports `tables=[]` — against a `FROM person` control that refuses UNAUTHORIZED_TABLE. The
    marker is what stops that `[]` being recorded as "nothing was read".

    UNKNOWN rather than INCOMPLETE, and the distinction is the point: the engine does not KNOW
    that `all_ssns()` reads anything, only that it cannot rule it out. A view earns INCOMPLETE
    because its body is in the snapshot and the reach can be demonstrated. Asserting INCOMPLETE
    here — as the first draft of this test did — would claim knowledge the engine does not have.
    """
    lineage = lineage_for(_ast("SELECT all_ssns() AS x"), [],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert "all_ssns" in lineage.unresolved
    assert lineage.tables == []


def test_a_function_over_a_read_table_still_reports_the_table():
    """Incomplete does not mean empty: what WAS resolved is still the audit record's best evidence."""
    lineage = lineage_for(_ast("SELECT id FROM claim WHERE all_ssns() IS NOT NULL"), ["claim"],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert lineage.tables == ["claim"]
    assert "all_ssns" in lineage.unresolved


@pytest.mark.parametrize("sql", [
    "SELECT id FROM claim WHERE created_at < now()",
    "SELECT age(created_at) FROM claim",
    "SELECT current_timestamp FROM claim",
])
def test_a_pure_builtin_that_sqlglot_leaves_anonymous_does_not_poison_the_marker(sql):
    """The degeneracy control, and it is not hypothetical. Measured: of eighteen ordinary
    expressions only `now()` and `age()` parse to `Anonymous` — `coalesce`, `round`, `substr`,
    `md5`, `date_trunc`, `extract`, `random`, `uuid` and `string_agg` are all typed. So a rule
    that reads every `Anonymous` as unresolved reports UNKNOWN on `WHERE created_at < now()`,
    which is most real queries, and the marker stops meaning anything.

    `_PURE` is the whitelist that prevents it. It is deliberately short, because an unlisted
    function stays unresolved — the `views.py` pattern, where an unlisted shape must not pass.
    """
    lineage = lineage_for(_ast(sql), ["claim"], inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == COMPLETE, f"{sql} must not poison the marker"


def test_an_aggregate_is_not_an_unresolved_function():
    """`count`/`upper` reach nothing past their arguments. Without this the marker says INCOMPLETE
    on every real query and stops meaning anything."""
    lineage = lineage_for(_ast("SELECT count(*), upper(region) FROM claim"), ["claim"],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == COMPLETE
    assert lineage.unresolved == []


# -- source 2: a view's body reads what base_tables does not report -------------------------------


def test_reading_a_view_makes_lineage_incomplete():
    """The M27 floor declines views over FILTERED tables. An unfiltered view is approved and its
    body still reads base tables the lineage never names."""
    views = [ViewDefinition(object_id="claim_view", definition="SELECT * FROM claim",
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT id FROM claim_view"), ["claim_view"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == INCOMPLETE
    assert "claim_view" in lineage.unresolved


def test_a_demonstrable_gap_outranks_an_unclear_one():
    """A statement that reads a view AND calls an unclassifiable function is INCOMPLETE, not
    UNKNOWN: naming the view is more use to an auditor than recording that something was unclear,
    and UNKNOWN must not mask a gap the engine can actually point at."""
    views = [ViewDefinition(object_id="claim_view", definition="SELECT * FROM claim",
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT id FROM claim_view WHERE all_ssns() IS NOT NULL"),
                          ["claim_view"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == INCOMPLETE
    assert "claim_view" in lineage.unresolved
    assert "all_ssns" in lineage.unresolved, "the unclear reach is still recorded, not dropped"


# -- sources 3 and 4: we cannot tell whether a name is a view -------------------------------------


def test_an_unavailable_inventory_is_unknown_not_incomplete():
    """UNKNOWN and INCOMPLETE are different claims. Incomplete says "something reaches past this";
    unknown says "we could not establish whether anything does". M52 is the finding that the two
    had one value."""
    failed = Job(id="discover:views", source_id="s", kind="discover", status="failed")
    lineage = lineage_for(_ast("SELECT id FROM claim"), ["claim"],
                          inventory_for(_snapshot(jobs=[failed])))
    assert lineage.completeness == UNKNOWN


def test_a_snapshot_that_was_never_asked_about_views_is_unknown():
    """`inventory_for` deliberately treats a snapshot with NO `discover:views` job as available,
    because refusing a query on it would be harsh. Recording "cannot confirm" costs nothing, so
    the audit record is stricter than the guard — the same fact, different response."""
    lineage = lineage_for(_ast("SELECT id FROM claim"), ["claim"],
                          inventory_for(_snapshot()))  # no jobs at all
    assert lineage.completeness == UNKNOWN


# -- source 5: base_tables can over-report, and wrong is not the same as missing -------------------


def test_an_unresolvable_scope_is_unknown_because_the_tables_may_be_wrong():
    """When `build_scope` raises, `base_tables` returns `find_all(exp.Table)` — CTE aliases and
    all — so the record can name an object the query never read. The marker needs a state for
    WRONG, not only for MISSING."""
    lineage = lineage_for(_ast("SELECT id FROM claim"), ["claim", "x"],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])), scope_resolved=False)
    assert lineage.completeness == UNKNOWN


# -- the residual this design cannot close, pinned rather than described --------------------------


@pytest.mark.xfail(strict=True, reason="sqlglot assigns a function's node type from a NAME "
                                       "registry, so a customer UDF named after a builtin parses "
                                       "to a typed node and is indistinguishable from the builtin "
                                       "by parsing alone. Measured: all_ssns()/pg_sleep() are "
                                       "Anonymous, while left(x,2), log(x), trim(x) and "
                                       "concat(x,y) are Left/Log/Trim/Concat. Closing it needs a "
                                       "function inventory FROM THE SOURCE -- the same shape as "
                                       "the view inventory, with the same availability problem -- "
                                       "which is v2. Pinned so the limit cannot be forgotten, and "
                                       "so it flips the day the inventory lands.")
def test_a_udf_named_after_a_builtin_is_still_unresolved():
    lineage = lineage_for(_ast("SELECT log(x) FROM claim"), ["claim"],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == INCOMPLETE


# -- it has to reach the audit store, in the tier that is always on -------------------------------


def _trace_record(**settings_kw):
    """The trace as it would leave the process, built by the production builder."""
    import sys

    sys.path.insert(0, "tests")
    from test_verity_trace_sink import _Settings, _event

    from mnemiq.observability.trace_sink import VerityTraceSink

    return VerityTraceSink(_Settings(**settings_kw))._build_trace(_event())


def test_lineage_reaches_the_trace_at_all():
    """M56: `agent/trace.py` sets `tables_used=list(approved.tables)` on the Trace hanging off the
    same `answer` object `Runtime.ask` hands the sink, and the emitter never mentions it. The CLI
    prints it and the MCP server returns it; the audit store is the one consumer not told."""
    record = _trace_record()
    assert "lineage" in record, "the audit store must be told what the answer read"


def test_lineage_ships_in_the_always_tier_not_behind_the_text_opt_in():
    """Criterion 6, and the trap it exists to close: lineage placed beside the executed SQL under
    `_send_text()` satisfies every other criterion in a text-enabled deployment and delivers
    nothing to a default-closed one. That is the substitute M56's row rejects, since the
    deployments most likely to keep text off are the ones most likely to need an audit trail.

    Asserted against a DEFAULT settings object -- every tier flag off -- so an implementation that
    classifies lineage as text fails here and only here.
    """
    record = _trace_record()  # no opt-ins
    assert "lineage" in record
    assert record["lineage"]["completeness"] in (COMPLETE, INCOMPLETE, UNKNOWN)


def test_the_trace_never_carries_a_bare_table_list():
    """The whole design in one assertion. A list without a marker is worse than the absent field
    it replaces, so the two must be impossible to ship apart."""
    record = _trace_record()
    lineage = record.get("lineage")
    assert lineage is not None
    assert "tables" in lineage and "completeness" in lineage, (
        "tables and completeness ship together or not at all")


def test_real_lineage_survives_the_journey_to_the_trace():
    """The three tests above are satisfied by the fixture's Trace, which carries no lineage and so
    records UNKNOWN — correct behaviour, and a weak assertion: they would pass on an emitter that
    never threads the real thing. This one puts a populated Trace in and asserts it comes out."""
    import sys

    sys.path.insert(0, "tests")
    from test_verity_trace_sink import _Answer, _Settings, _event

    from mnemiq.observability.trace_sink import VerityTraceSink

    class _TraceWithLineage:
        target_sql = "SELECT id FROM claim_view"
        enrichment_version = "v7"
        tables_used = ["claim_view"]
        lineage_completeness = INCOMPLETE
        lineage_unresolved = ["claim_view"]

    answer = _Answer()
    answer.trace = _TraceWithLineage()
    record = VerityTraceSink(_Settings())._build_trace(_event(answer=answer))

    assert record["lineage"]["tables"] == ["claim_view"]
    assert record["lineage"]["completeness"] == INCOMPLETE
    assert record["lineage"]["unresolved"] == ["claim_view"]


def test_the_decider_puts_the_marker_on_the_verdict():
    """The other end of the same thread: `decide` computes lineage where it computes `tables`, so
    the two cannot be produced by different code paths and disagree — which is M7's shape."""
    from mnemiq.sql.decide import decide
    from mnemiq.sql.policy import AccessPolicy
    from mnemiq.sql.verdict import Approved

    class _Ok:
        def execute(self, sql):
            return []

    verdict = decide("SELECT id FROM claim", {"claim": {"id"}}, adapter=_Ok(),
                     dialect="duckdb", target="duckdb", policy=AccessPolicy())
    assert isinstance(verdict, Approved)
    assert verdict.lineage is not None
    assert verdict.lineage.completeness == UNKNOWN, (
        "no views were passed, so the inventory was never asked — cannot confirm")
    assert verdict.tables == ["claim"]
