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


# -- the spellings the guard resolves, the marker must resolve too --------------------------------


@pytest.mark.parametrize("sql", [
    "SELECT id FROM claim_view",
    "SELECT id FROM CLAIM_VIEW",
    "SELECT id FROM public.claim_view",
])
def test_a_view_is_recognised_in_every_spelling_check_views_recognises(sql):
    """`check_views` folds case and falls back to the bare segment, so it refuses UNGOVERNED_VIEW
    on all three. A marker using exact-string `name in views` reported COMPLETE on the last two —
    the audit record asserting "every read was resolved" about a statement that read a view, which
    is the one claim this artifact exists to prevent. Both sides now share `spellings`."""
    views = [ViewDefinition(object_id="claim_view", definition="SELECT * FROM claim",
                            dialect="duckdb")]
    lineage = lineage_for(_ast(sql), [sql.split()[-1]],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == INCOMPLETE


def test_a_view_that_reaches_nothing_new_is_not_a_gap():
    """INCOMPLETE means reach was DEMONSTRATED, and membership is not demonstration. A view whose
    body names only tables already in the list adds nothing to account for — the first version of
    this module reported INCOMPLETE on it, and on a view defined `SELECT 1 AS x` that reaches
    nothing at all."""
    views = [ViewDefinition(object_id="claim_view", definition="SELECT * FROM claim",
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT v.id FROM claim_view v JOIN claim c ON c.id = v.id"),
                          ["claim_view", "claim"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == COMPLETE, "the body's reach is already in the list"


def test_object_names_and_engine_reason_codes_do_not_share_a_list():
    """They ship to the audit store and into the published contract. Mixed, a consumer rendering
    `unresolved` as objects shows `scope-unresolved` as a table name, and one filtering for reason
    codes matches a real object named after one."""
    lineage = lineage_for(_ast("SELECT all_ssns() AS x"), [], inventory_for(_snapshot()))
    assert "all_ssns" in lineage.unresolved
    assert "view-inventory-never-asked" in lineage.reasons
    assert not any(r in lineage.unresolved for r in lineage.reasons)


def test_a_views_reach_into_a_different_qualified_object_is_not_accounted_for():
    """The two comparisons in `lineage_for` have OPPOSITE polarity, and using one widening helper
    for both was a regression that made COMPLETE reachable for a view when it never had been.

    Measured on the broken version: inventory `{"pg.claim_v": SELECT * FROM claim}`, statement
    reading `pg.claim_v` joined to `mysql.claim`, tables `["pg.claim_v", "mysql.claim"]` →
    `complete`, while the view actually reads `pg.claim`, which the list never names. Widening the
    view LOOKUP adds refusals and is safe; widening the REACH comparison adds claims of
    completeness and is not. `object_key` exists for this hazard and the fix is to respect it.
    """
    views = [ViewDefinition(object_id="pg.claim_v", definition="SELECT * FROM claim",
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT id FROM pg.claim_v JOIN mysql.claim USING (id)"),
                          ["pg.claim_v", "mysql.claim"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == INCOMPLETE, (
        "a bare-vs-qualified match must count as unresolved, never as accounted for")
    assert "pg.claim_v" in lineage.unresolved


def test_the_fourth_spelling_resolves_like_the_other_three():
    """The membership test used `spellings` (four forms) and retrieval was hand-rolled with three,
    omitting `bare.lower()`. So `public.CLAIM_VIEW` matched, failed to retrieve, and fell into the
    cannot-parse branch — UNKNOWN where the other three spellings give INCOMPLETE. Conservative,
    and still two ways of asking one question, which is the drift the fix claimed to close."""
    views = [ViewDefinition(object_id="claim_view", definition="SELECT * FROM claim",
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT id FROM public.CLAIM_VIEW"), ["public.CLAIM_VIEW"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == INCOMPLETE


def test_a_cte_inside_a_view_body_is_not_reach():
    """A local alias names nothing outside the body. Counting it reported INCOMPLETE for a view
    whose only real base was already in the list — so `base_tables` rather than `find_all`, which
    is M31/M49's lesson applied one consumer later."""
    views = [ViewDefinition(object_id="claim_view",
                            definition="WITH c AS (SELECT * FROM claim) SELECT * FROM c",
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT v.id FROM claim_view v JOIN claim c ON c.id = v.id"),
                          ["claim_view", "claim"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == COMPLETE, "the body's only real base is already accounted for"


# -- the seven findings a second model found, one test each ---------------------------------------


def test_a_function_inside_a_view_body_is_reach_the_body_scan_must_see():
    """Checking a view's body only for base TABLES reported a view of `SELECT all_ssns()` as fully
    accounted for. The reach is one level down, not absent — so the function scan runs on the body
    as well as on the caller's statement."""
    views = [ViewDefinition(object_id="secret_view", definition="SELECT all_ssns() AS ssn",
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT ssn FROM secret_view"), ["secret_view"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert "all_ssns" in lineage.unresolved


def test_a_qualified_call_does_not_inherit_a_builtins_exemption():
    """`public.now()` parses to an `Anonymous` whose `name` is the bare leaf `now`, so matching
    `_PURE` on the leaf let a schema-qualified callable — a UDF named `now` in any schema — inherit
    the builtin's exemption. A qualified call has an `exp.Dot` parent, which is the structural form
    of "this is not the builtin you whitelisted", and the qualifier is kept in `unresolved` so the
    record names what it could not account for rather than a leaf matching several things."""
    lineage = lineage_for(_ast("SELECT public.now() AS x FROM claim"), ["claim"],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert "public.now" in lineage.unresolved  # qualifier kept, arguments never


def test_a_quoted_identifier_that_matches_only_when_folded_is_unknown():
    """Quoting MAY decide identity and this function cannot tell whether it does. In Postgres a
    quoted identifier is case-sensitive, so `"Claim"` and `claim` are two objects; DuckDB folds
    them to one — measured in this repo's venv — and `qualify.py` states the rule correctly scoped
    as "case-sensitive in Postgres". `lineage_for` takes no dialect.

    So this is UNKNOWN, not INCOMPLETE. Folding both was the first version and asserted the wrong
    one; an earlier draft of this very test asserted INCOMPLETE, which is equally wrong on DuckDB,
    the repo's own default engine. The honest claim is that identity is engine-dependent here.
    """
    views = [ViewDefinition(object_id="v", definition='SELECT id FROM "Claim"', dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT id FROM v JOIN claim USING (id)"), ["v", "claim"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert any(u.startswith("case-ambiguous:") for u in lineage.unresolved)


def test_a_quoted_qualifier_decides_identity_as_much_as_a_quoted_leaf():
    """`object_key` composes catalog/db/name, so inspecting only the leaf's `quoted` flag folded a
    quoted mixed-case QUALIFIER — `"Public".claim` against `public.claim` read COMPLETE."""
    views = [ViewDefinition(object_id="v", definition='SELECT id FROM "Public".claim',
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT id FROM v"), ["v", "public.claim"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN


def test_a_function_label_never_carries_its_arguments():
    """`unresolved` ships in the emitter's ALWAYS tier, so a label rendered from the whole call
    expression posts question and database literals to an external audit store on a deployment
    running the default `verity_trace_send_text=False`. Measured before the fix:
    `["public.mask(ssn, 'sk-live-abc123')"]`. An identifier is an object id; a rendered call is
    question text, and the tier table is what keeps those apart."""
    for sql in ("SELECT public.mask(ssn, 'sk-live-abc123') FROM claim",
                # The sibling spelling. The first fix stripped arguments where the call is the
                # Dot's EXPRESSION and left them where it is the Dot's `this`, so `mask(...).tag`
                # still rendered the whole call -- one leak fixed by reasoning about one shape.
                "SELECT mask(ssn, 'sk-live-abc123').tag FROM claim"):
        lineage = lineage_for(_ast(sql), ["claim"], inventory_for(_snapshot(jobs=[_DISCOVERED])))
        assert not any("sk-live" in u for u in lineage.unresolved), f"secret in ALWAYS tier: {sql}"
        assert not any("(" in u or "'" in u for u in lineage.unresolved), sql


def test_a_field_access_on_a_builtin_is_not_a_qualified_call():
    """`isinstance(parent, exp.Dot)` matched both sides of the dot, so `now().y` read as a
    qualified callable and lost its `_PURE` exemption. Qualified means the call is the Dot's
    EXPRESSION -- `schema.func()` -- not its `this`."""
    lineage = lineage_for(_ast("SELECT now().y FROM claim"), ["claim"],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == COMPLETE


def test_a_table_valued_function_is_unclassified_not_a_demonstrated_gap():
    """`object_key` returns "" for `generate_series`/`unnest` because there is no identifier to
    key, and treating that as an unaccounted OBJECT reported INCOMPLETE for a view that reaches no
    object at all. It is reach we cannot classify, which is UNKNOWN."""
    views = [ViewDefinition(object_id="g", definition="SELECT * FROM generate_series(1, 10)",
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT * FROM g"), ["g"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert any("generate_series" in u for u in lineage.unresolved)


def test_one_sources_discovery_does_not_vouch_for_a_federated_snapshot():
    """`any(...)` let a single completed `discover:views` job mark a merged snapshot as asked, so a
    query of a catalog that was never discovered came back COMPLETE — defeating the never-asked
    distinction exactly where federation makes it matter."""
    from mnemiq.semantic.federation import FederatedSnapshot

    def _job(source):
        return Job(id="discover:views", source_id=source, kind="discover", status="done")

    registry = {"a": "schema_a", "legacy": "schema_legacy"}
    partial = FederatedSnapshot(version="v1", source_id="f", created_at="t",
                                jobs=[_job("a")], registry=registry)
    assert inventory_for(partial).asked is False, "one catalog discovered, one never asked"

    full = FederatedSnapshot(version="v1", source_id="f", created_at="t",
                             jobs=[_job("a"), _job("legacy")], registry=registry)
    assert inventory_for(full).asked is True


def test_the_public_serializers_never_ship_a_bare_table_list():
    """The HTTP and MCP projections emitted `tables_used` and none of the marker fields, so a trace
    with `tables_used=[]` and `completeness='unknown'` reached clients as an empty list — the exact
    misleading artifact this branch exists to make impossible, on the two surfaces a customer
    actually reads."""
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.agent.trace import Trace
    from mnemiq.contract import IdentityContext
    from mnemiq.server.serialize import answer_payload

    trace = Trace(question="q", plan_sql="SELECT all_ssns()", target_sql="SELECT all_ssns()",
                  result_shape="scalar", timing={}, enrichment_version="v7",
                  identity=IdentityContext(tenant_id="t", principal_id="p", roles=[]),
                  tables_used=[], lineage_completeness=UNKNOWN,
                  lineage_unresolved=["all_ssns"], lineage_reasons=[])
    payload = answer_payload(AgentAnswer(answer="x", trace=trace))

    # Asserted on the PAYLOAD, not on module source text. The first version grepped
    # `inspect.getsource` for the word "lineage", which survives dropping "completeness" from the
    # emitted dict or making its value unconditionally None -- it tested that a word appears in a
    # file, while claiming to test that a list and its marker cannot separate.
    assert payload["lineage"]["completeness"] == UNKNOWN
    assert payload["lineage"]["unresolved"] == ["all_ssns"]
    assert payload["tables_used"] == [], "the bare list is still there, and now it is qualified"

    # BOTH surfaces. The previous version asserted only the HTTP payload while its name and the
    # commit message claimed "the two surfaces a customer actually reads" -- so dropping
    # `completeness` from the MCP dict kept the suite green, which is the degradation this test
    # exists to catch, surviving on one of the two.
    from mnemiq.mcp.server import _db_read

    class _Runtime:
        def ask(self, question, identity, mode=None):
            return AgentAnswer(answer="x", trace=trace)

    mcp = _db_read(_Runtime(), IdentityContext(tenant_id="t", principal_id="p", roles=[]), "q")
    assert mcp["trace"]["lineage"]["completeness"] == UNKNOWN
    assert mcp["trace"]["lineage"]["unresolved"] == ["all_ssns"]
