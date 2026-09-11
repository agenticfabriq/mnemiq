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
    # NOT named — that is what `_PURE` is still for, and naming `now` as an unaccounted object
    # would be noise. But no call licenses COMPLETE any more, because sqlglot cannot tell a
    # builtin from a source UDF of the same name, so the marker records the uncertainty as a
    # reason rather than as a suspect.
    assert lineage.unresolved == [], f"{sql} must not name a builtin as unaccounted"
    assert lineage.completeness == UNKNOWN
    assert "unconfirmed-function-identity" in lineage.reasons


def test_an_aggregate_is_not_an_unresolved_function():
    """`count`/`upper` reach nothing past their arguments. Without this the marker says INCOMPLETE
    on every real query and stops meaning anything."""
    lineage = lineage_for(_ast("SELECT count(*), upper(region) FROM claim"), ["claim"],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.unresolved == [], "an aggregate is not an unaccounted object"
    assert lineage.completeness == UNKNOWN, "but a call is still a call"


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


def test_a_udf_named_after_a_builtin_no_longer_certifies_completeness():
    """This was a strict xfail, and the narrowing closed it — from the other side.

    The residual: sqlglot assigns a function's node type from a NAME registry, so a source UDF
    called `log` parses to `exp.Log` and an `Anonymous` scan never sees it. I pinned that as an
    unclosable limit needing a function inventory, and went on emitting COMPLETE beside it — a
    documented false negative sitting next to a confident claim, which is the pairing a reviewer
    called affirmatively false.

    It is not closed by identifying the function. It is closed by no longer claiming what depends
    on identifying it: a call the inventory cannot clear downgrades COMPLETE, so a shadowing UDF cannot ride in on a
    COMPLETE it did not earn. The name is still unresolvable and the audit record says so.
    """
    lineage = lineage_for(_ast("SELECT log(x) FROM claim"), ["claim"],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert "unconfirmed-function-identity" in lineage.reasons


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
    # UNKNOWN, not COMPLETE, and the change is deliberate. This test asserted COMPLETE until a
    # fourth review pass named the class: a lexical match is not identity. The view's body says
    # `claim` and the caller says `claim`, and whether those are one object depends on the schema
    # each was bound in -- which `ViewDefinition` does not carry. What the test still pins is that
    # this is NOT a demonstrated gap, which is the property it was written for.
    assert lineage.completeness == UNKNOWN
    assert lineage.completeness != INCOMPLETE, "nothing here is a demonstrated gap"
    assert any(u.startswith("unconfirmed-identity:") for u in lineage.unresolved)


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
    # Not INCOMPLETE is the property: the CTE alias must not read as reach. Not COMPLETE either,
    # since a view is involved and cross-context identity cannot be confirmed.
    assert lineage.completeness == UNKNOWN
    assert not any(u == "c" for u in lineage.unresolved), "a CTE alias is not a base table"


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
    # `case-ambiguous:` became `unconfirmed-identity:` when the rule generalised: quoting was one
    # reason a lexical match might not be identity, and binding context is another. One label for
    # one question.
    assert any(u.startswith("unconfirmed-identity:") for u in lineage.unresolved)


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
    assert lineage.unresolved == [], "the builtin keeps its exemption from being NAMED"


def test_a_table_valued_function_is_unclassified_not_a_demonstrated_gap():
    """`object_key` returns "" for `generate_series`/`unnest` because there is no identifier to
    key, and treating that as an unaccounted OBJECT reported INCOMPLETE for a view that reaches no
    object at all. It is reach we cannot classify, which is UNKNOWN."""
    views = [ViewDefinition(object_id="g", definition="SELECT * FROM generate_series(1, 10)",
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT * FROM g"), ["g"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    # Labelled by the SOURCE-SHAPE whitelist now rather than by the empty-object_key branch: the
    # whitelist runs first and recognises a table-valued function as an unmodelled source, which
    # is the same judgement reached one step earlier and by the rule `views.py` already owns.
    assert any(u.startswith("unmodelled-source:") for u in lineage.unresolved)


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


def test_a_quoted_function_name_cannot_carry_a_literal_into_the_always_tier():
    """sqlglot strips identifier delimiters when populating `node.name`, so a UDF named
    `"fn(sk-live-secret)"` put that string in `name` — and the previous guard sanitised the
    COMPOSED label while falling back to that same unchecked value. Third spelling to carry a
    literal through this one field, which is why sanitising moved to a single admission point that
    fails closed rather than to another shape-specific branch."""
    lineage = lineage_for(_ast('SELECT public."fn(sk-live-secret)"() FROM claim'), ["claim"],
                          inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert not any("sk-live" in u for u in lineage.unresolved)
    assert "unnameable-function" in lineage.unresolved


def test_an_unmodelled_source_in_a_view_body_is_not_a_clean_bill():
    """`base_tables` can return FEWER tables for a shape it does not model while scope
    construction still reports success — a Postgres view over `LATERAL (VALUES ((SELECT
    max(store_id) FROM customer)))` gave COMPLETE with `customer` unaccounted. `views.py` refuses
    on unmodelled sources for exactly this reason and the whitelist is reused rather than
    re-derived."""
    views = [ViewDefinition(
        object_id="v",
        definition="SELECT * FROM LATERAL (VALUES ((SELECT max(store_id) FROM customer))) AS lv(x)",
        dialect="postgres")]
    lineage = lineage_for(_ast("SELECT x FROM v"), ["v"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert any(u.startswith("unmodelled-source:") for u in lineage.unresolved)


def test_discovery_in_progress_is_not_discovery_done():
    """`Job.status` is a free string and only `failed` marked the inventory unavailable, so a job
    recorded `running` counted as coverage and lineage came back COMPLETE over an inventory still
    being built. Absence, failure and IN-PROGRESS are three states; two were sharing the confident
    one."""
    running = Job(id="discover:views", source_id="s", kind="discover", status="running")
    inventory = inventory_for(_snapshot(jobs=[running]))
    assert inventory.asked is False
    lineage = lineage_for(_ast("SELECT id FROM claim"), ["claim"], inventory)
    assert lineage.completeness == UNKNOWN


def test_federated_coverage_is_correspondable_not_merely_counted():
    """`merge_snapshots` re-keys each job's `source_id` to its CATALOG, which is unique by
    construction and is what `registry` is keyed by — so coverage checks that the discovered set
    names the registry's sources. Counting cardinalities pinned `asked=False` forever whenever two
    catalogs shared a spec id, and documenting that limit was not the same as closing it."""
    from mnemiq.semantic.federation import FederatedSnapshot

    def _job(source):
        return Job(id="discover:views", source_id=source, kind="discover", status="done")

    registry = {"a": "schema_a", "b": "schema_b"}
    both = FederatedSnapshot(version="v", source_id="f", created_at="t",
                             jobs=[_job("a"), _job("b")], registry=registry)
    assert inventory_for(both).asked is True

    one = FederatedSnapshot(version="v", source_id="f", created_at="t",
                            jobs=[_job("a"), _job("a")], registry=registry)
    assert inventory_for(one).asked is False, "one catalog discovered twice is not two catalogs"


def test_the_cli_never_prints_a_table_list_without_its_marker():
    """The third surface, missed while the commit message said there were two. `tables: []` on a
    function-backed query reads to a human as "nothing was touched" when it means "we could not
    tell"."""
    import inspect

    from mnemiq import cli

    src = inspect.getsource(cli._cmd_ask if hasattr(cli, "_cmd_ask") else cli)
    assert "lineage_completeness" in src, "the CLI prints the list; it must print the marker"


def test_a_direct_unmodelled_source_is_not_reported_complete():
    """The whitelist was applied to view BODIES and not to the caller's own statement.
    `SELECT x FROM LATERAL (VALUES ((SELECT max(store_id) FROM customer)))` yields
    `base_tables == []` while scope resolution reports SUCCESS — so the empty list looked settled
    and the marker said COMPLETE over a read of `customer` it never named.

    Note this is also an AUTHORIZATION bypass and that half is NOT fixed here: `check_access`
    reads the same empty `base_tables`, so `decide` approves the statement with `customer` absent
    from `visible`. Pre-existing, same family as M43, filed rather than fixed — the marker's job
    is to stop the audit record from calling it accounted for.
    """
    lineage = lineage_for(sqlglot.parse_one(
        "SELECT x FROM LATERAL (VALUES ((SELECT max(store_id) FROM customer))) AS lv(x)",
        read="postgres"), [], inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert any(u.startswith("unmodelled-source:") for u in lineage.unresolved)


def test_a_label_must_match_an_identifier_not_merely_avoid_five_characters():
    """`_safe_label` rejected a character blocklist, so `"DOB 1990-01-01"()` passed: sqlglot strips
    the delimiters and the name arrives with spaces and hyphens but no bracket or quote.
    Blocklisting is the pattern `views.py` documents as unfixable — an unlisted shape passes —
    which I had quoted approvingly two commits before writing one."""
    from mnemiq.sql.lineage import _functions_in

    assert _functions_in(
        sqlglot.parse_one('SELECT "DOB 1990-01-01"() FROM claim', read="postgres")
    )[0] == ["unnameable-function"]
    # ...and an ordinary name still survives, or the guard would be a blanket.
    assert _functions_in(
        sqlglot.parse_one("SELECT all_ssns() FROM claim", read="duckdb"))[0] == ["all_ssns"]


def test_the_cli_json_surface_carries_lineage_too():
    """The fourth surface. A machine consumer reading `--json` got answer/deferred/mode/sql and no
    audit artifact at all — worse than a bare list, since there was neither marker nor tables."""
    import inspect

    from mnemiq import cli

    src = inspect.getsource(cli._cmd_ask)
    json_block = src[src.index("if args.json"):src.index("print(ans.answer)")]
    assert '"lineage"' in json_block and '"tables_used"' in json_block


def test_a_bare_name_in_a_view_body_is_not_the_callers_bare_name():
    """The class four review passes each found one shape of: lexical equality is not object
    identity across independently bound scopes.

    A view `a.v` whose body says `claim` reads `a.claim`; a caller under schema `b` writing bare
    `claim` reads `b.claim`. The strings match and the objects do not, and `ViewDefinition`
    carries no creation schema, so the binding context is not recoverable here. Measured before
    the narrowing: COMPLETE, with the read of `a.claim` unrecorded — which is the one outcome
    worse than shipping no marker, because it affirmatively certifies a false audit record.
    """
    views = [ViewDefinition(object_id="a.v", definition="SELECT * FROM claim", dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT v.id FROM a.v v JOIN claim c ON c.id = v.id"),
                          ["a.v", "claim"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert any(u.startswith("unconfirmed-identity:") for u in lineage.unresolved)


def test_an_unmodelled_source_does_not_hide_a_demonstrated_gap():
    """The uncertainty is recorded WITHOUT short-circuiting the body scan. The `continue` meant a
    view mixing a modelled source with an unmodelled one downgraded a gap the engine could point
    at to one it merely suspected — inverting this module's own rule that demonstrated outranks
    unclear."""
    views = [ViewDefinition(
        object_id="v",
        definition="SELECT customer.id FROM customer CROSS JOIN LATERAL (VALUES (1)) AS l(x)",
        dialect="postgres")]
    lineage = lineage_for(_ast("SELECT id FROM v"), ["v"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == INCOMPLETE, "customer is absent from the list and demonstrable"
    assert any(u.startswith("unmodelled-source:") for u in lineage.unresolved), "still recorded"


def test_every_label_goes_through_one_admission_point():
    """`_safe_label` was called "the single admission point" and then bypassed three times in the
    same function — a raw `object_key` interpolated into a prefix, a raw view name on parse
    failure, and a raw key for a demonstrated gap. A view named `"DOB 1990-01-01"` therefore put
    that string into the ALWAYS tier: the fourth disclosure of one shape, through a path created
    while closing the third. Every append now routes through `_add`, which keeps an engine prefix
    and grammar-checks only the caller-derived subject."""
    views = [ViewDefinition(object_id="v", definition='SELECT id FROM "DOB 1990-01-01"',
                            dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT id FROM v JOIN claim USING (id)"),
                          ["v", "dob 1990-01-01"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    for label in lineage.unresolved:
        subject = label.partition(":")[2] or label
        assert "1990" not in subject and " " not in subject, f"raw object name admitted: {label}"


def test_complete_still_happens_for_the_ordinary_case():
    """The narrowing must not make the marker degenerate. A statement reading no view, with a
    discovered inventory and no unclassifiable call, is still COMPLETE — otherwise UNKNOWN would
    mean nothing and the artifact would be a constant."""
    lineage = lineage_for(_ast("SELECT id, amount FROM claim WHERE region = 'west'"),
                          ["claim"], inventory_for(_snapshot(jobs=[_DISCOVERED])))
    assert lineage.completeness == COMPLETE


def test_one_leaky_view_does_not_certify_every_later_view_as_a_gap():
    """A statement-scoped gap accumulator meant that once ANY view reached past the table list,
    every later view was named as reaching too — order-dependent, so it would not have reproduced
    reliably, and the inverse of the false certification this design exists to remove: a clean
    view named as a demonstrated gap purely by its position in the list.

    Both orderings asserted, because the defect was invisible in one of them."""
    views = [ViewDefinition(object_id="leaky_view", definition="SELECT id FROM secret_table",
                            dialect="duckdb"),
             ViewDefinition(object_id="clean_view", definition="SELECT id FROM claim",
                            dialect="duckdb")]
    sql = ("SELECT b.id FROM clean_view b JOIN leaky_view a ON a.id = b.id "
           "JOIN claim c ON c.id = b.id")
    for order in (["clean_view", "leaky_view", "claim"], ["leaky_view", "clean_view", "claim"]):
        lineage = lineage_for(_ast(sql), order,
                              inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
        assert "leaky_view" in lineage.unresolved, order
        assert "clean_view" not in lineage.unresolved, f"certified by position: {order}"


def test_a_name_containing_a_colon_cannot_pose_as_a_classification():
    """`_add` split on the first colon and admitted whatever preceded it, so a caller-derived name
    containing one rode in intact. The earlier test could not catch this: it partitioned on ":"
    exactly as the code did and asserted only on the part after, which is the half a colon-bearing
    name never lands in — a test that shares the code's assumption cannot falsify it."""
    from mnemiq.sql.lineage import _add

    for hostile in ("DOB 1990-01-01:x", "evil name: with spaces", "sk-live-abc:123"):
        out: list[str] = []
        _add(out, hostile)
        assert out == ["unnameable-function"], f"admitted {out!r} for {hostile!r}"

    # ...and a real engine classification still passes through with its subject intact.
    out = []
    _add(out, "unconfirmed-identity:claim")
    assert out == ["unconfirmed-identity:claim"]


def test_a_view_that_reaches_no_object_at_all_is_still_complete():
    """The narrowing must not become "any view means UNKNOWN". A body reaching no object has no
    identity to confirm, so there is nothing the marker could be uncertain about.

    Pinned because the invariant was first written as "COMPLETE survives only where no view is
    read", which this case falsifies — the behaviour was right and the sentence was wrong, and an
    over-broad invariant is how a later edit narrows something that did not need it.
    """
    views = [ViewDefinition(object_id="const_view", definition="SELECT 1 AS x", dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT x FROM const_view"), ["const_view"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == COMPLETE
    assert lineage.unresolved == [] and lineage.reasons == []


def test_a_view_body_that_calls_a_function_also_loses_complete():
    """The any-call downgrade was wired at ONE of its two call sites: the view-body loop took
    `_functions_in(parsed)[0]` and dropped the flag, so a view defined `SELECT log(1) AS x` still
    yielded COMPLETE. The class declared closed, surviving one level down — the same shape as
    applying the source whitelist to bodies and not to the statement, which a previous commit
    fixed and whose message I then used against its predecessor. Inverted, two commits later."""
    views = [ViewDefinition(object_id="v", definition="SELECT log(1) AS x", dialect="duckdb")]
    lineage = lineage_for(_ast("SELECT x FROM v"), ["v"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN
    assert "unconfirmed-function-identity" in lineage.reasons


def test_a_quoted_callable_is_tokened_even_when_its_name_is_identifier_shaped():
    """The quoted branch had NO test that could fail. Both existing quoted-name cases carry names
    that fail `_IDENTIFIER` on their own, so both still passed with the branch deleted — coverage
    that proves the grammar, not the provenance. This payload passes the grammar, so only reading
    the quoting can stop it, on a default-on disclosure surface."""
    from mnemiq.sql.lineage import _functions_in

    names, saw_call = _functions_in(
        _ast('SELECT "sk_live_abc123"() FROM claim'))
    assert names == ["unnameable-function"], "an identifier-shaped quoted payload was admitted"
    assert saw_call is True


def test_an_unquoted_function_name_is_still_named_and_that_is_the_feature():
    """Recorded so the ledger is honest. `sk_live_abc123()` unquoted is reported verbatim, and no
    guard can change that: naming a genuinely unresolved callable is the artifact's purpose, and a
    function's name is an object id. The quoted branch closes a SPELLING, not the shape — a secret
    chosen as an unquoted identifier is indistinguishable from a real function name, and treating
    it as sensitive would mean naming nothing."""
    from mnemiq.sql.lineage import _functions_in

    assert _functions_in(_ast("SELECT sk_live_abc123() FROM claim"))[0] == ["sk_live_abc123"]


@pytest.mark.parametrize("seed", ["1", "2", "3", "4", "5"])
def test_case_distinct_views_never_certify_by_hash_order(seed):
    """`spellings` returns a SET, so `next(...)` chose a view by hash iteration order. With
    distinct Postgres views `"V"` and `v`, a query bound to `"V"` — whose body reads `secret` —
    returned COMPLETE on three of five PYTHONHASHSEED values by inspecting `v`'s constant body.

    Non-deterministic false certification is the worst variant of this class: a test can pass and
    the deployment still be wrong. Several spellings reaching the SAME view is fine; several
    reaching different ones is exactly where identity would have to be resolved and cannot be, so
    the lookup refuses to choose.

    Parametrized over seeds because a single run cannot observe the defect — the subprocess is the
    only way to vary hash ordering, and asserting once would have been an assertion about luck.
    """
    import json
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent('''
        import sqlglot
        from mnemiq.contract.semantic import Job, Snapshot, ViewDefinition
        from mnemiq.sql.lineage import lineage_for
        from mnemiq.sql.views import inventory_for
        import json
        D = Job(id="discover:views", source_id="s", kind="discover", status="done")
        # `dialect="postgres"`: the scenario is Postgres case-sensitivity, and DuckDB folds these
        # two spellings to one object -- as this repo's own source comment records -- so declaring
        # duckdb would describe a source shape that cannot exist in the dialect it names.
        views = [ViewDefinition(object_id="V", definition="SELECT id FROM secret",
                                dialect="postgres"),
                 ViewDefinition(object_id="v", definition="SELECT 1 AS id", dialect="postgres")]
        snap = Snapshot(version="v1", source_id="s", created_at="t", views=views, jobs=[D])
        lin = lineage_for(sqlglot.parse_one(\'SELECT id FROM "V"\', read="postgres"), ["V"],
                          inventory_for(snap))
        print(json.dumps({"c": lin.completeness, "u": lin.unresolved}))
    ''')
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                         timeout=60, cwd=root,
                         env={"PYTHONHASHSEED": seed, "PYTHONPATH": str(root / "src"),
                              "PATH": "/usr/bin:/bin"})
    # The child's failure must be legible. Without this a child that cannot import surfaces as an
    # IndexError from `splitlines()[-1]` with the real traceback discarded, and one that hangs
    # hangs the suite with no diagnostic at all.
    assert out.returncode == 0, f"child failed at seed {seed}:\n{out.stderr}"
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["c"] != COMPLETE, f"certified by hash order at seed {seed}"
    assert any(u.startswith("ambiguous-view:") for u in result["u"])


def test_an_ambiguous_view_still_reports_a_gap_every_candidate_demonstrates():
    """Refusing to choose must not hide what every choice would have shown. The `continue` on
    ambiguity skipped the body scan, so when EVERY candidate body reached past the caller's list
    the gap was recorded as UNKNOWN — "unclear" where the engine could name the view.

    That inverts this module's own ordering rule, and it is the third time an early `continue` in
    this file has turned a demonstrated gap into an unclear one. The ambiguity is about WHICH body
    was read, not about whether the read is accounted for.
    """
    both_reach = [ViewDefinition(object_id="V", definition="SELECT id FROM secret",
                                 dialect="postgres"),
                  ViewDefinition(object_id="v", definition="SELECT id FROM other_secret",
                                 dialect="postgres")]
    lineage = lineage_for(_ast('SELECT id FROM "V"'), ["V"],
                          inventory_for(_snapshot(views=both_reach, jobs=[_DISCOVERED])))
    assert lineage.completeness == INCOMPLETE
    assert "V" in lineage.unresolved
    assert any(u.startswith("ambiguous-view:") for u in lineage.unresolved), "still recorded"

    # ...and when the candidates DISAGREE, the ambiguity is the whole answer.
    one_reaches = [ViewDefinition(object_id="V", definition="SELECT id FROM secret",
                                  dialect="postgres"),
                   ViewDefinition(object_id="v", definition="SELECT 1 AS id", dialect="postgres")]
    mixed = lineage_for(_ast('SELECT id FROM "V"'), ["V"],
                        inventory_for(_snapshot(views=one_reaches, jobs=[_DISCOVERED])))
    assert mixed.completeness == UNKNOWN


def test_an_ambiguous_pair_with_an_unparseable_body_is_not_a_demonstrated_gap():
    """The `b is not None` half of the ambiguous guard had no test: rewriting the predicate as
    `b is None or _reaches_past(...)` — which would name a view as a demonstrated gap off a body
    that never parsed — passed the whole suite. A body we cannot read demonstrates nothing."""
    views = [ViewDefinition(object_id="V", definition="SELECT id FROM secret", dialect="postgres"),
             ViewDefinition(object_id="v", definition="NOT SQL AT ALL {{{", dialect="postgres")]
    lineage = lineage_for(_ast('SELECT id FROM "V"'), ["V"],
                          inventory_for(_snapshot(views=views, jobs=[_DISCOVERED])))
    assert lineage.completeness == UNKNOWN, "an unreadable body cannot demonstrate a gap"
    assert "V" not in lineage.unresolved


# --------------------------------------------------------------------------------------------
# Issue #5 -- the marker fired on `count(*)`, so it appeared on nearly every real answer
# --------------------------------------------------------------------------------------------


def _reasons(sql, *, inventory, dialect="duckdb"):
    import sqlglot

    from mnemiq.sql.lineage import lineage_for
    from mnemiq.sql.views import ViewInventory

    return list(lineage_for(sqlglot.parse_one(sql, read=dialect), ["customer"], ViewInventory(),
                            functions=inventory).reasons)


def test_an_aggregate_no_longer_downgrades_lineage():
    """Issue #5: every answer carried `unconfirmed-function-identity`, `count(*)` included.

    A source that defines no functions of its own cannot have a UDF in the query, whatever the
    call is spelled, so the calls clear.
    """
    from mnemiq.sql.functions import FunctionInventory

    defines_nothing = FunctionInventory.of([], covers_view_bodies=True)
    for sql in ("SELECT country, count(*) AS n FROM customer GROUP BY country",
                "SELECT date_trunc('month', created_at) FROM customer",
                "SELECT CASE WHEN id > 1 THEN 'a' ELSE 'b' END FROM customer"):
        assert "unconfirmed-function-identity" not in _reasons(sql, inventory=defines_nothing)


def test_a_source_that_defines_anything_downgrades_every_call():
    """Deliberately coarse, and the reason is three leaks.

    Naming each call and matching it against the catalogue failed three different ways: the
    written name is not what executes, the rendered name is not what DuckDB binds (`count(*)`
    becomes `count_star`), and a view body binds as written. Each version certified a macro that
    read another table. The engine cannot say WHICH call is a UDF, so if the source defines any
    function it declines for all of them.
    """
    from mnemiq.sql.functions import FunctionInventory

    defines_one = FunctionInventory.of(["helper"], covers_view_bodies=True)
    assert "unconfirmed-function-identity" in _reasons(
        "SELECT count(*) AS n FROM customer", inventory=defines_one)


def test_without_an_inventory_nothing_changes():
    """The regression guard. The old behaviour is this function's behaviour with no inventory --
    the change is what asking BUYS, not a relaxed default."""
    from mnemiq.sql.functions import FunctionInventory

    for blind in (FunctionInventory.never_asked(), FunctionInventory.unavailable("denied")):
        assert "unconfirmed-function-identity" in _reasons(
            "SELECT country, count(*) AS n FROM customer GROUP BY country", inventory=blind)


def test_a_statement_with_no_call_at_all_never_needed_an_inventory():
    """The floor: nothing to confirm means nothing to withhold, whatever the source defines."""
    from mnemiq.sql.functions import FunctionInventory

    assert "unconfirmed-function-identity" not in _reasons(
        "SELECT id FROM customer", inventory=FunctionInventory.never_asked())


def test_the_trace_says_why_it_could_not_confirm():
    """Four causes used to write one marker, so a source whose `user_functions` fails on every
    request read exactly like one that defines a helper -- the issue #5 fix quietly not
    applying, with nothing in the audit record saying so.

    These reach `Trace.lineage_reasons` and the HTTP and MCP payloads, so swapping or dropping
    one changes what an auditor is told.
    """
    import sqlglot

    from mnemiq.contract.semantic import ViewDefinition
    from mnemiq.sql.functions import FunctionInventory
    from mnemiq.sql.lineage import lineage_for
    from mnemiq.sql.views import ViewInventory

    sql = "SELECT count(*) AS n FROM customer"
    for code, inventory in {
        "function-inventory-unavailable": FunctionInventory.unavailable("denied"),
        "function-inventory-never-asked": FunctionInventory.never_asked(),
    }.items():
        reasons = _reasons(sql, inventory=inventory)
        assert "unconfirmed-function-identity" in reasons
        assert code in reasons, f"{code} missing from {reasons}"

    # A source that simply defines a helper gets the bare marker: nothing failed, nothing was
    # unasked, and the licence is not the reason.
    # `covers_view_bodies=False` on purpose. With True the fourth code's `not inventory.names`
    # half is never consulted, so dropping it would leave the suite green -- and an attachment
    # that defines a helper AND cannot cover bodies would then blame view bodies for the helper.
    plain = _reasons(sql, inventory=FunctionInventory.of(["helper"], covers_view_bodies=False))
    assert "unconfirmed-function-identity" in plain
    assert not [r for r in plain if r.startswith("function-inventory-")]

    # The fourth cause needs an actual view body, since that is the only place the licence is
    # consulted. A Postgres source with NO functions of its own still cannot certify a body,
    # and without this code that trace reads as "this source defines a function".
    views = ViewInventory({"v": ViewDefinition(
        object_id="v", definition="SELECT count(*) AS n FROM customer", dialect="duckdb")})
    body = lineage_for(sqlglot.parse_one("SELECT n FROM v", read="duckdb"), ["v"], views,
                       functions=FunctionInventory.of([], covers_view_bodies=False))
    assert "unconfirmed-function-identity" in body.reasons
    assert "function-inventory-covers-no-view-bodies" in body.reasons
