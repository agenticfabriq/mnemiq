"""`mnemiq enrich --add`: a run over a subset of tables merges into the source's latest snapshot instead of replacing it.

Without the merge, enriching a 600-table schema in batches loses every batch but the last; and the joins between
batches, which no single run sees, are re-derived over the union (`union_relationships`)."""
import pytest

from mnemiq.contract import Column, Snapshot, SourceBinding
from mnemiq.contract.semantic import Example, Job, Quality, TableFacts, ViewDefinition
from mnemiq.enrichment.accumulate import MERGE_FIELDS, merge_run
from mnemiq.sql.fanout_check import PARTITIONS_JOB


def _snap(tables: dict[str, str], *, jobs=(), facts=(), examples=(), partitions=None, version="v",
          definitions=()) -> Snapshot:
    """One binding, one column `id` described as `tables[t]`, and a done profile job per table."""
    all_jobs = [Job(id=f"profile:{t}", source_id="src", kind="profile", status="done") for t in tables]
    all_jobs += list(jobs)
    if partitions is not None:
        all_jobs.append(Job(id=PARTITIONS_JOB, source_id="src", kind="profile:partitions", status=partitions[0],
                            detail=partitions[1]))
    return Snapshot(
        version=version, source_id="src", created_at="2026-11-23T00:00:00Z",
        source_bindings=[SourceBinding(id=f"sb:{t}", source_id="src", object_id=t, source_object=t,
                                       binding_type="table") for t in tables],
        columns=[Column(id=f"{t}.id", object_id=t, name="id", description=d) for t, d in tables.items()],
        jobs=all_jobs, table_facts=list(facts), examples=list(examples), definitions=list(definitions),
    )


def _described(snap: Snapshot, table: str) -> str | None:
    return next(c.description for c in snap.columns if c.object_id == table)


def test_every_snapshot_field_has_a_merge_rule():
    # A field added to Snapshot later must be given a rule here, or --add would silently keep or drop it.
    assert set(MERGE_FIELDS) == set(Snapshot.model_fields)


def test_kept_tables_keep_their_descriptions():
    merged, kept_failed = merge_run(_snap({"a": "kept-a"}), _snap({"b": "run-b"}))
    assert {b.object_id for b in merged.source_bindings} == {"a", "b"}
    assert _described(merged, "a") == "kept-a"
    assert _described(merged, "b") == "run-b"
    assert kept_failed == []


def test_re_adding_a_table_replaces_every_per_table_entry():
    kept = _snap({"a": "old-a", "c": "kept-c"},
                 facts=[TableFacts(object_id="a", grain="old grain"), TableFacts(object_id="c", grain="c grain")],
                 examples=[Example(question="old a?", sql="SELECT 1 FROM a", tables=["a"], object_id="a"),
                           Example(question="c?", sql="SELECT 1 FROM c", tables=["c"], object_id="c")],
                 jobs=[Job(id="profile:a.id", source_id="src", kind="profile:column", status="failed")])
    kept = kept.model_copy(update={
        "views": [ViewDefinition(object_id="a", definition="old", dialect="duckdb")],
        "qualities": [Quality(id="q:a", object_id="a", name="n", status="ok", threshold="1", observed_value="1")]})
    run = _snap({"a": "new-a"}, facts=[TableFacts(object_id="a", grain="new grain")],
                examples=[Example(question="new a?", sql="SELECT 2 FROM a", tables=["a"], object_id="a")])
    merged, _ = merge_run(kept, run)

    assert [c.description for c in merged.columns if c.object_id == "a"] == ["new-a"]
    assert [f.grain for f in merged.table_facts if f.object_id == "a"] == ["new grain"]
    assert sorted(e.question for e in merged.examples) == ["c?", "new a?"]
    assert [j.status for j in merged.jobs if j.id == "profile:a"] == ["done"]
    assert not [j for j in merged.jobs if j.id == "profile:a.id"], "the kept column job of a re-added table went"
    assert not merged.views and not merged.qualities, "a re-added table takes the run's views and qualities (none)"
    assert _described(merged, "c") == "kept-c"


def test_an_example_reading_a_re_added_table_is_dropped_even_when_it_belongs_to_another():
    kept = _snap({"a": "a", "c": "c"},
                 examples=[Example(question="a and c?", sql="SELECT 1 FROM c JOIN a", tables=["c", "a"],
                                   object_id="c")])
    merged, _ = merge_run(kept, _snap({"a": "a2"}))
    assert merged.examples == []


def test_a_re_added_table_whose_profile_failed_keeps_its_earlier_enrichment():
    kept = _snap({"a": "kept-a"})
    run = _snap({}, jobs=[Job(id="profile:a", source_id="src", kind="profile", status="failed", detail="boom")])
    merged, kept_failed = merge_run(kept, run)

    assert _described(merged, "a") == "kept-a"
    assert [j.status for j in merged.jobs if j.id == "profile:a"] == ["done"]
    assert kept_failed == ["a"]


def test_a_new_table_whose_profile_failed_keeps_the_runs_failed_job():
    run = _snap({}, jobs=[Job(id="profile:b", source_id="src", kind="profile", status="failed", detail="boom")])
    merged, kept_failed = merge_run(_snap({"a": "kept-a"}), run)
    assert [j.status for j in merged.jobs if j.id == "profile:b"] == ["failed"]
    assert kept_failed == []


@pytest.mark.parametrize(("kept", "run", "expected"), [
    (("done", None), ("estimated", None), "estimated"),
    (("estimated", None), ("done", None), "estimated"),
    (("done", None), ("done", None), "done"),
    (("partial", "c.flag: boom"), ("done", None), "partial"),
    (("done", None), ("partial", "b.flag: boom"), "partial"),
])
def test_the_partitions_job_reports_the_weaker_of_the_two_runs(kept, run, expected):
    merged, _ = merge_run(_snap({"c": "c"}, partitions=kept), _snap({"b": "b"}, partitions=run))
    (job,) = [j for j in merged.jobs if j.id == PARTITIONS_JOB]
    assert job.status == expected


def test_a_kept_partition_failure_goes_once_its_table_is_profiled_again():
    # The warning names the tables whose current-row joins the guard will refuse; a table this run profiled again
    # no longer belongs in it.
    kept = _snap({"a": "a", "c": "c"}, partitions=("partial", "a.flag: boom; c.flag: bang"))
    merged, _ = merge_run(kept, _snap({"a": "a2"}, partitions=("done", None)))
    (job,) = [j for j in merged.jobs if j.id == PARTITIONS_JOB]
    assert (job.status, job.detail) == ("partial", "c.flag: bang")

    merged, _ = merge_run(_snap({"a": "a"}, partitions=("partial", "a.flag: boom")),
                                _snap({"a": "a2"}, partitions=("done", None)))
    (job,) = [j for j in merged.jobs if j.id == PARTITIONS_JOB]
    assert (job.status, job.detail) == ("done", None)


def test_the_runs_global_entries_win():
    from mnemiq.contract import Definition

    kept = _snap({"a": "a"}, definitions=[Definition(id="d:old", term="old", domain="x", definition="old")])
    run = _snap({"b": "b"}, definitions=[Definition(id="d:new", term="new", domain="x", definition="new")])
    merged, _ = merge_run(kept, run)
    assert [d.id for d in merged.definitions] == ["d:new"]
    assert merged.created_at == run.created_at


# Joins across subsets. No single run sees both ends of a join between subsets: the run's adapter is scoped to its own
# tables, and `TableScopedAdapter.foreign_keys` keeps only keys with both ends in scope. `union_relationships` reads
# them again through an adapter scoped to the union.
import fnmatch  # noqa: E402
import sqlite3  # noqa: E402

from mnemiq.adapters.scoped import TableScopedAdapter  # noqa: E402
from mnemiq.adapters.sqlite import SQLiteAdapter  # noqa: E402
from mnemiq.config import SourceSpec  # noqa: E402
from mnemiq.enrichment.accumulate import estimated_tables, union_relationships, union_spec  # noqa: E402
from mnemiq.enrichment.pipeline import enrich_structural  # noqa: E402
from tests.test_statistics_enrich import SHOP, _Recording, _touches  # noqa: E402


@pytest.fixture
def shop_path(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(SHOP)
    con.commit()
    con.close()
    return str(path)


def _spec(path: str, tables: tuple[str, ...]) -> SourceSpec:
    return SourceSpec(id="shop", kind="sqlite", target=path, catalog="shop", schema="main", tables=tables)


def _run(inner, tables: tuple[str, ...], statistics=None) -> Snapshot:
    scoped = TableScopedAdapter(inner, tables, "shop")
    return enrich_structural(scoped, "shop") if statistics is None else enrich_structural(scoped, "shop",
                                                                                         statistics=statistics)


def _accumulate(inner, path, *runs):
    """Each run over its own subset, merged in order, then joins over the union."""
    merged = runs[0]
    for run in runs[1:]:
        merged, _ = merge_run(merged, run)
    union = TableScopedAdapter(inner, union_spec(_spec(path, ()), merged).tables, "shop")
    return union_relationships(union, merged, harvest_values=True)


@pytest.mark.parametrize("order", [("orders", "customers"), ("customers", "orders")])
def test_a_declared_key_crossing_subsets_survives_in_both_directions(shop_path, order):
    inner = SQLiteAdapter(shop_path)
    runs = [_run(inner, (table,)) for table in order]
    assert not any(r.to == "customers" for run in runs for r in run.relationships), "no single run sees the key"
    relationships, job = _accumulate(inner, shop_path, *runs)
    assert ("orders", "customers") in [(r.from_, r.to) for r in relationships]
    assert (job.id, job.status) == ("infer:relationships", "done")


def test_an_inferred_join_crossing_subsets_is_found_in_exact_mode(shop_path):
    inner = SQLiteAdapter(shop_path)
    relationships, _ = _accumulate(inner, shop_path, _run(inner, ("product",)), _run(inner, ("order_lines",)))
    assert any(r.from_ == "order_lines" and r.to == "product" for r in relationships)


def test_statistics_mode_infers_between_exact_tables_of_different_runs_and_never_to_an_estimated_one(shop_path):
    inner = _Recording(SQLiteAdapter(shop_path))
    runs = [_run(inner, ("product",), statistics=("product",)),
            _run(inner, ("order_lines",), statistics=("order_lines",)),
            _run(inner, ("orders",), statistics=())]
    relationships, _ = _accumulate(inner, shop_path, *runs)
    pairs = [(r.from_, r.to) for r in relationships]
    assert ("order_lines", "product") in pairs, "both ends profiled exactly, each in its own run"
    assert ("order_lines", "orders") not in pairs, "orders is estimated: a candidate would read its rows"


def test_a_kept_table_keeps_its_own_runs_profile_mode(shop_path):
    inner = _Recording(SQLiteAdapter(shop_path))
    exact_customers = _run(inner, ("customers",))
    estimated_orders = _run(inner, ("orders",), statistics=())
    merged, _ = merge_run(exact_customers, estimated_orders)
    assert estimated_tables(merged) == frozenset({"orders"})
    merged, _ = merge_run(estimated_orders, exact_customers)
    assert estimated_tables(merged) == frozenset({"orders"})


def test_union_spec_escapes_names_that_look_like_patterns():
    snap = _snap({"a[1]": "x", "b": "y"})
    (pattern_a, pattern_b) = union_spec(_spec("unused", ("a*",)), snap).tables
    assert fnmatch.fnmatchcase("a[1]", pattern_a) and not fnmatch.fnmatchcase("a1", pattern_a)
    assert pattern_b == "b"


def test_the_union_pass_reads_no_estimated_tables_rows_and_infers_nothing_to_one(shop_path):
    # The statistics-mode guarantee holds across runs: product is estimated in its own run, so the join to it that an
    # exact profile would infer (test_an_inferred_join_crossing_subsets_is_found_in_exact_mode) is not looked for.
    inner = _Recording(SQLiteAdapter(shop_path))
    runs = [_run(inner, ("product",), statistics=()), _run(inner, ("order_lines",), statistics=("order_lines",))]
    inner.sql.clear()
    relationships, _ = _accumulate(inner, shop_path, *runs)
    assert not any(r.to == "product" for r in relationships)
    assert not [sql for sql in inner.sql if _touches(sql, "product")], "an estimated table's rows were read"


def test_joins_within_one_subset_and_across_subsets_are_all_kept(shop_path):
    # One subset already holds a join (order_lines to product); the key from orders to customers crosses two others.
    # A merge that kept the interim joins instead of reading them again would lose the crossing one.
    inner = SQLiteAdapter(shop_path)
    first = _run(inner, ("product", "order_lines"))
    assert any(r.from_ == "order_lines" and r.to == "product" for r in first.relationships)
    relationships, _ = _accumulate(inner, shop_path, first, _run(inner, ("orders",)), _run(inner, ("customers",)))
    pairs = [(r.from_, r.to) for r in relationships]
    assert ("order_lines", "product") in pairs and ("orders", "customers") in pairs


# Views. `discover:views` covers a source: done only once the source answered for every view in scope, and the view
# floor relies on it (`inventory_for`, `check_views`). A run discovers views within its own subset, so its `done` cannot
# vouch for an earlier run whose discovery failed: taken as the merged status, it let a row-filtered view through
# (Codex's adversarial review of --add).
import sqlglot  # noqa: E402

from mnemiq.contract.semantic import Job as _Job  # noqa: E402
from mnemiq.enrichment.accumulate import union_views, views_spec  # noqa: E402
from mnemiq.sql.verdict import RefusalCode  # noqa: E402
from mnemiq.sql.views import check_views, inventory_for  # noqa: E402

CLAIM_V_SQL = sqlglot.parse_one("SELECT id, amount FROM claim_v", read="duckdb")


def _discovery(status: str) -> _Job:
    return _Job(id="discover:views", source_id="src", kind="discover", status=status)


@pytest.fixture
def claims_path(tmp_path):
    path = tmp_path / "claims.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE claim (id INTEGER PRIMARY KEY, region TEXT, amount REAL);
        CREATE TABLE other (x INTEGER);
        CREATE VIEW claim_v AS SELECT id, region, amount FROM claim;
        INSERT INTO claim VALUES (1, 'west', 10.0), (2, 'east', 20.0);
        INSERT INTO other VALUES (1);
    """)
    con.commit()
    con.close()
    return str(path)


def test_a_successful_subset_cannot_clear_an_earlier_view_discovery_failure():
    kept = _snap({"claim": "c", "claim_v": "v"}, jobs=[_discovery("failed")])
    merged, _ = merge_run(kept, _snap({"other": "o"}, jobs=[_discovery("done")]))
    refusal = check_views(CLAIM_V_SQL, inventory_for(merged), {"claim"})
    assert refusal is not None and refusal.code is RefusalCode.VIEW_INVENTORY_UNAVAILABLE


def test_a_kept_view_not_bound_as_a_table_stays_in_the_interim_inventory():
    kept = _snap({"claim": "c"}, jobs=[_discovery("done")]).model_copy(update={"views": [
        ViewDefinition(object_id="claim_v", definition="SELECT id, region, amount FROM claim", dialect="duckdb")]})
    merged, _ = merge_run(kept, _snap({"other": "o"}, jobs=[_discovery("done")]))
    refusal = check_views(CLAIM_V_SQL, inventory_for(merged), {"claim"})
    assert refusal is not None and refusal.code is RefusalCode.UNGOVERNED_VIEW


def test_views_are_discovered_again_over_the_union(claims_path):
    inner = SQLiteAdapter(claims_path)
    first = _run(inner, ("claim*",))
    assert "claim_v" in {v.object_id for v in first.views}
    merged, _ = merge_run(first, _run(inner, ("other",)))
    scoped = TableScopedAdapter(inner, views_spec(_spec(claims_path, ("other",)), first, merged).tables, "shop")
    views, job = union_views(scoped, first, merged)
    assert "claim_v" in {v.object_id for v in views} and (job.id, job.status) == ("discover:views", "done")
    final = merged.model_copy(update={"views": views, "jobs": [*(j for j in merged.jobs if j.id != job.id), job]})
    refusal = check_views(CLAIM_V_SQL, inventory_for(final), {"claim"})
    assert refusal is not None and refusal.code is RefusalCode.UNGOVERNED_VIEW, "the view floor still fires"


def test_union_views_stays_failed_while_an_earlier_discovery_never_completed(claims_path):
    # The earlier run's scope is not recorded, so a fresh discovery over the union cannot know it covered those views.
    kept = _snap({"claim": "c"}, jobs=[_discovery("failed")])
    merged, _ = merge_run(kept, _snap({"other": "o"}, jobs=[_discovery("done")]))
    _views, job = union_views(SQLiteAdapter(claims_path), kept, merged)
    assert job.status == "failed" and "without --add" in (job.detail or "")


def test_a_kept_tables_facts_job_stays_and_a_re_added_tables_is_replaced():
    facts = [_Job(id="facts:a", source_id="src", kind="facts", status="failed"),
             _Job(id="facts:c", source_id="src", kind="facts", status="done")]
    run = _snap({"a": "a2"}, jobs=[_Job(id="facts:a", source_id="src", kind="facts", status="done")])
    merged, _ = merge_run(_snap({"a": "a", "c": "c"}, jobs=facts), run)
    assert sorted((j.id, j.status) for j in merged.jobs if j.kind == "facts") == [("facts:a", "done"),
                                                                                    ("facts:c", "done")]
