"""M176: an Oracle view reading a filtered table under a spelling Oracle folds onto it is refused.

#148 (M154) made the known-object check fold a view body's names as the view's engine folds, so
on Oracle `orderſ` (long s) is recognised as ORDERS. The restricting comparison, the body's
mentions against the filtered set, still folded with `str.lower`, which keeps ſ apart from s, so a
view whose body read `orderſ` was approved where `orders` was refused, and read the filtered
table unfiltered. Both sides of the restricting comparison now fold the engine's way as well as
the wide one: a match refuses, so the union is never narrower than either. Measured on main
`2b28f1eb` through `decide` as oracle.
"""
from __future__ import annotations

import pytest
import sqlglot

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, ApprovedWrite, Refusal, RefusalCode
from mnemiq.sql.views import ViewDefinition

SCHEMA = {"orders": {"id", "store_id"}, "client": {"id"}, "ΚΟΣΜΟΣ": {"id"}, "stores": {"id"}, "v": {"id"}}


def _read_view(body: str, filtered: str, dialect: str = "oracle"):
    policy = AccessPolicy(row_filters={filtered: "1 = 1"}, policy_schema=SCHEMA)
    views = {"v": ViewDefinition(object_id="v", dialect=dialect, definition=body)}
    return decide("SELECT id FROM v", {"v": {"id"}}, dialect=dialect, target=dialect, policy=policy, views=views)


@pytest.mark.parametrize("filtered", ["orders", "ORDERS"])
@pytest.mark.parametrize("body", [
    "SELECT id FROM orderſ",
    "SELECT id FROM ORDERſ",
    "SELECT id FROM app.orderſ",
    "SELECT id FROM (SELECT id FROM orderſ) x",
    "SELECT id FROM stores WHERE id IN (SELECT id FROM orderſ)",
])
def test_a_long_s_spelling_of_a_filtered_table_is_refused_as_the_plain_one_is(body, filtered):
    verdict = _read_view(body, filtered)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, (body, verdict)
    plain = _read_view(body.replace("ſ", "s").replace("ORDERs", "ORDERS"), filtered)
    assert isinstance(plain, Refusal) and plain.code is RefusalCode.UNGOVERNED_VIEW, plain  # the control


@pytest.mark.parametrize(("body", "filtered"), [
    ("SELECT id FROM clıent", "client"),  # dotless i: Oracle folds it to I
    ("SELECT id FROM κοσμος", "ΚΟΣΜΟΣ"),  # final sigma: Oracle folds it to Σ
])
def test_the_dotless_i_and_the_final_sigma_are_refused_too(body, filtered):
    verdict = _read_view(body, filtered)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, (body, verdict)


def test_a_long_s_spelling_of_an_unfiltered_table_is_still_read():
    # The fold adds refusals only where a filtered table is reached: `ſtores` is Oracle's STORES.
    verdict = _read_view("SELECT id FROM ſtores", "orders")
    assert isinstance(verdict, Approved), verdict


def _read_through_a_nested_view(spelled: str, dialect: str = "oracle"):
    schema = {"orders": {"id"}, "sales": {"id"}, "v": {"id"}}
    policy = AccessPolicy(row_filters={"orders": "1 = 1"}, policy_schema=schema)
    views = {"sales": ViewDefinition(object_id="sales", dialect=dialect, definition="SELECT id FROM orders"),
             "v": ViewDefinition(object_id="v", dialect=dialect, definition=f"SELECT id FROM {spelled}")}
    return decide("SELECT id FROM v", {"v": {"id"}}, dialect=dialect, target=dialect, policy=policy, views=views)


@pytest.mark.parametrize("spelled", ["ſales", "app.ſales", "SALEſ", "(SELECT id FROM ſales) x"])
def test_a_view_read_under_a_spelling_oracle_folds_onto_it_is_walked(spelled):
    # The known-object check vouched for `ſales` as SALES, and the nested lookup folded with
    # `str.lower` alone, so it found no view, walked nothing, and `v` was approved over the
    # filtered ORDERS that SALES reads (Codex's review of #192).
    verdict = _read_through_a_nested_view(spelled)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, (spelled, verdict)
    plain = _read_through_a_nested_view(spelled.replace("ſ", "s").replace("SALEs", "SALES"))
    assert isinstance(plain, Refusal) and plain.code is RefusalCode.UNGOVERNED_VIEW, plain  # the control


@pytest.mark.parametrize("body", ['SELECT id FROM app."ſALES"', "SELECT id FROM app.ſALES"])
def test_every_view_a_fold_matches_is_walked_not_the_first(body):
    # Two views a wider fold cannot tell apart: APP.SALES reads nothing filtered and sorts first.
    # Walking only the first match approved `app."ſALES"`, which Oracle reads as the filtered
    # APP.ſALES (Codex's second review of #192). The unquoted spelling is Oracle's APP.SALES, and
    # is refused too: the lookup does not resolve quoting, so it walks every candidate.
    schema = {"orders": {"id"}, "safe_table": {"id"}, "APP.ſALES": {"id"}, "APP.SALES": {"id"}, "v": {"id"}}
    views = {"APP.ſALES": ViewDefinition(object_id="APP.ſALES", dialect="oracle", definition="SELECT id FROM orders"),
             "APP.SALES": ViewDefinition(object_id="APP.SALES", dialect="oracle", definition="SELECT id FROM safe_table"),
             "v": ViewDefinition(object_id="v", dialect="oracle", definition=body)}
    policy = AccessPolicy(row_filters={"orders": "1 = 1"}, policy_schema=schema)
    verdict = decide("SELECT id FROM v", {"v": {"id"}}, dialect="oracle", target="oracle", policy=policy, views=views)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, (body, verdict)


def _over_twins(body: str, dialect: str, views: dict[str, str], visible=("v",)):
    defs = {k: ViewDefinition(object_id=k, dialect=dialect, definition=d) for k, d in views.items()}
    if body:
        defs.setdefault("v", ViewDefinition(object_id="v", dialect=dialect, definition=body))
    schema = {"orders": {"id"}, "safe_table": {"id"}, **{k: {"id"} for k in defs}}
    policy = AccessPolicy(row_filters={"orders": "1 = 1"}, policy_schema=schema)
    return decide("SELECT id FROM v", {k: {"id"} for k in visible}, dialect=dialect, target=dialect,
                  policy=policy, views=defs)


@pytest.mark.parametrize(("dialect", "body", "harmless", "filtered"), [
    ("oracle", "SELECT id FROM ſALES", "ſALES", "SALES"),  # unquoted: Oracle reads SALES
    ("postgres", "SELECT id FROM SALES", "SALES", "sales"),  # unquoted: Postgres reads sales
])
def test_an_exact_key_does_not_hide_the_view_the_engine_reads(dialect, body, harmless, filtered):
    # The body's raw spelling is a key, of a view over nothing filtered; the engine folds the same
    # spelling onto another key, a view over the filtered ORDERS. Trusting the exact key walked
    # only the harmless view and approved (Codex's third review of #192).
    verdict = _over_twins(body, dialect, {harmless: "SELECT id FROM safe_table", filtered: "SELECT id FROM orders"})
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict


def test_a_view_with_no_twin_is_still_read():
    verdict = _over_twins("SELECT id FROM safe_v", "duckdb",
                          {"safe_v": "SELECT id FROM safe_table", "sales": "SELECT id FROM orders"})
    assert isinstance(verdict, Approved), verdict


def test_a_refusal_through_a_twin_names_only_the_view_the_query_read():
    # The query reads `v`, which the caller holds; `V` folds onto it and reads the filtered table.
    # The refusal is about `v`, and the twin's name stays in the log.
    verdict = _over_twins("SELECT id FROM safe_table", "oracle", {"V": "SELECT id FROM orders"})
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict
    assert verdict.subject == "v" and "'V'" not in verdict.message, verdict


def test_a_granted_table_s_view_twin_is_refused_under_the_table_s_name():
    # Postgres keeps a quoted "V" apart from `v`; the caller holds the table `v` and not the view.
    # The lowercase fallback walks "V", the conservative reading, and the refusal names what the
    # query wrote, never "V" (Codex's fourth review of #192).
    verdict = _over_twins("", "postgres", {"V": "SELECT id FROM orders"})
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict
    assert verdict.subject == "v" and "'V'" not in verdict.message, verdict


def test_layered_twins_are_walked_once_each(monkeypatch):
    # Eight case twins a level, each reading the next level's name, which all eight twins there
    # match: walked twin by twin, 8 ** 5 bodies (Codex's fourth review of #192 measured 37,448
    # parses at 40 views). A view already walked clean is not walked again.
    import collections
    import itertools

    from mnemiq.sql import views as module

    levels = 5
    defs = {}
    for level in range(levels):
        read = f"SELECT id FROM abc{level + 1}" if level + 1 < levels else "SELECT id FROM safe_table"
        for letters in itertools.product(*("aA", "bB", "cC")):
            name = "".join(letters) + str(level)
            defs[name] = read
    layered = dict(defs)
    parsed = []
    body_of = module.body_of
    monkeypatch.setattr(module, "body_of", lambda view: parsed.append(view.object_id) or body_of(view))
    verdict = _over_twins("SELECT id FROM abc0", "postgres", defs)
    assert isinstance(verdict, Approved), verdict
    walked = collections.Counter(name for name in parsed if name in layered)
    assert set(walked) == set(layered) and max(walked.values()) == 1, walked.most_common(3)  # every twin, once


def test_a_view_walked_clean_near_the_top_still_meets_the_nesting_limit_deeper_down():
    # `v` reads `x`, which reads `y`, directly and again through a chain that brings `x` to the
    # limit's last level, where `y` falls past it. Walked clean near the top, `x` must still be
    # walked again down there and refused: a clean walk is remembered only for that depth or above.
    from mnemiq.sql.views import MAX_DEPTH

    def chain(first: int) -> dict[str, str]:
        links = {f"c{i}": f"SELECT id FROM c{i + 1}" for i in range(first, MAX_DEPTH - 2)}
        links[f"c{MAX_DEPTH - 2}"] = "SELECT id FROM x"
        return {**links, "x": "SELECT id FROM y", "y": "SELECT id FROM safe_table"}

    deep = _over_twins("SELECT id FROM x UNION ALL SELECT id FROM c1", "duckdb", chain(1))
    assert isinstance(deep, Refusal) and deep.code is RefusalCode.UNRESOLVABLE_VIEW, deep
    shallow = _over_twins("SELECT id FROM x UNION ALL SELECT id FROM c2", "duckdb", chain(2))
    assert isinstance(shallow, Approved), shallow  # the control: one level shorter fits


def test_the_query_s_own_spelling_is_folded_as_its_engine_reads_it():
    # The caller holds an object keyed `ſALES`; Oracle reads the unquoted `ſALES` the query writes
    # as SALES, a view over the filtered ORDERS. The top of the walk folded ASCII alone, found no
    # view, and approved (Codex's fifth review of #192). Refused under the query's own spelling.
    schema = {"orders": {"id"}, "SALES": {"id"}, "ſALES": {"id"}}
    policy = AccessPolicy(row_filters={"orders": "1 = 1"}, policy_schema=schema)
    views = {"SALES": ViewDefinition(object_id="SALES", dialect="oracle", definition="SELECT id FROM orders")}
    verdict = decide("SELECT id FROM ſALES", {"ſALES": {"id"}}, dialect="oracle", target="oracle",
                     policy=policy, views=views)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict
    assert verdict.subject == "ſALES" and "'SALES'" not in verdict.message, verdict


def test_the_inventory_is_folded_once_a_request_not_once_a_reference(monkeypatch):
    # Every table reference scanned and folded the whole inventory, and every view walked refolded
    # it to recognise its body's names: 5.5 s at 2,000 views and references (Codex's fifth review
    # of #192). Counted in folds, which grow with the inventory and the references, not with both.
    from mnemiq.sql import views as module
    from mnemiq.sql.identifiers import IdentifierRules

    folds = []

    def rules_for(dialect):
        real = module_rules_for(dialect)
        return IdentifierRules(lambda text: folds.append(text) or real.fold(text), real.quoting_preserves_case)

    module_rules_for = module.rules_for
    monkeypatch.setattr(module, "rules_for", rules_for)
    n = 300
    defs = {f"v{i}": "SELECT id FROM safe_table" for i in range(n)}
    query = " UNION ALL ".join(f"SELECT id FROM v{i}" for i in range(n))
    schema = {"orders": {"id"}, "safe_table": {"id"}, **{k: {"id"} for k in defs}}
    views = {k: ViewDefinition(object_id=k, dialect="oracle", definition=d) for k, d in defs.items()}
    verdict = module.check_views(sqlglot.parse_one(query, read="oracle"), views, {"orders"},
                                 known=set(schema), granted=set(defs))
    assert verdict is None, verdict
    assert len(folds) < 40 * n, len(folds)  # one pass of the inventory, not one a reference


@pytest.mark.parametrize("filtered_is", ["a view", "the table"])
def test_letters_no_single_narrow_fold_joins_are_joined(filtered_is):
    # Oracle reads the unquoted `ᾀſ` as ᾈS. Lowercase keeps ſ apart from s, and the vouching fold
    # keeps ᾀ (its Python uppercase is two letters), so neither joined `ᾀſ` to ᾈS, and `v` was
    # approved over the filtered ORDERS behind the view ᾈS (Codex's sixth review of #192). A known
    # quoted table `ᾀſ` passes the known-object check. The restricting key is uppercase then
    # casefold, which joins every letter Oracle folds to its fold: all 6,679 letters Oracle Free 23
    # answers, swept live.
    schema = {"orders": {"id"}, "ᾀſ": {"id"}, "ᾈS": {"id"}, "v": {"id"}}
    if filtered_is == "a view":
        filters, body_views = {"orders": "1 = 1"}, {"ᾈS": "SELECT id FROM orders"}
    else:
        filters, body_views = {"ᾈS": "1 = 1"}, {}
    defs = {k: ViewDefinition(object_id=k, dialect="oracle", definition=d) for k, d in body_views.items()}
    defs["v"] = ViewDefinition(object_id="v", dialect="oracle", definition="SELECT id FROM ᾀſ")
    policy = AccessPolicy(row_filters=filters, policy_schema=schema)
    verdict = decide("SELECT id FROM v", {"v": {"id"}}, dialect="oracle", target="oracle", policy=policy, views=defs)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict


def test_the_filtered_set_is_spelled_once_a_request(monkeypatch):
    # Each view walked rebuilt every filtered table's spellings: 4.3 s at 2,000 views and 2,000
    # filters (Codex's sixth review of #192). Grown together here, counted in coarse folds.
    from mnemiq.sql import views as module

    folds = []
    coarse = module._coarse
    monkeypatch.setattr(module, "_coarse", lambda text: folds.append(text) or coarse(text))
    n = 300
    defs = {f"v{i}": ViewDefinition(object_id=f"v{i}", dialect="oracle", definition="SELECT id FROM safe_table")
            for i in range(n)}
    filtered = {f"f{i}" for i in range(n)}
    query = " UNION ALL ".join(f"SELECT id FROM v{i}" for i in range(n))
    known = {"safe_table", *filtered, *defs}
    verdict = module.check_views(sqlglot.parse_one(query, read="oracle"), defs, filtered, known=known, granted=set(defs))
    assert verdict is None, verdict
    assert len(folds) < 40 * n, len(folds)


@pytest.mark.parametrize("dialect", ["postgres", "duckdb", "oracle"])
def test_the_repair_names_a_filtered_table_only_where_the_caller_holds_that_one(dialect):
    # The caller holds `v` and `ä`; the filter is on `Ä`. The repair compared lowercased names and
    # told the caller to query 'Ä' directly, a table it does not hold (Codex's seventh review of
    # #192). Exact names now: no repair here, the refusal stands.
    schema = {"v": {"id"}, "ä": {"id"}, "Ä": {"id"}}
    policy = AccessPolicy(row_filters={"Ä": "1 = 1"}, policy_schema=schema)
    views = {"v": ViewDefinition(object_id="v", dialect=dialect, definition='SELECT id FROM "Ä"')}
    verdict = decide("SELECT id FROM v", {"v": {"id"}, "ä": {"id"}}, dialect=dialect, target=dialect,
                     policy=policy, views=views)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict
    assert "Ä" not in verdict.message, verdict.message
    held = decide("SELECT id FROM v", {"v": {"id"}, "Ä": {"id"}}, dialect=dialect, target=dialect,
                  policy=policy, views=views)
    assert "Query 'Ä' directly." in held.message, held.message  # the control: held, it is offered


def test_repeated_references_to_a_twin_group_are_walked_once(monkeypatch):
    # N case twins (Postgres keeps quoted case apart) read under one spelling N times: each
    # reference enumerated and visited the whole group, N squared (Codex's seventh review of #192).
    import itertools

    from mnemiq.sql import views as module

    lookups = []

    class Counting(dict):
        def get(self, key, default=None):
            lookups.append(key)
            return super().get(key, default)

    class Request(module._Request):
        def __post_init__(self):
            super().__post_init__()
            self.clean = Counting(self.clean)

    monkeypatch.setattr(module, "_Request", Request)
    twins = ["".join(c) for c in itertools.product(*("tT",) * 7)]  # 128 spellings of `ttttttt`
    defs = {k: ViewDefinition(object_id=k, dialect="postgres", definition="SELECT id FROM safe_table") for k in twins}
    query = " UNION ALL ".join("SELECT id FROM ttttttt" for _ in twins)
    verdict = module.check_views(sqlglot.parse_one(query, read="postgres"), defs, {"orders"},
                                 known={"safe_table", "orders", *defs}, granted={"ttttttt"})
    assert verdict is None, verdict
    assert len(lookups) < 4 * len(twins), len(lookups)


def test_distinct_spellings_of_one_twin_group_share_one_walk(monkeypatch):
    # Each of N quoted twins referenced once under its own spelling: a group per spelling
    # enumerated and visited all N, N squared (Codex's eighth review of #192). Every spelling a
    # fold joins has the same candidates, so the group is the fold's.
    import itertools

    from mnemiq.sql import views as module

    lookups = []

    class Counting(dict):
        def get(self, key, default=None):
            lookups.append(key)
            return super().get(key, default)

    class Request(module._Request):
        def __post_init__(self):
            super().__post_init__()
            self.clean = Counting(self.clean)

    monkeypatch.setattr(module, "_Request", Request)
    # Case twins on Postgres, and the long s on Oracle, which lowercase keeps apart from s and the
    # coarse key joins: a group keyed by lowercase as well split those (its ninth review).
    for dialect, letters in (("postgres", "tT"), ("oracle", "sſ")):
        lookups.clear()
        twins = ["".join(c) for c in itertools.product(*(letters,) * 7)]
        defs = {k: ViewDefinition(object_id=k, dialect=dialect, definition="SELECT id FROM safe_table")
                for k in twins}
        query = " UNION ALL ".join(f'SELECT id FROM "{k}"' for k in twins)
        verdict = module.check_views(sqlglot.parse_one(query, read=dialect), defs, {"orders"},
                                     known={"safe_table", "orders", *defs}, granted=set(twins))
        assert verdict is None, (dialect, verdict)
        assert len(lookups) < 4 * len(twins), (dialect, len(lookups))
    # Each reference under its own qualifier, all sharing the bare name's twins: a group per
    # qualified spelling enumerated and visited the shared twins each time (its tenth review).
    lookups.clear()
    twins = ["".join(c) for c in itertools.product(*("tT",) * 7)]
    defs = {k: ViewDefinition(object_id=k, dialect="postgres", definition="SELECT id FROM safe_table") for k in twins}
    query = " UNION ALL ".join(f"SELECT id FROM s{i}.ttttttt" for i in range(len(twins)))
    verdict = module.check_views(sqlglot.parse_one(query, read="postgres"), defs, {"orders"},
                                 known={"safe_table", "orders", *defs}, granted=set(twins))
    assert verdict is None, verdict
    assert len(lookups) < 4 * len(twins), len(lookups)


def test_qualified_twins_are_walked_once_from_every_enclosing_catalog(monkeypatch):
    # N views, each in its own schema, all reading one qualified name with N case twins. A qualified
    # view resolves the names inside it in its own prefix, so its walk is the same from any catalog,
    # yet each enclosing catalog enumerated the twins again, N squared (Codex's eleventh review of
    # #192: 262,656 lookups at N=512).
    import itertools

    from mnemiq.sql import views as module

    lookups = []

    class Counting(dict):
        def get(self, key, default=None):
            lookups.append(key)
            return super().get(key, default)

    class Request(module._Request):
        def __post_init__(self):
            super().__post_init__()
            self.clean = Counting(self.clean)

    monkeypatch.setattr(module, "_Request", Request)
    twins = ["p." + "".join(c) for c in itertools.product(*("tT",) * 7)]
    defs = {k: ViewDefinition(object_id=k, dialect="postgres", definition="SELECT id FROM safe_table") for k in twins}
    wrappers = [f"s{i}.v" for i in range(len(twins))]
    defs |= {w: ViewDefinition(object_id=w, dialect="postgres", definition="SELECT id FROM p.ttttttt")
             for w in wrappers}
    query = " UNION ALL ".join(f"SELECT id FROM {w}" for w in wrappers)
    verdict = module.check_views(sqlglot.parse_one(query, read="postgres"), defs, {"orders"},
                                 known={"safe_table", "orders", *defs}, granted=set(wrappers))
    assert verdict is None, verdict
    assert len(lookups) < 4 * len(defs), len(lookups)


@pytest.mark.parametrize("first, second", [("b.v1", "a.v1"), ("a.v1", "b.v1")])
def test_a_bare_view_walked_clean_in_one_catalog_is_walked_again_in_another(first, second):
    # A bare view resolves the names inside it in the catalog it is read from, so `w` reading `x`
    # reaches the clean `b.x` from `b.v1` and the filtered one through `a.x` from `a.v1`: a record
    # of its bucket shared across catalogs, as a qualified bucket's is, approved the second read.
    from mnemiq.sql.views import check_views

    defs = {k: ViewDefinition(object_id=k, dialect="postgres", definition=body) for k, body in (
        ("a.v1", "SELECT id FROM w"), ("b.v1", "SELECT id FROM w"), ("w", "SELECT id FROM x"),
        ("a.x", "SELECT id FROM orders"), ("b.x", "SELECT id FROM safe_table"))}
    query = f"SELECT id FROM {first} UNION ALL SELECT id FROM {second}"
    verdict = check_views(sqlglot.parse_one(query, read="postgres"), defs, {"orders"},
                          known={"safe_table", "orders", *defs}, granted={"a.v1", "b.v1"})
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict


def test_on_duckdb_a_long_s_view_name_is_still_unknown():
    verdict = _read_through_a_nested_view("ſales", dialect="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict


def test_on_duckdb_the_long_s_names_another_object_and_is_refused_as_unknown():
    # DuckDB folds ASCII alone (M154), so `orderſ` is not ORDERS there: unknown, not ungoverned.
    verdict = _read_view("SELECT id FROM orderſ", "orders", dialect="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict


def _read_with_sales_view(query: str, dialect: str, view: str = "SALES", body: str = "SELECT id FROM orders",
                          visible=None):
    schema = {"safe_table": {"id"}, "orders": {"id"}, view: {"id"}, "v": {"id"}}
    policy = AccessPolicy(row_filters={"orders": "1 = 1"}, policy_schema=schema)
    views = {view: ViewDefinition(object_id=view, dialect=dialect, definition=body)}
    return decide(query, visible or {"safe_table": {"id"}}, dialect=dialect, target=dialect, policy=policy,
                  views=views)


@pytest.mark.parametrize(("dialect", "view", "cte"), [
    ("oracle", "SALES", "ſales"), ("oracle", "SALES", "sales"), ("duckdb", "SALES", "ſales"),
    ("duckdb", "sales", "sales"), ("duckdb", "SALES", "Sales")])
def test_a_name_the_query_s_own_cte_binds_is_not_looked_up_as_a_view(dialect, view, cte):
    # The engine binds the name to the CTE, so the view of that name is never read: looked up, the
    # wide fold matched `ſales` to an ungranted SALES and refused the query, telling the caller a view
    # of that name reads a filtered table (Codex's twelfth review of #192). Main refused the plain
    # spelling the same way and named SALES itself.
    verdict = _read_with_sales_view(f"WITH {cte} AS (SELECT id FROM safe_table) SELECT id FROM {cte}", dialect, view)
    assert isinstance(verdict, Approved), verdict


@pytest.mark.parametrize("dialect", ["oracle", "duckdb"])
def test_a_cte_s_own_body_reading_a_filtered_view_is_still_refused(dialect):
    verdict = _read_with_sales_view("WITH x AS (SELECT id FROM SALES) SELECT id FROM x", dialect,
                                    visible={"safe_table": {"id"}, "SALES": {"id"}})
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict


@pytest.mark.parametrize("dialect", ["oracle", "duckdb"])
def test_a_name_a_view_body_s_own_cte_binds_is_not_looked_up_as_a_view(dialect):
    # The same inside a view: its engine binds `sales` in its body to the body's own CTE.
    schema = {"safe_table": {"id"}, "orders": {"id"}, "SALES": {"id"}, "v": {"id"}}
    policy = AccessPolicy(row_filters={"orders": "1 = 1"}, policy_schema=schema)
    views = {"SALES": ViewDefinition(object_id="SALES", dialect=dialect, definition="SELECT id FROM orders"),
             "v": ViewDefinition(object_id="v", dialect=dialect,
                                 definition="WITH sales AS (SELECT id FROM safe_table) SELECT id FROM sales")}
    verdict = decide("SELECT id FROM v", {"v": {"id"}}, dialect=dialect, target=dialect, policy=policy, views=views)
    assert isinstance(verdict, Approved), verdict


@pytest.mark.parametrize("door", ["read", "write"])
def test_a_cte_oracle_folds_onto_the_name_read_binds_it_through_either_door(door):
    # Oracle folds the unquoted `ſales` to SALES, so `FROM SALES` reads the CTE. Bound by any other
    # engine's fold the name is not the CTE's, and the SALES view behind it was looked up.
    schema = {"safe_table": {"id"}, "orders": {"id"}, "SALES": {"id"}, "scratch": {"id"}}
    policy = AccessPolicy(row_filters={"orders": "1 = 1"}, policy_schema=schema)
    views = {"SALES": ViewDefinition(object_id="SALES", dialect="oracle", definition="SELECT id FROM orders")}
    query = "WITH ſales AS (SELECT id FROM safe_table) SELECT id FROM SALES"
    if door == "read":
        verdict = decide(query, {"safe_table": {"id"}}, dialect="oracle", target="oracle", policy=policy,
                         views=views)
        assert isinstance(verdict, Approved), verdict
    else:
        visible = {"safe_table": {"id"}, "scratch": {"id"}}
        verdict = decide_write(f"INSERT INTO scratch (id) {query}", visible,
                               GrantSet(frozenset(visible), writable=frozenset({"scratch"})),
                               dialect="oracle", target="oracle", writes_enabled=True, policy=policy, views=views)
        assert isinstance(verdict, ApprovedWrite), verdict


def test_a_cte_s_binding_is_asked_only_of_a_reference_some_view_could_answer_to(monkeypatch):
    # Asking costs a walk of every CTE in scope, so asked of every reference 2,000 CTEs were
    # 4,004,001 name resolutions with no view at all (Codex's thirteenth review of #192).
    from mnemiq.sql import views as module

    asked = []
    real = module.cte_binder

    def recording(*args, **kwargs):
        bound = real(*args, **kwargs)
        return lambda t: asked.append(t.name) or bound(t)

    monkeypatch.setattr(module, "cte_binder", recording)
    query = "WITH " + ", ".join(f"c{i} AS (SELECT id FROM safe_table)" for i in range(200)) + " SELECT id FROM " + \
        ", ".join(f"c{i}" for i in range(200)) + ", sales"
    defs = {"SALES": ViewDefinition(object_id="SALES", dialect="duckdb", definition="SELECT id FROM safe_table")}
    verdict = module.check_views(sqlglot.parse_one(query, read="duckdb"), defs, {"orders"},
                                 known={"safe_table", "orders", "SALES"}, granted={"safe_table", "SALES"},
                                 dialect="duckdb")
    assert verdict is None, verdict
    assert asked == ["sales"], asked


def _name_resolutions(monkeypatch) -> list:
    from mnemiq.sql import scope

    calls = []
    real = scope.resolve_name
    monkeypatch.setattr(scope, "resolve_name", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def test_references_to_ctes_named_like_views_resolve_each_cte_name_once(monkeypatch):
    # Each reference some view could answer to is asked whether a CTE binds it, and each asking
    # resolved every CTE name in scope again: 2,000 CTEs named like 2,000 views, each read once,
    # were 4,002,000 resolutions (Codex's fourteenth review of #192).
    from mnemiq.sql.views import check_views

    n = 400
    query = "WITH " + ", ".join(f"c{i} AS (SELECT 1 AS id)" for i in range(n)) + " SELECT 1 FROM " + \
        ", ".join(f"c{i}" for i in range(n))
    defs = {f"c{i}": ViewDefinition(object_id=f"c{i}", dialect="duckdb", definition="SELECT id FROM orders")
            for i in range(n)}
    calls = _name_resolutions(monkeypatch)
    verdict = check_views(sqlglot.parse_one(query, read="duckdb"), defs, {"orders"}, known={"orders", *defs},
                          granted=set(defs), dialect="duckdb")
    assert verdict is None, verdict
    assert len(calls) < 4 * n, len(calls)


def test_a_view_body_s_cte_chain_resolves_each_cte_name_once(monkeypatch):
    # The same inside a view body, where every reference is asked: one view over a chain of 2,000
    # CTEs, each reading the one before, was 4,002,001 resolutions (Codex's fourteenth review).
    from mnemiq.sql.views import check_views

    n = 400
    body = "WITH c0 AS (SELECT id FROM safe_table), " + \
        ", ".join(f"c{i} AS (SELECT id FROM c{i - 1})" for i in range(1, n)) + f" SELECT id FROM c{n - 1}"
    defs = {"v": ViewDefinition(object_id="v", dialect="duckdb", definition=body)}
    calls = _name_resolutions(monkeypatch)
    verdict = check_views(sqlglot.parse_one("SELECT id FROM v", read="duckdb"), defs, {"orders"},
                          known={"orders", "safe_table", "v"}, granted={"v"}, dialect="duckdb")
    assert verdict is None, verdict
    assert len(calls) < 4 * n, len(calls)
