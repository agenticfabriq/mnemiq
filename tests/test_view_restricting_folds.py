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

from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Approved, Refusal, RefusalCode
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


def test_on_duckdb_a_long_s_view_name_is_still_unknown():
    verdict = _read_through_a_nested_view("ſales", dialect="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict


def test_on_duckdb_the_long_s_names_another_object_and_is_refused_as_unknown():
    # DuckDB folds ASCII alone (M154), so `orderſ` is not ORDERS there: unknown, not ungoverned.
    verdict = _read_view("SELECT id FROM orderſ", "orders", dialect="duckdb")
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict
