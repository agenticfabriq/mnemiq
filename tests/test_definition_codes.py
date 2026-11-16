"""A definition that names codes its column never holds, named at load.

A definition written for other data (another plant's codes, an older code list) is followed by the writer: it filters on
a code that matches no row, the value check refuses it, and the answer is a decline the definition caused. Nothing said
so at load. `stale_codes` reads each definition for the codes it names for an indexed column, and reports the column
when it holds none of them. A definition that names at least one real value is never reported: a code list may name
codes the data has not seen yet, and a definition may name the code it says to exclude.
"""
from __future__ import annotations

import json
import logging

import duckdb

from mnemiq.config import Settings
from mnemiq.contract import Definition, Snapshot, SourceBinding
from mnemiq.contract.semantic import Column
from mnemiq.semantic.definition_codes import stale_codes
from mnemiq.semantic.values import NoValueIndex, ValueIndex


def _index(values: dict[tuple[str, str], list[str]]) -> ValueIndex:
    """A real value index over an in-memory store, holding `values[(object, column)]`."""
    index = ValueIndex(duckdb.connect())
    for (obj, col), vals in values.items():
        for v in vals:
            index._con.execute("INSERT INTO value_index VALUES ('src', ?, ?, ?)", [obj, col, v])
    return index


def _snapshot(definitions: list[tuple[str, str]], columns: list[tuple[str, str]]) -> Snapshot:
    return Snapshot(version="v1", source_id="src", created_at="2026-11-15T00:00:00Z",
                    columns=[Column(id=f"{o}.{c}", object_id=o, name=c, data_type="VARCHAR") for o, c in columns],
                    definitions=[Definition(id=f"d:{t}", term=t, domain="plant", definition=text)
                                 for t, text in definitions])


PLANTS = {("orders", "PLANT_CODE"): ["XY98"], ("returns", "PLANT_CODE"): ["ZZ77"]}
PLANT_COLUMNS = [("orders", "PLANT_CODE"), ("returns", "PLANT_CODE")]


def test_a_definition_whose_codes_its_column_never_holds_is_named():
    # The codes are bare and come before the column, in one sentence across a semicolon, as a definition written for
    # another plant's data says it.
    snap = _snapshot([("plant", "Orders ship from plant AB12 and returns go to plant CD34; PLANT_CODE carries it.")],
                     PLANT_COLUMNS)
    (found,) = stale_codes(snap, _index(PLANTS))
    assert (found.term, found.column, found.codes) == ("plant", "PLANT_CODE", ("AB12", "CD34"))
    assert set(found.objects) == {"orders", "returns"}
    assert set(found.nearest) == {"XY98", "ZZ77"}


def test_quoted_single_letter_codes_are_read():
    snap = _snapshot([("ship mode", "SHIP_MODE 'A' is air freight and SHIP_MODE 'G' is ground.")],
                     [("shipments", "SHIP_MODE")])
    (found,) = stale_codes(snap, _index({("shipments", "SHIP_MODE"): ["R", "S", "T"]}))
    assert found.codes == ("A", "G")


def test_a_code_list_that_names_one_real_value_is_not_reported():
    # A list of the codes a column may hold, some of which this data has not seen: not stale.
    snap = _snapshot([("grade", "QUALITY_TIER 'Q10' is the only passing grade. Defects: D20 chip, D99 crack.")],
                     [("parts", "QUALITY_TIER")])
    assert stale_codes(snap, _index({("parts", "QUALITY_TIER"): ["Q10", "D20", "D13"]})) == []


def test_naming_the_code_to_exclude_beside_the_one_to_keep_is_not_reported():
    # The data holds no test rows at all, so the excluded code is absent; the definition is still right.
    snap = _snapshot([("test rows", "SORT_FLAG = '0' marks test rows; every count filters SORT_FLAG = '2'.")],
                     [("parts", "SORT_FLAG")])
    assert stale_codes(snap, _index({("parts", "SORT_FLAG"): ["2"]})) == []


def test_ordinary_uppercase_words_and_prose_numbers_are_not_codes():
    snap = _snapshot([("bins", "Count SLOT_NO per DAILY run within 30 days, NOT NULL rows only: SELECT DISTINCT SLOT_NO.")],
                     [("parts", "SLOT_NO")])
    assert stale_codes(snap, _index({("parts", "SLOT_NO"): ["01", "05", "58"]})) == []


def test_a_definition_that_names_no_column_is_not_judged():
    # AB12 has the shape of a plant code, but the definition is about grades and names no column.
    snap = _snapshot([("grade", "Grade AB12 is the premium grade.")], PLANT_COLUMNS)
    assert stale_codes(snap, _index(PLANTS)) == []


def test_apostrophes_do_not_shift_which_quotes_pair():
    # Paired naively, "lot's ... 'A'" and "order's ... 'G'" make three phrases and no code.
    snap = _snapshot([("ship mode", "The lot's SHIP_MODE is 'A', the order's is 'G'.")], [("shipments", "SHIP_MODE")])
    (found,) = stale_codes(snap, _index({("shipments", "SHIP_MODE"): ["R", "S", "T"]}))
    assert found.codes == ("A", "G")


def test_an_apostrophe_does_not_pair_with_a_quote():
    # "lot's" opens no literal, so 'Q1' is read as the code it is, stale or not.
    text = "The lot's grade is QUALITY_TIER '{}', and the batch's is the same."
    columns = [("lots", "QUALITY_TIER")]
    held = _index({("lots", "QUALITY_TIER"): ["B7", "C9"]})
    assert stale_codes(_snapshot([("grade", text.format("B7"))], columns), held) == []
    (found,) = stale_codes(_snapshot([("grade", text.format("Q1"))], columns), held)
    assert found.codes == ("Q1",)


def test_a_table_name_shaped_like_a_code_is_not_a_code():
    # Table QX1 has the shape of the bay codes (QX2, QX3): naming the table names no bay.
    snap = _snapshot([("bay", "Bays are the BAY_CODE values of table QX1.")], [("QX1", "BAY_CODE")])
    assert stale_codes(snap, _index({("QX1", "BAY_CODE"): ["QX2", "QX3"]})) == []


def test_a_quoted_phrase_unlike_the_columns_values_is_not_a_code():
    snap = _snapshot([("plant", "A range 'from May to June' of PLANT_CODE rows, or 'any plant'.")], PLANT_COLUMNS)
    assert stale_codes(snap, _index(PLANTS)) == []


def test_a_column_named_only_as_a_plain_word_is_not_checked():
    # `status` is also an English word: a definition saying "status 'pending'" may mean the word. Known limit.
    snap = _snapshot([("status", "An order whose status is 'paid' has not shipped.")], [("orders", "status")])
    assert stale_codes(snap, _index({("orders", "status"): ["open", "shut"]})) == []


def test_a_column_not_indexed_in_every_table_cannot_be_judged():
    # The code may be in the table whose values were not read: unknown, not absent.
    snap = _snapshot([("plant", "Every order ships from plant AB12, on PLANT_CODE.")], PLANT_COLUMNS)
    assert stale_codes(snap, _index({("orders", "PLANT_CODE"): ["XY98"]})) == []


def test_no_value_index_judges_nothing():
    snap = _snapshot([("plant", "Every order ships from plant AB12, on PLANT_CODE.")], PLANT_COLUMNS)
    assert stale_codes(snap, NoValueIndex()) == []


def test_the_value_index_lists_a_columns_values():
    index = _index({("orders", "PLANT_CODE"): ["XY98", "AB12"]})
    assert index.values("orders", "PLANT_CODE") == ["AB12", "XY98"]
    assert index.values("orders", "OTHER") == []
    assert NoValueIndex().values("orders", "PLANT_CODE") == []


def _boot(tmp_path, definitions: list[tuple[str, str]], policy: dict | None = None) -> Settings:
    """A store holding a snapshot of `orders(PLANT_CODE)` with `definitions`, and its value index, as enrichment leaves
    it; `build_runtime` reads both."""
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import save_snapshot

    source = tmp_path / "src.duckdb"
    duckdb.connect(str(source)).execute("CREATE TABLE orders (PLANT_CODE TEXT); INSERT INTO orders VALUES ('XY98')").close()
    store = tmp_path / "store.duckdb"
    con = init_store(str(store))
    snap = Snapshot(version="v1", source_id="only", created_at="2026-11-15T00:00:00Z",
                    source_bindings=[SourceBinding(id="sb:orders", source_id="only", object_id="orders",
                                                   source_object="orders", binding_type="table")],
                    columns=[Column(id="orders.PLANT_CODE", object_id="orders", name="PLANT_CODE", data_type="VARCHAR")],
                    definitions=[Definition(id=f"d:{t}", term=t, domain="plant", definition=text)
                                 for t, text in definitions])
    save_snapshot(con, snap)
    ValueIndex(con)._con.execute("INSERT INTO value_index VALUES ('only', 'orders', 'PLANT_CODE', 'XY98')")
    con.close()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "duckdb", "target": str(source),
                                     "catalog": "src", "schema": "main"}]))
    authz = tmp_path / "authz.json"
    authz.write_text(json.dumps(policy or {"roles": {"analyst": ["orders"]}}))
    return Settings(llm_base_url="http://127.0.0.1:9/v1", llm_api_key="k", llm_model="m", llm_window_check=False,
                    sources_path=str(manifest), store_path=str(store), authz_path=str(authz))


def _stale_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "holds none of them" in r.getMessage()]


def test_build_runtime_warns_about_a_definition_whose_codes_its_data_lacks(tmp_path, caplog):
    """The join: the boot door over a stored snapshot and value index, nothing stubbed but the model server."""
    from mnemiq.runtime import build_runtime

    settings = _boot(tmp_path, [("plant", "Every order ships from plant AB12; PLANT_CODE carries it.")])
    with caplog.at_level(logging.WARNING):
        build_runtime(settings)
    lines = _stale_lines(caplog)
    assert len(lines) == 1, caplog.text
    line = lines[0]
    assert "'plant'" in line and "AB12" in line and "PLANT_CODE" in line and "XY98" in line


def test_build_runtime_is_silent_when_every_definition_names_a_real_code(tmp_path, caplog):
    from mnemiq.runtime import build_runtime

    settings = _boot(tmp_path, [("plant", "Every order ships from plant XY98; PLANT_CODE carries it.")])
    with caplog.at_level(logging.WARNING):
        build_runtime(settings)
    assert _stale_lines(caplog) == []


def test_a_row_filtered_tables_values_are_not_listed(tmp_path, caplog):
    """As the value check withholds them from a row-filtered identity (M5): the index is read without any filter."""
    from mnemiq.runtime import build_runtime

    policy = {"roles": {"west": {"read": ["orders"], "row_filters": {"ORDERS": "PLANT_CODE = 'XY98'"}}}}
    settings = _boot(tmp_path, [("plant", "Every order ships from plant AB12; PLANT_CODE carries it.")], policy)
    with caplog.at_level(logging.WARNING):
        build_runtime(settings)
    lines = _stale_lines(caplog)
    assert len(lines) == 1, caplog.text
    line = lines[0]
    assert "AB12" in line and "XY98" not in line and "row filter" in line


def test_a_hot_swapped_snapshot_is_checked_for_stale_codes(tmp_path, monkeypatch, caplog):
    """`reload_if_stale` replaces the snapshot, definitions included: the new one is checked against the index."""
    from mnemiq.runtime import Runtime

    class _Settings:
        control_dsn = "postgresql://x"
        ack_advisories = ""

    store = duckdb.connect()
    ValueIndex(store)._con.execute("INSERT INTO value_index VALUES ('only', 'orders', 'PLANT_CODE', 'XY98')")
    new = _snapshot([("plant", "Every order ships from plant AB12, on PLANT_CODE.")], [("orders", "PLANT_CODE")])
    rt = Runtime.__new__(Runtime)
    rt.settings, rt.con, rt.authz = _Settings(), store, None
    rt.snapshot, rt.loaded_versions = object(), {"only": "v1"}
    monkeypatch.setattr("mnemiq.runtime.load_current_snapshot", lambda _s, _c: (new, {"only": "v2"}))
    monkeypatch.setattr("mnemiq.runtime._warn_prompt_window", lambda *a, **k: None)
    monkeypatch.setattr("mnemiq.runtime._warn_policy_advisories", lambda *a, **k: None)
    with caplog.at_level(logging.WARNING):
        rt.reload_if_stale()
    lines = _stale_lines(caplog)
    assert len(lines) == 1 and "AB12" in lines[0], caplog.text


def test_a_value_index_that_cannot_be_read_never_stops_a_swap(monkeypatch, caplog):
    """The check is advisory: a store it cannot read is said, and the swap, on a user's request, goes on."""
    from mnemiq.runtime import Runtime

    class _Settings:
        control_dsn = "postgresql://x"
        ack_advisories = ""

    new = _snapshot([("plant", "Every order ships from plant AB12, on PLANT_CODE.")], [("orders", "PLANT_CODE")])
    rt = Runtime.__new__(Runtime)
    rt.settings, rt.con, rt.authz = _Settings(), object(), None
    rt.snapshot, rt.loaded_versions = object(), {"only": "v1"}
    monkeypatch.setattr("mnemiq.runtime.load_current_snapshot", lambda _s, _c: (new, {"only": "v2"}))
    monkeypatch.setattr("mnemiq.runtime._warn_prompt_window", lambda *a, **k: None)
    monkeypatch.setattr("mnemiq.runtime._warn_policy_advisories", lambda *a, **k: None)
    with caplog.at_level(logging.WARNING):
        try:
            rt.reload_if_stale()
        except Exception as exc:
            raise AssertionError(f"the swap raised {exc!r}") from exc
    assert rt.snapshot is new
    assert "definitions were not checked against the value index" in caplog.text


def test_superset_mode_judges_nothing(caplog):
    """Superset mode harvests no values and gives its writer none, so a value index left in its store judges nothing."""
    from mnemiq.runtime import _warn_stale_codes

    store = duckdb.connect()
    ValueIndex(store)._con.execute("INSERT INTO value_index VALUES ('only', 'orders', 'PLANT_CODE', 'XY98')")
    snap = _snapshot([("plant", "Every order ships from plant AB12, on PLANT_CODE.")], [("orders", "PLANT_CODE")])
    with caplog.at_level(logging.WARNING):
        _warn_stale_codes(snap, store, None, superset=True)
    assert _stale_lines(caplog) == []
    with caplog.at_level(logging.WARNING):
        _warn_stale_codes(snap, store, None, superset=False)
    assert len(_stale_lines(caplog)) == 1


def test_a_definition_that_only_excludes_a_code_is_not_reported():
    # Following it filters the code OUT, and every row is kept: the definition is right on data with no X rows.
    snap = _snapshot([("ship mode", "Exclude rows with SHIP_MODE = 'X'; never count SHIP_MODE <> 'Y' twice.")],
                     [("shipments", "SHIP_MODE")])
    assert stale_codes(snap, _index({("shipments", "SHIP_MODE"): ["A", "G"]})) == []


def test_a_code_both_excluded_and_named_plainly_counts():
    # "not 'Q1'" excludes, and the sentence after names Q1 plainly as the good tier: the definition selects on it.
    snap = _snapshot([("tier", "A part is defective when its QUALITY_TIER is not 'Q1'; Q1 is a good part.")],
                     [("parts", "QUALITY_TIER")])
    (found,) = stale_codes(snap, _index({("parts", "QUALITY_TIER"): ["Q2", "Q3"]}))
    assert found.codes == ("Q1",)


def test_an_apostrophe_before_a_punctuated_code_opens_nothing():
    # Opened at "lot's", a literal would run to the quote before -A and swallow it, leaving -G alone and unheld.
    snap = _snapshot([("ship mode", "The lot's SHIP_MODE='-A' or '-G'.")], [("shipments", "SHIP_MODE")])
    assert stale_codes(snap, _index({("shipments", "SHIP_MODE"): ["-A"]})) == []


def test_an_unreadable_policy_lists_no_values(tmp_path, caplog):
    """A policy that cannot be read lists no roles, and that is not a policy without row filters: say nothing of the
    values (M5)."""
    from mnemiq.authz.grants import FileAuthzProvider
    from mnemiq.runtime import _warn_stale_codes

    store = duckdb.connect()
    ValueIndex(store)._con.execute("INSERT INTO value_index VALUES ('only', 'orders', 'PLANT_CODE', 'XY98')")
    snap = _snapshot([("plant", "Every order ships from plant AB12, on PLANT_CODE.")], [("orders", "PLANT_CODE")])
    with caplog.at_level(logging.WARNING):
        _warn_stale_codes(snap, store, FileAuthzProvider(str(tmp_path / "missing.json")), superset=False)
    lines = _stale_lines(caplog)
    assert len(lines) == 1 and "AB12" in lines[0] and "XY98" not in lines[0], caplog.text
    assert "could not be read" in lines[0]


def test_an_all_capitals_word_is_not_a_code():
    # NOTE has the shape of the region codes, but a bare code must hold a digit: an acronym or a shouted word is prose.
    snap = _snapshot([("region", "REGION_CODE is the sales region, as the FAQ and the NOTE say.")],
                     [("sales", "REGION_CODE")])
    assert stale_codes(snap, _index({("sales", "REGION_CODE"): ["EAST", "WEST"]})) == []


def test_inequality_and_exclusion_forms_are_exclusions():
    held = _index({("shipments", "SHIP_MODE"): ["R"]})
    for text in ["Count rows where SHIP_MODE != 'X'.", "Count rows where SHIP_MODE!='X'.",
                 "Count rows where SHIP_MODE NOT IN ('X', 'Y').", "Every SHIP_MODE other than 'X' counts.",
                 "Every SHIP_MODE except 'X' counts.", "Count shipments excluding 'X'.",
                 "A shipment counts when its SHIP_MODE is not 'X'.", "Rows with SHIP_MODE = 'X' are excluded."]:
        snap = _snapshot([("ship mode", text)], [("shipments", "SHIP_MODE")])
        assert stale_codes(snap, held) == [], text


def test_a_nearby_negation_that_selects_still_counts():
    # Neither is an exclusion: each selects on X, which the data does not hold.
    held = _index({("shipments", "SHIP_MODE"): ["R"]})
    for text in ["SHIP_MODE is not only 'X' but also 'Y'.", "Do not omit SHIP_MODE = 'X'."]:
        snap = _snapshot([("ship mode", text)], [("shipments", "SHIP_MODE")])
        assert [f.term for f in stale_codes(snap, held)] == ["ship mode"], text


def test_the_policys_row_filters_are_read_once(tmp_path):
    """One read of the file, so a policy edited between reads cannot pair one version's roles with another's filters."""
    from mnemiq.authz.grants import FileAuthzProvider

    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"roles": {"analyst": ["orders"],
                                          "west": {"read": ["orders"], "row_filters": {"ORDERS": "PLANT_CODE = 'A'"}}}}))
    assert FileAuthzProvider(str(path)).row_filtered_objects() == frozenset({"ORDERS"})
    for broken in ["{", json.dumps({"roles": ["analyst"]}), json.dumps({"roles": {"w": {"row_filters": ["ORDERS"]}}}),
                   json.dumps({"no roles": {}})]:
        path.write_text(broken)
        assert FileAuthzProvider(str(path)).row_filtered_objects() is None, broken
    assert FileAuthzProvider(str(tmp_path / "missing.json")).row_filtered_objects() is None


def test_the_warning_reads_the_policy_once(tmp_path, monkeypatch, caplog):
    import builtins

    from mnemiq.authz.grants import FileAuthzProvider
    from mnemiq.runtime import _warn_stale_codes

    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"roles": {"analyst": ["orders"]}}))
    opened: list[str] = []
    real_open = builtins.open

    def counting_open(file, *a, **k):
        if str(file) == str(path):
            opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", counting_open)
    store = duckdb.connect()
    ValueIndex(store)._con.execute("INSERT INTO value_index VALUES ('only', 'orders', 'PLANT_CODE', 'XY98')")
    snap = _snapshot([("plant", "Every order ships from plant AB12, on PLANT_CODE.")], [("orders", "PLANT_CODE")])
    with caplog.at_level(logging.WARNING):
        _warn_stale_codes(snap, store, FileAuthzProvider(str(path)), superset=False)
    assert len(opened) == 1, opened
    (line,) = _stale_lines(caplog)
    assert "XY98" in line


def test_the_row_filter_list_agrees_with_each_roles_grants(tmp_path):
    """Two places read a policy's row filters: `grants_for`, which the engine applies, and `row_filtered_objects`, which
    decides what this warning may quote. Pinned together, so a filter `grants_for` learns to read cannot be missed here."""
    from mnemiq.authz.grants import FileAuthzProvider
    from mnemiq.contract import IdentityContext

    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"roles": {
        "analyst": ["orders"],
        "west": {"read": ["orders", "returns"], "row_filters": {"ORDERS": "region = 'w'", "returns": "region = 'w'"}},
        "east": {"read": ["orders"], "row_filters": {"orders": "region = 'e'"}},
        "writer": {"read": ["orders"], "write": ["orders"]}}}))
    policy = FileAuthzProvider(str(path))
    per_role = set()
    for role in policy.policy_roles():
        grants = policy.grants_for(IdentityContext(tenant_id="t", principal_id="p", roles=[role]))
        per_role |= {table.lower() for table in grants.row_filters}
    assert {table.lower() for table in policy.row_filtered_objects()} == per_role == {"orders", "returns"}
