"""Which snapshot tables a person may read, as Superset decides when asked as them: a dry run naming every table, and
when Superset refuses it, the tables the physical datasets it lists for them name, each kept only when a dry run of it
is accepted too.

Most tests drive `resolve_grants` over an adapter whose client is a recording stand-in that answers `execute_sql` and
`list_datasets` the way Superset does: a dry run accepted only when the person may read every table it names, and the
datasets paged. A few run it over the real client against the fake Superset, so the refusals reach the grant as the
client reads them off the wire.
"""

from __future__ import annotations

import logging

import pytest
import sqlglot
from sqlglot import exp

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.authz.grants import UNAVAILABLE, AuthzProvider, GrantSet, classified_levels
from mnemiq.contract import IdentityContext
from mnemiq.contract.semantic import Column, Dimension, Snapshot, SourceBinding
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.client import (
    DelegatedToken,
    SupersetDenied,
    SupersetMcp,
    SupersetPermissionDenied,
    SupersetRefusedCaller,
    SupersetResponseTooLarge,
    SupersetToolError,
    SupersetUnavailable,
)
from mnemiq.superset.grants import (
    DATASET_FIELDS,
    PAGE_SIZE,
    FixedGrants,
    NotABuilder,
    SupersetMisconfigured,
    resolve_grants,
)
from tests.superset_fake import FakeSuperset, RecordingClient

DATABASE_ID = 7
SCHEMA = "sales"
NOT_A_BUILDER = "Superset refused this user's queries. Building with mnemiq needs Superset's Alpha and sql_lab roles."
# What `execute_sql` answers a dry run it accepts.
DRY_RUN_OK = {"success": True, "rows": None, "columns": None, "row_count": None, "affected_rows": None,
              "error": None, "error_type": None, "statements": []}
PARTY_REFUSAL = "You need access to the following tables: `sales.party`, roles: [party_admin]"
# A role denial as `MCPPermissionDeniedError` words it, naming a table the person cannot see.
ROLE_DENIAL = "Permission denied: can_execute_sql_query on SQLLab for user west (tool: execute_sql) reading sales.party"
LIST_ROLE_DENIAL = "Permission denied: can_read on Dataset for user west (tool: list_datasets)"
GENERIC_DENIAL = "Permission denied for execute_sql: You don't have access to this resource."
# The opening of the error Superset's response size guard raises for a response it does not cut
# (`format_size_limit_error` in `superset/mcp_service/utils/response_size_utils.py`).
TOO_LARGE = "Response too large: 61,234 bytes (limit: 50,000)\n\nThis response would overwhelm the LLM context window."


# More `list_datasets` requests than any test here needs, so a read that never stops fails rather than hangs.
_MOST_PAGES = 1000


def _refused(text: str = PARTY_REFUSAL, error_type: str = "TABLE_SECURITY_ACCESS_ERROR") -> dict:
    return {"success": False, "rows": None, "columns": None, "error": text, "error_type": error_type}


def _access(readable, *, unparsable=()):
    """Answers a dry run as Superset's access check does for a person who may read the tables in `readable`: accepted
    when every table it names is readable, refused naming those that are not, and failed when it names one in
    `unparsable`, a table Superset cannot parse for access."""
    readable, unparsable = set(readable), set(unparsable)

    def answer(request: dict) -> dict:
        names = _names(request["sql"])
        if unparsable & set(names):
            return _refused("Unable to parse SQL", error_type="INVALID_SQL_ERROR")
        missing = [name for name in names if name not in readable]
        if missing:
            return _refused("You need access to the following tables: " + ", ".join(f"`sales.{n}`" for n in missing))
        return DRY_RUN_OK

    return answer


def _in_turn(*answers):
    """Answers each dry run with the next of `answers`, the last one for every dry run after it."""
    queue = list(answers)

    def answer(request: dict):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return answer


def _names(sql: str) -> list[str]:
    """The tables a dry run names, in order."""
    return [table.name for table in sqlglot.parse_one(sql, read="postgres").find_all(exp.Table)]


def _snapshot(*tables: str, columns: list[Column] = (), dimensions: list[Dimension] = ()) -> Snapshot:
    return Snapshot(version="v1", source_id="superset", created_at="t", columns=list(columns),
                    dimensions=list(dimensions),
                    source_bindings=[SourceBinding(id=f"sb:{t}", source_id="superset", object_id=t, source_object=t,
                                                   binding_type="table") for t in tables])


def _dataset(dataset_id: int, table_name: str, *, database_id: int = DATABASE_ID, schema: str | None = SCHEMA,
             sql: str | None = None) -> dict:
    return {"id": dataset_id, "table_name": table_name, "schema": schema, "database_id": database_id, "sql": sql}


class _Superset:
    """Answers `execute_sql` with `probe`, and `list_datasets` as Superset pages `datasets`: its `schema` filter
    applied, ordered by id, page `page` (counted from 1) holding the `page_size` datasets after the first
    `(page - 1) * page_size`, each carrying only the fields asked for.

    `cut_above` maps a dataset id to the largest page size at which a page holding it still fits Superset's response
    size; a larger page holding it is answered as `cut` says: "error", the error the size guard raises for a tool it
    does not cut, or "flag", a page marked `_response_truncated`. `asked` records every `list_datasets` request, cut or
    not, `pages` each page answered whole: its request and the ids it carried, and `probes` the tables each dry run
    named."""

    def __init__(self, probe, datasets: list[dict] = (), *, cut_above: dict[int, int] | None = None,
                 cut: str = "error", list_answer=None) -> None:
        self.probe = probe
        self.datasets = sorted(datasets, key=lambda d: d["id"])
        self.cut_above = cut_above or {}
        self.cut = cut
        self.list_answer = list_answer
        self.asked: list[dict] = []
        self.pages: list[tuple[dict, list[int]]] = []
        self.probes: list[list[str]] = []

    def __call__(self, request: dict):
        if "sql" in request:
            self.probes.append(_names(request["sql"]))
            return self.probe(request) if callable(self.probe) else self.probe
        self.asked.append(request)
        assert len(self.asked) <= _MOST_PAGES, "the listing never stopped asking for pages"
        if self.list_answer is not None:
            return self.list_answer(request) if callable(self.list_answer) else self.list_answer
        return self._page(request)

    def _page(self, request: dict):
        rows = [d for d in self.datasets
                if all(d.get(f["col"]) == f["value"] for f in request.get("filters", []) if f["opr"] == "eq")]
        size, page = request["page_size"], request["page"]
        window = rows[(page - 1) * size:page * size]
        if any(size > self.cut_above[d["id"]] for d in window if d["id"] in self.cut_above):
            if self.cut == "error":
                return SupersetResponseTooLarge("Superset's answer to list_datasets was larger than its MCP service "
                                                "will send; ask for less", detail=f"Error: {TOO_LARGE}")
            return {"datasets": [], "page": page, "page_size": size, "_response_truncated": True,
                    "_truncation_notes": ["datasets cut to fit"]}
        self.pages.append((request, [d["id"] for d in window]))
        fields = request.get("select_columns") or list(DATASET_FIELDS)
        return {"count": len(window), "total_count": len(rows), "page": page, "page_size": size,
                "has_next": page * size < len(rows), "columns_requested": fields,
                "datasets": [{k: d[k] for k in fields if k in d} for d in window]}


def _adapter(superset: _Superset) -> SupersetSqlAdapter:
    return SupersetSqlAdapter(RecordingClient(superset), database_id=DATABASE_ID, schema=SCHEMA)


def _tools(adapter: SupersetSqlAdapter) -> list[str]:
    return [call.tool for call in adapter.client.calls]


# --- the probe ---------------------------------------------------------------------------------------------------


def test_database_wide_access_grants_every_snapshot_table():
    snapshot = _snapshot("policy", "claim", "party",
                         columns=[Column(id="party.ssn", object_id="party", name="ssn", pii_level="pii")])
    adapter = _adapter(_Superset(DRY_RUN_OK))
    grants = resolve_grants(adapter, snapshot)
    assert grants.objects == {"policy", "claim", "party"}
    assert grants.available is True
    assert grants.pii_clearance == {"pii"}
    assert (grants.writable, grants.row_filters, grants.pii_mask) == (frozenset(), {}, frozenset())
    # One dry run, as the person, and no dataset listing.
    assert _tools(adapter) == ["execute_sql"]
    request = adapter.client.calls[0].request
    assert request["dry_run"] is True
    assert request["sql"] == 'SELECT 1 FROM "policy", "claim", "party" WHERE 1 = 0'
    assert (request["database_id"], request["schema"]) == (DATABASE_ID, SCHEMA)


def test_probe_quotes_stored_names():
    """Each table is named in the probe exactly as Postgres stores it: quoted, so a mixed-case name or one holding a
    space or a quote reaches that table and no other, and granted under that same stored name."""
    stored = ("Policy Lines", "party", "Party", 'say "hi"')
    adapter = _adapter(_Superset(DRY_RUN_OK))
    grants = resolve_grants(adapter, _snapshot(*stored))
    sql = adapter.client.calls[0].request["sql"]
    assert sql == 'SELECT 1 FROM "Policy Lines", "party", "Party", "say ""hi""" WHERE 1 = 0'
    tables = sqlglot.parse_one(sql, read="postgres").find_all(exp.Table)
    assert [(t.name, t.this.args.get("quoted")) for t in tables] == [(name, True) for name in stored]
    assert grants.objects == set(stored)


def test_a_table_bound_twice_is_probed_once():
    snapshot = _snapshot("policy", "policy", "claim")
    adapter = _adapter(_Superset(DRY_RUN_OK))
    assert resolve_grants(adapter, snapshot).objects == {"policy", "claim"}
    assert adapter.client.calls[0].request["sql"] == 'SELECT 1 FROM "policy", "claim" WHERE 1 = 0'


def test_only_table_bindings_are_probed():
    snapshot = _snapshot("policy")
    snapshot.source_bindings.append(SourceBinding(id="sb:feed", source_id="superset", object_id="feed",
                                                  source_object="feed", binding_type="stream"))
    adapter = _adapter(_Superset(DRY_RUN_OK))
    assert resolve_grants(adapter, snapshot).objects == {"policy"}
    assert adapter.client.calls[0].request["sql"] == 'SELECT 1 FROM "policy" WHERE 1 = 0'


def test_a_snapshot_without_tables_asks_nothing_and_grants_nothing():
    adapter = _adapter(_Superset(DRY_RUN_OK))
    grants = resolve_grants(adapter, _snapshot())
    assert grants.objects == frozenset() and grants.available is True
    assert adapter.client.calls == []


@pytest.mark.parametrize("path", ["probe", "datasets"])
def test_every_classified_level_is_cleared(path):
    """Superset has no column rules for SQL, so whatever the person may read they read raw: every level the snapshot
    classifies anything at, its columns' own, their certified dimensions' and its dimensions', is cleared."""
    columns = [Column(id="party.ssn", object_id="party", name="ssn", pii_level="pii"),
               Column(id="claim.diagnosis", object_id="claim", name="diagnosis", pii_level="phi"),
               Column(id="policy.card", object_id="policy", name="card", dimension_pii_levels=["pci"]),
               Column(id="policy.region", object_id="policy", name="region", pii_level="none")]
    dimensions = [Dimension(id="party.dob", label="dob", source="party", pii_level="dob")]
    snapshot = _snapshot("policy", "claim", "party", columns=columns, dimensions=dimensions)
    probe = DRY_RUN_OK if path == "probe" else _access({"policy"})
    grants = resolve_grants(_adapter(_Superset(probe, [_dataset(1, "policy")])), snapshot)
    assert grants.pii_clearance == classified_levels(snapshot) == {"pii", "phi", "pci", "dob"}
    assert grants.objects == ({"policy", "claim", "party"} if path == "probe" else {"policy"})


# --- the dataset fallback ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("stored", [("policy", "claim", "party"), ("policy", "claim")])
def test_dataset_access_grants_exactly_those_tables(stored):
    """The datasets name the candidates, and Superset, asked as the person, confirms them."""
    datasets = [
        _dataset(1, "policy"),                                         # granted
        _dataset(2, "claim", database_id=8),                           # another database
        # Virtual, though named for a stored table: names neither that table nor the one its SQL reads.
        _dataset(3, "claim", sql="SELECT * FROM party WHERE region = 'west'"),
        _dataset(4, "Party"),                                          # names stored `party`, read unquoted
    ]
    superset = _Superset(_access({"policy", "party"}), datasets)
    adapter = _adapter(superset)
    grants = resolve_grants(adapter, _snapshot(*stored))
    candidates = ["policy", "party"] if "party" in stored else ["policy"]
    assert grants.objects == set(candidates)
    assert grants.available is True
    assert _tools(adapter) == ["execute_sql", "list_datasets", "execute_sql"]
    assert superset.probes == [list(stored), candidates]


def test_a_dataset_names_the_table_stored_under_its_own_name():
    """A dataset names the table stored under exactly its name as a candidate, probed quoted, so a mixed-case name or
    one holding a space is confirmed and granted under that stored name."""
    datasets = [_dataset(1, "Policy Lines"), _dataset(2, "Party")]
    superset = _Superset(_access({"Policy Lines", "Party"}), datasets)
    grants = resolve_grants(_adapter(superset), _snapshot("Policy Lines", "Party", "claim"))
    assert grants.objects == {"Policy Lines", "Party"}
    assert superset.probes[1] == ["Policy Lines", "Party"]


def test_a_name_postgres_cannot_read_unquoted_names_only_itself():
    """Only a plain word read unquoted folds: a dataset whose name holds a space, or is a word Postgres reserves, names
    the table stored under its own name and never the table its lower case names, so nothing is asked about those."""
    datasets = [_dataset(1, "Policy Lines"), _dataset(2, "Order"), _dataset(3, "Select")]
    superset = _Superset(_access(set()), datasets)
    assert resolve_grants(_adapter(superset), _snapshot("policy lines", "order", "select")).objects == frozenset()
    assert len(superset.probes) == 1


def test_a_dataset_in_another_schema_is_never_granted():
    """The schema is compared as stored, whatever the listing's own filter let through."""
    page = {"datasets": [_dataset(1, "policy", schema="Sales"), _dataset(2, "claim", schema=None),
                         _dataset(3, "party", schema="sales ")]}
    superset = _Superset(_access({"policy", "party"}), list_answer=page)
    assert resolve_grants(_adapter(superset), _snapshot("policy", "claim", "party")).objects == frozenset()
    assert len(superset.probes) == 1


def test_an_empty_sql_is_a_physical_dataset():
    """Superset reads a dataset as virtual when its SQL is not empty (`SqlaTable.kind`), and only then."""
    datasets = [_dataset(1, "policy", sql=""), _dataset(2, "claim", sql=" ")]
    superset = _Superset(_access({"policy", "claim"}), datasets)
    grants = resolve_grants(_adapter(superset), _snapshot("policy", "claim", "secret"))
    assert grants.objects == {"policy"}
    assert superset.probes[1:] == [["policy"]]


def test_a_dataset_in_another_catalog_is_not_granted():
    """The listing carries no catalog, so a physical dataset in another catalog looks like one for the snapshot's table.
    Superset, asked as the person, refuses that table, so it is not granted, and the other candidates still are."""
    datasets = [_dataset(1, "policy"), _dataset(2, "claim"), _dataset(3, "party")]
    superset = _Superset(_access({"policy", "party"}), datasets)  # `claim`'s dataset is another catalog's
    grants = resolve_grants(_adapter(superset), _snapshot("policy", "claim", "party", "secret"))
    assert grants.objects == {"policy", "party"}
    assert superset.probes == [["policy", "claim", "party", "secret"], ["policy", "claim", "party"],
                               ["policy"], ["claim"], ["party"]]


def test_a_case_folded_name_superset_refuses_is_not_granted():
    """A dataset `Party` makes a stored `party` a candidate, which the fallback asks Superset about like any other:
    refused, it is not granted, and a single refused candidate is not asked about again."""
    superset = _Superset(_access(set()), [_dataset(1, "Party")])
    grants = resolve_grants(_adapter(superset), _snapshot("party", "claim"))
    assert grants.objects == frozenset() and grants.available is True
    assert superset.probes == [["party", "claim"], ["party"]]


def test_confirmed_candidates_are_granted_with_one_probe():
    superset = _Superset(_access({"policy", "claim"}), [_dataset(1, "policy"), _dataset(2, "claim")])
    adapter = _adapter(superset)
    grants = resolve_grants(adapter, _snapshot("policy", "claim", "secret"))
    assert grants.objects == {"policy", "claim"}
    assert _tools(adapter) == ["execute_sql", "list_datasets", "execute_sql"]
    confirmation = adapter.client.calls[2].request
    assert confirmation["dry_run"] is True
    assert confirmation["sql"] == 'SELECT 1 FROM "policy", "claim" WHERE 1 = 0'


def test_one_refused_candidate_does_not_deny_the_rest(caplog):
    """A candidate Superset refuses, or cannot check, is left out alone, the reason logged; the others are granted.
    Superset's text naming the refused table stays in the log."""
    datasets = [_dataset(i, f"t{i}") for i in range(1, 6)]
    superset = _Superset(_access({"t1", "t2", "t4", "t5"}, unparsable={"t5"}), datasets)
    with caplog.at_level(logging.INFO, logger="mnemiq.superset"):
        grants = resolve_grants(_adapter(superset), _snapshot(*(f"t{i}" for i in range(1, 6)), "secret"))
    assert grants.objects == {"t1", "t2", "t4"}
    assert superset.probes[2:] == [["t1"], ["t2"], ["t3"], ["t4"], ["t5"]]
    assert "`sales.t3`" in caplog.text and "Unable to parse SQL" in caplog.text
    assert "t3" not in repr(grants) and "t5" not in repr(grants)


@pytest.mark.parametrize("stage", ["all candidates", "a candidate alone"])
def test_permission_denied_on_a_confirmation_probe_is_not_a_builder(stage, caplog):
    denial = SupersetPermissionDenied("Superset's roles for this user do not permit execute_sql", detail=ROLE_DENIAL)
    answers = [_refused(), denial] if stage == "all candidates" else [_refused(), _refused(), denial]
    superset = _Superset(_in_turn(*answers), [_dataset(1, "policy"), _dataset(2, "claim")])
    with caplog.at_level(logging.WARNING, logger="mnemiq.superset"), pytest.raises(NotABuilder) as caught:
        resolve_grants(_adapter(superset), _snapshot("policy", "claim", "party"))
    assert caught.value.message == NOT_A_BUILDER
    assert "sales.party" in caplog.text


CONFIRMATION_FAILURES = {
    "tool error": (SupersetToolError("Superset's execute_sql tool failed", detail="boom"), SupersetMisconfigured),
    "refused caller": (SupersetRefusedCaller("Superset did not accept the credential", detail="401"),
                       SupersetRefusedCaller),
    "generic denial": (SupersetDenied("Superset refused execute_sql and did not say why", detail=GENERIC_DENIAL),
                       SupersetDenied),
}


@pytest.mark.parametrize(("failure", "raised"), CONFIRMATION_FAILURES.values(), ids=CONFIRMATION_FAILURES.keys())
def test_a_confirmation_probe_fails_as_the_first_probe_does(failure, raised):
    superset = _Superset(_in_turn(_refused(), failure), [_dataset(1, "policy")])
    with pytest.raises(raised):
        resolve_grants(_adapter(superset), _snapshot("policy", "claim"))


def test_list_datasets_request_is_pinned():
    """The fields the fallback reads and nothing more, in the database's own order, filtered to the schema; field
    names from `ListDatasetsRequest` and `DatasetFilter` (`superset/mcp_service/dataset/schemas.py`)."""
    adapter = _adapter(_Superset(_refused(), [_dataset(1, "policy")]))
    resolve_grants(adapter, _snapshot("policy"))
    assert adapter.client.calls[1].request == {
        "filters": [{"col": "schema", "opr": "eq", "value": SCHEMA}],
        "select_columns": ["id", "table_name", "schema", "database_id", "sql"],
        "order_column": "id",
        "order_direction": "asc",
        "page": 1,
        "page_size": 100,
    }
    assert PAGE_SIZE == 100


def test_datasets_are_read_page_by_page_until_a_short_page():
    datasets = [_dataset(i, f"t{i}") for i in range(1, 251)]
    superset = _Superset(_access({f"t{i}" for i in range(1, 251)}), datasets)
    adapter = _adapter(superset)
    grants = resolve_grants(adapter, _snapshot(*(f"t{i}" for i in range(1, 251)), "other"))
    assert grants.objects == {f"t{i}" for i in range(1, 251)}
    assert [(r["page"], r["page_size"]) for r, _ in superset.pages] == [(1, 100), (2, 100), (3, 100)]


def test_a_full_last_page_is_followed_by_an_empty_one():
    datasets = [_dataset(i, f"t{i}") for i in range(1, 201)]
    superset = _Superset(_refused(), datasets)
    resolve_grants(_adapter(superset), _snapshot("t1"))
    assert [(r["page"], len(ids)) for r, ids in superset.pages] == [(1, 100), (2, 100), (3, 0)]


@pytest.mark.parametrize("cut", ["error", "flag"])
def test_a_cut_dataset_page_is_asked_again_smaller(cut):
    """A page too large for Superset's response size is asked again for its first datasets with a smaller page, down
    to one dataset, and the read goes on at that size: every dataset is read once, in order, none skipped."""
    big = _dataset(120, "big_view", sql="SELECT " + "x, " * 20_000 + "1")
    datasets = [_dataset(i, f"t{i}") for i in range(1, 231) if i != 120] + [big]
    superset = _Superset(_access({f"t{i}" for i in range(1, 231)} - {"t120"}), datasets, cut_above={120: 5}, cut=cut)
    grants = resolve_grants(_adapter(superset), _snapshot(*(f"t{i}" for i in range(1, 231))))
    assert grants.available is True
    assert grants.objects == {f"t{i}" for i in range(1, 231)} - {"t120"}
    sizes = [r["page_size"] for r, _ in superset.pages]
    assert sizes[0] == 100 and set(sizes[1:]) == {5}
    asked = [(request["page"], request["page_size"]) for request in superset.asked]
    assert asked[:5] == [(1, 100), (2, 100), (3, 50), (5, 25), (21, 5)]


@pytest.mark.parametrize("position", [1, 5, 99, 100, 101, 150, 230])
@pytest.mark.parametrize("fits", [0, 1, 4, 5, 24, 25, 49, 50, 99])
def test_paging_never_skips_or_repeats_a_dataset(position, fits):
    """Whatever page holds a dataset too large to fit, and whatever page size first fits it, the pages answered
    whole carry each dataset exactly once, in order, and each asks for the datasets right after those already read:
    Superset reads page `p` of size `s` as the `s` datasets after the first `(p - 1) * s` (`max(request.page - 1, 0)` in
    `list_datasets`, `offset(page * page_size)` in `BaseDAO.list`)."""
    datasets = [_dataset(i, f"t{i}") for i in range(1, 231)]
    superset = _Superset(_access({f"t{i}" for i in range(1, 231)}), datasets, cut_above={position: fits})
    grants = resolve_grants(_adapter(superset), _snapshot(*(f"t{i}" for i in range(1, 231)), "secret"))
    if fits == 0:
        assert grants is UNAVAILABLE
        return
    read: list[int] = []
    for request, ids in superset.pages:
        assert (request["page"] - 1) * request["page_size"] == len(read)
        read.extend(ids)
    assert read == list(range(1, 231))
    assert grants.objects == {f"t{i}" for i in range(1, 231)}


@pytest.mark.parametrize("cut", ["error", "flag"])
def test_truncated_dataset_page_denies(cut, caplog):
    """A single dataset still too large for Superset's response size cannot be read, so nothing is granted, and the
    grant says it could not be read."""
    datasets = [_dataset(1, "policy"), _dataset(2, "huge", sql="SELECT 1")]
    superset = _Superset(_refused(), datasets, cut_above={2: 0}, cut=cut)
    with caplog.at_level(logging.WARNING, logger="mnemiq.superset"):
        grants = resolve_grants(_adapter(superset), _snapshot("policy"))
    assert grants is UNAVAILABLE
    assert [request["page_size"] for request in superset.asked] == [100, 50, 25, 5, 1, 1]
    assert "list_datasets" in caplog.text


MALFORMED = {
    # What `list_datasets` answers a role without data-model metadata access (`DatasetError`).
    "no datasets": {"error": "You don't have permission to access dataset details for your role.",
                    "error_type": "DataModelMetadataRestricted", "timestamp": "2026-11-12T00:00:00Z"},
    "datasets not a list": {"datasets": {"id": 1}},
    "a dataset not an object": {"datasets": [["policy"]]},
    "virtual marker missing": {"datasets": [{k: v for k, v in _dataset(1, "policy").items() if k != "sql"}]},
    "database missing": {"datasets": [{k: v for k, v in _dataset(1, "policy").items() if k != "database_id"}]},
    "id not a number": {"datasets": [{**_dataset(1, "policy"), "id": "1"}]},
    "database id a flag": {"datasets": [{**_dataset(1, "policy"), "database_id": True}]},
    "table name not text": {"datasets": [{**_dataset(1, "policy"), "table_name": None}]},
    "schema not text": {"datasets": [{**_dataset(1, "policy"), "schema": 5}]},
    "sql not text": {"datasets": [{**_dataset(1, "policy"), "sql": False}]},
    "more than asked": lambda request: {"datasets": [_dataset(i, f"t{i}") for i in range(1, 102)]
                                        if request["page"] == 1 else []},
    "ids out of order": {"datasets": [_dataset(2, "claim"), _dataset(1, "policy")]},
}


@pytest.mark.parametrize("payload", MALFORMED.values(), ids=MALFORMED.keys())
def test_a_malformed_page_denies(payload, caplog):
    """A page the fallback cannot read field by field grants nothing and says the grants could not be read: a dataset
    whose virtual marker is missing is never taken for a physical one."""
    adapter = _adapter(_Superset(_refused(), list_answer=payload))
    with caplog.at_level(logging.WARNING, logger="mnemiq.superset"):
        assert resolve_grants(adapter, _snapshot("policy", "claim", "t1")) is UNAVAILABLE
    assert "list_datasets" in caplog.text


def test_a_page_repeating_one_already_read_denies():
    """Ids ascend across pages in the order asked for, so a page that starts at or before the last id read is not the
    next page."""
    first = [_dataset(i, f"t{i}") for i in range(1, 101)]

    def pages(request: dict) -> dict:
        return {"datasets": first if request["page"] == 1 else first[-1:]}

    adapter = _adapter(_Superset(_refused(), list_answer=pages))
    assert resolve_grants(adapter, _snapshot("t1")) is UNAVAILABLE


def test_query_failed_probe_takes_the_dataset_fallback(caplog):
    """A probe Superset could not parse for access establishes nothing about database-wide access; the person gets the
    datasets Superset lists for them, and the failure is logged."""
    adapter = _adapter(_Superset(_access({"policy"}, unparsable={"claim"}), [_dataset(1, "policy")]))
    with caplog.at_level(logging.WARNING, logger="mnemiq.superset"):
        grants = resolve_grants(adapter, _snapshot("policy", "claim"))
    assert grants.objects == {"policy"}
    assert _tools(adapter) == ["execute_sql", "list_datasets", "execute_sql"]
    assert "Unable to parse SQL" in caplog.text


def test_a_name_mnemiq_would_not_send_is_never_granted():
    """A stored name holding a template marker keeps every dry run naming it from being sent, which establishes
    nothing: the datasets are read, and Superset never confirms that table, so it is not granted."""
    adapter = _adapter(_Superset(DRY_RUN_OK, [_dataset(1, "a{{b")]))
    assert resolve_grants(adapter, _snapshot("a{{b", "claim")).objects == frozenset()
    assert _tools(adapter) == ["list_datasets"]


# --- failures ----------------------------------------------------------------------------------------------------


def test_permission_denied_is_not_a_builder():
    adapter = _adapter(_Superset(SupersetPermissionDenied("Superset's roles for this user do not permit execute_sql",
                                                          detail=ROLE_DENIAL)))
    with pytest.raises(NotABuilder) as caught:
        resolve_grants(adapter, _snapshot("policy"))
    assert caught.value.message == str(caught.value) == NOT_A_BUILDER


@pytest.mark.parametrize("tool", ["execute_sql", "list_datasets"])
def test_a_role_denial_from_the_wire_is_not_a_builder(tool):
    """A role denial as Superset sends it, after its `Error: `, reaches the grant over the real client as a person who
    is not a builder, from the probe and from the listing alike."""
    text = ROLE_DENIAL if tool == "execute_sql" else LIST_ROLE_DENIAL
    with FakeSuperset() as fake:
        fake.reply("execute_sql", lambda request, authorization: PermissionError(text) if tool == "execute_sql"
                   else _refused())
        fake.reply("list_datasets", lambda request, authorization: PermissionError(text))
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok")), database_id=DATABASE_ID,
                                     schema=SCHEMA)
        with pytest.raises(NotABuilder):
            resolve_grants(adapter, _snapshot("policy"))


def test_a_role_denial_on_list_datasets_is_not_a_builder():
    denial = SupersetPermissionDenied("Superset's roles for this user do not permit list_datasets",
                                      detail=LIST_ROLE_DENIAL)
    adapter = _adapter(_Superset(_refused(), list_answer=denial))
    with pytest.raises(NotABuilder) as caught:
        resolve_grants(adapter, _snapshot("policy"))
    assert caught.value.message == NOT_A_BUILDER


def test_an_expired_token_is_not_told_about_roles():
    """A token Superset no longer accepts (401 at the transport) is the caller's, not a role's: it propagates as
    itself, for the probe and for the listing alike."""
    with FakeSuperset() as fake:
        fake.refuse_token("stale")
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("stale")), database_id=DATABASE_ID,
                                     schema=SCHEMA)
        with pytest.raises(SupersetRefusedCaller):
            resolve_grants(adapter, _snapshot("policy"))
    adapter = _adapter(_Superset(_refused(), list_answer=SupersetRefusedCaller("no", detail="401")))
    with pytest.raises(SupersetRefusedCaller):
        resolve_grants(adapter, _snapshot("policy"))


def test_a_generic_denial_is_not_told_about_roles():
    """Superset's generic denial cannot say whether the credential or a resource was refused, so it is not read as
    missing roles: it propagates as `SupersetDenied`."""
    with FakeSuperset() as fake:
        fake.reply("execute_sql", lambda request, authorization: RuntimeError(GENERIC_DENIAL))
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok")), database_id=DATABASE_ID,
                                     schema=SCHEMA)
        with pytest.raises(SupersetDenied) as caught:
            resolve_grants(adapter, _snapshot("policy"))
    assert not isinstance(caught.value, NotABuilder)


@pytest.mark.parametrize("on", ["probe", "list_datasets"])
def test_tool_search_left_on_is_misconfigured(on):
    """With Superset's tool search on, its tools are not offered by name; the message names the setting to change, so
    nobody is told to request roles they may already hold."""
    with FakeSuperset() as fake:
        unknown = RuntimeError(f"Unknown tool: {'execute_sql' if on == 'probe' else 'list_datasets'}")
        fake.reply("execute_sql", lambda request, authorization: unknown if on == "probe" else _refused())
        fake.reply("list_datasets", lambda request, authorization: unknown)
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok")), database_id=DATABASE_ID,
                                     schema=SCHEMA)
        with pytest.raises(SupersetMisconfigured) as caught:
            resolve_grants(adapter, _snapshot("policy"))
    message = caught.value.message
    assert message == str(caught.value)
    assert message.startswith("Superset's MCP service is not set up as mnemiq needs")
    assert 'MCP_TOOL_SEARCH_CONFIG["enabled"] = False' in message
    assert "Alpha" not in message


def test_any_other_tool_error_is_misconfigured(caplog):
    error = SupersetToolError("Superset's list_datasets tool failed", detail="Internal error in list_datasets: boom")
    adapter = _adapter(_Superset(_refused(), list_answer=error))
    with caplog.at_level(logging.WARNING, logger="mnemiq.superset"), pytest.raises(SupersetMisconfigured) as caught:
        resolve_grants(adapter, _snapshot("policy"))
    assert "boom" not in caught.value.message
    assert "boom" in caplog.text


_STATEMENT_FAILURES = {
    "unavailable": SupersetUnavailable("Superset's MCP service could not be reached"),
    "timed out": QueryTimedOut("Superset ended the statement at its time limit"),
    "cancelled": QueryCancelled("Superset reports the statement was stopped"),
}
UNANSWERED = {
    **{f"probe {name}": ("probe", failure) for name, failure in _STATEMENT_FAILURES.items()},
    "listing unavailable": ("list_datasets", SupersetUnavailable("Superset's MCP service answered HTTP 502")),
    **{f"confirmation {name}": ("confirmation", failure) for name, failure in _STATEMENT_FAILURES.items()},
    **{f"candidate alone {name}": ("candidate alone", failure) for name, failure in _STATEMENT_FAILURES.items()},
}


@pytest.mark.parametrize(("on", "failure"), UNANSWERED.values(), ids=UNANSWERED.keys())
def test_unavailable_denies_and_says_so(on, failure):
    """Superset not answering, the first dry run, the listing or any dry run confirming a candidate, grants nothing,
    and the grants say they could not be read rather than that the person may read nothing."""
    datasets = [_dataset(1, "policy"), _dataset(2, "party")]
    probes = {"probe": _in_turn(failure),
              "list_datasets": _in_turn(_refused()),
              "confirmation": _in_turn(_refused(), failure),
              "candidate alone": _in_turn(_refused(), _refused(), failure)}
    superset = _Superset(probes[on], datasets, list_answer=failure if on == "list_datasets" else None)
    grants = resolve_grants(_adapter(superset), _snapshot("policy", "party", "claim"))
    assert grants.available is False
    assert grants.objects == frozenset()


def test_unavailable_from_the_wire_denies_and_says_so():
    with FakeSuperset() as fake:
        fake.fail_http(502)
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("tok")), database_id=DATABASE_ID,
                                     schema=SCHEMA)
        assert resolve_grants(adapter, _snapshot("policy")).available is False


def test_probe_refusal_names_no_table(caplog):
    """Superset's refusal text names a table the person cannot see. It reaches the log, and neither the grants nor any
    exception's message."""
    # (a) A refused probe: the fallback runs, and the text is in the log only.
    adapter = _adapter(_Superset(_access({"policy"}), [_dataset(1, "policy")]))
    with caplog.at_level(logging.WARNING, logger="mnemiq.superset"):
        grants = resolve_grants(adapter, _snapshot("policy", "party"))
    assert grants.objects == {"policy"}
    assert _tools(adapter) == ["execute_sql", "list_datasets", "execute_sql"]
    assert "`sales.party`" in caplog.text
    assert "party" not in repr(grants)
    caplog.clear()
    # (b) A role denial: not a builder, and the denial's own text is in the log only.
    adapter = _adapter(_Superset(SupersetPermissionDenied("Superset's roles for this user do not permit execute_sql",
                                                          detail=ROLE_DENIAL)))
    with caplog.at_level(logging.WARNING, logger="mnemiq.superset"), pytest.raises(NotABuilder) as caught:
        resolve_grants(adapter, _snapshot("policy", "party"))
    assert "party" not in caught.value.message and "party" not in str(caught.value)
    assert "sales.party" in caplog.text


def test_grants_through_the_real_client():
    """Over the real client against the fake Superset: the probe, every page and the confirmation run as the person, a
    page too large for Superset's response size reaches the grant as a page to ask again smaller, and the tables the
    datasets name, once Superset confirms them, become the grant."""
    datasets = [_dataset(i, f"t{i}") for i in range(1, 121)]
    superset = _Superset(_access({f"t{i}" for i in range(1, 121)}), datasets, cut_above={60: 50})

    def list_datasets(request, authorization):
        answer = superset(request)
        # The size guard's own text; the fake puts `Error: ` before it, as Superset does.
        return RuntimeError(TOO_LARGE) if isinstance(answer, SupersetResponseTooLarge) else answer

    with FakeSuperset() as fake:
        fake.reply("execute_sql", lambda request, authorization: superset(request))
        fake.reply("list_datasets", list_datasets)
        adapter = SupersetSqlAdapter(SupersetMcp(fake.url, DelegatedToken("person-token")), database_id=DATABASE_ID,
                                     schema=SCHEMA)
        grants = resolve_grants(adapter, _snapshot(*(f"t{i}" for i in range(1, 121)), "secret"))
    assert grants.objects == {f"t{i}" for i in range(1, 121)}
    assert {call.authorization for call in fake.calls} == {"Bearer person-token"}
    assert [(c.tool, c.request.get("page"), c.request.get("page_size")) for c in fake.calls] == [
        ("execute_sql", None, None), ("list_datasets", 1, 100), ("list_datasets", 1, 50), ("list_datasets", 2, 50),
        ("list_datasets", 3, 50), ("execute_sql", None, None)]
    confirmation = fake.calls[-1].request
    assert confirmation["dry_run"] is True and _names(confirmation["sql"]) == [f"t{i}" for i in range(1, 121)]


# --- the provider ------------------------------------------------------------------------------------------------


def test_fixed_grants_returns_its_grants_for_anyone():
    grants = GrantSet(frozenset({"policy"}), pii_clearance=frozenset({"pii"}))
    provider: AuthzProvider = FixedGrants(grants)
    assert provider.grants_for(IdentityContext(tenant_id="t", principal_id="west", roles=["analyst"])) is grants
    assert provider.grants_for(IdentityContext(tenant_id="t", principal_id="east")) is grants
