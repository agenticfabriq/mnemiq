"""Which snapshot tables a person may read, as Superset decides when asked as them.

mnemiq keeps no grants of its own in Superset mode. It asks Superset in the person's name, through the client the
adapter carries, in two steps:

1. A dry run of one statement naming every table the snapshot has. `execute_sql` checks a statement's access before it
   prepares it, dry run or not (`security_manager.raise_for_access` with `force_dataset_match=True`, in
   `superset/mcp_service/sql_lab/tool/execute_sql.py`). A dry run Superset accepts therefore means the person may read
   every table it names, through access to the whole database or because each table is a dataset granted to them.
2. When Superset refuses that, the physical datasets Superset lists for the person on the configured database and
   schema name candidate tables, and Superset confirms them the same way: a dry run naming the candidates, and, when
   that is refused, one naming each alone. Only a table Superset confirms is granted: the listing only narrows what is
   asked, and the dry run is the access check execution itself makes, so the grant is what a query on the table is
   allowed.

Superset has no column rules for SQL, so whatever the person may read they read raw: every level the snapshot
classifies anything at is cleared. Superset's refusal text can name a table the person cannot see, so it is logged and
never reaches the grants or an exception's message.

A snapshot enriched directly from Oracle was read from rows with the enrichment login (`harvested.rows_read`): its
counts always, and its values where they were harvested. Its grants withhold what was read of every object but each
granted table Superset, asked as the person, says no row rule applies to for them (`GrantSet.values_withheld`): the
extension's `table_rules`, asked through the request's adapter, which answers from what the grant check's own dry runs
were told (`SupersetSqlAdapter.table_rules`). Every dry run on Oracle asks `table_rules` of each table it names, and
the grant check names each table it grants, so this costs no call of its own where the check was answered; the rest
are asked, fifty a call. Asking for every granted table on every request, rather than for those a request goes on to
use, keeps one decision for the whole request: its prompt, its value checks, its history and the schema it shows. A
snapshot enriched through Superset asks nothing more.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from sqlglot import exp

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.authz.grants import UNAVAILABLE, GrantSet, classified_levels
from mnemiq.contract import IdentityContext
from mnemiq.contract.semantic import Snapshot
from mnemiq.sql.identifiers import resolve_stored
from mnemiq.sql.reserved_words import BARE_TABLE_UNSAFE, RESERVED
from mnemiq.superset.adapter import SupersetSqlAdapter
from mnemiq.superset.client import (
    SupersetPermissionDenied,
    SupersetResponseTooLarge,
    SupersetToolError,
    SupersetUnavailable,
)
from mnemiq.superset.extension import TABLES_READ_ANSWER_BYTES
from mnemiq.superset.harvested import rows_read, snapshot_objects
from mnemiq.superset.names import stored_name, superset_name
from mnemiq.superset.payload import SupersetAccessRefused, SupersetQueryFailed
from mnemiq.superset.settings import superset_dialect

logger = logging.getLogger(__name__)

LIST_DATASETS = "list_datasets"

# The share of the bytes `tables_read` answers in that a part of a grant probe is sized to on Oracle (`_probe_parts`),
# and the bytes of that answer beside its tables: its envelope, and the rendering's own words.
_PROBE_SHARE = 0.75
_ANSWER_OVERHEAD = 200

# The datasets asked for on a page until a page is cut: the most `ListDatasetsRequest.page_size` allows
# (`MAX_PAGE_SIZE` in `superset/mcp_service/constants.py`).
PAGE_SIZE = 100

# The fields of a dataset the fallback reads, and the only ones it asks for, so that a page stays small: its id, which
# orders the pages; its table, schema and database; and its SQL, which only a virtual dataset has. Each is a column of
# Superset's dataset model, which `list_datasets` serves by name.
DATASET_FIELDS = ("id", "table_name", "schema", "database_id", "sql")

# A name Postgres reads unquoted as one identifier, and folds: ASCII letters, digits and underscores, not opening with
# a digit.
_PLAIN_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

NOT_A_BUILDER = ("Superset refused this user's queries. Building with mnemiq needs Superset's Alpha and sql_lab "
                 "roles.")
GRANTS_UNREADABLE = ("Superset's answer about what this user may read could not be read, so nothing was answered; "
                     "mnemiq's log records why.")


class _Stated(Exception):
    """A failure whose message, as `message` and as its `str()`, is mnemiq's own sentence."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotABuilder(_Stated):
    """Superset's roles for this person do not permit a tool mnemiq asks with, so they cannot build with mnemiq.

    Raised for that alone: a credential Superset refuses and a denial Superset does not explain propagate as
    themselves, so nobody is told to request roles they may already hold."""

    def __init__(self) -> None:
        super().__init__(NOT_A_BUILDER)


class SupersetMisconfigured(_Stated):
    """Superset's MCP service answered a call with an error that is not a refusal, so it is not set up as mnemiq needs.
    The message carries mnemiq's sentence for the error, which names the setting to change where it knows one."""


class GrantsUnreadable(_Stated):
    """What the person may read could not be read from Superset (`resolve_grants` returned grants that are not
    available), so the request is refused rather than answered under no grants: an empty schema would read as access
    to nothing, and a deferral would blame an access policy Superset mode does not hold. Why was logged where it was
    found; the message is mnemiq's own sentence."""

    def __init__(self) -> None:
        super().__init__(GRANTS_UNREADABLE)


@dataclass(frozen=True)
class FixedGrants:
    """The grants one request was decided under, as the `AuthzProvider` retrieval asks for: the same for any identity,
    since they were decided by asking Superset as the person making that request."""

    grants: GrantSet

    def grants_for(self, identity: IdentityContext) -> GrantSet:
        return self.grants


@dataclass(frozen=True)
class ListedDataset:
    """A dataset as `list_datasets` lists it: its id, its table, schema and database, and its SQL, which only a
    virtual dataset has."""

    id: int
    table_name: str
    schema: str | None
    database_id: int
    sql: str | None


class UnreadableAnswer(Exception):
    """What Superset was asked cannot be read: Superset did not answer, or answered with something mnemiq cannot read.
    Already logged where it was found."""


def resolve_grants(adapter: SupersetSqlAdapter, snapshot: Snapshot) -> GrantSet:
    """The snapshot tables the person whose token `adapter` carries may read, as Superset decides, and the objects whose
    harvested values they may not use (`_values_withheld`).

    The snapshot's tables are its table bindings, each probed by the name the source stores it under and granted by
    its object id. Raises `NotABuilder` when the person's roles do not permit `execute_sql` or `list_datasets`, and
    `SupersetMisconfigured` when Superset's MCP service answers either with any other tool error; a credential Superset
    refuses (`SupersetRefusedCaller`) and a denial it does not explain (`SupersetDenied`) propagate. Returns
    `UNAVAILABLE` when Superset does not answer, when any dry run reaches its time limit or is cancelled, and when the
    datasets cannot be read whole."""
    tables = table_bindings(snapshot)
    if not tables:
        return GrantSet(frozenset(), values_withheld=_values_withheld(adapter, snapshot, frozenset()))
    clearance = classified_levels(snapshot)
    try:
        if all(_accepted(adapter, part) for part in _probe_parts(adapter, list(tables))):
            granted = frozenset(object_id for ids in tables.values() for object_id in ids)
        else:
            datasets = read_datasets(adapter)
            candidates = _candidates(datasets, tables, adapter.database_id, adapter.schema)
            confirmed = _confirmed(adapter, candidates)
            logger.info("granted %d of %d tables: those Superset confirmed of the %d its %d datasets for this user "
                        "name", len(confirmed), len(tables), len(candidates), len(datasets))
            granted = frozenset(object_id for name in confirmed for object_id in tables[name])
        withheld = _values_withheld(adapter, snapshot, granted)
    except UnreadableAnswer:
        return UNAVAILABLE
    return GrantSet(granted, pii_clearance=clearance, values_withheld=withheld)


def _values_withheld(adapter: SupersetSqlAdapter, snapshot: Snapshot, granted: frozenset[str]) -> frozenset[str]:
    """The objects of `snapshot` of which a request of the person whose token `adapter` carries may not show what
    enrichment read from their rows (their harvested values, and their counts): none where enrichment read no row past
    the rules (`rows_read`), and nothing is asked; else every object it names but each one in `granted` that Superset
    says no row rule applies to for the person under every name the snapshot binds it to
    (`SupersetSqlAdapter.table_rules`, each name as Superset shows it, in the configured schema).

    A rule that applies under any of its names, and an answer mnemiq cannot read, withhold it. A table not granted is
    not asked about; nor is an object bound to anything but a table, or bound under a plain lower-case name. None is
    shown. Raises `UnreadableAnswer` when Superset does not answer, as for any question of the grants."""
    if not rows_read(snapshot):
        return frozenset()
    bindings: dict[str, list[tuple[str, str]]] = {}
    for binding in snapshot.source_bindings:
        bindings.setdefault(binding.object_id, []).append((binding.binding_type, binding.source_object))
    names: dict[str, list[str]] = {}
    for object_id, bound in bindings.items():
        if object_id in granted and all(kind == "table" and stored_name(superset_name(stored)) == stored
                                        for kind, stored in bound):
            names[object_id] = [superset_name(stored) for _, stored in bound]
    free: set[str] = set()
    if names:
        try:
            rules = adapter.table_rules(list(dict.fromkeys(name for named in names.values() for name in named)))
        except SupersetUnavailable as exc:
            logger.warning("Superset's row rules got no answer (%s: %s); granting nothing", exc, exc.detail)
            raise UnreadableAnswer from exc
        free = {object_id for object_id, named in names.items() if all(rules.get(name) is False for name in named)}
        logger.info("what was read from rows: %d of %d granted tables shown, where Superset says no row rule applies "
                    "to this user", len(free), len(granted))
    return snapshot_objects(snapshot) - free


def _accepted(adapter: SupersetSqlAdapter, names: list[str]) -> bool:
    """Whether Superset, asked as the person, accepts a dry run naming every table in `names`. A refusal, and a
    statement Superset cannot check (which says nothing about access either way), are both a no, logged.

    Raises `NotABuilder` for a role that does not permit `execute_sql`, `SupersetMisconfigured` for any other tool
    error, and `UnreadableAnswer` when Superset does not answer or the dry run reaches its time limit or is cancelled; a
    refused credential and an unexplained denial propagate."""
    try:
        adapter.dry_run(_probe(names))
    except SupersetAccessRefused as exc:
        # Superset's text, which can name a table, was logged where the payload was read.
        logger.info("Superset refused a dry run naming %d tables for this user: %s", len(names), exc)
        return False
    except SupersetQueryFailed as exc:
        logger.warning("Superset could not check a dry run naming %d tables for this user: %s", len(names), exc)
        return False
    except (SupersetUnavailable, QueryTimedOut, QueryCancelled) as exc:
        logger.warning("a dry run naming %d tables got no answer from Superset (%s %s); granting nothing", len(names),
                       exc, getattr(exc, "detail", ""))
        raise UnreadableAnswer from exc
    except SupersetPermissionDenied as exc:
        raise _not_a_builder(exc) from exc
    except SupersetToolError as exc:
        raise _misconfigured(exc) from exc
    return True


def _confirmed(adapter: SupersetSqlAdapter, candidates: list[str]) -> list[str]:
    """The candidates Superset confirms the person may read, part by part (`_probe_parts`; one part on PostgreSQL):
    all of a part when it accepts a dry run naming them all, else each it accepts a dry run of alone. A single
    candidate refused is not asked about again. Superset's access check is per table, so the parts decide what one
    dry run naming them all would."""
    confirmed = []
    for part in _probe_parts(adapter, candidates):
        if _accepted(adapter, part):
            confirmed += part
        elif len(part) > 1:  # a single candidate was just asked about alone
            confirmed += [name for name in part if _accepted(adapter, [name])]
    for name in candidates:
        if name not in confirmed:
            logger.info("Superset did not confirm %r for this user, so it is not granted", name)
    return confirmed


def _probe_parts(adapter: SupersetSqlAdapter, names: list[str]) -> list[list[str]]:
    """`names` in the parts each one dry run names: on Oracle, as many as keep the answer of the extension's
    `tables_read` of that dry run, which each is held to, within the share `_PROBE_SHARE` of the bytes it answers in
    (`TABLES_READ_ANSWER_BYTES`), each table's bytes estimated from above (`_answer_bytes`), so no answer is cut for
    size and none is refused for it; elsewhere all of them, in one. In order; none empty."""
    if adapter.dialect != "oracle" or not names:
        return [list(names)] if names else []
    budget = TABLES_READ_ANSWER_BYTES * _PROBE_SHARE
    parts: list[list[str]] = [[]]
    size = _ANSWER_OVERHEAD
    for name in names:
        cost = _answer_bytes(name, adapter.schema)
        if parts[-1] and size + cost > budget:
            parts.append([])
            size = _ANSWER_OVERHEAD
        parts[-1].append(name)
        size += cost
    return parts


def _answer_bytes(stored: str, schema: str) -> int:
    """An estimate from above of the bytes the table stored as `stored` adds to `tables_read`'s answer for a probe
    naming it, as the tool measures them (its JSON text, every character past ASCII escaped): its entry, the catalog
    counted as long as the schema, and its place in the rendering with a rule on every table, the table and its alias
    each quoted, the indentation one level deeper than rc2's (`tests/fixtures/superset_rc2_executed.json` holds rc2's
    own sizes, which a test holds this above)."""
    shown = superset_name(stored)
    quoted, alias = (exp.to_identifier(name, quoted=True).sql(dialect="oracle") for name in (shown, stored))
    entry = json.dumps({"catalog": schema, "schema": schema, "table": shown})
    wrap = json.dumps(f"(\n    SELECT\n      *\n    FROM {quoted}\n    WHERE\n      1 = 1\n  ) {alias},\n  ")
    return len(entry) + 2 + len(wrap) - 2


def table_bindings(snapshot: Snapshot) -> dict[str, set[str]]:
    """Each name the snapshot's tables are stored under, in the snapshot's order, with the object ids bound to it."""
    tables: dict[str, set[str]] = {}
    for binding in snapshot.source_bindings:
        if binding.binding_type == "table":
            tables.setdefault(binding.source_object, set()).add(binding.object_id)
    return tables


# The name the live contract test reads it by (`tests/superset_live/test_contract.py`).
_tables = table_bindings


def _probe(names) -> str:
    """One statement naming every table in `names`, each quoted as stored so that it reaches that table and no other,
    and returning no rows. The adapter sends it with each named as Superset names the table (`for_superset`), as
    Superset's access check matches it."""
    tables = ", ".join(exp.to_identifier(name, quoted=True).sql(superset_dialect()) for name in names)
    return f"SELECT 1 FROM {tables} WHERE 1 = 0"


def read_datasets(adapter: SupersetSqlAdapter) -> list[ListedDataset]:
    """Every dataset Superset lists for the person in the configured schema, in id order. Raises `UnreadableAnswer` when
    the list cannot be read whole: Superset does not answer, one dataset alone is too large for its response size, or a
    page does not read as one.

    Superset reads page `p` of size `s` as the `s` datasets after the first `(p - 1) * s`: `list_datasets` passes
    `page - 1` on, and `BaseDAO.list` (`superset/daos/base.py`) offsets by `page * page_size`. A page too large is
    asked again for its first datasets with a smaller page that divides the last one (`_smaller`), so the datasets
    read so far stay a whole number of pages and the next page starts right after them. Once the datasets of the page
    that was too large are read, the read goes back to that page's size: its window ends on a multiple of that size, so
    the next page is aligned too. Pages are read until one holds fewer datasets than it asked for."""
    datasets: list[ListedDataset] = []
    size = PAGE_SIZE
    # For each page that was too large and not yet read through, innermost last: its size, and how many datasets
    # will have been read once its window has.
    windows: list[tuple[int, int]] = []
    while True:
        page = len(datasets) // size + 1
        try:
            payload = adapter.client.call(LIST_DATASETS, _page_request(adapter.schema, page, size))
        except SupersetPermissionDenied as exc:
            raise _not_a_builder(exc) from exc
        except SupersetResponseTooLarge:
            # Superset's response size guard cuts only the tools it names (`INFO_TOOLS`, `DATA_QUERY_TOOLS` and their
            # kin in `superset/mcp_service/utils/response_size_utils.py`), and `list_datasets` is not among them, so a
            # page too large is refused whole rather than sent cut. A page marked `_response_truncated` reads the same.
            cut = True
        except SupersetToolError as exc:
            raise _misconfigured(exc) from exc
        except SupersetUnavailable as exc:
            logger.warning("list_datasets got no answer from Superset (%s: %s); granting nothing", exc, exc.detail)
            raise UnreadableAnswer from exc
        else:
            cut = bool(payload.get("_response_truncated"))
        if cut:
            if size == 1:
                logger.warning("list_datasets: dataset %d alone is too large for Superset's response size; granting "
                               "nothing", len(datasets) + 1)
                raise UnreadableAnswer
            windows.append((size, len(datasets) + size))
            size = _smaller(size)
            continue
        found = _page_datasets(payload, size, datasets[-1].id if datasets else None)
        if found is None:
            logger.warning("list_datasets answered page %d of %d with no page of datasets mnemiq can read (keys %s, "
                           "error_type %s); granting nothing", page, size, sorted(payload), payload.get("error_type"))
            raise UnreadableAnswer
        datasets.extend(found)
        if len(found) < size:
            return datasets
        while windows and len(datasets) == windows[-1][1]:
            size, _ = windows.pop()


def _page_request(schema: str, page: int, size: int) -> dict:
    """`list_datasets`'s request for one page, in the field names of `ListDatasetsRequest` and `DatasetFilter`
    (`superset/mcp_service/dataset/schemas.py`). It asks for the configured schema's datasets only, a filter that can
    only narrow what the fallback compares itself, and orders them by id: by default Superset orders them by when each
    last changed, so a dataset changed during the read would move between pages."""
    return {
        "filters": [{"col": "schema", "opr": "eq", "value": schema}],
        "select_columns": list(DATASET_FIELDS),
        "order_column": "id",
        "order_direction": "asc",
        "page": page,
        "page_size": size,
    }


def _smaller(size: int) -> int:
    """The page size to ask with once a page of `size` was too large: the largest that is at most half of it and
    divides it, so 100, 50, 25, 5, 1. Every size used before divides the datasets read so far, and this one divides
    each of them."""
    return next(smaller for smaller in range(size // 2, 0, -1) if size % smaller == 0)


def _page_datasets(payload: dict, size: int, after: int | None) -> list[ListedDataset] | None:
    """The datasets a page carries; None unless it holds a list of at most `size` datasets, each with every field the
    fallback reads in the type Superset gives it, ids ascending past `after`, the last id already read. A dataset whose
    SQL is missing is unreadable, never taken for a physical one."""
    items = payload.get("datasets")
    if not isinstance(items, list) or len(items) > size:
        return None
    datasets: list[ListedDataset] = []
    for item in items:
        dataset = _dataset(item)
        if dataset is None or (after is not None and dataset.id <= after):
            return None
        datasets.append(dataset)
        after = dataset.id
    return datasets


def _dataset(item) -> ListedDataset | None:
    if not isinstance(item, dict) or any(name not in item for name in DATASET_FIELDS):
        return None
    dataset_id, table_name, schema, database_id, sql = (item[name] for name in DATASET_FIELDS)
    # `type(...) is int`, since a JSON true reads as a bool, which Python also counts an int.
    if type(dataset_id) is not int or type(database_id) is not int or not isinstance(table_name, str):
        return None
    if not (schema is None or isinstance(schema, str)) or not (sql is None or isinstance(sql, str)):
        return None
    return ListedDataset(dataset_id, table_name, schema, database_id, sql)


def _candidates(datasets: list[ListedDataset], tables: dict[str, set[str]], database_id: int, schema: str) -> list[str]:
    """The stored names of the snapshot tables that the physical datasets on the configured database and schema name,
    in the snapshot's order: the tables to ask Superset to confirm, never a grant by themselves. The schema is compared
    as Superset names it, as the configured schema is (`SupersetSqlAdapter.schema`), and as Superset compares it.
    Superset reads a dataset as virtual when its SQL is not empty (`SqlaTable.kind` in
    `superset/connectors/sqla/models.py`); a virtual dataset names its own query's rows, never the tables that query
    reads."""
    named: set[str] = set()
    for dataset in datasets:
        if dataset.database_id == database_id and dataset.schema == schema and not dataset.sql:
            named |= _candidate_names(dataset.table_name)
    return [name for name in tables if name in named]


def _candidate_names(table_name: str) -> set[str]:
    """The stored names a dataset named `table_name` may stand for. On Oracle the one Superset reads for it
    (`stored_name`): a dataset `orders` is the table ORDERS, and a dataset `OrderItems` the table stored so. On
    PostgreSQL, read as Postgres reads a name (`mnemiq.sql.identifiers`): the table stored under exactly that name, as
    a quoted reference reads it, and, where the name is a plain word Postgres reads unquoted, the table it folds to, so
    a dataset `Party` names a stored `party`; any other name names only itself. These are candidates only: the
    fallback asks Superset about every one of them, and grants none on its name alone."""
    if superset_dialect() == "oracle":
        return {stored_name(table_name)}
    names = {table_name}
    if _PLAIN_WORD.fullmatch(table_name) and not _reserved(table_name):
        names.add(resolve_stored(table_name, superset_dialect()))
    return names


def _reserved(word: str) -> bool:
    """Whether any engine mnemiq lists reserves `word` or refuses it as a bare table name. Every engine's list, not
    Postgres's alone, which holds only part of what Postgres refuses there; a word missed only adds a candidate for
    Superset to confirm or refuse."""
    folded = word.casefold()
    return any(folded in words for words in (*RESERVED.values(), *BARE_TABLE_UNSAFE.values()))


def _not_a_builder(exc: SupersetPermissionDenied) -> NotABuilder:
    logger.warning("Superset's roles for this user do not permit a tool mnemiq asks with (%s): %s", exc, exc.detail)
    return NotABuilder()


def _misconfigured(exc: SupersetToolError) -> SupersetMisconfigured:
    logger.warning("Superset's MCP service answered with a tool error (%s): %s", exc, exc.detail)
    # The client's sentence names Superset's MCP service where it is the subject, so this one names Superset alone.
    return SupersetMisconfigured(f"Superset is not set up as mnemiq needs: {exc}")
