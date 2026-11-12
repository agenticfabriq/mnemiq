"""Which snapshot tables a person may read, as Superset decides when asked as them.

mnemiq keeps no grants of its own in Superset mode. It asks Superset in the person's name, through the client the
adapter carries, in two steps:

1. A dry run of one statement naming every table the snapshot has. `execute_sql` checks a statement's access before it
   prepares it, dry run or not (`security_manager.raise_for_access` with `force_dataset_match=True`, in
   `superset/mcp_service/sql_lab/tool/execute_sql.py`). A dry run Superset accepts therefore means the person may read
   every table it names, through access to the whole database or because each table is a dataset granted to them.
2. When Superset refuses that, the tables of the physical datasets Superset lists for the person on the configured
   database and schema, and nothing else.

Superset has no column rules for SQL, so whatever the person may read they read raw: every level the snapshot
classifies anything at is cleared. Superset's refusal text can name a table the person cannot see, so it is logged and
never reaches the grants or an exception's message.
"""

from __future__ import annotations

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
from mnemiq.superset.client import SupersetPermissionDenied, SupersetToolError, SupersetUnavailable
from mnemiq.superset.payload import SupersetAccessRefused, SupersetQueryFailed
from mnemiq.superset.settings import SUPERSET_DIALECT

logger = logging.getLogger(__name__)

LIST_DATASETS = "list_datasets"

# The datasets asked for on a page until a page is cut: the most `ListDatasetsRequest.page_size` allows
# (`MAX_PAGE_SIZE` in `superset/mcp_service/constants.py`).
PAGE_SIZE = 100

# The fields of a dataset the fallback reads, and the only ones it asks for, so that a page stays small: its id, which
# orders the pages; its table, schema and database; and its SQL, which only a virtual dataset has. Each is a column of
# Superset's dataset model, which `list_datasets` serves by name.
DATASET_FIELDS = ("id", "table_name", "schema", "database_id", "sql")

# How the error opens that Superset's response size guard raises for a response too large to send
# (`format_size_limit_error` in `superset/mcp_service/utils/response_size_utils.py`). The guard cuts only the tools it
# names (`INFO_TOOLS`, `DATA_QUERY_TOOLS` and their kin there), and `list_datasets` is not among them, so a page too
# large is refused whole with this error rather than sent cut. A page marked `_response_truncated` reads the same way.
_TOO_LARGE = "Response too large:"

# A name Postgres reads unquoted as one identifier, and folds: ASCII letters, digits and underscores, not opening with
# a digit.
_PLAIN_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

NOT_A_BUILDER = ("Superset refused this user's queries. Building with mnemiq needs Superset's Alpha and sql_lab "
                 "roles.")


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


@dataclass(frozen=True)
class FixedGrants:
    """The grants one request was decided under, as the `AuthzProvider` retrieval asks for: the same for any identity,
    since they were decided by asking Superset as the person making that request."""

    grants: GrantSet

    def grants_for(self, identity: IdentityContext) -> GrantSet:
        return self.grants


@dataclass(frozen=True)
class _Dataset:
    id: int
    table_name: str
    schema: str | None
    database_id: int
    sql: str | None


def resolve_grants(adapter: SupersetSqlAdapter, snapshot: Snapshot) -> GrantSet:
    """The snapshot tables the person whose token `adapter` carries may read, as Superset decides.

    The snapshot's tables are its table bindings, each probed by the name the source stores it under and granted by
    its object id. Raises `NotABuilder` when the person's roles do not permit `execute_sql` or `list_datasets`, and
    `SupersetMisconfigured` when Superset's MCP service answers either with any other tool error; a credential Superset
    refuses (`SupersetRefusedCaller`) and a denial it does not explain (`SupersetDenied`) propagate. Returns
    `UNAVAILABLE` when Superset does not answer, when the probe reaches its time limit or is cancelled, and when the
    datasets cannot be read whole."""
    tables = _tables(snapshot)
    if not tables:
        return GrantSet(frozenset())
    clearance = classified_levels(snapshot)
    try:
        adapter.dry_run(_probe(tables))
    except SupersetAccessRefused:
        # Superset's text, which can name a table, was logged where the payload was read.
        logger.info("Superset refused the probe of %d tables; granting the datasets it lists for this user",
                    len(tables))
    except SupersetQueryFailed as exc:
        # Says nothing about access to the whole database either way.
        logger.warning("Superset could not check the probe of %d tables (%s); granting the datasets it lists for this "
                       "user", len(tables), exc)
    except (SupersetUnavailable, QueryTimedOut, QueryCancelled) as exc:
        logger.warning("the probe of %d tables got no answer from Superset (%s: %s); granting nothing", len(tables),
                       exc, getattr(exc, "detail", ""))
        return UNAVAILABLE
    except SupersetPermissionDenied as exc:
        raise _not_a_builder(exc) from exc
    except SupersetToolError as exc:
        raise _misconfigured(exc) from exc
    else:
        return GrantSet(frozenset(object_id for ids in tables.values() for object_id in ids),
                        pii_clearance=clearance)
    datasets = _read_datasets(adapter)
    if datasets is None:
        return UNAVAILABLE
    granted = _granted(datasets, tables, adapter.database_id, adapter.schema)
    logger.info("granted %d of %d tables through the %d datasets Superset lists for this user", len(granted),
                len(tables), len(datasets))
    return GrantSet(frozenset(granted), pii_clearance=clearance)


def _tables(snapshot: Snapshot) -> dict[str, set[str]]:
    """Each name the snapshot's tables are stored under, in the snapshot's order, with the object ids bound to it."""
    tables: dict[str, set[str]] = {}
    for binding in snapshot.source_bindings:
        if binding.binding_type == "table":
            tables.setdefault(binding.source_object, set()).add(binding.object_id)
    return tables


def _probe(names) -> str:
    """One statement naming every table in `names`, each quoted as stored so that it reaches that table and no other,
    and returning no rows."""
    tables = ", ".join(exp.to_identifier(name, quoted=True).sql(SUPERSET_DIALECT) for name in names)
    return f"SELECT 1 FROM {tables} WHERE 1 = 0"


def _read_datasets(adapter: SupersetSqlAdapter) -> list[_Dataset] | None:
    """Every dataset Superset lists for the person in the configured schema, in id order; None when the list cannot be
    read whole: Superset does not answer, one dataset alone is too large for its response size, or a page does not
    read as one.

    Superset reads page `p` of size `s` as the `s` datasets after the first `(p - 1) * s`: `list_datasets` passes
    `page - 1` on, and `BaseDAO.list` (`superset/daos/base.py`) offsets by `page * page_size`. A page too large is
    asked again for its first datasets with a smaller page that divides the last one (`_smaller`), so the datasets
    read so far stay a whole number of pages and the next page starts right after them, and the read goes on at that
    size. Pages are read until one holds fewer datasets than it asked for."""
    datasets: list[_Dataset] = []
    size = PAGE_SIZE
    while True:
        page = len(datasets) // size + 1
        try:
            payload = adapter.client.call(LIST_DATASETS, _page_request(adapter.schema, page, size))
        except SupersetPermissionDenied as exc:
            raise _not_a_builder(exc) from exc
        except SupersetToolError as exc:
            if not exc.detail.startswith(_TOO_LARGE):
                raise _misconfigured(exc) from exc
            cut = True
        except SupersetUnavailable as exc:
            logger.warning("list_datasets got no answer from Superset (%s: %s); granting nothing", exc, exc.detail)
            return None
        else:
            cut = bool(payload.get("_response_truncated"))
        if cut:
            if size == 1:
                logger.warning("list_datasets: dataset %d alone is too large for Superset's response size; granting "
                               "nothing", len(datasets) + 1)
                return None
            size = _smaller(size)
            continue
        found = _page_datasets(payload, size, datasets[-1].id if datasets else None)
        if found is None:
            logger.warning("list_datasets answered page %d of %d with no page of datasets mnemiq can read (keys %s, "
                           "error_type %s); granting nothing", page, size, sorted(payload), payload.get("error_type"))
            return None
        datasets.extend(found)
        if len(found) < size:
            return datasets


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


def _page_datasets(payload: dict, size: int, after: int | None) -> list[_Dataset] | None:
    """The datasets a page carries; None unless it holds a list of at most `size` datasets, each with every field the
    fallback reads in the type Superset gives it, ids ascending past `after`, the last id already read. A dataset whose
    SQL is missing is unreadable, never taken for a physical one."""
    items = payload.get("datasets")
    if not isinstance(items, list) or len(items) > size:
        return None
    datasets: list[_Dataset] = []
    for item in items:
        dataset = _dataset(item)
        if dataset is None or (after is not None and dataset.id <= after):
            return None
        datasets.append(dataset)
        after = dataset.id
    return datasets


def _dataset(item) -> _Dataset | None:
    if not isinstance(item, dict) or any(name not in item for name in DATASET_FIELDS):
        return None
    dataset_id, table_name, schema, database_id, sql = (item[name] for name in DATASET_FIELDS)
    # `type(...) is int`, since a JSON true reads as a bool, which Python also counts an int.
    if type(dataset_id) is not int or type(database_id) is not int or not isinstance(table_name, str):
        return None
    if not (schema is None or isinstance(schema, str)) or not (sql is None or isinstance(sql, str)):
        return None
    return _Dataset(dataset_id, table_name, schema, database_id, sql)


def _granted(datasets: list[_Dataset], tables: dict[str, set[str]], database_id: int, schema: str) -> set[str]:
    """The object ids of the tables that the physical datasets on the configured database and schema reach. The schema
    is compared as stored, as Superset compares it. Superset reads a dataset as virtual when its SQL is not empty
    (`SqlaTable.kind` in `superset/connectors/sqla/models.py`); a virtual dataset grants its own query's rows, never
    the tables that query reads."""
    granted: set[str] = set()
    for dataset in datasets:
        if dataset.database_id != database_id or dataset.schema != schema or dataset.sql:
            continue
        for name in _names_reached(dataset.table_name):
            granted |= tables.get(name, set())
    return granted


def _names_reached(table_name: str) -> set[str]:
    """The stored names a dataset named `table_name` stands for, read as Postgres reads a name
    (`mnemiq.sql.identifiers`): the table stored under exactly that name, as a quoted reference reads it, and, where
    the name is a plain word Postgres reads unquoted, the table it folds to, so a dataset `Party` stands for a stored
    `party`. Any other name stands only for itself."""
    names = {table_name}
    if _PLAIN_WORD.fullmatch(table_name) and not _reserved(table_name):
        names.add(resolve_stored(table_name, SUPERSET_DIALECT))
    return names


def _reserved(word: str) -> bool:
    """Whether any engine mnemiq lists reserves `word` or refuses it as a bare table name. Every engine's list, not
    Postgres's alone, which holds only part of what Postgres refuses there: a word taken for one Postgres reads
    unquoted grants the table it folds to, so a list too short would grant a table the name cannot reach."""
    folded = word.casefold()
    return any(folded in words for words in (*RESERVED.values(), *BARE_TABLE_UNSAFE.values()))


def _not_a_builder(exc: SupersetPermissionDenied) -> NotABuilder:
    logger.warning("Superset's roles for this user do not permit a tool mnemiq asks with (%s): %s", exc, exc.detail)
    return NotABuilder()


def _misconfigured(exc: SupersetToolError) -> SupersetMisconfigured:
    logger.warning("Superset's MCP service answered with a tool error (%s): %s", exc, exc.detail)
    return SupersetMisconfigured(f"Superset's MCP service is not set up as mnemiq needs: {exc}")
