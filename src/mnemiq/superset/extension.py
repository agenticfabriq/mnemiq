"""The names of the extension's tools, and the one way to call them.

mnemiq's Superset extension registers its tools on Superset's MCP service under `TOOL_PREFIX`. A Superset that does not
have the extension installed does not say so: rc2's MCP service masks the not-found error, and the client receives only
`Internal error in <tool>: An unexpected error occurred. Error ID: <id>. ...`
(`GlobalErrorHandlerMiddleware._handle_error` in `superset/mcp_service/middleware.py`), the same text an unexpected
exception inside an installed tool produces. Superset's default tool search hides tools from `tools/list`, so the list
cannot settle it either. `require_extension` settles it by one call of a probe: `dataset_metrics` with no datasets, which
the installed extension answers with an empty list and never fails on. `call_tool` asks it once a call fails that way.

A build needs more than an extension: it needs one that makes every object a build makes, which an older one does not.
`require_build_tools` asks for that before a build is planned (`ExtensionOutdated`), and `require_table_rules` and
`require_tables_read` ask for the extension's `table_rules` and `tables_read`, which every statement Superset mode
sends on Oracle needs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from enum import Enum

from mnemiq.superset.client import SupersetToolError, SupersetUnavailable

__all__ = ["DATASET_METRICS_BATCH", "EXTENSION_TOOLS", "PRESENCE_PROBE", "ExtensionMissing", "ExtensionOutdated",
           "SessionAnswer", "TABLE_RULES_BATCH", "TOOL_PREFIX", "call_tool", "dashboard_editorship", "dataset_metrics", "require_build_tools",
           "require_extension", "require_table_rules", "require_tables_read", "row_rule_applies",
           "session_binding", "table_rules_apply", "tables_read"]

logger = logging.getLogger(__name__)

TOOL_PREFIX = "extensions.agenticfabriq.mnemiq."

# The base name of every tool the extension offers; the single source the client and the test fake build names from.
EXTENSION_TOOLS = ("create_dashboard", "create_dataset", "create_chart", "patch_chart", "add_saved_metric",
                   "delete_dataset", "set_layout", "dataset_metrics", "editors_of", "get_layout", "row_rules",
                   "table_rules", "tables_read", "session_binding")

# The most dataset ids one `dataset_metrics` call takes: the extension refuses a request naming more
# (`reads.MAX_DATASETS` in the extension).
DATASET_METRICS_BATCH = 20
# The most table names one `table_rules` call takes: the extension refuses a request naming more
# (`reads.MAX_TABLES` in the extension).
TABLE_RULES_BATCH = 50

# The probe, which the installed extension answers with an empty list and never fails on, without touching Superset.
PRESENCE_PROBE = ("dataset_metrics", {"dataset_ids": []})
# The build's probe: the extension's own create of a dataset, which a build sends and an older extension lacks. This
# one refuses an empty request as a request it cannot read, before it touches Superset.
_BUILD_PROBE = "create_dataset"
_MASKED = "An unexpected error occurred"


class ExtensionMissing(RuntimeError):
    """Superset does not offer the extension's tools. mnemiq's own sentence is the message; what Superset said is in
    `.detail` only."""

    def __init__(self, tool: str, *, detail: str = "") -> None:
        super().__init__(self._said(tool))
        self.detail = detail

    @staticmethod
    def _said(tool: str) -> str:
        return f"Superset does not offer mnemiq's extension tool {tool}: the extension is not installed there"


class ExtensionOutdated(ExtensionMissing):
    """Superset offers the extension, but one older than this mnemiq, without a tool `needed_by` calls (a build, by
    default). Answered as a missing extension is, with its own sentence; what Superset said is in `.detail` only."""

    def __init__(self, tool: str, *, detail: str = "", needed_by: str = "a build") -> None:
        self.needed_by = needed_by
        super().__init__(tool, detail=detail)

    def _said(self, tool: str) -> str:
        return (f"Superset's mnemiq extension is older than this mnemiq and does not offer {tool}, which "
                f"{self.needed_by} calls: update the extension installed there")


def _is_not_offered(tool: str, failure: SupersetToolError) -> bool:
    """Whether `failure` reads as Superset not having `tool`: the masked internal error that names it, or an explicit
    unknown-tool error."""
    message = failure.detail.removeprefix("Error: ")
    return message.startswith(f"Internal error in {tool}: {_MASKED}") or "unknown tool" in message.lower()


def require_extension(client) -> None:
    """Return when Superset offers the extension's tools, as the client's caller sees them; raise `ExtensionMissing`
    when it does not. One call, of the probe. Any other failure is raised as the client raised it."""
    probe, request = TOOL_PREFIX + PRESENCE_PROBE[0], PRESENCE_PROBE[1]
    try:
        client.call(probe, dict(request))
    except SupersetToolError as failure:
        if _is_not_offered(probe, failure):
            raise ExtensionMissing(probe, detail=failure.detail) from None
        raise


def require_build_tools(client) -> None:
    """Return when Superset offers the extension's tools a build calls, as the client's caller sees them. One call,
    of the build's probe: `create_dataset` with an empty request, which this extension refuses (`{"refused": ...}`)
    without touching Superset. A Superset that does not offer it is asked the extension's own probe: with no extension
    at all, `ExtensionMissing`; with one, it is older than this mnemiq (`ExtensionOutdated`). Any answer other than
    the refusal raises `SupersetToolError`, and any other failure is raised as the client raised it."""
    _require_tool(client, _BUILD_PROBE, needed_by="a build")


def require_table_rules(client) -> None:
    """Return when Superset offers the extension's `table_rules` (extension 0.3.0 and later), as `require_build_tools`
    asks for a build's tools: one call of it with an empty request, which it refuses without touching Superset, and
    `ExtensionMissing` or `ExtensionOutdated` when Superset does not offer it."""
    _require_tool(client, "table_rules", needed_by="Superset mode on Oracle")


def require_tables_read(client) -> None:
    """Return when Superset offers the extension's `tables_read` (extension 0.3.0 and later), as `require_table_rules`
    asks for `table_rules`."""
    _require_tool(client, "tables_read", needed_by="Superset mode on Oracle")


def _require_tool(client, base: str, *, needed_by: str) -> None:
    tool = TOOL_PREFIX + base
    try:
        answer = client.call(tool, {})
    except SupersetToolError as failure:
        if not _is_not_offered(tool, failure):
            raise
        require_extension(client)
        raise ExtensionOutdated(tool, detail=failure.detail, needed_by=needed_by) from None
    if not (isinstance(answer, dict) and "refused" in answer):
        raise SupersetToolError(f"the extension's {base} did not refuse an empty request", detail=str(answer)[:200])


def call_tool(client, base: str, request: dict, *, before_call: Callable[[], None] | None = None) -> dict:
    """Call the extension's tool `base` as the client's caller. Raises `ExtensionMissing` when Superset does not offer
    it; any other failure is raised as the client raised it. `before_call` runs before the call, and again before the
    probe that may follow it (a build renews its lease there)."""
    tool = TOOL_PREFIX + base
    if before_call is not None:
        before_call()
    try:
        return client.call(tool, request)
    except SupersetToolError as failure:
        if not _is_not_offered(tool, failure):
            raise
        original = failure
    if before_call is not None:
        before_call()
    try:
        require_extension(client)
    except ExtensionMissing:
        raise ExtensionMissing(tool, detail=original.detail) from None
    except SupersetToolError:
        pass  # the probe failed for its own reason: the tool's failure is what is raised
    raise original


def dataset_metrics(client, dataset_ids: tuple[int, ...]) -> list[dict]:
    """The extension's saved metrics of these datasets, each `{dataset_id, metrics: [...]}`, asked
    `DATASET_METRICS_BATCH` datasets a call, in order. A dataset the caller cannot access is absent, and so is one
    an answer leaves out for any other reason: a caller reads an absent dataset's metrics as unknown."""
    found: list[dict] = []
    ids = list(dataset_ids)
    for start in range(0, len(ids), DATASET_METRICS_BATCH):
        answer = call_tool(client, "dataset_metrics", {"dataset_ids": ids[start:start + DATASET_METRICS_BATCH]})
        found.extend(answer.get("datasets") or [])
    return found


def dashboard_editorship(client, dashboard_id: int, *, before_call: Callable[[], None] | None = None) -> dict | None:
    """Whether the caller edits the dashboard, as the extension's `editors_of` answers it: `{dashboard_id, is_editor,
    editors}` (the editors as labels, for the person's panel only), `{dashboard_id, is_editor, too_large}` when the
    labels were too many to send, or None when the caller may not read the dashboard or it does not exist.
    `before_call` is `call_tool`'s."""
    answer = call_tool(client, "editors_of", {"dashboard_ids": [dashboard_id]}, before_call=before_call)
    entries = answer.get("dashboards") if isinstance(answer, dict) else None
    if not isinstance(entries, list):
        raise SupersetToolError("the extension's editors_of answered no list of dashboards", detail=str(answer)[:200])
    return next((entry for entry in entries if isinstance(entry, dict) and entry.get("dashboard_id") == dashboard_id),
                None)


def row_rule_applies(client, dataset_id: int) -> bool | None:
    """Whether a row rule applies to the caller on the dataset, as the extension's `row_rules` answers it, read
    strictly: the answer must be exactly `{"datasets": [...]}` holding at most this one entry, exactly
    `{"dataset_id": <this id>, "applies": <a boolean>}`. None where mnemiq cannot tell: anything else (a refusal or a
    size cut beside it, a malformed or repeated entry, an entry for another dataset, an entry saying more), a dataset
    left out, or a failure of the call, which is logged. A refusal of the caller is raised, as the client raises it.

    The one reading of `row_rules`, for the value check of a chart on a dataset the build made."""
    try:
        answer = client.call(TOOL_PREFIX + "row_rules", {"dataset_ids": [dataset_id]})
    except (SupersetUnavailable, SupersetToolError) as failure:
        logger.warning("the extension's row_rules for dataset %s failed: %s", dataset_id, failure)
        return None
    entries = answer.get("datasets") if isinstance(answer, dict) and set(answer) == {"datasets"} else None
    if not isinstance(entries, list) or len(entries) != 1:
        return None
    [entry] = entries
    if not isinstance(entry, dict) or set(entry) != {"dataset_id", "applies"} \
            or type(entry["dataset_id"]) is not int or entry["dataset_id"] != dataset_id \
            or not isinstance(entry["applies"], bool):
        return None
    return entry["applies"]


def table_rules_apply(client, *, database_id: int, schema: str, tables: Sequence[str],
                      raise_unavailable: bool = False) -> dict[str, bool | None]:
    """Whether a row rule applies to the caller on each of `tables`, each name in Superset's spelling, in `schema` (in
    Superset's spelling too) on the database, as the extension's `table_rules` answers it: Superset's own lookup of
    the rules a statement reading the table takes. Asked `TABLE_RULES_BATCH` names a call, each name once, in order;
    no call for no name.

    Each answer is read strictly: it must be exactly `{"tables": [...]}` holding one entry for each name the call
    asked, in the order asked, each exactly `{"table": <that name>, "applies": <a boolean>}`. Every name of a call is
    None, where mnemiq cannot tell, when its answer is anything else (a database the caller cannot see, a size cut, a
    refusal, an entry left out, repeated, out of order, for another table or saying more) or the call fails, which is
    logged: an extension older than the tool fails it. With `raise_unavailable`, a call Superset did not answer
    (`SupersetUnavailable`) is raised instead, as the client raises it: Superset not answering is not an answer mnemiq
    cannot read. A refusal of the caller is raised, as the client raises it."""
    names = list(dict.fromkeys(tables))
    found: dict[str, bool | None] = {}
    for start in range(0, len(names), TABLE_RULES_BATCH):
        asked = names[start:start + TABLE_RULES_BATCH]
        found.update(zip(asked, _table_rules(client, database_id, schema, asked, raise_unavailable), strict=True))
    return found


def _table_rules(client, database_id: int, schema: str, asked: list[str],
                 raise_unavailable: bool = False) -> list[bool | None]:
    """One call's answers for `asked`, in order, read as `table_rules_apply` reads them."""
    unknown: list[bool | None] = [None] * len(asked)
    try:
        answer = client.call(TOOL_PREFIX + "table_rules", {"database_id": database_id, "schema": schema,
                                                           "tables": asked})
    except (SupersetUnavailable, SupersetToolError) as failure:
        if raise_unavailable and isinstance(failure, SupersetUnavailable):
            raise
        logger.warning("the extension's table_rules for %d tables failed: %s", len(asked), failure)
        return unknown
    entries = answer.get("tables") if isinstance(answer, dict) and set(answer) == {"tables"} else None
    if not isinstance(entries, list) or len(entries) != len(asked):
        return unknown
    for entry, name in zip(entries, asked, strict=True):
        if not isinstance(entry, dict) or set(entry) != {"table", "applies"} or entry["table"] != name \
                or not isinstance(entry["applies"], bool):
            return unknown
    return [entry["applies"] for entry in entries]


# One table as `tables_read` answers it: its catalog and schema (each a name, or None), and its name.
ReadTable = tuple[str | None, str | None, str]


def tables_read(client, *, database_id: int, schema: str, sql: str) -> frozenset[ReadTable] | None:
    """The tables Superset's own access check reads in `sql`, one statement, as the extension's `tables_read` answers
    as the caller, each `(catalog, schema, table)` as Superset qualifies a table it applies row rules to (the schema
    `schema`, in Superset's spelling, and the database's default catalog, where the statement names none of its own).

    Read strictly: the answer must be exactly `{"tables": [...]}`, each entry exactly `{"catalog", "schema", "table"}`,
    the table a name and the catalog and schema each a name or null. None, where mnemiq cannot tell, for anything else
    (a database the caller cannot see, a text Superset cannot read as one statement, a size cut, a refusal, a
    malformed entry, an entry saying more) and for a tool error, which is logged: an extension older than the tool
    fails it. Superset not answering (`SupersetUnavailable`) and a refusal of the caller are raised, as the client
    raises them."""
    try:
        answer = client.call(TOOL_PREFIX + "tables_read", {"database_id": database_id, "schema": schema, "sql": sql})
    except SupersetToolError as failure:
        logger.warning("the extension's tables_read failed: %s", failure)
        return None
    entries = answer.get("tables") if isinstance(answer, dict) and set(answer) == {"tables"} else None
    if not isinstance(entries, list):
        return None
    found: set[ReadTable] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"catalog", "schema", "table"}:
            return None
        catalog, read_schema, table = entry["catalog"], entry["schema"], entry["table"]
        if not (isinstance(table, str) and table) or not all(part is None or isinstance(part, str) and part
                                                             for part in (catalog, read_schema)):
            return None
        found.add((catalog, read_schema, table))
    return frozenset(found)


class SessionAnswer(Enum):
    """What the extension's `session_binding` answers, read strictly."""

    BOUND = "bound"            # {"bound": true}
    NOT_BOUND = "not bound"    # {"bound": false}
    UNREAD = "unread"          # {"unread": true}, or the tool failing for its own reason
    NOT_ORACLE = "not oracle"  # {"refused": "backend"}
    NOT_SHOWN = "not shown"    # {"not_found": "database"}


# Each answer `session_binding` gives, exactly as it gives it, and what it says.
_SESSION_ANSWERS = (({"bound": True}, SessionAnswer.BOUND), ({"bound": False}, SessionAnswer.NOT_BOUND),
                    ({"unread": True}, SessionAnswer.UNREAD), ({"refused": "backend"}, SessionAnswer.NOT_ORACLE),
                    ({"not_found": "database"}, SessionAnswer.NOT_SHOWN))


def session_binding(client, *, database_id: int, service: str, schema: str) -> SessionAnswer | None:
    """What the extension's `session_binding` says, asked as the client's caller, of the session Superset opens for them
    on the database, as `execute_sql` opens one: whether it logs in as `schema`, keeps it as its current schema and
    reached `service` (both as Oracle stores them), or why it cannot say.

    Read strictly: each `SessionAnswer` only for the answer the tool gives exactly, and None for any other, an answer
    mnemiq cannot read, which is logged. A masked failure of the call is settled as `_require_tool` settles one: the
    tool asked again with an empty request, then the extension's own probe, so an extension without the tool raises
    `ExtensionOutdated` (needed by Superset mode on Oracle) and a Superset without the extension `ExtensionMissing`;
    the tool that is there and failed for its own reason, and any other tool error, is `UNREAD`. Superset not answering
    (`SupersetUnavailable`) and a refusal of the caller are raised, as the client raises them."""
    tool = TOOL_PREFIX + "session_binding"
    try:
        answer = client.call(tool, {"database_id": database_id, "service": service, "schema": schema})
    except SupersetToolError as failure:
        logger.warning("the extension's session_binding failed: %s %s", failure, failure.detail)
        if _is_not_offered(tool, failure):
            _require_tool(client, "session_binding", needed_by="Superset mode on Oracle")
        return SessionAnswer.UNREAD
    for given, kind in _SESSION_ANSWERS:
        if answer == given and type(next(iter(answer.values()))) is type(next(iter(given.values()))):
            return kind
    logger.warning("the extension's session_binding answered %s", str(answer)[:200])
    return None
