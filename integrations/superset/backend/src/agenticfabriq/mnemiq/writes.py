"""The extension's write tools, one function per tool: what each reads, checks, writes and answers. `tools.py`
registers them on Superset's MCP service, which runs each as the caller its token names: `mcp_auth_hook`
(`superset/mcp_service/auth.py`) has set `g.user` and checked the tool's class permission and the token's scopes.

Every tool keeps these rules:

- It writes through Superset's commands (`superset/commands/`), never its models.
- A write to an existing row first takes the row's lock (`lock_row`: rc2's own locking read of the row, as its
  version restore takes it) and holds the locked entity until the command returns: the command's `@transaction`
  (`superset/utils/decorators.py`) commits in the same session, and that commit releases the lock. Only under the lock
  does a tool compare the `changed_on` its caller read, so of two racing writers the second waits, then sees the first
  one's `changed_on`, and answers `{"stale": true}`. `{"busy": true}` is for a lock-contention error
  (`superset/versioning/db_errors.py`, `is_lock_contention_error`), a wait past the lock timeout among them
  (`lock_row`), for a statement the metadata database canceled (its `statement_timeout`), and for a metadata
  connection lost mid-statement or not to be had from the pool (`_failed`): the same request may be sent again. A database error outside a command is answered too, by the same rule, never raised
  out of the tool (`_answering`).
- A create carrying a build's marker makes its object once (`_find_or_create`), so a create whose answer was lost may
  be sent again. Once the checks that refuse the caller pass, it takes one lock keyed by the marker, on a connection
  of its own (`_marker_lock`), and holds it from the look for the object whose description's last line is that
  marker through the create's commit: rc2's own checks (a chart's validation and query check) and the create command
  run under it, on a miss only. A second send of the create waits for the first and then answers what it made. On
  Postgres the command's own transaction also takes a second lock, on another key of the marker, and looks again just
  before the command (`_last_look`), so a marker's lock lost with its connection still lets no second object be made.
  A wait past the lock timeout, a lock granted only after mnemiq's deadline for the call, or a look the metadata
  database cannot answer, answers `{"busy": true}`. A SQLite metadata database takes no such lock.
- Its answer is short and structured. Superset's own text never reaches it: any of Superset's exceptions is logged
  here and answered by its kind.
- A caller who is not an editor of the object is told so before anything about the object's state (a stale
  `changed_on`, another build's marker): no retry changes it, and the answer names whom to ask.
- An object the caller may not read is answered as one that does not exist.
- It takes no user id from the caller. The editors a tool sets come from a dashboard's editors, and an object whose
  editors may read data for being its editors (a dataset, in rc2) gets as editors only the caller and the users whose
  grants already reach that data (`_data_editors`), never a role or a group, so that being made an editor widens no
  one's access.

Superset is imported inside the tools, never at module level, so this module imports without it and the request
checks run in mnemiq's unit tests.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agenticfabriq.mnemiq import params

logger = logging.getLogger(__name__)

# The chart types MCP's chart config has none for: `create_chart` takes them as their own form data (`{viz_type,
# params}`), and every other type as MCP's chart config (`{config, set?}`).
EXTENSION_MADE = ("funnel", "sankey_v2")

OK = {"ok": True}
STALE = {"stale": True}
BUSY = {"busy": True}
FAILED = {"failed": True}
EXISTS = {"exists": True}
BAD_MARKER = {"refused": "marker"}

# What opens a Jinja block; Superset renders a stored expression as a template, so none is saved.
_TEMPLATE = re.compile(r"\{[{%#]")
_ISO_4217 = re.compile(r"[A-Z]{3}")
# The keys a create's `set` may carry: the display formats mnemiq's `owed_format_params` makes, which MCP's chart config
# cannot. Nothing `set` carries may change which rows a chart reads or what it is bound to.
_CREATE_SET_KEYS = frozenset({"column_config", "number_format", "currency_format"})
_CHART_ROW_LIMIT_PROBE = 2  # the rows `create_chart`'s check asks for, as MCP's `_compile_chart` asks for two


class Refused(Exception):
    """A request a tool answers without touching Superset; `answer` is what it answers."""

    def __init__(self, answer: dict) -> None:
        super().__init__(answer)
        self.answer = answer


# ---------------------------------------------------------------- reading a request

def _an_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("not a positive integer")
    return value


def _ids(value: Any) -> list[int]:
    if not isinstance(value, list):
        raise ValueError("not a list")
    return [_an_id(item) for item in value]


def _names(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError("not a list of names")
    return list(value)


def _text(limit: int | None) -> Callable[[Any], str]:
    def read(value: Any) -> str:
        if not isinstance(value, str) or not value.strip() or (limit is not None and len(value) > limit):
            raise ValueError("not a short text")
        return value

    return read


def _optional_text(limit: int | None) -> Callable[[Any], str | None]:
    def read(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or (limit is not None and len(value) > limit):
            raise ValueError("not a text")
        return value

    return read


def _a_string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("not a string")
    return value


def _an_object(value: Any) -> dict:
    """A JSON object with text keys, as a copy that shares nothing with the request."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError("not a JSON object")
    return json.loads(json.dumps(value, allow_nan=False))


def _an_instant(value: Any) -> datetime:
    """The `changed_on` a caller read, as MCP's info tools serialise Superset's naive datetime: ISO 8601 text."""
    if not isinstance(value, str):
        raise ValueError("not ISO 8601 text")
    found = datetime.fromisoformat(value)
    if found.tzinfo is not None:
        raise ValueError("Superset's changed_on has no zone")
    return found


def _one_of(choices: tuple[str, ...]) -> Callable[[Any], str]:
    def read(value: Any) -> str:
        if value not in choices:
            raise ValueError("not one of the choices")
        return value

    return read


def _a_currency(value: Any) -> dict | None:
    """rc2's metric currency, `{symbol, symbolPosition}`, with an ISO 4217 code: its renderer drops any other symbol."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"symbol", "symbolPosition"}:
        raise ValueError("not a currency")
    if not isinstance(value["symbol"], str) or not _ISO_4217.fullmatch(value["symbol"]):
        raise ValueError("not an ISO 4217 code")
    if value["symbolPosition"] not in ("prefix", "suffix"):
        raise ValueError("not a symbol position")
    return {"symbol": value["symbol"], "symbolPosition": value["symbolPosition"]}


# The fields of a saved metric the builder may set, each with its reader; the column sizes are `SqlMetric`'s
# (`superset/connectors/sqla/models.py`). Certification (`extra`) is not the builder's to set.
_METRIC_FIELDS: dict[str, Callable[[Any], Any]] = {
    "metric_name": _text(255), "expression": _text(None), "verbose_name": _optional_text(1024),
    "d3format": _optional_text(128), "currency": _a_currency, "description": _optional_text(None)}
_METRIC_REQUIRED = ("metric_name", "expression")
_METRIC_TEXT = ("metric_name", "expression", "verbose_name", "description")


def _a_metric(value: Any) -> dict:
    if not isinstance(value, Mapping) or not set(value) <= set(_METRIC_FIELDS):
        raise ValueError("not a new metric")
    if any(name not in value for name in _METRIC_REQUIRED):
        raise ValueError("a new metric needs a name and an expression")
    metric = {name: _METRIC_FIELDS[name](found) for name, found in value.items()}
    if any(isinstance(metric.get(name), str) and _TEMPLATE.search(metric[name]) for name in _METRIC_TEXT):
        raise Refused({"refused": "template"})
    return metric


@dataclass(frozen=True)
class _Field:
    read: Callable[[Any], Any]
    required: bool = True
    default: Any = None


_DASHBOARD_ID = _Field(_an_id)
_MARKER = _Field(_a_string)
_EXPECT_CHANGED_ON = _Field(_an_instant)

# Every write tool's request, field by field.
REQUESTS: dict[str, dict[str, _Field]] = {
    "create_dashboard": {"title": _Field(_text(500)), "marker": _MARKER},
    "create_dataset": {"database_id": _Field(_an_id), "schema": _Field(_text(255)),
                       "table_name": _Field(_text(250)), "sql": _Field(_text(None), False),
                       "description": _Field(_optional_text(None), False), "dashboard_id": _DASHBOARD_ID,
                       "marker": _MARKER},
    # Exactly one shape, which `create_chart` checks: `{viz_type, params}`, or `{config, set?}`.
    "create_chart": {"dataset_id": _Field(_an_id), "viz_type": _Field(_one_of(EXTENSION_MADE), False),
                     "slice_name": _Field(_text(250)), "params": _Field(_an_object, False),
                     "config": _Field(_an_object, False), "set": _Field(_an_object, False),
                     "dashboard_id": _DASHBOARD_ID, "marker": _MARKER},
    "patch_chart": {"chart_id": _Field(_an_id), "set": _Field(_an_object, False, {}),
                    "unset": _Field(_names, False, []), "expect_changed_on": _EXPECT_CHANGED_ON},
    "add_saved_metric": {"dataset_id": _Field(_an_id), "metric": _Field(_a_metric),
                         "expect_changed_on": _EXPECT_CHANGED_ON},
    "delete_dataset": {"dataset_id": _Field(_an_id), "dashboard_id": _DASHBOARD_ID},
    "set_layout": {"dashboard_id": _DASHBOARD_ID, "positions": _Field(_an_object),
                   "add_chart_ids": _Field(_ids, False, []), "remove_chart_ids": _Field(_ids, False, []),
                   "expect_changed_on": _EXPECT_CHANGED_ON},
}


def read(tool: str, request: Any, requests: Mapping[str, dict[str, _Field]] = REQUESTS) -> dict:
    """`request` read field by field for `tool`, whose fields `requests` names (the write tools' by default; `reads`
    passes its own). Raises `Refused` for a request that is not an object, a missing or misread field, or a field
    the tool does not take (a misspelt one is refused, never ignored)."""
    fields = requests[tool]
    if not isinstance(request, Mapping):
        raise Refused({"refused": "request"})
    unknown = sorted(str(key)[:64] for key in request if key not in fields)
    if unknown:
        raise Refused({"refused": "request", "unknown": unknown[:5]})
    found: dict = {}
    for name, field in fields.items():
        if name not in request:
            if field.required:
                raise Refused({"refused": "request", "field": name})
            found[name] = copy.deepcopy(field.default)
            continue
        try:
            found[name] = field.read(request[name])
        except (TypeError, ValueError):
            raise Refused({"refused": "request", "field": name}) from None
    return found


def claim_marker(text: Any, dashboard_id: int | None) -> params.Marker | None:
    """The marker `text` is, when it is exactly one marker line for `dashboard_id` (None: the dashboard shell's own,
    `new`); else None. Compared field by field, the dashboard as an integer, never as text."""
    if not isinstance(text, str):
        return None
    found = params.parse_marker(text)
    if found is None or found.dashboard_id != dashboard_id:
        return None
    if text != params.format_marker(found.build_id, found.step_key, found.dashboard_id):
        return None
    return found


def with_marker(description: str | None, marker: str) -> str:
    """`description` ending with `marker` as its last line, after its prose. A marker already on the last line is
    replaced, so a description never ends with two."""
    lines = (description or "").rstrip().splitlines()
    if lines and params.parse_marker(lines[-1]) is not None:
        lines.pop()
    prose = "\n".join(lines).rstrip()
    return f"{prose}\n{marker}" if prose else marker


def _answering(tool: str, requests: Mapping[str, dict[str, _Field]] = REQUESTS) -> Callable[
        [Callable[[dict], dict]], Callable[[Any], dict]]:
    """Reads the request for `tool` (its fields in `requests`) before the body runs, and answers a `Refused` with its
    answer.

    A write tool (one `REQUESTS` reads) has one boundary for database errors around its whole body: a SQLAlchemy
    error that reaches it, from anywhere a command does not answer it (the checks before a lock, taking or letting go
    a marker's lock, a bound on lock waits), is answered by `_failed` once the session is rolled back:
    `{"busy": true}` for a lost race or a metadata connection lost or not to be had, where the same request may be
    sent again, else `{"failed": true}`. Never raised out of the tool, which mnemiq would count as the tool failing.
    A lock's own helpers have let it go and closed its connection by then. A read tool raises it as before."""
    writing = tool in REQUESTS

    def wrap(body: Callable[[dict], dict]) -> Callable[[Any], dict]:
        @functools.wraps(body)
        def run(request: Any) -> dict:
            try:
                return body(read(tool, request, requests))
            except Refused as refused:
                return refused.answer
            except Exception as exc:
                from sqlalchemy.exc import SQLAlchemyError

                if not writing or not isinstance(exc, SQLAlchemyError):
                    raise
                with contextlib.suppress(SQLAlchemyError):
                    _session().rollback()
                return _failed(tool, exc)

        return run

    return wrap


# ---------------------------------------------------------------- inside Superset

def _session():
    from superset import db

    return db.session


def _security():
    from superset import security_manager

    return security_manager


def _stored_object(text: str | None) -> dict | None:
    """A stored JSON column read as an object: empty when nothing is stored, None when it is not an object."""
    if not text:
        return {}
    try:
        found = json.loads(text)
    except (TypeError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def _dumps(value: dict | None) -> str | None:
    return None if value is None else json.dumps(value)


def _not_found(kind: str) -> dict:
    return {"not_found": kind}


def _not_editor(resource: Any) -> dict:
    """The answer to a caller who is not an editor of `resource`: its editors' labels, for the person's panel."""
    return {"not_editor": True, "editors": sorted({subject.label for subject in resource.editors})}


def _chain(exc: BaseException | None) -> Iterator[BaseException]:
    """`exc` and each exception it was raised from or while handling, once each: a command reraises the database error
    it caught as its own failure, `from` it (`superset/utils/decorators.py`, `on_error`)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _from_lock_contention(exc: BaseException | None) -> bool:
    """Whether a database lock-contention error is anywhere in `exc`'s chain (`_chain`)."""
    from sqlalchemy.exc import DBAPIError
    from superset.versioning.db_errors import is_lock_contention_error

    return any(isinstance(found, DBAPIError) and is_lock_contention_error(found) for found in _chain(exc))


def _from_lost_connection(exc: BaseException | None) -> bool:
    """Whether `exc`'s chain (`_chain`) holds a metadata connection that could not be had or was lost: no pooled
    connection came free in time (SQLAlchemy's `TimeoutError`), or SQLAlchemy found the connection lost mid-statement
    and invalidated it (a `DBAPIError` with `connection_invalidated`). The write did not happen, or whether it was
    committed is not known."""
    from sqlalchemy.exc import DBAPIError
    from sqlalchemy.exc import TimeoutError as PoolTimeoutError

    return any(isinstance(found, PoolTimeoutError)
               or (isinstance(found, DBAPIError) and getattr(found, "connection_invalidated", False))
               for found in _chain(exc))


def _from_canceled_statement(exc: BaseException | None) -> bool:
    """Whether `exc`'s chain (`_chain`) holds a Postgres statement the database canceled (SQLSTATE 57014,
    query_canceled), as its `statement_timeout` cancels a lock wait longer than it: rc2's contention codes do not
    hold 57014. The SQLSTATE is read from the driver's error as rc2's `is_lock_contention_error` reads it (`pgcode`,
    or `sqlstate`)."""
    from sqlalchemy.exc import DBAPIError

    for found in _chain(exc):
        driver = getattr(found, "orig", None) if isinstance(found, DBAPIError) else None
        if driver is not None and (getattr(driver, "pgcode", None) or getattr(driver, "sqlstate", None)) == "57014":
            return True
    return False


def _failed(tool: str, exc: BaseException) -> dict:
    """The answer for a failure no rule above it names: `{"busy": true}` for a lost lock race (`_from_lock_contention`),
    a statement the database canceled (`_from_canceled_statement`, a lock wait cut short by its `statement_timeout`
    among them) or a metadata connection lost or not to be had (`_from_lost_connection`), where the same request may
    be sent again (a create finds what an earlier send made by its marker); else `{"failed": true}`."""
    if _from_lock_contention(exc):
        logger.info("mnemiq extension: %s lost a lock race", tool)
        return BUSY
    if _from_canceled_statement(exc):
        logger.info("mnemiq extension: %s had a statement canceled by the metadata database", tool)
        return BUSY
    if _from_lost_connection(exc):
        logger.warning("mnemiq extension: %s lost its metadata database connection", tool, exc_info=exc)
        return BUSY
    logger.warning("mnemiq extension: %s failed", tool, exc_info=exc)
    return FAILED


def _command_answer(tool: str, exc: Exception, *, kind: str, editable: Any = None) -> dict:
    """The answer for a command's failure: the object gone, the caller not its editor (with `editable`'s editors, or
    the object as unreadable without one), the command's own validation, or anything else."""
    from superset.commands.chart.exceptions import ChartNotFoundError
    from superset.commands.dataset.exceptions import DatasetNotFoundError
    from superset.commands.exceptions import CommandInvalidError, ForbiddenError, ObjectNotFoundError
    from superset.exceptions import SupersetSecurityException

    if isinstance(exc, (ChartNotFoundError, DatasetNotFoundError, ObjectNotFoundError)):
        return _not_found(kind)
    if isinstance(exc, (ForbiddenError, SupersetSecurityException)):
        return _not_editor(editable) if editable is not None else _not_found(kind)
    if isinstance(exc, CommandInvalidError):
        logger.warning("mnemiq extension: %s refused as invalid: %s", tool, exc.normalized_messages())
        return {"refused": "invalid"}
    return _failed(tool, exc)


def _command_errors() -> tuple[type[Exception], ...]:
    """What a command's failure may be: any of Superset's exceptions (`superset/exceptions.py`), a command's own
    (`CommandException`) and a refused access (`SupersetSecurityException`) among them."""
    from superset.exceptions import SupersetException

    return (SupersetException,)


# How long a write waits for a row another transaction holds, unless Superset's config sets `MNEMIQ_LOCK_TIMEOUT_S`:
# well inside mnemiq's own deadline for a call, so a tool stuck behind a long transaction answers `{"busy": true}`
# while mnemiq still waits for it, rather than writing after mnemiq gave up on the call.
LOCK_TIMEOUT_S = 30
# mnemiq's deadline for a call, in seconds (its default): a lock wait as long as this may end after mnemiq gave up.
CALL_DEADLINE_S = 45
# The clock a create measures its own time by, from the tool's start: monotonic, so a change of the wall clock moves
# nothing.
_clock = time.monotonic


def _lock_timeout_s() -> float:
    """How long a write waits for a lock another transaction holds, in seconds: `MNEMIQ_LOCK_TIMEOUT_S` when that is
    a number above 0 and below `CALL_DEADLINE_S`, else `LOCK_TIMEOUT_S`, with a warning naming the setting. The bound
    also keeps the value finite, a whole number of milliseconds once rounded, and under Postgres' largest
    `lock_timeout` (2147483647 ms), which a larger value would make the write fail on."""
    from flask import current_app

    value = current_app.config.get("MNEMIQ_LOCK_TIMEOUT_S", LOCK_TIMEOUT_S)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < CALL_DEADLINE_S:
        logger.warning("mnemiq extension: MNEMIQ_LOCK_TIMEOUT_S must be a number of seconds above 0 and below %s "
                       "(mnemiq's deadline for a call); waiting %s", CALL_DEADLINE_S, LOCK_TIMEOUT_S)
        value = LOCK_TIMEOUT_S
    return value


def _lock_timeout() -> str:
    """The `lock_timeout` to set, as Postgres reads it: `_lock_timeout_s()` in milliseconds."""
    return f"{max(1, round(_lock_timeout_s() * 1000))}ms"


# `SET LOCAL lock_timeout`, written `set_config(..., true)` so the value is a bound parameter: it holds until the
# transaction it is set in ends.
_SET_LOCK_TIMEOUT = "SELECT set_config('lock_timeout', :timeout, true)"


def _bound_lock_waits(session: Any) -> None:
    """On a Postgres metadata database, bounds each lock wait for the rest of `session`'s transaction
    (`_SET_LOCK_TIMEOUT`). A wait past it fails with SQLSTATE 55P03 (lock_not_available), which rc2's
    `is_lock_contention_error` counts as contention (`superset/versioning/db_errors.py`, `_PG_LOCK_CONTENTION`).
    SQLite takes no row locks, and MySQL has no lock timeout that ends with the transaction: there nothing is set,
    and the database's own setting applies."""
    from sqlalchemy import text

    if session.get_bind().dialect.name == "postgresql":
        session.execute(text(_SET_LOCK_TIMEOUT), {"timeout": _lock_timeout()})


def lock_row(model: type, entity_id: int) -> Any:
    """Row `entity_id` of `model`, locked for update and refreshed to its committed state, or None when no such row is
    visible (a row in the trash is not). A blocking `SELECT ... FOR UPDATE` of the primary-key row, so it serialises
    against every other writer that locks the row, rc2's own dataset `PUT` included.

    On Postgres the wait is bounded first (`_bound_lock_waits`), until the transaction ends, which is the command's
    commit or `_locked`'s rollback. It bounds each lock wait in that transaction on its own, this row's lock and any
    lock a write of the command waits for after it, not the write as a whole. A wait past it fails with SQLSTATE 55P03,
    so the tool answers `{"busy": true}`. SQLite takes no row locks, and MySQL has no lock timeout that ends with the
    transaction: there the database's own setting applies.

    This is rc2's `lock_entity_for_update` (`superset/versioning/api_helpers.py`) with eager loads off, as rc2 reads a
    row it locks for a version restore (`superset/commands/version_restore.py`). `lock_entity_for_update` cannot lock
    a chart on Postgres: `Slice.table` is `lazy="subquery"`, which SQLAlchemy loads with a `SELECT DISTINCT ...
    FOR UPDATE`, and Postgres rejects `FOR UPDATE` with `DISTINCT`. With eager loads off only the row itself is read;
    its relationships (editors, metrics, charts) load when first read, in the same locked transaction. Every locked
    tool takes its lock here, so the rule is one rule."""
    session = _session()
    _bound_lock_waits(session)
    return (session.query(model).populate_existing().enable_eagerloads(False)
            .filter(model.id == entity_id).with_for_update().one_or_none())


def _locked(model: type, entity_id: int, write: Callable[[Any], dict]) -> dict:
    """`write(entity)`'s answer, `entity` being row `entity_id` of `model` as `lock_row` locks it, or None when no
    such row is visible. `write` holds `entity` until its command returns, so the refreshed row stays the one the
    command finds; the command's commit releases the lock. Whatever `write` answers, the session's transaction ends
    here, so a refusal that ran no command releases the lock too."""
    from sqlalchemy.exc import DBAPIError
    from superset.versioning.db_errors import is_lock_contention_error

    try:
        try:
            entity = lock_row(model, entity_id)
        except DBAPIError as exc:
            if not is_lock_contention_error(exc):
                raise
            logger.info("mnemiq extension: the lock on %s %s is contended", model.__name__, entity_id)
            return BUSY
        return write(entity)
    finally:
        _session().rollback()


# ---------------------------------------------------------------- a create carrying a build's marker

class _NotLocked(Exception):
    """The marker's lock was not taken; `answer` is what the create answers instead."""

    def __init__(self, answer: dict) -> None:
        super().__init__(answer)
        self.answer = answer


# SQLAlchemy's names for a MySQL metadata database: MariaDB's dialect is MySQL's, named apart by a `mariadb://` URI.
_MYSQL = ("mysql", "mariadb")
# The marker's lock on Postgres: a session's advisory lock on the marker's hash. A session's lock, unlike a
# transaction's, is held across the commits and rollbacks of the connection it was taken on, until it is released or
# the database session ends.
_PG_LOCK = "SELECT pg_advisory_lock(hashtextextended(:marker, 0))"
_PG_UNLOCK = "SELECT pg_advisory_unlock(hashtextextended(:marker, 0))"
# The second guard on Postgres: the create command's own transaction takes a transaction's advisory lock on another
# key of the marker, so it never waits for the session's lock on the first, and its commit or rollback releases it.
_PG_CREATE_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended('mnemiq-create:' || :marker, 0))"
# The marker's lock on MySQL: a named lock, which belongs to a connection, not a transaction.
_MYSQL_LOCK = "SELECT GET_LOCK(:name, :seconds)"
_MYSQL_UNLOCK = "SELECT RELEASE_LOCK(:name)"


def _like_suffix(marker: str) -> str:
    """A `LIKE` pattern, escaped with a backslash, for a text that ends with `marker`: `%`, then the marker with the
    backslash, `%` and `_` each escaped, so the marker is read literally (a step key's `_` is not any one character)."""
    return "%" + marker.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _marked(model: type, marker: str) -> list[Any]:
    """The rows of `model`, not in the trash, whose description's last line is exactly `marker`, by id. Two filters:
    the database's keeps a description that ends with the marker (`LIKE`, `_like_suffix`; SQLAlchemy writes the
    escape character as each dialect reads it, where a hand-written `ESCAPE '\\'` is an unterminated string to MySQL's
    default mode); Python's then keeps a row only when its last line is the marker alone, so a marker with text typed
    before it on its line is not found, as one with text after it or under it is not."""
    # `populate_existing`: a row the session already holds (an earlier read in the same session) is read as it is now.
    rows = (_session().query(model).populate_existing()
            .filter(model.description.like(_like_suffix(marker), escape="\\"), model.deleted_at.is_(None))
            .order_by(model.id))
    return [row for row in rows if (row.description or "").splitlines()[-1:] == [marker]]


def _isolation_refusal(session: Any) -> dict | None:
    """`{"refused": "isolation"}` when `session` reads at another level than READ COMMITTED on a Postgres or MySQL
    metadata database (MariaDB's dialect included), logged; else None. SQLite has no level to check.

    The look must see an object the moment another create commits it, which READ COMMITTED gives: rc2 sets it unless
    `SQLALCHEMY_ENGINE_OPTIONS` names another level, and only for a `postgresql` or a `mysql` backend, so not for a
    `mariadb://` URI (`SupersetAppInitializer.set_db_default_isolation`). Under a level whose snapshot is fixed at the
    transaction's first read, which may come before the lock, a create could miss the object another one made, and
    under READ UNCOMMITTED it could find one another create has written but may still roll back. So the session's own
    level is read, on its own connection, and any other level is refused."""
    if session.get_bind().dialect.name == "sqlite":
        return None
    level = session.connection().get_isolation_level()
    if not isinstance(level, str) or level.upper() != "READ COMMITTED":
        logger.warning("mnemiq extension: a create carrying a marker needs the metadata database's READ COMMITTED "
                       "isolation, not %s", level)
        return {"refused": "isolation"}
    return None


def _lock_name(marker: str) -> str:
    """MySQL's name for `marker`'s lock."""
    return "mnemiq:" + hashlib.sha256(marker.encode()).hexdigest()[:56]


def _take_marker_lock(connection: Any, dialect: str, marker: str) -> None:
    """Takes `marker`'s lock on `connection`, in a short transaction of its own, committed once the lock is held (a
    session's lock and a named lock both outlive it). On Postgres, a session's advisory lock on the marker's hash, its
    wait bounded first as `lock_row`'s is (`_SET_LOCK_TIMEOUT`, which ends with that transaction), so a wait past the
    lock timeout fails with SQLSTATE 55P03. On MySQL, a named lock: `mnemiq:` and the marker's SHA-256, cut to 63
    characters (MySQL allows 64), waited for `_lock_timeout_s()` in whole seconds, rounded down so the wait stays
    inside the setting's bound, and at least one; `GET_LOCK` answers 1 when it holds the lock (0 when its wait timed
    out, NULL on an error).

    A lock not taken in time, or contended (`is_lock_contention_error`), raises `_NotLocked` with `{"busy": true}`;
    any other failure is raised. Whatever stops the take, the connection is invalidated first: SQLAlchemy then closes
    it rather than putting it back in the pool, and the database session ending ends any lock it may hold. A failed
    take may hold one: Postgres can grant a waited-for lock just as the wait's lock timeout fires and still fail the
    statement with 55P03 (`LockErrorCleanup` keeps a lock granted while it waited), and the transaction's rollback
    does not end a session's lock; a commit can fail after the lock was taken."""
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError
    from superset.versioning.db_errors import is_lock_contention_error

    try:
        with connection.begin():
            if dialect == "postgresql":
                connection.execute(text(_SET_LOCK_TIMEOUT), {"timeout": _lock_timeout()})
                connection.execute(text(_PG_LOCK), {"marker": marker})
            elif connection.execute(text(_MYSQL_LOCK), {"name": _lock_name(marker),
                                                        "seconds": max(1, int(_lock_timeout_s()))}).scalar() != 1:
                raise _NotLocked(BUSY)
    except BaseException as exc:
        connection.invalidate()
        if isinstance(exc, _NotLocked):
            logger.info("mnemiq extension: the lock on a marker is contended")
            raise
        if isinstance(exc, DBAPIError) and is_lock_contention_error(exc):
            logger.info("mnemiq extension: the lock on a marker is contended")
            raise _NotLocked(BUSY) from exc
        raise


def _release_marker_lock(connection: Any, dialect: str, marker: str) -> None:
    """Releases `marker`'s lock on `connection`. A session's lock outlives the rollback a pool's reset makes, so a
    release that raises, or does not answer that it released the lock, invalidates the connection: SQLAlchemy closes
    it rather than putting it back in the pool, and the database session ending ends the lock. Never raises, so an
    exception the locked work raised is the one that goes on."""
    from sqlalchemy import text

    statement, key = ((_PG_UNLOCK, {"marker": marker}) if dialect == "postgresql"
                      else (_MYSQL_UNLOCK, {"name": _lock_name(marker)}))
    try:
        released = connection.execute(text(statement), key).scalar()
    except Exception as exc:  # whatever stopped the release, the connection must not go back to the pool holding it
        logger.warning("mnemiq extension: a marker's lock was not released; its connection is closed", exc_info=exc)
        connection.invalidate()
        return
    if released != 1:  # Postgres answers true, MySQL 1
        logger.warning("mnemiq extension: a marker's lock was not released (%r); its connection is closed", released)
        connection.invalidate()


@contextlib.contextmanager
def _marker_lock(marker: str, started: float) -> Iterator[None]:
    """Holds the lock keyed by `marker` over the block, so that of two creates carrying one marker the second looks
    for the object only once the first has committed it or given up. The lock is taken on a connection of its own
    (`engine.connect()`), never the session's: a session's lock on Postgres, a named lock on MySQL
    (`_take_marker_lock`), each held across every commit and rollback the block makes in the session, until it is
    released after the block (`_release_marker_lock`), when the block raises too, and the connection closed. Raises
    `_NotLocked` when the lock is not taken: `{"busy": true}` for a wait past the lock timeout, or when no connection
    comes free in the pool in time (SQLAlchemy's `TimeoutError`).

    A lock granted more than `CALL_DEADLINE_S` after `started` (the tool's start, on `_clock`) is let go at once,
    with `{"busy": true}` and the block not run: mnemiq has stopped waiting for this answer, and its own retry, sent
    the same, decides.

    - SQLite: no lock. A SQLite metadata database is Superset's single-person development default, where a create sent
      again while its own lost first one still runs does not happen in practice.
    - Any other metadata database than Postgres, MySQL or SQLite: refused (`{"refused": "metadata_database"}`), rather
      than created on unlocked."""
    from sqlalchemy.exc import TimeoutError as PoolTimeoutError

    engine = _session().get_bind()
    dialect = engine.dialect.name
    if dialect == "sqlite":
        yield
        return
    if dialect != "postgresql" and dialect not in _MYSQL:
        logger.warning("mnemiq extension: a create carrying a marker takes no lock on a %s metadata database", dialect)
        raise _NotLocked({"refused": "metadata_database"})
    try:
        connection = engine.connect()
    except PoolTimeoutError:
        logger.info("mnemiq extension: no connection came free to take a marker's lock on")
        raise _NotLocked(BUSY) from None
    try:
        _take_marker_lock(connection, dialect, marker)
        try:
            if _clock() - started > CALL_DEADLINE_S:
                logger.info("mnemiq extension: a marker's lock came free after mnemiq's deadline for the call")
                raise _NotLocked(BUSY)
            yield
        finally:
            _release_marker_lock(connection, dialect, marker)
    finally:
        connection.close()


def _found_answer(found: list[Any], *, kind: str) -> dict:
    """The answer for the objects of `kind` a lookup found carrying a marker (`_marked`, at least one). One is answered
    `{"<kind>_id": id}` when the caller may read it and edits it, else as one that does not exist or with its editors
    (`_not_editor`). More than one stops the create with `{"candidates": [ids]}`, for the build to name: the ids of
    those the caller may read, the others answered as objects that do not exist, so the list may be empty."""
    security = _security()
    readable = {"dashboard": security.can_access_dashboard, "chart": security.can_access_chart,
                "dataset": security.can_access_datasource}[kind]
    if len(found) > 1:
        return {"candidates": [row.id for row in found if readable(row)]}
    (row,) = found
    if not readable(row):
        return _not_found(kind)
    if not security.is_editor(row):
        return _not_editor(row)
    return {f"{kind}_id": row.id}


def _look(model: type, marker: str, *, kind: str) -> dict | None:
    """Under the marker's lock: the answer for the objects of `model` that `marker` names (`_marked`, answered by
    `_found_answer`), or None when there is none, once the session's isolation is the one the look needs
    (`_isolation_refusal`, whose refusal is the answer otherwise). Before it looks, the session's own lock waits are
    bounded as `lock_row` bounds them (`_bound_lock_waits`), for the rest of its transaction: a statement of the
    create's that waits for a row another transaction holds then fails with 55P03, answered `{"busy": true}`, rather
    than waiting unbounded while the marker's lock is held.

    A look the metadata database cannot answer (its connection gone, the session's transaction aborted) answers
    `{"busy": true}`, after rolling the session back, which ends an aborted transaction: whether the object is there
    is not known, and the same request may be sent again. Never a refusal, which would stop the build though another
    send may have made the object."""
    from sqlalchemy.exc import SQLAlchemyError

    session = _session()
    try:
        if (refusal := _isolation_refusal(session)) is not None:
            return refusal
        _bound_lock_waits(session)
        found = _marked(model, marker)
        return _found_answer(found, kind=kind) if found else None
    except SQLAlchemyError as exc:
        logger.warning("mnemiq extension: a create could not look for its object; answered busy", exc_info=exc)
        with contextlib.suppress(SQLAlchemyError):
            session.rollback()
        return BUSY


def _last_look(model: type, marker: str, *, kind: str) -> dict | None:
    """On Postgres, just before a create's command and in the command's own transaction: the answer for the objects
    of `model` that `marker` names, or None when there is none, looked for again under a second lock.

    The marker's lock is a session's lock on a connection of its own, which sits idle while the create works; if that
    database session ends, the lock goes with it, while the create's own session works on, and a second send of the
    create may take the lock, find nothing and make the object. So the session's lock waits are bounded again
    (`_bound_lock_waits`: rc2's checks may have ended the transaction the look bounded), the transaction takes an
    advisory lock on another key of the marker (`_PG_CREATE_LOCK`), held until the command's commit or rollback, and
    looks again (`_marked`, `_found_answer`), in a fresh READ COMMITTED snapshot. Of two creates that both passed the
    look, the second to take this lock finds the first's object, committed, and answers it. Between this and the
    command nothing commits or rolls back the session. MySQL and SQLite take no second lock: None.

    Any database error here (a wait past the lock timeout, a connection gone, a statement that failed) answers
    `{"busy": true}`, as the first look's does (`_look`), after rolling the session back, which ends the transaction
    and releases this lock with nothing made: whether another send made the object is not known, and a refusal would
    stop the build though it may have."""
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    session = _session()
    if session.get_bind().dialect.name != "postgresql":
        return None
    try:
        _bound_lock_waits(session)
        session.execute(text(_PG_CREATE_LOCK), {"marker": marker})
        found = _marked(model, marker)
        return _found_answer(found, kind=kind) if found else None
    except SQLAlchemyError as exc:
        logger.warning("mnemiq extension: a create could not look for its object before its command; answered busy",
                       exc_info=exc)
        with contextlib.suppress(SQLAlchemyError):
            session.rollback()
        return BUSY


def _find_or_create(model: type, marker: str, *, kind: str, started: float, create: Callable[[], dict],
                    then: Callable[[dict], dict] | None = None) -> dict:
    """The object of `model` that `marker` names, or `create()`'s answer when there is none, all under the marker's
    lock (`_marker_lock`): a create whose answer was lost, sent again, answers the object the first one made instead
    of making a second, and a second send while the first still runs waits for the first and then answers what it
    made. `then(answer)` runs under the lock too, after the object was found or made (a dataset's editors).

    The caller's own checks run before this (its request, the objects it names, being an editor of the dashboard), so a
    caller they refuse never waits for the lock. Under it run the look (`_look`) and, on a miss, `create()`: rc2's own
    checks (a chart's validation, mapping and query check), on Postgres the last look in the command's own transaction
    (`_last_look`, which `create()` calls just before its command), the create command and its commit. The lock is on a
    connection of its own, so whatever rc2 commits or rolls back in the session keeps it held, and no other send of the
    create can make the object while a check of this one refuses it: that refusal is answered as it is. Should that
    connection's database session end while the create works, the lock goes with it; the last look then still finds an
    object another send made meanwhile, and the lost connection is closed when the lock is let go
    (`_release_marker_lock`). The session's transaction ends before the lock is let go. A lock not taken answers
    `{"busy": true}` or the refusal that says why, and so does one granted after mnemiq's deadline for the call, counted
    from `started`, the tool's start."""
    try:
        with _marker_lock(marker, started):
            try:
                answer = _look(model, marker, kind=kind)
                if answer is None:
                    answer = create()
                return then(answer) if then is not None else answer
            finally:
                _session().rollback()
    except _NotLocked as not_locked:
        _session().rollback()
        return not_locked.answer


def _dashboard(dashboard_id: int) -> Any:
    """The dashboard, when the caller may read it; else None."""
    from superset.daos.dashboard import DashboardDAO

    dashboard = DashboardDAO.find_by_id(dashboard_id)
    if dashboard is None or not _security().can_access_dashboard(dashboard):
        return None
    return dashboard


def _dataset(dataset_id: int) -> Any:
    from superset.daos.dataset import DatasetDAO

    dataset = DatasetDAO.find_by_id(dataset_id)
    if dataset is None or not _security().can_access_datasource(dataset):
        return None
    return dataset


def _readable_chart(chart_id: int) -> bool:
    from superset.daos.chart import ChartDAO

    chart = ChartDAO.find_by_id(chart_id)
    return chart is not None and _security().can_access_chart(chart)


def _editor_ids(dashboard: Any) -> list[int]:
    return [subject.id for subject in dashboard.editors]


@contextlib.contextmanager
def _as_user(user: Any) -> Iterator[None]:
    """`g.user` is `user` inside the block, and the request's own user again after it, also when the block raises:
    rc2's `override_user` (`superset/utils/core.py`) puts the user back only when its block returns."""
    from flask import g

    had, previous = hasattr(g, "user"), getattr(g, "user", None)
    g.user = user
    try:
        yield
    finally:
        if had:
            g.user = previous
        else:
            del g.user


def _reads_by_a_grant(dataset: Any) -> bool:
    """Whether a grant `g.user`'s roles hold already reaches `dataset` and the table under it, wherever rc2 checks
    (`superset/security/manager.py`, `raise_for_access`): `can_access_database` (all-datasource, all-database or
    database access) or `datasource_access` on the dataset's own `perm`. These are the grants SQL Lab's strict check
    of a table admits (`force_dataset_match`, as `Query.raise_for_access` and MCP's `execute_sql` ask it), besides
    being an editor of a dataset over the table. Catalog and schema access read the dataset but are skipped there,
    so one of them alone does not count. Nor do being an editor; the legs a chart's or a dashboard's request opens
    (an embedded guest, a viewer in promiscuous mode); a semantic layer's grant, which covers semantic views only;
    and a deployment's `EXTRA_RAISE_FOR_ACCESS_BYPASS`, whose reach the extension cannot know."""
    security = _security()
    return bool(security.can_access_database(dataset.database)
                or security.can_access("datasource_access", dataset.perm or ""))


def _data_editors(dataset: Any, dashboard: Any, kept: Iterable[Any]) -> list[int]:
    """The editor subjects for an object whose editors may read `dataset`'s data. rc2 admits a dataset's editors to
    its data (`raise_for_access`) and, in SQL Lab, to the table under it, and a role or a group made an editor makes
    every member one. So: every subject in `kept` (the object's editors now, whoever added them), the caller's own
    user subject, and each user among `dashboard`'s editors whose grants already reach `dataset` and its table,
    judged as that user (`_reads_by_a_grant`). Never a role or a group."""
    from superset.subjects.types import SubjectType
    from superset.subjects.utils import get_user_subject
    from superset.utils.core import get_user_id

    chosen = [subject.id for subject in kept]
    caller = get_user_subject(user_id) if (user_id := get_user_id()) is not None else None
    if caller is not None and caller.id not in chosen:
        chosen.append(caller.id)
    for subject in dashboard.editors:
        if dataset is None or subject.id in chosen or subject.type != SubjectType.USER or subject.user is None:
            continue
        with _as_user(subject.user):
            readable = _reads_by_a_grant(dataset)
        if readable:
            chosen.append(subject.id)
    return chosen


def _chart_data_by_editorship() -> bool:
    """Whether a chart's editors may read its dataset's data for being its editors: rc2 admits a chart's editors and
    viewers to it when `ENABLE_VIEWERS` is on and `VIEWER_PROMISCUOUS_MODE` is set (`raise_for_access`,
    `has_promiscuous_chart_access`)."""
    from flask import current_app
    from superset import is_feature_enabled

    return bool(is_feature_enabled("ENABLE_VIEWERS") and current_app.config.get("VIEWER_PROMISCUOUS_MODE"))


def _chart_editors(dataset: Any, dashboard: Any, kept: Iterable[Any]) -> list[int]:
    """The editor subjects for a chart on `dataset` made for `dashboard`: the dashboard's editors, unless a chart's
    editors may read its data for being its editors (`_chart_data_by_editorship`); then they are chosen as a dataset's
    are (`_data_editors`), `kept` being the chart's editors now."""
    if _chart_data_by_editorship():
        return _data_editors(dataset, dashboard, kept)
    return _editor_ids(dashboard)


def _claimed_dataset_editors(dataset: Any, dashboard: Any) -> list[int]:
    """A dataset's editors for `dashboard`: those it has, and whom `_data_editors` adds."""
    return _data_editors(dataset, dashboard, dataset.editors)


# ---------------------------------------------------------------- the tools

@_answering("create_dashboard")
def create_dashboard(found: dict) -> dict:
    """An unpublished dashboard with an empty layout, the caller as its editor, and the build's marker as its
    description, made once for the marker: under the marker's lock, held from the look through the create's commit
    (`_find_or_create`), so sent again, it answers the dashboard the first one made, waiting for it while it is being
    made. The marker is the shell's own (`new`): the dashboard has no id until this returns."""
    started = _clock()
    if claim_marker(found["marker"], None) is None:
        return BAD_MARKER
    from superset.commands.dashboard.create import CreateDashboardCommand
    from superset.models.dashboard import Dashboard

    def create() -> dict:
        if (made := _last_look(Dashboard, found["marker"], kind="dashboard")) is not None:
            return made
        try:
            dashboard = CreateDashboardCommand({"dashboard_title": found["title"], "published": False,
                                                "description": found["marker"]}).run()
        except _command_errors() as exc:
            return _command_answer("create_dashboard", exc, kind="dashboard")
        return {"dashboard_id": dashboard.id}

    return _find_or_create(Dashboard, found["marker"], kind="dashboard", started=started, create=create)


def _dataset_creation_answer(exc: Exception) -> dict:
    """The answer for a dataset create's failure (one with `DatasetSoftDeletedTwinExistsError` is `_twin_answer`'s). A
    caller who may not read the table, or what a virtual dataset's SQL reads (`DatasetDataAccessIsNotAllowed` among
    the errors listed), is answered `{"not_found": "dataset"}`, with a dataset over the table or none. A table a
    dataset already holds (`DatasetExistsValidationError`) is answered `{"exists": true}` only to a caller Superset's
    own access check admits; anything else as `_command_answer` answers it."""
    from superset.commands.dataset.exceptions import DatasetInvalidError

    listed = exc.get_list_classnames() if isinstance(exc, DatasetInvalidError) else []
    if "DatasetDataAccessIsNotAllowed" in listed:
        return _not_found("dataset")
    if "DatasetExistsValidationError" in listed:
        return EXISTS
    return _command_answer("create_dataset", exc, kind="dataset")


def _twin_answer(command: Any) -> dict:
    """The answer for a dataset's create refused with `DatasetSoftDeletedTwinExistsError`: a dataset in the trash holds
    its table. As for a live dataset over the table, the answer depends on whether the caller may read the table, or
    what a virtual dataset's SQL reads: `raise_for_access` is asked as the caller, with the arguments
    `CreateDatasetCommand.validate` gives it (the table, or `sql` with its catalog and schema), on what `validate`
    resolved into the command's properties (the database read from its id, the default catalog when none was given,
    a schema read from the SQL). A caller it refuses is answered `{"not_found": "dataset"}`, and SQL it cannot parse
    `{"refused": "invalid"}`, as the same request is answered with no dataset in the trash; `{"exists": true}` only
    to a caller it admits. A database failure while it checks is answered by `_failed`."""
    from sqlalchemy.exc import SQLAlchemyError
    from superset.exceptions import SupersetParseError, SupersetSecurityException
    from superset.sql.parse import Table

    resolved = command._properties
    database, catalog, schema, sql = (resolved.get(key) for key in ("database", "catalog", "schema", "sql"))
    # No database to ask about: answered as no dataset, never as an `exists` no check admitted.
    if database is None:
        return _not_found("dataset")
    try:
        if sql:
            _security().raise_for_access(database=database, sql=sql, catalog=catalog, schema=schema)
        else:
            _security().raise_for_access(database=database, table=Table(resolved["table_name"], schema, catalog))
    except SupersetSecurityException:
        return _not_found("dataset")
    except SupersetParseError:
        logger.warning("mnemiq extension: create_dataset refused as invalid: its SQL does not parse")
        return {"refused": "invalid"}
    # The check reads the metadata database: a failure there is answered as the create's own database failures are.
    except (*_command_errors(), SQLAlchemyError) as exc:
        return _failed("create_dataset", exc)
    return EXISTS


def _set_dataset_editors(found: dict, dataset_id: int) -> dict:
    """The dataset's editors, set under its row lock (`_claimed_dataset_editors`): those it has, the caller, and the
    dashboard's users whose grants already reach it. `OK`, or why not."""
    from superset.commands.dataset.update import UpdateDatasetCommand
    from superset.connectors.sqla.models import SqlaTable

    def write(dataset: Any) -> dict:
        if dataset is None or not _security().can_access_datasource(dataset):
            return _not_found("dataset")
        dashboard = _dashboard(found["dashboard_id"])
        if dashboard is None:
            return _not_found("dashboard")
        try:
            UpdateDatasetCommand(dataset.id, {"editors": _claimed_dataset_editors(dataset, dashboard)}).run()
        except _command_errors() as exc:
            return _command_answer("create_dataset", exc, kind="dataset", editable=dataset)
        return OK

    return _locked(SqlaTable, dataset_id, write)


@_answering("create_dataset")
def create_dataset(found: dict) -> dict:
    """A dataset over a table, or over `sql` when it is given (a virtual dataset), for the dashboard: `description`'s
    prose and the build's marker as its last line, in `CreateDatasetCommand`'s own transaction, once for the marker:
    under the marker's lock, held from the look through the create's commit and the editors step
    (`_find_or_create`), so sent again, it answers the dataset the first one made, waiting for it while it is being
    made, before rc2 can refuse the table as already holding one. A caller who may not read the table (or what a
    virtual dataset's SQL reads) is answered `{"not_found": "dataset"}`, with a dataset over the table, one in the
    trash, or none; a table that already has a dataset not made for this marker, live or in the trash, is answered
    `{"exists": true}` only to a caller Superset's own access check admits (`_dataset_creation_answer`, and
    `_twin_answer` after the marker's lock is released). The dashboard must be readable and edited by the caller,
    which is checked before the lock. After the create commits, and on the found path, the dataset's editors are set
    under its row lock (`_set_dataset_editors`), still under the marker's lock; if that answers anything but `OK`, so
    does this tool, and a retry finds the marked dataset and sets them again."""
    started = _clock()
    if claim_marker(found["marker"], found["dashboard_id"]) is None:
        return BAD_MARKER
    dashboard = _dashboard(found["dashboard_id"])
    if dashboard is None:
        return _not_found("dashboard")
    if not _security().is_editor(dashboard):
        return _not_editor(dashboard)
    from superset.commands.dataset.create import CreateDatasetCommand
    from superset.commands.dataset.exceptions import DatasetSoftDeletedTwinExistsError
    from superset.connectors.sqla.models import SqlaTable

    properties = {"database": found["database_id"], "schema": found["schema"], "table_name": found["table_name"],
                  "description": with_marker(found["description"], found["marker"])}
    if found["sql"] is not None:
        properties["sql"] = found["sql"]
    command = CreateDatasetCommand(properties)

    def create() -> dict:
        if (made := _last_look(SqlaTable, found["marker"], kind="dataset")) is not None:
            return made
        try:
            dataset = command.run()
        except DatasetSoftDeletedTwinExistsError:
            raise  # answered by the caller's access, once the marker's lock is released (`_twin_answer`)
        except _command_errors() as exc:
            return _dataset_creation_answer(exc)
        return {"dataset_id": dataset.id}

    def with_editors(answer: dict) -> dict:
        if "dataset_id" not in answer:
            return answer
        editors = _set_dataset_editors(found, answer["dataset_id"])
        return answer if editors == OK else editors

    try:
        return _find_or_create(SqlaTable, found["marker"], kind="dataset", started=started, create=create,
                               then=with_editors)
    except DatasetSoftDeletedTwinExistsError:
        return _twin_answer(command)


def _check_query(context: dict) -> dict | None:
    """Runs the query context's queries, two rows each, as `ChartDataCommand` runs a chart's, so a chart whose query
    fails is never saved (as MCP's `_compile_chart`, `superset/mcp_service/chart/compile.py`, checks its own). None
    when they run; else the answer."""
    return _probe(context)[0]


def _has_rows(result: Any) -> bool:
    queries = result.get("queries") if isinstance(result, dict) else None
    return bool(queries) and any(isinstance(q, dict) and q.get("data") for q in queries)


def _probe(context: dict) -> tuple[dict | None, Any]:
    """`_check_query`'s run: (the answer or None, the command's result)."""
    from marshmallow import ValidationError
    from sqlalchemy.exc import SQLAlchemyError
    from superset.charts.schemas import ChartDataQueryContextSchema
    from superset.commands.chart.data.get_data_command import ChartDataCommand
    from superset.exceptions import SupersetException, SupersetSecurityException
    from superset.mcp_service.chart.query_result import query_result_failure

    probe = copy.deepcopy(context)
    for query in probe["queries"]:
        query["row_limit"] = _CHART_ROW_LIMIT_PROBE
    try:
        command = ChartDataCommand(ChartDataQueryContextSchema().load(probe))
        command.validate()
        result = command.run()
    except SupersetSecurityException:
        return _not_found("dataset"), None
    # Any other of Superset's exceptions (a command's, or one raised building the query object, such as
    # `QueryObjectValidationError`), the schema's and the database's: the MCP service would pass its words on.
    except (SupersetException, ValidationError, SQLAlchemyError, ValueError, KeyError) as exc:
        logger.warning("mnemiq extension: create_chart's query did not run", exc_info=exc)
        return {"refused": "query_failed"}, None
    if query_result_failure(result) is not None:
        logger.warning("mnemiq extension: create_chart's query failed")
        return {"refused": "query_failed"}, None
    return None, result


def _funnel_form(found: dict) -> tuple[str, dict, dict] | dict:
    """The request's own `viz_type` and `params` read for the dataset: (viz_type, form data, query context), or the
    refusal."""
    dataset_id, viz_type, form_data = found["dataset_id"], found["viz_type"], found["params"]
    # The params name the chart's own dataset and type, or nothing: the row's binding is the request's.
    if form_data.setdefault("viz_type", viz_type) != viz_type:
        return {"refused": "params"}
    bound = {"datasource": f"{dataset_id}__table", "datasource_id": dataset_id, "datasource_type": "table"}
    form_data.setdefault("datasource", bound["datasource"])
    if any(key in form_data and form_data[key] != value for key, value in bound.items()):
        return {"refused": "datasource"}
    try:
        context = params.query_context(viz_type, form_data, dataset_id)
    except ValueError:
        return {"refused": "params"}
    return viz_type, form_data, context


def _config_form(found: dict, dataset: Any) -> tuple[str, dict, dict | None] | dict:
    """MCP's chart config made into a chart as its `generate_chart` saves one: the config validated
    (`ValidationPipeline`), mapped to form data (`map_config_to_form_data`, whose warnings key is dropped), the
    request's `set` merged over it, and the query run (`_compile_chart`; a histogram's by the extension's own
    `_check_query`, as rc2's compile leaves a histogram's column out of its query), so a chart whose query fails is never saved.
    (viz_type, form data, query context), or the refusal. Runs under the marker's lock, which is held on a connection
    of its own, so whatever rc2 commits or rolls back here keeps it held."""
    from superset.mcp_service.chart.chart_utils import map_config_to_form_data
    from superset.mcp_service.chart.compile import _compile_chart
    from superset.mcp_service.chart.validation import ValidationPipeline

    dataset_id = found["dataset_id"]
    validated = ValidationPipeline.validate_request_with_warnings({
        "dataset_id": dataset_id, "config": found["config"], "chart_name": found["slice_name"], "save_chart": True})
    if not validated.is_valid or validated.request is None:
        return {"refused": "config"}
    try:
        # A plugin's `post_map_validate` refuses by raising ValueError, as a histogram's does a non-numeric column.
        form_data = map_config_to_form_data(validated.request.config, dataset_id=dataset_id)
    except ValueError:
        return {"refused": "config"}
    form_data.pop("_mcp_warnings", None)
    form_data.update(found["set"])
    viz_type = form_data.get("viz_type")
    if not isinstance(viz_type, str) or not viz_type:
        return {"refused": "config"}
    try:
        context = params.query_context(viz_type, form_data, dataset_id)
        check = params.histogram_check_context(form_data, dataset_id) if viz_type == "histogram_v2" else None
    except ValueError:
        return {"refused": "config"}
    if check is not None:
        # rc2's `_compile_chart` builds its query from metrics and groupby and leaves a histogram's column (named
        # under `column` alone) out, so the extension's own check runs a query that selects the groups and the column.
        refused, result = _probe(check)
        if refused is not None:
            return refused
        if len(check["queries"][0]["columns"]) > 1 and not _has_rows(result):
            # The probe keeps only rows with every group set. None left may be an empty chart, which renders, or
            # observations with no usable group, which the operator cannot unstack: a look without the group filters
            # tells them apart.
            refused, result = _probe(params.histogram_check_context(form_data, dataset_id, grouped=False))
            if refused is not None:
                return refused
            if _has_rows(result):
                logger.warning("mnemiq extension: create_chart's histogram has no row with a group set")
                return {"refused": "query_failed"}
    elif not _compile_chart(form_data, dataset.id).success:
        logger.warning("mnemiq extension: create_chart's query failed")
        return {"refused": "query_failed"}
    return viz_type, form_data, context


@_answering("create_chart")
def create_chart(found: dict) -> dict:
    """A chart on the dataset, for the dashboard, in one of two shapes: a funnel or a sankey as `{viz_type, params}`,
    or any family as MCP's own chart config, `{config}` with the format keys `set` (only `_CREATE_SET_KEYS`) to merge into
    its params. Its query
    is run first and the chart saved only if it runs; then `CreateChartCommand` saves its params, its query context,
    the editors `_chart_editors` chooses and the build's marker, once for the marker. The editors are read in the
    create's own transaction. The chart joins the dashboard when a layout places it (`set_layout`).

    The request, the dataset and the dashboard are checked first (a funnel's or a sankey's params too, which needs no
    database). Then the marker's lock is taken and held from the look through the create's commit
    (`_find_or_create`): a chart the marker names already is answered before its query is checked, so a retry of a
    chart whose answer was lost never runs the source query again, which a slow source database, the likely cause of
    the lost answer, could now fail; and a retry sent while its first send still runs waits for the first, then
    answers the chart it made, never a refusal of its own check while the first may still make the chart. Only a
    look that finds nothing runs the config's validation and mapping and the query check, whose refusal is then
    answered as it is."""
    started = _clock()
    if claim_marker(found["marker"], found["dashboard_id"]) is None:
        return BAD_MARKER
    from_config = found["config"] is not None
    own = not from_config
    if from_config:
        shaped = found["viz_type"] is None and found["params"] is None
    else:
        shaped = found["viz_type"] is not None and found["params"] is not None and found["set"] is None
    if not shaped:
        return {"refused": "request"}
    if from_config and not set(found["set"] or ()) <= _CREATE_SET_KEYS:
        return {"refused": "set"}
    dataset_id = found["dataset_id"]
    prepared = _funnel_form(found) if own else None
    if isinstance(prepared, dict):
        return prepared
    dataset = _dataset(dataset_id)
    if dataset is None:
        return _not_found("dataset")
    dashboard = _dashboard(found["dashboard_id"])
    if dashboard is None:
        return _not_found("dashboard")
    if not _security().is_editor(dashboard):
        return _not_editor(dashboard)
    from superset.commands.chart.create import CreateChartCommand
    from superset.models.slice import Slice

    def create() -> dict:
        if prepared is None:
            found["set"] = found["set"] or {}
            checked = _config_form(found, dataset)
        else:
            checked = _check_query(prepared[2]) or prepared
        if isinstance(checked, dict):
            return checked
        viz_type, form_data, context = checked
        editors = _chart_editors(dataset, dashboard, ())
        # rc2's checks may have ended the transaction the look bounded (Superset's metastore cache commits the session
        # when it stores a chart's data): `_last_look` bounds the command's own transaction again.
        if (made := _last_look(Slice, found["marker"], kind="chart")) is not None:
            return made
        try:
            chart = CreateChartCommand({
                "slice_name": found["slice_name"], "viz_type": viz_type, "datasource_id": dataset_id,
                "datasource_type": "table", "params": json.dumps(form_data),
                "query_context": json.dumps(context) if context is not None else None,
                "editors": editors, "description": found["marker"]}).run()
        except _command_errors() as exc:
            return _command_answer("create_chart", exc, kind="dataset")
        return {"chart_id": chart.id}

    return _find_or_create(Slice, found["marker"], kind="chart", started=started, create=create)


def _patch(found: dict, chart: Any) -> dict:
    security = _security()
    if chart is None or not security.can_access_chart(chart):
        return _not_found("chart")
    if not security.is_editor(chart):
        return _not_editor(chart)
    if chart.changed_on != found["expect_changed_on"]:
        return STALE
    if chart.datasource_type != "table":
        return {"refused": "datasource"}
    form_data = _stored_object(chart.params)
    if form_data is None:
        return {"refused": "params"}
    try:
        patched = params.patch(chart.viz_type, form_data, found["set"], found["unset"], chart.datasource_id)
    except params.PatchRefused as refused:
        return {"refused": "patch", "reason": refused.reason}
    except ValueError:
        return {"refused": "params"}
    from superset.commands.chart.update import UpdateChartCommand

    try:
        UpdateChartCommand(chart.id, {"params": json.dumps(patched.params), "viz_type": patched.viz_type,
                                      "query_context": _dumps(patched.query_context)}).run()
    except _command_errors() as exc:
        return _command_answer("patch_chart", exc, kind="chart", editable=chart)
    return OK


@_answering("patch_chart")
def patch_chart(found: dict) -> dict:
    """Sets and unsets only the named `params` keys of a chart, keeping every other key as it is, under the chart's
    lock, once the `changed_on` the caller read is still the chart's. The query context is regenerated from the merged
    params for a family that has one, and cleared for the rest. One `UpdateChartCommand`."""
    from superset.models.slice import Slice

    return _locked(Slice, found["chart_id"], lambda chart: _patch(found, chart))


def _add_metric(found: dict, dataset: Any) -> dict:
    from superset.sql.parse import has_aggregate

    security = _security()
    if dataset is None or not security.can_access_datasource(dataset):
        return _not_found("dataset")
    if not security.is_editor(dataset):
        return _not_editor(dataset)
    if dataset.changed_on != found["expect_changed_on"]:
        return STALE
    metric = found["metric"]
    # Superset's own rule for a saved metric (`superset/sql/parse.py`, `has_aggregate`), which fails open: mnemiq
    # holds the measure to its own fail-closed check before it asks.
    if not has_aggregate(metric["expression"], dataset.database.backend):
        return {"refused": "row_level"}
    try:
        stubs = params.metric_stub_list([{"id": m.id, "metric_name": m.metric_name} for m in dataset.metrics], metric)
    except params.MetricExists:
        return EXISTS
    from superset.commands.dataset.update import UpdateDatasetCommand

    try:
        UpdateDatasetCommand(dataset.id, {"metrics": stubs}).run()
    except _command_errors() as exc:
        return _command_answer("add_saved_metric", exc, kind="dataset", editable=dataset)
    return OK


@_answering("add_saved_metric")
def add_saved_metric(found: dict) -> dict:
    """Adds one saved metric to a dataset and keeps every one it has, under the dataset's lock, once the `changed_on`
    the caller read is still the dataset's: `UpdateDatasetCommand` replaces the whole list, so each existing metric
    goes back as its id and name alone."""
    from superset.connectors.sqla.models import SqlaTable

    return _locked(SqlaTable, found["dataset_id"], lambda dataset: _add_metric(found, dataset))


def _live_chart_ids(dataset_id: int) -> list[int]:
    """The charts on the dataset that are not in the trash, whoever made them: one in the trash is restored without
    its dataset only to show nothing."""
    from superset.models.slice import Slice

    rows = (_session().query(Slice.id)
            .filter(Slice.datasource_id == dataset_id, Slice.datasource_type == "table", Slice.deleted_at.is_(None))
            .order_by(Slice.id))
    return [row.id for row in rows]


def _soft_deleted(dataset_id: int) -> bool:
    """Whether the dataset is in the trash rather than gone."""
    from superset.connectors.sqla.models import SqlaTable
    from superset.models.helpers import skip_visibility_filter

    session = _session()
    with skip_visibility_filter(session, SqlaTable):
        row = session.query(SqlaTable.deleted_at).filter(SqlaTable.id == dataset_id).one_or_none()
    return row is not None and row.deleted_at is not None


def _delete(found: dict, dataset: Any) -> dict:
    if dataset is None or not _security().can_access_datasource(dataset):
        return _not_found("dataset")
    if not _security().is_editor(dataset):
        return _not_editor(dataset)
    # A second guard: mnemiq asks only for a dataset its store recorded as made for the dashboard.
    marker = params.parse_marker(dataset.description)
    if marker is None:
        return {"refused": "no_marker"}
    if marker.dashboard_id != found["dashboard_id"]:
        return {"refused": "another_dashboard"}
    if in_use := _live_chart_ids(dataset.id):
        return {"in_use": True, "chart_ids": in_use}
    dataset_id, dataset_uuid = dataset.id, str(dataset.uuid)
    from superset.commands.dataset.delete import DeleteDatasetCommand
    from superset.commands.dataset.restore import RestoreDatasetCommand

    try:
        DeleteDatasetCommand([dataset_id]).run()
    except _command_errors() as exc:
        return _command_answer("delete_dataset", exc, kind="dataset", editable=dataset)
    # No lock keeps a chart from being made on the dataset meanwhile (a chart's dataset has no foreign key, and
    # `CreateChartCommand` takes no lock), so the charts are counted again now that the delete is committed.
    soft = _soft_deleted(dataset_id)
    if appeared := _live_chart_ids(dataset_id):
        restored = False
        if soft:
            try:
                RestoreDatasetCommand(dataset_uuid).run()
                restored = True
            except _command_errors() as exc:
                logger.warning("mnemiq extension: a dataset a chart now uses could not be restored", exc_info=exc)
        return {"in_use": True, "chart_ids": appeared, "restored": restored}
    return {"ok": True, "soft_deleted": soft}


@_answering("delete_dataset")
def delete_dataset(found: dict) -> dict:
    """Deletes a dataset the build made for this dashboard (its marker says so) once no chart uses it, under the
    dataset's lock. Superset soft-deletes it, into the trash; a chart made on it while it was deleted brings it back
    (`RestoreDatasetCommand`) and the answer says the dataset is in use."""
    from superset.connectors.sqla.models import SqlaTable

    return _locked(SqlaTable, found["dataset_id"], lambda dataset: _delete(found, dataset))


def _member_chart_ids(dashboard_id: int) -> list[int]:
    """The dashboard's charts as its membership table holds them, a chart in the trash included: rc2 keeps a trashed
    chart a member so its restore puts it back (`DashboardDAO.set_dash_metadata`)."""
    from sqlalchemy import select
    from superset.models.dashboard import dashboard_slices

    rows = _session().execute(select(dashboard_slices.c.slice_id).where(dashboard_slices.c.dashboard_id == dashboard_id))
    return sorted(row.slice_id for row in rows)


def _layout(found: dict, dashboard: Any) -> dict:
    security = _security()
    if dashboard is None or not security.can_access_dashboard(dashboard):
        return _not_found("dashboard")
    if not security.is_editor(dashboard):
        return _not_editor(dashboard)
    if dashboard.changed_on != found["expect_changed_on"]:
        return STALE
    added, removed = found["add_chart_ids"], found["remove_chart_ids"]
    # `set_dash_metadata` reads the added charts with no access check of its own.
    if blocked := sorted({chart_id for chart_id in added if not _readable_chart(chart_id)}):
        return {"refused": "chart_access", "chart_ids": blocked}
    current_positions = _stored_object(dashboard.position_json) or {}
    refusal = params.check_positions(found["positions"], current_positions=current_positions,
                                     current_chart_ids=_member_chart_ids(dashboard.id), add_chart_ids=added,
                                     remove_chart_ids=removed)
    if refusal is not None:
        return refusal
    metadata = params.layout_metadata(found["positions"], _stored_object(dashboard.json_metadata) or {})
    from superset.commands.dashboard.update import UpdateDashboardCommand

    try:
        UpdateDashboardCommand(dashboard.id, {"json_metadata": json.dumps(metadata)}).run()
    except _command_errors() as exc:
        return _command_answer("set_layout", exc, kind="dashboard", editable=dashboard)
    return OK


@_answering("set_layout")
def set_layout(found: dict) -> dict:
    """Writes a dashboard's layout, which also sets its charts (`json_metadata.positions`, through
    `UpdateDashboardCommand`), under the dashboard's lock, once the `changed_on` the caller read is still the
    dashboard's. The layout must hold exactly the dashboard's charts plus those added minus those removed, keep every
    tab rc2 knows reachable under its id (`params.check_positions`), and add only charts the caller may read. The
    dashboard's native filters and the settings rc2 would otherwise reset are sent back as they are
    (`params.layout_metadata`), so no report on a filter is deactivated."""
    from superset.models.dashboard import Dashboard

    return _locked(Dashboard, found["dashboard_id"], lambda dashboard: _layout(found, dashboard))
