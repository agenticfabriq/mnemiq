"""The extension's write tools, one function per tool: what each reads, checks, writes and answers. `tools.py`
registers them on Superset's MCP service, which runs each as the caller its token names: `mcp_auth_hook`
(`superset/mcp_service/auth.py`) has set `g.user` and checked the tool's class permission and the token's scopes.

Every tool keeps these rules:

- It writes through Superset's commands (`superset/commands/`), never its models.
- A write to an existing row first takes the row's lock (`lock_row`: rc2's own locking read of the row, as its
  version restore takes it) and holds the locked entity until the command returns: the command's `@transaction`
  (`superset/utils/decorators.py`) commits in the same session, and that commit releases the lock. Only under the lock
  does a tool compare the `changed_on` its caller read, so of two racing writers the second waits, then sees the first
  one's `changed_on`, and answers `{"stale": true}`. `{"busy": true}` is for a lock-contention error alone
  (`superset/versioning/db_errors.py`, `is_lock_contention_error`), a wait past the lock timeout among them
  (`lock_row`), and the same request may be sent again.
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
import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agenticfabriq.mnemiq import params

logger = logging.getLogger(__name__)

# The chart types only the extension creates; MCP's `generate_chart` makes every other one.
EXTENSION_MADE = ("funnel", "sankey_v2")

OK = {"ok": True}
STALE = {"stale": True}
BUSY = {"busy": True}
FAILED = {"failed": True}
CLAIMED_BY_ANOTHER = {"claimed_by_another": True}
EXISTS = {"exists": True}
BAD_MARKER = {"refused": "marker"}

# What opens a Jinja block; Superset renders a stored expression as a template, so none is saved.
_TEMPLATE = re.compile(r"\{[{%#]")
_ISO_4217 = re.compile(r"[A-Z]{3}")
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
    "claim_chart": {"chart_id": _Field(_an_id), "dashboard_id": _DASHBOARD_ID, "marker": _MARKER},
    "claim_dataset": {"dataset_id": _Field(_an_id), "dashboard_id": _DASHBOARD_ID, "marker": _MARKER},
    "create_chart": {"dataset_id": _Field(_an_id), "viz_type": _Field(_one_of(EXTENSION_MADE)),
                     "slice_name": _Field(_text(250)), "params": _Field(_an_object), "dashboard_id": _DASHBOARD_ID,
                     "marker": _MARKER},
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
    replaced, so the same claim made twice leaves one."""
    lines = (description or "").rstrip().splitlines()
    if lines and params.parse_marker(lines[-1]) is not None:
        lines.pop()
    prose = "\n".join(lines).rstrip()
    return f"{prose}\n{marker}" if prose else marker


def _answering(tool: str, requests: Mapping[str, dict[str, _Field]] = REQUESTS) -> Callable[
        [Callable[[dict], dict]], Callable[[Any], dict]]:
    """Reads the request for `tool` (its fields in `requests`) before the body runs, and answers a `Refused` with its
    answer."""

    def wrap(body: Callable[[dict], dict]) -> Callable[[Any], dict]:
        @functools.wraps(body)
        def run(request: Any) -> dict:
            try:
                return body(read(tool, request, requests))
            except Refused as refused:
                return refused.answer

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


def _from_lock_contention(exc: BaseException | None) -> bool:
    """Whether a database lock-contention error is anywhere in `exc`'s chain: a command reraises the database error
    it caught as its own failure, `from` it (`superset/utils/decorators.py`, `on_error`)."""
    from sqlalchemy.exc import DBAPIError
    from superset.versioning.db_errors import is_lock_contention_error

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, DBAPIError) and is_lock_contention_error(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _failed(tool: str, exc: BaseException) -> dict:
    if _from_lock_contention(exc):
        logger.info("mnemiq extension: %s lost a lock race", tool)
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


def _lock_timeout() -> str:
    """The `lock_timeout` to set, as Postgres reads it: `MNEMIQ_LOCK_TIMEOUT_S` seconds when that is a number above 0
    and below `CALL_DEADLINE_S`, else `LOCK_TIMEOUT_S`, with a warning naming the setting. The bound also keeps the
    value finite, a whole number of milliseconds once rounded, and under Postgres' largest `lock_timeout`
    (2147483647 ms), which a larger value would make the write fail on."""
    from flask import current_app

    value = current_app.config.get("MNEMIQ_LOCK_TIMEOUT_S", LOCK_TIMEOUT_S)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < CALL_DEADLINE_S:
        logger.warning("mnemiq extension: MNEMIQ_LOCK_TIMEOUT_S must be a number of seconds above 0 and below %s "
                       "(mnemiq's deadline for a call); waiting %s", CALL_DEADLINE_S, LOCK_TIMEOUT_S)
        value = LOCK_TIMEOUT_S
    return f"{max(1, round(value * 1000))}ms"


def lock_row(model: type, entity_id: int) -> Any:
    """Row `entity_id` of `model`, locked for update and refreshed to its committed state, or None when no such row is
    visible (a row in the trash is not). A blocking `SELECT ... FOR UPDATE` of the primary-key row, so it serialises
    against every other writer that locks the row, rc2's own dataset `PUT` included.

    On Postgres the wait is bounded first: `SET LOCAL lock_timeout` (written `set_config(..., true)`, so the value is a
    bound parameter) holds until the transaction ends, which is the command's commit or `_locked`'s rollback. It
    bounds each lock wait in that transaction on its own, this row's lock and any lock a write of the command waits
    for after it, not the write as a whole. A wait past it fails with SQLSTATE 55P03
    (lock_not_available), which rc2's `is_lock_contention_error` counts as contention (`superset/versioning/
    db_errors.py`, `_PG_LOCK_CONTENTION`), so the tool answers `{"busy": true}`. SQLite takes no row locks, and MySQL
    has no lock timeout that ends with the transaction: there the database's own setting applies.

    This is rc2's `lock_entity_for_update` (`superset/versioning/api_helpers.py`) with eager loads off, as rc2 reads a
    row it locks for a version restore (`superset/commands/version_restore.py`). `lock_entity_for_update` cannot lock
    a chart on Postgres: `Slice.table` is `lazy="subquery"`, which SQLAlchemy loads with a `SELECT DISTINCT ...
    FOR UPDATE`, and Postgres rejects `FOR UPDATE` with `DISTINCT`. With eager loads off only the row itself is read;
    its relationships (editors, metrics, charts) load when first read, in the same locked transaction. Every locked
    tool takes its lock here, so the rule is one rule."""
    from sqlalchemy import text

    session = _session()
    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SELECT set_config('lock_timeout', :timeout, true)"), {"timeout": _lock_timeout()})
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
    """The editor subjects for a chart on `dataset` made or claimed for `dashboard`: the dashboard's editors, unless
    a chart's editors may read its data for being its editors (`_chart_data_by_editorship`); then they are chosen as
    a dataset's are (`_data_editors`), `kept` being the chart's editors now."""
    if _chart_data_by_editorship():
        return _data_editors(dataset, dashboard, kept)
    return _editor_ids(dashboard)


def _claimed_chart_editors(chart: Any, dashboard: Any) -> list[int]:
    return _chart_editors(chart.datasource, dashboard, chart.editors)


def _claimed_dataset_editors(dataset: Any, dashboard: Any) -> list[int]:
    return _data_editors(dataset, dashboard, dataset.editors)


# ---------------------------------------------------------------- the tools

@_answering("create_dashboard")
def create_dashboard(found: dict) -> dict:
    """An unpublished dashboard with an empty layout, the caller as its editor, and the build's marker as its
    description. The marker is the shell's own (`new`): the dashboard has no id until this returns."""
    if claim_marker(found["marker"], None) is None:
        return BAD_MARKER
    from superset.commands.dashboard.create import CreateDashboardCommand

    try:
        dashboard = CreateDashboardCommand({"dashboard_title": found["title"], "published": False,
                                            "description": found["marker"]}).run()
    except _command_errors() as exc:
        return _command_answer("create_dashboard", exc, kind="dashboard")
    return {"dashboard_id": dashboard.id}


def _claim(found: dict, resource: Any, *, kind: str, extra: Callable[[Any], dict | Mapping],
           editors: Callable[[Any, Any], list[int]], command: Callable[[int, dict], Any]) -> dict:
    """What `claim_chart` and `claim_dataset` share: under the resource's lock, refuse one the caller does not edit or
    whose dashboard the caller does not edit, then one another build's marker holds; then one command sets the
    editors `editors(resource, dashboard)` chooses and the marker (and `extra(resource)`'s properties)."""
    readable = resource is not None and (_security().can_access_chart(resource) if kind == "chart"
                                         else _security().can_access_datasource(resource))
    if not readable:
        return _not_found(kind)
    dashboard = _dashboard(found["dashboard_id"])
    if dashboard is None:
        return _not_found("dashboard")
    for edited in (resource, dashboard):
        if not _security().is_editor(edited):
            return _not_editor(edited)
    # Read under the lock: of two builds claiming one chart, the second reads the first one's marker.
    held = params.parse_marker(resource.description)
    if held is not None and held != claim_marker(found["marker"], found["dashboard_id"]):
        return CLAIMED_BY_ANOTHER
    more = extra(resource)
    if "refused" in more:
        return dict(more)
    properties = {"editors": editors(resource, dashboard),
                  "description": with_marker(resource.description, found["marker"]), **more}
    try:
        command(resource.id, properties).run()
    except _command_errors() as exc:
        return _command_answer(f"claim_{kind}", exc, kind=kind, editable=resource)
    return OK


def _chart_query_context(chart: Any) -> dict:
    """The properties a claim writes beside the editors and the marker: the query context of the chart's family,
    built from its params (null for a family that has none), or a refusal."""
    if chart.datasource_type != "table":
        return {"refused": "datasource"}
    form_data = _stored_object(chart.params)
    if form_data is None:
        return {"refused": "params"}
    try:
        context = params.query_context(chart.viz_type, form_data, chart.datasource_id)
    except ValueError:
        return {"refused": "params"}
    return {"query_context": _dumps(context)}


@_answering("claim_chart")
def claim_chart(found: dict) -> dict:
    """Makes a chart the dashboard's: its editors become the dashboard's (or, where a chart's editors may read its
    data, are chosen as a dataset's: `_chart_editors`), its description ends with the build's marker, and it gets the
    query context of its family, in one `UpdateChartCommand` under the chart's lock. A chart another build's marker
    holds is refused; the same claim made again is answered as the first was."""
    if claim_marker(found["marker"], found["dashboard_id"]) is None:
        return BAD_MARKER
    from superset.commands.chart.update import UpdateChartCommand
    from superset.models.slice import Slice

    return _locked(Slice, found["chart_id"], lambda chart: _claim(
        found, chart, kind="chart", extra=_chart_query_context, editors=_claimed_chart_editors,
        command=UpdateChartCommand))


@_answering("claim_dataset")
def claim_dataset(found: dict) -> dict:
    """Makes a dataset the dashboard's, as `claim_chart` makes a chart, through `UpdateDatasetCommand`. Its editors
    are those it has, the caller, and the dashboard's users whose grants already reach it (`_data_editors`): a co-editor
    left out is answered `{"not_editor": true, "editors": [...]}` when it asks to change or delete the dataset."""
    if claim_marker(found["marker"], found["dashboard_id"]) is None:
        return BAD_MARKER
    from superset.commands.dataset.update import UpdateDatasetCommand
    from superset.connectors.sqla.models import SqlaTable

    return _locked(SqlaTable, found["dataset_id"], lambda dataset: _claim(
        found, dataset, kind="dataset", extra=lambda _: {}, editors=_claimed_dataset_editors,
        command=UpdateDatasetCommand))


def _check_query(context: dict) -> dict | None:
    """Runs the query context's queries, two rows each, as `ChartDataCommand` runs a chart's, so a chart whose query
    fails is never saved (as MCP's `_compile_chart`, `superset/mcp_service/chart/compile.py`, checks its own). None
    when they run; else the answer."""
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
        return _not_found("dataset")
    # Any other of Superset's exceptions (a command's, or one raised building the query object, such as
    # `QueryObjectValidationError`), the schema's and the database's: the MCP service would pass its words on.
    except (SupersetException, ValidationError, SQLAlchemyError, ValueError, KeyError) as exc:
        logger.warning("mnemiq extension: create_chart's query did not run", exc_info=exc)
        return {"refused": "query_failed"}
    if query_result_failure(result) is not None:
        logger.warning("mnemiq extension: create_chart's query failed")
        return {"refused": "query_failed"}
    return None


@_answering("create_chart")
def create_chart(found: dict) -> dict:
    """A funnel or a sankey on the dataset, for the dashboard: its query is run first and the chart saved only if it
    runs; then `CreateChartCommand` saves its params, its query context, the editors `_chart_editors` chooses and the
    build's marker. The editors are read in the create's own transaction. The chart joins the dashboard when a layout
    places it (`set_layout`)."""
    if claim_marker(found["marker"], found["dashboard_id"]) is None:
        return BAD_MARKER
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
    dataset = _dataset(dataset_id)
    if dataset is None:
        return _not_found("dataset")
    dashboard = _dashboard(found["dashboard_id"])
    if dashboard is None:
        return _not_found("dashboard")
    if not _security().is_editor(dashboard):
        return _not_editor(dashboard)
    if (failure := _check_query(context)) is not None:
        return failure
    from superset.commands.chart.create import CreateChartCommand

    try:
        chart = CreateChartCommand({
            "slice_name": found["slice_name"], "viz_type": viz_type, "datasource_id": dataset_id,
            "datasource_type": "table", "params": json.dumps(form_data), "query_context": json.dumps(context),
            "editors": _chart_editors(dataset, dashboard, ()), "description": found["marker"]}).run()
    except _command_errors() as exc:
        return _command_answer("create_chart", exc, kind="dataset")
    return {"chart_id": chart.id}


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
