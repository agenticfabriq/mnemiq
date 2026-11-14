"""Stand-ins for the Superset modules the extension's write tools import inside themselves, so a unit test can run a
tool's own logic in mnemiq's environment, which has neither Superset nor its libraries. Each stand-in carries only
what `agenticfabriq.mnemiq.writes` reads from it; the exceptions keep rc2's hierarchy (`superset/exceptions.py`,
`superset/commands/exceptions.py`), which is what the tools' answers depend on.

`install(monkeypatch)` puts them in `sys.modules` for one test and answers the namespace a test configures them with.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace


class SupersetException(Exception):
    pass


class SupersetErrorException(SupersetException):
    pass


class SupersetSecurityException(SupersetErrorException):
    pass


class QueryObjectValidationError(SupersetException):
    pass


class CommandException(SupersetException):
    pass


class CommandInvalidError(CommandException):
    def normalized_messages(self) -> dict:
        return {"_schema": [str(self)]}


class ObjectNotFoundError(CommandException):
    pass


class ForbiddenError(CommandException):
    pass


class DeleteFailedError(CommandException):
    pass


class ChartNotFoundError(CommandException):
    pass


class DatasetNotFoundError(CommandException):
    pass


class DatasetRestoreFailedError(CommandException):
    pass


class SQLAlchemyError(Exception):
    pass


class DBAPIError(SQLAlchemyError):
    pass


class MarshmallowValidationError(Exception):
    pass


# rc2's Postgres SQLSTATEs for lock contention: serialization_failure, deadlock_detected and lock_not_available, the
# last what a `lock_timeout` raises (`superset/versioning/db_errors.py`, `_PG_LOCK_CONTENTION`).
PG_LOCK_CONTENTION = ("40001", "40P01", "55P03")


def is_lock_contention_error(exc) -> bool:
    """rc2's `is_lock_contention_error` (`superset/versioning/db_errors.py`) for a Postgres driver: decided by the
    driver error's SQLSTATE alone."""
    return getattr(getattr(exc, "orig", None), "pgcode", None) in PG_LOCK_CONTENTION


def database_error(sqlstate: str) -> DBAPIError:
    """A database error as SQLAlchemy wraps a psycopg2 one, its SQLSTATE on the driver error (`orig.pgcode`)."""
    error = DBAPIError(f"(psycopg2.errors) SQLSTATE {sqlstate}")
    error.orig = SimpleNamespace(pgcode=sqlstate)
    return error


class SubjectType:
    """rc2's `SubjectType` (`superset/subjects/types.py`): what an editor subject stands for."""

    USER = 1
    ROLE = 2
    GROUP = 3


def install(monkeypatch) -> SimpleNamespace:
    """The stand-ins, installed for this test. The answer holds what a test may set (`chart_data`: what running a
    query context does, a payload or an exception; `delete`, `restore`: what each command's `run` does, None or an
    exception; `update`: what `UpdateDashboardCommand.run` does; `config`: Superset's config, as
    `flask.current_app.config`; `features`: the feature flags `is_feature_enabled` answers on; `g`: `flask.g`, whose
    `user` is the request's user; `user_subjects`: each user's own subject, by user id, as `get_user_subject` finds it;
    `charts`, `dashboards`, `datasets`: what `ChartDAO`, `DashboardDAO` and `DatasetDAO` find, by id) and what the
    tools did (`ran`: each command run, in order, with its arguments)."""
    state = SimpleNamespace(chart_data={"queries": [{"data": [], "status": "success"}]}, delete=None, restore=None,
                            update=None, ran=[], config={}, features=set(), g=SimpleNamespace(), user_subjects={},
                            charts={}, dashboards={}, datasets={}, next_chart_id=500)

    class ChartDataQueryContextSchema:
        def load(self, context: dict) -> dict:
            return context

    class ChartDataCommand:
        def __init__(self, query_context) -> None:
            self.query_context = query_context

        def validate(self) -> None:
            if isinstance(state.chart_data, BaseException):
                raise state.chart_data

        def run(self) -> dict:
            state.ran.append(("chart_data", self.query_context))
            return state.chart_data

    def query_result_failure(result):
        failed = [q for q in result.get("queries", []) if q.get("error") or q.get("status") == "failed"]
        return failed[0] if failed else None

    def command(name: str, outcome: str):
        class Command:
            def __init__(self, *arguments) -> None:
                self.argument = arguments[0] if len(arguments) == 1 else arguments

            def run(self) -> None:
                state.ran.append((name, self.argument))
                if (raised := getattr(state, outcome)) is not None:
                    raise raised

        Command.__name__ = name
        return Command

    class CreateChartCommand:
        def __init__(self, properties: dict) -> None:
            self.properties = properties

        def run(self) -> SimpleNamespace:
            state.ran.append(("CreateChartCommand", self.properties))
            state.next_chart_id += 1
            return SimpleNamespace(id=state.next_chart_id)

    def get_user_id():
        return getattr(getattr(state.g, "user", None), "id", None)

    modules = {
        "flask": {"current_app": SimpleNamespace(config=state.config), "g": state.g},
        "superset": {"is_feature_enabled": lambda name: name in state.features},
        "superset.utils.core": {"get_user_id": get_user_id},
        "superset.subjects.types": {"SubjectType": SubjectType},
        "superset.subjects.utils": {"get_user_subject": lambda user_id: state.user_subjects.get(user_id)},
        "superset.models.slice": {"Slice": type("Slice", (), {"id": "slices.id"})},
        "superset.connectors.sqla.models": {"SqlaTable": type("SqlaTable", (), {"id": "tables.id"})},
        "superset.daos.dashboard": {"DashboardDAO": SimpleNamespace(
            find_by_id=lambda dashboard_id: state.dashboards.get(dashboard_id))},
        "superset.daos.dataset": {"DatasetDAO": SimpleNamespace(
            find_by_id=lambda dataset_id: state.datasets.get(dataset_id))},
        "superset.commands.chart.update": {"UpdateChartCommand": command("UpdateChartCommand", "update")},
        "superset.commands.chart.create": {"CreateChartCommand": CreateChartCommand},
        "superset.commands.dataset.update": {"UpdateDatasetCommand": command("UpdateDatasetCommand", "update")},
        "marshmallow": {"ValidationError": MarshmallowValidationError},
        "sqlalchemy": {"text": lambda sql: sql},
        "sqlalchemy.exc": {"SQLAlchemyError": SQLAlchemyError, "DBAPIError": DBAPIError},
        "superset.exceptions": {"SupersetException": SupersetException,
                                "SupersetSecurityException": SupersetSecurityException,
                                "QueryObjectValidationError": QueryObjectValidationError},
        "superset.commands.exceptions": {"CommandException": CommandException,
                                         "CommandInvalidError": CommandInvalidError,
                                         "ObjectNotFoundError": ObjectNotFoundError, "ForbiddenError": ForbiddenError},
        "superset.commands.chart.exceptions": {"ChartNotFoundError": ChartNotFoundError},
        "superset.commands.dataset.exceptions": {"DatasetNotFoundError": DatasetNotFoundError},
        "superset.charts.schemas": {"ChartDataQueryContextSchema": ChartDataQueryContextSchema},
        "superset.commands.chart.data.get_data_command": {"ChartDataCommand": ChartDataCommand},
        "superset.mcp_service.chart.query_result": {"query_result_failure": query_result_failure},
        "superset.commands.dataset.delete": {"DeleteDatasetCommand": command("DeleteDatasetCommand", "delete")},
        "superset.commands.dataset.restore": {"RestoreDatasetCommand": command("RestoreDatasetCommand", "restore")},
        "superset.commands.dashboard.update": {"UpdateDashboardCommand": command("UpdateDashboardCommand", "update")},
        "superset.daos.chart": {"ChartDAO": SimpleNamespace(find_by_id=lambda chart_id: state.charts.get(chart_id))},
        "superset.versioning.db_errors": {"is_lock_contention_error": is_lock_contention_error},
    }
    made: dict[str, types.ModuleType] = {}

    def stub(name: str) -> types.ModuleType:
        """A stand-in module for `name` (and its parents), in `sys.modules` for this test only."""
        if name not in made:
            made[name] = types.ModuleType(name)
            monkeypatch.setitem(sys.modules, name, made[name])
            if "." in name:
                parent, _, child = name.rpartition(".")
                setattr(stub(parent), child, made[name])
        return made[name]

    for name, attributes in modules.items():
        stub(name).__dict__.update(attributes)
    return state
