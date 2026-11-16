"""Stand-ins for the Superset modules the extension's write tools import inside themselves, so a unit test can run a
tool's own logic in mnemiq's environment, which has neither Superset nor its libraries. Each stand-in carries only
what `agenticfabriq.mnemiq.writes` reads from it; the exceptions keep rc2's hierarchy (`superset/exceptions.py`,
`superset/commands/exceptions.py`), which is what the tools' answers depend on.

`install(monkeypatch)` puts them in `sys.modules` for one test and answers the namespace a test configures them with.
"""

from __future__ import annotations

import copy
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace


class SupersetException(Exception):
    pass


class SupersetErrorException(SupersetException):
    pass


class SupersetSecurityException(SupersetErrorException):
    pass


class SupersetParseError(SupersetErrorException):
    """rc2's error for SQL it cannot parse."""


class SupersetDisallowedSQLFunctionException(SupersetErrorException):
    """rc2's error for a query using a function its `DISALLOWED_SQL_FUNCTIONS` lists."""


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


class DatasetInvalidError(CommandInvalidError):
    """rc2's `DatasetInvalidError`: `get_list_classnames` names the validation errors it holds."""

    def __init__(self, classnames=()) -> None:
        super().__init__("Dataset parameters are invalid.")
        self.classnames = sorted(classnames)

    def get_list_classnames(self) -> list[str]:
        return self.classnames


class DatasetSoftDeletedTwinExistsError(CommandException):
    pass


class DatasetRestoreFailedError(CommandException):
    pass


class SQLAlchemyError(Exception):
    pass


class DBAPIError(SQLAlchemyError):
    """SQLAlchemy's wrapper of a driver's error; `connection_invalidated` is set when SQLAlchemy found the connection
    lost and invalidated it."""

    connection_invalidated = False


class PoolTimeoutError(SQLAlchemyError):
    """SQLAlchemy's `TimeoutError`: no connection came free in the pool within its timeout."""


class MarshmallowValidationError(Exception):
    pass


# rc2's Postgres SQLSTATEs for lock contention: serialization_failure, deadlock_detected and lock_not_available, the
# last what a `lock_timeout` raises (`superset/versioning/db_errors.py`, `_PG_LOCK_CONTENTION`).
PG_LOCK_CONTENTION = ("40001", "40P01", "55P03")


def is_lock_contention_error(exc) -> bool:
    """rc2's `is_lock_contention_error` (`superset/versioning/db_errors.py`) for a Postgres driver: decided by the
    driver error's SQLSTATE alone."""
    return getattr(getattr(exc, "orig", None), "pgcode", None) in PG_LOCK_CONTENTION


def database_error(sqlstate: str, invalidated: bool = False) -> DBAPIError:
    """A database error as SQLAlchemy wraps a psycopg2 one, its SQLSTATE on the driver error (`orig.pgcode`), with the
    connection marked invalidated when `invalidated` (SQLAlchemy found it lost)."""
    error = DBAPIError(f"(psycopg2.errors) SQLSTATE {sqlstate}")
    error.orig = SimpleNamespace(pgcode=sqlstate)
    error.connection_invalidated = invalidated
    return error


class Column:
    """A model's column, stood in: a filter on it answers what it asks, for a session stand-in to apply."""

    def __init__(self, name: str) -> None:
        self.name = name

    def like(self, pattern: str, escape: str | None = None) -> tuple:
        return ("like", self.name, pattern, escape)

    def is_(self, value) -> tuple:
        return ("is", self.name, value)


def model(name: str) -> type:
    """A model stood in: the columns a lookup by marker reads."""
    return type(name, (), {"id": Column("id"), "description": Column("description"),
                           "deleted_at": Column("deleted_at")})


@dataclass(frozen=True)
class Table:
    """rc2's `Table` (`superset/sql/parse.py`): a table, with its schema and catalog when known."""

    table: str
    schema: str | None = None
    catalog: str | None = None


class SubjectType:
    """rc2's `SubjectType` (`superset/subjects/types.py`): what an editor subject stands for."""

    USER = 1
    ROLE = 2
    GROUP = 3


def install(monkeypatch) -> SimpleNamespace:
    """The stand-ins, installed for this test. The answer holds what a test may set (`chart_data`: what running a
    query context does, a payload or an exception; `delete`, `restore`: what each command's `run` does, None or an
    exception; `update`: what `UpdateDashboardCommand.run` does; `compiled`: what MCP's `_compile_chart` answers, or
    raises; `config`: Superset's config, as
    `flask.current_app.config`; `features`: the feature flags `is_feature_enabled` answers on; `g`: `flask.g`, whose
    `user` is the request's user; `user_subjects`: each user's own subject, by user id, as `get_user_subject` finds it;
    `charts`, `dashboards`, `datasets`: what `ChartDAO`, `DashboardDAO` and `DatasetDAO` find, by id; `session`:
    `db.session`, which each command commits when its `run` returns and rolls back when it raises, as rc2's
    `@transaction` does; `rows`: the metadata database's dashboards and charts, by model name, to which each create
    command adds the row it makes; `create`: an exception a create command raises instead; `dataset_database`: the
    database a dataset's create reads from its id; `twin`: whether a dataset in the trash holds the table a dataset's
    create names; `security`: rc2's security manager as a dataset's create asks it, or None to ask nothing) and what
    the tools did (`ran`: each command run, in order, with its arguments)."""
    state = SimpleNamespace(chart_data={"queries": [{"data": [], "status": "success"}]}, delete=None, restore=None,
                            config_valid=True, mapped={"viz_type": "big_number_total", "metric": "count",
                                                       "_mcp_warnings": ["a sample warning"]},
                            compiled=SimpleNamespace(success=True),
                            update=None, create=None, ran=[], config={}, features=set(), g=SimpleNamespace(),
                            user_subjects={}, charts={}, dashboards={}, datasets={}, session=None,
                            rows={"Dashboard": [], "Slice": [], "SqlaTable": []}, next_id=500,
                            dataset_database=SimpleNamespace(perm=None, get_default_catalog=lambda: None),
                            twin=False, security=None)

    class Db:
        @property
        def session(self):
            return state.session

    def transaction(work):
        """What rc2's `@transaction` (`superset/utils/decorators.py`) does around a command's `run`: the session is
        committed when `work` returns and rolled back when it raises."""
        try:
            result = work()
        except Exception:
            if state.session is not None:
                state.session.rollback()
            raise
        if state.session is not None:
            state.session.commit()
        return result

    def created(model_name: str, command_name: str, properties: dict, validate=None) -> SimpleNamespace:
        state.ran.append((command_name, properties))
        if validate is not None:
            validate()
        if state.create is not None:
            raise state.create
        state.next_id += 1
        row = SimpleNamespace(id=state.next_id, description=properties.get("description"), deleted_at=None,
                              editors=[])
        if model_name == "SqlaTable":
            row.perm, row.schema_perm, row.database = None, None, state.dataset_database
            # rc2's create makes its creator an editor of the dataset.
            creator = getattr(getattr(state.g, "user", None), "subject", None)
            row.editors = [creator] if creator is not None else []
        state.rows[model_name].append(row)
        return row

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
            # A test may set a function of the query context to answer each probe as its query asks.
            return state.chart_data(self.query_context) if callable(state.chart_data) else state.chart_data

    def query_result_failure(result):
        failed = [q for q in result.get("queries", []) if q.get("error") or q.get("status") == "failed"]
        return failed[0] if failed else None

    def command(name: str, outcome: str):
        class Command:
            def __init__(self, *arguments) -> None:
                self.argument = arguments[0] if len(arguments) == 1 else arguments

            def run(self) -> None:
                def work() -> None:
                    state.ran.append((name, self.argument))
                    if (raised := getattr(state, outcome)) is not None:
                        raise raised

                transaction(work)

        Command.__name__ = name
        return Command

    def create_command(model_name: str, name: str):
        class Command:
            def __init__(self, properties: dict) -> None:
                self.properties = properties

            def run(self) -> SimpleNamespace:
                return transaction(lambda: created(model_name, name, self.properties))

        Command.__name__ = name
        return Command

    class CreateDatasetCommand:
        """rc2's `CreateDatasetCommand` (`superset/commands/dataset/create.py`), as far as the tools read it. Its
        `validate` reads the database from its id into the command's own copy of the properties (`_properties`), with
        the database's default catalog when none is given; refuses a table a dataset in the trash holds (`twin`) with
        `DatasetSoftDeletedTwinExistsError`; and asks `security.raise_for_access`, as the caller and with the
        arguments rc2 passes (the table, or `sql` with its catalog and schema), listing a refusal as
        `DatasetDataAccessIsNotAllowed` and SQL it cannot parse as a `ValidationError`. Then the dataset is made as
        every create command makes its row (`created`)."""

        def __init__(self, properties: dict) -> None:
            self.properties, self._properties = properties, properties.copy()

        def run(self) -> SimpleNamespace:
            return transaction(lambda: created("SqlaTable", "CreateDatasetCommand", self.properties, self.validate))

        def validate(self) -> None:
            database = self._properties["database"] = state.dataset_database
            catalog = self._properties.get("catalog")
            if not catalog:
                catalog = self._properties["catalog"] = database.get_default_catalog()
            schema, table_name, sql = (self._properties.get(key) for key in ("schema", "table_name", "sql"))
            if state.twin:
                raise DatasetSoftDeletedTwinExistsError()
            if state.security is None:
                return
            try:
                if sql:
                    state.security.raise_for_access(database=database, sql=sql, catalog=catalog, schema=schema)
                else:
                    state.security.raise_for_access(database=database, table=Table(table_name, schema, catalog))
            except SupersetSecurityException:
                raise DatasetInvalidError({"DatasetDataAccessIsNotAllowed"}) from None
            except SupersetParseError:
                raise DatasetInvalidError({"ValidationError"}) from None

    class ValidationPipeline:
        @staticmethod
        def validate_request_with_warnings(request_data: dict) -> SimpleNamespace:
            state.ran.append(("validate", request_data))
            if not state.config_valid:
                return SimpleNamespace(is_valid=False, request=None, error=SimpleNamespace(), warnings=None)
            return SimpleNamespace(is_valid=True, request=SimpleNamespace(config=request_data["config"]),
                                   warnings=None)

    def map_config_to_form_data(config: dict, dataset_id=None) -> dict:
        state.ran.append(("map", config, dataset_id))
        if isinstance(state.mapped, BaseException):
            raise state.mapped
        return copy.deepcopy(state.mapped)

    def _compile_chart(form_data: dict, dataset_id: int) -> SimpleNamespace:
        state.ran.append(("compile", form_data, dataset_id))
        if isinstance(state.compiled, BaseException):
            raise state.compiled
        return state.compiled

    def get_user_id():
        return getattr(getattr(state.g, "user", None), "id", None)

    modules = {
        "flask": {"current_app": SimpleNamespace(config=state.config), "g": state.g},
        "superset": {"is_feature_enabled": lambda name: name in state.features, "db": Db()},
        "superset.utils.core": {"get_user_id": get_user_id},
        "superset.subjects.types": {"SubjectType": SubjectType},
        "superset.subjects.utils": {"get_user_subject": lambda user_id: state.user_subjects.get(user_id)},
        "superset.models.slice": {"Slice": model("Slice")},
        "superset.models.dashboard": {"Dashboard": model("Dashboard")},
        "superset.connectors.sqla.models": {"SqlaTable": model("SqlaTable")},
        "superset.daos.dashboard": {"DashboardDAO": SimpleNamespace(
            find_by_id=lambda dashboard_id: state.dashboards.get(dashboard_id))},
        "superset.daos.dataset": {"DatasetDAO": SimpleNamespace(
            find_by_id=lambda dataset_id: state.datasets.get(dataset_id))},
        "superset.commands.chart.update": {"UpdateChartCommand": command("UpdateChartCommand", "update")},
        "superset.commands.chart.create": {"CreateChartCommand": create_command("Slice", "CreateChartCommand")},
        "superset.commands.dashboard.create": {
            "CreateDashboardCommand": create_command("Dashboard", "CreateDashboardCommand")},
        "superset.commands.dataset.create": {"CreateDatasetCommand": CreateDatasetCommand},
        "superset.sql.parse": {"Table": Table},
        "superset.commands.dataset.update": {"UpdateDatasetCommand": command("UpdateDatasetCommand", "update")},
        "marshmallow": {"ValidationError": MarshmallowValidationError},
        "sqlalchemy": {"text": lambda sql: sql},
        "sqlalchemy.exc": {"SQLAlchemyError": SQLAlchemyError, "DBAPIError": DBAPIError,
                           "TimeoutError": PoolTimeoutError},
        "superset.exceptions": {"SupersetException": SupersetException,
                                "SupersetSecurityException": SupersetSecurityException,
                                "SupersetParseError": SupersetParseError,
                                "QueryObjectValidationError": QueryObjectValidationError},
        "superset.commands.exceptions": {"CommandException": CommandException,
                                         "CommandInvalidError": CommandInvalidError,
                                         "ObjectNotFoundError": ObjectNotFoundError, "ForbiddenError": ForbiddenError},
        "superset.commands.chart.exceptions": {"ChartNotFoundError": ChartNotFoundError},
        "superset.commands.dataset.exceptions": {
            "DatasetNotFoundError": DatasetNotFoundError, "DatasetInvalidError": DatasetInvalidError,
            "DatasetSoftDeletedTwinExistsError": DatasetSoftDeletedTwinExistsError},
        "superset.charts.schemas": {"ChartDataQueryContextSchema": ChartDataQueryContextSchema},
        "superset.commands.chart.data.get_data_command": {"ChartDataCommand": ChartDataCommand},
        "superset.mcp_service.chart.validation": {"ValidationPipeline": ValidationPipeline},
        "superset.mcp_service.chart.chart_utils": {"map_config_to_form_data": map_config_to_form_data},
        "superset.mcp_service.chart.compile": {"_compile_chart": _compile_chart},
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
