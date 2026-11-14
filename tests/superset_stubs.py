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


def install(monkeypatch) -> SimpleNamespace:
    """The stand-ins, installed for this test. The answer holds what a test may set (`chart_data`: what running a
    query context does, a payload or an exception; `delete`, `restore`: what each command's `run` does, None or an
    exception) and what the tools did (`ran`: each command run, in order)."""
    state = SimpleNamespace(chart_data={"queries": [{"data": [], "status": "success"}]}, delete=None, restore=None,
                            ran=[])

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
            def __init__(self, argument) -> None:
                self.argument = argument

            def run(self) -> None:
                state.ran.append((name, self.argument))
                if (raised := getattr(state, outcome)) is not None:
                    raise raised

        Command.__name__ = name
        return Command

    modules = {
        "marshmallow": {"ValidationError": MarshmallowValidationError},
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
        "superset.versioning.db_errors": {"is_lock_contention_error": lambda exc: False},
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
