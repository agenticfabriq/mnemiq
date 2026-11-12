"""Statements run through Superset's `execute_sql` MCP tool, as a source adapter the query runner and the proof seam
use as they use any other.

Every statement runs as the person whose token the client carries, so Superset's permissions and row rules decide what
it reads. For a data statement the adapter sends the database, the schema, the statement and a time limit, and nothing
else: no `limit`, which Superset would put in place of the statement's own LIMIT; no `template_params`; and, on the ask
path, nothing that asks Superset to skip its result cache, which keys a result on the statement after its row rules are
applied and is consulted only once access is checked. The adapter enrichment reads through is built to send
`force_refresh` with every statement instead, since a re-enrich is meant to read the source as it is now.

The source's catalogue is read the same way, from Postgres's `pg_catalog` (`mnemiq.superset.pg_catalog`), each part of
it once per adapter, and each statement with `force_refresh`, so Superset runs it afresh rather than answer from its
result cache.
"""

from __future__ import annotations

import math
from functools import cached_property

import pyarrow as pa

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.superset.client import (
    SupersetDenied,
    SupersetMcp,
    SupersetPermissionDenied,
    SupersetRefusal,
    SupersetRefusedCaller,
    SupersetResponseTooLarge,
    SupersetUnavailable,
)
from mnemiq.superset.payload import ResultCut, SupersetQueryFailed, check_execute_response, read_execute_response
from mnemiq.superset.pg_catalog import PostgresCatalogue
from mnemiq.superset.settings import SUPERSET_DIALECT

# `execute_sql` renders every statement as a Jinja template before it checks access and runs it, so a statement holding
# one of these markers could run as something other than the statement mnemiq proved. Such a statement is never sent.
JINJA_DELIMITERS = ("{{", "{%", "{#")

# The whole seconds `ExecuteSqlRequest.timeout` accepts (`superset/mcp_service/sql_lab/schemas.py`).
_SUPERSET_TIMEOUT_RANGE = (1, 300)

# How much longer the transport waits than the time limit sent to Superset. Superset ends a statement at that limit
# and answers `timed_out`, which the runner reads as the statement's cost for the model to make cheaper. Were the
# transport to give up first, the same slow statement would read as Superset being unavailable, and the turn would
# end telling the person to try again.
TRANSPORT_MARGIN_S = 15.0

# What a result too large to send reads as. Superset's MCP service sends a result past its response size cut, where
# dropping rows brings it under that size, and otherwise sends an error in its place (`_handle_oversized_response` in
# `superset/mcp_service/middleware.py`). Either way the rows are not all there, and the statement is the caller's to
# make smaller, so both raise `ResultCut`.
_TOO_LARGE_TO_SEND = ("Superset would not send this result, which is larger than its response size allows; aggregate, "
                      "or select fewer rows or columns.")

# The refusals that end a request wherever they are met: of the caller, of the caller's roles, and one Superset does not
# explain. Each stays the same for every later call the request makes, so no rewrite or retry can help. Superset's
# refusal of a table is not among them: it is the model's to repair.
TERMINAL_REFUSALS = (SupersetRefusedCaller, SupersetPermissionDenied, SupersetDenied)


class SupersetSqlAdapter:
    """Runs statements on one Superset database and schema through `client`, each within `timeout_s` unless the
    caller gives its own limit. With `refresh_every_statement`, every statement, data statements included, asks
    Superset to run it rather than answer from its result cache.

    Each catalogue method reads its part of the catalogue from the database on the first call, never from Superset's
    result cache, and answers every later call with a copy of what it read: `decide` reads the function inventory on
    every decision, and one adapter serves one request, so the inventory is read live once per request. A read that
    raises is not remembered, and the next call asks again."""

    # The Postgres types with no equality operator, which `count(DISTINCT ...)` and `GROUP BY` refuse, as
    # `format_type` names them, upper-cased as profiling compares them. Profiling reads this in place of its table by
    # dialect: a direct Postgres adapter declares the same dialect and profiles as it always has.
    unaggregatable_types = frozenset({"JSON", "XML", "POINT", "LINE", "LSEG", "BOX", "PATH", "POLYGON", "CIRCLE"})

    def __init__(self, client: SupersetMcp, *, database_id: int, schema: str, dialect: str = SUPERSET_DIALECT,
                 timeout_s: float = 30.0, refresh_every_statement: bool = False) -> None:
        self.client = client
        self.dialect = dialect
        self._database_id = database_id
        self._schema = schema
        self._timeout_s = timeout_s
        self._refresh_every_statement = refresh_every_statement
        self._terminal_refusal: SupersetRefusal | None = None

    @property
    def terminal_refusal(self) -> SupersetRefusal | None:
        """The refusal the request this adapter serves ends as, or None: of the `TERMINAL_REFUSALS` any statement it
        sent was refused with, a refusal of the caller if there was one, else the first.

        The query runner reads each of them as a source that did not answer, the proof seam as one it could not ask,
        and the function inventory as one it could not read, so the refusal's own kind survives only here. Each stays
        the same for every later call the request makes, so the request ends as that refusal rather than as an outage
        to try again. A refusal of the caller outranks the others, since signing in again is the one thing the person
        can do that might help. One adapter serves one request, so what it remembers is never another person's."""
        return self._terminal_refusal

    @property
    def database_id(self) -> int:
        """The Superset database every statement runs on."""
        return self._database_id

    @property
    def schema(self) -> str:
        """The schema every statement runs in, spelled as Postgres stores it."""
        return self._schema

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """`sql`'s result. Raises `ResultCut` when Superset cut the result to size or would not send it at all."""
        return self._result(sql, timeout_s)

    def execute(self, sql: str) -> list[tuple]:
        return _tuples(self.execute_arrow(sql))

    def validate(self, sql: str) -> None:
        """The proof the proof seam asks for in place of `EXPLAIN`: the statement as a subquery that returns no rows.
        Superset checks it for access and applies its row rules as it would to the statement itself, and the database
        resolves every table and column in it without returning a row. Raises what the statement would."""
        body = sql.strip().rstrip(";").rstrip()
        read_execute_response(self._execute_sql(f"SELECT * FROM ({body}) AS mnemiq_probe WHERE 1 = 0", None))

    def dry_run(self, sql: str) -> None:
        """Have Superset check `sql` and prepare it without running it. A refusal raises what the same refusal raises
        from a run; a dry run Superset accepts returns nothing, whatever its payload carries."""
        check_execute_response(self._execute_sql(sql, None, dry_run=True))

    def source_unavailable(self, exc: BaseException) -> bool:
        """The proof seam's question: did the source not answer, rather than answer no.

        Superset unreachable, and every refusal of the caller: a token Superset no longer accepts, or roles that do
        not permit the tool, stays refused however the statement is rewritten, so it ends the turn rather than reaching
        the repair loop. A time limit reached and a cancel are unanswered at the proof too, as `QueryCancelled` has
        it; the runner asks for both before it asks this, so there they stay the statement's cost and the cancel.
        Superset's refusal of a table, a result cut to size and a tool error are answers, which the model repairs."""
        return isinstance(exc, (SupersetUnavailable, SupersetRefusal, QueryTimedOut, QueryCancelled))

    def introspect(self) -> list[str]:
        return list(self._tables)

    def list_columns(self) -> list[tuple[str, str, str]]:
        return list(self._columns)

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]:
        return list(self._foreign_keys)

    def view_definitions(self) -> list[tuple[str, str, str]]:
        return list(self._views)

    def user_functions(self) -> list[str]:
        return list(self._user_functions)

    def builtin_functions(self) -> list[str]:
        return list(self._builtin_functions)

    def reachable_user_functions(self) -> list[str]:
        return list(self._reachable_user_functions)

    def virtual_columns(self) -> list[tuple[str, str, str]]:
        return list(self._virtual_columns)

    @cached_property
    def _catalogue(self) -> PostgresCatalogue:
        return PostgresCatalogue(self._catalogue_rows, self._schema)

    def _catalogue_rows(self, sql: str) -> list[tuple]:
        """A catalogue statement's rows as the database has them now. Superset answers a statement it has run before
        from its result cache unless the request says `force_refresh` (`_try_get_cached_result` in
        `superset/sql/execution/executor.py`), for as long as a day by default, and a catalogue statement is the same
        text every time. From the cache, a function created since would read as a builtin and pass the decider unseen,
        and the pages of one read could come from different moments, skipping a row or repeating one.

        A page too large to send raises `ResultCut`, as a page cut to size does, so the catalogue asks for it again
        smaller."""
        return _tuples(self._result(sql, None, force_refresh=True))

    @cached_property
    def _tables(self) -> tuple[str, ...]:
        return tuple(self._catalogue.tables())

    @cached_property
    def _columns(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(self._catalogue.columns())

    @cached_property
    def _foreign_keys(self) -> tuple[tuple[str, str, str, str, str], ...]:
        return tuple(self._catalogue.foreign_keys())

    @cached_property
    def _views(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(self._catalogue.views())

    @cached_property
    def _user_functions(self) -> tuple[str, ...]:
        return tuple(self._catalogue.user_functions())

    @cached_property
    def _builtin_functions(self) -> tuple[str, ...]:
        return tuple(self._catalogue.builtin_functions())

    @cached_property
    def _reachable_user_functions(self) -> tuple[str, ...]:
        return tuple(self._catalogue.reachable_user_functions())

    @cached_property
    def _virtual_columns(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(self._catalogue.virtual_columns())

    def _result(self, sql: str, timeout_s: float | None, *, force_refresh: bool = False) -> pa.Table:
        """The table `sql` returns, read from `execute_sql`'s payload; a result too large to send raises `ResultCut`."""
        try:
            payload = self._execute_sql(sql, timeout_s, force_refresh=force_refresh)
        except SupersetResponseTooLarge as exc:
            raise ResultCut(_TOO_LARGE_TO_SEND) from exc
        return read_execute_response(payload)

    def _execute_sql(self, sql: str, timeout_s: float | None, *, dry_run: bool = False,
                     force_refresh: bool = False) -> dict:
        for delimiter in JINJA_DELIMITERS:
            if delimiter in sql:
                raise SupersetQueryFailed(f"the statement holds {delimiter}, which Superset reads as a template "
                                          f"marker, so it was not sent; write the statement without {delimiter}")
        limit = _superset_timeout(self._timeout_s if timeout_s is None else timeout_s)
        request = {"database_id": self._database_id, "sql": sql, "schema": self._schema, "timeout": limit}
        if dry_run:
            request["dry_run"] = True
        if force_refresh or self._refresh_every_statement:
            request["force_refresh"] = True
        try:
            return self.client.call("execute_sql", request, timeout_s=limit + TRANSPORT_MARGIN_S)
        except TERMINAL_REFUSALS as exc:
            kept = self._terminal_refusal
            if kept is None or (isinstance(exc, SupersetRefusedCaller) and not isinstance(kept, SupersetRefusedCaller)):
                self._terminal_refusal = exc
            raise


def _tuples(table: pa.Table) -> list[tuple]:
    """`table`'s rows as tuples, each in column order."""
    return [tuple(row.values()) for row in table.to_pylist()]


def _superset_timeout(timeout_s: float) -> int:
    """`timeout_s` as whole seconds in the range Superset accepts, rounded up so a statement never gets less time than
    it was given."""
    low, high = _SUPERSET_TIMEOUT_RANGE
    return max(low, min(high, math.ceil(timeout_s)))
