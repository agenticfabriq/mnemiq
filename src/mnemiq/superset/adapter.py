"""Statements run through Superset's `execute_sql` MCP tool, as a source adapter the query runner and the proof seam
use as they use any other.

Every statement runs as the person whose token the client carries, so Superset's permissions and row rules decide what
it reads. Every statement is sent with each table it reads named as Superset names it (`names.for_superset`), which
Superset's access check and row rules match on. For a data statement the adapter sends the database, the schema, the
statement and a time limit, and nothing else: no `limit`, which Superset would put in place of the statement's own
LIMIT, except to ask again, below that LIMIT, for an answer refused as larger than one response (`execute_answer`); no
`template_params`; and, on the ask path, nothing that asks Superset to skip its result cache, which keys a result on
the statement after its row rules are applied and is consulted only once access is checked. The adapter enrichment
reads through is built to send `force_refresh` with every statement instead, since a re-enrich is meant to read the
source as it is now.

The source's catalogue is read the same way, from Postgres's `pg_catalog` (`mnemiq.superset.pg_catalog`), each part of
it once per adapter, and each statement with `force_refresh`, so Superset runs it afresh rather than answer from its
result cache.
"""

from __future__ import annotations

import json
import math
import re
from functools import cached_property

import pyarrow as pa
import sqlglot
from sqlglot import exp

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.enrichment.profiling import _UNAGGREGATABLE
from mnemiq.superset.client import (
    SupersetCallTimedOut,
    SupersetDenied,
    SupersetMcp,
    SupersetPermissionDenied,
    SupersetRefusal,
    SupersetRefusedCaller,
    SupersetResponseTooLarge,
    SupersetUnavailable,
)
from mnemiq.execute.cut import mark_sent
from mnemiq.sql.guard import MAX_ROWS
from mnemiq.superset.payload import (
    ResultCut,
    SupersetQueryFailed,
    check_execute_response,
    read_execute_prefix,
    read_execute_response,
)
from mnemiq.superset.names import UnreadableStatement, for_superset
from mnemiq.superset.pg_catalog import PostgresCatalogue
from mnemiq.superset.settings import superset_dialect

# `execute_sql` renders every statement as a Jinja template before it checks access and runs it, so a statement holding
# one of these markers could run as something other than the statement mnemiq proved. Such a statement is never sent.
JINJA_DELIMITERS = ("{{", "{%", "{#")

# The whole seconds `ExecuteSqlRequest.timeout` accepts (`superset/mcp_service/sql_lab/schemas.py`).
_SUPERSET_TIMEOUT_RANGE = (1, 300)

# How much longer the transport waits than the time limit sent to Superset. Superset answers `timed_out` for a
# statement past that limit, which the runner reads as the statement's cost for the model to make cheaper. Its limit
# takes effect only once the database call returns, though, so a statement can still be running when the transport
# gives up; that is read as the time limit too (`SupersetCallTimedOut`), never as Superset being unavailable, which
# would end the turn telling the person to try again.
TRANSPORT_MARGIN_S = 15.0

# What a result too large to send reads as. Superset's MCP service sends a result past its response size cut, where
# dropping rows brings it under that size, and otherwise sends an error in its place (`_handle_oversized_response` in
# `superset/mcp_service/middleware.py`). Either way the rows are not all there: every read but an answer's raises
# `ResultCut`, the statement's to make smaller, and an answer's is its first rows, said so (`execute_answer`).
_TOO_LARGE_TO_SEND = ("Superset would not send this result, which is larger than its response size allows; aggregate, "
                      "or select fewer rows or columns.")

# What a refusal of a result too large to send says of its size: `Response too large: <N> bytes (limit: <M>)`
# (`superset/mcp_service/middleware.py`). An answer asked again is sized from it.
_REFUSED_SIZE = re.compile(r"Response too large: ([\d,]+) bytes \(limit: ([\d,]+)\)")
# How many times an answer refused as too large is asked again for fewer rows, each sized from the latest refusal, and
# the share of the limit each aims for. rc2's answer carries its rows twice (`rows` and `statements`), so a size read
# off a refusal already counts both.
_RE_ASKS = 3
_AIM = 0.8

# The refusals that end a request wherever they are met: of the caller, of the caller's roles, and one Superset does not
# explain. Each stays the same for every later call the request makes, so no rewrite or retry can help. Superset's
# refusal of a table is not among them: it is the model's to repair.
TERMINAL_REFUSALS = (SupersetRefusedCaller, SupersetPermissionDenied, SupersetDenied)

# The column types `count(DISTINCT ...)` and `GROUP BY` refuse, by the dialect of the database behind Superset, as its
# catalogue names them, upper-cased as profiling compares them (`SupersetSqlAdapter.unaggregatable_types`).
# PostgreSQL's have no equality operator, and are spelled as `format_type` names them. Oracle's are profiling's own
# table for Oracle (`profiling._UNAGGREGATABLE`), one list whichever adapter reads an Oracle source: its large objects,
# LONG and LONG RAW, VECTOR, XMLTYPE and the row ids.
_POSTGRES_UNAGGREGATABLE = frozenset({"JSON", "XML", "POINT", "LINE", "LSEG", "BOX", "PATH", "POLYGON", "CIRCLE"})
_ORACLE_UNAGGREGATABLE = _UNAGGREGATABLE["oracle"]


class SupersetSqlAdapter:
    """Runs statements on one Superset database and schema through `client`, each within `timeout_s` unless the
    caller gives its own limit. With `refresh_every_statement`, every statement, data statements included, asks
    Superset to run it rather than answer from its result cache. Its `dialect` is Superset mode's (`superset_dialect`),
    read as the adapter is built, unless the caller names another.

    Each catalogue method reads its part of the catalogue from the database on the first call, never from Superset's
    result cache, and answers every later call with a copy of what it read: `decide` reads the function inventory on
    every decision, and one adapter serves one request, so the inventory is read live once per request. A read that
    raises is not remembered, and the next call asks again."""

    def __init__(self, client: SupersetMcp, *, database_id: int, schema: str, dialect: str | None = None,
                 timeout_s: float = 30.0, refresh_every_statement: bool = False) -> None:
        self.client = client
        self.dialect = superset_dialect() if dialect is None else dialect
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
    def unaggregatable_types(self) -> frozenset[str]:
        """The column types this adapter's database cannot aggregate, by the adapter's dialect: on Oracle profiling's
        own for Oracle, PostgreSQL's otherwise. Profiling reads them in place of its own table by dialect, so a direct
        PostgreSQL adapter, which declares the same dialect and none of these, profiles as it always has."""
        return _ORACLE_UNAGGREGATABLE if self.dialect == "oracle" else _POSTGRES_UNAGGREGATABLE

    @property
    def database_id(self) -> int:
        """The Superset database every statement runs on."""
        return self._database_id

    @property
    def schema(self) -> str:
        """The schema every statement runs in, as Superset names it and its datasets record it: as stored on
        PostgreSQL, and on Oracle in SQLAlchemy's spelling (`shop` for a stored SHOP, `names.superset_name`)."""
        return self._schema

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """`sql`'s result. Raises `ResultCut` when Superset cut the result to size or would not send it at all."""
        return self._result(sql, timeout_s)

    def execute_arrow_afresh(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """`sql`'s result as the database answers it now: Superset is asked to run it (`force_refresh`) rather than
        answer from its result cache, as the plan check's reads of a filter's value need. Raises as `execute_arrow`."""
        return self._result(sql, timeout_s, force_refresh=True)

    def execute_answer(self, sql: str, timeout_s: float | None = None) -> pa.Table:
        """`sql`'s result as an answer: whole, or, when it is larger than Superset's MCP service sends in one response,
        its first rows, marked with how many (`mark_sent`), for the answer to say it stops there. Superset cuts such an
        answer to the rows that fit, or refuses it when no cut fits. A refused answer is asked again with
        `execute_sql`'s `limit`, which Superset puts in place of the statement's own LIMIT (Oracle's FETCH FIRST): each
        ask is sized from the latest refusal, below the last and below the statement's own outer row limit
        (`_outer_limit`), so a re-ask never answers more than the statement asks. A refusal that repeats the last one's
        size says the limit did not bind, the result holding fewer rows than were asked: one row is asked for, and the
        next ask is sized from it. Up to `_RE_ASKS` asks again; one still refused, a single row refused, an answer cut
        to no rows, or a statement whose row limit cannot be read raises `ResultCut`, as `execute_arrow` does for any
        answer too large.

        Only for the answer to a question, and its reruns: every other read (a check, a catalogue page, a plan's
        probe) compares or decides on all of a result, and reads it with `execute_arrow`, which a cut result never
        reaches."""
        bound = _outer_limit(sql, self.dialect)
        limit: int | None = None  # the rows asked for; None, the statement as written
        refused_at: int | None = None  # the size of the last refusal
        allowed = 0
        probing = False
        measured: pa.Table | None = None  # the one row measured, the answer's first: kept should a later ask be refused
        refused_limit = bound  # the last limit refused, which no later ask reaches
        for asked in range(_RE_ASKS + 1):
            try:
                payload = self._execute_sql(sql, timeout_s, row_limit=limit)
            except SupersetResponseTooLarge as refused:
                sizes = _refused_size(refused)
                if sizes is None or bound is None or asked == _RE_ASKS or limit == 1:
                    if measured is not None:
                        return mark_sent(measured, measured.num_rows)
                    raise ResultCut(_TOO_LARGE_TO_SEND) from refused
                if limit is not None:
                    refused_limit = limit
                sent_bytes, allowed = sizes
                if limit is not None and sent_bytes == refused_at:
                    limit, probing = 1, True
                else:
                    current = bound if limit is None else limit
                    limit = max(1, min(current - 1, math.floor(current * allowed / sent_bytes * _AIM)))
                refused_at = sent_bytes
                continue
            if payload.get("_response_truncated"):
                table = read_execute_prefix(payload)
                if not table.num_rows:
                    raise ResultCut(_TOO_LARGE_TO_SEND)
                return mark_sent(table, table.num_rows)
            table = read_execute_response(payload)
            if probing:
                probing = False
                fit = math.floor(allowed * _AIM / max(1, _payload_size(payload)))
                if fit <= 1 or asked == _RE_ASKS:
                    return mark_sent(table, table.num_rows)
                # Sized from one row, which may be narrower than the rest: kept, and the ask stays below the refused one.
                measured = table
                limit = max(2, min(fit, refused_limit - 1))
                continue
            return mark_sent(table, limit) if limit is not None and table.num_rows >= limit else table
        raise AssertionError("unreachable: the last ask returns or raises")

    def execute(self, sql: str) -> list[tuple]:
        return _tuples(self.execute_arrow(sql))

    def validate(self, sql: str) -> None:
        """The proof the proof seam asks for in place of `EXPLAIN`: the statement as a subquery that returns no rows.
        Superset checks it for access and applies its row rules as it would to the statement itself, and the database
        resolves every table and column in it without returning a row. Raises what the statement would.

        Written in the adapter's dialect: Oracle refuses `AS` before a table alias, so there the alias follows the
        subquery bare."""
        body = sql.strip().rstrip(";").rstrip()
        alias = "mnemiq_probe" if self.dialect == "oracle" else "AS mnemiq_probe"
        read_execute_response(self._execute_sql(f"SELECT * FROM ({body}) {alias} WHERE 1 = 0", None))

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
                     force_refresh: bool = False, row_limit: int | None = None) -> dict:
        _refuse_template_markers(sql)
        try:
            # Each table as Superset names it, which its access check and row rules match on (`names`).
            sql = for_superset(sql)
        except UnreadableStatement as exc:
            raise SupersetQueryFailed(f"mnemiq could not read the statement to name its tables as Superset does "
                                      f"({exc}), so it was not sent; send one SELECT") from exc
        # The text Superset renders as a template is the text sent: checked again as it is sent.
        _refuse_template_markers(sql)
        limit = _superset_timeout(self._timeout_s if timeout_s is None else timeout_s)
        request = {"database_id": self._database_id, "sql": sql, "schema": self._schema, "timeout": limit}
        if dry_run:
            request["dry_run"] = True
        if row_limit is not None:
            request["limit"] = row_limit  # Superset sets it as the statement's LIMIT, in place
        if force_refresh or self._refresh_every_statement:
            request["force_refresh"] = True
        try:
            return self.client.call("execute_sql", request, timeout_s=limit + TRANSPORT_MARGIN_S)
        except SupersetCallTimedOut as exc:
            # Superset took the statement and was still running it at the time limit plus the margin: its own limit
            # takes effect only once the database call returns, so the statement outran it. The statement's cost,
            # as Superset's own `timed_out` answer is, never Superset being unavailable.
            raise QueryTimedOut("the statement was still running in Superset past its time limit") from exc
        except TERMINAL_REFUSALS as exc:
            kept = self._terminal_refusal
            if kept is None or (isinstance(exc, SupersetRefusedCaller) and not isinstance(kept, SupersetRefusedCaller)):
                self._terminal_refusal = exc
            raise


def _refuse_template_markers(sql: str) -> None:
    """Raise `SupersetQueryFailed` for a statement holding a marker `execute_sql` renders as a template."""
    for delimiter in JINJA_DELIMITERS:
        if delimiter in sql:
            raise SupersetQueryFailed(f"the statement holds {delimiter}, which Superset reads as a template "
                                      f"marker, so it was not sent; write the statement without {delimiter}")


def _refused_size(refused: SupersetResponseTooLarge) -> tuple[int, int] | None:
    """The bytes a refused answer held and the bytes Superset allows, read off its refusal, or None when it does not
    say."""
    found = _REFUSED_SIZE.search(refused.detail or "")
    if found is None:
        return None
    sent_bytes, allowed = (int(group.replace(",", "")) for group in found.groups())
    return (sent_bytes, allowed) if sent_bytes > 0 and allowed > 0 else None


def _outer_limit(sql: str, dialect: str) -> int | None:
    """The rows `sql` asks for at most, as `dialect` reads its outer row limit, never past the deciders' cap
    (`MAX_ROWS`): the bound an answer asked again stays under. None when the statement, or its limit, cannot be read.

    PostgreSQL's limit is its LIMIT. Oracle's is its row-limiting clause, `FETCH FIRST|NEXT n ROW|ROWS ONLY`, after
    an `OFFSET m ROWS` or not, and the bound is `n` either way, the most rows it returns. Superset puts `execute_sql`'s
    `limit` in that clause's place, as it does a LIMIT's (`set_limit_value` in `superset/sql/parse.py`). A share
    (`PERCENT`) and a count that keeps its ties (`WITH TIES`) are no number of rows, so neither is read: 15 rows asked
    in place of a 20 PERCENT that is 10 rows would answer more than the statement asks. Nor is a clause that states no
    count (`FETCH FIRST ROW ONLY`, one row), below which no re-ask could go."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError:
        return None
    if not isinstance(tree, exp.Query):
        return None
    written = tree.args.get("limit")
    if written is None:
        return MAX_ROWS
    if dialect == "oracle":
        value = _fetched_rows(written)
    else:
        value = written.expression if isinstance(written, exp.Limit) else None
    if not isinstance(value, exp.Literal) or value.is_string or not value.this.isdigit():
        return None
    return min(int(value.this), MAX_ROWS)


def _fetched_rows(written: exp.Expression) -> exp.Expression | None:
    """The count of an Oracle row-limiting clause (`exp.Fetch`, sqlglot's `count`), or None for anything else, and
    for a clause whose count is a share of the rows or keeps its ties."""
    if not isinstance(written, exp.Fetch):
        return None
    options = written.args.get("limit_options")
    if options is not None and (options.args.get("percent") or options.args.get("with_ties")):
        return None
    return written.args.get("count")


def _payload_size(payload: dict) -> int:
    """The bytes `payload` takes as JSON, near enough to Superset's own measure of a response, rows twice and all."""
    return len(json.dumps(payload, default=str).encode())


def _tuples(table: pa.Table) -> list[tuple]:
    """`table`'s rows as tuples, each in column order."""
    return [tuple(row.values()) for row in table.to_pylist()]


def _superset_timeout(timeout_s: float) -> int:
    """`timeout_s` as whole seconds in the range Superset accepts, rounded up so a statement never gets less time than
    it was given."""
    low, high = _SUPERSET_TIMEOUT_RANGE
    return max(low, min(high, math.ceil(timeout_s)))
