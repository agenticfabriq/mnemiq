"""Reading what Superset's `execute_sql` tool answers: a result into an Arrow table, and a failure into the exception
that says whose it is to fix.

`execute_sql` (`superset/mcp_service/sql_lab/tool/execute_sql.py`) answers `success`. A success carries the statement's
rows, one JSON object per row keyed by column name, and its columns, each a name and the pandas dtype the column had in
Superset's result (`_data_to_statement_data`). A failure carries `error`, Superset's text, and `error_type`: the query's
own status (`timed_out`, `stopped`, `failed`) or the name of a `SupersetErrorType` (`superset/errors.py`).

Each exception here carries mnemiq's own sentence as its message, except `SupersetQueryFailed`, which carries Superset's
text on purpose: the failure is the statement's own, and the model repairs the statement from it. A refusal of access
can name a table the person asking cannot see, so its text goes to the log and nowhere else.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

import pyarrow as pa

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.superset.client import SupersetUnavailable

logger = logging.getLogger(__name__)

# The ending of every `SupersetErrorType` that refuses access (`TABLE_SECURITY_ACCESS_ERROR`, its datasource, database
# and query kin). A database Superset does not find reads the same way: the caller cannot read through it either, and
# Superset's text for it points at a tool the model does not have.
_ACCESS_REFUSAL_SUFFIX = "_SECURITY_ACCESS_ERROR"
_DATABASE_NOT_FOUND = "DATABASE_NOT_FOUND_ERROR"
# `execute_sql` catches the database asking for an OAuth2 sign-in of its own (`OAuth2RedirectError`), and that sign-in's
# setup failing (`OAuth2Error`), and answers each as a payload. Neither is the statement's fault, and mnemiq can give
# no such sign-in, so both read as Superset being unable to answer.
_OAUTH2 = frozenset({"OAUTH2_REDIRECT", "OAUTH2_REDIRECT_ERROR"})

# Superset reports a column's pandas dtype, case and width included: `int64`, the nullable `Int64`, `float32`.
_INTEGER = re.compile(r"[Ii]nt\d*")
_FLOAT = re.compile(r"[Ff]loat\d*")
_BOOLEAN = frozenset({"bool", "boolean"})
# A timestamp column, `datetime64[<unit>]`, or `datetime64[<unit>, <zone>]` when its values carry a zone.
_TIMESTAMP = re.compile(r"datetime64\[\w+(?P<zone>,[^\]]+)?\]")

_CUT = "Superset cut this result to fit its response size; aggregate, or select fewer rows or columns."


class ResultCut(RuntimeError):
    """The result was larger than Superset's MCP service sends: cut to fit (`_response_truncated`), so the rows that
    arrived are not all of them, or not sent at all (the adapter raises this for that too). Never read as the answer,
    since a count or a sum over part of the rows reads as one over all of them: the statement's to make smaller."""


class SupersetAccessRefused(RuntimeError):
    """Superset refused access to something the statement reads. Its message is mnemiq's own: Superset's text can name
    a table the person asking cannot see, so it is logged and goes nowhere else."""


class SupersetQueryFailed(RuntimeError):
    """The statement's own fault, which the model repairs: Superset's or the database's text, or mnemiq's sentence for
    a statement mnemiq would not send or could not read the result of."""


def read_execute_response(payload: dict) -> pa.Table:
    """The table a successful `execute_sql` payload carries, or the exception the payload means."""
    check_execute_response(payload)
    if payload.get("_response_truncated"):
        raise ResultCut(_CUT)
    columns, rows = payload.get("columns"), payload.get("rows")
    if columns is None or rows is None:
        # Superset's answer to statements none of which returns rows: no SELECT among them.
        raise SupersetQueryFailed("the statement returned no result set; send one SELECT")
    return table_from_payload(columns, rows)


def check_execute_response(payload: dict) -> None:
    """Raise what a failed `execute_sql` payload means; return for a successful one, whatever it carries."""
    if payload.get("success") is True:
        return
    error_type = payload.get("error_type") or ""
    error = payload.get("error") or ""
    if error_type == "timed_out":
        raise QueryTimedOut("Superset ended the statement at its time limit")
    if error_type == "stopped":
        raise QueryCancelled("Superset reports the statement was stopped before it finished")
    if error_type.endswith(_ACCESS_REFUSAL_SUFFIX) or error_type == _DATABASE_NOT_FOUND:
        logger.warning("Superset refused access to something a statement reads (%s): %s", error_type, error)
        raise SupersetAccessRefused("Superset refused access to something this query reads.")
    if error_type in _OAUTH2:
        raise SupersetUnavailable("The database behind Superset asks for an OAuth2 sign-in of its own, which mnemiq "
                                  "cannot give", detail=error)
    raise SupersetQueryFailed(error or "Superset reported that the statement failed, and gave no reason")


def table_from_payload(columns: list[dict], rows: list[dict]) -> pa.Table:
    """`rows` as an Arrow table with one column for each of `columns`, in their order, typed by the dtype Superset
    reports for it. No rows is an empty table with the columns."""
    names = [column["name"] for column in columns]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            # Each row is keyed by column name, so two columns of one name cannot both be read from it.
            raise SupersetQueryFailed(f"two output columns share the name {name}; give each its own alias")
        seen.add(name)
    arrays = [_array(column["type"], [row.get(column["name"]) for row in rows]) for column in columns]
    return pa.Table.from_arrays(arrays, names=names)


def _array(dtype: str, values: list) -> pa.Array:
    if _INTEGER.fullmatch(dtype):
        return pa.array(values, type=pa.int64())
    if _FLOAT.fullmatch(dtype):
        return pa.array(values, type=pa.float64())
    if dtype in _BOOLEAN:
        return pa.array(values, type=pa.bool_())
    if (timestamp := _TIMESTAMP.fullmatch(dtype)) is not None:
        return _timestamps(values, zoned=timestamp.group("zone") is not None)
    return _inferred(values)


def _timestamps(values: list, *, zoned: bool) -> pa.Array:
    """A timestamp column's values, which arrive as ISO 8601 text.

    `_data_to_statement_data` takes the rows from the result's DataFrame (`to_dict(orient="records")`), so each value
    is a pandas `Timestamp`. `JsonSafeRows` (`superset/mcp_service/utils/serialization.py`) passes a datetime on as it
    is, and pydantic writes it as ISO 8601: `2024-03-01T09:30:00`, a fraction of a second as six digits, a zone as `Z`
    or `+02:00`, and a null timestamp as null. The unit is microseconds, the finest pydantic writes. A zoned column
    keeps its instants, in UTC.

    A result Superset serves from its own cache reaches `_data_to_statement_data` as a list of rows, not a DataFrame,
    and every column is reported `object`; there a timestamp column is left to `_inferred` and reads as text."""
    parsed = [None if value is None else datetime.fromisoformat(value) for value in values]
    return pa.array(parsed, type=pa.timestamp("us", tz="UTC") if zoned else pa.timestamp("us"))


def _inferred(values: list) -> pa.Array:
    """An `object` column, or a dtype with no type of its own here, typed as pyarrow reads its values. A column whose
    values pyarrow cannot give one type (numbers and strings mixed) is text."""
    try:
        return pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError):
        return pa.array([_as_text(value) for value in values], type=pa.string())


def _as_text(value):
    """A value as text, as it arrived: a string unchanged, anything else in its JSON spelling."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
