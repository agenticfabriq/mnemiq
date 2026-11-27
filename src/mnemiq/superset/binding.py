"""On Oracle, a person's statements are sent only once Superset's connection for them is shown to run them as the schema
the snapshot was read from.

Superset mode on Oracle answers from one schema (MNEMIQ_SUPERSET_SCHEMA), and the snapshot it serves was read from that
schema on one Oracle service, as the snapshot's `superset:oracle` job records them. Every name mnemiq writes and every
check it makes of a statement is made for that schema, while Oracle reads a statement's unqualified names in the
session's current schema, and Superset opens each person's session as its database connection says: with one login for
everyone, or, where the database impersonates its users, as the person. So mnemiq sends a person's statements only once
Superset's session for them is shown to log in as that schema's owner, keep that schema as its current schema, and reach
the service the snapshot was read from.

`binding_refusal` asks Superset, as the person, through the adapter their request runs on:

- `get_database_info` for the database Superset mode answers from (MNEMIQ_SUPERSET_DATABASE_ID): its backend must be
  `oracle`, and nothing else is asked of a database Superset describes otherwise;
- `BINDING_SQL`, the session's login user, service and current schema, sent as every Oracle statement is
  (`SupersetSqlAdapter`: each table named as Superset names it, Superset's dry run, the extension's `tables_read` and
  `table_rules`), and asked to run afresh rather than be answered from Superset's result cache, which could hold a
  session read before the connection changed. The service must be the job's, and the login user and the current
  schema must each be the job's schema.

Each value is compared as Oracle stores it: the job records the service and the schema as the enrichment's own session
read them (`SYS_CONTEXT`), and the binding statement reads the person's the same way, so no name is mapped between
spellings. Superset's spelling of the schema (`shop` for a stored `SHOP`) is another name here, and is refused.

`Runtime.bound` keeps a bound answer for the person; a refusal and a read that failed are not kept.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from mnemiq.adapters.base import QueryCancelled, QueryTimedOut
from mnemiq.superset.client import DEFAULT_TIMEOUT_S, SupersetRefusal, SupersetToolError, SupersetUnavailable
from mnemiq.superset.payload import ResultCut, SupersetAccessRefused, SupersetQueryFailed

logger = logging.getLogger(__name__)

# The session's login user, the service it reached and its current schema, each as Oracle stores it. DUAL is the only
# table it reads.
BINDING_SQL = "SELECT USER, SYS_CONTEXT('USERENV','SERVICE_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM DUAL"

# The backend Superset's `get_database_info` names an Oracle database by: `DatabaseInfo.backend`
# (`superset/mcp_service/database/schemas.py`), the backend name of the database's SQLAlchemy URL.
ORACLE_BACKEND = "oracle"

# What a person is told when their connection was not shown to be the snapshot's. Why goes to the log only: it names
# the deployment's login, service and schema.
NOT_BOUND = ("Superset's connection to Oracle was not shown to run your statements as the schema this deployment answers "
             "from, so nothing is answered for you until an operator has Superset log in to Oracle as that schema's "
             "owner; mnemiq's log records why.")
# What a person is told when the check could not be read.
BINDING_UNREAD = ("mnemiq could not check the connection Superset runs your statements on in Oracle, so nothing was "
                  "answered; try again. mnemiq's log records why.")


class ConnectionNotBound(SupersetRefusal):
    """A request refused because Superset's connection for the person asking was not shown to run their statements as
    the snapshot's schema, on its service. The message, also `message`, is mnemiq's sentence for the person; `detail`
    says why, and goes to the log only."""

    def __init__(self, detail: str) -> None:
        super().__init__(NOT_BOUND, detail=detail)
        self.message = NOT_BOUND


class BindingUnread(SupersetUnavailable):
    """The check of a person's connection got no answer it could read: the request is refused, and the next one asks
    again. `detail` says why, and goes to the log only."""

    def __init__(self, detail: str) -> None:
        super().__init__(BINDING_UNREAD, detail=detail)


def binding_refusal(adapter, *, database_id: int, service: str, schema: str) -> str | None:
    """None when Superset, asked as the person whose token `adapter` carries, describes database `database_id` as
    Oracle and runs their statements in a session that logs in as `schema`, keeps it as its current schema and reached
    `service` (each as Oracle stores it); else why not, in words for the log.

    Raises `BindingUnread` where an answer cannot be read, or the binding statement got no result (a database error, a
    time limit, a result cut, a statement mnemiq would not send); Superset not answering (`SupersetUnavailable`), a
    refusal of the person (`SupersetRefusal`) and a Superset without the extension's tools (`ExtensionMissing`) raise as
    themselves. Superset's refusal of access to the statement's table is a refusal, since no rewrite of it gets past it."""
    try:
        info = adapter.client.call("get_database_info", {"identifier": database_id}, timeout_s=DEFAULT_TIMEOUT_S)
    except SupersetToolError as exc:
        raise BindingUnread(f"get_database_info failed: {exc} {exc.detail}") from exc
    if "error" in info or "error_type" in info:
        logger.warning("Superset did not describe database %s to this person: %s (%s)", database_id,
                       info.get("error"), info.get("error_type"))
        return f"Superset did not describe database {database_id} to this person ({info.get('error_type')})"
    found, backend = info.get("id"), info.get("backend")
    # `type(...) is int`, since a JSON true reads as a bool, which Python also counts an int.
    if type(found) is not int or found != database_id or not isinstance(backend, str):
        raise BindingUnread(f"Superset's description of database {database_id} could not be read: id {found!r}, "
                            f"backend {backend!r}")
    if backend != ORACLE_BACKEND:
        return f"Superset describes database {database_id} as backend {backend!r}, not {ORACLE_BACKEND!r}"
    try:
        rows = [tuple(row.values()) for row in adapter.execute_arrow_afresh(BINDING_SQL).to_pylist()]
    except SupersetAccessRefused as exc:
        return f"Superset refused this person the statement that reads the session ({exc})"
    except (SupersetQueryFailed, SupersetToolError, QueryTimedOut, QueryCancelled, ResultCut) as exc:
        raise BindingUnread(f"the statement that reads the session got no result: {type(exc).__name__}: {exc}") from exc
    if len(rows) != 1 or len(rows[0]) != 3 or not all(isinstance(value, str) and value for value in rows[0]):
        raise BindingUnread(f"the statement that reads the session answered {len(rows)} rows, not one row of three "
                            "values each a name")
    return session_refusal(rows[0], service=service, schema=schema)


def session_refusal(session: Sequence, *, service: str, schema: str) -> str | None:
    """None when `session`, the binding statement's row (login user, service, current schema), logs in as `schema`,
    keeps it as its current schema and reached `service`, each compared exactly, as Oracle stores it; else why not."""
    user, reached, current = session
    reasons = []
    if reached != service:
        reasons.append(f"its service is {reached!r}, and the snapshot was read from {service!r}")
    if user != schema:
        reasons.append(f"it logs in as {user!r}, not as the owner of the snapshot's schema {schema!r}")
    if current != schema:
        reasons.append(f"its current schema is {current!r}, not the snapshot's {schema!r}")
    return f"Superset's session for this person: {'; '.join(reasons)}" if reasons else None
