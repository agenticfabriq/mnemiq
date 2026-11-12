"""The service key, and the adapter `mnemiq enrich` reads the source through in Superset mode.

Enrichment builds the snapshot offline, before anyone asks anything, so it reads the source as a Superset user of its
own: a service user, whose API key is in a file `MNEMIQ_SUPERSET_SERVICE_KEY_FILE` names. This module reads that file
and nothing else does. The key is not a setting, and only the CLI imports this module (`tests/test_superset_guards.py`
holds every module to that), so the ask path, which carries each person's own delegated token, never holds it.

What the service user reads describes the source, not what any one person may read. So enrichment through Superset
reads the catalogue and the counts and harvests no values, and records the database and schema it read in the
snapshot's `superset:source` job, which a server in Superset mode requires before it serves the snapshot.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from collections.abc import Mapping
from typing import TYPE_CHECKING

from mnemiq.superset.adapter import TRANSPORT_MARGIN_S, SupersetSqlAdapter
from mnemiq.superset.client import (
    DEFAULT_TIMEOUT_S,
    Bearer,
    SupersetMcp,
    SupersetRefusal,
    SupersetRefusedCaller,
    SupersetToolError,
    SupersetUnavailable,
)
from mnemiq.superset.settings import SupersetConfigError, check_superset_enrichment

if TYPE_CHECKING:
    from mnemiq.config import Settings

logger = logging.getLogger(__name__)

# The environment variable naming the file that holds the service key. Read here and nowhere else.
SERVICE_KEY_ENV = "MNEMIQ_SUPERSET_SERVICE_KEY_FILE"

# The time limit each enrichment statement is given: the most `ExecuteSqlRequest.timeout` accepts
# (`superset/mcp_service/sql_lab/schemas.py`), since profiling a large table may rightly run long.
ENRICHMENT_TIMEOUT_S = 300.0

# The one backend Superset mode reads, as `get_database_info` names it: `DatabaseInfo.backend`
# (`superset/mcp_service/database/schemas.py`) is the backend name of the database's SQLAlchemy URL.
_POSTGRES = "postgresql"

# A key as a header carries it: visible ASCII, no spaces.
_KEY_BYTES = re.compile(rb"[!-~]+")

# The most a key file may hold, in bytes: far more than any key, so a larger file is not one.
_KEY_FILE_LIMIT = 64 * 1024


class ServiceKey(Bearer):
    """The API key of the Superset user enrichment reads the source as. Like every `Bearer`, it never prints."""


def service_credential_from_env(environ: Mapping[str, str] = os.environ) -> ServiceKey:
    """The service key in the file `MNEMIQ_SUPERSET_SERVICE_KEY_FILE` names.

    Refused with a sentence, which never holds the key, unless the variable names a regular file that grants its group
    and others nothing, no larger than `_KEY_FILE_LIMIT`, holding one key: visible ASCII with no spaces, once
    surrounding whitespace is stripped. The file is checked as it was opened, so the file read is the file checked,
    and opening one that is not a regular file never waits on it.

    The variable's value is repeated in a refusal only once it has opened as a file: an operator may have put the key
    itself there in place of its file's path, and a value that opens nothing says only why, as the system put it. That
    refusal is raised apart from the system's error, which names what it tried to open."""
    named = environ.get(SERVICE_KEY_ENV, "")
    if not named.strip():
        raise SupersetConfigError(f"{SERVICE_KEY_ENV} is not set: enriching through Superset reads the source as a "
                                  "service user, and that variable names the file holding its API key")
    reason = None
    try:
        descriptor = os.open(named, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        reason = exc.strerror or "the system gave no reason"
    except ValueError:  # a NUL, which no path holds
        reason = "it is not a path"
    if reason is not None:
        raise SupersetConfigError(f"the file {SERVICE_KEY_ENV} names cannot be opened ({reason})")
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise SupersetConfigError(f"{SERVICE_KEY_ENV} names {named!r}, which is not a regular file")
        if mode & 0o077:
            raise SupersetConfigError(f"{SERVICE_KEY_ENV} names {named!r}, which grants its group or others access "
                                      f"(mode {stat.S_IMODE(mode):04o}): make it its owner's alone, with chmod 600")
        content = b""
        while len(content) <= _KEY_FILE_LIMIT and (chunk := os.read(descriptor, _KEY_FILE_LIMIT + 1 - len(content))):
            content += chunk
    finally:
        os.close(descriptor)
    if len(content) > _KEY_FILE_LIMIT:
        raise SupersetConfigError(f"{SERVICE_KEY_ENV} names {named!r}, which is larger than "
                                  f"{_KEY_FILE_LIMIT // 1024} KiB, far larger than a key")
    content = content.strip()
    if not content:
        raise SupersetConfigError(f"{SERVICE_KEY_ENV} names {named!r}, which holds no key")
    if not _KEY_BYTES.fullmatch(content):
        raise SupersetConfigError(f"{SERVICE_KEY_ENV} names {named!r}, whose key holds a space or a character other "
                                  "than visible ASCII, which a header cannot carry")
    return ServiceKey(content.decode("ascii"))


def enrichment_adapter(settings: Settings, credential: ServiceKey) -> SupersetSqlAdapter:
    """The adapter enrichment reads the source through: Superset's MCP service at `settings.superset_mcp_url`, called
    with the service key, on the configured database and schema.

    Every statement it sends asks Superset to run it afresh (`force_refresh`), so a re-enrich reads the source as it is
    now. Each is given `ENRICHMENT_TIMEOUT_S`, and the adapter adds its own margin for the transport; the client's own
    deadline is no shorter, so it never cuts a statement's short. The backend check is a metadata call, and waits only
    the client's default deadline.

    Before it returns, it asks Superset which backend the database is, and refuses anything but PostgreSQL. Refuses
    anything but a `ServiceKey`, and settings `check_superset_enrichment` refuses, before Superset is asked anything."""
    if not isinstance(credential, ServiceKey):
        raise TypeError(f"enrichment reads the source with the service key, not a {type(credential).__name__}")
    check_superset_enrichment(settings)
    database_id = settings.superset_database_id
    client = SupersetMcp(settings.superset_mcp_url, credential, timeout_s=ENRICHMENT_TIMEOUT_S + TRANSPORT_MARGIN_S)
    _require_postgres(client, database_id)
    return SupersetSqlAdapter(client, database_id=database_id, schema=settings.superset_schema,
                              timeout_s=ENRICHMENT_TIMEOUT_S, refresh_every_statement=True)


def _require_postgres(client: SupersetMcp, database_id: int) -> None:
    """Refuse unless Superset's `get_database_info` describes database `database_id` as PostgreSQL.

    The request is `GetDatabaseInfoRequest`'s `identifier`, the database's numeric id, and what is read of the answer
    is `DatabaseInfo`'s `id` and `backend` (`superset/mcp_service/database/schemas.py`). Superset answers a database
    it does not find, or one the caller may not see, with a `DatabaseError` (`error`, `error_type`) in place of the
    description. Whatever Superset said is logged, and the refusal says only mnemiq's sentence."""
    try:
        info = client.call("get_database_info", {"identifier": database_id}, timeout_s=DEFAULT_TIMEOUT_S)
    except SupersetRefusedCaller as exc:
        raise SupersetConfigError(f"Superset did not accept the service key the file {SERVICE_KEY_ENV} names: check "
                                  "that the key is current and that its user exists in Superset") from exc
    except SupersetRefusal as exc:
        raise SupersetConfigError(f"Superset refused to describe database {database_id} to the service user ({exc}): "
                                  "check that the key is current and that its user may see the database") from exc
    except (SupersetUnavailable, SupersetToolError) as exc:
        raise SupersetConfigError(f"Superset could not describe database {database_id}, so enrichment read nothing: "
                                  f"{exc}") from exc
    if "error" in info or "error_type" in info:
        logger.warning("Superset did not describe database %s: %s (%s)", database_id, info.get("error"),
                       info.get("error_type"))
        raise SupersetConfigError(f"Superset did not describe database {database_id}, and its answer is logged: check "
                                  "MNEMIQ_SUPERSET_DATABASE_ID, and that the service user may see the database")
    found, backend = info.get("id"), info.get("backend")
    if type(found) is not int or found != database_id or not isinstance(backend, str):
        logger.warning("Superset's description of database %s could not be read: id %r, backend %r", database_id,
                       found, backend)
        raise SupersetConfigError(f"Superset's description of database {database_id} could not be read, and is "
                                  "logged: enrichment read nothing")
    if backend != _POSTGRES:
        logger.warning("Superset describes database %s as backend %r", database_id, backend)
        raise SupersetConfigError(f"Superset database {database_id} is not a PostgreSQL database (its backend is "
                                  "logged), and Superset mode reads PostgreSQL only")
