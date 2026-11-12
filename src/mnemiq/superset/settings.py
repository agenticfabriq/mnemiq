"""The rules a deployment must meet to start in Superset mode, checked once at startup.

In Superset mode every query runs through Superset's MCP service carrying the asking person's own verified token, so
Superset's permissions and row rules decide what each query may read. A configuration that would let anything else
answer (a door whose tokens Superset does not accept, a connection mnemiq opens itself, a write) refuses to start.
Each refusal names the setting to change and never repeats a credential, because this message reaches the console and
whatever log the operator keeps.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from mnemiq.config import without_credentials

if TYPE_CHECKING:
    from mnemiq.auth.config import AuthConfig
    from mnemiq.config import Settings

logger = logging.getLogger(__name__)

# Superset mode supports a Superset database on PostgreSQL only, so the SQL mnemiq writes and checks there is
# PostgreSQL's.
SUPERSET_DIALECT = "postgres"

# The job a snapshot enriched through Superset records, naming the database and schema it was read from. A server in
# Superset mode looks for it, so it never serves a snapshot built over a direct connection, which may hold sample
# values that Superset's row rules would have kept from the person asking.
SUPERSET_SOURCE_JOB = "superset:source"

# The environment variable that turns on Superset mode, `Settings.superset_mcp_url` under its `MNEMIQ_` prefix.
_SUPERSET_URL_VARIABLE = "MNEMIQ_SUPERSET_MCP_URL"

# Catalogue queries write the schema into their SQL as a string literal, so it is held to a plain identifier.
_PLAIN_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# The hosts that name this machine, where a token sent over http goes no further than loopback.
_THIS_MACHINE = frozenset({"127.0.0.1", "::1", "localhost"})


class SupersetConfigError(SystemExit):
    """A configuration Superset mode cannot honour. A SystemExit, so the server stops with the message alone."""


def refuse_a_direct_source(settings: Settings | None) -> None:
    """Raise in Superset mode: when `settings` are in it, or, with no settings given, when the environment sets
    MNEMIQ_SUPERSET_MCP_URL. Every place mnemiq opens a connection to the deployment's configured source calls this
    first (`source_adapter`, `adapter_for` and `run_acme`), so a deployment configured for Superset never reads that
    source with mnemiq's own credential instead, whether or not its caller passed the settings along."""
    superset_mode = settings.superset_mode if settings is not None else _superset_url_in_environment()
    if superset_mode:
        raise SupersetConfigError("MNEMIQ_SUPERSET_MCP_URL is set, and Superset mode runs every query through Superset "
                                  "and opens no database connection of its own")


def _superset_url_in_environment() -> bool:
    """Whether the environment sets MNEMIQ_SUPERSET_MCP_URL to anything but blank, read as `Settings` reads it: its
    name matched in any case, and a blank value read as unset, as `Settings.superset_mode` reads one."""
    return any(name.upper() == _SUPERSET_URL_VARIABLE and value.strip() for name, value in os.environ.items())


def check_superset_settings(settings: Settings, auth: AuthConfig | None) -> None:
    """Refuse to start in Superset mode unless every answer can come from Superset as the person asking.

    `auth` is the identity configuration `auth_config_from_settings` reads from the same settings, so a door is judged
    by whether it is open, exactly as the server will run it. With no MNEMIQ_SUPERSET_MCP_URL (direct mode) nothing
    here applies, except one thing: a Superset database id set without the URL is half a Superset configuration, and
    would otherwise start in direct mode, reading the source with mnemiq's own credential.

    An http URL to a host other than this machine is accepted with one warning: Superset and mnemiq on one private
    network, reached by a Compose service name, is the usual deployment, and that network is the operator's to judge.
    """
    if not settings.superset_mode:
        if settings.superset_database_id is not None:
            raise SupersetConfigError(
                "MNEMIQ_SUPERSET_DATABASE_ID is set but MNEMIQ_SUPERSET_MCP_URL is not, so this deployment would read "
                "the source directly with its own credential: set MNEMIQ_SUPERSET_MCP_URL to answer through "
                "Superset, or unset MNEMIQ_SUPERSET_DATABASE_ID")
        return None

    scheme, host = _scheme_and_host(settings.superset_mcp_url)
    if settings.superset_database_id is None:
        raise SupersetConfigError("MNEMIQ_SUPERSET_DATABASE_ID is required with MNEMIQ_SUPERSET_MCP_URL: the id "
                                  "Superset gives the database mnemiq answers from")
    if not _PLAIN_IDENTIFIER.fullmatch(settings.superset_schema):
        raise SupersetConfigError(
            "MNEMIQ_SUPERSET_SCHEMA must be a plain identifier, letters, digits and _ not starting with a digit, "
            f"because catalogue queries write it into their SQL as a literal (got {settings.superset_schema!r})")
    if auth is None or auth.oidc is None:
        raise SupersetConfigError(
            "Superset mode needs MNEMIQ_OIDC_ISSUER: mnemiq forwards each request's verified token to Superset "
            "unchanged, so Superset answers as the person it names, and with no issuer there is no token to verify")
    if auth.mcp_http:
        raise SupersetConfigError(
            "MNEMIQ_MCP_TRANSPORT=http is not served in Superset mode, which answers only on the HTTP API's /v1, the "
            "door that forwards each person's verified token to Superset: unset MNEMIQ_MCP_TRANSPORT")
    doors = [name for name, open_ in (("MNEMIQ_OPENWEBUI_JWT_SECRET", auth.openwebui is not None),
                                      ("MNEMIQ_OPENWEBUI_AUDIENCE", auth.oidc.openwebui_audience is not None))
             if open_]
    if doors:
        raise SupersetConfigError(
            f"{' and '.join(doors)} must be unset in Superset mode: it forwards each request's verified token to "
            "Superset unchanged, and only a token for MNEMIQ_OIDC_AUDIENCE is meant for Superset. Open WebUI's signed "
            "header is a token Open WebUI mints itself, and a token for Open WebUI's endpoint carries another audience")
    if settings.write_enabled:
        raise SupersetConfigError("MNEMIQ_WRITE_ENABLED must be off in Superset mode, which is read-only: mnemiq holds "
                                  "no connection of its own to write through")
    if settings.sources_path:
        raise SupersetConfigError(
            "MNEMIQ_SOURCES_PATH must be unset in Superset mode: mnemiq reads one database, the one "
            "MNEMIQ_SUPERSET_DATABASE_ID names, through Superset, and a manifest names sources mnemiq connects to "
            "itself")
    if settings.authz_path:
        raise SupersetConfigError(
            "MNEMIQ_AUTHZ_PATH must be unset in Superset mode, where Superset's roles and row rules decide what each "
            "person may read: mnemiq applies no policy file of its own there, so its column masks and row filters "
            "would govern nothing")
    phases = [name for name, on in (("MNEMIQ_ENRICH_FACTS", settings.enrich_facts),
                                    ("MNEMIQ_ENRICH_EXAMPLES", settings.enrich_examples)) if on]
    if phases:
        raise SupersetConfigError(
            f"{' and '.join(phases)} must be off in Superset mode, which does not support these experimental "
            "enrichment phases: the examples phase, for one, keeps a query only once it returns rows, and the rows "
            "enrichment reads are not the asking person's")

    if scheme == "http" and host not in _THIS_MACHINE:
        logger.warning(
            "MNEMIQ_SUPERSET_MCP_URL is http to %s: each person's token crosses the network to Superset unencrypted. "
            "That suits a private network such as a Compose stack's own; anywhere else use https", host)
    return None


def refuse_superset_mode(settings: Settings, entry: str, *, writes: bool = False) -> None:
    """Refuse `entry`, an entry point that answers as the one identity its settings name (a CLI command, or MCP over
    stdio), in Superset mode, which answers each question as the person whose verified token reaches the HTTP API's
    /v1. `writes` names an entry point that writes, which Superset mode never does. Outside Superset mode this checks
    what `check_superset_settings` checks there: a Superset database id set without the URL."""
    if settings.superset_mode:
        if writes:
            raise SupersetConfigError(f"{entry} does not run in Superset mode, which is read-only: mnemiq makes no write "
                                      "through Superset, and holds no connection of its own to write through")
        raise SupersetConfigError(
            f"{entry} does not run in Superset mode: it answers as the one identity its settings name, and Superset "
            "mode answers each question as the person whose verified token reaches the HTTP API's /v1. Serve that "
            "with `mnemiq serve --http`, or unset MNEMIQ_SUPERSET_MCP_URL to answer directly")
    check_superset_settings(settings, None)


def _scheme_and_host(url: str) -> tuple[str, str]:
    """MNEMIQ_SUPERSET_MCP_URL's scheme and host, refused unless it is plainly an http or https URL naming a host.

    Every query and each person's token go to this URL, so it is refused where the client that connects could read it
    differently from this check: a URL parser drops whitespace and control characters without a word, and reads a
    character outside ASCII by its own rules.
    """
    problem = None
    try:
        parts = urlsplit(url)
        valid_port = parts.port is None or parts.port > 0
    except ValueError:
        parts, valid_port = None, False
    if any(not "!" <= ch <= "~" for ch in url):
        problem = "it holds whitespace, a control character or a character outside ASCII"
    elif parts is None or not valid_port:
        problem = "it is not a URL with a valid port"
    elif parts.scheme not in ("http", "https"):
        problem = "its scheme is not http or https"
    elif "@" in parts.netloc:
        problem = "it carries a user name or password, where mnemiq sends Superset each person's own token instead"
    elif not parts.hostname:
        problem = "it names no host"
    elif "?" in url or "#" in url:
        # Neither can appear in a scheme or a host, so either one starts a query or a fragment, an empty one included.
        problem = "it has a query or a fragment"
    if problem:
        raise SupersetConfigError(f"MNEMIQ_SUPERSET_MCP_URL must be the http or https URL of Superset's MCP service: "
                                  f"{problem} (got {without_credentials(url)!r})")
    return parts.scheme, parts.hostname
