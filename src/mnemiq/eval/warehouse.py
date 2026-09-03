"""Shared plumbing for benchmarking a vendor NL->SQL product on BIRD.

The vendor engines (Snowflake Cortex Analyst, Databricks Genie) answer a question with SQL --
or with a clarification, which is a *deferral*, not a wrong answer. This module holds the
per-question record both runners produce and the grader consumes, so a Snowflake run and a
Databricks run are scored by identical rules.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class VendorResult:
    """One question, one vendor answer."""

    case_id: str
    db_id: str
    question: str
    gold_sql: str
    difficulty: str = ""

    #: what the vendor produced. Exactly one of these is meaningful.
    sql: str | None = None
    deferral: str | None = None
    error: str | None = None

    latency_ms: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        if self.sql:
            return "sql"
        if self.deferral:
            return "deferred"
        return "error"


def load_results(path: str) -> dict[str, VendorResult]:
    """Existing results keyed by case_id, so a re-run resumes instead of re-paying."""
    if not path or not os.path.exists(path):
        return {}
    out: dict[str, VendorResult] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["case_id"]] = VendorResult(**rec)
    return out


def append_result(path: str, result: VendorResult) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(asdict(result)) + "\n")


def databricks_workspace(*, host: str | None = None, profile: str | None = None):
    """A Databricks client, preferring browser OAuth over a stored token.

    With `host`, the SDK runs the OAuth user-to-machine flow: a browser opens, the workspace
    authorizes, and the SDK caches the result in ~/.databricks/token-cache.json. No personal
    access token is ever generated, pasted into a config file, or left to rotate.

    Without `host`, falls back to a named profile in ~/.databrickscfg.
    """
    from databricks.sdk import WorkspaceClient

    if host:
        return WorkspaceClient(host=host, auth_type="external-browser")
    return WorkspaceClient(profile=profile or "DEFAULT")


def schema_for(db_id: str) -> str:
    """The warehouse schema name for a benchmark database.

    Lowercased, because the loader creates schemas that way and Databricks resolves names
    case-insensitively. Anything outside [a-z0-9_] becomes an underscore: Spider 2.0-lite ships
    `Db-IMDB` and `sqlite-sakila`, and a hyphen is not legal in an unquoted Unity Catalog
    identifier -- the CREATE SCHEMA parses as a subtraction and fails. Every script that names
    a schema must agree on this mapping, or the loader writes one name and the runner reads
    another.
    """
    return "".join(c if c.isalnum() or c == "_" else "_" for c in db_id.lower())


def databricks_sql_connection(workspace, warehouse_id: str | None = None):
    """(connection, warehouse_id) for a SQL warehouse, authenticated the way `workspace` is.

    Not `access_token=workspace.config.token`: that field is only populated for personal-access-
    token auth and is None under OAuth, so the connect fails before a single query runs. Passing
    the SDK's own `authenticate` as a credentials provider covers both, and -- because the
    connector calls the header factory per request rather than once at connect -- the hour-long
    OAuth access token is refreshed underneath a run that takes six.
    """
    from databricks import sql as dbsql

    if not warehouse_id:
        warehouses = [w for w in workspace.warehouses.list()]
        if not warehouses:
            raise RuntimeError("no SQL warehouse in this workspace")
        # Prefer a warehouse that is already running: a cold start is two minutes the caller
        # pays before the first question, and serverless warehouses idle down between runs.
        running = [w for w in warehouses if "RUNNING" in str(w.state)]
        warehouse_id = (running or warehouses)[0].id

    connection = dbsql.connect(
        server_hostname=workspace.config.host.replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: workspace.config.authenticate,
    )
    return connection, warehouse_id
