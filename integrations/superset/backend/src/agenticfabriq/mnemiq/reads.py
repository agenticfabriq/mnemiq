"""The extension's read tools, one function per tool, as `writes` has the write tools: `tools.py` registers them on
Superset's MCP service, which runs each as the caller its token names (`mcp_auth_hook`, `superset/mcp_service/auth.py`).

They change nothing. One, `session_binding`, opens a connection to the database as the caller and runs one fixed
statement that reads no data; the others read Superset's own metadata. Each keeps the write tools' rules that apply to
a read:

- Its answer is short and structured; Superset's own text never reaches it.
- An object the caller may not read is left out of an answer about several, or answered as one that does not exist,
  by the same check Superset's own tools use (`can_access_datasource`, `can_access_chart`, `can_access_dashboard`),
  so an object out of reach reads as one that is not there. Except `table_rules`, which answers whether a row rule
  applies to the caller on a table through every dataset of it Superset's own lookup finds, the ones the caller may
  not read among them, and only that: never the rule, its clause, its roles or the dataset.
- It takes no user id from the caller.

An editor's label (`Subject.label`) is returned only for an object the caller may read, and only in the answer
itself, which mnemiq passes to the person's panel; no label goes into a log line or an error.

Superset's MCP response size guard (`superset/mcp_service/middleware.py`, `ResponseSizeGuardMiddleware`) does not
truncate an extension tool's answer: over its limit (50,000 bytes by default) the call fails with Superset's own text.
So every read tool keeps its answer under `MAX_ANSWER_BYTES` itself: `row_rules`, whose at most 20 entries are a few
bytes each, by its size alone; `get_layout` refuses a layout over it with `{"too_large": True}`, and `table_rules`,
which echoes the names it is asked, a request whose answer would be over it, the same way, as `tables_read` does an
answer naming more tables than fit; and the tools that answer a
list of objects by one helper (`fit`), which keeps their envelope and answers each entry that does not fit as
`{<id field>: N, "too_large": True}` (entries fill in request order, and once one does not fit the rest are marked). A
caller treats a marked entry as unanswered.

Superset is imported inside the tools, as in `writes`, so this module imports without it.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

from agenticfabriq.mnemiq import writes
from agenticfabriq.mnemiq.writes import _Field

logger = logging.getLogger(__name__)

MAX_DATASETS = 20
MAX_CHARTS = 100
MAX_DASHBOARDS = 20
MAX_TABLES = 50
# A table's name, and a schema's, as long as Superset holds one (`SqlaTable.table_name` and `SqlaTable.schema`).
MAX_TABLE_NAME = 250
MAX_SCHEMA_NAME = 255
# The longest statement `tables_read` reads, in bytes of its UTF-8 text: rc2's own default bound on a script it parses
# (`SQL_MAX_PARSE_LENGTH` in `superset/config.py`).
MAX_SQL_BYTES = 1_000_000
# The markers Superset renders a text as a template by, which mnemiq never sends (`JINJA_DELIMITERS` in
# `mnemiq/superset/adapter.py`): `tables_read` reads no text holding one, wherever it stands in it.
TEMPLATE_MARKERS = ("{{", "{%", "{#")
# The longest Oracle service name `session_binding` takes, as a schema's.
MAX_SERVICE_NAME = MAX_SCHEMA_NAME
# What `session_binding` runs, word for word: the session's login user, the service it reached and its current schema,
# each as Oracle stores it. DUAL is the only table it reads.
SESSION_SQL = "SELECT USER, SYS_CONTEXT('USERENV','SERVICE_NAME'), SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM DUAL"
# The backend of a database `session_binding` reads a session of, as rc2's `Database.backend` names it.
ORACLE = "oracle"
# The rule `tables_read` renders a statement with on every table it reads: a predicate true of every row, read by the
# statement's own `parse_predicate`, so the rendering is what Superset would run were a rule to apply to each table.
PLACEHOLDER_RULE = "1 = 1"
# The only nodes a template processor's environment parses plain text to (`jinja2.nodes`): `tables_read` reads no text
# its database's processor would read as anything more.
_PLAIN_TEXT_NODES = frozenset({"Template", "Output", "TemplateData"})
# An answer larger than this, in bytes of its JSON text, is not sent: well under the guard's 50,000.
MAX_ANSWER_BYTES = 40_000
MAX_LAYOUT_BYTES = MAX_ANSWER_BYTES
TOO_LARGE = {"too_large": True}


def _distinct_ids(limit: int):
    def read(value: Any) -> list[int]:
        found = list(dict.fromkeys(writes._ids(value)))
        if len(found) > limit:
            raise ValueError("too many ids")
        return found

    return read


def _distinct_names(limit: int):
    def read(value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("not a list")
        found = list(dict.fromkeys(writes._text(MAX_TABLE_NAME)(item) for item in value))
        if len(found) > limit:
            raise ValueError("too many names")
        return found

    return read


def _statement_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > MAX_SQL_BYTES:
        raise ValueError("not a statement's text")
    return value


# Every read tool's request, field by field.
REQUESTS: dict[str, dict[str, _Field]] = {
    "dataset_metrics": {"dataset_ids": _Field(_distinct_ids(MAX_DATASETS))},
    "editors_of": {"chart_ids": _Field(_distinct_ids(MAX_CHARTS), False, []),
                   "dataset_ids": _Field(_distinct_ids(MAX_DATASETS), False, []),
                   "dashboard_ids": _Field(_distinct_ids(MAX_DASHBOARDS), False, [])},
    "get_layout": {"dashboard_id": writes._DASHBOARD_ID},
    "row_rules": {"dataset_ids": _Field(_distinct_ids(MAX_DATASETS))},
    "table_rules": {"database_id": _Field(writes._an_id), "schema": _Field(writes._text(MAX_SCHEMA_NAME)),
                    "tables": _Field(_distinct_names(MAX_TABLES))},
    "tables_read": {"database_id": _Field(writes._an_id), "schema": _Field(writes._text(MAX_SCHEMA_NAME)),
                    "sql": _Field(_statement_text)},
    "session_binding": {"database_id": _Field(writes._an_id), "service": _Field(writes._text(MAX_SERVICE_NAME)),
                        "schema": _Field(writes._text(MAX_SCHEMA_NAME))},
}


def _size(answer: Any) -> int:
    return len(json.dumps(answer).encode())


def fit(entries: list, build, mark) -> dict:
    """`build(entries)`'s answer with as many of `entries`, in order, as keep the answer sent within
    `MAX_ANSWER_BYTES`; from the first that does not fit on, each entry is `mark(entry)` in its place. What is measured
    is what would be sent: the entries kept, this one, and the marks of every entry after it. So whenever an entry is
    kept the answer that ends the loop is no larger than the one just measured, and the final answer is within the
    cap (the marks alone, which a request's id limits, are far below it)."""
    marks = [mark(entry) for entry in entries]
    kept = 0
    for index, entry in enumerate(entries):
        if _size(build([*entries[:index + 1], *marks[index + 1:]])) > MAX_ANSWER_BYTES:
            break
        kept = index + 1
    return build([*entries[:kept], *marks[kept:]])


def read(tool: str, request: Any) -> dict:
    """`request` read for the read tool `tool`; raises `Refused` as `writes.read` does."""
    return writes.read(tool, request, REQUESTS)


def _certification(metric: Any) -> dict:
    """The certification a metric's `extra` holds (`CertificationMixin` in rc2: `extra.certification`, with
    `certified_by` and `details`); empty when there is none or `extra` is not a JSON object."""
    extra = writes._stored_object(getattr(metric, "extra", None)) or {}
    found = extra.get("certification")
    return found if isinstance(found, dict) else {}


def metric_answer(metric: Any) -> dict:
    """One metric as the tool answers it. `currency` is rc2's own object, `{symbol, symbolPosition}`, as `CurrencyType`
    reads it; an empty object (what rc2 reads from text it cannot parse) and anything else stored there are answered
    as none."""
    certification = _certification(metric)
    currency = getattr(metric, "currency", None)
    return {"metric_name": metric.metric_name, "expression": metric.expression, "d3format": metric.d3format,
            "currency": (dict(currency) or None) if isinstance(currency, dict) else None,
            "certified_by": certification.get("certified_by"), "certification_details": certification.get("details"),
            "warning_text": metric.warning_text}


@writes._answering("dataset_metrics", REQUESTS)
def dataset_metrics(found: dict) -> dict:
    """Every saved metric of each dataset the caller may access (`can_access_datasource`), in the order asked. A
    dataset the caller may not access, or that does not exist, is left out, so nothing tells the two apart. An empty
    request answers `{"datasets": []}` without touching Superset: mnemiq's client probes with it to tell a Superset
    without this extension from a tool that fails."""
    datasets = []
    for dataset_id in found["dataset_ids"]:
        dataset = writes._dataset(dataset_id)
        if dataset is not None:
            datasets.append({"dataset_id": dataset_id, "metrics": [metric_answer(m) for m in dataset.metrics]})
    return fit(datasets, lambda kept: {"datasets": kept}, lambda d: {"dataset_id": d["dataset_id"], **TOO_LARGE})


def _chart(chart_id: int) -> Any:
    """The chart, when the caller may read it; else None."""
    from superset.daos.chart import ChartDAO

    chart = ChartDAO.find_by_id(chart_id)
    if chart is None or not writes._security().can_access_chart(chart):
        return None
    return chart


def _labels(resource: Any) -> list[str]:
    return sorted({subject.label for subject in resource.editors})


_ID_FIELDS = {"charts": "chart_id", "datasets": "dataset_id", "dashboards": "dashboard_id"}


@writes._answering("editors_of", REQUESTS)
def editors_of(found: dict) -> dict:
    """The editors' labels of each chart, dataset and dashboard asked for that the caller may read, for the person's
    panel (who to ask for access), each with whether the caller edits it (`is_editor`), so a change refuses a
    non-editor before it changes anything. An object the caller may not read, or that does not exist, is left out.
    Asked for no dashboard, the answer holds none: as it was before dashboards were asked for."""
    security = writes._security()
    entries = [("charts", {"chart_id": chart_id, "is_editor": bool(security.is_editor(chart)),
                           "editors": _labels(chart)})
               for chart_id in found["chart_ids"] if (chart := _chart(chart_id)) is not None]
    entries += [("datasets", {"dataset_id": dataset_id, "is_editor": bool(security.is_editor(dataset)),
                              "editors": _labels(dataset)})
                for dataset_id in found["dataset_ids"] if (dataset := writes._dataset(dataset_id)) is not None]
    entries += [("dashboards", {"dashboard_id": dashboard_id, "is_editor": bool(security.is_editor(board)),
                                "editors": _labels(board)})
                for dashboard_id in found["dashboard_ids"] if (board := writes._dashboard(dashboard_id)) is not None]
    kinds = ("charts", "datasets", "dashboards") if found["dashboard_ids"] else ("charts", "datasets")

    def build(kept: list) -> dict:
        return {key: [entry for kind, entry in kept if kind == key] for key in kinds}

    def mark(pair: tuple) -> tuple:
        kind, entry = pair
        # An entry cut for size keeps whether the caller edits it, one bool: only its labels are left out.
        return kind, {_ID_FIELDS[kind]: entry[_ID_FIELDS[kind]], "is_editor": entry["is_editor"], **TOO_LARGE}

    return fit(entries, build, mark)


@writes._answering("get_layout", REQUESTS)
def get_layout(found: dict) -> dict:
    """The dashboard's stored `position_json` as the text it is stored in, and its `changed_on` as ISO 8601 text (what
    MCP's `get_dashboard_info` answers, and `set_layout` takes back as `expect_changed_on`), for a dashboard the caller
    may read; else `{"not_found": "dashboard"}`. An answer over `MAX_ANSWER_BYTES` is `{"too_large": True}`."""
    dashboard = writes._dashboard(found["dashboard_id"])
    if dashboard is None:
        return writes._not_found("dashboard")
    answer = {"dashboard_id": found["dashboard_id"], "position_json": dashboard.position_json,
              "changed_on": dashboard.changed_on.isoformat() if dashboard.changed_on is not None else None}
    if _size(answer) > MAX_ANSWER_BYTES:
        return dict(TOO_LARGE)
    return answer


@writes._answering("row_rules", REQUESTS)
def row_rules(found: dict) -> dict:
    """Whether any row rule applies to the caller on each dataset the caller may access (`can_access_datasource`), in
    the order asked: whether Superset's `security_manager.get_rls_filters` finds a rule attached to the dataset that
    applies to the caller. The guest rules of an embedded dashboard are not counted; a guest is never the caller of
    an MCP tool. Only whether: never a rule's clause, name or roles, which an Alpha may not list. A dataset the caller may not access, or that does not exist, is left out, so nothing tells the two apart.
    At most `MAX_DATASETS` short entries, so the answer is far under `MAX_ANSWER_BYTES`."""
    datasets = []
    for dataset_id in found["dataset_ids"]:
        dataset = writes._dataset(dataset_id)
        if dataset is not None:
            datasets.append({"dataset_id": dataset_id,
                             "applies": bool(writes._security().get_rls_filters(dataset))})
    return {"datasets": datasets}


@writes._answering("table_rules", REQUESTS)
def table_rules(found: dict) -> dict:
    """Whether a row rule applies to the caller on each table asked, by its name in `schema` on the database, in the
    order asked: whether Superset's own lookup of the rules a statement reading the table takes,
    `superset.utils.rls.get_predicates_for_table`, finds any for the caller. The table is looked up as `apply_rls`
    looks up a table a statement reads with no catalog of its own, `Table(name, schema, <the database's default
    catalog>)`, beside that catalog, so the answer is the lookup Superset makes for mnemiq's statements, which name no
    catalog: through every dataset of the table that lookup finds, whether or not the caller may list it. Only whether:
    never a rule's clause, name or roles, nor which dataset carries it.

    A database the caller may not see (Superset's own `DatabaseDAO` filter, as `get_database_info` finds one), or that
    does not exist, is answered `{"not_found": "database"}`, and no table is looked up. A request whose answer would be
    over `MAX_ANSWER_BYTES` (it echoes each name) is answered `{"too_large": True}` before Superset is asked."""
    if _size(_table_rules_answer(found["tables"], lambda name: False)) > MAX_ANSWER_BYTES:
        return dict(TOO_LARGE)
    from superset.daos.database import DatabaseDAO
    from superset.sql.parse import Table
    from superset.utils.rls import get_predicates_for_table

    database = DatabaseDAO.find_by_id(found["database_id"])
    if database is None:
        return writes._not_found("database")
    catalog = database.get_default_catalog()
    return _table_rules_answer(found["tables"], lambda name: bool(get_predicates_for_table(
        Table(name, found["schema"], catalog), database, catalog)))


@writes._answering("tables_read", REQUESTS)
def tables_read(found: dict) -> dict:
    """The tables Superset's own access check reads in one statement's text, as the caller: rc2's
    `process_jinja_sql` (`superset/sql/parse.py`), as `security_manager.raise_for_access` reads a statement, with the
    empty template parameters `execute_sql` renders a statement with. Each table is qualified as the executor qualifies
    a table it applies row rules to (`SQLExecutor._prepare_sql`): by the request's schema, and the database's default
    catalog, where the statement names none of its own. Answered `{"tables": [{catalog, schema, table}, ...], "ruled":
    <text>}`, the tables sorted, each once, and `ruled` the statement as rc2's own `apply_rls` renders it with
    `PLACEHOLDER_RULE` on every one of them, by the method the database's engine applies rules by (`get_rls_method`),
    then `.format()`: what Superset would run were a rule to apply to every table it reads. No rule is read.

    A text holding a template marker (`TEMPLATE_MARKERS`), in a literal or a comment as anywhere, is answered
    `{"unreadable": True}` before anything else: it is never rendered, so no answer carries what a template renders
    to, and Superset is not asked. A database the caller may not see, or that does not exist, is answered
    `{"not_found": "database"}`, as `table_rules` answers one, and the text is not read. A text the database's own
    template processor (`get_template_processor`, which a deployment's `CUSTOM_TEMPLATE_PROCESSORS` decide) parses to
    anything but text, or cannot parse, is answered `{"unreadable": True}`, and is not rendered either. So is a text
    rc2 cannot read, or reads as other than one statement, or as one it does not fully model
    (`has_unparseable_statement`, which its access check refuses under strict scoping), or cannot render with the
    placeholder rule; an answer over `MAX_ANSWER_BYTES` is answered `{"too_large": True}`."""
    from jinja2.exceptions import TemplateError
    from sqlglot.errors import SqlglotError
    from superset.daos.database import DatabaseDAO
    from superset.exceptions import SupersetParseError, SupersetSecurityException
    from superset.sql.parse import process_jinja_sql

    if any(marker in found["sql"] for marker in TEMPLATE_MARKERS):
        return {"unreadable": True}
    database = DatabaseDAO.find_by_id(found["database_id"])
    if database is None:
        return writes._not_found("database")
    if not _plain_text(found["sql"], database):
        return {"unreadable": True}
    catalog = database.get_default_catalog()
    try:
        parsed = process_jinja_sql(found["sql"], database, {})
    except (SupersetParseError, SupersetSecurityException, TemplateError):
        return {"unreadable": True}
    if len(parsed.script.statements) != 1 or parsed.script.has_unparseable_statement:
        return {"unreadable": True}
    [statement] = parsed.script.statements
    tables = {table.qualify(catalog=catalog, schema=found["schema"]) for table in parsed.tables}
    try:
        statement.apply_rls(catalog, found["schema"],
                            {table: [statement.parse_predicate(PLACEHOLDER_RULE)] for table in tables},
                            database.db_engine_spec.get_rls_method())
        ruled = statement.format()
    except (SupersetParseError, SqlglotError):
        return {"unreadable": True}
    answer = {"tables": [{"catalog": table.catalog, "schema": table.schema, "table": table.table}
                         for table in sorted(tables, key=lambda t: (t.catalog or "", t.schema or "", t.table))],
              "ruled": ruled}
    return dict(TOO_LARGE) if _size(answer) > MAX_ANSWER_BYTES else answer


def _plain_text(sql: str, database: Any) -> bool:
    """Whether the database's own template processor reads `sql` as text and nothing more: its environment parses it
    (never renders it) to `_PLAIN_TEXT_NODES` alone. A processor with no environment, and a text it cannot parse, are
    not plain text."""
    from jinja2.exceptions import TemplateError
    from superset.jinja_context import get_template_processor

    env = getattr(get_template_processor(database=database), "env", None)
    if env is None:
        return False
    try:
        pending = [env.parse(sql)]
    except TemplateError:
        return False
    while pending:
        node = pending.pop()
        if type(node).__name__ not in _PLAIN_TEXT_NODES:
            return False
        pending.extend(node.iter_child_nodes())
    return True


@writes._answering("session_binding", REQUESTS)
def session_binding(found: dict) -> dict:
    """Whether the session Superset opens on the database for the caller, as `execute_sql` opens one for a statement in
    `schema`, logs in as `schema`, keeps it as its current schema and reached `service`, each as Oracle stores it:
    `{"bound": true}` or `{"bound": false}`, and never a value the session read.

    The database is found as `table_rules` finds one, through Superset's own `DatabaseDAO` filter: a database the
    caller may not see, or that does not exist, is answered `{"not_found": "database"}`, and one whose backend is not
    Oracle `{"refused": "backend"}`; on neither is anything run. Otherwise the tool runs `SESSION_SQL`, and nothing the
    caller sends, on the connection rc2's executor opens for the caller (`SQLExecutor._execute_statements`):
    `get_raw_connection` with the database's default catalog and the schema as Superset shows it (its dialect's
    `normalize_name`), so the caller's own login where the database impersonates its users, through the engine spec's
    `execute`. It reads no table of the caller's and no row of data.

    A session that cannot be read (no connection, the statement failing, an answer other than one row of three
    values) is answered `{"unread": true}`, with no message: what failed is logged here by its kind alone."""
    from superset.daos.database import DatabaseDAO

    database = DatabaseDAO.find_by_id(found["database_id"])
    if database is None:
        return writes._not_found("database")
    if database.backend != ORACLE:
        return {"refused": "backend"}
    try:
        shown = database.get_dialect().normalize_name(found["schema"])
        with database.get_raw_connection(catalog=database.get_default_catalog(), schema=shown) as connection:
            with contextlib.closing(connection.cursor()) as cursor:
                database.db_engine_spec.execute(cursor, SESSION_SQL, database)
                rows = [tuple(row) for row in cursor.fetchall()]
    except Exception as exc:  # any failure to read the session is answered as one, never raised  # noqa: BLE001
        logger.warning("session_binding could not read the session on database %s (%s)", found["database_id"],
                       type(exc).__name__)
        return {"unread": True}
    if len(rows) != 1 or len(rows[0]) != 3:
        return {"unread": True}
    return {"bound": session_bound(rows[0], service=found["service"], schema=found["schema"])}


def session_bound(row: tuple, *, service: str, schema: str) -> bool:
    """Whether `row`, `SESSION_SQL`'s (login user, service, current schema), logs in as `schema`, keeps it as its
    current schema and reached `service`, each compared exactly, as Oracle stores it."""
    user, reached, current = row
    return user == schema and current == schema and reached == service


def _table_rules_answer(names: list[str], applies) -> dict:
    return {"tables": [{"table": name, "applies": applies(name)} for name in names]}
