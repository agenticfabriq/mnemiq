"""What a dashboard build's tests run against: a Superset whose state lives in memory and is served on `FakeSuperset`,
and a build store with the Postgres store's own rules over a dict, on a clock the test moves.

`World` answers every tool a build calls as rc2 and mnemiq's extension do, as far as a build can tell. The extension's
creates (`create_dashboard`, `create_dataset`, `create_chart`) first look for the object carrying the request's marker
as its description's last line, and answer with it, or with `candidates` when there are several, as the extension's
`_find_or_create` does; only when there is none do they make one, with the marker. `create_chart` saves a chart from
MCP's chart config with the `form_data` MCP's mappers write, the request's `set` merged over it, and its name and its
SQL metrics' labels as rc2's validators leave them (`rc2_saved`), or a funnel or a sankey from its own form data; and
the query context of the extension's own `params.query_context`, so the query is the one the extension saves.
`get_chart_data` runs a chart's saved query context and `execute_sql` mnemiq's own statement, both on one DuckDB of
fixed rows, by two separate paths, so a chart whose numbers are right is checked and one the test skews is not. MCP's
own creates (`generate_chart`, `create_dataset`, `create_virtual_dataset`) are served too, though no build calls them.
Superset's clock is the test's clock, written with no zone as rc2's `datetime.now` default writes `created_on`.

`MemoryBuildStore` keeps builds and measures as `BuildStore` does, through the store module's own transitions and
checks (`_claim`, `_renew`, `_finish`, `_plan`, `_intent`, ...), each change made whole under one lock, as the
Postgres store's compare-and-set makes it one write. Its clock is the control Postgres's.
"""

from __future__ import annotations

import copy
import html
import itertools
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import duckdb

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.superset import build_store as bs
from mnemiq.superset.build_store import BuildNotFound, StoredBuild
from mnemiq.superset.extension import EXTENSION_TOOLS, TOOL_PREFIX
from tests.superset_extension_path import use_extension_source

use_extension_source()

from agenticfabriq.mnemiq import params as extension_params  # noqa: E402
from agenticfabriq.mnemiq import reads as extension_reads  # noqa: E402
from agenticfabriq.mnemiq import writes as extension_writes  # noqa: E402

TOKEN = "tok-builder"
FRESH_TOKEN = "tok-builder-signed-in-again"
# Who each bearer is, to Superset: a fresh sign-in is the same person.
USERS = {TOKEN: "builder", FRESH_TOKEN: "builder", "tok-colleague": "colleague", "tok-west": "west", "tok-east": "east"}
DATABASE_ID = 7
SCHEMA = "public"
POLICY_DATASET = 12

POLICY_ROWS = [
    (1, "east", "web", "active", 120, "2026-01-05 09:00:00"),
    (2, "east", "agent", "active", 80, "2026-01-19 14:30:00"),
    (3, "west", "web", "lapsed", 200, "2026-02-02 08:00:00"),
    (4, "west", "agent", "active", 60, "2026-02-20 10:00:00"),
    (5, "north", "web", "active", 150, "2026-03-03 11:00:00"),
    (6, "north", "agent", "lapsed", 90, "2026-03-28 16:00:00"),
    (7, "east", "web", "quoted", 40, "2026-03-30 12:00:00"),
]
CLAIM_ROWS = [
    (1, "east", 500, "2026-01-10 09:00:00"),
    (2, "west", 300, "2026-02-11 09:00:00"),
    (3, "west", 250, "2026-03-12 09:00:00"),
    (4, "north", 75, "2026-03-15 09:00:00"),
]
TABLES = {
    "policy": [("policy_id", "INTEGER"), ("region", "VARCHAR"), ("channel", "VARCHAR"), ("status", "VARCHAR"),
               ("premium", "INTEGER"), ("issued", "TIMESTAMP")],
    "claim": [("claim_id", "INTEGER"), ("claim_region", "VARCHAR"), ("amount", "INTEGER"), ("filed", "TIMESTAMP")],
}
SNAPSHOT = Snapshot(version="v1", source_id="only", created_at="t", columns=[
    Column(id=f"{table}.{name}", object_id=table, name=name, data_type=kind)
    for table, columns in TABLES.items() for name, kind in columns])
GRANTS = GrantSet(frozenset(TABLES))
# The saved metric planted on `policy`'s dataset, which a plan reuses; and Superset's own `count`.
PLANTED = ("total_premium", "SUM(premium)")
USD = {"symbol": "USD", "symbolPosition": "prefix"}

EMPTY = {"success": True, "columns": [], "rows": []}
_GRAIN_UNIT = {"PT1S": "second", "PT1M": "minute", "PT1H": "hour", "P1D": "day", "P1W": "week", "P1M": "month",
               "P3M": "quarter", "P1Y": "year"}
_OPERATORS = {"==": "=", "!=": "<>", ">": ">", "<": "<", ">=": ">=", "<=": "<=", "LIKE": "LIKE", "ILIKE": "ILIKE",
              "NOT LIKE": "NOT LIKE"}
_FILTER_OP = {"=": "==", "!=": "!=", ">": ">", "<": "<", ">=": ">=", "<=": "<=", "LIKE": "LIKE", "ILIKE": "ILIKE",
              "NOT LIKE": "NOT LIKE", "IN": "IN", "NOT IN": "NOT IN", "IS NULL": "IS NULL",
              "IS NOT NULL": "IS NOT NULL"}


# What rc2's MCP removes from a chart name or a metric label before and after cleaning it (zero-width characters, C0
# controls but for tab, newline and carriage return, the next-line character, and line and paragraph separators), and
# what it refuses: a script scheme or an event handler (`sanitize_user_input` in
# `superset/mcp_service/utils/sanitization.py`).
_RC2_REMOVED = re.compile("[" + "".join(chr(c) for c in (*range(0x00, 0x09), 0x0B, 0x0C, *range(0x0E, 0x20), 0x85,
                                                         0x200B, 0x200C, 0x200D, 0x2028, 0x2029, 0xFEFF)) + "]")
_RC2_REFUSED = (re.compile(r"\b(javascript|vbscript|data):", re.IGNORECASE), re.compile(r"on\w+\s*=", re.IGNORECASE))
_TAG = re.compile(r"</?[A-Za-z][^>]*>")


def rc2_saved(text: str) -> str | ValueError:
    """What rc2's MCP saves of a chart name or an SQL metric's label (`GenerateChartRequest.sanitize_chart_name`,
    `ColumnRef.sanitize_label`), for the texts a build sends: the ends stripped and the characters `_RC2_REMOVED`
    names removed; every entity unescaped and the tags dropped, then the text written as HTML writes it (`&`, `<`, `>`
    and a no-break space as entities) with `&amp;` back as `&`; and a script scheme or an event handler refused, as a
    validation error (the `ValueError` returned, which the fake sends as the tool's error)."""
    text = _RC2_REMOVED.sub("", text.strip())
    while (unescaped := html.unescape(text)) != text:
        text = unescaped
    text = _TAG.sub("", text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\u00a0", "&nbsp;")
    text = _RC2_REMOVED.sub("", text.replace("&amp;", "&"))
    if any(pattern.search(text) for pattern in _RC2_REFUSED):
        return ValueError("1 validation error for GenerateChartRequest: Value error, the text contains potentially "
                          "malicious content")
    return text


class Clock:
    """The test's clock: UTC, moved only by `advance`, shared by the store (the control Postgres's clock) and by
    Superset (its app server's)."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += timedelta(seconds=seconds)


# ---------------------------------------------------------------- the store


class MemoryBuildStore:
    """`BuildStore`'s interface and rules, its rows in memory and its clock the test's. Counts what it is asked, so a
    test can see a claim was refused before any Superset call."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self._builds: dict[tuple[str, str], dict] = {}
        self._measures: dict[tuple[str, int, str], dict] = {}
        self._lock = threading.Lock()
        self.claims: list[str] = []

    def _now(self) -> datetime:
        return self.clock.now()

    def _live_on(self, tenant: str, dashboard_id: int, other_than: str | None = None) -> bool:
        return any(row["tenant_id"] == tenant and row["dashboard_id"] == dashboard_id and row["status"] == bs.BUILDING
                   and row["build_id"] != other_than for row in self._builds.values())

    def create_build(self, tenant, principal, goal, plan, steps, *, dashboard_id=None, profile=None) -> StoredBuild:
        tenant, principal = bs._text(tenant, "tenant"), bs._text(principal, "principal")
        goal = bs._text(goal, "goal")
        profile = None if profile is None else bs._text(profile, "profile")
        dashboard_id = bs._id(dashboard_id, "a build's dashboard id", optional=True)
        plan = bs._plan(plan)
        build_id = str(uuid.uuid4())
        step_list = bs._new_steps(build_id, steps)
        with self._lock:
            now = self._now()
            if dashboard_id is not None:
                self._release_abandoned(tenant, dashboard_id)
            if dashboard_id is not None and self._live_on(tenant, dashboard_id):
                raise bs.DashboardBusy()
            row = {"tenant_id": tenant, "build_id": build_id, "principal_id": principal, "dashboard_id": dashboard_id,
                   "goal": goal, "profile": profile, "plan": plan, "steps": step_list, "status": bs.BUILDING,
                   "created_at": now, "updated_at": now}
            self._builds[(tenant, build_id)] = row
            return self._stored(row)

    def unfinished_builds(self, tenant, principal, *, dashboard_id=None, limit=10) -> list[StoredBuild]:
        """`BuildStore.unfinished_builds`: newest touched first, ties by build id ascending, as Postgres orders them."""
        tenant, principal = bs._text(tenant, "tenant"), bs._text(principal, "principal")
        dashboard_id = bs._id(dashboard_id, "a dashboard id", optional=True)
        if type(limit) is not int or not 1 <= limit <= bs.MAX_UNFINISHED:
            raise ValueError(f"a list holds 1 to {bs.MAX_UNFINISHED} builds")
        with self._lock:
            rows = [row for row in self._builds.values() if row["tenant_id"] == tenant
                    and row["principal_id"] == principal and row["status"] == bs.BUILDING
                    and (dashboard_id is None or row["dashboard_id"] == dashboard_id)]
            rows.sort(key=lambda row: row["build_id"])
            rows.sort(key=lambda row: row["updated_at"], reverse=True)
            return [self._stored(row) for row in rows[:limit]]

    def dashboard_building(self, tenant, dashboard_id) -> bool:
        with self._lock:
            self._release_abandoned(tenant, dashboard_id)
            return self._live_on(tenant, dashboard_id)

    def _release_abandoned(self, tenant: str, dashboard_id: int) -> None:
        """`build_store._release_abandoned`: each build of the dashboard still building, untouched for
        `ABANDONED_AFTER_S` with no claim of it holding, has its next step failed as abandoned."""
        for row in self._builds.values():
            if (row["tenant_id"], row["dashboard_id"], row["status"]) == (tenant, dashboard_id, bs.BUILDING):
                released = bs._released(copy.deepcopy(row["steps"]), self._now(), row["updated_at"])
                if released is not None:
                    self._write(row, released, row["dashboard_id"])

    def load_build(self, tenant, build_id, principal) -> StoredBuild | None:
        tenant, principal = bs._text(tenant, "tenant"), bs._text(principal, "principal")
        with self._lock:
            row = self._builds.get((tenant, build_id))
            if row is None or row["principal_id"] != principal:
                return None
            return self._stored(row)

    def claim_next_step(self, tenant, build_id, principal, intent, lease_s=bs.DEFAULT_LEASE_S):
        tenant, principal = bs._text(tenant, "tenant"), bs._text(principal, "principal")
        intent = bs._intent(intent)
        lease = timedelta(seconds=bs._lease(lease_s))
        with self._lock:
            self.claims.append(intent["key"])
            row = self._row(tenant, build_id, principal)
            steps = copy.deepcopy(row["steps"])
            changed, outcome = bs._claim(steps, intent, self._now(), lease, str(uuid.uuid4()))
            if changed is not None:
                self._write(row, changed, row["dashboard_id"])
            return outcome

    def renew_lease(self, tenant, build_id, key, lease_id, lease_s=bs.DEFAULT_LEASE_S) -> bool:
        lease = timedelta(seconds=bs._lease(lease_s))
        with self._lock:
            try:
                row = self._row(tenant, build_id, None)
            except BuildNotFound:
                return False
            changed, renewed = bs._renew(copy.deepcopy(row["steps"]), key, lease_id, self._now(), lease)
            if changed is not None:
                self._write(row, changed, row["dashboard_id"])
            return renewed is True

    def finish_step(self, tenant, build_id, key, status, object_id, *, lease_id, reason=None) -> StoredBuild:
        if status not in bs._FINISHED_AS:
            raise ValueError("a step finishes done, failed or uncertain")
        object_id = bs._id(object_id, "a step's object id", optional=True)
        reason = bs._reason(reason)
        with self._lock:
            row = self._row(tenant, build_id, None)
            steps, dashboard_id = bs._finish(copy.deepcopy(row["steps"]), row["dashboard_id"], key, lease_id, status,
                                             object_id, reason)
            if dashboard_id is not None and dashboard_id != row["dashboard_id"] \
                    and self._live_on(tenant, dashboard_id, other_than=row["build_id"]):
                # As the Postgres store: a holder left untouched is released, and the finish is tried once more.
                self._release_abandoned(tenant, dashboard_id)
                if self._live_on(tenant, dashboard_id, other_than=row["build_id"]):
                    raise bs.DashboardBusy()
            self._write(row, steps, dashboard_id)
            return self._stored(row)

    def measures_for(self, tenant, dashboard_id) -> list[bs.DashboardMeasure]:
        with self._lock:
            rows = [row for (t, d, _), row in self._measures.items() if t == tenant and d == dashboard_id]
            return [self._measure(row) for row in sorted(rows, key=lambda r: r["name"])]

    def put_measure(self, tenant, dashboard_id, measure, *, chart_ids=(), made_dataset_ids=(), check_status=None,
                    saved_metric=None):
        tenant = bs._text(tenant, "tenant")
        dashboard_id = bs._id(dashboard_id, "a dashboard id")
        row = {"name": measure.name, "definition": bs._definition(measure, saved_metric),
               "chart_ids": bs._ids(chart_ids, "c"),
               "made_dataset_ids": bs._ids(made_dataset_ids, "m"),
               "check_status": bs._check_status({} if check_status is None else check_status)}
        with self._lock:
            row["updated_at"] = self._now()
            self._measures[(tenant, dashboard_id, measure.name)] = row
            return self._measure(row)

    def delete_measure(self, tenant, dashboard_id, name) -> bool:
        with self._lock:
            return self._measures.pop((tenant, dashboard_id, name), None) is not None

    def raw_steps(self, tenant: str, build_id: str) -> list[dict]:
        """The steps as kept, lease ids and all, as a test reads a row past the store."""
        with self._lock:
            return copy.deepcopy(self._builds[(tenant, build_id)]["steps"])

    def _row(self, tenant: str, build_id: str, principal: str | None) -> dict:
        row = self._builds.get((tenant, build_id))
        if row is None or (principal is not None and row["principal_id"] != principal):
            raise BuildNotFound()
        return row

    def _write(self, row: dict, steps: list[dict], dashboard_id: int | None) -> None:
        row.update(steps=steps, dashboard_id=dashboard_id, status=bs._build_status(steps), updated_at=self._now())

    def _stored(self, row: dict) -> StoredBuild:
        now = self._now()
        return StoredBuild(tenant_id=row["tenant_id"], build_id=row["build_id"], principal_id=row["principal_id"],
                           dashboard_id=row["dashboard_id"], goal=row["goal"], plan=copy.deepcopy(row["plan"]),
                           steps=tuple(bs._step_state(step, now) for step in row["steps"]), status=row["status"],
                           created_at=bs._iso(row["created_at"]), updated_at=bs._iso(row["updated_at"]),
                           profile=row["profile"])

    def _measure(self, row: dict) -> bs.DashboardMeasure:
        return bs._dashboard_measure((row["name"], copy.deepcopy(row["definition"]), list(row["chart_ids"]),
                                      list(row["made_dataset_ids"]), dict(row["check_status"]), row["updated_at"]))


# ---------------------------------------------------------------- Superset


@dataclass
class Hook:
    """What a test makes happen around one tool's work: `before` runs before it (raise to answer an error, so
    nothing is done), `instead` in its place (an answer it returns is sent, and the tool does nothing), `after` once
    it is done and committed, before the answer leaves (sleep past the client's deadline, refuse the token, move the
    clock). Each gets the request, the answer (after) and the world."""

    before: Callable | None = None
    # Run once the tool's work is committed; an answer it returns is sent in place of the tool's own.
    after: Callable | None = None
    # Answers in the tool's place, doing nothing, when it returns an answer rather than None.
    instead: Callable | None = None


@dataclass
class Late:
    """A call held at Superset's door (`World.run_late`): `release` lets it run, `ran` is set once it has, and
    `answers` holds what it answered."""

    release: threading.Event
    ran: threading.Event
    answers: list


class World:
    """Superset, as a build reads and writes it. `world.datasets`, `world.charts` and `world.dashboards` are its rows,
    and its clock is the test's."""

    def __init__(self, fake, clock: Clock) -> None:
        self.fake, self.clock = fake, clock
        self.lock = threading.RLock()
        self.datasets: dict[int, dict] = {}
        # The datasets on which a row rule applies to every caller, as the extension's `row_rules` answers.
        self.row_ruled: set[int] = set()
        # An extension older than `create_chart`'s `leave_out_no_value`: it refuses the field, as the extension
        # refuses a field it does not know.
        self.without_leave_out = False
        self.charts: dict[int, dict] = {}
        self.dashboards: dict[int, dict] = {}
        # A row rule on `policy` for each user it applies to, as SQL Superset adds to every statement and chart query
        # that user runs over the table; and the charts an alert or a report holds, which `delete_chart` refuses.
        self.user_rules: dict[str, str] = {}
        self.reported: set[int] = set()
        self.trashed: dict[int, dict] = {}
        self.hooks: dict[str, Hook] = {}
        # A chart's numbers a test puts off: chart title to the amount added to its first row's last value.
        self.skewed: dict[str, float] = {}
        self._ids = {"chart": itertools.count(100), "dataset": itertools.count(50), "dashboard": itertools.count(300)}
        self._tick = itertools.count(1)
        self.db = duckdb.connect()
        for table, columns in TABLES.items():
            self.db.execute(f'CREATE TABLE "{table}" ({", ".join(f"{n} {k}" for n, k in columns)})')
        self.db.executemany('INSERT INTO "policy" VALUES (?, ?, ?, ?, ?, ?)', POLICY_ROWS)
        self.db.executemany('INSERT INTO "claim" VALUES (?, ?, ?, ?)', CLAIM_ROWS)
        self.datasets[POLICY_DATASET] = self._dataset_row(POLICY_DATASET, "policy", None, None, "analyst",
                                                          self.superset_now() - timedelta(days=30))
        self.datasets[POLICY_DATASET]["metrics"].append(
            {"metric_name": PLANTED[0], "expression": PLANTED[1], "verbose_name": "Total premium",
             "d3format": None, "currency": USD, "certified_by": "finance", "certification_details": None,
             "warning_text": None, "description": None})
        for tool, handler in self._handlers().items():
            fake.reply(tool, self._serving(tool, handler))

    # --- the clock and the rows

    def superset_now(self) -> datetime:
        """Superset's app server's `datetime.now()`: the test's clock, with no zone written."""
        return self.clock.now().replace(tzinfo=None)

    def _stamp(self) -> datetime:
        # Two writes in one test instant still get distinct `changed_on`s, as two real writes would.
        return self.superset_now() + timedelta(microseconds=next(self._tick))

    def _dataset_row(self, dataset_id: int, table_name: str, sql: str | None, description: str | None, user: str,
                     created_on: datetime) -> dict:
        return {"id": dataset_id, "table_name": table_name, "schema": SCHEMA, "database_id": DATABASE_ID, "sql": sql,
                "description": description, "created_by": user, "created_on": created_on,
                "changed_on": created_on, "editors": [user],
                "metrics": [{"metric_name": "count", "expression": "COUNT(*)", "verbose_name": "COUNT(*)",
                             "d3format": None, "currency": None, "certified_by": None,
                             "certification_details": None, "warning_text": None, "description": None}]}

    def plant_chart(self, *, title: str, dataset_id: int, viz_type: str, user: str = "builder",
                    description: str | None = None, created_on: datetime | None = None) -> int:
        """A chart made outside the build (by hand, or by another build), as Superset keeps it."""
        with self.lock:
            chart_id = next(self._ids["chart"])
            dataset = self.datasets[dataset_id]
            moment = created_on or self.superset_now()
            self.charts[chart_id] = {
                "id": chart_id, "slice_name": title, "viz_type": viz_type, "params": {"viz_type": viz_type},
                "datasource_id": dataset_id, "datasource_name": f"{dataset['schema']}.{dataset['table_name']}",
                "description": description, "query_context": None, "created_by": user, "created_on": moment,
                "changed_on": moment, "editors": [user]}
            return chart_id

    def dashboard_of(self, title: str) -> dict:
        [found] = [d for d in self.dashboards.values() if d["dashboard_title"] == title]
        return found

    def calls(self, tool: str) -> list:
        """The calls of `tool`: the extension's tool of that name when it offers one (so `create_dataset` is the
        extension's), else Superset's own. MCP's `create_dataset` is read off `fake.calls` by its name."""
        name = TOOL_PREFIX + tool if tool in EXTENSION_TOOLS else tool
        return [call for call in self.fake.calls if call.tool == name]

    # --- what a test makes of one call: its answer lost, or the call run late

    def lose_answer(self, tool: str, *, hold_s: float) -> threading.Event:
        """The next call of `tool` does its work and commits it, then holds its answer `hold_s` seconds, past the
        deadline of a client that waits less, which then raises `SupersetUnavailable`: a create whose answer was lost
        after Superset made the object. Later calls answer at once. The event is set once the held call has ended."""
        ended, held = threading.Event(), []

        def hold(request, answer, _world):
            if not held:
                held.append(True)
                time.sleep(hold_s)
                ended.set()

        self.hooks[tool] = Hook(after=hold)
        return ended

    def run_late(self, tool: str) -> Late:
        """The next call of `tool` waits at Superset's door, doing nothing, until `release` is set, so a client whose
        deadline passes meanwhile raises `SupersetUnavailable` with nothing made; once released, the call runs, late,
        its answer is kept in `answers`, and `ran` is set. Later calls run at once."""
        late = Late(release=threading.Event(), ran=threading.Event(), answers=[])
        first = []

        def wait(request, _world):
            if not first:
                first.append(True)
                late.release.wait(30)

        def keep(request, answer, _world):
            if late.release.is_set() and not late.ran.is_set():
                late.answers.append(answer)
                late.ran.set()

        self.hooks[tool] = Hook(before=wait, after=keep)
        return late

    # --- serving

    def _serving(self, tool: str, handler: Callable):
        def answer(request: dict, authorization: str | None):
            user = USERS.get((authorization or "").removeprefix("Bearer "), "someone")
            hook = self.hooks.get(tool.removeprefix(TOOL_PREFIX))
            if hook is not None and hook.before is not None:
                hook.before(request, self)
            if hook is not None and hook.instead is not None and (found := hook.instead(request, self)) is not None:
                return found
            with self.lock:
                try:
                    found = handler(request, user)
                except extension_writes.Refused as refused:  # what the extension answers a request it refuses
                    found = refused.answer
            if hook is not None and hook.after is not None:
                replaced = hook.after(request, found, self)
                if isinstance(replaced, dict):
                    return replaced
            return found

        return answer

    def _handlers(self) -> dict[str, Callable]:
        plain = {"list_datasets": self._list_datasets, "get_dataset_info": self._get_dataset_info,
                 "list_metrics": self._list_metrics, "get_chart_info": self._get_chart_info,
                 "get_chart_data": self._get_chart_data, "list_charts": self._list_charts,
                 "list_dashboards": self._list_dashboards, "generate_chart": self._generate_chart,
                 "create_dataset": self._create_dataset, "create_virtual_dataset": self._create_virtual_dataset,
                 "execute_sql": self._execute_sql}
        plain["delete_chart"] = self._delete_chart
        extension = {"create_dashboard": self._create_dashboard, "create_dataset": self._make_dataset,
                     "create_chart": self._create_chart, "patch_chart": self._patch_chart,
                     "add_saved_metric": self._add_saved_metric, "delete_dataset": self._delete_dataset,
                     "set_layout": self._set_layout, "dataset_metrics": self._dataset_metrics,
                     "get_layout": self._get_layout, "editors_of": self._editors_of,
                     "row_rules": self._row_rules}
        return {**plain, **{TOOL_PREFIX + name: handler for name, handler in extension.items()}}

    # --- listing, as `ModelListCore` lists: filters, `created_by_me`, the columns asked for, a page

    @staticmethod
    def _matches(row: dict, filters: list[dict]) -> bool:
        for item in filters:
            value = row.get(item["col"])
            if item["opr"] == "eq" and value != item["value"]:
                return False
        return True

    def _page(self, rows: list[dict], request: dict, user: str, kind: str, shown: Callable[[dict], dict]) -> dict:
        order = request.get("order_column") or "id"
        assert order == "id", f"the world orders only by id, not {order}"
        found = [row for row in sorted(rows, key=lambda r: r["id"], reverse=request.get("order_direction") == "desc")
                 if self._matches(row, request.get("filters", []))
                 and (not request.get("created_by_me") or row["created_by"] == user)]
        size, page = request.get("page_size", 10), request.get("page", 1)
        columns = request.get("select_columns") or ["id"]
        items = [{key: value for key, value in shown(row).items() if key in columns}
                 for row in found[(page - 1) * size: page * size]]
        return {kind: items, "count": len(items), "total_count": len(found), "page": page, "page_size": size,
                "total_pages": max(1, -(-len(found) // size))}

    @staticmethod
    def _iso(moment: datetime | None) -> str | None:
        return None if moment is None else moment.isoformat()

    def _dataset_shown(self, row: dict) -> dict:
        return {"id": row["id"], "table_name": row["table_name"], "schema": row["schema"],
                "database_id": row["database_id"], "sql": row["sql"], "description": row["description"],
                "is_virtual": row["sql"] is not None, "created_on": self._iso(row["created_on"]),
                "changed_on": self._iso(row["changed_on"]),
                "metrics": [{"metric_name": m["metric_name"], "verbose_name": m["verbose_name"],
                             "expression": m["expression"], "description": m["description"],
                             "d3format": m["d3format"]} for m in row["metrics"]]}

    def _list_datasets(self, request: dict, user: str) -> dict:
        return self._page(list(self.datasets.values()), request, user, "datasets", self._dataset_shown)

    def _get_dataset_info(self, request: dict, user: str) -> dict:
        row = self.datasets.get(request["identifier"])
        if row is None:
            return {"error": "not found", "error_type": "not_found"}
        shown = self._dataset_shown(row)
        # `changed_on` and `created_on` are answered only when asked for (`get_dataset_info`'s default columns).
        columns = request.get("select_columns") or ["id", "table_name", "schema", "database_id", "is_virtual",
                                                    "description", "sql", "metrics"]
        return {key: value for key, value in shown.items() if key in columns}

    def _list_metrics(self, request: dict, user: str) -> dict:
        row = self.datasets[request["dataset_id"]]
        metrics = [{"name": m["metric_name"], "expression": m["expression"], "verbose_name": m["verbose_name"],
                    "d3format": m["d3format"], "warning_text": m["warning_text"], "description": m["description"]}
                   for m in row["metrics"]]
        return {"metrics": metrics, "total_count": len(metrics), "page": 1, "page_size": request.get("page_size"),
                "total_pages": 1, "success": True}

    def _dataset_metrics(self, request: dict, user: str) -> dict:
        found = []
        for dataset_id in request["dataset_ids"]:
            if dataset_id in self.datasets:
                found.append({"dataset_id": dataset_id, "metrics": [
                    {key: m[key] for key in ("metric_name", "expression", "d3format", "currency", "certified_by",
                                             "certification_details", "warning_text")}
                    for m in self.datasets[dataset_id]["metrics"]]})
        return {"datasets": found}

    def _row_rules(self, request: dict, user: str) -> dict:
        found = extension_reads.read("row_rules", request)
        return {"datasets": [{"dataset_id": i, "applies": i in self.row_ruled}
                             for i in found["dataset_ids"] if i in self.datasets]}

    def _chart_shown(self, row: dict) -> dict:
        return {"id": row["id"], "slice_name": row["slice_name"], "viz_type": row["viz_type"],
                "form_data": copy.deepcopy(row["params"]), "description": row["description"],
                "datasource_name": row["datasource_name"], "datasource_type": "table",
                "created_on": self._iso(row["created_on"]), "changed_on": self._iso(row["changed_on"])}

    def _list_charts(self, request: dict, user: str) -> dict:
        return self._page(list(self.charts.values()), request, user, "charts", self._chart_shown)

    def _get_chart_info(self, request: dict, user: str) -> dict:
        row = self.charts.get(request["identifier"])
        if row is None:
            return {"error": "not found", "error_type": "not_found"}
        columns = request.get("select_columns") or ["id", "slice_name", "viz_type", "description", "changed_on"]
        return {key: value for key, value in self._chart_shown(row).items() if key in columns}

    def _dashboard_shown(self, row: dict) -> dict:
        return {"id": row["id"], "dashboard_title": row["dashboard_title"], "description": row["description"],
                "published": row["published"], "created_on": self._iso(row["created_on"]),
                "changed_on": self._iso(row["changed_on"])}

    def _list_dashboards(self, request: dict, user: str) -> dict:
        return self._page(list(self.dashboards.values()), request, user, "dashboards", self._dashboard_shown)

    # --- making things, as MCP makes them: served, though no build calls them

    def _metric(self, column: dict) -> Any:
        """`create_metric_object` (`superset/mcp_service/chart/chart_utils.py`) for the two kinds a build sends, an SQL
        metric under its label as rc2 saves it (`rc2_saved`)."""
        if column.get("saved_metric"):
            return column["name"]
        return {"aggregate": None, "column": None, "expressionType": "SQL", "sqlExpression": column["sql_expression"],
                "label": rc2_saved(column["label"]), "optionName": "metric_sql_0", "hasCustomLabel": True,
                "datasourceWarning": False}

    @staticmethod
    def _filters(config: dict) -> list[dict]:
        return [{"clause": "WHERE", "expressionType": "SIMPLE", "subject": f["column"], "operator": _FILTER_OP[f["op"]],
                 "comparator": f.get("value")} for f in config.get("filters") or []]

    def _temporal(self, dataset_id: int, column: str) -> bool:
        table = self.datasets[dataset_id]["table_name"]
        return dict(TABLES.get(table, [])).get(column) == "TIMESTAMP"

    def form_data(self, dataset_id: int, config: dict) -> dict:
        """The `form_data` MCP's mappers write for the configs a build sends (`map_xy_config`, `map_pie_config`,
        `map_big_number_config`, `map_table_config`, `map_gauge_config`, `map_treemap_config`), each row limit a config
        leaves unset as its `*ChartConfig.row_limit` default, and a pie's, gauge's and treemap's `sort_by_metric` as
        its default, true (`superset/mcp_service/chart/schemas.py`)."""
        kind = config["chart_type"]
        if kind == "xy":
            viz = {"line": "echarts_timeseries_line", "bar": "echarts_timeseries_bar", "area": "echarts_area",
                   "scatter": "echarts_timeseries_scatter"}[config["kind"]]
            x = config["x"]["name"]
            form = {"viz_type": viz, "metrics": [self._metric(c) for c in config["y"]], "x_axis": x}
            if self._temporal(dataset_id, x):
                form["granularity_sqla"] = x
                if config.get("time_grain"):
                    form["time_grain_sqla"] = config["time_grain"]
            else:
                form.update(x_axis_sort_series_type="name", x_axis_sort_series_ascending=True, time_grain_sqla=None,
                            granularity_sqla=None)
            if config.get("group_by"):
                form["groupby"] = [c["name"] for c in config["group_by"] if c["name"] != x]
            form["row_limit"] = config.get("row_limit", 10000)
            if (config.get("y_axis") or {}).get("format"):
                form["y_axis_format"] = config["y_axis"]["format"]
        elif kind == "pie":
            form = {"viz_type": "pie", "groupby": [config["dimension"]["name"]], "metric": self._metric(config["metric"]),
                    "number_format": config.get("number_format", "SMART_NUMBER"), "row_limit": config.get("row_limit", 100),
                    "show_total": False, "donut": False, "show_labels": True,
                    "sort_by_metric": config.get("sort_by_metric", True)}
        elif kind == "big_number":
            form = {"viz_type": "big_number_total", "metric": self._metric(config["metric"])}
            if config.get("y_axis_format"):
                form["y_axis_format"] = config["y_axis_format"]
        elif kind == "gauge":
            form = {"viz_type": "gauge_chart", "metric": self._metric(config["metric"]),
                    "groupby": [c["name"] for c in config.get("groupby") or []],
                    "number_format": config.get("number_format", "SMART_NUMBER"), "row_limit": 10,
                    "sort_by_metric": config.get("sort_by_metric", True)}
        elif kind == "treemap_v2":
            form = {"viz_type": "treemap_v2", "groupby": [c["name"] for c in config["groupby"]],
                    "metric": self._metric(config["metric"]), "row_limit": config.get("row_limit", 100),
                    "sort_by_metric": config.get("sort_by_metric", True)}
        elif kind == "table":
            names = [c["name"] for c in config["columns"] if "name" in c and not c.get("saved_metric")]
            metrics = [self._metric(c) for c in config["columns"] if c.get("saved_metric") or c.get("sql_expression")]
            form = {"viz_type": "table", "query_mode": "aggregate", "all_columns": names, "groupby": names,
                    "metrics": metrics, "row_limit": config.get("row_limit", 1000)}
            if config.get("column_config"):
                form["column_config"] = copy.deepcopy(config["column_config"])
        else:
            raise ValueError(f"the world maps no {kind} chart")
        if config.get("currency_format"):
            form["currency_format"] = copy.deepcopy(config["currency_format"])
        if config.get("filters"):
            form["adhoc_filters"] = self._filters(config)
        return form

    def _generate_chart(self, request: dict, user: str) -> dict:
        assert request.get("save_chart") is True and request.get("generate_preview") is False, request
        dataset_id = request["dataset_id"]
        # The request's texts as rc2's validators leave them, before anything is made: one refused makes nothing.
        name = rc2_saved(request["chart_name"])
        labels = [rc2_saved(c["label"]) for c in _labelled_columns(request["config"])]
        if refused := next((text for text in (name, *labels) if isinstance(text, ValueError)), None):
            return refused
        if dataset_id not in self.datasets:
            return {"success": False, "error": {"error": "dataset not found", "error_type": "NotFound"}}
        form = self.form_data(dataset_id, request["config"])
        chart_id = next(self._ids["chart"])
        dataset = self.datasets[dataset_id]
        moment = self._stamp()
        self.charts[chart_id] = {
            "id": chart_id, "slice_name": name, "viz_type": form["viz_type"], "params": form,
            "datasource_id": dataset_id, "datasource_name": f"{dataset['schema']}.{dataset['table_name']}",
            "description": None, "query_context": None, "created_by": user, "created_on": moment,
            "changed_on": moment, "editors": [user]}
        return {"chart": {"id": chart_id, "slice_name": name, "viz_type": form["viz_type"]},
                "success": True, "form_data": form}

    def _create_dataset(self, request: dict, user: str) -> dict:
        assert set(request) <= {"database_id", "schema", "table_name"}, request
        if any(d["table_name"] == request["table_name"] and d["schema"] == request.get("schema") and d["sql"] is None
               for d in self.datasets.values()):
            return {"error": "Dataset already exists", "error_type": "DatasetExistsError"}
        dataset_id = next(self._ids["dataset"])
        self.datasets[dataset_id] = self._dataset_row(dataset_id, request["table_name"], None, None, user, self._stamp())
        return self._dataset_shown(self.datasets[dataset_id])

    def _create_virtual_dataset(self, request: dict, user: str) -> dict:
        name = request["dataset_name"]
        if any(d["table_name"] == name and d["schema"] == request.get("schema_name") for d in self.datasets.values()):
            return {"id": None, "dataset_name": name, "sql": request["sql"], "database_id": request["database_id"],
                    "error": f"Dataset {name} already exists"}
        dataset_id = next(self._ids["dataset"])
        self.datasets[dataset_id] = self._dataset_row(dataset_id, name, request["sql"], request.get("description"),
                                                      user, self._stamp())
        return {"id": dataset_id, "dataset_name": name, "sql": request["sql"], "database_id": request["database_id"],
                "columns": [], "url": None, "error": None}

    # --- the extension's tools, under its rules

    def _editor_of(self, row: dict, user: str) -> bool:
        return user in row["editors"]

    @staticmethod
    def _marked(rows: dict[int, dict], marker: str) -> list[dict]:
        """The rows whose description's last line is exactly `marker`, by id: the extension's `_marked`."""
        return [rows[i] for i in sorted(rows) if (rows[i]["description"] or "").splitlines()[-1:] == [marker]]

    def _find_or_create(self, rows: dict[int, dict], marker: str, kind: str, user: str,
                        create: Callable[[], dict]) -> dict:
        """The extension's `_find_or_create`: the object `marker` names, answered when the caller edits it; every one,
        when there are several (the builder reads every object here); `create()`'s answer when there is none."""
        found = self._marked(rows, marker)
        if not found:
            return create()
        if len(found) > 1:
            return {"candidates": [row["id"] for row in found]}
        (row,) = found
        if not self._editor_of(row, user):
            return {"not_editor": True, "editors": sorted(row["editors"])}
        return {f"{kind}_id": row["id"]}

    def _create_dashboard(self, request: dict, user: str) -> dict:
        found = extension_writes.read("create_dashboard", request)
        if extension_writes.claim_marker(found["marker"], None) is None:
            return {"refused": "marker"}

        def create() -> dict:
            dashboard_id = next(self._ids["dashboard"])
            moment = self._stamp()
            self.dashboards[dashboard_id] = {
                "id": dashboard_id, "dashboard_title": found["title"], "published": False,
                "description": found["marker"], "position_json": None, "json_metadata": "{}",
                "editors": [user, "colleague"], "members": [], "created_by": user, "created_on": moment,
                "changed_on": moment}
            return {"dashboard_id": dashboard_id}

        return self._find_or_create(self.dashboards, found["marker"], "dashboard", user, create)

    def _edited_dashboard(self, found: dict, user: str) -> dict | tuple[dict]:
        """The request's dashboard, as `(row,)`, when the caller edits it; else the extension's answer."""
        dashboard = self.dashboards.get(found["dashboard_id"])
        if dashboard is None:
            return {"not_found": "dashboard"}
        if not self._editor_of(dashboard, user):
            return {"not_editor": True, "editors": sorted(dashboard["editors"])}
        return (dashboard,)

    def _make_dataset(self, request: dict, user: str) -> dict:
        """The extension's `create_dataset`: a dataset over the table, or over `sql` (a virtual one), with the marker
        after its prose, and the dashboard's editors as its editors (the World does not judge whose grants reach the
        data; the extension's own tests do). A table name the schema already holds a dataset of is answered
        `exists`."""
        found = extension_writes.read("create_dataset", request)
        if extension_writes.claim_marker(found["marker"], found["dashboard_id"]) is None:
            return {"refused": "marker"}
        edited = self._edited_dashboard(found, user)
        if isinstance(edited, dict):
            return edited
        (dashboard,) = edited

        def create() -> dict:
            if any(d["table_name"] == found["table_name"] and d["schema"] == found["schema"]
                   and d["database_id"] == found["database_id"] for d in self.datasets.values()):
                return {"exists": True}
            dataset_id = next(self._ids["dataset"])
            self.datasets[dataset_id] = self._dataset_row(
                dataset_id, found["table_name"], found["sql"],
                extension_writes.with_marker(found["description"], found["marker"]), user, self._stamp())
            self.datasets[dataset_id]["editors"] = list(dashboard["editors"])
            return {"dataset_id": dataset_id}

        return self._find_or_create(self.datasets, found["marker"], "dataset", user, create)

    def _create_chart(self, request: dict, user: str) -> dict:
        """The extension's `create_chart`, in its two shapes: `{viz_type, params}` (a funnel or a sankey), or MCP's
        chart config as `{config, set?}`, mapped as MCP maps it (`form_data`) with `set` merged over it, the chart's
        name and labels as rc2's validators leave them; a config they refuse is refused as the extension refuses it."""
        if self.without_leave_out and "leave_out_no_value" in request:
            return {"refused": "request", "unknown": ["leave_out_no_value"]}
        found = extension_writes.read("create_chart", request)
        if extension_writes.claim_marker(found["marker"], found["dashboard_id"]) is None:
            return {"refused": "marker"}
        from_config = found["config"] is not None
        if from_config:
            shaped = found["viz_type"] is None and found["params"] is None
        else:
            shaped = found["viz_type"] is not None and found["params"] is not None and found["set"] is None \
                and not found["leave_out_no_value"]
        if not shaped:
            return {"refused": "request"}
        if from_config and set(found["set"] or ()) & {"datasource", "datasource_id", "datasource_type", "viz_type"}:
            return {"refused": "set"}
        dataset_id = found["dataset_id"]
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return {"not_found": "dataset"}
        edited = self._edited_dashboard(found, user)
        if isinstance(edited, dict):
            return edited
        (dashboard,) = edited
        if from_config:
            name = rc2_saved(found["slice_name"])
            labels = [rc2_saved(c["label"]) for c in _labelled_columns(found["config"])]
            if any(isinstance(text, ValueError) for text in (name, *labels)):
                return {"refused": "config"}
            form = {**self.form_data(dataset_id, found["config"]), **(found["set"] or {})}
            if found["leave_out_no_value"]:
                # As the extension writes it: from the ranked metric as the mapping saved it.
                held = SimpleNamespace(metrics=[SimpleNamespace(metric_name=m["metric_name"], expression=m["expression"])
                                                for m in dataset["metrics"]])
                ranked = extension_writes.ranked_metric_sql(form, held)
                if ranked is None:
                    return {"refused": "leave_out_no_value"}
                form["adhoc_filters"] = [*(form.get("adhoc_filters") or []),
                                         extension_writes._having_filter(extension_writes.not_null_condition(ranked))]
            viz_type = form["viz_type"]
        else:
            name, viz_type, form = found["slice_name"], found["viz_type"], found["params"]
            form.setdefault("viz_type", viz_type)
            form.setdefault("datasource", f"{dataset_id}__table")
        try:
            context = extension_params.query_context(viz_type, form, dataset_id)
        except ValueError:
            return {"refused": "config" if from_config else "params"}

        def create() -> dict:
            chart_id = next(self._ids["chart"])
            moment = self._stamp()
            self.charts[chart_id] = {
                "id": chart_id, "slice_name": name, "viz_type": viz_type, "params": form,
                "datasource_id": dataset_id, "datasource_name": f"{dataset['schema']}.{dataset['table_name']}",
                "description": found["marker"], "query_context": context, "created_by": user, "created_on": moment,
                "changed_on": moment, "editors": list(dashboard["editors"])}
            return {"chart_id": chart_id}

        return self._find_or_create(self.charts, found["marker"], "chart", user, create)

    def _patch_chart(self, request: dict, user: str) -> dict:
        found = extension_writes.read("patch_chart", request)
        chart = self.charts.get(found["chart_id"])
        if chart is None:
            return {"not_found": "chart"}
        if not self._editor_of(chart, user):
            return {"not_editor": True, "editors": sorted(chart["editors"])}
        if chart["changed_on"] != found["expect_changed_on"]:
            return {"stale": True}
        patched = extension_params.patch(chart["viz_type"], chart["params"], found["set"], found["unset"],
                                         chart["datasource_id"])
        chart.update(params=patched.params, viz_type=patched.viz_type, query_context=patched.query_context,
                     changed_on=self._stamp())
        return {"ok": True}

    def _add_saved_metric(self, request: dict, user: str) -> dict:
        """The extension's `add_saved_metric`: the dataset's editors only, once its `changed_on` is the one read, a
        name the dataset already has refused."""
        found = extension_writes.read("add_saved_metric", request)
        dataset = self.datasets.get(found["dataset_id"])
        if dataset is None:
            return {"not_found": "dataset"}
        if not self._editor_of(dataset, user):
            return {"not_editor": True, "editors": sorted(dataset["editors"])}
        if dataset["changed_on"] != found["expect_changed_on"]:
            return {"stale": True}
        if any(m["metric_name"] == found["metric"]["metric_name"] for m in dataset["metrics"]):
            return {"exists": True}
        metric = {"verbose_name": None, "d3format": None, "currency": None, "certified_by": None,
                  "certification_details": None, "warning_text": None, "description": None, **found["metric"]}
        dataset["metrics"].append(metric)
        dataset["changed_on"] = self._stamp()
        return {"ok": True}

    def _delete_dataset(self, request: dict, user: str) -> dict:
        """The extension's `delete_dataset`: a dataset whose marker names this dashboard, once no chart uses it."""
        found = extension_writes.read("delete_dataset", request)
        dataset = self.datasets.get(found["dataset_id"])
        if dataset is None:
            return {"not_found": "dataset"}
        if not self._editor_of(dataset, user):
            return {"not_editor": True, "editors": sorted(dataset["editors"])}
        marker = extension_params.parse_marker(dataset["description"])
        if marker is None:
            return {"refused": "no_marker"}
        if marker.dashboard_id != found["dashboard_id"]:
            return {"refused": "another_dashboard"}
        if in_use := sorted(c["id"] for c in self.charts.values() if c["datasource_id"] == dataset["id"]):
            return {"in_use": True, "chart_ids": in_use}
        del self.datasets[dataset["id"]]
        return {"ok": True, "soft_deleted": True}

    def _delete_chart(self, request: dict, user: str) -> dict:
        """MCP's `delete_chart` under `SOFT_DELETE`: the chart's editors only, a chart a report holds refused; the
        chart goes to the trash and stays a member of its dashboards, as rc2 keeps it."""
        chart = self.charts.get(request["identifier"])
        if chart is None:
            return {"success": False, "error": "No chart found", "error_type": "NotFound"}
        if not self._editor_of(chart, user):
            return {"success": False, "permission_denied": True, "error": "no", "error_type": "Forbidden"}
        if chart["id"] in self.reported:
            return {"success": False, "error": "reports exist", "error_type": "ReportsExist"}
        self.trashed[chart["id"]] = self.charts.pop(chart["id"])
        return {"success": True, "deleted_id": chart["id"], "deleted_name": chart["slice_name"],
                "soft_deleted": True}

    def _get_layout(self, request: dict, user: str) -> dict:
        found = extension_reads.read("get_layout", request)
        dashboard = self.dashboards.get(found["dashboard_id"])
        if dashboard is None:
            return {"not_found": "dashboard"}
        return {"dashboard_id": dashboard["id"], "position_json": dashboard["position_json"],
                "changed_on": self._iso(dashboard["changed_on"])}

    def _editors_of(self, request: dict, user: str) -> dict:
        found = extension_reads.read("editors_of", request)
        answer = {"charts": [{"chart_id": i, "is_editor": user in self.charts[i]["editors"],
                              "editors": sorted(self.charts[i]["editors"])}
                             for i in found["chart_ids"] if i in self.charts],
                  "datasets": [{"dataset_id": i, "is_editor": user in self.datasets[i]["editors"],
                                "editors": sorted(self.datasets[i]["editors"])}
                               for i in found["dataset_ids"] if i in self.datasets]}
        if found["dashboard_ids"]:
            answer["dashboards"] = [{"dashboard_id": i, "is_editor": user in self.dashboards[i]["editors"],
                                     "editors": sorted(self.dashboards[i]["editors"])}
                                    for i in found["dashboard_ids"] if i in self.dashboards]
        return answer

    def _set_layout(self, request: dict, user: str) -> dict:
        found = extension_writes.read("set_layout", request)
        dashboard = self.dashboards.get(found["dashboard_id"])
        if dashboard is None:
            return {"not_found": "dashboard"}
        if not self._editor_of(dashboard, user):
            return {"not_editor": True, "editors": sorted(dashboard["editors"])}
        if dashboard["changed_on"] != found["expect_changed_on"]:
            return {"stale": True}
        added = found["add_chart_ids"]
        if blocked := sorted(c for c in added if c not in self.charts):
            return {"refused": "chart_access", "chart_ids": blocked}
        current = json.loads(dashboard["position_json"]) if dashboard["position_json"] else {}
        refusal = extension_params.check_positions(found["positions"], current_positions=current,
                                                   current_chart_ids=dashboard["members"], add_chart_ids=added,
                                                   remove_chart_ids=found["remove_chart_ids"],
                                                   trashed_chart_ids=[m for m in dashboard["members"]
                                                                      if m in self.trashed])
        if refusal is not None:
            return refusal
        metadata = extension_params.layout_metadata(found["positions"], json.loads(dashboard["json_metadata"]))
        # A chart in the trash stays a member through a layout write that leaves it out, as seen live on rc2: a chart a
        # Remove moved to the trash and took off the layout was still in `dashboard_slices` after the write
        # (`DashboardDAO.set_dash_metadata` rebuilds `dashboard.slices` from the layout's charts; the trashed row is
        # not removed).
        trashed = {chart_id for chart_id in dashboard["members"] if chart_id in self.trashed}
        members = sorted(set(extension_params._layout_chart_ids(found["positions"])) | trashed)
        dashboard.update(position_json=json.dumps(metadata["positions"]), members=members,
                         json_metadata=json.dumps({k: v for k, v in metadata.items() if k != "positions"}),
                         changed_on=self._stamp())
        return {"ok": True}

    # --- reading numbers: a chart's saved query, and mnemiq's own statement

    def _get_chart_data(self, request: dict, user: str) -> dict:
        chart = self.charts.get(request["identifier"])
        if chart is None:
            return {"error": "not found", "error_type": "NotFound"}
        if chart["query_context"] is None:
            # A rebuild from form_data is rc2's, not the world's: no chart a build makes is left without a query.
            return {"error": "the world rebuilds no query", "error_type": "NoQueryContext"}
        dataset = self.datasets[chart["datasource_id"]]
        results = [self._run_query(query, dataset, user) for query in chart["query_context"]["queries"]]
        columns, rows = results[0]
        if chart["slice_name"] in self.skewed and rows:
            last = columns[-1]
            rows[0] = {**rows[0], last: rows[0][last] + self.skewed[chart["slice_name"]]}
        if not rows:
            return {"error": f"No data available for chart {chart['id']}", "error_type": "NoData"}
        answer = {"chart_id": chart["id"], "chart_name": chart["slice_name"], "chart_type": chart["viz_type"],
                  "columns": [{"name": name} for name in columns], "data": rows, "row_count": len(rows),
                  "total_rows": len(rows)}
        if len(results) > 1:
            answer["query_results"] = [{"query_index": i, "columns": c, "data": r, "row_count": len(r)}
                                       for i, (c, r) in enumerate(results)]
        return answer

    def _as(self, user: str, sql: str) -> tuple[list, list]:
        """`sql` run as `user`, its cursor's description and rows: with a row rule of theirs, the name `policy` reads
        the rows it keeps (a view in a schema of their own, first on the search path), as Superset adds the rule to
        every statement they run."""
        rule = self.user_rules.get(user)
        if rule is None:
            cursor = self.db.execute(sql)
            return cursor.description, cursor.fetchall()
        schema = f"ruled_{user}"
        self.db.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        self.db.execute(f'CREATE OR REPLACE VIEW "{schema}"."policy" AS SELECT * FROM main."policy" WHERE {rule}')
        self.db.execute(f"SET search_path = '{schema},main'")
        try:
            cursor = self.db.execute(sql)
            return cursor.description, cursor.fetchall()
        finally:
            self.db.execute("SET search_path = 'main'")

    def _run_query(self, query: dict, dataset: dict, user: str = "builder") -> tuple[list[str], list[dict]]:
        """A query of a saved query context, as Superset compiles one for a physical dataset."""
        saved = {m["metric_name"]: m["expression"] for m in dataset["metrics"]}
        selected, labels, groups = [], [], []
        for column in query.get("columns") or []:
            if isinstance(column, str):
                expression, label = f'"{column}"', column
            else:
                expression, label = f'"{column["sqlExpression"]}"', column["label"]
                if column.get("timeGrain"):
                    expression = f"DATE_TRUNC('{_GRAIN_UNIT[column['timeGrain']]}', {expression})"
            selected.append(f'{expression} AS "{label}"')
            labels.append(label)
            groups.append(expression)
        for metric in query.get("metrics") or []:
            if isinstance(metric, str):
                expression, label = saved[metric], metric
            else:
                expression, label = metric["sqlExpression"], metric["label"]
            selected.append(f'{expression} AS "{label}"')
            labels.append(label)
        where = []
        for item in query.get("filters") or []:
            column, op, value = f'"{item["col"]}"', item["op"], item.get("val")
            if op == "TEMPORAL_RANGE":
                continue
            if op in ("IN", "NOT IN"):
                listed = ", ".join(self._literal(v) for v in value)
                where.append(f"{column} {op} ({listed})")
            elif op in ("IS NULL", "IS NOT NULL"):
                where.append(f"{column} {op}")
            else:
                where.append(f"{column} {_OPERATORS[op]} {self._literal(value)}")
        source = dataset["sql"] and f"({dataset['sql']}) AS virtual_table" or f'"{dataset["table_name"]}"'
        sql = f"SELECT {', '.join(selected)} FROM {source}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        # The query's own order, as rc2's `get_sqla_query` writes it (`sa.desc`/`sa.asc`, no NULLS clause), with
        # Postgres's NULL placement spelled out, since DuckDB's differs: NULL first under DESC, last under ASC.
        order = []
        for item, ascending in query.get("orderby") or []:
            expression = saved[item] if isinstance(item, str) and item in saved else \
                f'"{item}"' if isinstance(item, str) else item["sqlExpression"]
            order.append(f"{expression} {'ASC NULLS LAST' if ascending else 'DESC NULLS FIRST'}")
        if groups:
            sql += " GROUP BY " + ", ".join(groups)
        if having := (query.get("extras") or {}).get("having"):
            sql += f" HAVING {having}"
        if order or groups:
            sql += " ORDER BY " + ", ".join(order or groups)
        if query.get("row_limit"):
            sql += f" LIMIT {int(query['row_limit'])}"
        found = self._as(user, sql)[1]
        return labels, [{label: self._json(v) for label, v in zip(labels, row)} for row in found]

    @staticmethod
    def _literal(value) -> str:
        return str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
            else "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def _json(value):
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    def _execute_sql(self, request: dict, user: str) -> dict:
        sql = request["sql"]
        if request.get("force_refresh") or request.get("dry_run") or sql.rstrip().endswith(
                "AS mnemiq_probe WHERE 1 = 0"):
            return EMPTY
        described, rows = self._as(user, sql)
        names = [d[0] for d in described]
        types = []
        for index, description in enumerate(described):
            kind = str(description[1]).upper()
            types.append("datetime64[ns]" if "TIMESTAMP" in kind else "int64" if "INT" in kind
                         else "float64" if kind in ("DOUBLE", "FLOAT") or kind.startswith("DECIMAL") else "object")
        out = []
        for row in rows:
            out.append({name: (value.isoformat() if isinstance(value, datetime) else
                               float(value) if types[i] == "float64" and value is not None else value)
                        for i, (name, value) in enumerate(zip(names, row))})
        return {"success": True, "columns": [{"name": n, "type": t} for n, t in zip(names, types)], "rows": out}


def _labelled_columns(config: dict) -> list[dict]:
    """Every `ColumnRef` of a `generate_chart` config that carries a label: the SQL metrics a build sends."""
    found: list[dict] = []
    for value in config.values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict) and isinstance(item.get("label"), str):
                found.append(item)
    return found


def dataset_id_of(world: World, table_name: str) -> int:
    [found] = [d["id"] for d in world.datasets.values() if d["table_name"] == table_name]
    return found
