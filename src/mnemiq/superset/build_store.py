"""What a dashboard build keeps, and where: each build's plan, its steps and their leases, and each dashboard's list of
measures, in two tables of mnemiq's control Postgres (`MNEMIQ_CONTROL_DSN`), made the first time a store connects.

Nothing kept is a value. A build's row holds the goal it was asked for, its plan, and for each step its state, the
intent it was claimed with and the id of the object it made. A measure's row holds the measure's definition, the ids of
the charts that show it and of the datasets mnemiq made for it, and each chart's check status.

The guarantee is upstream, not here: the planner is given no value (it sees the schema, saved metrics' names and
expressions, and dashboards' titles), a value check's numbers ride back in its response only, and a measure's value is
computed through Superset as whoever asks, each time, and kept nowhere. The store's own checks are a backstop. An intent
holds four named fields and a measure's row ids and statuses, so neither has room for a number; a step's reason is
one of `REASONS`, never text it was given. A plan is held to the shape the planner declares beside its writer
(`planner.PLAN_JSON_SHAPE`): an allow-list of its keys at every depth refuses any other key and an object or a list
where the writer writes a single value, so a value has no key to be kept under. Beneath it,
a key that reads, folded (NFKC, stripped, casefolded), as one a value, a row or a sample is kept under
(`VALUE_SHAPED_KEYS`) is refused too.

A build is read only by its tenant and the principal who started it. A dashboard's measures are the dashboard's: any
principal of the tenant reads them, and Superset checks editorship on each action that changes the dashboard. A
dashboard has one build building at a time (`DashboardBusy`), since its measures are kept as the last build to write
them leaves them; Postgres holds that rule, as a unique index over the dashboards of builds still building. A build
left untouched for `ABANDONED_AFTER_S`, no claim of it holding, does not keep its dashboard: the next build of the
dashboard fails its next step as `abandoned`, in the transaction that creates the new one, and so does a new
dashboard's step that Superset gave the id the old build holds, before it finishes once more.

**Steps and leases.** A build runs one step at a time, in order: its dashboard (a new shell, or the read of an existing
one), then its datasets, its charts, its layout and its checks. `claim_next_step` claims the first step not done before
any Superset call, with one compare-and-set `UPDATE` of the build's row: the step list as it was read is the condition,
so of two claims decided on the same row, one writes and the other finds the row changed, reads it again and is
`StepBusy`. The claim records the step's intent, its `started_at` and a lease, all on Postgres's clock, so workers on
different machines agree on when a lease passes.

A worker renews its lease before each of its calls (`renew_lease`), and starts no call once that fails. A lease that
has passed turns its step `uncertain`, and the next claim takes that step to reconcile it, keeping the first claim's
intent and `started_at`. A worker whose call ended unanswered finishes its step `uncertain` itself, so the next claim
need not wait for the lease: each call a step makes may be sent again, a create among them, which the extension answers
with the object the first one made. Each claim has a lease id of its own, so a worker whose step was taken from it can
neither renew nor finish it (`LeaseLost`).
"""

from __future__ import annotations

import copy
import hashlib
import math
import threading
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg.types.json import Jsonb

from mnemiq.superset.chart_check import ChartStatus
from mnemiq.superset.marker import format_marker
from mnemiq.superset.measures import Measure, MeasureKind

if TYPE_CHECKING:
    from mnemiq.config import Settings

__all__ = ["ABANDONED_AFTER_S", "BUILD_TABLE", "DEFAULT_LEASE_S", "LOCK_TIMEOUT_MS", "MAX_ID", "MAX_UNFINISHED",
           "MEASURE_TABLE", "PROPOSED", "REASONS",
           "STATEMENT_TIMEOUT_MS", "STEP_KINDS", "STORE_MISSING", "VALUE_SHAPED_KEYS",
           "BuildNotFound", "BuildStore", "BuildStoreMissing", "DashboardBusy", "DashboardMeasure", "LeaseLost",
           "StepBusy", "StepState", "StoredBuild"]

BUILD_TABLE = "mnemiq_superset_build"
MEASURE_TABLE = "mnemiq_superset_measure"

# The kinds of step, in the order a build runs them. A build's steps keep this order, and its first is its dashboard.
STEP_KINDS = ("dashboard", "dataset", "chart", "layout", "check")
# The kinds whose step makes (or, for an existing dashboard, reads) an object, and so finishes `done` with its id.
_MAKES_AN_OBJECT = frozenset({"dashboard", "dataset", "chart"})

PENDING = "pending"
DONE = "done"
UNCERTAIN = "uncertain"
FAILED = "failed"
# What a worker may finish its step as. `pending` and the lease are the store's own.
_FINISHED_AS = frozenset({DONE, FAILED, UNCERTAIN})

# A build's own status, read from its steps on every write: any step failed, every step done, or neither.
BUILDING = "building"
# A build planned and waiting for the person to confirm it: it holds no dashboard, and has no step to claim, until
# `confirm_build` makes it `building`.
PROPOSED = "proposed"

# The names a value, a row, a result or a sample is kept under. No key of a stored plan reads, folded (`_folded`), as
# one of them.
VALUE_SHAPED_KEYS = frozenset({"value", "values", "row", "rows", "data", "result", "results", "sample", "samples"})
_INTENT_KEYS = ("key", "kind", "name", "dataset_id")

# Why a step failed, or was left uncertain, as a code: what a later read of the build says, never Superset's words.
REASONS = frozenset({"not_editor", "not_found", "claimed_by_another", "extension_refused", "not_saved", "too_large",
                     "candidates", "not_made", "dataset_exists", "layout_unreadable", "kept_changing",
                     "measure_conflict", "tool_error", "abandoned", "unreadable", "busy", "superset_silent",
                     "lease_passed"})

DEFAULT_LEASE_S = 600  # longer than any one Superset call's deadline
# How long a build may go untouched (every claim, renewal and finish touches it), no claim of it holding, before
# another build of its dashboard takes the dashboard over.
ABANDONED_AFTER_S = 2 * DEFAULT_LEASE_S
_MAX_LEASE_S = 86_400
# The largest id the store keeps: Postgres's `integer`, which every id column is. A caller that takes an id from a
# request bounds it by this before any work.
MAX_ID = 2**31 - 1
# How many times a write decided on a row that changed under it reads the row again. Only a lease's holder writes a
# build while the lease holds, so a second attempt is the most that is ever needed.
_ATTEMPTS = 3
_CONTENDED = object()

# How long one of the store's statements may wait for a lock, and run in all, before Postgres cancels it. Every server
# thread shares the store's one connection, so without a bound one statement blocked on a held row would stall every
# build behind it. Each statement here is a single-row read or write that takes milliseconds.
LOCK_TIMEOUT_MS = 10_000
STATEMENT_TIMEOUT_MS = 30_000

STORE_MISSING = ("Building a dashboard needs mnemiq's control Postgres, where each build keeps its plan and its steps; "
                 "set MNEMIQ_CONTROL_DSN.")

_BUILD_COLUMNS = ("tenant_id, build_id, principal_id, dashboard_id, goal, profile, plan, steps, status, created_at, "
                  "updated_at, planned_against")
_MEASURE_COLUMNS = "name, definition, chart_ids, made_dataset_ids, check_status, updated_at"

_CREATE_BUILD = f"""
CREATE TABLE IF NOT EXISTS {BUILD_TABLE} (
    tenant_id text NOT NULL,
    build_id uuid NOT NULL,
    principal_id text NOT NULL,
    dashboard_id integer,
    goal text NOT NULL,
    profile text,
    plan jsonb NOT NULL,
    steps jsonb NOT NULL,
    status text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, build_id))"""
_CREATE_MEASURE = f"""
CREATE TABLE IF NOT EXISTS {MEASURE_TABLE} (
    tenant_id text NOT NULL,
    dashboard_id integer NOT NULL,
    name text NOT NULL,
    definition jsonb NOT NULL,
    chart_ids integer[] NOT NULL DEFAULT '{{}}',
    made_dataset_ids integer[] NOT NULL DEFAULT '{{}}',
    check_status jsonb NOT NULL DEFAULT '{{}}',
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, dashboard_id, name))"""
# One build building per dashboard of a tenant, held by Postgres: a second insert, or a new dashboard's step naming a
# dashboard another build is building, fails on it, however the two writes interleave.
_LIVE_DASHBOARD = f"{BUILD_TABLE}_live_dashboard"
_CREATE_LIVE_DASHBOARD = (f"CREATE UNIQUE INDEX IF NOT EXISTS {_LIVE_DASHBOARD} ON {BUILD_TABLE} "
                          f"(tenant_id, dashboard_id) WHERE status = '{BUILDING}' AND dashboard_id IS NOT NULL")
# The dashboard's `changed_on` a proposal was planned against, read again when it is confirmed. Added to a table made
# before proposals were kept, so an existing deployment's builds read as started at once (null).
_ADD_PLANNED_AGAINST = f"ALTER TABLE {BUILD_TABLE} ADD COLUMN IF NOT EXISTS planned_against text"
# A builder's builds by status: what the list of their unfinished builds reads.
_CREATE_PRINCIPAL_INDEX = (f"CREATE INDEX IF NOT EXISTS {BUILD_TABLE}_principal ON {BUILD_TABLE} "
                           "(tenant_id, principal_id, status)")
# The most unfinished builds a list holds.
MAX_UNFINISHED = 50
# Two servers starting at once would both run `CREATE TABLE IF NOT EXISTS`, and Postgres can fail the second with a
# unique violation on its catalogue rather than find the table; the tables are made under this advisory lock instead.
_DDL_LOCK = int.from_bytes(hashlib.sha256(b"mnemiq.superset.build_store").digest()[:8], "big", signed=True)


class BuildStoreMissing(Exception):
    """A build asked of a deployment with no control Postgres configured. The message names the setting."""

    def __init__(self) -> None:
        super().__init__(STORE_MISSING)


class BuildNotFound(LookupError):
    """No build of this id for this tenant and principal: another's build reads exactly as one that does not exist."""

    def __init__(self) -> None:
        super().__init__("no such build for this tenant and principal")


class DashboardBusy(Exception):
    """Another build of this dashboard, for this tenant, is still building. A dashboard's measures are kept as its
    builds write them, the last write standing, so one build works on a dashboard at a time. The other build is not
    named: it may be another principal's, and a build is read only by its own."""

    def __init__(self) -> None:
        super().__init__("another build of this dashboard is still building; it must finish or stop first")


class LeaseLost(Exception):
    """A worker finished a step whose claim is no longer its own: the step was finished, or its lease passed and
    another claim took it to reconcile. What the worker made is found by that reconciliation, by its marker."""

    def __init__(self) -> None:
        super().__init__("this step's claim is no longer this worker's")


@dataclass(frozen=True)
class StepState:
    """One step of a build. `status` is `pending`, `done`, `uncertain` or `failed`; a step whose lease has passed reads
    `uncertain`. `intent` is what the claim recorded (`key`, `kind`, `name` and `dataset_id`), `started_at` when, both
    kept from the first claim while the step is reconciled.
    `lease_id` names the claim holding the step, if any, and `lease_until` when its lease passes. `reason` is why it
    failed or was left uncertain, one of `REASONS`. Times are ISO 8601 in UTC, on Postgres's clock."""

    key: str
    kind: str
    status: str
    object_id: int | None
    started_at: str | None
    intent: dict | None = None
    lease_id: str | None = None
    lease_until: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class StepBusy:
    """No step claimed: `key` is the step whose lease still holds, or None when the build has moved past the step the
    intent named, so the caller read it before another request ran a step."""

    key: str | None


@dataclass(frozen=True)
class StoredBuild:
    tenant_id: str
    build_id: str
    principal_id: str
    dashboard_id: int | None
    goal: str
    plan: dict
    steps: tuple[StepState, ...]
    status: str
    created_at: str
    updated_at: str
    # The profile the build was started under, whose settings each of its steps runs under; None: the deployment's
    # default.
    profile: str | None = None
    # The dashboard's `changed_on` the build was planned against, kept for a proposal onto a dashboard; None otherwise.
    planned_against: str | None = None

    def step(self, key: str) -> StepState:
        for step in self.steps:
            if step.key == key:
                return step
        raise KeyError(key)


@dataclass(frozen=True)
class DashboardMeasure:
    """One of a dashboard's measures: its definition, the charts that show it, the datasets mnemiq made for it, each
    chart's check status, and, for a saved metric a chart reuses, that metric's name."""

    measure: Measure
    chart_ids: tuple[int, ...]
    made_dataset_ids: tuple[int, ...]
    check_status: dict[int, ChartStatus]
    updated_at: str
    saved_metric: str | None = None


class BuildStore:
    """The builds and measure lists in the control Postgres at `dsn`.

    One connection, made on the first call and again after it breaks, never when the store is made; each call holds it
    alone, so one store is shared safely across a server's threads. Its session waits at most `LOCK_TIMEOUT_MS` for a
    lock and `STATEMENT_TIMEOUT_MS` for a statement, then raises. Every argument is checked before anything is sent,
    and nothing fails soft: a store that cannot answer raises, since a build step that is not recorded must not run."""

    def __init__(self, dsn: str | None) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise BuildStoreMissing()
        self._dsn = dsn
        self._con: psycopg.Connection | None = None
        self._lock = threading.Lock()

    @classmethod
    def for_settings(cls, settings: Settings) -> BuildStore:
        """The store at the deployment's `control_dsn`, or BuildStoreMissing naming MNEMIQ_CONTROL_DSN."""
        return cls(getattr(settings, "control_dsn", None))

    def close(self) -> None:
        with self._lock:
            self._discard()

    def __enter__(self) -> BuildStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- builds

    def create_build(self, tenant: str, principal: str, goal: str, plan: Mapping[str, Any],
                     steps: Sequence[tuple[str, str]], *, dashboard_id: int | None = None,
                     profile: str | None = None, proposed: bool = False,
                     planned_against: str | None = None) -> StoredBuild:
        """A new build of `steps`, each a `(key, kind)` pair, every one pending. `dashboard_id` is the existing
        dashboard a build works on; a new dashboard's id is the build's once its dashboard step is done. `profile` is
        the profile it was started under, None for the deployment's default. A `proposed` build waits for
        `confirm_build`, holding no dashboard meanwhile; `planned_against` is the dashboard's `changed_on` its plan
        was made against, read again when it is confirmed."""
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        goal = _text(goal, "goal")
        profile = None if profile is None else _text(profile, "profile")
        dashboard_id = _id(dashboard_id, "a build's dashboard id", optional=True)
        if type(proposed) is not bool:
            raise ValueError("whether a build is proposed is true or false")
        planned_against = None if planned_against is None else _text(planned_against, "dashboard's changed_on")
        plan = _plan(plan)
        build_id = uuid.uuid4()
        step_list = _new_steps(str(build_id), steps)
        with self._connection() as con, _live_dashboard(), con.transaction():
            if dashboard_id is not None and not proposed:
                _release_abandoned(con, tenant, dashboard_id)
            row = con.execute(
                f"INSERT INTO {BUILD_TABLE} (tenant_id, build_id, principal_id, dashboard_id, goal, profile, plan, "
                f"steps, status, planned_against) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                f"RETURNING {_BUILD_COLUMNS}, now()",
                (tenant, build_id, principal, dashboard_id, goal, profile, Jsonb(plan), Jsonb(step_list),
                 PROPOSED if proposed else BUILDING, planned_against),
            ).fetchone()
        return _stored_build(row)

    def confirm_build(self, tenant: str, build_id: str, principal: str) -> StoredBuild:
        """Start the person's proposed build: `building` from now, its dashboard's one live build. `DashboardBusy`
        while another build of the dashboard is building, the proposal kept; BuildNotFound unless the build is this
        tenant's and principal's and still proposed, so a second confirm, or two at once, starts it once."""
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        build = _build_uuid(build_id)
        if build is None:
            raise BuildNotFound()
        with self._connection() as con, _live_dashboard(), con.transaction():
            row = con.execute(
                f"SELECT dashboard_id FROM {BUILD_TABLE} WHERE tenant_id = %s AND build_id = %s "
                "AND principal_id = %s AND status = %s FOR UPDATE", (tenant, build, principal, PROPOSED)).fetchone()
            if row is None:
                raise BuildNotFound()
            if row[0] is not None:
                _release_abandoned(con, tenant, row[0])
            written = con.execute(
                f"UPDATE {BUILD_TABLE} SET status = %s, updated_at = now() WHERE tenant_id = %s AND build_id = %s "
                f"AND status = %s RETURNING {_BUILD_COLUMNS}, now()", (BUILDING, tenant, build, PROPOSED)).fetchone()
        if written is None:  # not reached while the row is held: a second guard, not a path
            raise BuildNotFound()
        return _stored_build(written)

    def withdraw_proposal(self, tenant: str, build_id: str, principal: str) -> bool:
        """Delete the person's build while it is still proposed: what a proposal planned again replaces. False for
        any other build, which is kept."""
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        build = _build_uuid(build_id)
        if build is None:
            return False
        with self._connection() as con:
            row = con.execute(
                f"DELETE FROM {BUILD_TABLE} WHERE tenant_id = %s AND build_id = %s AND principal_id = %s "
                "AND status = %s RETURNING 1", (tenant, build, principal, PROPOSED)).fetchone()
        return row is not None

    def load_build(self, tenant: str, build_id: str, principal: str) -> StoredBuild | None:
        """The build, read by its tenant and its principal only; None for any other, as for an id that names none."""
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        build = _build_uuid(build_id)
        if build is None:
            return None
        with self._connection() as con:
            row = con.execute(
                f"SELECT {_BUILD_COLUMNS}, now() FROM {BUILD_TABLE} "
                "WHERE tenant_id = %s AND build_id = %s AND principal_id = %s",
                (tenant, build, principal),
            ).fetchone()
        return None if row is None else _stored_build(row)

    def claim_next_step(self, tenant: str, build_id: str, principal: str, intent: Mapping[str, Any],
                        lease_s: int = DEFAULT_LEASE_S) -> StepState | StepBusy | None:
        """Claim the build's next step, the first not done, for `lease_s` seconds, recording `intent`: the step's
        `key` and `kind`, its planned `name` and `dataset_id`. A lease that has passed turns its step `uncertain` first,
        and that step is the one claimed, with its first intent and `started_at` kept.

        The claimed step comes back with its `lease_id`, and its status says what to do: `pending` to run it,
        `uncertain` to reconcile it. `StepBusy` while another claim's lease holds, or when `intent` names a step that is
        not the next. None when nothing is left to claim: every step is done, or one failed and the build stopped.
        BuildNotFound unless the build is this tenant's and this principal's."""
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        intent = _intent(intent)
        lease = timedelta(seconds=_lease(lease_s))
        build = _build_uuid(build_id)
        if build is None:
            raise BuildNotFound()
        lease_id = str(uuid.uuid4())

        def claim(steps: list[dict], dashboard_id: int | None, now: datetime):
            changed, outcome = _claim(steps, intent, now, lease, lease_id)
            return changed, dashboard_id, outcome

        outcome, _ = self._change(tenant, build, principal, claim)
        return StepBusy(key=None) if outcome is _CONTENDED else outcome

    def renew_lease(self, tenant: str, build_id: str, key: str, lease_id: str,
                    lease_s: int = DEFAULT_LEASE_S) -> bool:
        """Extend the claim `lease_id` holds on step `key` to `lease_s` seconds from now, if it still holds it and its
        lease has not passed. False otherwise: the worker starts no further call, and leaves the step to be
        reconciled."""
        tenant, key, lease_id = _text(tenant, "tenant"), _text(key, "step key"), _text(lease_id, "lease id")
        lease = timedelta(seconds=_lease(lease_s))
        build = _build_uuid(build_id)
        if build is None:
            return False

        def renew(steps: list[dict], dashboard_id: int | None, now: datetime):
            changed, renewed = _renew(steps, key, lease_id, now, lease)
            return changed, dashboard_id, renewed

        try:
            renewed, _ = self._change(tenant, build, None, renew)
        except BuildNotFound:
            return False
        return renewed is True

    def finish_step(self, tenant: str, build_id: str, key: str, status: str, object_id: int | None, *,
                    lease_id: str, reason: str | None = None) -> StoredBuild:
        """Record step `key` as `done`, `failed` or `uncertain`, with the id of the object it made or read (required
        for a done dashboard, dataset or chart step), ending the claim `lease_id` holds on it. A done dashboard step's
        id is the build's dashboard, refused with DashboardBusy while another build of that dashboard is building; a
        build of it left untouched for `ABANDONED_AFTER_S` is released first, as `create_build` releases one, so a
        build that named the id before Superset gave it to this dashboard cannot hold it for good. LeaseLost when the
        claim is no longer `lease_id`'s; nothing is written then. `reason` is one of `REASONS`, why the step failed or
        is left uncertain."""
        tenant, key = _text(tenant, "tenant"), _text(key, "step key")
        if not isinstance(status, str) or status not in _FINISHED_AS:
            raise ValueError(f"a step finishes as one of {sorted(_FINISHED_AS)}")
        object_id = _id(object_id, "a step's object id", optional=True)
        reason = _reason(reason)
        lease_id = _text(lease_id, "lease id")
        build = _build_uuid(build_id)
        if build is None:
            raise BuildNotFound()

        def finish(steps: list[dict], dashboard_id: int | None, now: datetime):
            return (*_finish(steps, dashboard_id, key, lease_id, status, object_id, reason), True)

        try:
            outcome, stored = self._change(tenant, build, None, finish)
        except DashboardBusy:
            # Only a done dashboard step gives a build its dashboard, so the dashboard another build holds is this
            # one: a holder left untouched is released, and the finish is tried once more.
            if status != DONE or object_id is None:
                raise
            with self._connection() as con, con.transaction():
                _release_abandoned(con, tenant, object_id)
            outcome, stored = self._change(tenant, build, None, finish)
        if outcome is _CONTENDED:
            raise LeaseLost()
        return stored

    def unfinished_builds(self, tenant: str, principal: str, *, dashboard_id: int | None = None,
                          limit: int = 10) -> list[StoredBuild]:
        """The principal's builds still building, newest touched first, at most `limit` (1 to `MAX_UNFINISHED`):
        with `dashboard_id`, only that dashboard's, so a build with no dashboard yet is listed only without it."""
        tenant, principal = _text(tenant, "tenant"), _text(principal, "principal")
        dashboard_id = _id(dashboard_id, "a dashboard id", optional=True)
        if type(limit) is not int or not 1 <= limit <= MAX_UNFINISHED:
            raise ValueError(f"a list holds 1 to {MAX_UNFINISHED} builds")
        where, values = "tenant_id = %s AND principal_id = %s AND status = %s", [tenant, principal, BUILDING]
        if dashboard_id is not None:
            where += " AND dashboard_id = %s"
            values.append(dashboard_id)
        with self._connection() as con:
            rows = con.execute(f"SELECT {_BUILD_COLUMNS}, now() FROM {BUILD_TABLE} WHERE {where} "
                               "ORDER BY updated_at DESC, build_id LIMIT %s", (*values, limit)).fetchall()
        return [_stored_build(row) for row in rows]

    def dashboard_building(self, tenant: str, dashboard_id: int) -> bool:
        """Whether a build of the tenant's is building on the dashboard, once each one left untouched for
        `ABANDONED_AFTER_S` has been released: what an action on the dashboard's measures asks first, since a build
        keeps the measures as it last writes them."""
        tenant = _text(tenant, "tenant")
        dashboard_id = _id(dashboard_id, "a dashboard id")
        with self._connection() as con, con.transaction():
            _release_abandoned(con, tenant, dashboard_id)
            row = con.execute(f"SELECT 1 FROM {BUILD_TABLE} WHERE tenant_id = %s AND dashboard_id = %s AND "
                              "status = %s LIMIT 1", (tenant, dashboard_id, BUILDING)).fetchone()
        return row is not None

    # --- a dashboard's measures

    def measures_for(self, tenant: str, dashboard_id: int) -> list[DashboardMeasure]:
        """The dashboard's measures, by name: the tenant's, whoever put them."""
        tenant = _text(tenant, "tenant")
        dashboard_id = _id(dashboard_id, "a dashboard id")
        with self._connection() as con:
            rows = con.execute(
                f"SELECT {_MEASURE_COLUMNS} FROM {MEASURE_TABLE} WHERE tenant_id = %s AND dashboard_id = %s "
                'ORDER BY name COLLATE "C"',
                (tenant, dashboard_id),
            ).fetchall()
        return [_dashboard_measure(row) for row in rows]

    def put_measure(self, tenant: str, dashboard_id: int, measure: Measure, *, chart_ids: Iterable[int] = (),
                    made_dataset_ids: Iterable[int] = (),
                    check_status: Mapping[int, ChartStatus | str] | None = None,
                    saved_metric: str | None = None) -> DashboardMeasure:
        """Keep `measure` on the dashboard, replacing one of its name: its definition, the charts that show it, the
        datasets mnemiq made for it, and each chart's check status, a chart id to a `ChartStatus`. `saved_metric`
        names the saved metric a measure is, for one a chart reuses rather than writes."""
        tenant = _text(tenant, "tenant")
        dashboard_id = _id(dashboard_id, "a dashboard id")
        definition = _definition(measure, saved_metric)
        charts = _ids(chart_ids, "chart_ids")
        made = _ids(made_dataset_ids, "made_dataset_ids")
        statuses = _check_status({} if check_status is None else check_status)
        with self._connection() as con:
            row = con.execute(
                f"INSERT INTO {MEASURE_TABLE} (tenant_id, dashboard_id, name, definition, chart_ids, made_dataset_ids, "
                "check_status) VALUES (%s, %s, %s, %s, %s::integer[], %s::integer[], %s) "
                "ON CONFLICT (tenant_id, dashboard_id, name) DO UPDATE SET definition = EXCLUDED.definition, "
                "chart_ids = EXCLUDED.chart_ids, made_dataset_ids = EXCLUDED.made_dataset_ids, "
                f"check_status = EXCLUDED.check_status, updated_at = now() RETURNING {_MEASURE_COLUMNS}",
                (tenant, dashboard_id, measure.name, Jsonb(definition), charts, made, Jsonb(statuses)),
            ).fetchone()
        return _dashboard_measure(row)

    def delete_measure(self, tenant: str, dashboard_id: int, name: str) -> bool:
        """Remove the dashboard's measure `name`. False when it has none of that name."""
        tenant, name = _text(tenant, "tenant"), _text(name, "measure name")
        dashboard_id = _id(dashboard_id, "a dashboard id")
        with self._connection() as con:
            cur = con.execute(f"DELETE FROM {MEASURE_TABLE} WHERE tenant_id = %s AND dashboard_id = %s AND name = %s",
                              (tenant, dashboard_id, name))
        return cur.rowcount > 0

    # --- the connection, and the one way a build's row is written after it is made

    def _change(self, tenant: str, build: uuid.UUID, principal: str | None,
                change: Callable[[list[dict], int | None, datetime], tuple[list[dict] | None, int | None, Any]],
                ) -> tuple[Any, StoredBuild | None]:
        """Read the build's steps and dashboard with Postgres's `now()`, let `change` decide, and write what it returns
        with one compare-and-set `UPDATE`: only while the steps are still as read. A write that finds them changed reads
        the row again and lets `change` decide again. `change` returns the new steps (None: write nothing), the
        dashboard id, and the result; this returns the result and the build as written."""
        # A proposal has no step to change until it is confirmed: it reads here as no build.
        where = (f"tenant_id = %s AND build_id = %s AND status <> '{PROPOSED}'"
                 + ("" if principal is None else " AND principal_id = %s"))
        key = (tenant, build) if principal is None else (tenant, build, principal)
        with self._connection() as con:
            for _ in range(_ATTEMPTS):
                row = con.execute(f"SELECT steps, dashboard_id, now() FROM {BUILD_TABLE} WHERE {where}",
                                  key).fetchone()
                if row is None:
                    raise BuildNotFound()
                steps, dashboard_id, now = row
                new_steps, new_dashboard_id, result = change(copy.deepcopy(steps), dashboard_id, now)
                if new_steps is None:
                    return result, None
                with _live_dashboard():
                    written = con.execute(
                        f"UPDATE {BUILD_TABLE} SET steps = %s, dashboard_id = %s, status = %s, updated_at = now() "
                        f"WHERE tenant_id = %s AND build_id = %s AND steps = %s RETURNING {_BUILD_COLUMNS}, now()",
                        (Jsonb(new_steps), new_dashboard_id, _build_status(new_steps), tenant, build, Jsonb(steps)),
                    ).fetchone()
                if written is not None:
                    return result, _stored_build(written)
        return _CONTENDED, None

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        with self._lock:
            if self._con is None or self._con.closed or self._con.broken:
                self._discard()
                con = psycopg.connect(self._dsn, autocommit=True)
                try:
                    _bound_waits(con)
                    _make_tables(con)
                except BaseException:
                    con.close()
                    raise
                self._con = con
            try:
                yield self._con
            except psycopg.OperationalError:
                self._discard()  # the next call connects afresh
                raise

    def _discard(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            finally:
                self._con = None


@contextmanager
def _live_dashboard() -> Iterator[None]:
    """A write that would give a dashboard a second build building raises DashboardBusy instead of Postgres's unique
    violation of `_LIVE_DASHBOARD`; any other violation is raised as it is."""
    try:
        yield
    except psycopg.errors.UniqueViolation as exc:
        if exc.diag.constraint_name != _LIVE_DASHBOARD:
            raise
        raise DashboardBusy() from None


def _release_abandoned(con: psycopg.Connection, tenant: str, dashboard_id: int) -> None:
    """In the transaction that creates a build of the dashboard, or before a new dashboard's step is finished once
    more: each build of the dashboard still building, untouched for `ABANDONED_AFTER_S` with no claim of it holding,
    has its next step failed as abandoned, so it no longer holds the dashboard. Its row is held `FOR UPDATE` while it
    is decided."""
    rows = con.execute(f"SELECT build_id, steps, updated_at, now() FROM {BUILD_TABLE} WHERE tenant_id = %s AND "
                       "dashboard_id = %s AND status = %s FOR UPDATE", (tenant, dashboard_id, BUILDING)).fetchall()
    for build, steps, updated_at, now in rows:
        released = _released(steps, now, updated_at)
        if released is not None:
            con.execute(f"UPDATE {BUILD_TABLE} SET steps = %s, status = %s, updated_at = now() "
                        "WHERE tenant_id = %s AND build_id = %s",
                        (Jsonb(released), _build_status(released), tenant, build))


def _bound_waits(con: psycopg.Connection) -> None:
    """Bound how long the session's statements wait for a lock and run, read when the store connects."""
    con.execute("SELECT set_config('lock_timeout', %s, false), set_config('statement_timeout', %s, false)",
                (f"{LOCK_TIMEOUT_MS}ms", f"{STATEMENT_TIMEOUT_MS}ms"))


def _make_tables(con: psycopg.Connection) -> None:
    with con.transaction():
        con.execute("SELECT pg_advisory_xact_lock(%s)", (_DDL_LOCK,))
        con.execute(_CREATE_BUILD)
        con.execute(_ADD_PLANNED_AGAINST)
        con.execute(_CREATE_LIVE_DASHBOARD)
        con.execute(_CREATE_PRINCIPAL_INDEX)
        con.execute(_CREATE_MEASURE)


# --- How each write changes a build's steps: pure, over the step list as read, and `now` on Postgres's clock.

def _claim(steps: list[dict], intent: dict, now: datetime, lease: timedelta,
           lease_id: str) -> tuple[list[dict] | None, StepState | StepBusy | None]:
    held = [step for step in steps if _holds(step, now)]
    if held:
        return None, StepBusy(key=held[0]["key"])
    for step in steps:
        if _passed(step, now):
            step.update(status=UNCERTAIN, lease_id=None, lease_until=None)
    if any(step["status"] == FAILED for step in steps):
        return None, None
    target = next((step for step in steps if step["status"] != DONE), None)
    if target is None:
        return None, None
    if target["key"] != intent["key"]:
        return None, StepBusy(key=None)
    if target["kind"] != intent["kind"]:
        raise ValueError(f"step {target['key']!r} is a {target['kind']} step, not a {intent['kind']} step")
    if target["status"] == PENDING or target["intent"] is None:
        target.update(intent=intent, started_at=_iso(now))
    # An uncertain step keeps the intent and `started_at` of the claim that left it so.
    target.update(lease_id=lease_id, lease_until=_iso(now + lease))
    return steps, _step_state(target, now)


def _renew(steps: list[dict], key: str, lease_id: str, now: datetime,
           lease: timedelta) -> tuple[list[dict] | None, bool]:
    step = next((step for step in steps if step["key"] == key), None)
    if step is None or step["lease_id"] != lease_id or not _holds(step, now):
        return None, False
    step["lease_until"] = _iso(now + lease)
    return steps, True


def _released(steps: list[dict], now: datetime, updated_at: datetime) -> list[dict] | None:
    """`steps` with the next step not done failed as abandoned, when the build was last touched `ABANDONED_AFTER_S`
    or more before `now` and no claim of it holds; None when it still holds its dashboard."""
    if updated_at > now - timedelta(seconds=ABANDONED_AFTER_S) or any(_holds(step, now) for step in steps):
        return None
    target = next((step for step in steps if step["status"] != DONE), None)
    if target is None:
        return None
    target.update(status=FAILED, reason="abandoned", lease_id=None, lease_until=None)
    return steps


def _finish(steps: list[dict], dashboard_id: int | None, key: str, lease_id: str, status: str,
            object_id: int | None, reason: str | None = None) -> tuple[list[dict], int | None]:
    step = next((step for step in steps if step["key"] == key), None)
    if step is None:
        raise ValueError(f"this build has no step {key!r}")
    if step["lease_id"] != lease_id:
        raise LeaseLost()
    if status == DONE and step["kind"] in _MAKES_AN_OBJECT and object_id is None:
        raise ValueError(f"a done {step['kind']} step names the object it made")
    if status == DONE and step["kind"] == "dashboard":
        if dashboard_id is not None and dashboard_id != object_id:
            raise ValueError("this build's dashboard step finishes with the build's own dashboard")
        dashboard_id = object_id
    step.update(status=status, object_id=object_id, lease_id=None, lease_until=None, reason=reason)
    return steps, dashboard_id


def _holds(step: dict, now: datetime) -> bool:
    return step["lease_id"] is not None and datetime.fromisoformat(step["lease_until"]) > now


def _passed(step: dict, now: datetime) -> bool:
    return step["lease_id"] is not None and datetime.fromisoformat(step["lease_until"]) <= now


def _build_status(steps: list[dict]) -> str:
    if any(step["status"] == FAILED for step in steps):
        return FAILED
    if all(step["status"] == DONE for step in steps):
        return DONE
    return BUILDING


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


# --- Reading rows back.

def _step_state(step: dict, now: datetime) -> StepState:
    return StepState(key=step["key"], kind=step["kind"], status=UNCERTAIN if _passed(step, now) else step["status"],
                     object_id=step["object_id"], started_at=step["started_at"], intent=_kept_intent(step["intent"]),
                     lease_id=step["lease_id"], lease_until=step["lease_until"], reason=step.get("reason"))


def _kept_intent(intent: dict | None) -> dict | None:
    """A stored intent as it reads: its `_INTENT_KEYS`. An older build's holds an `id_floor` too, the highest id it
    read before its create, which no step reads any more; it is left out."""
    return None if intent is None else {name: value for name, value in intent.items() if name in _INTENT_KEYS}


def _stored_build(row: tuple) -> StoredBuild:
    (tenant, build, principal, dashboard_id, goal, profile, plan, steps, status, created_at, updated_at,
     planned_against, now) = row
    return StoredBuild(tenant_id=tenant, build_id=str(build), principal_id=principal, dashboard_id=dashboard_id,
                       goal=goal, plan=plan, steps=tuple(_step_state(step, now) for step in steps), status=status,
                       created_at=_iso(created_at), updated_at=_iso(updated_at), profile=profile,
                       planned_against=planned_against)


def _dashboard_measure(row: tuple) -> DashboardMeasure:
    name, definition, chart_ids, made_dataset_ids, check_status, updated_at = row
    measure = Measure(name=name, label=definition["label"], expression=definition["expression"],
                      kind=MeasureKind(definition["kind"]), dataset_id=definition["dataset_id"],
                      description=definition["description"], asked_by=definition["asked_by"],
                      currency=definition["currency"])
    return DashboardMeasure(measure=measure, chart_ids=tuple(chart_ids), made_dataset_ids=tuple(made_dataset_ids),
                            check_status={int(chart): ChartStatus(status) for chart, status in check_status.items()},
                            updated_at=_iso(updated_at), saved_metric=definition.get("saved_metric"))


# --- What is checked before anything is sent. No message repeats a value it was given.

def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"a {what} is a non-empty string")
    return value


def _id(value: object, what: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= MAX_ID:
        raise ValueError(f"{what} is a positive integer id")
    return value


def _ids(values: Iterable[int], what: str) -> list[int]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise ValueError(f"{what} is a list of ids")
    ids = [_id(value, f"each of {what}") for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{what} names an id twice")
    return ids


def _reason(value: object) -> str | None:
    if value is not None and value not in REASONS:
        raise ValueError(f"a step's reason is one of {sorted(REASONS)}")
    return value


def _lease(lease_s: object) -> int:
    if isinstance(lease_s, bool) or not isinstance(lease_s, int) or not 0 < lease_s <= _MAX_LEASE_S:
        raise ValueError(f"a lease is a whole number of seconds from 1 to {_MAX_LEASE_S}")
    return lease_s


def _build_uuid(build_id: object) -> uuid.UUID | None:
    if not isinstance(build_id, str):
        return None
    try:
        return uuid.UUID(build_id)
    except ValueError:
        return None


def _new_steps(build_id: str, steps: Sequence[tuple[str, str]]) -> list[dict]:
    """The build's steps, every one pending, refused unless they run in `STEP_KINDS`' order from one dashboard step
    first, each key unique and one the build's markers can carry."""
    if isinstance(steps, (str, bytes, Mapping)) or not isinstance(steps, Iterable):
        raise ValueError("a build's steps are a list of (key, kind) pairs")
    made: list[dict] = []
    for position, pair in enumerate(steps):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"step {position} is not a (key, kind) pair")
        key, kind = pair
        if kind not in STEP_KINDS:
            raise ValueError(f"step {position}'s kind is not one of {list(STEP_KINDS)}")
        try:
            format_marker(build_id, key, None)
        except ValueError:
            raise ValueError(f"step {position}'s key is not one a marker carries: 1 to 40 of a-z, 0-9 and _") from None
        if any(step["key"] == key for step in made):
            raise ValueError(f"step key {key!r} is used twice")
        if (kind == "dashboard") != (position == 0):
            raise ValueError("a build's first step, and only its first, is its dashboard")
        if made and STEP_KINDS.index(kind) < STEP_KINDS.index(made[-1]["kind"]):
            raise ValueError(f"step {key!r} is a {kind} step after a {made[-1]['kind']} step; a build runs "
                             f"{', '.join(STEP_KINDS)}, in that order")
        made.append({"key": key, "kind": kind, "status": PENDING, "object_id": None, "started_at": None,
                     "intent": None, "lease_id": None, "lease_until": None, "reason": None})
    if not made:
        raise ValueError("a build has at least one step, its dashboard")
    return made


def _intent(intent: object) -> dict:
    """The intent, refused unless it holds the step's `key` and `kind`, and at most its planned `name` (a string) and
    `dataset_id` (an id); with all four keys, absent ones None. `started_at` is the store's own."""
    if not isinstance(intent, Mapping):
        raise ValueError("an intent is a mapping")
    other = [str(name) for name in intent if name not in _INTENT_KEYS]
    if other:
        raise ValueError(f"an intent holds {list(_INTENT_KEYS)} only, not {sorted(other)}")
    key, kind, name = intent.get("key"), intent.get("kind"), intent.get("name")
    if not isinstance(key, str) or not key:
        raise ValueError("an intent names its step's key")
    if kind not in STEP_KINDS:
        raise ValueError(f"an intent's kind is one of {list(STEP_KINDS)}")
    if name is not None and not isinstance(name, str):
        raise ValueError("an intent's name is a string")
    dataset_id = _id(intent.get("dataset_id"), "an intent's dataset id", optional=True)
    return {"key": key, "kind": kind, "name": name, "dataset_id": dataset_id}


def _plan(plan: object) -> dict:
    """A plain JSON copy of the plan (tuples become lists), refused when it is not a mapping of JSON, when a key at any
    depth is one a value is kept under, or when it is not shaped as the planner writes a plan (`_plan_shape`)."""
    if not isinstance(plan, Mapping):
        raise ValueError("a build's plan is a mapping")
    copied = _plain_json(plan, "plan")
    _shaped(copied, _plan_shape(), "plan")
    return copied


def _plan_shape() -> Any:
    """The shape the planner declares beside its writer (`planner.PLAN_JSON_SHAPE`), read when a plan is checked."""
    from mnemiq.superset import planner

    return planner.PLAN_JSON_SHAPE


def _shaped(doc: Any, shape: Any, path: str, *, writer: str = "the planner",
            kept: str = "a build keeps only its plan") -> None:
    """Refuse what `shape` (`planner.PLAN_JSON_SHAPE`'s form) does not admit, naming where by the keys admitted on
    the way, never a key or a value it was given. `writer` and `kept` word the refusal for the store that checks: the
    conversation store holds its messages to a shape of the same form."""
    from mnemiq.superset.planner import SCALAR

    if isinstance(shape, tuple):
        refusals = []
        for each in shape:
            try:
                return _shaped(doc, each, path, writer=writer, kept=kept)
            except ValueError as refused:
                refusals.append(refused)
        raise refusals[0]
    if isinstance(shape, dict):
        if not isinstance(doc, Mapping):
            raise ValueError(f"{path} holds something else where {writer} writes an object")
        for key, item in doc.items():
            if key not in shape:
                raise ValueError(f"{path} holds a key {writer} never writes there: {kept}")
            _shaped(item, shape[key], f"{path}.{key}", writer=writer, kept=kept)
    elif isinstance(shape, list):
        if not isinstance(doc, list):
            raise ValueError(f"{path} holds something else where {writer} writes a list")
        for index, item in enumerate(doc):
            _shaped(item, shape[0], f"{path}[{index}]", writer=writer, kept=kept)
    elif shape == SCALAR:
        if isinstance(doc, (Mapping, list)):
            raise ValueError(f"{path} holds an object or a list where {writer} writes a single value")
    else:
        raise ValueError(f"{path}: {writer}'s declared shape is not one the store reads")


def _plain_json(doc: object, path: str, *, holder: str = "a build") -> Any:
    if isinstance(doc, Mapping):
        copied = {}
        for key, item in doc.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} has a key that is not a string")
            if _folded(key) in VALUE_SHAPED_KEYS:
                raise ValueError(f"{path} holds a key named {key!r}: {holder} keeps no value, row or result")
            copied[key] = _plain_json(item, f"{path}.{key}", holder=holder)
        return copied
    if isinstance(doc, (list, tuple)):
        return [_plain_json(item, f"{path}[{index}]", holder=holder) for index, item in enumerate(doc)]
    if isinstance(doc, float) and not math.isfinite(doc):
        raise ValueError(f"{path} holds a number JSON cannot carry")
    if doc is None or isinstance(doc, (str, int, float)):
        return doc
    raise ValueError(f"{path} holds a {type(doc).__name__}, which is not JSON")


def _folded(key: str) -> str:
    """A key as the plan check compares it: NFKC (a fullwidth or other compatibility spelling is the plain one), with
    surrounding space stripped, casefolded."""
    return unicodedata.normalize("NFKC", key).strip().casefold()


def _definition(measure: object, saved_metric: object = None) -> dict:
    """A measure's definition under its seven keys, and an eighth, `saved_metric`, for a saved metric a chart reuses.
    The name is the row's own key."""
    if not isinstance(measure, Measure):
        raise ValueError("a dashboard's measure is a Measure")
    _text(measure.name, "measure name")
    _text(measure.expression, "measure expression")
    for what, text in (("label", measure.label), ("description", measure.description),
                       ("asked_by", measure.asked_by)):
        if not isinstance(text, str):
            raise ValueError(f"a measure's {what} is a string")
    if measure.currency is not None and not isinstance(measure.currency, str):
        raise ValueError("a measure's currency is a string")
    try:
        kind = MeasureKind(measure.kind)
    except ValueError:
        raise ValueError(f"a measure's kind is one of {[kind.value for kind in MeasureKind]}") from None
    definition = {"expression": measure.expression, "label": measure.label, "kind": kind.value,
                  "currency": measure.currency, "description": measure.description, "asked_by": measure.asked_by,
                  "dataset_id": _id(measure.dataset_id, "a measure's dataset id")}
    if saved_metric is not None:
        definition["saved_metric"] = _text(saved_metric, "saved metric's name")
    return definition


def _check_status(statuses: object) -> dict[str, str]:
    """`{chart id: status}`, each status a `ChartStatus`, as jsonb holds it: the chart id as text."""
    if not isinstance(statuses, Mapping):
        raise ValueError("a check status maps a chart id to a status")
    kept: dict[str, str] = {}
    for chart, status in statuses.items():
        chart = _id(chart, "a check status's chart id")
        if not isinstance(status, str):
            raise ValueError(f"a check status is one of {[status.value for status in ChartStatus]}")
        try:
            kept[str(chart)] = ChartStatus(status).value
        except ValueError:
            raise ValueError(f"a check status is one of {[status.value for status in ChartStatus]}") from None
    return kept
