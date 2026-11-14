"""What a follow-up in Superset mode is reminded of: the earlier turns, none with the rows the client sent back, and the
newest turn with SQL run again as the person asking now.

A follow-up such as "and how many claims does it have?" needs the rows the earlier turn returned, since its SQL computes
"it" without naming it. In direct mode the client sends those rows back with the turn. Here they are never trusted:
they may be another person's, or older than what this person may read now, since Superset's row rules and grants can
change between two questions. So every turn's rows are dropped, and the newest turn with SQL is decided again, as a new
question's SQL is decided, and run again through the request's own adapter. The rows it returns are what this person
may read now, and those are the rows the prompt carries.

Only the newest turn is run: it is the one "it" points back to, and one more statement bounds what a follow-up costs.

The statement is an ask-path data statement like any other: the adapter sends no `force_refresh`, so Superset's own
result cache may answer it, and that cache checks the person's access before it does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from mnemiq.agent.history import MAX_COLUMNS, MAX_ROWS, scope_history
from mnemiq.authz.grants import GrantSet
from mnemiq.contract import HistoryTurn, Snapshot
from mnemiq.execute.runner import ExecutionError, run
from mnemiq.sql.decide import decide
from mnemiq.sql.fanout_check import guard_on, key_facts
from mnemiq.sql.policy import build_access_policy
from mnemiq.sql.schema import column_types, visible_schema
from mnemiq.sql.verdict import Approved, Refusal
from mnemiq.sql.views import inventory_for

if TYPE_CHECKING:
    from mnemiq.superset.adapter import SupersetSqlAdapter

logger = logging.getLogger(__name__)


def superset_history(turns: list[HistoryTurn] | None, *, grants: GrantSet, snapshot: Snapshot,
                     adapter: SupersetSqlAdapter, timeout_s: float, values, guard_fanout: bool | None,
                     ) -> list[HistoryTurn]:
    """The turns this request is reminded of: scoped to `grants` as direct mode scopes them, each without the rows and
    columns the client sent, and the newest turn with SQL holding the rows that SQL returns now.

    That SQL is decided under `grants` with what the answer path decides a new question's SQL with: `values` and
    `guard_fanout` are the answering agent's own, and `adapter` is the request's, whose live function inventory the
    decider reads. Approved, it runs through `adapter` as the person within `timeout_s`, and the turn keeps the first
    `MAX_ROWS` rows of the first `MAX_COLUMNS` columns, named as the result names them.

    A refusal of a table, a failed run and a result Superset cut leave the turn without rows, and what Superset said
    goes to the log only. A refusal of the caller, of the caller's roles, or one Superset does not explain, met while
    the turn is decided or run (`SupersetSqlAdapter.terminal_refusal`), is raised instead: it stays the same for every
    later call this request makes, so the request ends here, as it does when the grants meet the same refusal."""
    kept = [turn.model_copy(update={"rows": [], "columns": []}) for turn in scope_history(turns, grants.fingerprint)]
    newest = next((i for i in reversed(range(len(kept))) if kept[i].sql.strip()), None)
    if newest is None:
        return kept
    result = _rows_now(kept[newest].sql, grants=grants, snapshot=snapshot, adapter=adapter, timeout_s=timeout_s,
                       values=values, guard_fanout=guard_fanout)
    if (refused := adapter.terminal_refusal) is not None:
        logger.warning("Superset refused this request (%s) while an earlier turn was run again; the request ends here",
                       type(refused).__name__)
        raise refused
    if result is not None:
        columns, rows = result
        kept[newest] = kept[newest].model_copy(update={"columns": columns, "rows": rows})
    return kept


def _rows_now(sql: str, *, grants: GrantSet, snapshot: Snapshot, adapter, timeout_s: float, values,
              guard_fanout: bool | None) -> tuple[list[str], list[list[object]]] | None:
    """The columns and the first rows `sql` returns when decided under `grants` and run through `adapter`, or None when
    it is not approved or does not run whole.

    Decided as `plan_query` decides the writer's SQL, argument for argument, in the source's dialect, so nothing is
    looser for a turn than for a new question. With no table granted a new question is deferred before anything is
    decided, and a turn is not run either."""
    if not visible_schema(snapshot, grants):
        logger.info("no table is granted, so the earlier turn is not run again and keeps no rows")
        return None
    try:
        outcome = run_decided(sql, adapter=adapter, snapshot=snapshot, grants=grants, timeout_s=timeout_s,
                              guard_fanout=guard_fanout, values=values)
    except ExecutionError as exc:
        # Every failure of the run reaches here as one: a result cut to size, a time limit, a cancel, a refusal, and
        # Superset not answering. What Superset said is in the exception's detail, or in the one it was raised from.
        detail = getattr(exc.__cause__, "detail", "")
        logger.warning("an earlier turn did not run again, and keeps no rows: %s%s", exc.full_text,
                       f" ({detail})" if detail else "")
        return None
    if isinstance(outcome, Refusal):
        # The subject and the source's words are the operator's to read, never the model's.
        logger.warning("an earlier turn's SQL was refused when decided again (%s, %r): it is not run, and keeps no "
                       "rows. %s", outcome.code.value, outcome.subject, outcome.source_detail or "")
        return None
    table = outcome.table
    width = min(table.num_columns, MAX_COLUMNS)
    head = table.slice(0, MAX_ROWS)
    columns = [head.column(i).to_pylist() for i in range(width)]
    return list(table.column_names[:width]), [list(row) for row in zip(*columns)]


@dataclass(frozen=True)
class Decided:
    """A statement mnemiq approved, and the table it returned when run through the request's adapter."""

    approved: Approved
    table: pa.Table


def decide_statement(sql: str, *, adapter, snapshot: Snapshot, grants: GrantSet, guard_fanout: bool | None,
                     values=None) -> Approved | Refusal:
    """`sql` decided under `grants` as `plan_query` decides the writer's SQL, argument for argument, in the source's
    dialect, so nothing is looser here than for a new question."""
    dialect = adapter.dialect
    return decide(
        sql, visible_schema(snapshot, grants), adapter=adapter, dialect=dialect, target=dialect, values=values,
        policy=build_access_policy(snapshot, grants), registry=getattr(snapshot, "registry", {}),
        views=inventory_for(snapshot), keys=key_facts(snapshot) if guard_on(guard_fanout, snapshot) else None,
        types=column_types(snapshot), gregorian_calendar=getattr(adapter, "gregorian_calendar", False) is True,
    )


def run_decided(sql: str, *, adapter, snapshot: Snapshot, grants: GrantSet, timeout_s: float,
                guard_fanout: bool | None, values=None) -> Decided | Refusal:
    """Decide `sql` under `grants`, and when it is approved run the approved statement through `adapter` as the person.

    A refusal is returned. A run that fails raises `ExecutionError`, whose detail is the caller's to log."""
    verdict = decide_statement(sql, adapter=adapter, snapshot=snapshot, grants=grants, guard_fanout=guard_fanout,
                               values=values)
    if not isinstance(verdict, Approved):
        return verdict
    return Decided(verdict, run(adapter, verdict.target_sql, timeout_s=timeout_s).table)
