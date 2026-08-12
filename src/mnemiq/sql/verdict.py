from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RefusalCode(StrEnum):
    PARSE_ERROR = "parse_error"
    NOT_A_SINGLE_STATEMENT = "not_a_single_statement"
    NOT_SELECT_ONLY = "not_select_only"
    SELECT_STAR = "select_star"
    UNAUTHORIZED_TABLE = "unauthorized_table"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"
    EXPLAIN_FAILED = "explain_failed"
    LOGIC_LINT = "logic_lint"
    VALUE_GROUNDING = "value_grounding"
    NOT_A_WRITE = "not_a_write"
    UNBOUNDED_WRITE = "unbounded_write"
    UNAUTHORIZED_WRITE = "unauthorized_write"
    UNAUTHORIZED_COLUMN = "unauthorized_column"
    MASKED_COLUMN_IN_PREDICATE = "masked_column_in_predicate"
    INVALID_ROW_FILTER = "invalid_row_filter"
    # Deployment-level, not grant-level: this deployment does not do writes at all. Distinct from
    # UNAUTHORIZED_WRITE, which is a statement about THIS identity's grants (M3).
    WRITES_DISABLED = "writes_disabled"


# A refusal the model can act on is a repair; a refusal it cannot act on is a dead end.
# UNAUTHORIZED_TABLE is deliberately absent: an unauthorized reference is never repaired
# into existence -- it is deferred (plan_query.py).
REPAIRABLE = frozenset(
    {
        RefusalCode.PARSE_ERROR,
        RefusalCode.NOT_A_SINGLE_STATEMENT,
        RefusalCode.NOT_SELECT_ONLY,
        RefusalCode.SELECT_STAR,
        RefusalCode.UNKNOWN_TABLE,
        RefusalCode.UNKNOWN_COLUMN,
        RefusalCode.EXPLAIN_FAILED,
        RefusalCode.LOGIC_LINT,
        RefusalCode.VALUE_GROUNDING,
    }
)


@dataclass
class Refusal:
    code: RefusalCode
    message: str  # written for the model to repair against, not for a log
    subject: str | None = None

    @property
    def repairable(self) -> bool:
        return self.code in REPAIRABLE


@dataclass
class Approved:
    plan_sql: str  # the dialect the model wrote in
    target_sql: str  # what will actually run on the source
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    # Set by `plan_query`, never by `decide` -- the decider does not repair, it only rules.
    # It rides here because it is a fact about THIS plan: the SQL that was approved is not the
    # SQL the model first proposed. Without it the corrector is the one mode difference nobody
    # can observe (M33).
    corrected: bool = False


@dataclass
class ApprovedWrite:
    plan_sql: str
    target_sql: str
    target: str  # the mutated table
    tables: list[str] = field(default_factory=list)  # all referenced tables


Verdict = Approved | Refusal
