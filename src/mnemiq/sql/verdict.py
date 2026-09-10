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
    # A call this engine cannot model, so it cannot say what the query reads. Not
    # "a forbidden function" -- there is no list of those, and that is the point (M43).
    UNMODELLED_CALL = "unmodelled_call"
    LOGIC_LINT = "logic_lint"
    VALUE_GROUNDING = "value_grounding"
    NOT_A_WRITE = "not_a_write"
    UNBOUNDED_WRITE = "unbounded_write"
    UNAUTHORIZED_WRITE = "unauthorized_write"
    UNAUTHORIZED_COLUMN = "unauthorized_column"
    MASKED_COLUMN_IN_PREDICATE = "masked_column_in_predicate"
    INVALID_ROW_FILTER = "invalid_row_filter"
    # A granted view whose body this engine cannot resolve, while a policy is active. Refusing
    # is the floor the full fix keeps as its error branch: an unresolvable view is one whose
    # base tables' filters cannot be applied, and answering from it anyway is M27 (M27).
    UNRESOLVABLE_VIEW = "unresolvable_view"
    # A granted view that reads a row-filtered table. The engine cannot apply the filter
    # through a view, so it declines rather than answer past it -- distinct from
    # UNRESOLVABLE_VIEW, which is a view it could not read at all (M27).
    # The source could not be asked what views it has, so the engine cannot confirm that a name
    # in the query is not a view reading around a row filter. Distinct from UNGOVERNED_VIEW,
    # which names a view it CAN see: this one reports a gap in what it knows, not in the policy.
    VIEW_INVENTORY_UNAVAILABLE = "view_inventory_unavailable"
    UNGOVERNED_VIEW = "ungoverned_view"
    # Deployment-level, not grant-level: this deployment does not do writes at all. Distinct from
    # UNAUTHORIZED_WRITE, which is a statement about THIS identity's grants (M3).
    WRITES_DISABLED = "writes_disabled"
    # A write whose CTEs sit outside the scope every guard walks, so the guards see an empty
    # statement and approve it ungoverned. Refused by shape rather than modelled, the same way
    # UNGOVERNED_VIEW refuses rather than reaching through a view.
    UNSCOPED_CTE = "unscoped_cte"
    # A write whose target this engine cannot resolve to exactly one table. An authorization
    # decision must not be made against a guess: the previous fallback picked the first table in
    # the tree, which for a multi-target DELETE is a SOURCE.
    AMBIGUOUS_WRITE_TARGET = "ambiguous_write_target"


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
        # Repairable on purpose. The measured cost of this guard is a legitimate scalar
        # function sqlglot does not model -- Postgres `age()` was the one case in 459 --
        # and the repair loop can rewrite that into arithmetic the engine does model.
        # Without this the guard's false positives become deferrals instead of retries.
        RefusalCode.UNMODELLED_CALL,
        RefusalCode.LOGIC_LINT,
        RefusalCode.VALUE_GROUNDING,
    }
)


@dataclass
class Refusal:
    code: RefusalCode
    message: str  # written for the model to repair against, not for a log
    subject: str | None = None
    # The source's own words, when a refusal has any. Deliberately NOT in `message`:
    # `plan_query` forwards `message` to the caller as a deferral reason, and a source's
    # exception is made of the caller's schema -- it names the relation and the column it
    # refused and can carry a DSN. Splitting it out is what lets the repair loop keep the
    # database's complaint (the cheapest accuracy lever there is) without the wire keeping
    # it too.
    #
    # The prompt is NOT a safe destination, and this field does not pretend otherwise:
    # retrieval scoping means the model may never have been shown the object a rejection
    # names, and a DSN is not schema at all. So `plan_query` withholds its model-authored
    # DEFERRAL text on a turn it fed source words into -- the stated reason, the declared
    # terms, the unauthorized subject -- keyed off `Feedback`, which carries that provenance
    # and has no default for it. Withholding here is one half; the other is refusing to let
    # the model hand it back.
    #
    # Bounded claim, deliberately. The approved SQL is model-authored too and ships to the
    # caller as `trace.target_sql`, so a model can still launder text through a string
    # literal. Suppressing that would cost the transparency the SQL surface exists for, and
    # a partial-containment check is a guard with a false-positive rate nobody has measured.
    source_detail: str | None = None

    @property
    def repairable(self) -> bool:
        return self.code in REPAIRABLE

    @property
    def repair_text(self) -> str:
        """What the model repairs against: our refusal, plus the source's own words."""
        return f"{self.message} {self.source_detail}" if self.source_detail else self.message


@dataclass
class Approved:
    plan_sql: str  # the dialect the model wrote in
    target_sql: str  # what will actually run on the source
    tables: list[str] = field(default_factory=list)
    # Whether `tables` is the whole story. It rides beside the list and never apart from it: a
    # bare list is worse than no list, because absent reads as "not recorded" and `[]` reads as
    # "nothing was read" (M56).
    lineage: object = None
    columns: list[str] = field(default_factory=list)
    # Set by `plan_query`, never by `decide` -- the decider does not repair, it only rules.
    # It rides here because it is a fact about THIS plan: the SQL that was approved is not the
    # SQL the model first proposed. Without it the corrector is the one mode difference nobody
    # can observe (M33).
    corrected: bool = False
    # What the access policy narrowed, table-granular, on the SAME rule as `lineage`: the fact
    # travels with the answer or not at all -- INCLUDING its absence. `None` is "not evaluated",
    # `[]` is the claim "the policy narrowed nothing". They are different sentences and a default
    # of `[]` would say the second whenever nobody said anything, which is how M56 started:
    # `Approved` is constructed outside any decider (`eval/verify_replay.py`), and there the
    # honest value is None.
    narrowed: list | None = None


@dataclass
class ApprovedWrite:
    plan_sql: str
    target_sql: str
    target: str  # the mutated table
    tables: list[str] = field(default_factory=list)  # all referenced tables
    # The write path narrows too, and its target is narrowed OUTSIDE the read loop, so this is the
    # only place a governed UPDATE/DELETE's own filtering can be reported from. `None` vs `[]`
    # carries the same distinction as on `Approved`.
    narrowed: list | None = None


Verdict = Approved | Refusal
