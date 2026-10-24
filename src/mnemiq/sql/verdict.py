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
    # Nothing here can be attributed, because this source's name resolution cannot be
    # reproduced: it defines a name a builtin also has, it could not say, or -- once -- the
    # statement would not render, which is the one arrival that is about the statement rather
    # than the source. Kept apart from UNMODELLED_CALL because the two have opposite repairs:
    # that one names a function to avoid, this one usually condemns every query against the
    # source (M43's residual). `_cannot_resolve` is where the arrivals are told apart.
    UNRESOLVABLE_CALLS = "unresolvable_calls"
    # A source that is not a named table or a subquery over one -- LATERAL, VALUES, UNNEST, a
    # table literal -- so `base_tables` cannot say what the query reads (M65).
    UNMODELLED_SOURCE = "unmodelled_source"
    LOGIC_LINT = "logic_lint"
    VALUE_GROUNDING = "value_grounding"
    # An aggregate over rows a join has multiplied: runs fine, silently inflated (M109).
    FAN_OUT = "fan_out"
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
    # A row or column policy, decided for an engine whose binding of a CTE name the scope rule does
    # not model (M150). A deployment's configuration, not the statement: never repairable.
    POLICY_TARGET_UNSUPPORTED = "policy_target_unsupported"
    # The source accepts the statement as asked and refuses it once the access policy is applied:
    # a row filter, or the snapshot it was checked against, no longer matches the source (M158).
    # Never repairable: every rewrite of the question meets the same policy.
    POLICY_REWRITE_REJECTED = "policy_rewrite_rejected"
    # The source could not be asked whether a statement is valid: a connection fault, a timeout,
    # an interrupt, as the adapter reads its driver's errors (Codex's review of #159). Not the
    # statement's, so never repairable: every rewrite meets the same silence. Distinct from
    # EXPLAIN_FAILED, which is the source's answer, no.
    SOURCE_UNAVAILABLE = "source_unavailable"
    # A LIMIT or FETCH that states no plain row count -- a percentage, WITH TIES, an expression --
    # so the row cap can neither keep it nor replace it without changing the answer (M162).
    UNREADABLE_LIMIT = "unreadable_limit"
    # A bare niladic name (`current_date`, `session_user`) where a table the decider knows has
    # a column so named: this engine reads the column, and no check sees a column through the
    # function's node (M194). Repairable: the column qualified, or a call the engine keeps.
    AMBIGUOUS_NILADIC_NAME = "ambiguous_niladic_name"
    # The statement has no rendering in the target's dialect: sqlglot's generator refused to print
    # it (a DuckDB MAP literal keyed by a column or a lambda's parameter, for Postgres or Oracle,
    # which name a struct's field by a plain identifier: Codex's forty-seventh review of #160),
    # where the error escaped both deciders. Repairable: another spelling of the question may
    # print. A policy's rewrite that does not print where the statement does is
    # POLICY_TARGET_UNSUPPORTED, the deployment's.
    UNPRINTABLE_FOR_TARGET = "unprintable_for_target"


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
        RefusalCode.AMBIGUOUS_NILADIC_NAME,
        # Repairable on purpose. The measured cost of this guard is a legitimate scalar
        # function sqlglot does not model -- Postgres `age()` was the one case in 459 --
        # and the repair loop can rewrite that into arithmetic the engine does model.
        # Without this the guard's false positives become deferrals instead of retries.
        RefusalCode.UNMODELLED_CALL,
        # Repairable for the same reason: the message names the shape, and the same query can be
        # written with tables, joins and subqueries. Measured cost: none of 9,687 gold queries
        # (BIRD mini-dev in SQLite, Postgres and MySQL, KaggleDBQA, BEAVER, Spider2-lite local)
        # reads through one, measured before LATERAL subqueries were admitted -- the stricter
        # rule, so the count can only have fallen (M65).
        RefusalCode.UNMODELLED_SOURCE,
        # The three below were absent while nothing read this set, so nothing had checked them
        # against the loop that ignored it. Each says what to do differently, in its own
        # message, which is the test for belonging here:
        #
        #   MASKED_COLUMN_IN_PREDICATE -- "may only be selected, not used in a filter"
        #   UNGOVERNED_VIEW            -- "Query the table directly."
        #   UNRESOLVABLE_VIEW          -- the view is the problem, and the model chose to use it
        #
        # The last is the weakest of the three: its messages name the view without saying to
        # avoid it, and a self-referential view is the source's DDL rather than anything the
        # model did. Listed anyway, because the safe direction here is the one that changes no
        # behaviour, and a wasted retry costs a model call while a wrong refusal costs an answer.
        RefusalCode.MASKED_COLUMN_IN_PREDICATE,
        RefusalCode.UNGOVERNED_VIEW,
        RefusalCode.UNRESOLVABLE_VIEW,
        # UNRESOLVABLE_CALLS is deliberately absent. It condemns every statement against the
        # source, not one function, so there is nothing for a rewrite to avoid and a retry loop
        # would spend every attempt to reach the deferral it starts at. The fix belongs to the
        # deployer, who sees this in the trace, not to the model. POLICY_REWRITE_REJECTED is
        # absent for the same reason: no rewrite of the question escapes the policy (M158), and
        # SOURCE_UNAVAILABLE because no source read the statement, so there is nothing to repair
        # against.
        #
        # `plan_query` reads this now (M98). It used to retry every refusal but
        # UNAUTHORIZED_TABLE, so an unrepairable code was retried until the attempts ran out and
        # arrived as INVALID_QUERY -- three model calls to reach a verdict the first one had.
        RefusalCode.LOGIC_LINT,
        RefusalCode.VALUE_GROUNDING,
        # The message names the table the join repeats and says how to restructure: aggregate
        # each table on its own, then join the aggregates.
        RefusalCode.FAN_OUT,
        # The message says to state the bound as a plain count, which the same query can always do.
        RefusalCode.UNREADABLE_LIMIT,
        # The message names the engine the statement cannot be printed for; another spelling can be.
        RefusalCode.UNPRINTABLE_FOR_TARGET,
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
    # caller as `trace.asked_sql` (the executed `target_sql` stays in the audit trace, M51), so a
    # model can still launder text through a string literal. Suppressing that would cost the
    # transparency the SQL surface exists for, and
    # a partial-containment check is a guard with a false-positive rate nobody has measured.
    source_detail: str | None = None

    # When the CODE is not enough to say. One code can be reached by arrivals that differ on
    # this: `UNRESOLVABLE_CALLS` condemns the whole source when it defines a name a builtin also
    # has, and is about this one statement when the statement would not render -- whose message
    # asks for a rewrite. Left None, the code decides, which is what every other refusal wants.
    repairable_override: bool | None = None

    @property
    def repairable(self) -> bool:
        if self.repairable_override is not None:
            return self.repairable_override
        return self.code in REPAIRABLE

    @property
    def repair_text(self) -> str:
        """What the model repairs against: our refusal, plus the source's own words -- with any
        credential in them taken out first (M96: a model can copy what it reads into a literal)."""
        if not self.source_detail:
            return self.message
        from mnemiq.redact import scrub_credentials

        return f"{self.message} {scrub_credentials(self.source_detail)}"


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
    # The row limit the guard imposed, when the query did not bound itself tighter (M118). A
    # result with this many rows may have been cut; `None` means any limit is the query's own.
    row_cap: int | None = None
    # What the access policy narrowed, table-granular, on the SAME rule as `lineage`: the fact
    # travels with the answer or not at all -- INCLUDING its absence. `None` is "not evaluated",
    # `[]` is the claim "the policy narrowed nothing". They are different sentences and a default
    # of `[]` would say the second whenever nobody said anything, which is how M56 started:
    # `Approved` is constructed outside any decider (`eval/verify_replay.py`), and there the
    # honest value is None.
    narrowed: list | None = None
    # M51: the statement as asked -- shaped and row-capped, in the target's dialect -- before any
    # row or column policy rewrote it. What a CALLER is shown: the executed `target_sql` carries
    # the policy's own predicate, which under an entitlements filter names a table and literals
    # the caller is never granted. Equal to `target_sql` when the policy changed nothing, which
    # is when `narrowed` is empty. None where no decider set it, so a caller is shown nothing
    # rather than the executed statement by default.
    asked_sql: str | None = None
    # The same statement as asked, in the PLAN's dialect, for the MODEL calls -- the judge, the
    # selector, the synthesizer -- to read through `model_sql()` in place of `plan_sql`. Shown the
    # rewritten `plan_sql`, the judge read the policy's own predicate as a filter the question
    # never implied and withdrew every answer under a row filter (measured by the ui lane, M156),
    # and an entitlements predicate would carry tables and literals the role is not granted to the
    # model server. All three read `model_sql()` (`Verifier.verify`, the selector's `ClusterView`s
    # and `Agent`'s synthesis step); a new model call must too. Equal to `plan_sql` when nothing
    # was narrowed, so a prompt with no policy is byte-identical. Not `asked_sql`: that is the
    # target's dialect, which would change every prompt on a source whose target differs. Never
    # what runs. None where no decider set it.
    asked_plan: str | None = None

    def model_sql(self) -> str:
        """What a model call may be shown of this statement (M156): the statement as asked, or
        `plan_sql` where no policy was evaluated (`narrowed` None, as `mnemiq.eval.verify_replay`
        builds it). With a policy evaluated and no `asked_plan`, it RAISES rather than fall back:
        the fallback would be the rewritten plan, the predicate and any entitlements literals
        included, which is what this exists to keep from the model server."""
        if self.asked_plan is not None:
            return self.asked_plan
        if self.narrowed is None:
            return self.plan_sql
        raise ValueError("a policy was evaluated but the statement as asked was not recorded, "
                         "so there is no SQL a model call may be shown")


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
    # M51, as on `Approved`: the write as asked, before the row-filter rewrite.
    asked_sql: str | None = None


Verdict = Approved | Refusal
