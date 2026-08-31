from enum import StrEnum

from typing import Literal

from pydantic import BaseModel, Field


class DeferralReason(StrEnum):
    """Why there is no answer -- categorised by what the CALLER should do next, not by where in
    the code it arose (register M6).

    It lives here, beside IdentityContext and Trace, because it crosses the same boundaries: the
    planner raises it, the agent carries it, and MCP / the CLI / the metrics all read it. Before
    this, every terminal state was `deferred=True` plus a sentence, so a governing agent could not
    tell "ask for a grant" from "rephrase the question" from "page an operator" -- three different
    required responses behind one boolean.
    """

    AUTHORIZATION = "authorization"  # you lack access -> request a grant
    POLICY_UNAVAILABLE = "policy_unavailable"  # the policy could not be read -> page an operator
    NO_TABLES = "no_tables"  # nothing retrieved -> rephrase, or check enrichment
    UNANSWERABLE = "unanswerable"  # the model could not form a query from these tables
    # M35: the question names a business term nobody certified a meaning for -> get it defined.
    #
    # A separate code on M6's own test, which is what the caller should DO next. UNANSWERABLE says
    # "rephrase, these tables cannot answer that"; this says "the tables can answer it, and nobody
    # has said what the term MEANS" -- and the fix is external and specific, closer in shape to
    # AUTHORIZATION's "request a grant" than to "try different words". Rephrasing cannot help.
    #
    # It is also the only thing that makes the guard MEASURABLE. Beacon's grader reads deferred as
    # a boolean and never inspects the reason, but the reason is persisted and queryable -- and
    # today every deferral in the corpus carries the identical string `unanswerable`, so a pass
    # cannot be told from luck. A distinct code lands in `output.reason` with no harness change.
    UNDEFINED_TERM = "undefined_term"
    INVALID_QUERY = "invalid_query"  # could not produce valid SQL within the budget
    VERIFICATION = "verification"  # we answered, then the verifier declined to stand behind it
    DISAGREEMENT = "disagreement"  # candidates diverged too much to pick one
    EXECUTION_FAILED = "execution_failed"  # the source rejected every attempt -- NOT a deferral
    MODEL_UNAVAILABLE = "model_unavailable"  # the model provider did not answer -- NOT a deferral


class IdentityContext(BaseModel):
    tenant_id: str
    principal_id: str
    email: str | None = None
    roles: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)


class HistoryTurn(BaseModel):
    """One prior turn of a conversation, carried forward as structure, not prose.

    A follow-up like "and how many claims does it have?" needs an antecedent for "it",
    and the previous SQL does not contain one -- `ORDER BY total DESC LIMIT 1` computes
    the top region without naming it. So the answer's RESULT is what resolves the
    pronoun, and a few rows of it ride along beside the query that produced them.

    `grant_fingerprint` is the authorization boundary this turn was answered under. A
    turn is replayed only to the same boundary: otherwise a privileged turn's values
    (a person's name, a masked column) could steer a query asked by an identity that
    may not see them, and the new answer would disclose through the filter what the
    policy withheld from the column.
    """

    question: str
    sql: str = ""  # the query that ran -- what was actually asked of the data
    tables_used: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    rows: list[list[object]] = Field(default_factory=list)  # bounded; the antecedent
    grant_fingerprint: str = ""


class Trace(BaseModel):
    question: str
    plan_sql: str
    target_sql: str
    result_shape: str
    timing: dict[str, float]
    enrichment_version: str
    identity: IdentityContext
    tables_used: list[str] = Field(default_factory=list)
    # M56: whether `tables_used` is the whole story. `complete` | `incomplete` | `unknown`, and
    # `unknown` by default because a Trace built without one has not established completeness.
    lineage_completeness: Literal["complete", "incomplete", "unknown"] = "unknown"
    # Object ids and function names -- the caller's vocabulary.
    lineage_unresolved: list[str] = Field(default_factory=list)
    # This engine's reason codes. A separate field because a consumer rendering `unresolved` as
    # objects would otherwise show `scope-unresolved` as a table name.
    lineage_reasons: list[str] = Field(default_factory=list)
    definitions_used: list[str] = Field(default_factory=list)
