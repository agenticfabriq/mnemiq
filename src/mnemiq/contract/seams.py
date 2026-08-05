from enum import StrEnum

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


class Trace(BaseModel):
    question: str
    plan_sql: str
    target_sql: str
    result_shape: str
    timing: dict[str, float]
    enrichment_version: str
    identity: IdentityContext
    tables_used: list[str] = Field(default_factory=list)
    definitions_used: list[str] = Field(default_factory=list)
