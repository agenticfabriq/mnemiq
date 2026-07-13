from pydantic import BaseModel, Field


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
