from pydantic import BaseModel, ConfigDict, Field

from .values import CodedValue, JoinKey, MeasureExpr


class SourceBinding(BaseModel):
    id: str
    source_id: str
    object_id: str
    source_object: str
    binding_type: str
    tenant_id: str | None = None
    freshness_ref: str | None = None


class Column(BaseModel):
    id: str
    object_id: str
    name: str
    data_type: str | None = None
    semantic_type: str | None = None
    description: str | None = None
    pii_level: str | None = None
    coded_values: list[CodedValue] = Field(default_factory=list)
    # Profile counts. None means "profiling did not run", never zero -- a fabricated
    # zero would make every unprofiled column look empty.
    row_count: int | None = None
    distinct_count: int | None = None
    null_count: int | None = None


class Dimension(BaseModel):
    id: str
    label: str
    source: str
    expr: str | None = None
    data_type: str | None = None
    semantic_type: str | None = None


class Metric(BaseModel):
    id: str
    label: str
    status: str
    owner: str
    grain: str
    measure: MeasureExpr
    time_dimension: str
    compatible_dimensions: list[str] = Field(default_factory=list)


class Definition(BaseModel):
    id: str
    term: str
    domain: str
    definition: str
    bound_objects: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    from_: str = Field(alias="from")
    to: str
    cardinality: str
    join_keys: list[JoinKey] = Field(default_factory=list)


class Quality(BaseModel):
    id: str
    object_id: str
    name: str
    status: str
    threshold: str
    observed_value: str


class CompatibilityProfile(BaseModel):
    source_system: str
    source_version: str
    mapping_status: str
    lossless_concepts: list[str] = Field(default_factory=list)
    lossy_concepts: list[str] = Field(default_factory=list)
    unsupported_concepts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    id: str
    question: str
    # Ground truth is a QUERY, not a string. Result-based grading means any query returning
    # the same facts is correct -- a gold string cannot express that (spec 6.9).
    gold_sql: str | None = None
    answerable: bool = True  # False: the right behaviour is to defer, not to answer
    expected_answer: str | None = None  # human-readable documentation; never graded against
    metric_id: str | None = None
    dimensions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Skill(BaseModel):
    id: str
    question: str
    sql: str
    dialect: str
    source_id: str
    approval_ref: str | None = None


class Job(BaseModel):
    id: str
    source_id: str
    kind: str
    status: str
    tenant_id: str | None = None
    checkpoints: list[str] = Field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 1


class Snapshot(BaseModel):
    version: str
    source_id: str
    created_at: str
    columns: list[Column] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    source_bindings: list[SourceBinding] = Field(default_factory=list)
    qualities: list[Quality] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    jobs: list[Job] = Field(default_factory=list)
    compatibility_profiles: list[CompatibilityProfile] = Field(default_factory=list)
