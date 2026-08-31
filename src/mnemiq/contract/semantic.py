from pydantic import BaseModel, ConfigDict, Field

from .values import CodedValue, JoinKey, MeasureExpr

# The closed vocabulary of `Column.pii_level`. Lives here, beside the field, so the producer
# (enrichment) and the consumers (the value-index gate, and the authz clearance check) read
# one set and cannot drift -- `pii_clearance`/`pii_mask` are sets drawn from these same levels,
# so a clearance value outside this set clears nothing (M40).
PII_LEVELS: tuple[str, ...] = ("none", "pii", "phi")


class SourceBinding(BaseModel):
    id: str
    source_id: str
    object_id: str
    source_object: str
    binding_type: str
    tenant_id: str | None = None
    freshness_ref: str | None = None


class CodeScheme(BaseModel):
    """The code system a column's values are drawn from (ICD-10, NAICS, ...). Set by the
    ontology binder from observed values, never by the LLM."""
    id: str
    label: str


class Column(BaseModel):
    id: str
    object_id: str
    name: str
    data_type: str | None = None
    semantic_type: str | None = None
    description: str | None = None
    pii_level: str | None = None
    coded_values: list[CodedValue] = Field(default_factory=list)
    code_scheme: CodeScheme | None = None
    # Profile counts. None means "profiling did not run", never zero -- a fabricated
    # zero would make every unprofiled column look empty.
    row_count: int | None = None
    distinct_count: int | None = None
    null_count: int | None = None


class ViewDefinition(BaseModel):
    """A view's body, as the source states it.

    Structural, not semantic: discovery reports it, no LLM touches it. It exists because a
    view is the one object a governed engine cannot reason about from its columns alone --
    the rows it returns are defined by SQL held in the source, and until that SQL is in the
    snapshot a granted view over a filtered table returns unfiltered rows (M27).

    `dialect` is the SOURCE's, not the executor's: `pg_get_viewdef` returns Postgres even when
    the adapter that fetched it executes DuckDB, and parsing it as anything else silently
    mangles casts and quoting.
    """

    object_id: str
    definition: str
    dialect: str


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
    # Fail-closed visibility. A definition is shown only if it is `public` (a standard whose text
    # names no table -- MPAA, ICD-10) OR it is bound to objects the identity is granted. An unbound,
    # non-public definition (a confidential internal taxonomy) is therefore visible to no one, rather
    # than to everyone as `all([]) == True` would otherwise imply.
    public: bool = False
    bound_objects: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class TableFacts(BaseModel):
    object_id: str
    grain: str | None = None
    gotchas: list[str] = Field(default_factory=list)
    canonical_measures: dict[str, str] = Field(default_factory=dict)  # business name -> expr
    default_time_column: str | None = None


class Example(BaseModel):
    question: str
    sql: str
    tables: list[str] = Field(default_factory=list)
    object_id: str


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
    db_id: str | None = None  # which source this question targets; None = the single/default source
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
    # WHY, when the status alone cannot say. A job records that something failed; without this the
    # cause survives only in the log of the run that produced it, and a snapshot outlives its run
    # -- so a reader months later can see THAT a column was not measured and never learn whether
    # it was temp space, a privilege, or a driver fault (**M73**). Optional and defaulted, so every
    # snapshot written before it parses unchanged, and `content_version` excludes jobs, so adding
    # it invalidates no cached model.
    detail: str | None = None
    tenant_id: str | None = None
    checkpoints: list[str] = Field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 1


class CertifiedRef(BaseModel):
    """Which certified version supplied a piece of meaning in this snapshot.

    M24: `apply_certified` applied `rec.payload` and dropped `rec.envelope`, so mnemiq carried the
    CONTENT of a certified record and could not say which VERSION produced it -- and an emitted
    trace could therefore never name what it relied on. Verity D131 is the mirror: it kept the
    reference at ingest and threw it away.

    Deliberately a list of typed refs rather than a `{object_id: version}` dict. `object_id` does
    not identify a record -- `loss_ratio` is legitimately both the metric and the glossary
    definition (Verity D124 was exactly that mistake) -- and this shape is already the wire type
    the trace carries, so it crosses without translation.
    """

    object_type: str
    object_id: str
    version_hash: str


class Snapshot(BaseModel):
    version: str
    source_id: str
    created_at: str
    # The version of the OntologyRecords this snapshot was enriched against (local digest merged
    # with governed/certified schemes). Folded into `content_version` so a shifted ontology
    # vocabulary invalidates the enrichment even when it does not alter any column's `code_scheme`
    # (which carries only the scheme id/label, never concept content). Empty = no ontology.
    ontology_version: str = ""
    columns: list[Column] = Field(default_factory=list)
    # Structural, and part of what the snapshot ASSERTS about the source: change a view's body
    # and the rows it returns change, so this is folded into `content_version` alongside the
    # columns rather than treated as bookkeeping.
    views: list[ViewDefinition] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    source_bindings: list[SourceBinding] = Field(default_factory=list)
    qualities: list[Quality] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    jobs: list[Job] = Field(default_factory=list)
    compatibility_profiles: list[CompatibilityProfile] = Field(default_factory=list)
    table_facts: list[TableFacts] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    # PROVENANCE, not content: which certified versions supplied the meaning above. Deliberately
    # absent from `content_version`'s body, so two snapshots asserting the same meaning from
    # different record versions still share a cache entry (M24).
    certified_refs: list[CertifiedRef] = Field(default_factory=list)
