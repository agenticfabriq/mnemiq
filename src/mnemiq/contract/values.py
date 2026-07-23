from pydantic import BaseModel


class MeasureExpr(BaseModel):
    expr: str
    source: str


class CodedValue(BaseModel):
    code: str
    # None until semantic enrichment assigns it: structural profiling can observe that a
    # code exists, but only the semantic pass can say what it means.
    meaning: str | None = None
    # Provenance of `meaning`: "ontology" | "correlated" | "lookup" | "dictionary" | None.
    source: str | None = None


class JoinKey(BaseModel):
    left: str
    right: str
