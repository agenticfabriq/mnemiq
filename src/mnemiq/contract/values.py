from pydantic import BaseModel


class MeasureExpr(BaseModel):
    expr: str
    source: str


class CodedValue(BaseModel):
    code: str
    meaning: str


class JoinKey(BaseModel):
    left: str
    right: str
