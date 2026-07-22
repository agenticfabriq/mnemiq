from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError


class ColumnEntry(BaseModel):
    description: str | None = None
    codes: dict[str, str] = Field(default_factory=dict)


class DataDictionary(BaseModel):
    columns: dict[str, ColumnEntry] = Field(default_factory=dict)


def load_dictionary(path: str) -> DataDictionary:
    """Parse an operator data dictionary (JSON). Raises ValueError on malformed input."""
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return DataDictionary.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, OSError) as exc:
        raise ValueError(f"invalid data dictionary at {path}: {exc}") from exc
