from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

# The pii_level vocabulary is owned by the contract, beside the field it validates; imported
# here (and re-exported for prompts.py) so the enrichment prompt and the authz clearance check
# read one set (M40).
from mnemiq.contract import PII_LEVELS

SEMANTIC_TYPES: tuple[str, ...] = (
    "identifier",
    "code",
    "name",
    "address",
    "email",
    "phone",
    "date",
    "timestamp",
    "amount",
    "count",
    "ratio",
    "boolean",
    "free_text",
    "other",
)

_MAX_DESCRIPTION = 400
_MAX_MEANING = 200

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ColumnAnnotation(BaseModel):
    name: str
    description: str | None = None
    semantic_type: str | None = None
    pii_level: str | None = None
    code_meanings: dict[str, str] = Field(default_factory=dict)


class TableAnnotation(BaseModel):
    table: str
    columns: list[ColumnAnnotation] = Field(default_factory=list)


def _extract_json(raw: str) -> dict:
    """Models fence their JSON and chat around it. Take the outermost object, or nothing."""
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None  # null means "I don't know" -- keep it that way, never stringify it
    text = " ".join(value.split())[:limit].strip()
    return text or None


def _in_vocabulary(value: object, vocabulary: tuple[str, ...]) -> str | None:
    return value if isinstance(value, str) and value in vocabulary else None


def parse_annotation(raw: str, table: str, allowed: dict[str, set[str]]) -> TableAnnotation:
    """Screen a model proposal into an annotation. Never raises; drops what it cannot verify.

    `allowed` maps each column we asked about to the codes we actually observed. The model
    may only annotate those columns, and may only explain those codes -- so a hallucination
    (or a successful prompt injection) is inert by construction rather than by good behavior.
    """
    payload = _extract_json(raw)
    proposed = payload.get("columns")
    if not isinstance(proposed, list):
        return TableAnnotation(table=table)

    columns: list[ColumnAnnotation] = []
    seen: set[str] = set()
    for item in proposed:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name not in allowed or name in seen:
            continue  # a column we never asked about does not exist
        seen.add(name)

        raw_meanings = item.get("code_meanings")
        meanings: dict[str, str] = {}
        if isinstance(raw_meanings, dict):
            for code, meaning in raw_meanings.items():
                if code not in allowed[name]:
                    continue  # a code we never observed is not a code
                text = _clean_text(meaning, _MAX_MEANING)
                if text is not None:
                    meanings[code] = text

        columns.append(
            ColumnAnnotation(
                name=name,
                description=_clean_text(item.get("description"), _MAX_DESCRIPTION),
                semantic_type=_in_vocabulary(item.get("semantic_type"), SEMANTIC_TYPES),
                pii_level=_in_vocabulary(item.get("pii_level"), PII_LEVELS),
                code_meanings=meanings,
            )
        )
    return TableAnnotation(table=table, columns=columns)
