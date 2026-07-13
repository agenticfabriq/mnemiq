from __future__ import annotations

from dataclasses import dataclass, field

from mnemiq.adapters.base import SourceAdapter

# Naming convention for identifier columns in the source catalog. Key columns are
# join candidates, never coded vocabularies -- "policy_identifier = 1" has no meaning.
KEY_SUFFIXES = ("_identifier", "_id")


def is_key_like(column: str) -> bool:
    return column.endswith(KEY_SUFFIXES)


# Columns whose *values* are personal data. Their contents must never be harvested into the
# snapshot or rendered into a prompt: in a small sample a name column looks like a coded
# vocabulary, and we would copy real people into a searchable artifact and ship them to a
# model. Names are also entity labels, not controlled vocabularies -- there is nothing to
# learn from the value set even when it is safe.
#
# This is the deterministic first line: it fires before any LLM sees anything. The model's
# own pii/phi classification is the second line, for columns whose names give nothing away.
SENSITIVE_TOKENS = (
    "email",
    "phone",
    "mobile",
    "fax",
    "address",
    "street",
    "postal",
    "zipcode",
    "birth",
    "dob",
    "ssn",
    "social_security",
    "tax_id",
    "passport",
    "license",
    "national_id",
    "account_number",
    "credit_card",
    "iban",
    "gender",
    "ethnicity",
    "salary",
    "location",
)


def is_sensitive_name(column: str) -> bool:
    lowered = column.lower()
    if lowered == "name" or lowered.endswith("_name"):
        return True  # a name is an entity label, and often a person
    return any(token in lowered for token in SENSITIVE_TOKENS)


@dataclass
class ColumnInfo:
    name: str
    data_type: str


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)


def introspect(adapter: SourceAdapter) -> list[TableInfo]:
    tables: dict[str, TableInfo] = {}
    for table, column, dtype in adapter.list_columns():
        tables.setdefault(table, TableInfo(table)).columns.append(ColumnInfo(column, dtype))
    return list(tables.values())
