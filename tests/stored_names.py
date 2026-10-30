"""A table name spelled as each engine stores it unquoted, for tests that run one statement on several.

The deciders decide a table reference only where the engine reads it as a name the grant holds, as
written (M223): Oracle stores an unquoted `claim` as CLAIM and reads `claim` as CLAIM, so a grant
keyed `claim` there is a quoted lowercase table. A test whose subject is not the table's name writes
it once and spells it per engine with these.
"""
from __future__ import annotations

import re

from mnemiq.sql.identifiers import resolve


def stored(name: str, engine: str | None) -> str:
    """How `engine` stores the unquoted `name`."""
    return resolve(name, quoted=False, dialect=engine)


def as_stored(text: str, engine: str | None, *names: str) -> str:
    """`text` with each of `names`, as a whole word, spelled as `engine` stores it unquoted."""
    for name in names:
        text = re.sub(rf"(?<![\w.\"]){re.escape(name)}(?![\w\"])", stored(name, engine), text)
    return text


def keyed(mapping: dict, engine: str | None, *names: str) -> dict:
    """`mapping` with each key among `names` spelled as `engine` stores it unquoted."""
    return {stored(k, engine) if k in names else k: v for k, v in mapping.items()}


def policy_keyed(policy, engine: str | None, *names: str):
    """An `AccessPolicy` with each table among `names` spelled as `engine` stores it unquoted."""
    from dataclasses import replace

    def table(t: str) -> str:
        return stored(t, engine) if t in names else t

    return replace(policy,
                   denied={(table(t), c) for t, c in policy.denied},
                   masked={(table(t), c) for t, c in policy.masked},
                   row_filters=keyed(policy.row_filters, engine, *names),
                   policy_schema=keyed(policy.policy_schema or {}, engine, *names))
