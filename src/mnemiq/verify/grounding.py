from __future__ import annotations

import re

from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved
from mnemiq.verify.verdict import VerifyVerdict

_DQUOTED = re.compile(r'"([^"]+)"')
# single-quoted, but not a possessive ("Sanders's", "Women's") -- require a non-letter before the '
_SQUOTED = re.compile(r"(?<![A-Za-z])'([^']+)'")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _salient(question: str) -> list[str]:
    """Literals a correct query is very likely to reference verbatim: quoted phrases and 4-digit
    years. Deliberately narrow -- we only flag values we are confident should appear verbatim, so a
    'missing' is a real signal. We drop values with '/' (dates get reformatted by the model, e.g.
    '2012/8/23' -> '2012-08-23', so their absence is not evidence of a dropped filter)."""
    vals: list[str] = []
    for pattern in (_DQUOTED, _SQUOTED):
        vals.extend(m.group(1).strip() for m in pattern.finditer(question))
    vals.extend(m.group(0) for m in _YEAR.finditer(question))
    return [v for v in vals if v and "/" not in v]


def grounding_check(packet: ContextPacket, approved: Approved) -> VerifyVerdict | None:
    sql = (approved.plan_sql or "").lower()  # case-insensitive: question "discount" vs SQL 'Discount'
    missing = [v for v in _salient(packet.question) if v.lower() not in sql]
    if missing:
        shown = ", ".join(repr(v) for v in missing)
        return VerifyVerdict(0.3, True, f"The query may not use a value the question specifies: {shown}.", "grounding")
    return None
