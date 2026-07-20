from __future__ import annotations

import re

from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved
from mnemiq.verify.verdict import VerifyVerdict

_QUOTED = re.compile(r"'([^']+)'|\"([^\"]+)\"")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _salient(question: str) -> list[str]:
    """Literals a correct query is very likely to reference: quoted phrases and 4-digit years.
    Deliberately narrow -- we only flag values we are confident the SQL should contain, so a
    'missing' is a real signal, not a guess."""
    vals: list[str] = []
    for m in _QUOTED.finditer(question):
        text = (m.group(1) or m.group(2) or "").strip()
        if text:
            vals.append(text)
    vals.extend(m.group(0) for m in _YEAR.finditer(question))
    return vals


def grounding_check(packet: ContextPacket, approved: Approved) -> VerifyVerdict | None:
    sql = approved.plan_sql or ""
    missing = [v for v in _salient(packet.question) if v not in sql]
    if missing:
        shown = ", ".join(repr(v) for v in missing)
        return VerifyVerdict(0.3, True, f"The query may not use a value the question specifies: {shown}.", "grounding")
    return None
