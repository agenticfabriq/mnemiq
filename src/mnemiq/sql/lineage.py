"""What an answer read, and whether that record is the whole story.

M56: the trace carried no lineage at all, while `agent/trace.py` had `tables_used` sitting on the
same object the emitter was handed. M43 is why the fix is not simply to send that list: a read
through a function returns rows with `tables=[]`, so a bare list would record "nothing was read"
about a statement that read two SSNs. **Absent reads as *not recorded*; `[]` reads as *nothing was
read*.** The marker is what keeps those apart.

Three-valued for the reason `rls_tables = 0` had to be a third state, and this codebase has now
recorded eleven instances of an absence and a failure sharing one value:

* `COMPLETE`   -- every read was resolved.
* `INCOMPLETE` -- something reaches past what was resolved, and `unresolved` names it.
* `UNKNOWN`    -- we could not establish whether anything does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

COMPLETE = "complete"
INCOMPLETE = "incomplete"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Lineage:
    """The objects an answer read, plus whether that list can be relied on."""

    tables: list[str] = field(default_factory=list)
    completeness: str = UNKNOWN
    unresolved: list[str] = field(default_factory=list)


def lineage_for(ast, tables, views, *, scope_resolved: bool = True) -> Lineage:
    """NOT IMPLEMENTED -- the deliberately naive version, so each test fails on its own
    assertion rather than on an import error."""
    return Lineage(tables=list(tables), completeness=COMPLETE, unresolved=[])
