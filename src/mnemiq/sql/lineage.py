"""What an answer read, and whether that record is the whole story.

M56: the trace carried no lineage at all, while `agent/trace.py` had `tables_used` sitting on the
same object the emitter was handed. M43 is why the fix is not simply to send that list: a read
through a function returns rows with `tables=[]`, so a bare list would record "nothing was read"
about a statement that read two SSNs. **Absent reads as *not recorded*; `[]` reads as *nothing was
read*.** The marker is what keeps those apart.

Three-valued for the reason `rls_tables = 0` had to be a third state, and this codebase has now
recorded eleven instances of an absence and a failure sharing one value:

* `COMPLETE`   -- every read was resolved.
* `INCOMPLETE` -- reach beyond the list is DEMONSTRABLE. A view is the case: its body is in the
  snapshot, so the engine can see that it reads tables the list does not name.
* `UNKNOWN`    -- completeness could not be established. A function is the case, and the
  distinction from INCOMPLETE is the point: the engine does not know that `all_ssns()` reads
  anything, only that it cannot rule it out.

**Functions are UNKNOWN rather than INCOMPLETE, and that is measured rather than chosen.** sqlglot
types a function from a NAME registry, so `now()` and `age()` land in `Anonymous` beside a genuine
UDF while `coalesce`, `round`, `substr`, `md5` and a dozen others parse to typed nodes. The engine
therefore cannot tell a pure builtin from a table-reading UDF, in either direction. `_PURE`
below is a whitelist for the false-positive half -- an unlisted function is unresolved, which is
the pattern `views.py` already uses because an unlisted shape must not pass -- and the
false-negative half, a UDF named after a typed builtin, is pinned as a strict xfail. Both dissolve
when a function inventory exists, which is the same shape as the view inventory and v2.
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


# Functions this engine treats as reaching nothing past their arguments, for the small set that
# sqlglot leaves `Anonymous`. Deliberately short: an unlisted function is UNRESOLVED, so the list
# growing is a decision someone makes rather than a default. A same-named UDF shadows an entry
# here, which is the same registry problem the strict xfail pins from the other side.
_PURE = frozenset({"now", "age", "current_timestamp", "current_date", "current_time", "today"})


def lineage_for(ast, tables, views, *, scope_resolved: bool = True) -> Lineage:
    """The objects an answer read, and whether that list is the whole story.

    Order matters only in that UNKNOWN is not allowed to mask a demonstrable INCOMPLETE: a
    statement reading a view AND calling an unclassifiable function is reported INCOMPLETE, because
    naming the view is more use to an auditor than recording that something was unclear.
    """
    from sqlglot import exp

    tables = list(tables)
    unresolved: list[str] = []
    unknown_reasons: list[str] = []

    # Demonstrable reach: a view's body names tables this list does not.
    for name in tables:
        if name in views:
            unresolved.append(name)

    # Unclassifiable reach: a function whose body this engine cannot see.
    for node in ast.find_all(exp.Anonymous):
        name = (node.name or "").lower()
        if name and name not in _PURE and name not in unknown_reasons:
            unknown_reasons.append(name)

    # Unestablished at all: the inventory could not be read, was never asked for, or the table
    # list itself may be wrong because the scope did not resolve.
    if not getattr(views, "available", True):
        unknown_reasons.append("view-inventory-unavailable")
    elif not getattr(views, "asked", False):
        # Default FALSE, not True: a caller that passed a bare mapping has established nothing
        # about views, and an audit record must not read "nobody asked" as "asked and found none".
        unknown_reasons.append("view-inventory-never-asked")
    if not scope_resolved:
        unknown_reasons.append("scope-unresolved")

    if unresolved:
        return Lineage(tables=tables, completeness=INCOMPLETE,
                       unresolved=unresolved + unknown_reasons)
    if unknown_reasons:
        return Lineage(tables=tables, completeness=UNKNOWN, unresolved=unknown_reasons)
    return Lineage(tables=tables, completeness=COMPLETE, unresolved=[])
