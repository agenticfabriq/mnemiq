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
    """The objects an answer read, plus whether that list can be relied on.

    `unresolved` and `reasons` are separate because they are different NAMESPACES. The first
    holds object ids and function names -- things in the caller's world. The second holds this
    engine's own reason codes. Mixed into one array they ship to the audit store
    indistinguishable, so a consumer rendering objects would show `scope-unresolved` as a table
    and one filtering for reason codes would match a real object named after one.
    """

    tables: list[str] = field(default_factory=list)
    completeness: str = UNKNOWN
    unresolved: list[str] = field(default_factory=list)  # object ids and function names
    reasons: list[str] = field(default_factory=list)  # this engine's codes, never a caller's name


# Functions this engine treats as reaching nothing past their arguments, for the small set that
# sqlglot leaves `Anonymous`. Deliberately short: an unlisted function is UNRESOLVED, so the list
# growing is a decision someone makes rather than a default. A same-named UDF shadows an entry
# here, which is the same registry problem the strict xfail pins from the other side.
_PURE = frozenset({"now", "age", "current_timestamp", "current_date", "current_time", "today"})


def _safe_label(label: str, fallback: str) -> str:
    """An object id, or a generic token. Applied to EVERY label, including the fallback.

    `unresolved` ships in the emitter's ALWAYS tier, which is on by default and is not behind the
    text opt-in, so a label is an identifier or it does not go. Three spellings have now carried a
    literal through this one field -- a rendered qualified call, its field-access sibling, and a
    QUOTED FUNCTION NAME, because sqlglot strips the delimiters when populating `node.name` and
    the previous guard sanitised the composed label while falling back to that same unchecked
    name. Each was found by someone thinking of a shape.

    So this is the single place a label is admitted, and it fails closed: anything holding a
    bracket or a quote is not an identifier, and if the fallback is no better the value is
    replaced by a token rather than trusted. An auditor learns that a function could not be
    accounted for; they do not learn what was inside it.
    """
    for candidate in (label, fallback):
        if candidate and not any(ch in candidate for ch in "('\"`"):
            return candidate
    return "unnameable-function"


def _unclassified_functions(ast) -> list[str]:
    """Function calls whose reach this engine cannot rule out, in one AST.

    Reused for the caller's statement AND for a view's body: a view defined `SELECT all_ssns()`
    reads through a function, and checking its body only for base TABLES reported the view as
    fully accounted for. The reach is one level down, not absent.

    `_PURE` applies only to an UNQUALIFIED call. `public.now()` parses to an `Anonymous` whose
    `name` is the bare leaf `now`, so matching on the leaf let a schema-qualified callable inherit
    a builtin's exemption -- a UDF named `now` in any schema. A qualified call has an `exp.Dot`
    parent, which is the structural form of "this is not the builtin you whitelisted".
    """
    from sqlglot import exp

    out: list[str] = []
    for node in ast.find_all(exp.Anonymous):
        name = (node.name or "").lower()
        if not name:
            continue
        # Qualified means `qualifier.func()` -- the call is the Dot's EXPRESSION. Testing only
        # `isinstance(parent, exp.Dot)` also matched `func(...).field`, where the call is the
        # Dot's `this`, and rendering that qualifier rendered the call again: the first fix
        # stripped arguments from one spelling and left them in its sibling. It also read
        # `now().y` as qualified, so a whitelisted builtin became unresolved.
        parent = node.parent
        qualified = isinstance(parent, exp.Dot) and parent.expression is node
        if not qualified and name in _PURE:
            continue

        label = name
        if qualified:
            qualifier = parent.this
            # An identifier-shaped qualifier only. Anything else is not a schema name.
            if isinstance(qualifier, exp.Column | exp.Identifier):
                label = f"{qualifier.sql()}.{name}".lower()

        label = _safe_label(label, name)
        if label not in out:
            out.append(label)
    return out


def lineage_for(ast, tables, views, *, scope_resolved: bool = True) -> Lineage:
    """The objects an answer read, and whether that list is the whole story.

    Order matters only in that UNKNOWN is not allowed to mask a demonstrable INCOMPLETE: a
    statement reading a view AND calling an unclassifiable function is reported INCOMPLETE, because
    naming the view is more use to an auditor than recording that something was unclear.
    """

    from mnemiq.sql.qualify import object_key
    from mnemiq.sql.scope import base_tables
    from mnemiq.sql.views import body_of, spellings, unrecognised_source

    tables = list(tables)
    # Two lists, because the two states are decided by different evidence: a view whose body we
    # READ and whose reach we can point at, versus a function whose body we cannot see at all.
    reaching_views: list[str] = []
    unclassified: list[str] = []
    reasons: list[str] = []
    # The two comparisons here
    # have opposite polarity, which is the trap: widening the VIEW lookup adds matches and every
    # added match is an INCOMPLETE, so `spellings` is safe there for the reason `views.py` gives
    # ("every added match is a refusal"). Widening the REACH comparison adds matches and every
    # added match is a claim of COMPLETENESS. Using one helper for both dropped every qualifier on
    # both sides, so a view reaching `pg.claim` counted as accounted-for against a list naming
    # `mysql.claim` -- the audit record affirmatively asserting a read was resolved when it was
    # not, which is `object_key`'s own documented hazard re-done inside the artifact meant to
    # prevent exactly that assertion.
    # Two indexes, because quoting MAY decide identity and this function cannot tell whether it
    # does. In Postgres a quoted identifier is case-sensitive, so `"Claim"` and `claim` are two
    # objects; DuckDB folds them to one, measured in this repo's own venv, and `qualify.py` states
    # the rule correctly scoped as "case-sensitive in Postgres". `lineage_for` takes no dialect.
    #
    # So a quoted name that matches only case-insensitively is UNKNOWN, never INCOMPLETE: on one
    # engine it is a different object and on another it is the same one, and INCOMPLETE claims
    # reach was DEMONSTRATED. Folding both -- the previous version -- asserted the wrong one of
    # the two; asserting the other would be equally wrong on the other engine.
    resolved_folded = {t.lower() for t in tables}
    resolved_exact = set(tables)

    # DEMONSTRABLE reach: a view whose body names an object this list does not.
    #
    # Membership alone is not demonstration, and saying so was this module's first bug: a view
    # defined `SELECT 1 AS x` reaches nothing, and `FROM claim_view JOIN claim` may name every
    # base the body reads. The body is in the snapshot, so the engine can actually look -- and
    # the docstring above claimed it did before it did.
    #
    # Spellings via the shared helper, not `name in views`: `check_views` resolves `CLAIM_VIEW`
    # and `public.claim_view` to a keyed `claim_view`, and a marker that did not would report
    # COMPLETE on a statement the guard recognised as reading a view. Two normalisations of one
    # question drift, which is M7.
    view_keys = spellings(set(views))
    for name in tables:
        if not (spellings({name}) & view_keys):
            continue
        # Every spelling `spellings` generates, not three of the four. The hand-rolled version
        # omitted `bare.lower()`, so `public.CLAIM_VIEW` matched the membership test and then
        # failed to retrieve -- landing in the cannot-parse branch and reporting UNKNOWN where
        # the other three spellings report INCOMPLETE. Conservative in outcome and still exactly
        # the drift this was meant to close: two ways of asking one question.
        view = next((views[k] for k in spellings({name}) if k in views), None)
        parsed = body_of(view) if view is not None else None
        if parsed is None:
            unclassified.append(name)  # a view we cannot PARSE: unclear, not demonstrated
            continue
        # The source-shape whitelist `views.py` already applies, reused rather than re-derived.
        # `base_tables` can return FEWER tables for a shape the resolver does not model while
        # scope construction still reports success -- a Postgres view over `LATERAL (VALUES
        # ((SELECT max(store_id) FROM customer)))` yielded COMPLETE with `customer` unaccounted.
        # An unmodelled source is exactly the case where a table list cannot be trusted, and
        # `views.py` refuses on it for the same reason.
        if unrecognised_source(parsed) is not None:
            label = f"unmodelled-source:{unrecognised_source(parsed)}".lower()
            if label not in unclassified:
                unclassified.append(label)
            continue
        # `base_tables`, not `find_all`: a CTE alias inside the body is not a real read, and
        # counting one reported INCOMPLETE for a view whose only true base was already in the
        # list. That is M31/M49's lesson -- the scope-aware resolver exists so that three guards
        # stopped asking "is this name a real table" three different ways -- applied one consumer
        # later, in a function that had reached for `find_all` anyway.
        gap = False
        for node in base_tables(parsed):
            key = object_key(node)
            if not key:
                # A table-valued function -- `generate_series`, `unnest`, `range`. `object_key`
                # returns "" because there is no identifier to key, and treating that as an
                # unaccounted OBJECT reported INCOMPLETE for a view that reaches no object at
                # all. It is not demonstrated reach; it is reach we cannot classify.
                label = f"table-function:{node.sql().split('(')[0].lower()}"
                if label not in unclassified:
                    unclassified.append(label)
                continue
            # Any segment quoted, not just the leaf: `object_key` composes catalog/db/name, so a
            # quoted mixed-case QUALIFIER (`"Public".claim`) decides identity exactly as a quoted
            # leaf does, and inspecting `node.this` alone still folded it.
            quoted = any(bool(getattr(part, "quoted", False)) for part in node.parts)
            if key in resolved_exact:
                continue  # matched exactly: accounted for under either engine's rule
            if key.lower() in resolved_folded:
                if quoted:
                    ambiguous = f"case-ambiguous:{key}"
                    if ambiguous not in unclassified:  # dedup, like every sibling append
                        unclassified.append(ambiguous)
                continue
            gap = True
        # A function inside the body reaches where the body's table list cannot show.
        for fn in _unclassified_functions(parsed):
            if fn not in unclassified:
                unclassified.append(fn)
        if gap:
            reaching_views.append(name)

    # UNCLASSIFIABLE reach: a function whose body this engine cannot see.
    for fn in _unclassified_functions(ast):
        if fn not in unclassified:
            unclassified.append(fn)

    if not getattr(views, "available", True):
        reasons.append("view-inventory-unavailable")
    elif not getattr(views, "asked", False):
        # Default FALSE, not True: a caller that passed a bare mapping has established nothing
        # about views, and an audit record must not read "nobody asked" as "asked and found none".
        reasons.append("view-inventory-never-asked")
    if not scope_resolved:
        reasons.append("scope-unresolved")

    unresolved = reaching_views + unclassified
    if reaching_views:
        # A gap the engine can point at outranks one it merely suspects: naming the view is more
        # use to an auditor than recording that something was unclear. The unclear reach is still
        # carried, not dropped.
        return Lineage(tables=tables, completeness=INCOMPLETE, unresolved=unresolved,
                       reasons=reasons)
    if unresolved or reasons:
        return Lineage(tables=tables, completeness=UNKNOWN, unresolved=unresolved, reasons=reasons)
    return Lineage(tables=tables, completeness=COMPLETE)
