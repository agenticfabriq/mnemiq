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

import re
from dataclasses import dataclass, field

# What an unquoted SQL identifier may look like, applied per dot-separated segment. Deliberately
# narrow: a name that needed quoting to be legal is a name this label cannot carry safely.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_$#]*")

# The classifications this engine may prefix a label with. A CLOSED set, because anything else in
# front of a colon is part of a caller-derived name and must face the grammar.
_ENGINE_PREFIXES = frozenset({"unconfirmed-identity", "table-function", "unmodelled-source"})

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

    A GRAMMAR, not a blocklist. The first version rejected five characters, which let
    `"DOB 1990-01-01"()` through: sqlglot strips the delimiters, so the name arrives as
    `DOB 1990-01-01` -- spaces and hyphens, no bracket, no quote. Blocklisting is the pattern
    `views.py` documents as unfixable ("each round of review found another it had not thought
    of, because an unlisted shape PASSED"), which I quoted approvingly two commits before
    writing one. So a label must MATCH what an identifier is, segment by segment, and anything
    else becomes a token. An auditor learns that a function could not be accounted for; they do
    not learn what was inside it.
    """
    for candidate in (label, fallback):
        if candidate and all(_IDENTIFIER.fullmatch(part) for part in candidate.split(".")):
            return candidate
    return "unnameable-function"


def _add(out: list[str], value: str) -> None:
    """The ONE way a value reaches `unresolved`, deduplicated and grammar-checked.

    `_safe_label` was introduced as "the single admission point" and then bypassed three times in
    the same function: `case-ambiguous:` interpolated an `object_key` straight in, a view name was
    appended raw on parse failure, and a demonstrated gap appended a raw key. So a view named
    `"DOB 1990-01-01"` put that string into the emitter's ALWAYS tier, which is on by default and
    not behind the text opt-in -- the fourth disclosure of this shape, through a path created
    while closing the third.

    A prefixed label keeps its prefix (an engine code) and grammar-checks only the subject, so a
    caller-controlled name can never ride in on a classification.
    """
    prefix, sep, subject = value.partition(":")
    if sep and prefix in _ENGINE_PREFIXES:
        admitted = f"{prefix}:{_safe_label(subject, subject)}"
    else:
        # Not a known classification, so the WHOLE value is caller-derived and gets the grammar.
        # Splitting on the first colon and trusting whatever preceded it let a name containing a
        # colon ride in intact -- `DOB 1990-01-01:x` was admitted whole. The test could not catch
        # it either, because it partitioned on ":" exactly as the code did and asserted only on
        # the part after, which is the half a colon-bearing name does not land in. A test that
        # shares the code's assumption cannot falsify it.
        admitted = _safe_label(value, value)
    if admitted not in out:
        out.append(admitted)


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
            _add(unclassified, name)  # a view we cannot PARSE: unclear, not demonstrated
            continue
        # The source-shape whitelist `views.py` already applies, reused rather than re-derived.
        # `base_tables` can return FEWER tables for a shape the resolver does not model while
        # scope construction still reports success -- a Postgres view over `LATERAL (VALUES
        # ((SELECT max(store_id) FROM customer)))` yielded COMPLETE with `customer` unaccounted.
        # An unmodelled source is exactly the case where a table list cannot be trusted, and
        # `views.py` refuses on it for the same reason.
        # Recorded WITHOUT short-circuiting. The `continue` here stopped the body scan, so a view
        # mixing a modelled source with an unmodelled one downgraded a DEMONSTRATED gap to
        # UNKNOWN -- `SELECT customer.id FROM customer CROSS JOIN LATERAL (VALUES (1))` hid
        # `customer` behind `unmodelled-source:lateral`. That inverts this module's own rule that
        # a gap the engine can point at outranks one it merely suspects.
        reaches_past_the_list = False
        shape = unrecognised_source(parsed)
        if shape is not None:
            _add(unclassified, f"unmodelled-source:{shape.lower()}")
        # `base_tables`, not `find_all`: a CTE alias inside the body is not a real read, and
        # counting one reported INCOMPLETE for a view whose only true base was already in the
        # list. That is M31/M49's lesson -- the scope-aware resolver exists so that three guards
        # stopped asking "is this name a real table" three different ways -- applied one consumer
        # later, in a function that had reached for `find_all` anyway.
        # A lexical match is NOT identity, and this is the class four review passes kept finding
        # one shape at a time: a view `a.v` whose body says `claim` reads `a.claim`, while a
        # caller under schema `b` writing bare `claim` reads `b.claim`. The strings are equal and
        # the objects are not. `ViewDefinition` carries no creation schema, so the binding context
        # a body was written in is not recoverable here -- and without it a match CANNOT be
        # confirmed, only observed.
        #
        # So a view's reach never yields COMPLETE. It yields INCOMPLETE when a body table matches
        # nothing the caller named under any spelling, because that is a gap under every binding
        # context; and UNKNOWN when it matches lexically, because that is where identity would
        # have to be resolved and cannot be. COMPLETE survives only for statements that read no
        # view at all, where every name lives in one context. A narrower marker that is right
        # beats a broader one that certifies a false audit record -- which is the one outcome
        # worse than shipping no marker.
        for node in base_tables(parsed):
            key = object_key(node)
            if not key:
                _add(unclassified, f"table-function:{node.sql().split('(')[0].lower()}")
                continue
            if key in resolved_exact or key.lower() in resolved_folded:
                _add(unclassified, f"unconfirmed-identity:{key}")
            else:
                # PER VIEW. A statement-scoped accumulator meant that once any view contributed a
                # gap, every LATER view in the list was named as reaching past the table list --
                # order-dependent, so it would not have reproduced reliably, and the inverse of
                # the false certification this design exists to remove: a clean view certified as
                # a demonstrated gap purely by its position. The previous `gap = False` bool was
                # correct and the rename bought nothing, since the collected keys were only ever
                # read for truthiness.
                reaches_past_the_list = True
        for fn in _unclassified_functions(parsed):
            _add(unclassified, fn)
        if reaches_past_the_list:
            _add(reaching_views, name)

    # The caller's OWN statement gets the whitelist too, not just the bodies it reads through.
    # `SELECT x FROM LATERAL (VALUES ((SELECT max(store_id) FROM customer)))` yields
    # `base_tables == []` while scope resolution reports SUCCESS, so the list is empty and looked
    # settled -- COMPLETE over a read of `customer` the record never named. Applying the check to
    # bodies and not to the statement was the same blind spot one level out.
    shape = unrecognised_source(ast)
    if shape is not None:
        _add(unclassified, f"unmodelled-source:{shape.lower()}")

    # UNCLASSIFIABLE reach: a function whose body this engine cannot see.
    for fn in _unclassified_functions(ast):
        _add(unclassified, fn)

    if not getattr(views, "available", True):
        reasons.append("view-inventory-unavailable")
    elif not getattr(views, "asked", False):
        # Default FALSE, not True: a caller that passed a bare mapping has established nothing
        # about views, and an audit record must not read "nobody asked" as "asked and found none".
        reasons.append("view-inventory-never-asked")
    if not scope_resolved:
        reasons.append("scope-unresolved")

    # Through the same helper as everything else. This comprehension was the third place that
    # bypassed the "single admission point" -- sanitised but not deduplicated, and not against
    # `unclassified` either.
    unresolved: list[str] = []
    for value in reaching_views + unclassified:
        _add(unresolved, value)
    if reaching_views:
        # A gap the engine can point at outranks one it merely suspects: naming the view is more
        # use to an auditor than recording that something was unclear. The unclear reach is still
        # carried, not dropped.
        return Lineage(tables=tables, completeness=INCOMPLETE, unresolved=unresolved,
                       reasons=reasons)
    if unresolved or reasons:
        return Lineage(tables=tables, completeness=UNKNOWN, unresolved=unresolved, reasons=reasons)
    return Lineage(tables=tables, completeness=COMPLETE)
