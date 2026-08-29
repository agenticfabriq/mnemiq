from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.contract import ViewDefinition
from mnemiq.sql.qualify import object_key
from mnemiq.sql.verdict import Refusal, RefusalCode

# A view nested deeper than this is either a cycle the stack missed or a schema nobody should
# be governing by reading definitions. Refusing beats walking forever.
MAX_DEPTH = 8


def body_of(view: ViewDefinition) -> exp.Expression | None:
    """The SELECT a view stands for, whichever way its source spells it.

    Postgres hands back a bare SELECT; DuckDB and SQLite hand back the whole
    `CREATE VIEW x AS SELECT ...`. Normalising through the parser rather than by stripping the
    prefix keeps one SQL implementation -- `str.partition` would be a second, and a worse one.
    """
    try:
        parsed = sqlglot.parse_one(view.definition, read=view.dialect)
    except Exception:
        return None
    if isinstance(parsed, exp.Create):
        parsed = parsed.expression
    return parsed if isinstance(parsed, exp.Query) else None


def _unrecognised_source(body: exp.Expression) -> str | None:
    """The name of a source shape this engine does not model, or None if all are plain.

    A **whitelist**, and that is the whole design. The previous rule enumerated dangerous
    shapes -- a table-valued function, then laterals -- and each round of review found another
    it had not thought of, because an unlisted shape PASSED. Here an unlisted shape refuses:
    every source must be a named table or a subquery over one, and anything else (LATERAL,
    UNNEST, a function call, a table literal, whatever a future dialect adds) lands in the
    refuse branch without anyone having to notice it first.

    That inversion is the same one `writes_enabled` took in M3. Forgetting must fail closed.
    """
    # Found by node type, not by args key. The first version read `args["from"]`, which this
    # sqlglot spells `from_`, so it enumerated NOTHING -- and a whitelist that finds no sources
    # approves everything. It was failing open in exactly the way it exists to prevent, and only
    # the function tests caught it. Node types do not get renamed out from under a lookup.
    # FIRST, and position-independent: a table node whose `.this` is not an Identifier is a
    # function wearing a table's clothes, wherever it sits. Checking source POSITIONS caught
    # `Subquery(Table)` and `Subquery(Pivot)` and still missed `((query_table('customer')))`,
    # because a Subquery inside a Subquery satisfies the Query test while its root is in no
    # From or Join slot. Containers nest arbitrarily; the node type does not move.
    for table in body.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return type(table.this).__name__

    # SECOND, by position, for sources that are not Table nodes at all -- LATERAL, UNNEST,
    # VALUES. The two checks are independent on purpose: neither subsumes the other.
    sources = [node.this for node in body.find_all(exp.From)]
    sources += [join.this for join in body.find_all(exp.Join)]
    # Two conditions, and the second is the one round seven needed. A `Subquery` is allowed on
    # the assumption its inner sources reappear as From/Join nodes -- but sqlglot represents a
    # parenthesised join as `Subquery(Table-with-joins)` and a pivot as `Subquery(Pivot)`, and
    # in both the ROOT source is in neither position. `FROM (query_table('customer') JOIN film)`
    # therefore enumerated only `film`, and the real table name lives inside a string literal
    # where nothing can read it. DuckDB executes both shapes.
    #
    # A table source is checked on `.this` being an Identifier rather than on having a name: a
    # function call parses as `Table(Anonymous)`, which is what an empty name was standing in
    # for, and the type is the fact while the empty name was a symptom of it.
    for source in sources:
        if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier):
            continue
        if isinstance(source, exp.Subquery) and isinstance(source.this, exp.Query):
            continue  # a real subquery: its own FROM/JOIN nodes are enumerated by this walk
        return type(source).__name__
    return None


def spellings(names: set[str]) -> set[str]:
    """Every spelling a name might be written in: itself, folded, and its bare last segment.

    One function for both comparisons -- what a body MENTIONS and what the source KNOWS -- 
    because when only the filtered set was widened, the known-object check refused every
    qualified reference it should have accepted. Two sets compared against each other must be
    normalised the same way or the comparison is between two different vocabularies.
    """
    out: set[str] = set()
    for name in names:
        bare = name.rsplit(".", 1)[-1]
        out |= {name, name.lower(), bare, bare.lower()}
    return out


def _mentions(body: exp.Expression) -> set[str]:
    """Every table name the body mentions, in both spellings, from the RAW tree.

    Deliberately not `base_tables`. That resolves scopes to tell a CTE reference from a base
    table, which is right for authorization and wrong here: scope resolution can only ever
    return FEWER names, and it dropped the read inside a lateral. Over-inclusion costs a
    spurious refusal on a CTE that shares a filtered table's name; under-inclusion cost the
    control. Both spellings because a body may say `public.customer` where the snapshot's
    object-id is `customer`.
    """
    tables = list(body.find_all(exp.Table))
    names = {object_key(t) for t in tables} | {t.name for t in tables if t.name}
    # Folded and bare-segmented, because unquoted identifiers are case-insensitive in all three
    # engines and a body may qualify what the snapshot keys bare. Widening here can only ADD
    # matches, and every added match is a refusal.
    return spellings(names)


class ViewInventory(dict):
    """The views a source reports, plus whether the source could be ASKED.

    A plain `{}` asserts "this source has no views". A failed discovery produces the SAME empty
    mapping and means "we do not know what views exist" -- and the governance layer must not read
    the second as the first. Measured before this existed: with `row_filters={'claim'}` and a view
    `claim_v` over it, `check_views` REFUSED `ungoverned_view` when discovery succeeded and
    APPROVED when discovery failed, from the identical call.

    The producer already knew. `enrichment/pipeline.py` records `discover:views` with
    `status='failed'` under a comment saying an empty list "must be treated as 'cannot reason
    about', never as 'there are none'". Nothing read it -- the tenth instance in this codebase of
    an absence and a failure sharing one value.

    Modelled on `authz/grants.py`'s EMPTY and UNAVAILABLE, which deny identically and exist so the
    engine can say WHY. Kept as a dict subclass so every existing caller that passes a plain
    mapping keeps meaning what it meant: a bare dict asserts it is complete, and only a caller
    that knows otherwise says so.
    """

    def __init__(self, mapping=None, available: bool = True, asked: bool = True) -> None:
        super().__init__(mapping or {})
        self.available = available
        # Whether the source was ASKED at all, which `available` cannot carry: a snapshot with a
        # `done` job holding no views and a snapshot with no job at all are both available with an
        # empty mapping, and nothing could tell them apart. `check_views` deliberately treats both
        # as answerable -- refusing a query because a hand-built snapshot lacks a job would be
        # harsh. An AUDIT RECORD is stricter, because saying "cannot confirm" costs nothing where
        # refusing costs an answer. Same fact, different response, so it needs its own field.
        self.asked = asked


# Denies exactly as much as it must, and says why. Distinct from `ViewInventory({})`, which is a
# source that answered and reported no views.
VIEWS_UNAVAILABLE = ViewInventory(available=False, asked=False)


def inventory_for(snapshot) -> ViewInventory:
    """The view inventory a snapshot supports, and whether it could be built at all.

    ONE place answers this, because two would drift: the read path builds its inventory in
    `plan_query` and the write path in `Runtime.write`, and M46 is what happens when the two
    doors disagree about a view.

    No snapshot at all is UNAVAILABLE, not empty. `Runtime` substitutes `{}` there, which reads
    as "this source has no views" -- an engine holding no schema cannot assert that.

    A snapshot whose `discover:views` job FAILED is unavailable: the source was asked and did not
    answer. `enrichment/pipeline.py` has recorded that status all along, under a comment saying an
    empty list "must be treated as 'cannot reason about', never as 'there are none'". Nothing read
    it until now.

    A snapshot carrying no `discover:views` job at all is still treated as AVAILABLE, and now
    also records `asked=False` so a consumer that wants the stricter reading can have it. The producer always emits one, so absence means a hand-built snapshot rather than a
    failed read, and the stricter rule -- demand a `done` job -- would refuse every such snapshot
    on a question about provenance rather than about views. Named here so the choice is visible
    instead of implicit; a test pins it.
    """
    if snapshot is None:
        return VIEWS_UNAVAILABLE
    failed = any(
        getattr(j, "id", None) == "discover:views" and getattr(j, "status", None) == "failed"
        for j in getattr(snapshot, "jobs", ()) or ()
    )
    asked = any(
        getattr(j, "id", None) == "discover:views" for j in getattr(snapshot, "jobs", ()) or ()
    )
    return ViewInventory({v.object_id: v for v in snapshot.views}, available=not failed,
                         asked=asked)


def check_views(
    ast: exp.Expression,
    views: dict[str, ViewDefinition],
    filtered: set[str],
    known: set[str] | None = None,
) -> Refusal | None:
    """Refuse a granted view that reads a row-filtered table, and one this engine cannot read.

    M27: a view's rows are defined by SQL held in the source, so a filter on its base table
    reaches nothing -- the decider refused the base table correctly, and the repair loop used
    that refusal as a signpost to the same rows through the view.

    **This is the floor, not the fix.** The fix is to inline the body so the filter lands at
    the leaves, and it was built, merged, and taken back out. Applying a policy *through* a
    view needs column-level lineage, and a lineage model that does not understand `SELECT *`,
    `UNION`, aggregates or transitive renames fails OPEN on every shape it misses -- four
    review rounds found bypasses in ordinary view definitions, the last of them in
    `SELECT * FROM base`.

    The floor asks a question with no unmodelled shapes: *which tables does this body mention*.
    `find_all` answers that for every spelling of SQL -- a star mentions its tables, a union
    mentions both sides' -- and a body that will not parse is refused outright. There is no
    shape that yields FEWER tables than the body reads, so every way this can be wrong points
    at refusing.

    Only row filters trigger it. Column dispositions need not, because enrichment classifies a
    view's own columns independently -- `customer_list.phone` carries its own `pii_level` -- so
    CLS already governs them at the view. Including them was measured on Pagila: all seven
    views refuse for all three roles, including one with no row filters at all.
    """
    if not filtered:
        # No row filters: there is nothing a view could carry the caller around, so an unknown
        # inventory costs nothing either. This early-out is what keeps the refusal below narrow.
        return None
    if not getattr(views, "available", True):
        return Refusal(
            code=RefusalCode.VIEW_INVENTORY_UNAVAILABLE,
            message=(
                "This source could not report its views, so this engine cannot confirm that "
                "nothing in this query reads around a row filter. Retry once the source is "
                "reachable; this is not a limit on your grants."
            ),
        )
    return _walk(ast, views, filtered, known or set(), (), 0, "")


def _walk(
    ast: exp.Expression,
    views: dict[str, ViewDefinition],
    filtered: set[str],
    known: set[str],
    stack: tuple[str, ...],
    depth: int,
    catalog: str,
) -> Refusal | None:
    for node in ast.find_all(exp.Table):
        name = object_key(node)
        view = views.get(name) or views.get(node.name)
        if view is not None and name not in views:
            name = node.name  # a body may qualify a view the snapshot keys bare
        if view is None and catalog:
            # A FEDERATED view body is written in the SOURCE's own naming, so a NESTED view
            # reads `claim_v` where the merged inventory keys it `pg.claim_v`, and the lookup
            # above misses. Measured: `pg.claim_v2` over `pg.claim_v` over a filtered
            # `pg.claim` was APPROVED while the byte-identical single-source shape refused.
            # Resolved in the ENCLOSING view's catalog only, never globally, so two catalogs
            # holding a view of the same name cannot resolve to each other's.
            for candidate in (f"{catalog}.{object_key(node)}", f"{catalog}.{node.name}"):
                if candidate in views:
                    view, name = views[candidate], candidate
                    break
        if view is None:
            # Case-insensitive fallback, tried LAST so an exact match always wins. Unquoted
            # identifiers are case-insensitive in all three engines -- `_spellings` two screens
            # down folds for exactly this reason -- but this lookup did not. Measured: a body
            # writing `FROM CLAIM_V` against an inventory keyed `claim_v` missed, the nested view
            # was never walked, and its row-filtered base was never reached. APPROVED on the
            # single-source path as well as the federated one, so this is not federation's bug.
            #
            # `sorted` so a source holding two keys differing only in case resolves the same way
            # every run. Such a source cannot exist unambiguously under those same engine rules,
            # but a deterministic wrong answer is debuggable and a arbitrary one is not.
            # `.lower()` on the WHOLE candidate, catalog included. Lowercasing only the table
            # half left the prefix in its original case while the comparison below folds the real
            # key in full, so any catalog alias carrying an uppercase letter made this fallback
            # dead code. `SourceSpec.catalog` is a free-form DuckDB attach alias and nothing in
            # `merge_snapshots` or `qualify_object_id` normalises it. Measured: catalog `pg`
            # refused and catalog `PG` approved, on the identical statement.
            wanted = {object_key(node).lower(), node.name.lower()}
            if catalog:
                wanted |= {f"{catalog}.{w}".lower() for w in tuple(wanted)}
            for key in sorted(views):
                if key.lower() in wanted:
                    view, name = views[key], key
                    break
        if view is None:
            continue
        if name in stack:
            return Refusal(
                code=RefusalCode.UNRESOLVABLE_VIEW,
                message=(
                    f"The view {name!r} is defined in terms of itself "
                    f"({' -> '.join((*stack, name))}), so it cannot be resolved."
                ),
                subject=name,
            )
        if depth >= MAX_DEPTH:
            return Refusal(
                code=RefusalCode.UNRESOLVABLE_VIEW,
                message=f"The view {name!r} nests deeper than this engine will resolve.",
                subject=name,
            )
        body = body_of(view)
        if body is None:
            return Refusal(
                code=RefusalCode.UNRESOLVABLE_VIEW,
                message=(
                    f"{name!r} is a view whose definition this engine cannot read, so it "
                    "cannot confirm the row policy on the tables behind it."
                ),
                subject=name,
            )
        unrecognised = _unrecognised_source(body)
        if unrecognised is not None:
            return Refusal(
                code=RefusalCode.UNRESOLVABLE_VIEW,
                message=(
                    f"{name!r} reads through a {unrecognised.lower()} rather than a named "
                    "table, so this engine cannot tell which tables it touches."
                ),
                subject=name,
            )
        # The filtered set is widened the same way and by its bare last segment, so a body
        # saying `customer` still matches a filter keyed `public.customer`. Both directions,
        # because a miss here means not refusing.
        wanted = {f for f in filtered} | {f.lower() for f in filtered} \
            | {f.rsplit(".", 1)[-1] for f in filtered} \
            | {f.rsplit(".", 1)[-1].lower() for f in filtered}
        # A body may only read objects the snapshot knows. A caller writing
        # `SELECT a FROM 'customer.csv'` is already refused UNAUTHORIZED_TABLE because the name
        # is not in `visible`; a view body was never held to that, so a view could reach a file
        # the caller could not. That is M27's shape -- the view reaching past the caller's own
        # boundary -- and it closes every unknown-object spelling at once rather than the file
        # literal specifically.
        if known:
            # A CTE alias is a name the BODY defines; the source is not expected to know it.
            # Counting it as an unknown object refused every ordinary CTE view -- five of eight
            # new refusals in the review's matrix were legitimate SQL. This weakens nothing:
            # `_mentions` still collects the alias, so a CTE named after a filtered table still
            # trips the filtered check below.
            local = {cte.alias_or_name for cte in body.find_all(exp.CTE)}
            recognised = spellings(known) | spellings(set(views)) | spellings(local)
            # BOTH sides normalised. Widening only the known set still rejected `public.film`
            # against a snapshot that keys it `film`: a name is known when ANY of its spellings
            # matches any recognised one, not when its exact text appears.
            unknown = sorted(
                object_key(x)
                for x in body.find_all(exp.Table)
                if not (spellings({object_key(x)}) & recognised)
            )
            if unknown:
                return Refusal(
                    code=RefusalCode.UNRESOLVABLE_VIEW,
                    message=(
                        f"{name!r} reads {unknown[0]!r}, which is not an object in this source, "
                        "so this engine cannot tell what policy applies to it."
                    ),
                    subject=name,
                )
        reached = sorted(_mentions(body) & wanted)
        if reached:
            return Refusal(
                code=RefusalCode.UNGOVERNED_VIEW,
                message=(
                    f"{name!r} reads {reached[0]!r}, which is row-filtered for you, and this "
                    "engine cannot apply that filter through a view. Query the table directly."
                ),
                subject=name,
            )
        # The body's own catalog carries into it: a view two levels down is still written in
        # the source's naming, not the federation's.
        nested = _walk(body, views, filtered, known, (*stack, name), depth + 1,
                       name.rsplit(".", 1)[0] if "." in name else catalog)
        if nested is not None:
            return nested
    return None
