from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from mnemiq.contract import ViewDefinition
from mnemiq.sql.identifiers import rules_for
from mnemiq.sql.qualify import object_key
from mnemiq.sql.reading import as_duckdb_reads, dotted_map_keys
from mnemiq.sql.scope import cte_binder, cte_references
from mnemiq.sql.verdict import Refusal, RefusalCode

# A view nested deeper than this is either a cycle the stack missed or a schema nobody should
# be governing by reading definitions. Refusing beats walking forever.
MAX_DEPTH = 8

logger = logging.getLogger(__name__)


def body_of(view: ViewDefinition) -> exp.Expression | None:
    """The SELECT a view stands for, whichever way its source spells it.

    Postgres hands back a bare SELECT; DuckDB and SQLite hand back the whole
    `CREATE VIEW x AS SELECT ...`. Normalising through the parser rather than by stripping the
    prefix keeps one SQL implementation -- `str.partition` would be a second, and a worse one.
    """
    if dotted_map_keys(view.definition, view.dialect):
        return None  # the parser reads the key as its last part alone: not a body the checks can read
    try:
        parsed = as_duckdb_reads(sqlglot.parse_one(view.definition, read=view.dialect))
    except Exception:
        return None
    if isinstance(parsed, exp.Create):
        parsed = parsed.expression
    return parsed if isinstance(parsed, exp.Query) else None


def unrecognised_source(body: exp.Expression) -> str | None:
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
        # A LATERAL over a real subquery is that subquery, correlated: its FROM/JOIN nodes are
        # enumerated by this walk too, and scope resolution reads them (M65 measured `LATERAL
        # (SELECT id FROM customer)` refused as `unauthorized_table customer`). A LATERAL over
        # anything else -- `VALUES` above all, whose scalar subqueries scope resolution does not
        # reach, so `FROM LATERAL (VALUES ((SELECT max(ssn) FROM customer)))` read an ungranted
        # table with `base_tables == []` -- still refuses: sqlglot's `Values` is not a `Query`.
        if (
            isinstance(source, exp.Lateral)
            and isinstance(source.this, exp.Subquery)
            and isinstance(source.this.this, exp.Query)
        ):
            continue
        return type(source).__name__
    return None


def spellings(names: set[str], fold=str.lower) -> set[str]:
    """Every spelling a name might be written in: itself, folded, and its bare last segment.

    One function for both comparisons -- what a body MENTIONS and what the source KNOWS -- 
    because when only the filtered set was widened, the known-object check refused every
    qualified reference it should have accepted. Two sets compared against each other must be
    normalised the same way or the comparison is between two different vocabularies.

    `fold` folds every letter by default, which only ADDS matches -- right for what a body
    mentions, where a match refuses. The known-object check passes the view's own dialect fold
    instead, since there a match VOUCHES (M154).
    """
    out: set[str] = set()
    for name in names:
        bare = name.rsplit(".", 1)[-1]
        out |= {name, fold(name), bare, fold(bare)}
    return out


def _coarse(name: str) -> str:
    """A key that joins every spelling an engine may fold onto one name: uppercase, then casefold.
    Lowercase alone kept `orderſ` apart from `orders` where Oracle folds the long s to S, so a view
    reading `orderſ` read the filtered ORDERS and was approved (M176); the dotless i and the final
    sigma are the same shape, and one fold or the other was not enough for `ᾀſ`, which Oracle
    reads as ᾈS (Codex's sixth review of #192). Swept live on Oracle Free 23, this joins each of
    the 6,679 letters Oracle answers to Oracle's fold of it; DuckDB, Postgres and SQLite fold
    ASCII alone, which it joins too. Wider than every engine, so for comparisons that refuse."""
    return name.upper().casefold()


def _restricting(names: set[str]) -> set[str]:
    """Every spelling of `names` a RESTRICTING comparison matches on, lowercased and `_coarse`. A
    match refuses, so a key wider than the engine's only adds refusals."""
    return spellings(names) | spellings(names, _coarse)


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
    return _restricting(names)


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
    # COVERAGE, not existence. `any(...)` let one source's completed discovery vouch for every
    # source in a merged snapshot: a federated snapshot where one catalog was discovered and a
    # legacy one never was reported asked=True, so a query of the undiscovered catalog's view came
    # back COMPLETE. That defeats the never-asked distinction exactly where federation makes it
    # matter -- and it is this file's own lesson, since `merge_snapshots` dropped views and jobs
    # entirely until recently and produced the same silence one layer down.
    #
    # `registry` is catalog -> schema and exists only on a FederatedSnapshot, so its size is the
    # number of sources merged. A single-source snapshot has none and needs one job.
    # `status == "done"`, not "anything but failed". `Job.status` is a free string, so a job
    # recorded `running` -- a partial snapshot, or a producer/version skew -- counted as coverage
    # and lineage came back COMPLETE over an inventory that had not finished being built. Only
    # `failed` marked it unavailable, so every other value read as success. Absence, failure and
    # IN-PROGRESS are three states; two of them were sharing the confident one.
    discovery_jobs = {
        getattr(j, "source_id", None)
        for j in getattr(snapshot, "jobs", ()) or ()
        if getattr(j, "id", None) == "discover:views" and getattr(j, "status", None) == "done"
    }
    registry = getattr(snapshot, "registry", {}) or {}
    sources_needed = max(1, len(registry))
    # Correspondable now, not merely counted: `merge_snapshots` re-keys each job's `source_id` to
    # its CATALOG, which is unique by construction and is what `registry` is keyed by. The earlier
    # version compared the cardinality of spec ids against the cardinality of catalogs and could
    # not check they named the same things, so two catalogs over one source id pinned asked=False
    # forever. Documenting that limit was not the same as closing it.
    asked = len(discovery_jobs & set(registry)) >= sources_needed if registry else bool(
        discovery_jobs)
    return ViewInventory({v.object_id: v for v in snapshot.views}, available=not failed,
                         asked=asked)


def check_views(
    ast: exp.Expression,
    views: dict[str, ViewDefinition],
    filtered: set[str],
    known: set[str] | None = None,
    granted: set[str] | None = None,
    dialect: str | None = None,
) -> Refusal | None:
    """Refuse a granted view that reads a row-filtered table, and one this engine cannot read.

    **What a refusal names (M53, with M145 merged into it).** The message goes to the model as
    repair text and, once retries run out, to the caller -- so it names the view the QUERY read,
    which the caller is granted, and a dependency only where `granted` holds it. It used to name
    the base table and say it was row-filtered for the caller, to a caller granted only the
    view; and a nested view, an unknown object or a cycle's whole chain the same way. The full
    detail goes to the server log. `granted` is the caller's visible objects; None names nothing
    beyond the view the query read -- the safe direction for a caller that does not pass it.

    `known` is the objects the source holds, and None and empty mean opposite things (M45): None
    is not supplied, and skips the unknown-object rule; an empty set is a source that knows
    nothing, so every object a body reads is unknown and the view is refused. A caller must never
    turn an empty set into None (`... or None`), which would skip the rule exactly where it is
    widest. Both deciders pass `set(policy.policy_schema)`.

    A name a CTE in scope binds is the CTE, never a view, and is not looked up: `dialect` is the
    engine that runs the query, whose binding `check_access` reads too, and a view body is bound
    by its own engine. Looked up, a CTE named `ſales` matched an ungranted SALES through the wide
    fold and refused the query, saying a view of that name reads a filtered table, and main did the
    same for `sales` (Codex's twelfth review of #192). The CTE's body is still walked.

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
    # Exact names -- never `spellings`, whose bare last segment would let a grant on
    # `sales.customer` name a filtered `public.customer` the caller does not hold, nor folded,
    # which let a grant on `ä` name a filtered `Ä`.
    return _walk(ast, views, filtered, known, (), 0, "", "",
                 set(granted) if granted else set(), _Request(views, known, filtered),
                 cte_binder(ast, dialect))


def _refuse(code: RefusalCode, view: str, message: str, detail: str) -> Refusal:
    """A view refusal naming only the view the query read; the dependency it is about goes to the
    operator's log, never to the model or the caller (M53)."""
    # WARNING, as `plan_query` logs the names it withholds: what cannot be said to the caller still
    # has to be said somewhere, and INFO is dropped under default logging.
    logger.warning("view %r refused (%s): %s", view, code.value, detail)
    return Refusal(code=code, message=message, subject=view)


def _buckets(node: exp.Table, catalog: str) -> set[str]:
    """The `_coarse` keys of every spelling `node` may name a view by. Every view it may read is in
    one of those buckets: the key spelled exactly, every key equal lowercased, since names equal
    lowercased are equal coarse (checked for every code point), and every key an engine's fold
    joins to it.

    Every one, the exact key's twins included. Neither fold reads quoting, and the exact key is the
    raw text, so a raw spelling can name a harmless view while the engine folds it onto another:
    Oracle reads an unquoted `ſALES` as SALES, Postgres an unquoted `SALES` as sales (Codex's third
    review of #192), and walking only the first match, exact or not, approved around the filtered
    one (its second). Walking a view can only refuse, so an inventory that holds two such keys
    refuses a body either would read: the conservative reading, at the cost of a view whose twin
    reads a filtered table.

    Case-insensitive at least, since unquoted identifiers are in all three engines. Measured: a body
    writing `FROM CLAIM_V` against an inventory keyed `claim_v` missed, the nested view was never
    walked, and its row-filtered base was never reached. APPROVED on the single-source path as well
    as the federated one, so this is not federation's bug. And as wide as any engine's fold: Oracle
    reads `ſales` as SALES, which `str.lower` keeps apart from `sales`, so the known-object check
    vouched for a view this lookup never found, and the filtered table behind it went unwalked; the
    query's own names are read the same way, as the engine that runs it reads them."""
    spelled = {object_key(node), node.name}
    if catalog:
        # A FEDERATED view body is written in the SOURCE's own naming, so a NESTED view
        # reads `claim_v` where the merged inventory keys it `pg.claim_v`, and the lookup
        # above misses. Measured: `pg.claim_v2` over `pg.claim_v` over a filtered
        # `pg.claim` was APPROVED while the byte-identical single-source shape refused.
        # Resolved in the ENCLOSING view's catalog only, never globally, so two catalogs
        # holding a view of the same name cannot resolve to each other's. The whole candidate
        # folded, catalog included: folding only the table half made any catalog alias carrying
        # an uppercase letter miss (measured: catalog `pg` refused and `PG` approved).
        spelled |= {f"{catalog}.{s}" for s in tuple(spelled)}
    return {_coarse(s) for s in spelled}


def _unwalked(node: exp.Table, request: _Request, catalog: str, depth: int):
    """Every view `node` may read, bucket by bucket, from the buckets this request has not walked
    clean in `catalog` at `depth` or above. A bucket is recorded clean when the walk asks past its
    last view, which it does only having refused none: a refusal returns from the walk and leaves
    the rest unasked. So references sharing twins, under any spelling or qualifier, visit them
    once."""
    for bucket in sorted(_buckets(node, catalog)):
        # A bucket of qualified views is walked the same from every catalog, since each resolves
        # the names inside it in its own prefix: recorded once for all of them, a view in each of N
        # schemas reading N qualified twins no longer lists them N times (Codex's eleventh review).
        scope = "" if bucket in request.qualified else catalog
        if request.clean_buckets.get((bucket, scope), -1) >= depth:
            continue
        for name in request.by_coarse.get(bucket, ()):
            yield name, request.views[name]
        request.clean_buckets[(bucket, scope)] = depth


@dataclass
class _Request:
    """What one `check_views` call works out once and reads at every table and view it meets: the
    views and the buckets walked clean, and the inventory and the known objects indexed by each
    fold. Walking every fold twin re-walked each shared view once per path to it, 37,448 parses at
    40 views (Codex's fourth review of #192); every table reference scanned and folded the whole
    inventory, 5.5 s at 2,000 views (its fifth); and references grouped by spelling, by fold, or by
    qualifier each enumerated and visited the twins they shared, N squared (its seventh to tenth).

    `clean` is every view, and `clean_buckets` every bucket, walked without a refusal, by the
    catalog the names inside resolve in, at the deepest depth it was. Clean at a depth is clean at
    any depth no deeper, since the nesting limit leaves more room; a cycle refuses, so nothing on
    the stack is ever in either."""

    views: dict[str, ViewDefinition]
    known: set[str] | None
    filtered: set[str]
    clean: dict[tuple[str, str], int] = field(default_factory=dict)
    clean_buckets: dict[tuple[str, str], int] = field(default_factory=dict)
    by_coarse: dict[str, list[str]] = field(default_factory=dict)
    qualified: set[str] = field(default_factory=set)
    wanted: set[str] = field(default_factory=set)
    _recognised: dict[str | None, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in sorted(self.views):
            self.by_coarse.setdefault(_coarse(key), []).append(key)
        self.qualified = {b for b, keys in self.by_coarse.items() if all("." in k for k in keys)}
        # The filtered set is widened the same way as a body's mentions and by its bare last
        # segment, so a body saying `customer` still matches a filter keyed `public.customer`.
        # Both directions, because a miss here means not refusing.
        self.wanted = _restricting(set(self.filtered))

    def recognised(self, dialect: str | None) -> set[str]:
        """Every spelling of a known object or a view, folded as `dialect` folds."""
        if dialect not in self._recognised:
            fold = rules_for(dialect).fold
            self._recognised[dialect] = spellings(self.known or set(), fold) | spellings(set(self.views), fold)
        return self._recognised[dialect]


def _walk(
    ast: exp.Expression,
    views: dict[str, ViewDefinition],
    filtered: set[str],
    known: set[str] | None,
    stack: tuple[str, ...],
    depth: int,
    catalog: str,
    top: str,
    granted: set[str],
    request: _Request,
    bound,
) -> Refusal | None:
    for node in ast.find_all(exp.Table):
        # Asked only of a reference some view could answer to, and through one binder for the
        # whole tree: asked of every reference, each asking resolving every CTE in scope again,
        # 2,000 CTEs were 4,004,001 name resolutions with no view at all (Codex's thirteenth
        # review of #192), and 4,002,000 with a view named like each (its fourteenth).
        if any(b in request.by_coarse for b in _buckets(node, catalog)) and bound(node):
            continue
        for name, view in _unwalked(node, request, catalog, depth):
            # The view the query read, named in every refusal below: at the top, as the query
            # spells it, which the grant check has passed. Never an inventory key: a twin the
            # fallback walks may be one the caller does not hold, a quoted "V" beside a granted
            # table `v` (Codex's fourth review of #192), and its name stays in the log.
            here = top or object_key(node)
            # Only the view the query read, which the caller holds. `it` names the view the
            # problem is in where that is the same view, and says "a view it reads" where not.
            it = repr(here) if name == here else f"a view {here!r} reads"
            if name in stack:
                return _refuse(RefusalCode.UNRESOLVABLE_VIEW, here,
                               f"{it} is defined in terms of itself, so it cannot be resolved.",
                               f"cycle {' -> '.join((*stack, name))}")
            if depth >= MAX_DEPTH:
                return _refuse(RefusalCode.UNRESOLVABLE_VIEW, here,
                               f"{here!r} nests views deeper than this engine will resolve.",
                               f"depth {depth} at {name!r}")
            inherited = name.rsplit(".", 1)[0] if "." in name else catalog
            if request.clean.get((name, inherited), -1) >= depth:
                continue
            body = body_of(view)
            if body is None:
                return _refuse(RefusalCode.UNRESOLVABLE_VIEW, here,
                               f"{it} has a definition this engine cannot read, so it cannot confirm "
                               "the row policy on the tables behind it.",
                               f"no readable body for {name!r}")
            unrecognised = unrecognised_source(body)
            if unrecognised is not None:
                return _refuse(RefusalCode.UNRESOLVABLE_VIEW, here,
                               f"{it} reads through a {unrecognised.lower()} rather than a named "
                               "table, so this engine cannot tell which tables it touches.",
                               f"{name!r} reads through a {unrecognised.lower()}")
            # The body's references its own CTEs bind, as the view's engine binds them: neither an
            # unknown object below nor a view in the walk into the body.
            inner = {id(t) for t in cte_references(body, view.dialect, engine=view.dialect)}
            # A body may only read objects the snapshot knows. A caller writing
            # `SELECT a FROM 'customer.csv'` is already refused UNAUTHORIZED_TABLE because the name
            # is not in `visible`; a view body was never held to that, so a view could reach a file
            # the caller could not. That is M27's shape -- the view reaching past the caller's own
            # boundary -- and it closes every unknown-object spelling at once rather than the file
            # literal specifically.
            if known is not None:
                # A reference a CTE BINDS is a name the body defines; the source is not expected to
                # know it. Counting one as an unknown object refused every ordinary CTE view -- five
                # of eight new refusals in the review's matrix were legitimate SQL. Exempt by SCOPE,
                # not by name: pooling every alias in the body and matching it by bare name let
                # `WITH secret AS (...) SELECT ... FROM main.secret` read a real table the snapshot
                # does not know, and a `secret` read outside the CTE's scope too, with any inventory
                # (Codex's review of #132). Bound as the view's OWN engine binds it, since its source
                # runs the body: SQLite binds a later sibling, and DuckDB's rule refused such a view
                # (Codex's review of #138). Never a qualified name. `_mentions` still collects every
                # alias, so a CTE named after a filtered table still trips the filtered check.
                # Folded as the VIEW's engine folds, never wider: a match here vouches that a read is
                # of a known object, and folding every letter let a known `Ä` vouch for an unknown `ä`
                # that DuckDB, SQLite and Postgres keep apart (Codex's review of #147, M154).
                fold = rules_for(view.dialect).fold
                recognised = request.recognised(view.dialect)
                # BOTH sides normalised. Widening only the known set still rejected `public.film`
                # against a snapshot that keys it `film`: a name is known when ANY of its spellings
                # matches any recognised one, not when its exact text appears.
                unknown = sorted(
                    object_key(x)
                    for x in body.find_all(exp.Table)
                    if id(x) not in inner and not (spellings({object_key(x)}, fold) & recognised)
                )
                if unknown:
                    return _refuse(RefusalCode.UNRESOLVABLE_VIEW, here,
                                   f"{it} reads an object that is not in this source, so this engine "
                                   "cannot tell what policy applies to it.",
                                   f"{name!r} reads unknown {unknown[0]!r}")
            mentioned = _mentions(body)
            reached = sorted(mentioned & request.wanted)
            if reached:
                # The table is offered as the repair only when the view reaches exactly ONE filtered
                # policy key, the body wrote that key exactly, and the caller holds it. Not by any
                # widened spelling: `spellings` folds to the bare segment, which is right for refusing
                # and wrong for naming -- a body reading `sales.customer` reaches a filtered
                # `public.customer` that way, and the repair would send the model to a different
                # table's rows. Not by the view's name prefix either: a prefix may be a schema, not a
                # catalog, and nothing here can tell which. A federated body writes `customer` where
                # the key is `pg.customer`, so it gets no hint -- fail closed, the refusal still holds.
                keys = sorted(f for f in filtered if _restricting({f}) & mentioned)
                # Exact names, never folded: a caller holding `ä` was told to query 'Ä', the
                # filtered table it does not hold (Codex's seventh review of #192).
                written = {object_key(t) for t in body.find_all(exp.Table)}
                held = keys[0] if len(keys) == 1 and keys[0] in written and keys[0] in granted else None
                repair = f" Query {held!r} directly." if held else ""
                return _refuse(RefusalCode.UNGOVERNED_VIEW, here,
                               f"{it} reads a table that is row-filtered for you, and this engine "
                               f"cannot apply that filter through a view.{repair}",
                               f"{name!r} reads row-filtered {keys or reached}, offered {held!r}")
            # The body's own catalog carries into it: a view two levels down is still written in
            # the source's naming, not the federation's.
            nested = _walk(body, views, filtered, known, (*stack, name), depth + 1, inherited, here,
                           granted, request, lambda t: id(t) in inner)
            if nested is not None:
                return nested
            request.clean[(name, inherited)] = depth
    return None
