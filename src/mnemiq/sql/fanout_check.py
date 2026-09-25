from __future__ import annotations

import logging
import numbers
from decimal import Decimal, InvalidOperation
from types import MappingProxyType

from sqlglot import exp
from sqlglot.optimizer.merge_subqueries import merge_subqueries
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from mnemiq.contract import Snapshot
from mnemiq.sql.qualify import names_one_object, object_key
from mnemiq.sql.verdict import Refusal, RefusalCode

logger = logging.getLogger(__name__)

class KeyFacts(dict):
    """(object_id, column name) -> True when the column's non-null values are all distinct, False
    when some repeat. A column the profile never measured is ABSENT, and absence is never read as
    either.

    `partitions` maps (object_id, column, partition label) to the columns unique among that
    partition's rows (`Column.unique_within`). A plain dict works wherever a KeyFacts does: it
    simply has no partitions.
    """

    partitions: MappingProxyType = MappingProxyType({})  # read-only default; key_facts sets its own


PARTITIONS_JOB = "profile:partitions"  # recorded by enrich_structural once partitions are profiled
_warned: set[tuple[str, str]] = set()


def partitions_profiled(snapshot) -> bool:
    """Whether enrichment profiled key uniqueness per partition for this snapshot -- found or
    not. A database can legitimately record none, so the job, not the facts, is the signal.

    Coverage, not existence, as `inventory_for` learned for view discovery: a federated snapshot
    carries every source's jobs, re-keyed to its catalog, and one re-enriched source must not
    vouch for a legacy one. Every catalog in `registry` needs a `done` job; a single source, one.
    """
    done = {j.source_id for j in getattr(snapshot, "jobs", None) or []
            if j.id == PARTITIONS_JOB and j.status == "done"}
    registry = getattr(snapshot, "registry", {}) or {}
    return set(registry) <= done if registry else bool(done)


def guard_on(setting: bool | None, snapshot) -> bool:
    """Whether the fan-out check runs for this snapshot. An explicit setting wins. Unset (auto),
    it runs exactly when the snapshot's partitions were profiled: on an older snapshot a
    current-row SCD join would be refused -- the join is correct, the key facts cannot yet show
    it -- so auto leaves that snapshot as it was, and says so once."""
    if setting is not None:
        return setting
    if partitions_profiled(snapshot):
        return True
    seen = (getattr(snapshot, "source_id", ""), getattr(snapshot, "version", ""))
    if seen not in _warned:
        _warned.add(seen)
        logger.warning(
            "fan-out guard off for snapshot %s of %r: its partitions were not all profiled "
            "(enriched before per-partition profiling, or a partition query failed -- see its "
            "profile:partitions job), so current-row SCD joins would be refused. Re-enrich to "
            "turn it on, or set MNEMIQ_GUARD_FANOUT=1 to force it.", seen[1], seen[0])
    return False


def partition_label(value) -> str:
    """How a value names a profiled partition -- the one spelling profiling and the check share.

    `=true`, `=1`, `='Y'`, or `IS NULL`. A number is written as an integer when it is one, so a
    driver's 1.0 and a query's 1 meet; a string keeps its case, as SQL equality does.
    """
    if value is None:
        return "IS NULL"
    if isinstance(value, bool):
        return "=true" if value else "=false"
    if isinstance(value, numbers.Number):
        return f"={_exact(value)}"
    return f"='{value}'"


def _exact(value) -> str:
    """A number's exact decimal spelling: 1.0 and 1 meet, 1.50 and 1.5 meet, and every digit of an
    integer is kept. A float round trip spelled 2**53 and 2**53 + 1 the same, and profiling then
    gave one partition the other's facts.

    Built from the digits and exponent, never through `int()` or a context-bound `normalize()`: a
    query's literal is untrusted, and `1e5000` would exceed the integer-string limit while
    `1e99999999` would build a hundred-million-digit integer. Plain notation up to 30 places either
    side of the point, `str()` beyond. Two values never share a spelling: either form parses back to
    exactly the value it came from."""
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))  # a float's shortest repr
    except InvalidOperation:
        return str(value)
    if not number.is_finite():
        return str(number)
    sign, digits, exponent = number.as_tuple()
    kept = len(digits)
    while kept > 1 and digits[kept - 1] == 0:  # 1.50 -> 1.5, 10 -> 1E+1
        kept -= 1
    exponent += len(digits) - kept
    if digits[:kept] == (0,):
        return "0"  # -0 is 0 to SQL
    canonical = Decimal((sign, digits[:kept], exponent))
    return format(canonical, "f") if -30 <= exponent <= 30 else str(canonical)


def key_facts(snapshot: Snapshot) -> KeyFacts:
    """Which columns hold each value at most once, from the profile enrichment already took.

    Declared foreign keys are not enough and are not used: `fact_claim` and `fact_premium` share
    `policy_id` with no relationship declared between them, which is exactly how a chasm trap
    escapes schema metadata. The profile's `count(DISTINCT col)` is exact, so this is a fact about
    the data at profiling time, not an estimate.
    """
    facts = KeyFacts()
    partitions: dict[tuple[str, str, str], frozenset[str]] = {}
    for column in snapshot.columns:
        for label, unique in (column.unique_within or {}).items():
            partitions[(column.object_id, column.name, label)] = frozenset(unique)
        if column.row_count is None or column.distinct_count is None or column.null_count is None:
            continue
        facts[(column.object_id, column.name)] = (
            column.distinct_count == column.row_count - column.null_count
        )
    facts.partitions = partitions
    return facts


def _unique(facts: KeyFacts, table: str, columns: tuple[str, ...]) -> bool | None:
    """Whether `columns` together hold each value once in `table`: True, False, or None (unknown).

    The profile measures one column at a time. Any member that is unique makes the whole key
    unique; a single repeating column is a definite False. Several columns that each repeat may
    still be unique TOGETHER, which a per-column profile cannot tell -- unknown, and unknown
    never fires.
    """
    known = [facts.get((table, c)) for c in columns]
    if any(k is True for k in known):
        return True
    if len(columns) == 1 and known[0] is False:
        return False
    return None


def _spelling(name: str, names) -> str | None:
    """The snapshot's spelling of a column the SQL named, ignoring case, or None.

    Case only -- the rule `names_one_object` applies to tables, with the same open question
    (M55: a quoted identifier is case-sensitive in Postgres, and quoting is gone by now).
    """
    hits = [n for n in names if n.lower() == name.lower()]
    return hits[0] if len(hits) == 1 else None


def value_terms(e: exp.Expression | None) -> list[list[exp.Column]]:
    """An aggregate's argument split into additive terms, each as its value columns.

    A term is inflated when it reads a repeated table and no table that appears once per join row.
    Such a table gives the term its grain: join rows map one-to-one onto its rows, so a term
    computed per row sums each of them once. `SUM(li.quantity * p.unit_price)` is right though `p`
    repeats per line item, and reading it as two columns refused the canonical revenue query.
    Columns share a term only through a product or a quotient; any other function passes its
    arguments' terms through, so `ROUND(li.quantity + p.unit_price, 2)` still sums `p.unit_price`
    on its own. Splitting only at the operators it knows would approve every wrapper it does not.
    """
    return [term for term in _terms(e) if term]


# Multiplying out is exponential in nesting depth. Past this, a product falls back to one term per
# column, plus the constant term if both sides had one: coarser, and it refuses whatever the full
# expansion would -- every expanded term that fires contains a column that fires alone.
_MAX_TERMS = 64

Terms = list[list[exp.Column]]


def _product(left: Terms, right: Terms) -> Terms:
    if len(left) * len(right) <= _MAX_TERMS:
        return [a + b for a in left for b in right]
    # Dropping the constant term let an enclosing `* p.unit_price` merge back into `li`'s terms:
    # seven `(li.quantity + 1)` factors approved what one refused.
    columns = dict.fromkeys(c for term in left + right for c in term)
    return [[c] for c in columns] + ([[]] if [] in left and [] in right else [])


def _literal(e: exp.Literal) -> Terms:
    # Zero is no term at all; any other constant is the EMPTY term.
    return [] if e.is_number and not str(e.this).strip("0.") else [[]]


def _choice(e: exp.Expression, walk) -> Terms | None:
    # A choice -- IF, CASE, COALESCE, NULLIF -- is one of its value branches per row, each read by
    # `walk`; its conditions, a simple CASE's operand and NULLIF's comparand are never values.
    # `SUM(CASE WHEN p.region = 'east' THEN c.amount ELSE 0 END)` sums `c`; repeating `p` rows does
    # not corrupt it. The prototype that counted every column inside the aggregate fired on 14.6%
    # of BIRD's own gold for this shape. None when `e` is not a choice.
    if isinstance(e, exp.If):
        return walk(e.args.get("true")) + walk(e.args.get("false"))
    if isinstance(e, exp.Case):
        out: Terms = []
        for branch in e.args.get("ifs") or []:
            out += walk(branch.args.get("true"))
        return out + walk(e.args.get("default"))
    if isinstance(e, exp.Coalesce):
        out = walk(e.this)
        for arg in e.expressions:
            out += walk(arg)
        return out
    if isinstance(e, exp.Nullif):  # its first argument, or NULL
        return walk(e.this)
    return None


def _terms(e: exp.Expression | None) -> Terms:
    # A non-zero constant is the EMPTY term: dropped at the top, but under a product it leaves the
    # other factor alone -- `(li.quantity + 1) * p.unit_price` sums `p.unit_price` per line item.
    # Zero and NULL are no term at all, so `IIF(x, li.quantity, 0) * p.unit_price` is one term.
    if e is None or isinstance(e, exp.Null):
        return []
    if isinstance(e, exp.Literal):
        return _literal(e)
    if isinstance(e, exp.Column):
        return [[e]]
    if isinstance(e, (exp.Add, exp.Sub)):
        return _terms(e.left) + _terms(e.right)
    if isinstance(e, exp.Mul):
        return _product(_terms(e.left), _terms(e.right))
    if isinstance(e, exp.Div):
        # Each side once: recomputing the denominator per numerator term cost width ** depth.
        return _product(_terms(e.left), _alternatives(e.right))
    if (branches := _choice(e, _terms)) is not None:
        return branches
    # Any other node: its arguments' terms, side by side. An argument with no column in it -- a
    # precision, a CAST's type -- is a parameter, not a summand.
    out = []
    for child in e.iter_expressions():
        terms = _terms(child)
        if any(terms):
            out += terms
    return out or [[]]


def _alternatives(e: exp.Expression | None) -> Terms:
    # The values a denominator can take, each as the columns it reads. A choice is one alternative
    # per branch, as in a product: dividing by `COALESCE(li.quantity, 1)` divides by 1 on some
    # rows, and there the numerator is summed alone. Flattened into one factor, the denominator
    # approved `p.unit_price / COALESCE(li.quantity, 1)` while the product was refused. A sum is
    # NOT split -- 1 / (a + b) is not 1/a + 1/b -- so anything else is one factor per combination
    # of its arguments' alternatives.
    if e is None or isinstance(e, exp.Null):
        return []
    if isinstance(e, exp.Literal):
        return _literal(e)
    if isinstance(e, exp.Column):
        return [[e]]
    if (branches := _choice(e, _alternatives)) is not None:
        return branches
    out: Terms = [[]]
    for child in e.iter_expressions():
        alternatives = _alternatives(child)
        if any(alternatives):
            out = _product(out, alternatives)
    return out


def check_fanout(
    ast: exp.Expression,
    visible: dict[str, set[str]],
    facts: KeyFacts,
    dialect: str | None = None,
) -> Refusal | None:
    """Refuse an aggregate whose rows a join has multiplied (register M109).

    `FROM fact_claim JOIN fact_premium ON policy_id` with a SUM on each side repeats every claim
    once per premium row on its policy; the query runs, and the total is silently inflated. Per
    SELECT scope, over BASE tables only:

    1. collect equi-joins (JOIN ... ON, USING, WHERE a.x = b.y);
    2. ask `facts` whether each side's key is unique;
    3. a table is DUPLICATED when a walk outward from it enters another table by a key that
       repeats there;
    4. refuse SUM/AVG over a term (`value_terms`) that reads a duplicated table and no table
       shown to appear once per row, or a count-like aggregate (COUNT(x), COUNT(*), a SUM of
       constants) only when EVERY table is duplicated -- a count over a plain one-to-many counts
       the finer table, which may be what was asked.

    Never fires on MIN, MAX, or any DISTINCT aggregate (immune to repetition), or on a join whose
    key uniqueness is unknown. A CTE or subquery that aggregates, de-duplicates, windows or limits
    is opaque and treated as unique: an aggregated CTE is usually one row per key, and guessing
    otherwise would refuse the very rewrite this asks for. One that only passes a join's rows
    through is, where sqlglot can merge it, merged into its parent and checked flat (`_merged`).
    """
    found = _check_scopes(ast, visible, facts)
    if found is None and (ast.find(exp.Subquery) or ast.find(exp.CTE)):
        merged = _merged(ast, visible, dialect)
        if merged is not None:
            found = _check_scopes(merged, visible, facts)
    return found


def _merged(ast: exp.Expression, visible: dict[str, set[str]],
            dialect: str | None = None) -> exp.Expression | None:
    """The query with its pass-through derived tables merged into their parents, where sqlglot can.

    Opaque derived sources were a bypass: `SELECT SUM(t.x) FROM (SELECT c.x FROM claim c JOIN
    premium p ON ...) t` got an inflated sum approved, and BIRD gold that computes its inflation
    that way hid from the gold check. sqlglot's `merge_subqueries` merges a derived table only when
    the result means the same -- never one that aggregates or de-duplicates -- so a merged query is
    checked like any flat one. This NARROWS the bypass rather than closing it: sqlglot also leaves
    a CTE read more than once, and some derived tables under an outer join, unmerged, and those
    stay opaque (the CTE case is pinned in the tests). Anything qualify or merge cannot handle
    leaves the check as it was.
    """
    schema: dict = {}
    for object_id, columns in visible.items():
        node = schema
        *parents, name = object_id.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[name] = {c: "TEXT" for c in columns}
    # A catalog that mixes bare and qualified object ids is a schema sqlglot rejects (it wants one
    # nesting depth), which silently switched the merge off for every query against it. Without
    # a schema, qualify still resolves every column the query itself qualifies.
    for attempt in (schema, None):
        try:
            return merge_subqueries(qualify(ast.copy(), schema=attempt, dialect=dialect,
                                            quote_identifiers=False,
                                            validate_qualify_columns=False))
        except Exception as exc:  # noqa: BLE001 -- unmergeable is not evidence of fan-out
            if attempt is None:
                logger.warning("fan-out check could not merge derived tables: %s", exc)
    return None


def _check_scopes(ast: exp.Expression, visible: dict[str, set[str]],
                  facts: KeyFacts) -> Refusal | None:
    try:
        scopes = traverse_scope(ast)
    except Exception as exc:  # noqa: BLE001 -- an unresolvable scope is not evidence of fan-out
        logger.warning("fan-out check skipped, scope did not resolve: %s", exc)
        return None
    for scope in scopes:
        select = scope.expression
        if not isinstance(select, exp.Select):
            continue
        sources: dict[str, str] = {}  # alias (lowercased) -> visible object id
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table):
                hits = [t for t in visible if names_one_object(object_key(source), [t])]
                if len(hits) == 1:
                    sources[alias.lower()] = hits[0]
        if len(sources) < 2:
            continue
        found = _check_scope(select, sources, visible, facts)
        if found is not None:
            return found
    return None


def _check_scope(select: exp.Select, sources: dict[str, str], visible, facts: KeyFacts):
    sources = dict(sources)  # a SEMI/ANTI right side is dropped from it below

    def owner(col: exp.Column) -> str | None:
        if col.table:
            alias = col.table.lower()
            return alias if alias in sources else None
        hits = [a for a, t in sources.items() if _spelling(col.name, visible.get(t, ())) is not None]
        return hits[0] if len(hits) == 1 else None

    # alias pair (sorted) -> [(left column, right column)], in the snapshot's spelling
    pairs: dict[tuple[str, str], list[tuple[str, str]]] = {}
    # (a, b): each `a` row meets at most one `b` row, whatever b's key -- `b` is an ASOF right side
    at_most_one: set[tuple[str, str]] = set()

    def add(a: str, a_col: str, b: str, b_col: str, asof: str | None = None) -> None:
        if a == b:
            return
        a_name = _spelling(a_col, visible.get(sources[a], ()))
        b_name = _spelling(b_col, visible.get(sources[b], ()))
        if a_name is None or b_name is None:
            return
        key, pair = ((a, b), (a_name, b_name)) if a < b else ((b, a), (b_name, a_name))
        cols = pairs.setdefault(key, [])
        # `ON c.k = p.k WHERE c.k = p.k` is one key. Recorded twice, it read as a two-column
        # composite the profile cannot settle, and the check went silent.
        if pair not in cols:
            cols.append(pair)
        if asof in (a, b):
            at_most_one.add((b, a) if asof == a else (a, b))

    # (condition, its ASOF right side, the aliases it can FILTER -- None for all). A conjunct in
    # an outer join's ON removes no row of the preserved side: `LEFT JOIN x ON ... AND
    # d.is_current = 1` still joins every history row of `d`, and reading it as a filter on `d`
    # approved an inflated sum. WHERE and an inner ON filter every alias; a LEFT JOIN's ON only
    # its right side; a RIGHT or FULL JOIN's ON none.
    conditions: list[tuple[exp.Expression, str | None, set[str] | None]] = []
    # USING binds to the tables LEFT of the join, and after `a JOIN b USING (k)` the merged `k`
    # equals both `a.k` and `b.k` -- so a later `JOIN c USING (k)` is an edge to each of them.
    # Binding to "the one other table with that column" instead refuses nothing on a three-way
    # USING chain, where every table has it.
    frm = select.args.get("from_")
    earlier = [frm.this.alias_or_name.lower()] if frm and isinstance(frm.this, exp.Table) else []
    for join in select.args.get("joins") or []:
        right = join.this.alias_or_name.lower() if isinstance(join.this, exp.Table) else None
        if str(join.args.get("kind") or "").upper() in {"SEMI", "ANTI"}:
            # Each left row is kept or dropped once, and no right row reaches the result: no edge,
            # and no table of this join. Left in `sources`, it could never be repeated, and the
            # every-table COUNT rule could never fire past it.
            sources.pop(right, None)
            continue
        # ASOF matches each left row to at most one right row, so the left never fans into it --
        # but one right row can meet many left rows, and that direction stays an edge.
        asof = right if str(join.args.get("method") or "").upper() == "ASOF" else None
        if join.args.get("on") is not None:
            side = str(join.args.get("side") or "").upper()
            filters = None if not side else {right} if side == "LEFT" and right else set()
            conditions.append((join.args["on"], asof, filters))
        for ident in join.args.get("using") or []:
            if right not in sources:
                continue
            for left in earlier:
                if left in sources and _spelling(ident.name, visible.get(sources[left], ())):
                    add(left, ident.name, right, ident.name, asof)
        if right is not None:
            earlier.append(right)
    if select.args.get("where") is not None:
        conditions.append((select.args["where"].this, None, None))
    for condition, asof, _ in conditions:
        for eq in condition.find_all(exp.EQ):
            if eq.find_ancestor(exp.Select) is not select:
                continue  # inside a subquery: its columns belong to another scope
            left, right_col = eq.left, eq.right
            if isinstance(left, exp.Column) and isinstance(right_col, exp.Column):
                lo, ro = owner(left), owner(right_col)
                if lo and ro:
                    add(lo, left.name, ro, right_col.name, asof)
    if not pairs:
        return None

    # alias -> the columns an as-of filter in this scope narrows it by. Uniqueness is read
    # table-wide, so a versioned table joined as of the fact's date (`o.day BETWEEN d.valid_from
    # AND d.valid_to`) was refused, and every repair that kept the join was refused again until
    # the question deferred. A BETWEEN whose bounds are two of the table's own columns, tested
    # against another table's value, is that join: it MAY leave one version per key, unknown,
    # and unknown never fires. Nothing wider: counting a flag (`is_current`), an equality or an
    # IS NULL let an ordinary binary filter (`is_returned = 'N'`) switch the guard off for an
    # inflated sum -- a silent wrong number where a refusal is a visible deferral -- and syntax
    # cannot tell a current-row flag from one. The DATA can: `pinned` below.
    narrowed: dict[str, set[str]] = {}
    # alias -> the columns a filter pins to one row per value, per the profile's partitions
    # (`Column.unique_within`): among the rows where `is_current = 1`, `customer_id` is unique, a
    # definite fact, not a guess -- while among the rows where `is_returned = 0`, `product_id`
    # still repeats and nothing is recorded. A snapshot without partitions pins nothing.
    partitions = getattr(facts, "partitions", {})
    pinned: dict[str, set[str]] = {}

    def conjuncts(e: exp.Expression):
        e = e.unnest()  # flatten() unwraps parentheses inside the AND chain itself
        return e.flatten() if isinstance(e, exp.And) else [e]

    for condition, _, filters in conditions:
        for c in conjuncts(condition):
            if not isinstance(c, exp.Between):
                col, labels = _partition_filter(c)
                alias = owner(col) if col is not None and col.find_ancestor(exp.Select) is select \
                    else None
                if alias is None or (filters is not None and alias not in filters):
                    continue
                name = _spelling(col.name, visible.get(sources[alias], ()))
                if labels is _TRUTHY:
                    labels = _truthy_label(partitions, sources[alias], name)
                for label in labels:
                    unique = partitions.get((sources[alias], name, label))
                    if unique:
                        pinned.setdefault(alias, set()).update(u.lower() for u in unique)
                continue
            bounds, tested = [c.args.get("low"), c.args.get("high")], c.this
            if not all(isinstance(x, exp.Column) and x.find_ancestor(exp.Select) is select
                       for x in [*bounds, tested]):
                continue
            owners = {owner(x) for x in bounds}
            if len(owners) != 1 or None in owners or owner(tested) in owners | {None}:
                continue
            alias = owners.pop()
            if filters is not None and alias not in filters:
                continue
            names = [_spelling(x.name, visible.get(sources[alias], ())) for x in bounds]
            if None not in names:
                narrowed.setdefault(alias, set()).update(n.lower() for n in names)

    def repeats(alias: str, key: tuple[str, ...]) -> bool:
        if _unique(facts, sources[alias], key) is not False:
            return False
        if {k.lower() for k in key} <= pinned.get(alias, set()):
            return False  # the filter leaves one row per key value, per the profile
        return not narrowed.get(alias, set()) - {k.lower() for k in key}

    adj: dict[str, list[str]] = {a: [] for a in sources}
    # a -> {b: b's key columns}: each `a` row meets MANY `b` rows, because b's key repeats
    fans_into: dict[str, dict[str, tuple[str, ...]]] = {a: {} for a in sources}
    for (a, b), cols in pairs.items():
        a_cols, b_cols = tuple(c[0] for c in cols), tuple(c[1] for c in cols)
        adj[a].append(b)
        adj[b].append(a)
        if repeats(b, b_cols) and (a, b) not in at_most_one:
            fans_into[a][b] = b_cols
        if repeats(a, a_cols) and (b, a) not in at_most_one:
            fans_into[b][a] = a_cols

    def multiplier(t: str) -> tuple[str, tuple[str, ...]] | None:
        # Walk OUTWARD, asking of each edge only in the direction it is traversed. Testing every
        # visited table's edges -- including the one pointing back toward `t` -- charged the
        # policy row repeated per claim to the claim, and fired on a plain many-to-one AVG.
        seen, stack = {t}, [t]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w in seen:
                    continue
                if w in fans_into[u]:
                    return w, fans_into[u][w]
                seen.add(w)
                stack.append(w)
        return None

    dup = {a: m for a in sources if (m := multiplier(a)) is not None}
    if not dup:
        return None

    def keyed_to_all(t: str) -> bool:
        seen, stack = {t}, [t]
        while stack:
            for w in adj[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen) == len(sources)

    # A table carries a term at its own grain only if it appears at most once per join row:
    # repeated by no walk AND keyed to every other table. Absence from `dup` alone proves nothing
    # for a table no key reaches -- CROSS JOIN, a comma, a range or an expression join -- and read
    # as proof, such a table excused an inflated term. An owner `owner()` cannot resolve -- a CTE's
    # or derived table's column, an ambiguous bare name, any qualifier that is not a base table of
    # this scope -- counts as single: the pinned opaque limit.
    single = {a for a in sources if a not in dup and keyed_to_all(a)}

    def of_this_row(col: exp.Column) -> bool:
        # A nested scope's own columns -- `(SELECT MAX(rate) FROM fx)` -- are not values of this
        # row: read as owners they resolved to None, and None excused the term. A CORRELATED
        # reference to one of this scope's aliases is, and dropping it too left the subquery
        # ownerless and approved an inflated sum. An alias a nested SELECT defines itself shadows
        # ours. An unqualified nested column is dropped: SQL resolves it inside the nested scope
        # first and reaches this row only when the inner tables lack it, which this does not
        # look up -- so a BARE correlated reference is a known gap.
        inner = col.find_ancestor(exp.Select)
        alias = col.table.lower() if col.table else None
        if inner is not select and alias not in sources:
            return False
        while inner is not None and inner is not select:
            if alias in _defines(inner):
                return False
            inner = inner.find_ancestor(exp.Select)
        return inner is select

    summed: dict[str, None] = {}  # duplicated aliases whose values are aggregated, in order
    chasm: exp.Expression | None = None
    windowed = False
    for agg in select.find_all(exp.Sum, exp.Avg, exp.Count):
        if agg.find_ancestor(exp.Select) is not select:
            continue
        if isinstance(agg.this, exp.Distinct) or agg.args.get("distinct"):
            continue
        terms = [] if isinstance(agg, exp.Count) else value_terms(agg.this)
        if not terms:
            if chasm is None and set(dup) == set(sources):
                chasm = agg
            continue
        for term in terms:
            owners = list(dict.fromkeys(owner(col) for col in term if of_this_row(col)))
            if any(o in dup for o in owners) and not any(o is None or o in single for o in owners):
                summed.update(dict.fromkeys(o for o in owners if o in dup))
                window = agg.find_ancestor(exp.Window)
                windowed |= window is not None and window.find_ancestor(exp.Select) is select
    if summed:
        return _inflated(list(summed), sources, dup, select, windowed)
    if chasm is not None:
        return _counted_chasm(chasm, sources)
    return None


_TRUTHY = ["<truthy>"]  # a bare `F` or `F IS TRUE`; resolved by `_truthy_label`


def _truthy_label(partitions, table: str, column: str) -> list[str]:
    """Which partition a bare `F` or `F IS TRUE` picks: `=true` on a boolean column, `=1` on a
    0/1 flag, and nothing otherwise -- on values {1, 2} a bare `F` keeps both partitions, and
    reading it as `= 1` pinned a key the other partition repeats. The profile records every
    value's label, so the value set is known."""
    labels = {label for (t, c, label) in partitions if t == table and c == column}
    if labels and labels <= {"=true", "=false"}:
        return ["=true"]
    if labels and labels <= {"=0", "=1"}:
        return ["=1"]
    return []


def _partition_filter(c: exp.Expression) -> tuple[exp.Column | None, list[str]]:
    """The column a filter conjunct restricts, and the partition labels it may pick.

    `F = 1` -> `=1`; `F = 'Y'` -> `='Y'`; a bare `F` or `F IS TRUE` -> `_TRUTHY`, resolved against
    the column's profiled values (`_truthy_label`); `F IS NULL` -> `IS NULL`. Anything else picks
    no profiled partition.
    """
    if isinstance(c, exp.Column):
        return c, _TRUTHY
    if isinstance(c, exp.Is) and isinstance(c.this, exp.Column):
        if isinstance(c.expression, exp.Null):
            return c.this, ["IS NULL"]
        if isinstance(c.expression, exp.Boolean) and c.expression.this:
            return c.this, _TRUTHY
    if isinstance(c, exp.EQ):
        for col, value in ((c.left, c.right), (c.right, c.left)):
            if isinstance(col, exp.Column) and isinstance(value, exp.Boolean):
                return col, [partition_label(bool(value.this))]
            if isinstance(col, exp.Column) and isinstance(value, exp.Literal):
                if value.is_string:
                    return col, [partition_label(value.this)]
                try:
                    return col, [partition_label(Decimal(value.this))]
                except InvalidOperation:
                    return None, []
    return None, []


def _defines(select: exp.Select) -> set[str]:
    """The aliases a SELECT's own FROM and JOINs introduce, lowercased."""
    frm = select.args.get("from_")
    nodes = ([frm.this] if frm else []) + [j.this for j in select.args.get("joins") or []]
    return {n.alias_or_name.lower() for n in nodes}


def _grouping(select: exp.Select) -> list[str] | None:
    """The answer's grain as the query states it, or None when the query has no GROUP BY.

    Every grouping form counts: plain expressions (`GROUP BY 1` named by its column), ROLLUP,
    CUBE, GROUPING SETS, and DuckDB's `GROUP BY ALL` (every non-aggregate output). Reading only
    the plain list sent those to the one-overall-total advice, whose repair returns one figure
    where the question asked for one per group. A form this cannot name still comes back as []
    -- grouped, unnamed -- never as None.
    """
    group = select.args.get("group")
    if group is None:
        return None
    named = []
    for e in group.expressions:
        if isinstance(e, exp.Literal) and e.is_int and 1 <= int(e.this) <= len(select.expressions):
            e = select.expressions[int(e.this) - 1].unalias()
        named.append(e.sql())
    for form in ("rollup", "cube", "grouping_sets"):
        for node in group.args.get(form) or []:
            named += [c.sql() for c in node.find_all(exp.Column)]
    if group.args.get("all"):
        named += [e.unalias().sql() for e in select.expressions if not e.find(exp.AggFunc)]
    return list(dict.fromkeys(named))


def _inflated(aliases: list[str], sources: dict[str, str], dup, select: exp.Select,
              windowed: bool = False) -> Refusal:
    because, keys = [], {}
    for a in aliases:
        entered, key = dup[a]
        t, w = sources[a], sources[entered]
        because.append(f"each {t} row is repeated once per matching {w} row "
                       f"({w}.{', '.join(key)} is not unique)")
        keys.update(dict.fromkeys(key))
    tables = " and ".join(dict.fromkeys(sources[a] for a in aliases))
    key = ", ".join(keys)
    head = ("This query aggregates across a join that multiplies rows, so the result is "
            f"inflated: {'; '.join(because)}.")
    if windowed:
        # A window keeps one output row per input row; the GROUP BY advice below would discard
        # its partition and that row-level grain.
        return Refusal(code=RefusalCode.FAN_OUT, message=(
            f"{head} Compute the window on {tables}'s own rows in a CTE or subquery, keeping its "
            "PARTITION BY, then join that result instead of the raw table."))
    # The first wording said "grouped by the key you join or group on", and the local 14B did
    # exactly that: a CTE per fact grouped by the join key, inner-joined on it, which drops every
    # key present in only one table and repaired 0 of 8. So name the ANSWER's grain instead, and
    # say how to combine without an inner join -- per region it drops a region, one grain up.
    grouped = _grouping(select)
    by_key = grouped is not None and any(
        g.split(".")[-1].lower() in {k.lower() for k in keys} for g in grouped)
    group = select.args.get("group")
    subtotals = group is not None and any(group.args.get(f) for f in ("rollup", "cube",
                                                                       "grouping_sets"))
    if grouped is None:
        grain = "reduced to one overall total with no GROUP BY"
        combine = "then CROSS JOIN those one-row totals"
    else:
        columns = ", ".join(grouped) or "the columns your query groups by"
        # NULL-safe, because GROUP BY puts NULL keys in one group and `=` never matches NULL to
        # NULL: a plain FULL OUTER JOIN returns a NULL group as two half-rows.
        on = (f"joining on {columns} with IS NOT DISTINCT FROM and outputting COALESCE of the two "
              "sides' values (an inner join, or plain =, drops a group present on one side only "
              "or whose value is NULL)")
        if subtotals:
            # Flattened to its columns, a ROLLUP loses its subtotal and grand-total rows; kept
            # whole, a subtotal row and a real NULL group are both NULL in the column, and only
            # GROUPING() tells them apart.
            grain = (f"grouped exactly as your query groups ({group.sql()}) with "
                     f"GROUPING({columns}) kept as a column")
            combine = f"then FULL OUTER JOIN those results on that GROUPING column too, {on}"
        else:
            grain = (f"grouped by the answer's own columns ({columns}), joining in only the lookup "
                     "table that supplies a column it lacks")
            combine = f"then FULL OUTER JOIN those per-group results, {on}"
    avoid = "" if by_key else f", not by the join key ({key})"
    warn = "" if by_key else (
        f" Grouping each table by {key} and inner-joining the results drops every "
        f"{' or '.join(keys)} value that appears in only one table.")
    # An inner join can MEAN "only keys in both" (products that had returns); whole-table totals
    # widen that cohort silently. Usually it is an accident -- the loss ratio's was -- so the
    # restriction is kept only when the question sets it.
    cohort = (" These totals count every key; only if the question itself limits the answer to "
              "keys present in both tables, restrict each table with EXISTS on the other first.")
    return Refusal(
        code=RefusalCode.FAN_OUT,
        message=(
            f"{head} Aggregate each of {tables} in its own CTE or subquery, {grain}{avoid}; "
            f"{combine}.{warn}{cohort}"
        ),
    )


def _counted_chasm(agg: exp.Expression, sources: dict[str, str]) -> Refusal:
    tables = ", ".join(dict.fromkeys(sources.values()))
    return Refusal(
        code=RefusalCode.FAN_OUT,
        message=(
            f"{agg.sql()} counts rows of a join in which every table is repeated ({tables}), so "
            "it counts combinations rather than rows of any one table. Count the table you mean "
            "on its own, or use COUNT(DISTINCT <that table's key>)."
        ),
    )
