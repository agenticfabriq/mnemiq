from __future__ import annotations

import logging

from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from mnemiq.contract import Snapshot
from mnemiq.sql.qualify import names_one_object, object_key
from mnemiq.sql.verdict import Refusal, RefusalCode

logger = logging.getLogger(__name__)

# (object_id, column name) -> True when the column's non-null values are all distinct, False when
# some repeat. A column the profile never measured is ABSENT, and absence is never read as either.
KeyFacts = dict[tuple[str, str], bool]


def key_facts(snapshot: Snapshot) -> KeyFacts:
    """Which columns hold each value at most once, from the profile enrichment already took.

    Declared foreign keys are not enough and are not used: `fact_claim` and `fact_premium` share
    `policy_id` with no relationship declared between them, which is exactly how a chasm trap
    escapes schema metadata. The profile's `count(DISTINCT col)` is exact, so this is a fact about
    the data at profiling time, not an estimate.
    """
    facts: KeyFacts = {}
    for column in snapshot.columns:
        if column.row_count is None or column.distinct_count is None or column.null_count is None:
            continue
        facts[(column.object_id, column.name)] = (
            column.distinct_count == column.row_count - column.null_count
        )
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


def value_columns(e: exp.Expression | None) -> list[exp.Column]:
    """The columns an aggregate actually adds up: VALUE position only, never a condition.

    `SUM(CASE WHEN p.region = 'east' THEN c.amount ELSE 0 END)` sums `c`'s values; `p` appears
    only in the condition, and repeating `p` rows does not corrupt that sum. The prototype that
    counted every column inside the aggregate fired on 14.6% of BIRD's own gold for this shape.
    """
    return list(dict.fromkeys(col for term in _terms(e) for col in term))


def value_terms(e: exp.Expression | None) -> list[list[exp.Column]]:
    """An aggregate's argument split into additive terms, each as its value columns.

    A term is inflated only when EVERY table it reads is repeated. A table that is not repeated
    appears at most once per join row, so a term computed per row sums each of its rows once:
    `SUM(li.quantity * p.unit_price)` is right though `p` repeats per line item, and reading it as
    two columns refused the canonical revenue query. Columns share a term only through a product
    or a quotient; any other function passes its arguments' terms through, so
    `ROUND(li.quantity + p.unit_price, 2)` still sums `p.unit_price` on its own. Splitting only at
    the operators it knows would approve every wrapper it does not.
    """
    return [term for term in _terms(e) if term]


# Multiplying out is exponential in nesting depth; past this, a product falls back to reading
# each column as its own term, which refuses whatever the rule before terms refused.
_MAX_TERMS = 64


def _terms(e: exp.Expression | None) -> list[list[exp.Column]]:
    # A non-zero constant is the EMPTY term: dropped at the top, but under a product it leaves the
    # other factor alone -- `(li.quantity + 1) * p.unit_price` sums `p.unit_price` per line item.
    # Zero and NULL are no term at all, so `IIF(x, li.quantity, 0) * p.unit_price` is one term.
    if e is None or isinstance(e, exp.Null):
        return []
    if isinstance(e, exp.Literal):
        return [] if e.is_number and not str(e.this).strip("0.") else [[]]
    if isinstance(e, exp.Column):
        return [[e]]
    if isinstance(e, (exp.Add, exp.Sub)):
        return _terms(e.left) + _terms(e.right)
    if isinstance(e, exp.Mul):
        left, right = _terms(e.left), _terms(e.right)
        if len(left) * len(right) > _MAX_TERMS:
            return [[col] for col in dict.fromkeys(c for term in left + right for c in term)]
        return [a + b for a in left for b in right]
    if isinstance(e, exp.Div):  # 1 / (a + b) does not split: the denominator is one factor
        return [a + value_columns(e.right) for a in _terms(e.left)]
    if isinstance(e, exp.If):
        return _terms(e.args.get("true")) + _terms(e.args.get("false"))
    if isinstance(e, exp.Case):
        out: list[list[exp.Column]] = []
        for branch in e.args.get("ifs") or []:
            out += _terms(branch.args.get("true"))
        return out + _terms(e.args.get("default"))
    if isinstance(e, exp.Coalesce):  # it returns one of its arguments: each is a branch
        out = _terms(e.this)
        for arg in e.expressions:
            out += _terms(arg)
        return out
    # Any other node: its arguments' terms, side by side. An argument with no column in it -- a
    # precision, a CAST's type -- is a parameter, not a summand.
    out = []
    for child in e.iter_expressions():
        terms = _terms(child)
        if any(terms):
            out += terms
    return out or [[]]


def check_fanout(
    ast: exp.Expression,
    visible: dict[str, set[str]],
    facts: KeyFacts,
) -> Refusal | None:
    """Refuse an aggregate whose rows a join has multiplied (register M109).

    `FROM fact_claim JOIN fact_premium ON policy_id` with a SUM on each side repeats every claim
    once per premium row on its policy; the query runs, and the total is silently inflated. Per
    SELECT scope, over BASE tables only:

    1. collect equi-joins (JOIN ... ON, USING, WHERE a.x = b.y);
    2. ask `facts` whether each side's key is unique;
    3. a table is DUPLICATED when a walk outward from it enters another table by a key that
       repeats there;
    4. refuse SUM/AVG over a term whose tables are all duplicated (`value_terms`), or a
       count-like aggregate (COUNT(x), COUNT(*), a SUM of constants) only when EVERY table is
       duplicated -- a count over a plain one-to-many counts the finer table, which may be what
       was asked.

    Never fires on MIN, MAX, or any DISTINCT aggregate (immune to repetition), on a join whose
    key uniqueness is unknown, or through a CTE or subquery: a derived source is opaque and
    treated as unique, because an aggregated CTE is usually one row per key and guessing
    otherwise would refuse the very rewrite this asks for.
    """
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

    conditions: list[tuple[exp.Expression, str | None]] = []  # (condition, its ASOF right side)
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
            conditions.append((join.args["on"], asof))
        for ident in join.args.get("using") or []:
            if right not in sources:
                continue
            for left in earlier:
                if left in sources and _spelling(ident.name, visible.get(sources[left], ())):
                    add(left, ident.name, right, ident.name, asof)
        if right is not None:
            earlier.append(right)
    if select.args.get("where") is not None:
        conditions.append((select.args["where"].this, None))
    for condition, asof in conditions:
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

    adj: dict[str, list[str]] = {a: [] for a in sources}
    # a -> {b: b's key columns}: each `a` row meets MANY `b` rows, because b's key repeats
    fans_into: dict[str, dict[str, tuple[str, ...]]] = {a: {} for a in sources}
    for (a, b), cols in pairs.items():
        a_cols, b_cols = tuple(c[0] for c in cols), tuple(c[1] for c in cols)
        adj[a].append(b)
        adj[b].append(a)
        if _unique(facts, sources[b], b_cols) is False and (a, b) not in at_most_one:
            fans_into[a][b] = b_cols
        if _unique(facts, sources[a], a_cols) is False and (b, a) not in at_most_one:
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

    summed: dict[str, None] = {}  # duplicated aliases whose values are aggregated, in order
    chasm: exp.Expression | None = None
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
            owners = list(dict.fromkeys(owner(col) for col in term))
            if all(o in dup for o in owners):  # an unresolved owner (None) is never in `dup`
                summed.update(dict.fromkeys(owners))
    if summed:
        return _inflated(list(summed), sources, dup)
    if chasm is not None:
        return _counted_chasm(chasm, sources)
    return None


def _inflated(aliases: list[str], sources: dict[str, str], dup) -> Refusal:
    because = []
    for a in aliases:
        entered, key = dup[a]
        t, w = sources[a], sources[entered]
        because.append(f"each {t} row is repeated once per matching {w} row "
                       f"({w}.{', '.join(key)} is not unique)")
    tables = " and ".join(dict.fromkeys(sources[a] for a in aliases))
    return Refusal(
        code=RefusalCode.FAN_OUT,
        message=(
            "This query aggregates across a join that multiplies rows, so the result is "
            f"inflated: {'; '.join(because)}. Aggregate {tables} separately first -- one CTE or "
            "subquery per table, grouped by the key you join or group on -- then join those "
            "aggregated results instead of the raw tables."
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
