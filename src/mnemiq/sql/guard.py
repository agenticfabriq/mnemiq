from __future__ import annotations

import sqlglot
from sqlglot import exp

from mnemiq.sql.identifiers import resolve_name
from mnemiq.sql.qualify import object_key
from mnemiq.sql.verdict import Refusal, RefusalCode

MAX_ROWS = 1000

# Everything the engine is willing to run is one of these. Not a blocklist: a blocklist is a
# bet that you thought of every dangerous statement, and that bet is always lost eventually.
_ALLOWED_ROOTS = (exp.Select, exp.Union)


def check_shape(
    sql: str, dialect: str = "duckdb", max_rows: int = MAX_ROWS, executes_as: str | None = None,
    columns: dict[str, set[str]] | None = None,
) -> exp.Expression | Refusal:
    """Parse and enforce the shape of the query. Returns the AST, with a LIMIT guaranteed.

    The limit is *injected*, not requested: asking a model to add LIMIT is a request, and
    rewriting the tree is a guarantee.

    **Register M88.** `columns` is the schema -- table to column names, the same `visible` map
    `decide` hands `check_access` two lines later -- and supplying it ends a guess this file has
    been paying for in three findings. Is a bare `claim_amount` the COLUMN or the whole ROW?
    Without a schema that is unanswerable, so the rules fail closed and refuse legitimate queries
    (M84's `sum(claim_amount)`, M85's distinct-count, M88's `WHERE claim_amount > 10`). With one it
    is not a judgement call at all: DuckDB resolves the name to the COLUMN when one exists and to
    the row only when none does -- measured both ways -- so a name that is a known column is a
    column, and the row rules simply do not apply to it.

    Optional, and its absence changes nothing. A caller that cannot supply a schema keeps the
    fail-closed behaviour and the false refusals that come with it; loosening on a missing map
    would make forgetting to pass it a silent grant, which is the shape this whole file exists
    against.
    """
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception:
        return Refusal(
            code=RefusalCode.PARSE_ERROR,
            message="That is not valid SQL. Return a single SELECT statement.",
        )

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        # sqlglot.parse_one() would fold "SELECT 1; DROP TABLE t" into a Block and return it
        # without raising -- so we count statements ourselves.
        return Refusal(
            code=RefusalCode.NOT_A_SINGLE_STATEMENT,
            message="Return exactly one statement. Multiple statements are never executed.",
        )

    ast = statements[0]
    if not isinstance(ast, _ALLOWED_ROOTS):
        return Refusal(
            code=RefusalCode.NOT_SELECT_ONLY,
            message="Only SELECT queries are allowed. This engine never modifies data.",
            subject=type(ast).__name__.upper(),
        )

    # `executes_as`, not `dialect`. One is what we PARSE as and the other is what will RUN the
    # statement, and identifier folding belongs to the second: `decide` defaults to parsing duckdb
    # and targeting postgres, so keying the fold on the parse dialect answers for the wrong engine.
    # Production passes them equal, which is exactly why the mismatch would not have surfaced.
    if _has_projection_star(ast, executes_as or dialect, columns) \
            or _projects_a_row(ast, columns) \
            or _uses_a_row_outside_the_projection(ast, columns):
        return Refusal(
            code=RefusalCode.SELECT_STAR,
            message=(
                "Name the columns you need explicitly -- the engine must know which columns a "
                "query reads. That rules out `SELECT *` and `COLUMNS(...)`, and using a table's "
                "own name as a value anywhere in the statement, which means the whole row: "
                "`SELECT claim`, `WHERE claim[\'ssn\'] = ...`, `WHERE claim > ...`, `ORDER BY "
                "claim`. A field read like that never names the field as a column, so nothing "
                "can check whether you may read it. Name the column instead and qualify it: "
                "`FROM claim_amount c` then `WHERE c.claim_amount > 10`, or `SELECT c.ssn` -- a "
                "named column is allowed if your grants permit it, and refused by name if not."
            ),
        )

    return _with_limit(ast, max_rows)


def _output_selects(ast: exp.Expression) -> list[exp.Select]:
    """The SELECTs whose projections actually leave the engine.

    For a UNION that is every branch; for a plain query it is just the root. Inner SELECTs
    (subqueries, CTE bodies, EXISTS) do not return columns to the caller -- whatever they
    project is bounded by the explicit projection that wraps them.
    """
    if isinstance(ast, exp.Select):
        return [ast]
    # SetOperation, not Union: EXCEPT and INTERSECT are its SIBLINGS in sqlglot, not its
    # subclasses, so keying on Union read `... EXCEPT ...` as having no output projection at all.
    if isinstance(ast, exp.SetOperation):
        return [*_output_selects(ast.left), *_output_selects(ast.right)]
    if isinstance(ast, exp.Subquery):
        return _output_selects(ast.this)  # redundant parentheses nest Subquery in Subquery
    return []


def _is_star(projection: exp.Expression) -> bool:
    """`COUNT(*)` contains an `exp.Star` too, so this asks about the PROJECTION, not the tree.

    `exp.Columns` is the other spelling and it is not a subclass of `exp.Star`: DuckDB's
    `COLUMNS(*)` and `COLUMNS('regex')` expand to a column set the same way and parsed straight
    past a check that only knew `Star`. `SELECT COLUMNS(*) FROM claim` returned every column
    including a DENIED one, and `COLUMNS('s.*')` returned ONLY that column -- the denied value
    exfiltrated without its name ever appearing in the query.

    Searched rather than matched, because the expansion can sit inside another node and anything
    CONTAINING one returns an unbounded column set whatever wraps it -- `min(COLUMNS(*))` is still
    one output column per input column.

    Matched by NAME as well as by type, and that is not belt-and-braces: an unqualified
    `COLUMNS(*)` parses to `exp.Columns`, while a qualified `claim.COLUMNS(*)` parses to
    `Dot(Identifier, Anonymous(this="COLUMNS", ...))` -- no `exp.Columns` node anywhere in it. A
    type check alone therefore misses the qualified spelling.

    Searching for a bare `exp.Star` instead would be simpler and wrong: `COUNT(*)` contains one and
    collapses it to a single column, which is the most common analytics query there is. What is
    refused is the expansion, not the asterisk.
    """
    if isinstance(projection, exp.Star | exp.Columns):
        return True
    if isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
        return True
    for node in projection.walk():
        if isinstance(node, exp.Columns):
            return True
        if isinstance(node, exp.Anonymous) and str(node.this).upper() == "COLUMNS":
            return True
    return False


def _names_a_source(select: exp.Select, projection: exp.Expression,
                    columns: dict[str, set[str]] | None = None) -> exp.Expression | None:
    """The SOURCE this projection references as a whole row, or None if it references none.

    Keyed on naming a SOURCE, which is what separates it from the legitimate use -- `UNNEST(tags)`
    over a list column names a COLUMN, expands ROWS, and returns one column. And on the reference
    being BARE, since a qualified one names a column even where the column shares its table's name.
    The source name is folded because DuckDB folds: `SELECT CLAIM FROM claim` returns the struct.
    """
    sources = {name.lower(): source for name, source in _sources(select).items()}
    for column in projection.find_all(exp.Column):
        if column.table:
            continue  # qualified: it names a column
        source = _binds_to_a_row(column, columns)
        if source is not None and column.name.lower() in sources:
            return sources[column.name.lower()]
    return None


# Aggregates whose result carries NO VALUE out of their argument. That is the criterion, and it is
# narrower than it first looks -- an earlier wording said "result type cannot be the argument's",
# which is both wrong about these three and dangerous as a guide for extending the tuple. Measured
# on DuckDB against `claim(id, salary, ssn)`:
#
#   sum(claim)       -> Binder Error, no function matches      (no value escapes: it does not run)
#   avg(claim)       -> likewise
#   count(claim)     -> 2                                      (no value escapes: a cardinality)
#   max(claim)       -> the whole STRUCT, ssn included         <- LEAKS
#   array_agg(claim) -> a LIST of whole structs, ssn included  <- LEAKS
#
# `count` is the one that shows the old wording was wrong: it takes a struct without erroring at
# all, and is safe for the other reason -- it discards values and returns how many there were. And
# `array_agg`'s result type is `LIST(argument)`, which is not the argument's type, so the old
# criterion would have admitted the worst leak of the set.
#
# NOT `exp.AggFunc`, for the same reason: `max` and `min` are aggregates and both carry the row out
# through a projection that looks like one. Being an aggregate implies nothing here.
_COLLAPSING_AGGREGATES = (exp.Sum, exp.Avg, exp.Count)


def _every_bare_reference_is_aggregated(select: exp.Select, projection: exp.Expression, *,
                                        through_distinct: bool) -> bool:
    """The shared walk. `through_distinct` is the part each caller has to decide for itself.

    **Register M85.** sqlglot parses `count(DISTINCT claim_amount)` as `Count(this=Distinct(...))`,
    so a node sits between the column and the aggregate and a direct-parent test says no -- the M84
    deferral, one phrasing over, on what a model writes for "how many different claim amounts".

    Stepping over `Distinct` in a SHARED predicate was the first attempt and it widened the star
    walk too: `count(DISTINCT t) FROM (SELECT * FROM claim) t` stopped being examined at all. The
    two callers are asking different questions and can afford different answers. `_projects_a_row`
    asks whether a VALUE leaves, and `DISTINCT` changes nothing about that -- `count` still returns
    a cardinality. The star walk asks whether to LOOK at a projection at all, and a projection it
    declines to walk is one whose inner star nobody examines; there, anything short of certainty is
    a reason to keep walking.
    """
    sources = {name.lower() for name in _sources(select)}
    bare = [
        column for column in projection.find_all(exp.Column)
        if not column.table and column.name.lower() in sources
    ]
    if not bare:
        return False
    return all(
        isinstance(_past_distinct(column.parent) if through_distinct else column.parent,
                   _COLLAPSING_AGGREGATES)
        for column in bare
    )


def _past_distinct(node: exp.Expression | None) -> exp.Expression | None:
    """The node above, stepping over a `DISTINCT` wrapper and nothing else.

    Deliberately not a loop over "harmless" wrappers. Everything skipped has to be a node that
    cannot extract a field, and `Distinct` is the only one that qualifies; a general skip is how
    `Bracket` got walked past when this rule was first written.
    """
    return node.parent if isinstance(node, exp.Distinct) else node


def _projects_a_row(ast: exp.Expression,
                    columns: dict[str, set[str]] | None = None) -> bool:
    """Is a whole ROW projected as a value, anywhere in the statement?

    A third spelling of the same expansion, containing no star at all. DuckDB resolves a bare
    reference to a source NAME as the entire row: `SELECT claim FROM claim` returns a struct
    holding every value including a denied one, and `SELECT UNNEST(claim) FROM claim` spreads it
    back into columns. `check_cls` finds no `exp.Column` for the denied name in either, because the
    column is never spelled.

    EVERY select, not only the ones whose projections leave the engine. An explicit outer
    projection bounds the column COUNT and not the column SET once one item is a whole row, so
    `SELECT x FROM (SELECT claim AS x FROM claim) t` carried the struct out through a projection
    that looks entirely ordinary. Scanning every select is what reaches it: the derived table's
    body is itself a select, and it is there that the row is taken.

    A SUBQUERY source is skipped HERE and handled by the star walk instead, which recurses into
    that subquery's own projection with the CTE scopes already resolved. It is emphatically NOT
    covered by this scan: `_names_a_source` looks at `exp.Column` nodes, and a star or `COLUMNS(*)`
    inside the derived table yields none -- so `SELECT UNNEST(t) FROM (SELECT * FROM claim) t`
    passes this check entirely and is refused by the walk. Claiming the sibling scan covered it is
    how that shape briefly became permitted again.

    A TABLE source gets no exemption, whether or not it names a CTE. That refuses `SELECT c FROM c`
    over a CTE, whose struct is in fact bounded -- a deliberate false refusal, taken because the
    alternative is resolving CTE scopes a second way here, and a rarely-written shape is a better
    price than a second copy of a rule this file has already had wrong twice.
    """
    for select in ast.find_all(exp.Select):
        for projection in select.expressions:
            source = _names_a_source(select, projection, columns)
            if source is None or isinstance(source, exp.Subquery):
                continue
            if _every_bare_reference_is_aggregated(select, projection,
                                                   through_distinct=True):
                continue
            return True
    return False



def _binds_to_a_row(column: exp.Column, columns: dict[str, set[str]] | None) -> exp.Expression | None:
    """The source this bare name binds to as a WHOLE ROW, or None if it is a column.

    **Innermost scope first, and stopping there.** SQL resolves a bare name in the nearest scope
    that can answer it, and both directions of getting that wrong have now been measured:

    * unioning every column name in the statement let `other.claim` exempt the row reference in
      `SELECT claim FROM claim WHERE id IN (SELECT claim FROM other)`;
    * walking OUTWARD for columns let the same `other.claim` exempt the INNER row reference in
      `... FROM other WHERE EXISTS (SELECT 1 FROM claim WHERE claim['ssn'] = ...)`, where DuckDB
      binds `claim` to the inner table's struct.

    Both handed back a working extraction oracle past a `check_cls` that sees no `exp.Column` for
    the field. So each scope is asked in turn and the FIRST one that resolves the name decides: a
    column there means column, a source there means row, and only a scope that knows neither passes
    the question outward -- which is what keeps M86's correlated case refused.
    """
    lookup = (
        {key.lower(): {name.lower() for name in names} for key, names in columns.items()}
        if columns else {}
    )
    # A CTE reference is an `exp.Table`, so the schema lookup would hand it the columns of whatever
    # BASE table it shadows: `WITH other AS (SELECT id FROM other)` borrowed the granted `other`'s
    # column list, made a bare `claim` look like a column, and returned the whole `claim` struct
    # with a denied `ssn` in it. Matched without dialect folding on purpose -- over-matching a base
    # table that shares a CTE's name costs a refusal, and under-matching costs the row rule.
    root = column
    while root.parent is not None:
        root = root.parent
    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    name = column.name.lower()
    scope = column.parent
    while scope is not None:
        if isinstance(scope, exp.Select):
            sources = {key.lower(): source for key, source in _sources(scope).items()}
            # A column in THIS scope wins over a table of the same name in it -- which is what
            # DuckDB does, measured -- and a SUBQUERY source answers neither, so its columns stay
            # its own projection's business and cannot exempt anything.
            if any(
                isinstance(source, exp.Table)
                and source.name.lower() not in cte_names
                and name in lookup.get(object_key(source).lower(), frozenset())
                for source in sources.values()
            ):
                return None
            if name in sources:
                return sources[name]
        scope = scope.parent
    return None


def _uses_a_row_outside_the_projection(
    ast: exp.Expression, columns: dict[str, set[str]] | None = None
) -> bool:
    """Is a whole-row reference used anywhere but the projection?

    **Register M86.** `_projects_a_row` scans `select.expressions`, so WHERE, ORDER BY, GROUP BY
    and HAVING were unguarded. `check_cls` walks the whole statement but can only rule on columns
    that are SPELLED, and `claim['ssn']` yields `Column(claim)` with no `exp.Column` named `ssn` --
    so `policy.denies` is asked about `claim`, answers no, and the row count answers the predicate.
    A caller who may not read `ssn` recovers it one guess at a time.

    **Blanket, matching what the projection already does, after two narrower rules leaked.** The
    first refused subscripts, dots and calls on the reasoning that only those can reach a field.
    That is true of EXTRACTING a value and false about the threat: DuckDB compares structs
    field-by-field, so `WHERE claim > {'claim_identifier': 1, 'ssn': <guess>}` is a binary search
    over a denied column with no subscript, dot or call anywhere in it -- measured 1/0/0 across the
    real value. `ORDER BY claim` and `GROUP BY claim` leak the same way. Once comparison operators
    are dangerous too, what is left to permit is nothing, and the discriminating rule was only ever
    a list of the leaks that had been thought of.

    **Given a schema this costs nothing** (M88): `WHERE claim_amount > 10` is allowed, because
    `_binds_to_a_row` can see that `claim_amount` is a column of the table in scope. Without one it
    is refused, the same fail-closed price the projection rule pays, and the qualified spelling
    `WHERE c.claim_amount > 10` is the rewrite -- which the refusal message names.
    """
    for select in ast.find_all(exp.Select):
        for column in select.find_all(exp.Column):
            if column.table:
                continue  # qualified: it names a column, and `check_cls` can rule on it
            if _in_projection(column, select):
                continue  # `_projects_a_row` owns that clause, with its own exemptions
            # ONE lookup, which is also the scope rule. Asking `_binds_to_a_row` whether this is a
            # row and then a second function WHICH row is how two answers to one question get
            # into a file; this returns the source it bound to, or None for a column, a list-typed
            # name like `tags[1]`, or a name that resolves nowhere.
            source = _binds_to_a_row(column, columns)
            if source is None:
                continue
            # A DERIVED TABLE is bounded by its own projection, and the star walk decides whether
            # that projection reaches a base table -- the same division of labour `_projects_a_row`
            # keeps. Refusing here would forbid `UNNEST(t) FROM (SELECT id FROM claim) t`, which is
            # explicit and legitimate. A CTE reference is an `exp.Table` and gets no exemption, for
            # the reason `_projects_a_row` already gives: resolving CTE scopes a second way here is
            # a worse price than refusing a rarely-written shape.
            if isinstance(source, exp.Subquery):
                continue
            # M84: `sum` of a struct is a type error, so nothing escapes either way; M85: a
            # `DISTINCT` between the column and the aggregate changes neither. This is the
            # THIRD site making this judgement, and it kept deferring
            # `HAVING count(DISTINCT claim_amount) > 1` while the projection stopped.
            if isinstance(_past_distinct(column.parent), _COLLAPSING_AGGREGATES):
                continue
            return True
    return False


def _in_projection(column: exp.Expression, select: exp.Select) -> bool:
    """Is this column part of the SELECT list, rather than a clause hanging off it?"""
    node = column
    while node is not None and node is not select:
        if node.parent is select:
            return any(node is projection for projection in select.expressions)
        node = node.parent
    return False


def _sources(select: exp.Select) -> dict[str, exp.Expression]:
    """Everything the SELECT reads from, keyed by the name a star could be qualified with."""
    found: list[exp.Expression] = []
    # sqlglot spells this arg "from" in older versions and "from_" from v30 on. Reading the
    # wrong key silently yields no sources -- which would make every star look safe.
    from_ = select.args.get("from") or select.args.get("from_")
    if from_ is not None:
        found.append(from_.this)
    for join in select.args.get("joins") or []:
        found.append(join.this)
    return {source.alias_or_name: source for source in found}


# A CTE name resolves to its body AND to the scope that body was written in -- never to the
# scope it is read from. `Scope` is that pair, and pairing them is the whole point: a bare
# `name -> body` map has to borrow some caller's names to interpret the body, and the caller's
# names are not the ones the body was written against.
# Keyed by RESOLVED identifier, never by the name as typed -- `identifiers.resolve_name` owns that
# rule for every resolver in this package, after this file got it wrong twice in consecutive
# commits. Two spellings name one object or two
# depending on quoting, and getting that wrong in either direction has a cost: too wide and a CTE
# vouches for a base table, too narrow and ordinary SQL is refused.
Scope = dict[str, tuple[exp.Expression, "Scope"]]


def _visible_ctes(node: exp.Expression, outer: Scope, dialect: str) -> Scope:
    """The CTE names visible inside `node`, each bound to the scope ITS OWN body sees.

    Lexical, and lexical at the definition site. Two rules make that different from "the names
    around the reference", and both were live bypasses:

      * a non-recursive CTE cannot see a LATER sibling, so in
        `WITH a AS (SELECT * FROM claim), claim AS (...) SELECT * FROM a` the `claim` inside `a`
        is the base table, not the sibling defined after it;
      * an inner WITH at the reference site cannot rebind a name inside an outer CTE's body, so
        `WITH a AS (SELECT * FROM claim) SELECT * FROM (WITH claim AS (...) SELECT * FROM a) z`
        does not make `a`'s star explicit either.

    Both returned every column of `claim`, `ssn` included. Carrying one flat map -- or merging
    the caller's -- lets a projection vouch for a star it has no relationship to; `cls.py` records
    the same defect in the CLS resolver, where one alias meant two tables and the map kept
    whichever was seen last.
    """
    # "with" in older sqlglot, "with_" from v30 on -- the same rename `_sources` guards against
    # for "from". Reading only the old key made every CTE invisible: `source.name in ctes` was
    # never true, so a CTE source fell through to the base-table branch. That direction refuses
    # legitimate SQL rather than permitting a star, which is the only reason the tests caught it.
    with_ = node.args.get("with") or node.args.get("with_")
    if with_ is None:
        return outer

    recursive = bool(with_.args.get("recursive"))
    scope: Scope = dict(outer)
    for cte in with_.expressions:
        key = resolve_name(cte, dialect)
        # snapshot BEFORE binding this name: the body sees the outer scope and its preceding
        # siblings, plus itself only when the WITH is RECURSIVE
        body_scope: Scope = dict(scope)
        entry = (cte.this, body_scope)
        if recursive:
            body_scope[key] = entry
        scope[key] = entry
    return scope


def _star_reaches_base(node: exp.Expression, ctes: Scope, memo: dict[int, bool],
                       columns: dict[str, set[str]] | None,
                       stack: frozenset[int], dialect: str) -> bool:
    """Does this expression's own OUTPUT projection reach a base table's unbounded column set?

    The recursion is the point. "Its projection is explicit" is a claim about the derived table,
    and it is only true when that projection names columns -- `(SELECT a, b FROM t)` yes,
    `(SELECT * FROM t)` no. Answering it by looking one level down let a base-table star through
    inside a wrapper, so the column set was unbounded after all.

    Unreadable is not safe. A node `_output_selects` cannot decompose has no known projection,
    so it is treated as reaching a base table rather than as reaching nothing -- the same rule
    `_expands_a_base_table` already applies to sources it cannot see. Returning False there made
    every gap in that function a silent hole, which is how EXCEPT and doubled parentheses got
    through.

    `memo` keeps this linear, and is only sound BECAUSE the scopes above are lexical: each node
    then sits in exactly one scope, so its answer cannot depend on the path that reached it. While
    a CTE body was interpreted against its caller's names, the memo cached one caller's answer for
    all of them and two union branches gave opposite verdicts depending on their order. Without
    it, a CTE named twice per level is re-walked once per path -- 8s of pre-execution gate on
    1.2KB of model-written SQL.
    """
    if id(node) in stack:
        return False  # a recursive CTE names itself; the base case is the branch that ends
    if id(node) in memo:
        return memo[id(node)]

    selects = _output_selects(node)
    answer = True  # cannot be read -> cannot be vouched for
    if selects:
        inner, within = _visible_ctes(node, ctes, dialect), stack | {id(node)}
        answer = any(
            _expands_a_base_table(select, projection, inner, columns, memo, within, dialect)
            for select in selects
            for projection in select.expressions
            # A row reference joins the walk too: naming a DERIVED TABLE is bounded by that
            # table's projection exactly as a star over it is, and only this recursion can say
            # whether that projection reaches a base table.
            #
            # The aggregate exemption is applied to the ROW-REFERENCE half only, never to
            # `_is_star`: a star is a star whatever wraps it, and skipping the walk for one would
            # be the leak this guard was built for.
            #
            # It also stops at a SUBQUERY source, which is the whole difference between this caller
            # and `_projects_a_row`. Over a base table, `count(DISTINCT claim_amount)` carries no
            # value out and there is nothing here to examine (M85). Over a derived table the row
            # reference is a door onto that table's own projection, and declining to walk it is
            # declining to look at the star inside -- `count(DISTINCT t) FROM (SELECT * FROM claim)
            # t` is the shape, and a shared step-over let it through once already.
            if _is_star(projection) or (
                _skipped_row_source(select, projection, dialect, columns) is not None
            )
        )
    memo[id(node)] = answer
    return answer


def _expands_a_base_table(select: exp.Select, star: exp.Expression, ctes: Scope,
                          columns: dict[str, set[str]] | None, memo: dict[int, bool],
                          stack: frozenset[int], dialect: str) -> bool:
    """Would this star pull in the columns of a real source table?

    Over a base table the columns are unbounded: we would not know what we are returning, and
    column-level authorization would have nothing to check. Over a subquery or a CTE they are
    determined by that source's projection -- so that projection is what gets asked, rather than
    assumed. A wrapper does not make a star explicit, and treating it as if it did returned a
    DENIED column in full: `SELECT claim.ssn FROM claim` was refused `unauthorized_column` while
    `SELECT * FROM (SELECT * FROM claim) t` came back with the same values, masks and denials
    alike passing through untouched because neither `check_cls` nor `referenced_masked` sees an
    `exp.Column` for a star.
    """
    sources = _sources(select)
    if not sources:
        return True  # a star over sources we cannot see is not a star we can vouch for

    qualifier = star.table if isinstance(star, exp.Column) else None
    candidates = (
        [sources[qualifier]] if qualifier and qualifier in sources else list(sources.values())
    )

    for source in candidates:
        if isinstance(source, exp.Subquery):
            if _star_reaches_base(source.this, ctes, memo, columns, stack, dialect):
                return True
            continue  # derived table whose projection really is explicit
        # A CTE reference is always a BARE name. Matching on `source.name` alone let a
        # schema-qualified base table be vouched for by an unrelated CTE that merely shares it:
        # `WITH claim AS (SELECT id FROM policy) SELECT * FROM main.claim` resolves to the CTE
        # here and to the base table in the database, which returned every column of `claim`.
        if isinstance(source, exp.Table) and not source.db and not source.catalog \
                and resolve_name(source, dialect) in ctes:
            body, body_scope = ctes[resolve_name(source, dialect)]
            # `body_scope`, not `ctes`: the body is interpreted where it was written
            if _star_reaches_base(body, body_scope, memo, columns, stack, dialect):
                return True
            continue  # a CTE: same
        return True
    return False


def _skipped_row_source(select: exp.Select, projection: exp.Expression,
                        dialect: str,
                        columns: dict[str, set[str]] | None) -> exp.Expression | None:
    """A row source this projection names that the star walk must examine, or None.

    None means "nothing here for the walk": no bare source reference at all, or every one of them
    aggregated into a scalar over a BASE table. A subquery source is never None, because the walk is
    what reads that subquery's own projection.

    **Every source it names, not the first.** `_names_a_source` returns the first match, so asking
    it alone let `count(DISTINCT claim_amount) + count(DISTINCT t) FROM claim_amount, (SELECT *
    FROM claim) t` resolve to the base table, take the exemption, and leave the star inside the
    derived table unwalked -- the shape one addend over from the one this exemption is pinned not
    to open.
    """
    sources = {name.lower(): source for name, source in _sources(select).items()}
    named = [
        sources[column.name.lower()]
        for column in projection.find_all(exp.Column)
        if not column.table and _binds_to_a_row(column, columns) is not None
        and column.name.lower() in sources
    ]
    if not named:
        return None
    # A CTE reference is an `exp.Table`, so matching `exp.Subquery` alone missed the CTE spelling
    # of exactly the shape above: `WITH c AS (SELECT * FROM claim) SELECT count(DISTINCT c) FROM c`
    # took the exemption and left that star unwalked.
    #
    # Resolved through `resolve_name`, and on the table's NAME rather than `alias_or_name`, which is
    # the ALIAS when the reference is aliased -- `FROM c AS x` answers `x`, misses a CTE named `c`,
    # and reopens the same gap one spelling over. `_expands_a_base_table` resolves identifiers the
    # same way, which is the part that has to agree.
    #
    # It is NOT the same question otherwise, deliberately: this matches every CTE in the statement
    # while that one matches the lexically visible scope. Being wrong here costs a walk that was
    # not needed; being wrong there costs a star nobody examined. So this one over-approximates and
    # says so rather than pretending the two are one rule.
    root = select
    while root.parent is not None:
        root = root.parent
    cte_names = {
        name for name in (resolve_name(cte, dialect) for cte in root.find_all(exp.CTE)) if name
    }
    for source in named:
        if isinstance(source, exp.Subquery):
            return source
        if isinstance(source, exp.Table) and resolve_name(source, dialect) in cte_names:
            return source
    return None if _every_bare_reference_is_aggregated(
        select, projection, through_distinct=True
    ) else named[0]


def _has_projection_star(ast: exp.Expression, dialect: str,
                         columns: dict[str, set[str]] | None = None) -> bool:
    """A star that would return columns we cannot name.

    Three things this deliberately does not reject -- a guard that refuses legitimate
    analytics SQL is broken, not safe (all 99 TPC-DS queries must pass it):

    - `COUNT(*)` contains an exp.Star, so a naive find_all(exp.Star) would refuse `SELECT
      count(*) FROM claim`, the most common analytics query there is.
    - An *inner* star returns nothing to the caller: `EXISTS (SELECT * FROM t)` discards its
      projection, and `SELECT a FROM (SELECT * FROM t)` still returns exactly `a`.
    - A star over a *derived table or CTE* expands an explicit projection, so the output
      columns are known: `SELECT * FROM (SELECT a, b FROM t)` returns exactly a and b.

    What is refused is a star that expands a **base table**, directly or through a wrapper the
    walker can follow -- there the column set is unbounded, and neither we nor column-level
    authorization can say what comes back. The third exemption above is the narrow one: it holds
    when the derived table or CTE projects named columns, and not when it projects a star of its
    own. "A wrapper the walker can follow" is the honest limit rather than a universal: what it
    follows is what `_output_selects` decomposes, and a shape it cannot read is refused rather
    than waved through, so THAT gap costs a false refusal instead of a leak.

    Name resolution is the other half, and there a WIDER match vouches for a star rather than
    refusing it, so the two rules that could widen one are asserted rather than assumed: a CTE
    reference is a bare name, and a body is read in the scope it was written in.

    A CTE name matches on what the identifier RESOLVES to, which is the rule `qualify.py` already
    states for these engines: unquoted is case-insensitive, quoted is preserved. Matching the name
    as typed is wrong in both directions and both were live. Too wide:
    `WITH "Claim" AS (...) SELECT * FROM Claim` matched, while Postgres -- the default transpile
    target -- folds the reference to `claim` and resolves it to the base table, whose columns then
    reached no grant check at all; `decide` approved exactly that with `tables=['policy']`. Too
    narrow: requiring identical quoting refused `WITH "claim" AS (...) SELECT * FROM claim`, which
    is one object in every DOWN-folding engine here -- Oracle is the exception and the paragraph
    below is why -- and is the ordinary shape of a model quoting a definition but not its
    reference.

    The fold direction is the EXECUTING dialect's, because Oracle folds unquoted names up where
    the others fold them down -- one constant is wrong for one of them in the direction that
    vouches.

    The fold itself belongs to `identifiers.resolve_name`, which knows both rules per dialect
    -- so the false refusal this paragraph used to record, a quoted CTE against an unquoted
    reference on DuckDB, is gone: DuckDB folds quoted names and the resolver now says so.
"""
    # The top level keeps its own fail-open reading: a statement whose shape `_output_selects`
    # does not recognise returns no columns to a caller here, and refusing every such statement
    # would refuse writes and DDL this function is not the gate for. Fail-CLOSED starts one level
    # down, where the question is whether a wrapper can be vouched for.
    if not _output_selects(ast):
        return False
    return _star_reaches_base(ast, {}, {}, columns, frozenset(), dialect)


def _with_limit(ast: exp.Expression, max_rows: int) -> exp.Expression:
    existing = ast.args.get("limit")
    if existing is None:
        return ast.limit(max_rows)

    try:
        requested = int(existing.expression.name)
    except (AttributeError, ValueError):
        return ast.limit(max_rows)  # unreadable limit -> impose our own

    return ast if requested <= max_rows else ast.limit(max_rows)
