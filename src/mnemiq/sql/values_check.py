from __future__ import annotations

from collections.abc import Callable

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope

from mnemiq.semantic.values import shown
from mnemiq.sql.qualify import names_one_object
from mnemiq.sql.verdict import Refusal, RefusalCode


def _column_and_string(a: exp.Expression, b: exp.Expression) -> tuple[exp.Column, str] | None:
    """If one side is a column and the other a string literal, return (column, literal_text)."""
    for column, literal in ((a, b), (b, a)):
        if isinstance(column, exp.Column) and isinstance(literal, exp.Literal) and literal.is_string:
            return column, literal.this
    return None


def _scopes(ast: exp.Expression) -> dict[int, Scope]:
    """Each column node's own scope, by identity: the INNERMOST that lists it, since sqlglot
    lists an inner subquery's unqualified column in the outer scope as well, and the outer
    one read `doc` in `EXISTS (SELECT 1 FROM u WHERE doc = 'z')` among t's sources and refused
    the right literal naming `t.doc` (the review gate's third round on M174; `traverse_scope`
    yields the inner scopes first). Empty where sqlglot cannot build the scopes, which leaves
    every column to the statement-wide resolution below, as before M141."""
    try:
        out: dict[int, Scope] = {}
        for scope in traverse_scope(ast):
            for c in scope.columns:
                out.setdefault(id(c), scope)
        return out
    except (SqlglotError, RecursionError):
        return {}


def _is_star(projection: exp.Expression) -> bool:
    """A star that passes the source's columns through as they are: one with an EXCLUDE, a
    REPLACE or a RENAME changes what a name reaches (`* REPLACE (upper(doc) AS doc)` is no
    longer `t.doc`, `* RENAME (region AS doc)` is `t.region`), and the walk gives up on it
    rather than guess (the review gate's fourth round on M174)."""
    star = projection.this if isinstance(projection, exp.Column) else projection
    return isinstance(star, exp.Star) and not any(star.args.get(k) for k in ("except_", "replace", "rename"))


def _column_list(source: Scope) -> list[str] | None:
    """The column list a CTE's or a derived table's alias puts on the scope's output, by
    position (`WITH x(region, doc) AS (...)`, `(...) d(a, b)`), or None where it has none."""
    holder = source.expression.parent
    while isinstance(holder, exp.Paren):
        holder = holder.parent
    if not isinstance(holder, (exp.CTE, exp.Subquery)):
        return None
    alias = holder.args.get("alias")
    columns = alias.columns if isinstance(alias, exp.TableAlias) else None
    return [c.name for c in columns] if columns else None


def _folded(name: str) -> str:
    """Names meet as the engine folds an unquoted one: Postgres reads `x.CURRENCY_CODE` as
    `currency_code` where the walk compared exactly and answered nothing, a skip (the ui
    lane's probe of #178). A quoted mixed-case name beside its folded twin is the one shape
    this misreads, and the grant check refuses an unknown spelling on a base table first."""
    return name.lower()


def _through_scopes(column: exp.Column, scope: Scope, visible: dict[str, set[str]],
                    depth: int = 0) -> tuple[str, str] | None:
    """The base table and column a qualified column reads, followed through each CTE or derived
    table that projects it (M141). Only a projection that IS a column is followed, renamed or
    not; a function, a literal or a union has no single base column, so None, and the check
    skips. The scope's own answer says whether the qualifier is a base table or a local source:
    a CTE named after the base table it reads (`WITH t AS (SELECT doc FROM t)`) was discarded by
    a statement-wide CTE-name set, and a wrong literal against it passed (M174, Codex's review
    of #128). A column list renames by POSITION: `WITH x(region, doc) AS (SELECT doc, region
    FROM t)` makes `x.doc` the second projection, `t.region`, where matching by name checked it
    against `t.doc`'s values (the same finding); a star among the listed positions yields a
    column per column of its source, so the positions are unknown and the walk gives up (the
    review gate's fifth round). A projection without a qualifier is followed
    into the scope's one SELECTED source, not its one visible definition: three chained CTEs
    count every earlier CTE among `sources` and skipped the walk (the same finding). A name no
    projection carries comes through a star, into the scope's one selected source the same
    way (the review gate: an unqualified `doc` through `SELECT * FROM t` lost its owner once
    the statement-wide lookup became the fallback), a star with an EXCLUDE, a REPLACE or a
    RENAME excepted, which the walk gives up on (`_is_star`). The projections are read in
    order, as the engine names the output: a plain star BEFORE a projection of the asked name
    yields that name first where its source has the column, and the later projection is
    renamed (`WITH t AS (SELECT *, region AS doc FROM t)` keeps the star's `doc`, DuckDB's
    `doc_1` for the other: Codex's first review of #178, where the walk matched the alias by
    name and checked `t.region`); a star whose source the walk cannot read, or a modified one,
    before the name answers nothing. A base table under an alias column list (`FROM t
    q(region, doc)`) renames its columns by position, which a set of names cannot place, so
    the walk answers nothing there (the same review: `q.doc` is `t.region`, and the walk said
    `t.doc`). Names meet folded (`_folded`). Where this answers nothing, the caller keeps the
    statement-wide reading it had: a star over a join, a set operation, a recursive CTE and a
    correlated name each resolved there and nowhere here (the gate's second round)."""
    source = scope.sources.get(column.table) if column.table else None
    if isinstance(source, exp.Table):
        alias = source.args.get("alias")
        if isinstance(alias, exp.TableAlias) and alias.columns:
            return None  # renamed by position: a set of names cannot place them
        stored = next((c for c in visible.get(source.name, ()) if _folded(c) == _folded(column.name)), None)
        return (source.name, stored if stored is not None else column.name)
    if not isinstance(source, Scope) or depth > 32:
        return None
    if not isinstance(source.expression, exp.Select):
        return None
    selects = source.expression.selects
    listed = [_folded(name) for name in _column_list(source) or []]
    wanted = _folded(column.name)
    if any(isinstance(p.this if isinstance(p, exp.Column) else p, exp.Star) for p in selects[:len(listed)]):
        return None  # a star among the listed positions: a column per source column, positions unknown
    if wanted in listed[:len(selects)]:
        projection = selects[listed.index(wanted)]  # renamed by position
    else:
        # A list names the first projections alone and the engine keeps the rest by their own
        # names (`WITH x(k) AS (SELECT region, doc FROM t)` keeps `doc`: the review gate's
        # fourth round), so the projections past the list are matched by name, in order, a
        # star met first followed into the scope's one selected source.
        projection = None
        for candidate in selects[len(listed):]:
            star = candidate.this if isinstance(candidate, exp.Column) else candidate
            if isinstance(star, exp.Star):
                if not _is_star(candidate) or len(source.selected_sources) != 1:
                    return None
                (alias,) = source.selected_sources
                through = _through_scopes(exp.column(column.name, table=alias), source, visible, depth + 1)
                if through is None:
                    return None  # the star may yield the name: not known either way
                table, name = through
                if any(_folded(c) == _folded(name) for c in visible.get(table, ())):
                    return through  # the star yields the name first
                continue  # the star's source lacks it: the name is a later projection's
            if _folded(candidate.alias_or_name) == wanted:
                projection = candidate
                break
        if projection is None:
            return None
    inner = projection.unalias()
    if isinstance(inner, exp.Column) and inner.table:
        return _through_scopes(inner, source, visible, depth + 1)
    if isinstance(inner, exp.Column) and len(source.selected_sources) == 1:
        (alias,) = source.selected_sources
        return _through_scopes(exp.column(inner.name, table=alias), source, visible, depth + 1)
    return None


def check_values(
    ast: exp.Expression,
    visible: dict[str, set[str]],
    values,
    row_filtered: set[str] | None = None,
    masks: Callable[[str, str], bool] | None = None,
) -> Refusal | None:
    """Flag an equality/IN filter whose string literal does not exist in its indexed column.

    Fires only where the column resolves unambiguously to an indexed base column -- a bounded,
    non-key, non-PII string column whose full value set we hold. Anything we cannot resolve or
    have not indexed is skipped: the check never guesses, so a 'not found' is genuine.

    `row_filtered` names the tables this identity sees through a row filter. **The value index is
    built once per source with no predicate** (`semantic/values.py`), and this check runs BEFORE the
    RLS rewrite -- so listing the real values of a row-filtered table let an identity enumerate rows
    it cannot read, without executing anything (M5). For those tables the refusal still fires and
    still names the wrong literal; it just stops quoting the data, and says why.

    `masks` says whether a base column is masked for this identity, and such a column is not
    checked at all. `check_cls` refuses one used in a predicate directly, but not through a CTE
    or a derived table, which this check follows (M141) -- so without this it listed a masked
    column's raw values. Skipped rather than refused without values: "does not appear" alone
    answers whether a raw value exists, which the mask exists to withhold.
    """
    row_filtered = row_filtered or set()
    # The statement-wide reading, the fallback where the scope answers nothing: every table
    # node, a CTE's name excluded by a statement-wide set, as before M141.
    cte = {c.alias_or_name for c in ast.find_all(exp.CTE)}
    alias_to_table: dict[str, str] = {}
    for table in ast.find_all(exp.Table):
        alias = table.args.get("alias")
        if isinstance(alias, exp.TableAlias) and alias.columns:
            continue  # renamed by position (`t q(region, doc)`): a set of names cannot place them
        if table.name not in cte:
            alias_to_table[table.alias_or_name] = table.name
    referenced = set(alias_to_table.values())
    scopes = _scopes(ast)

    def resolve(column: exp.Column) -> tuple[str, str] | None:
        qualifier = column.table
        scope = scopes.get(id(column))
        if scope is not None:
            # The scope's own answer first (M141, M174): a base table, or a local source
            # followed to the column it projects, the column renamed included -- `x.doc` may be
            # `t.region`, and read by its name it would be checked against `t.doc`'s values.
            # An unqualified name is read among the scope's selected sources, a base table
            # holding it or a local source projecting it, one owner or no answer.
            if qualifier:
                found = _through_scopes(column, scope, visible)
                if found is not None:
                    return found
                if isinstance(scope.sources.get(qualifier), Scope):
                    return None  # a local source projecting no such column: nothing to check
            else:
                owners: list[tuple[str, str]] = []
                for alias, (_, source) in scope.selected_sources.items():
                    if isinstance(source, exp.Table):
                        found = _through_scopes(exp.column(column.name, table=alias), scope, visible)
                        if found is not None and any(_folded(c) == _folded(column.name) for c in visible.get(source.name, ())):
                            owners.append(found)
                    else:
                        found = _through_scopes(exp.column(column.name, table=alias), scope, visible)
                        if found is not None:
                            owners.append(found)
                if len(owners) == 1:
                    return owners[0]
        # The statement-wide reading, as before M174, where the scope answers nothing: a star
        # over a join, a set operation, a recursive CTE, a correlated name, a column sqlglot
        # lists in no scope (the review gate's second round, each refused here and skipped by
        # the scope alone).
        if qualifier:
            table = alias_to_table.get(qualifier)
            return (table, column.name) if table else None
        owners_wide = [t for t in referenced if column.name in visible.get(t, set())]
        return (owners_wide[0], column.name) if len(owners_wide) == 1 else None  # ambiguous -> skip

    def violation(column: exp.Column, literal: str) -> Refusal | None:
        resolved = resolve(column)
        if resolved is None:
            return None
        table, name = resolved
        if masks is not None and masks(table, name):
            return None
        if not values.has(table, name) or values.contains(table, name, literal):
            return None
        # FOLDED, and folding is safe here for the reason it is not safe for the filter
        # lookup itself: this set decides whether to WITHHOLD, so matching more of it discloses
        # less. Measured before this, with `row_filters` keyed `CLAIM` against a snapshot keyed
        # `claim` -- a pairing the policy fold now BINDS -- `check_values` answered "the real
        # values include: west, east", naming a value from rows the caller cannot read. The
        # exact-keyed control withheld it. So the fold that made the filter reach the query left
        # this branch still believing the table was unfiltered, and turned M5 back on.
        if names_one_object(table, row_filtered):
            return Refusal(
                code=RefusalCode.VALUE_GROUNDING,
                message=(
                    f"the value {literal!r} does not appear in {table}.{name}. "
                    "The valid values are not listed because your access to this table is "
                    "row-filtered and the value index is not."
                ),
            )
        near = values.nearest(table, name, literal)
        return Refusal(
            code=RefusalCode.VALUE_GROUNDING,
            message=(
                f"the value {literal!r} does not appear in {table}.{name}; "
                f"the real values include: {', '.join(shown(v) for v in near)}. Use the intended one."
            ),
        )

    for eq in ast.find_all(exp.EQ):
        pair = _column_and_string(eq.left, eq.right)
        if pair is not None:
            found = violation(*pair)
            if found is not None:
                return found

    for member in ast.find_all(exp.In):
        column = member.this
        if not isinstance(column, exp.Column):
            continue
        for item in member.expressions:
            if isinstance(item, exp.Literal) and item.is_string:
                found = violation(column, item.this)
                if found is not None:
                    return found

    return None
