from __future__ import annotations

from collections.abc import Callable

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope

from mnemiq.semantic.values import shown
from mnemiq.sql.qualify import names_one_object
from mnemiq.sql.identifiers import resolve, resolve_identifier, resolve_stored
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


def _column_list(source: Scope, dialect: str | None) -> list[str] | None:
    """The column list a CTE's or a derived table's alias puts on the scope's output, by
    position (`WITH x(region, doc) AS (...)`, `(...) d(a, b)`), each name as the engine
    resolves it, or None where it has none."""
    holder = source.expression.parent
    while isinstance(holder, exp.Paren):
        holder = holder.parent
    if not isinstance(holder, (exp.CTE, exp.Subquery)):
        return None
    alias = holder.args.get("alias")
    columns = alias.columns if isinstance(alias, exp.TableAlias) else None
    return [_written(c, dialect) for c in columns] if columns else None


def _written(node: exp.Expression | None, dialect: str | None) -> str:
    """A reference's name as the engine resolves it: by its quoting under the dialect's rules,
    the shared resolver (M55). `.lower()` read Postgres's quoted `"CURRENCY_CODE"` as
    `currency_code`, a different column there, and refused, missed or named the wrong column
    on nine rows the engine answers right (the ui lane's probe of #178 at `20753d6`), where
    DuckDB folds a quoted name too and was right under the fold."""
    return resolve_identifier(node, dialect) or (node.name if node is not None else "")


def _stored(name: str, dialect: str | None) -> set[str]:
    """The spellings a catalog name meets a reference under: as created, and as the engine folds
    an unquoted reference (`resolve_stored`), since a snapshot spells a column as its author
    did (`Country` in a BIRD world decided for SQLite, `region` for Oracle, whose catalog
    would hold `REGION`): the grant check's reading (M170), which withholds more and never
    less; a reference's own quoting is kept (`_written`)."""
    return {name, resolve_stored(name, dialect)}


def _stored_match(column: exp.Column, stored: set[str] | frozenset[str], dialect: str | None) -> str | None:
    """The catalog spelling a reference meets, the exact one first: a catalog holding both
    `"Amount"` and `amount` meets an unquoted `amount` by the latter, where a set's first
    match depended on its order (the review gate on the fourth round)."""
    written = _written(column.this, dialect)
    if written in stored:
        return written
    return next((c for c in sorted(stored) if written in _stored(c, dialect)), None)


def _projects_a_star(projection: exp.Expression) -> bool:
    return isinstance(projection.this if isinstance(projection, exp.Column) else projection, exp.Star)


def _names(source: Scope, wanted: str, dialect: str | None, visible: dict[str, set[str]],
           depth: int = 0) -> bool | None:
    """Whether a local source names the column in its own output, as the engine names it: True,
    False, or None where the walk cannot tell. A column list names the first projections by
    position (`WITH x(doc) AS (SELECT upper(region) FROM t)` has `doc`, the projection's own
    name never meeting the column: the review gate's fifth round) and renames those away
    (`WITH x(k) AS (SELECT doc FROM t)` has no `doc`); a projection past the list keeps its own
    name. A star among the listed positions yields a column per source column, so the positions
    are unknown, and only a listed name or a named projection from the star on answers (the
    gate's seventh round: `WITH x(k, m) AS (SELECT region AS doc, * FROM ...)`), otherwise the
    walk cannot tell. A star past the list is followed into the source's one selected source,
    in order: a base table's granted columns say, a local source this again, and a modified
    star, a star over a join or a base table renamed by position leave the answer unknown.
    Unknown, the source may yield the name and the engine binds there, so the caller keeps the
    statement-wide reading for the scope rather than ask the enclosing one (the review gate on
    the ninth round: `EXISTS (SELECT 1 FROM (SELECT * FROM u JOIN v ON true) y WHERE doc =
    'west')` with `v` holding `doc` binds to `y`, and `(SELECT * FROM (SELECT doc || '' AS doc
    FROM v) z) y` names `doc` through the star as an expression; asked of the enclosing scope,
    each refused a present value against `t.doc`)."""
    if depth > 32 or not isinstance(source.expression, exp.Select):
        return None
    selects = source.expression.selects
    listed = _column_list(source, dialect) or []
    first_star = next((i for i, p in enumerate(selects[:len(listed)]) if _projects_a_star(p)), None)
    if first_star is not None:
        if wanted in listed or any(_projection_name(p, dialect) == wanted
                                   for p in selects[first_star:] if not _projects_a_star(p)):
            return True
        return None
    if wanted in listed[:len(selects)]:
        return True
    for projection in selects[len(listed):]:
        if not _projects_a_star(projection):
            if _projection_name(projection, dialect) == wanted:
                return True
            continue
        if not _is_star(projection) or len(source.selected_sources) != 1:
            return None
        (alias,) = source.selected_sources
        _, inner = source.selected_sources[alias]
        if isinstance(inner, exp.Table):
            renamed = inner.args.get("alias")
            if isinstance(renamed, exp.TableAlias) and renamed.columns:
                return None
            if any(wanted in _stored(c, dialect) for c in visible.get(inner.name, ())):
                return True
            continue
        if not isinstance(inner, Scope):
            return None
        through = _names(inner, wanted, dialect, visible, depth + 1)
        if through is not False:
            return through
    return False


def _may_hold(source: exp.Expression | Scope, wanted: str, dialect: str | None,
              visible: dict[str, set[str]]) -> bool:
    """Whether a source can hold the column in its output: a base table by its granted columns,
    one renamed by position by any, a local source by `_names`, holding unless it answers no."""
    if isinstance(source, exp.Table):
        renamed = source.args.get("alias")
        if isinstance(renamed, exp.TableAlias) and renamed.columns:
            return True
        return any(wanted in _stored(c, dialect) for c in visible.get(source.name, ()))
    return not isinstance(source, Scope) or _names(source, wanted, dialect, visible) is not False


def _same_column(column: exp.Column, alias: str) -> exp.Column:
    """The column's name, quoting kept, under another qualifier: rebuilt by `exp.column(name,
    table=...)` it was unquoted, which the engine folds where the written one it does not."""
    return exp.Column(this=column.this.copy(), table=exp.to_identifier(alias))


def _projection_name(projection: exp.Expression, dialect: str | None) -> str:
    """The name a projection puts on the output, as the engine resolves it."""
    if isinstance(projection, exp.Alias):
        return _written(projection.args.get("alias"), dialect)
    if isinstance(projection, exp.Column):
        return _written(projection.this, dialect)
    return resolve(projection.alias_or_name, quoted=False, dialect=dialect)


def _through_scopes(column: exp.Column, scope: Scope, visible: dict[str, set[str]],
                    dialect: str | None, depth: int = 0) -> tuple[str, str] | None:
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
    `t.doc`). Names meet as the engine resolves them (`_written`, `_stored`). Where this answers nothing, the caller asks the
    enclosing scopes, then keeps the statement-wide reading it had: a star over a join, a set
    operation and a recursive CTE each resolved there and nowhere here (the gate's second
    round)."""
    source = scope.sources.get(column.table) if column.table else None
    if isinstance(source, exp.Table):
        alias = source.args.get("alias")
        if isinstance(alias, exp.TableAlias) and alias.columns:
            return None  # renamed by position: a set of names cannot place them
        stored = _stored_match(column, visible.get(source.name, ()), dialect)
        return (source.name, stored if stored is not None else column.name)
    if not isinstance(source, Scope) or depth > 32:
        return None
    if not isinstance(source.expression, exp.Select):
        return None
    selects = source.expression.selects
    listed = _column_list(source, dialect) or []
    wanted = _written(column.this, dialect)
    if any(_projects_a_star(p) for p in selects[:len(listed)]):
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
                through = _through_scopes(_same_column(column, alias), source, visible, dialect, depth + 1)
                if through is None:
                    return None  # the star may yield the name: not known either way
                table, name = through
                # A local source answers only where it projects the name; a base table answers
                # for any name, so its columns say whether the star yields it (the review gate
                # on the second round: held to the grant, a name a CTE projects from a column
                # outside it fell to the later projection).
                if not isinstance(source.sources.get(alias), exp.Table) or any(_stored(c, dialect) & _stored(name, dialect) for c in visible.get(table, ())):
                    return through  # the star yields the name first
                continue  # the star's source lacks it: the name is a later projection's
            if _projection_name(candidate, dialect) == wanted:
                projection = candidate
                break
        if projection is None:
            return None
    inner = projection.unalias()
    if isinstance(inner, exp.Column) and inner.table:
        return _through_scopes(inner, source, visible, dialect, depth + 1)
    if isinstance(inner, exp.Column) and len(source.selected_sources) == 1:
        (alias,) = source.selected_sources
        return _through_scopes(_same_column(inner, alias), source, visible, dialect, depth + 1)
    return None


def check_values(
    ast: exp.Expression,
    visible: dict[str, set[str]],
    values,
    row_filtered: set[str] | None = None,
    masks: Callable[[str, str], bool] | None = None,
    dialect: str | None = None,
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
    _ABSENT = object()  # no source of the scope can hold the name: the enclosing scope's turn
    _WIDE = object()  # the scope cannot tell: the statement-wide reading's turn, as before

    def in_scope(column: exp.Column, scope: Scope) -> tuple[str, str] | None | object:
        """The scope's own answer (M141, M174): a base table, or a local source followed to the
        column it projects, the column renamed included -- `x.doc` may be `t.region`, and read
        by its name it would be checked against `t.doc`'s values. An unqualified name is read
        among the scope's selected sources, a base table holding it or a local source
        projecting it, one owner or no answer. None is a skip; `_ABSENT` says no source of this
        scope can hold the name, and the enclosing scope is asked; `_WIDE` says the scope
        cannot tell, a source's output being one the walk cannot name (`_names`) or two
        sources owning the name, and the statement-wide reading answers as before."""
        qualifier = column.table
        if qualifier:
            return _through_scopes(column, scope, visible, dialect) if qualifier in scope.sources else _ABSENT
        wanted = _written(column.this, dialect)
        # The merged column of a USING join is the join's: an inner or LEFT join's the
        # left input's, a RIGHT join's the right input's, a FULL join's coalesced from
        # both (measured by the ui lane on Postgres and DuckDB). `SELECT doc FROM t
        # RIGHT JOIN t q(region, doc) USING (doc)` compares `q.doc`, `t.region`, where
        # the walk discarded q and took `t.doc` as the sole owner (Codex's second
        # review of #178); a FULL join's the same with the literal on q's side (the ui
        # lane). The candidates are the join's side; a source renamed by position
        # among them answers nothing, and so does a FULL join.
        candidates = list(scope.selected_sources)
        # This scope's own joins alone, in order, each USING the key narrowing the
        # candidates in turn (a later RIGHT join decides over an earlier inner one): a
        # join inside a subquery is another scope's, and read here it switched the check
        # off for the outer column (the review gate on the fourth round). A NATURAL
        # join's keys are every name its sides share, which a set of granted names
        # cannot settle, so the name answers nothing where a source holds or may hold it
        # (Codex's third review of #178: `t NATURAL RIGHT JOIN (SELECT region AS doc FROM
        # t) q` compares `t.region`); beside sources none of which can hold the name the
        # join is no part of its binding, and the enclosing scope is asked (Codex's seventh
        # review: `EXISTS (SELECT 1 FROM w NATURAL JOIN w b WHERE doc = 'west')` compares
        # the outer `t.doc`, and the skip let an absent value through where the
        # statement-wide reading had refused it).
        for join in scope.expression.args.get("joins") or []:
            if (join.method or "").upper() == "NATURAL":
                return None if any(_may_hold(source, wanted, dialect, visible)
                                   for _, source in scope.selected_sources.values()) else _ABSENT
            if not any(_written(key, dialect) == wanted for key in (join.args.get("using") or ())):
                continue
            side = (join.side or "").upper()
            if side == "FULL":
                return None
            right = join.this.alias_or_name
            candidates = [right] if side == "RIGHT" else [a for a in candidates if a != right]
        # A source renamed by position among the candidates, before any sole owner: the
        # statement-wide reading would place the name by it (the gate's second round).
        if any(isinstance(source, exp.Table) and isinstance(source.args.get("alias"), exp.TableAlias)
               and source.args["alias"].columns for alias, (_, source) in scope.selected_sources.items()
               if alias in candidates):
            return None
        owners: list[tuple[str, str]] = []
        unknown = False
        for alias, (_, source) in scope.selected_sources.items():
            if alias not in candidates:
                continue
            if isinstance(source, exp.Table):
                found = _through_scopes(_same_column(column, alias), scope, visible, dialect)
                if found is not None and any(wanted in _stored(c, dialect) for c in visible.get(source.name, ())):
                    owners.append(found)
            else:
                found = _through_scopes(_same_column(column, alias), scope, visible, dialect)
                if found is not None:
                    owners.append(found)
                else:
                    holds = _names(source, wanted, dialect, visible)
                    if holds:
                        # A local source naming the column that the walk could not follow
                        # to a base column (a USING output of a join inside it, a function,
                        # an expression through a star): the name is that source's, not a
                        # base table's, and the statement-wide reading placed it by one
                        # (Codex's third review of #178: `(SELECT doc FROM t RIGHT JOIN t
                        # q(region, doc) USING (doc)) x WHERE doc = 'west'` compares
                        # `t.region`, the walk said `t.doc`).
                        return None
                    if holds is None:
                        unknown = True  # the source may yield the name, and the engine binds there
        if len(owners) == 1:
            return owners[0]
        return _WIDE if owners or unknown else _ABSENT

    def resolve(column: exp.Column) -> tuple[str, str] | None:
        # The scope's own answer first, then each enclosing scope's, the nearest first, as the
        # engine binds a correlated name: `WITH x(region, doc) AS (SELECT * FROM t) SELECT
        # count(*) FROM x WHERE EXISTS (SELECT 1 FROM u WHERE doc = 'west')` compares `x.doc`,
        # `t.region`, which the list over a star leaves unresolved, and the name, held by no
        # source of the subquery's own scope, fell to the statement-wide reading, `t.doc`, a
        # present value refused (Codex's sixth review of #178). Only a scope none of whose
        # sources can hold the name is passed: a source whose output the walk cannot name may
        # yield it, and the engine binds there, so that scope keeps the statement-wide reading
        # it had (`_names`, the review gate on this round). A CTE's body the engine does not
        # correlate, and `_ABSENT` says no source of it can hold the name, so the name fails
        # there and whatever the walk answers past it is moot.
        scope = scopes.get(id(column))
        while scope is not None:
            answer = in_scope(column, scope)
            if answer is _WIDE:
                break
            if answer is not _ABSENT:
                return answer
            scope = scope.parent
        # The statement-wide reading, as before M174, where a scope cannot tell or no scope
        # answers: a star over a join, a set operation, a recursive CTE, a column sqlglot lists
        # in no scope (the review gate's second round, each refused here and skipped by the
        # scope alone).
        # The catalog's spelling is returned, as the scope's answer returns it: the index is
        # keyed by it, and the name as typed met nothing there (the gate on the fourth round).
        qualifier = column.table
        if qualifier:
            table = alias_to_table.get(qualifier)
            return (table, _stored_match(column, visible.get(table, ()), dialect) or column.name) if table else None
        owners_wide = [t for t in referenced if _stored_match(column, visible.get(t, ()), dialect) is not None]
        return (owners_wide[0], _stored_match(column, visible.get(owners_wide[0], ()), dialect)) if len(owners_wide) == 1 else None  # ambiguous -> skip

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
