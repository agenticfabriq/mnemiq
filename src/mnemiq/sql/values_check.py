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


def _star_yields(projection: exp.Expression, wanted: str, dialect: str | None):
    """How a star projection yields the name, as the engine expands it. "column": its source's
    column (a plain star, or one whose EXCLUDE, REPLACE and RENAME name other columns).
    ("replaced", column): a REPLACE by a column, bare or qualified, which the engine binds across
    the star's whole scope and not within the star's own source (`u.* REPLACE (region AS name)`
    over `t JOIN u` is `t.region`: the review gate on round 21; `* REPLACE (t.doc AS doc)` is
    `t.doc`: Codex's fifteenth review). ("column", identifier): the star's own column
    under another name, a RENAME giving the name where EXCLUDE took the original (`* EXCLUDE (doc)
    RENAME (region AS doc)` is `t.region`). ("renamed", identifier): a RENAME giving the name
    beside the star's own, where the engine lists the two in the table's column order, which the
    walk cannot see: the renamed column where the source lacks the original, nothing otherwise
    (`* RENAME (region AS doc)` over `t` is `t.doc` with `doc` declared first and `t.region` with
    `region` first, measured both ways: the review gate on round 21); two RENAMEs giving one name
    are two columns in that order too, unknown (Codex's fifteenth review). "expression": a
    REPLACE by an expression (`* REPLACE (upper(doc) AS doc)`), no base column behind the name.
    "none": not at all (EXCLUDE names it, or RENAME takes it: `* EXCLUDE (doc)` leaves the name
    to a later projection, which the walk's giving up on every modified star left to the
    statement-wide reading, a present value refused on main and the branch alike: Codex's
    thirteenth review of #178; the RENAME forms its fourteenth). None where the walk cannot tell:
    a qualified EXCLUDE entry on a star not qualified by that same source, which drops one
    source's column while another's may come through."""
    star = projection.this if isinstance(projection, exp.Column) else projection
    own = projection.args.get("table") if isinstance(projection, exp.Column) else None
    gone = False
    for excluded in star.args.get("except_") or ():
        if _written(excluded.this if isinstance(excluded, exp.Column) else excluded, dialect) != wanted:
            continue
        qualifier = excluded.args.get("table") if isinstance(excluded, exp.Column) else None
        if qualifier is None or (own is not None and _written(qualifier, dialect) == _written(own, dialect)):
            gone = True
            continue
        # `* EXCLUDE (t.doc)` over `t JOIN v` drops one source's column and the other's still comes
        # through: which, the walk cannot tell (the review gate on round 20).
        return None
    givers = []
    for renamed in star.args.get("rename") or ():
        before = renamed.this.this if isinstance(renamed.this, exp.Column) else renamed.this
        if _written(renamed.args.get("alias"), dialect) == wanted:
            givers.append(before)
        elif _written(before, dialect) == wanted:
            gone = True
    for replaced in star.args.get("replace") or ():
        if _written(replaced.args.get("alias"), dialect) == wanted:
            value = replaced.this
            if isinstance(value, exp.Column):
                return ("replaced", value)
            return "expression"
    if len(givers) > 1:
        return None  # two renamed columns under one name, in the table's order: unseen
    if givers:
        return ("column", givers[0]) if gone else ("renamed", givers[0])
    return "none" if gone else "column"


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
    in order, or the source a qualified star names: a base table's granted columns say, a local
    source this again; a star says how it yields the name (`_star_yields`: EXCLUDE naming it
    passes to the later projections, REPLACE or RENAME giving it is the source's own), a bare
    star over a join is read by the scope's rules, and a base table renamed by position leaves
    the answer unknown.
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
        yields = _star_yields(projection, wanted, dialect)
        if yields is None:
            return None
        if yields == "none":
            continue  # EXCLUDE names it, or RENAME takes it: a later projection's, if any
        if yields == "expression" or isinstance(yields, tuple):
            return True  # REPLACE or RENAME gives the name: the source's own output has it
        qualifier = projection.args.get("table") if isinstance(projection, exp.Column) else None
        if qualifier is not None:
            alias = _source_named(qualifier, source.selected_sources, dialect)
            if alias is None:
                return None
        elif len(source.selected_sources) != 1:
            # A bare star over a join yields the name as the scope binds a bare one, so the
            # scope's reading says whether it does: nothing of it where no source the grant
            # describes holds it (`b.*` over `u CROSS JOIN w` lacks `doc`, and the later `region
            # AS doc` is the name, a present value the statement-wide reading refused against
            # `t.doc`: Codex's fourteenth review of #178), the owner's where one does. False here
            # is the grant's answer: a table function or a VALUES list is invisible to it and may
            # hold the name, the same false refusal main makes (the review gate on round 21).
            answer = _in_scope(exp.Column(this=exp.Identifier(this=wanted, quoted=True)), source, visible,
                               dialect, depth + 1, listed=True)
            if answer is _ABSENT:
                continue
            return True if isinstance(answer, tuple) or answer is _HELD else None
        else:
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


def _source_named(qualifier: exp.Identifier | None, sources, dialect: str | None) -> str | None:
    """The key of `sources` a qualifier names, as the engine resolves names: the key as written,
    else the one key that, read unquoted, folds to the qualifier's spelling (`T.*` over `t` on
    DuckDB, where the written key missed and a qualified star's walk gave up: the review gate on
    round 19). The scope's map keeps its keys unquoted, so a quoted key in another case may meet
    an unquoted qualifier on an engine that keeps them apart, where the engine then finds no such
    source and runs nothing."""
    if qualifier is None:
        return None
    if qualifier.name in sources:
        return qualifier.name
    wanted = _written(qualifier, dialect)
    matches = [key for key in sources if _written(exp.to_identifier(key), dialect) == wanted]
    return matches[0] if len(matches) == 1 else None


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
    count every earlier CTE among `sources` and skipped the walk (the same finding); over
    several selected sources it is placed by that scope's join rules (`_in_scope`), as the
    engine binds a bare name there (Codex's tenth review of #178). A name no
    projection carries comes through a star, into the scope's one selected source or the one
    a qualified star names, the same way (the review gate: an unqualified `doc` through
    `SELECT * FROM t` lost its owner once
    the statement-wide lookup became the fallback); a star says how it yields the name
    (`_star_yields`), and a local source's named projections prove it absent, so EXCLUDE on
    the name and a source lacking it pass to the later projections. The projections are read in
    order, as the engine names the output: a plain star BEFORE a projection of the asked name
    yields that name first where its source has the column, and the later projection is
    renamed (`WITH t AS (SELECT *, region AS doc FROM t)` keeps the star's `doc`, DuckDB's
    `doc_1` for the other: Codex's first review of #178, where the walk matched the alias by
    name and checked `t.region`); a star whose source the walk cannot read, or a modified one,
    before the name answers nothing. A base table under an alias column list (`FROM t
    q(region, doc)`) renames its columns by position, which a set of names cannot place, so
    the walk answers nothing there (the same review: `q.doc` is `t.region`, and the walk said
    `t.doc`). Names meet as the engine resolves them (`_written`, `_stored`). Where this answers nothing, the caller asks the
    enclosing scopes, then keeps the statement-wide reading it had: a set operation and a
    recursive CTE each resolved there and nowhere here (the gate's second round). A plain star
    over several sources yields the name as the scope binds a bare one, by its join rules
    (Codex's eleventh review); a REPLACE by a column or a RENAME follows that column (the
    fourteenth)."""
    source = scope.sources.get(_source_named(column.args.get("table"), scope.sources, dialect)) if column.table else None
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
        # star met first followed into its source: the scope's one, the one a qualified star
        # names, or the join's rules over several.
        projection = None
        for candidate in selects[len(listed):]:
            star = candidate.this if isinstance(candidate, exp.Column) else candidate
            if isinstance(star, exp.Star):
                yields = _star_yields(candidate, wanted, dialect)
                if yields is None or yields == "expression":
                    return None  # a qualified EXCLUDE the walk cannot read; REPLACE by an expression: no base column
                if yields == "none":
                    continue  # EXCLUDE names it, or RENAME takes it: the name is a later projection's
                if isinstance(yields, tuple) and yields[0] == "replaced":
                    # A REPLACE by a bare column is bound across the star's scope, as a bare name in
                    # it is (Codex's fourteenth review of #178: `* REPLACE (doc AS doc)` is `t.doc`
                    # and the FULL join over it skipped; the review gate on round 21: `u.* REPLACE
                    # (region AS name)` over `t JOIN u` is `t.region`, which a lookup in `u` alone
                    # missed, refusing a present value against the later projection's column).
                    answer = _in_scope(yields[1].copy(), source, visible, dialect, depth + 1)
                    return answer if isinstance(answer, tuple) else None
                # The names to follow in the star's source: the asked one; the column a RENAME gives it
                # from where EXCLUDE took the original (`* EXCLUDE (doc) RENAME (region AS doc)` is
                # `t.region`); or, for a RENAME beside the star's own column, the original and then the
                # renamed, where the renamed is the name only where the source lacks the original, and
                # the two together are listed in the table's column order the walk cannot see, so the
                # scope answers nothing there (the review gate on round 21: `t(region, doc)` lists the
                # renamed `region` first, `t(doc, region)` the original).
                renamed_beside_own = isinstance(yields, tuple) and yields[0] == "renamed"
                if yields == "column":
                    names = [column.this]
                elif yields[0] == "column":
                    names = [yields[1]]
                else:
                    names = [column.this, yields[1]]
                qualifier = candidate.args.get("table") if isinstance(candidate, exp.Column) else None
                if qualifier is not None:
                    # A qualified star yields its named source's own columns, a USING merge aside
                    # (`t.*` over `t RIGHT JOIN b USING (doc)` is `t.doc`, not the merged column),
                    # so the walk follows that source alone; read as a bare star, `u.*` placed
                    # the name by the join and refused `t.region`'s present value (Codex's
                    # twelfth review of #178).
                    alias = _source_named(qualifier, source.selected_sources, dialect)
                    if alias is None:
                        return None
                elif len(source.selected_sources) != 1:
                    # A plain star over a join yields the name as the scope binds a bare one: the
                    # one source holding it under the grant, or a USING or NATURAL join's merged
                    # column (Codex's eleventh review of #178: `(SELECT * FROM t JOIN u ON true) q
                    # FULL JOIN t b USING (doc)` compares `coalesce(q.doc, b.doc)`, `t.doc` twice,
                    # and the walk gave up on the two sources, so the FULL join's left input had
                    # no owner and the scope skipped where main refused the absent value). Two
                    # sources holding it as two columns are listed in FROM order and the first is
                    # the name; a source the walk cannot name beside a holder answers nothing,
                    # since the star lists its columns too (`listed`); none holding it passes to
                    # the next name, then the later
                    # projections. Names are placed by the grant, as the single-source star
                    # places them: a source the grant does not describe, a table function or a
                    # VALUES list, is invisible to it, and the walk reads absence where that
                    # source may hold the name, the statement-wide reading then placing it by a
                    # base table, the same false refusal main makes (the review gate on round 21).
                    for ident in names:
                        answer = _in_scope(exp.Column(this=ident.copy()), source, visible, dialect, depth + 1, listed=True)
                        if answer is _ABSENT:
                            continue
                        if renamed_beside_own and ident is names[0]:
                            return None  # the original is there beside the renamed, in an order unseen
                        return answer if isinstance(answer, tuple) else None
                    continue
                else:
                    (alias,) = source.selected_sources
                inner = source.sources.get(alias)
                for ident in names:
                    # A local source's named projections prove a name absent, and the star yields
                    # nothing of it: `b.*, t.*` over `t JOIN (SELECT name FROM u) b` is `t.doc`
                    # through the second star, where the walk's None for `b.doc` stopped it and
                    # the FULL join over it skipped (Codex's thirteenth review of #178). `_names`
                    # answering None is no verdict: a bare star over a join inside the source is
                    # one the walk below still places by the scope's join rules (the review gate
                    # on round 20).
                    if isinstance(inner, Scope) and _names(inner, _written(ident, dialect), dialect, visible, depth + 1) is False:
                        continue
                    through = _through_scopes(exp.Column(this=ident.copy(), table=exp.to_identifier(alias)), source, visible, dialect, depth + 1)
                    if through is None:
                        return None  # the star may yield the name: not known either way
                    table, name = through
                    # A local source answers only where it projects the name; a base table answers
                    # for any name, so its columns say whether the star yields it (the review gate
                    # on the second round: held to the grant, a name a CTE projects from a column
                    # outside it fell to the later projection).
                    if not isinstance(inner, exp.Table) or any(_stored(c, dialect) & _stored(name, dialect) for c in visible.get(table, ())):
                        if renamed_beside_own and ident is names[0]:
                            return None  # the original is there beside the renamed, in an order unseen
                        return through  # the star yields the name first
                continue  # the star's source lacks every name: the name is a later projection's
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
    if isinstance(inner, exp.Column) and source.selected_sources:
        # Over several sources the projection is placed as the engine binds it in its own
        # scope, by the join rules: `(SELECT doc FROM t FULL JOIN t b USING (doc)) q` projects
        # `coalesce(t.doc, b.doc)`, `t.doc` twice, and the walk gave up on the two sources, so
        # `q FULL JOIN t c USING (doc) WHERE doc = 'west'` skipped where main refused the absent
        # value (Codex's tenth review of #178); where the rules answer nothing, or that the
        # scope cannot tell, the walk answers nothing.
        answer = _in_scope(inner, source, visible, dialect, depth + 1)
        return answer if isinstance(answer, tuple) else None
    return None


def _from_order(node: exp.Expression) -> list[str]:
    """The aliases of a FROM item and the joins hanging off it, in the order the engine lists a
    star's columns. A parenthesised join group is an alias-less Subquery wrapping the group's
    first item, which carries the group's joins, and `((SELECT region AS doc FROM t) b CROSS JOIN
    t)` lists `b` first where the scope's own order, base tables first, listed `t` (the review gate
    on round 22). The join rules read the scope's top-level joins alone; a USING join inside a
    group is a merge they do not see, as before."""
    joins = [alias for join in node.args.get("joins") or [] for alias in _from_order(join.this)]
    if isinstance(node, exp.Subquery) and not node.alias and isinstance(node.this, (exp.Table, exp.Subquery)):
        # A group wraps a FROM item; an alias-less derived table wraps a SELECT, whose own joins are
        # another scope's (the review gate on round 23).
        return _from_order(node.this) + joins
    return [node.alias_or_name] + joins


# The engines that list a star's duplicate names in FROM order and bind a bare name to the first,
# the second renamed (`doc_1`): DuckDB, measured on 1.5.4. Postgres and MySQL refuse such a bare
# name as ambiguous, so no reading of it is the engine's and the scope answers nothing there (the
# review gate on round 22).
_LISTS_DUPLICATE_STAR_COLUMNS = frozenset({"duckdb"})


_ABSENT = object()  # no source of the scope can hold the name: the enclosing scope's turn
_WIDE = object()  # the scope cannot tell: the statement-wide reading's turn, as before
_HELD = object()  # under a star: a source holds the name and the scope cannot place it (`_names` reads it as held)


def _in_scope(column: exp.Column, scope: Scope, visible: dict[str, set[str]], dialect: str | None,
              depth: int = 0, listed: bool = False) -> tuple[str, str] | None | object:
    """The scope's own answer (M141, M174): a base table, or a local source followed to the
    column it projects, the column renamed included -- `x.doc` may be `t.region`, and read
    by its name it would be checked against `t.doc`'s values. An unqualified name is read
    among the scope's selected sources, a base table holding it or a local source
    projecting it, one owner or no answer. None is a skip; `_ABSENT` says no source of this
    scope can hold the name, and the enclosing scope is asked; `_WIDE` says the scope
    cannot tell, a source's output being one the walk cannot name (`_names`) or two
    sources owning the name, and the statement-wide reading answers as before; where a join
    has made the name one side's (`narrowed`) that scope answers nothing instead, since the
    statement-wide reading would resurrect the other side. Under a star (`listed`) a source the
    walk cannot name beside a holder answers nothing: the star lists that source's columns too,
    and the engine binds the name to the first without the ambiguity error a bare name meets;
    two sources holding the name as two columns are, on DuckDB, the first in FROM order
    (`_LISTS_DUPLICATE_STAR_COLUMNS`), where a bare name would be the ambiguity error, and on
    the other engines `_HELD`: the star holds the name, which `_names` reports to the enclosing
    scope, and the scope cannot place it, where None read as unknown fell to the statement-wide
    reading. `_HELD` arrives only under a star; the top-level resolve never sees it."""
    if column.table:
        named = _source_named(column.args.get("table"), scope.sources, dialect)
        return _through_scopes(column, scope, visible, dialect, depth + 1) if named is not None else _ABSENT
    wanted = _written(column.this, dialect)
    # The merged column of a USING join is the join's: an inner or LEFT join's the
    # left input's, a RIGHT join's the right input's, a FULL join's coalesced from
    # both (measured by the ui lane on Postgres and DuckDB). `SELECT doc FROM t
    # RIGHT JOIN t q(region, doc) USING (doc)` compares `q.doc`, `t.region`, where
    # the walk discarded q and took `t.doc` as the sole owner (Codex's second
    # review of #178); a FULL join's the same with the literal on q's side (the ui
    # lane). The candidates are the join's side; a source renamed by position
    # among them answers nothing, and so does a FULL join.
    # In FROM order, as the engine lists a star's columns: sqlglot's scope lists base tables
    # before derived tables, which put `t` first under `(SELECT ...) b CROSS JOIN t` and placed
    # the star's `doc` by `t` where the engine lists `b`'s (round 22 of #178).
    in_from_order = (_from_order(scope.expression.args["from_"].this) if scope.expression.args.get("from_") is not None else []) + [
        alias for join in scope.expression.args.get("joins") or [] for alias in _from_order(join.this)]
    candidates = [a for a in in_from_order if a in scope.selected_sources] + [
        a for a in scope.selected_sources if a not in in_from_order]
    # This scope's own joins alone, in order, each USING the key narrowing the
    # candidates in turn (a later RIGHT join decides over an earlier inner one): a
    # join inside a subquery is another scope's, and read here it switched the check
    # off for the outer column (the review gate on the fourth round). A NATURAL join
    # is a USING join on every name its sides share, so it unsettles a name only where
    # both sides may hold it (`_may_hold`), and then by its side as a USING key is: a
    # name the right side cannot hold the join merges nothing of, and one the left
    # side cannot hold is the right side's (Codex's third review of #178: `t NATURAL
    # RIGHT JOIN (SELECT region AS doc FROM t) q` compares `t.region`; the seventh:
    # `EXISTS (SELECT 1 FROM w NATURAL JOIN w b WHERE doc = 'west')` compares the outer
    # `t.doc`; the eighth: `w NATURAL JOIN w b, t`, `t NATURAL JOIN t b` and five more
    # compare `t.doc`, where a skip on any NATURAL join in the scope let an absent
    # value through). A FULL join's merged column is coalesced from both sides, and
    # answers only where both resolve to one base column (`t NATURAL FULL JOIN t b`
    # compares `coalesce(t.doc, b.doc)`, both `t.doc`: Codex's ninth review), else
    # nothing. A merged column is one side's, so where that side's output the walk
    # cannot name the scope answers nothing rather than the statement-wide reading, which
    # resurrected the discarded side: `t NATURAL RIGHT JOIN (SELECT * FROM (SELECT region AS
    # doc FROM t) z CROSS JOIN u) q WHERE doc = 'west'` compares `q.doc`, `t.region`, and
    # was refused against `t.doc` (the same review). A join's left side is the FROM
    # source and the joins before it, not every other source of the scope: read as the
    # latter, a source joined LATER that held the name made an earlier NATURAL join a
    # merge, and `w NATURAL JOIN (SELECT region AS doc, 'west' AS k FROM t) b NATURAL LEFT
    # JOIN t c WHERE doc = 'west'` refused a present value, `b.doc`, against `t.doc` (the
    # review gate on round 13).
    from_ = scope.expression.args.get("from_")
    left = [from_.this.alias_or_name] if from_ is not None else []
    narrowed = False  # a join made the name one side's
    full_sides = None  # a FULL join's two inputs, by alias: both stay, one answer where they agree
    comma_starts_an_input = (dialect or "").lower() in _COMMA_BINDS_LOOSER
    for join in scope.expression.args.get("joins") or []:
        right = join.this.alias_or_name
        if comma_starts_an_input and not any(join.args.get(k) for k in ("side", "kind", "method", "on", "using")):
            # A comma binds more loosely than JOIN, so it begins a new left input: `v, t FULL
            # JOIN t b USING (doc)` compares `coalesce(t.doc, b.doc)`, and with `v` among the
            # join's left input the owners differed and the scope skipped where main refused
            # (the review gate on round 15).
            left = [right]
            continue
        if (join.method or "").upper() == "NATURAL":
            right_may = right not in scope.selected_sources or _may_hold(
                scope.selected_sources[right][1], wanted, dialect, visible)
            if not right_may:
                left.append(right)
                continue
            if not any(_may_hold(scope.selected_sources[a][1], wanted, dialect, visible)
                       for a in candidates if a in left and a in scope.selected_sources):
                candidates = [right]
                narrowed = True
                full_sides = None  # the name is the right side's alone; a FULL join before it is superseded
                left.append(right)
                continue
        elif not any(_written(key, dialect) == wanted for key in (join.args.get("using") or ())):
            left.append(right)
            continue
        side = (join.side or "").upper()
        narrowed = True
        if side == "FULL":
            # The join's left input is the sources before it still among the candidates, its
            # right input the joined source; counted as every source of the scope, a source
            # beside the join that cannot hold the name blocked the answer and `w, t FULL JOIN
            # t b USING (doc)` skipped where main refused (the review gate on round 15).
            full_sides = ([a for a in candidates if a in left], [right])
            candidates = full_sides[0] + full_sides[1]
        elif side == "RIGHT":
            candidates = [right]
            full_sides = None  # a later RIGHT join's merged column is its right input's alone
        else:
            candidates = [a for a in candidates if a != right]
        left.append(right)
    # A source renamed by position among the candidates, before any sole owner: the
    # statement-wide reading would place the name by it (the gate's second round).
    if any(isinstance(source, exp.Table) and isinstance(source.args.get("alias"), exp.TableAlias)
           and source.args["alias"].columns for alias, (_, source) in scope.selected_sources.items()
           if alias in candidates):
        return None
    owner_of: dict[str, tuple[str, str]] = {}  # by the candidate's alias
    unknown = False
    for alias, (_, source) in scope.selected_sources.items():
        if alias not in candidates:
            continue
        if isinstance(source, exp.Table):
            found = _through_scopes(_same_column(column, alias), scope, visible, dialect, depth + 1)
            if found is not None and any(wanted in _stored(c, dialect) for c in visible.get(source.name, ())):
                owner_of[alias] = found
        else:
            found = _through_scopes(_same_column(column, alias), scope, visible, dialect, depth + 1)
            if found is not None:
                owner_of[alias] = found
            else:
                holds = _names(source, wanted, dialect, visible, depth + 1)
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
    owners = list(owner_of.values())
    if full_sides is not None:
        # The merged column is coalesced from the join's two inputs: each must add an owner, and
        # every owner must be the one base column. An input that adds none, a table lacking the
        # name under the grant or a source the walk cannot name, is one whose values the
        # coalesced column still carries (the review gate on round 14); a source beside the
        # left input that cannot hold the name is no part of it.
        sides = [[owner_of[a] for a in side if a in owner_of] for side in full_sides]
        return owners[0] if all(sides) and len(set(owners)) == 1 else None
    if owners and len(set(owners)) == 1 and not (listed and unknown):
        # Two holders of one lineage are the one column (`t CROSS JOIN t tt` lists `doc` twice, each
        # `t.doc`): a bare name there is the engine's ambiguity error, and under a star the first
        # listed, the same column either way (Codex's fourteenth review of #178, where a FULL join
        # over such a star had no owner and skipped).
        # A bare name beside a source the walk cannot name is the one owner's, since the engine
        # raises where that source holds it too; a star lists both and raises nothing, so the
        # name may be the unnamed source's (`(VALUES ('west')) AS x(doc), t`: the review gate on
        # round 18, where `t` alone was counted and a present value refused).
        return owners[0]
    if listed and owners and not unknown and not narrowed and (dialect or "").lower() in _LISTS_DUPLICATE_STAR_COLUMNS:
        # Under a star, two sources holding the name as two columns are listed in FROM order and
        # the engine binds the bare name to the first (`t CROSS JOIN (SELECT region AS doc FROM t)
        # b` lists `doc` then `doc_1`): the first holder's lineage, which the walk sees in the
        # FROM order where it cannot see a table's column order (Codex's fifteenth review of
        # #178: a FULL join over such a star had no owner and skipped where main refused, and the
        # reversed order refused `t.region`'s present value against `t.doc` on main and here).
        return next(owner_of[alias] for alias in candidates if alias in owner_of)
    if listed and len(owners) > 1:
        # The star lists both columns: DuckDB binds the first, read above where the order is known,
        # and the other engines refuse the bare name as ambiguous, so the statement-wide reading has
        # no say here and would place the name by a base table (the review gate on round 23). The
        # scope holds the name, which the enclosing one must hear, and cannot place it.
        return _HELD
    if owners or unknown:
        return None if narrowed else _WIDE
    return _ABSENT


# The engines on which a join the parser gives no kind, side, method, ON or USING can only be a
# comma, which binds more loosely than JOIN: a bare JOIN and an APPLY are syntax errors on DuckDB
# (measured on 1.5.4), and Postgres documents the same grammar. MySQL takes a bare JOIN, which
# binds like any JOIN and reads as the same node (the review gate on round 16: `t JOIN w NATURAL
# LEFT JOIN (SELECT region AS doc FROM t) q` compares `t.doc` there), and Oracle takes CROSS and
# OUTER APPLY, which read as the node of a comma before a LATERAL; on both the sources before such
# a join stay its left input. SQLite binds a comma like JOIN, and the parser types its joins (a
# comma a CROSS JOIN, a bare JOIN `ON TRUE`), so the node never arises there.
_COMMA_BINDS_LOOSER = frozenset({"duckdb", "postgres", "postgresql"})


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
            answer = _in_scope(column, scope, visible, dialect)
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
