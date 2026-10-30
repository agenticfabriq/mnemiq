from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import NamedTuple

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope

from mnemiq.semantic.values import shown
from mnemiq.sql.qualify import names_one_object
from mnemiq.sql.identifiers import resolve, resolve_identifier, resolve_stored
from mnemiq.sql.verdict import Refusal, RefusalCode

logger = logging.getLogger(__name__)


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
    thirteenth review of #178; the RENAME forms its fourteenth). ("without", qualifiers): a plain
    star whose qualified EXCLUDE entries drop the named sources' own columns of the name, the other
    sources' and any merged column still coming through, so the star yields the name as the scope
    binds a bare one among the columns left (`* EXCLUDE (t.doc)` over `t CROSS JOIN (SELECT region
    AS doc FROM t) b` is `b`'s `doc`, `t.region`, and over `t LEFT JOIN b USING (doc)` the merged
    `doc`, `t.doc`: Codex's nineteenth review, where read as unknown the first fell to
    the statement-wide reading and refused the present value against the excluded `t.doc`, on
    main and the branch alike). None where the walk cannot tell: a qualified star's EXCLUDE
    naming another source, or such an EXCLUDE beside a RENAME giving the name."""
    star = projection.this if isinstance(projection, exp.Column) else projection
    own = projection.args.get("table") if isinstance(projection, exp.Column) else None
    gone = False
    without: list[exp.Identifier] = []
    for excluded in star.args.get("except_") or ():
        if _written(excluded.this if isinstance(excluded, exp.Column) else excluded, dialect) != wanted:
            continue
        qualifier = excluded.args.get("table") if isinstance(excluded, exp.Column) else None
        if qualifier is None or (own is not None and _written(qualifier, dialect) == _written(own, dialect)):
            gone = True
            continue
        if own is not None:
            return None  # a qualified star's EXCLUDE naming another source: unread
        # `* EXCLUDE (t.doc)` over `t JOIN v` drops one source's column and the other's still comes
        # through (the review gate on round 20): the walk reads the sources left.
        without.append(qualifier)
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
    if len(givers) > 1 or (givers and without):
        return None  # two renamed columns under one name, or one beside a source's column left: an order unseen
    if givers:
        return ("column", givers[0]) if gone else ("renamed", givers[0])
    if gone:
        return "none"
    return ("without", without) if without else "column"


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
        without: frozenset[str] = frozenset()
        if isinstance(yields, tuple) and yields[0] == "without":
            aliases = [_source_named(q, source.selected_sources, dialect) for q in yields[1]]
            if any(a is None for a in aliases):
                return None
            without = frozenset(aliases)
        elif yields == "expression" or isinstance(yields, tuple):
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
                               dialect, depth + 1, listed=True, without=without)
            if answer is _ABSENT:
                continue
            return True if isinstance(answer, tuple) or answer is _HELD else None
        else:
            (alias,) = source.selected_sources
            if alias in without:
                continue  # the one source's column is the excluded one: a later projection's, if any
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
                without: frozenset[str] = frozenset()
                if isinstance(yields, tuple) and yields[0] == "without":
                    # A qualified EXCLUDE drops the named source's column and the others' still come
                    # through, so the star yields the name as the scope binds a bare one among the
                    # sources left (Codex's nineteenth review of #178: `* EXCLUDE (t.doc)` over `t
                    # CROSS JOIN (SELECT region AS doc FROM t) b` is `b`'s `doc`, `t.region`, and read
                    # as unknown it fell to the statement-wide reading, which refused the present
                    # value against the excluded `t.doc`, on main and here alike).
                    aliases = [_source_named(q, source.selected_sources, dialect) for q in yields[1]]
                    if any(a is None for a in aliases):
                        return None
                    without = frozenset(aliases)
                    yields = "column"
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
                        answer = _in_scope(exp.Column(this=ident.copy()), source, visible, dialect, depth + 1, listed=True, without=without)
                        if answer is _ABSENT:
                            continue
                        if renamed_beside_own and ident is names[0]:
                            return None  # the original is there beside the renamed, in an order unseen
                        return answer if isinstance(answer, tuple) else None
                    continue
                else:
                    (alias,) = source.selected_sources
                    if alias in without:
                        continue  # the one source's column is the excluded one: the name is a later projection's
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


class _Merge(NamedTuple):
    """A USING or NATURAL join's merged column of the name: the join's side and the two inputs' columns."""

    side: str
    left: list
    right: list


# The engines that list a star's duplicate names in FROM order and bind a bare name to the first,
# the second renamed (`doc_1`): DuckDB, measured on 1.5.4. Postgres 16 refuses such a bare name as
# ambiguous and MySQL 8.4 the derived table itself for the duplicate name (both measured on
# throwaway containers, round 24), so no reading of it is the engine's and the scope holds the name
# without placing it (`_HELD`).
_LISTS_DUPLICATE_STAR_COLUMNS = frozenset({"duckdb"})

# The engines that bind a bare name to a USING or NATURAL join's merged column before every other
# source's column of the name: DuckDB, measured on 1.5.4 (`v, t FULL JOIN t b USING (doc)`, `t JOIN t
# c USING (doc) CROSS JOIN (SELECT region AS doc FROM t) b` and `(t FULL JOIN t b USING (doc)), v`
# each compare the merged column). Postgres 16 and MySQL 8.4 refuse the bare name as ambiguous in
# each of those shapes, the plain `t JOIN t c USING (doc)` alone resolving (measured on throwaway
# containers, round 25), so there the merged column is one lineage among the others.
_BINDS_A_MERGE_FIRST = frozenset({"duckdb"})


_ABSENT = object()  # no source of the scope can hold the name: the enclosing scope's turn
_WIDE = object()  # the scope cannot tell: the statement-wide reading's turn, as before
_HELD = object()  # a source or a merged column holds the name and the scope cannot place it (`_names`: held)
_UNKNOWN = object()  # a source the walk cannot name may hold the name, and the engine may bind there


def _in_scope(column: exp.Column, scope: Scope, visible: dict[str, set[str]], dialect: str | None,
              depth: int = 0, listed: bool = False, without: frozenset[str] = frozenset()) -> tuple[str, str] | None | object:
    """The scope's own answer (M141, M174): a base table, or a local source followed to the
    column it projects, the column renamed included -- `x.doc` may be `t.region`, and read
    by its name it would be checked against `t.doc`'s values. An unqualified name is read
    among the name's output columns of the FROM clause, as the engine lists a star's: each
    source holding the name is one, and a USING or NATURAL join on the name merges its two
    inputs' into one, the join's, listed in the join's place. The reading is the engine's
    binding of a bare name among those columns: a merged column before every other on DuckDB
    (`_BINDS_A_MERGE_FIRST`); one owner;
    the owner beside a source the walk
    cannot name, since the engine raises where that source holds it too; and nothing where two
    lineages are listed, the engine's ambiguity error. None is a skip; `_ABSENT` says no source
    of this scope can hold the name, and the enclosing scope is asked; `_WIDE` says the scope
    cannot tell and no join concerned the name, so the statement-wide reading answers as before,
    where a join that made the name one side's (`narrowed`) answers nothing instead, since that
    reading would resurrect the other side. Under a star (`listed`) the engine binds the name to
    the first column listed without the ambiguity error a bare name meets: on DuckDB
    (`_LISTS_DUPLICATE_STAR_COLUMNS`) the first column is the reading whatever follows, nothing
    where the walk cannot name it; elsewhere two columns of one lineage are that column and any
    other pair `_HELD`, Postgres refusing the bare name and MySQL the derived table."""
    if column.table:
        named = _source_named(column.args.get("table"), scope.sources, dialect)
        return _through_scopes(column, scope, visible, dialect, depth + 1) if named is not None else _ABSENT
    wanted = _written(column.this, dialect)
    comma_starts_an_input = (dialect or "").lower() in _COMMA_BINDS_LOOSER
    presence: dict[str, object] = {}
    seen: set[str] = set()
    order: list[str] = []  # the sources in FROM order, as the engine lists a star's columns
    narrowed = False  # a join made the name one side's

    def present(alias: str) -> object:
        """The source's column of the name: an owner, `_HELD`, `_UNKNOWN`, or None for a source
        without one, or one the scope does not list (a table function, a VALUES list: the grant's
        blind spot, recorded)."""
        if alias not in presence:
            presence[alias] = _column_of(alias)
        return presence[alias]

    def _column_of(alias: str) -> object:
        if alias not in scope.selected_sources:
            return None
        _, source = scope.selected_sources[alias]
        if isinstance(source, exp.Table):
            if isinstance(source.args.get("alias"), exp.TableAlias) and source.args["alias"].columns:
                # Renamed by position: the statement-wide reading would place the name by it (the
                # gate's second round), so the scope holds it and answers nothing.
                return _HELD
            found = _through_scopes(_same_column(column, alias), scope, visible, dialect, depth + 1)
            if found is not None and any(wanted in _stored(c, dialect) for c in visible.get(source.name, ())):
                return found
            return None
        found = _through_scopes(_same_column(column, alias), scope, visible, dialect, depth + 1)
        if found is not None:
            return found
        # A local source naming the column that the walk could not follow to a base column (a
        # USING output of a join inside it, a function, an expression through a star): the name
        # is that source's, not a base table's, and the statement-wide reading placed it by one
        # (Codex's third review of #178: `(SELECT doc FROM t RIGHT JOIN t q(region, doc) USING
        # (doc)) x WHERE doc = 'west'` compares `t.region`, the walk said `t.doc`).
        holds = _names(source, wanted, dialect, visible, depth + 1)
        return _HELD if holds else (_UNKNOWN if holds is None else None)

    def may_hold(col: str | _Merge) -> bool:
        """Whether a column may be the name's: a merged one is, the engine demanding the name on both
        inputs whatever the grant shows of them; a source the scope does not list may; a listed one
        by its presence."""
        return isinstance(col, _Merge) or col not in scope.selected_sources or present(col) is not None

    def columns(node: exp.Expression) -> list[str | _Merge]:
        """The name's output columns of a FROM item and the joins hanging off it. A parenthesised
        join group is an alias-less Subquery wrapping the group's first item, which carries the
        group's joins, where an alias-less derived table wraps a SELECT whose joins are its own
        scope's (the review gate on round 23); `((SELECT region AS doc FROM t) b CROSS JOIN t)`
        lists `b` first where sqlglot's scope, base tables first, lists `t` (the gate on round
        22), and `(t RIGHT JOIN (SELECT region AS doc FROM t) b USING (doc))` is one merged column,
        `b`'s, where the flat order read `t` first (Codex's sixteenth review)."""
        if isinstance(node, exp.Subquery) and not node.alias and isinstance(node.this, (exp.Table, exp.Subquery)):
            first = columns(node.this)
        else:
            first = [node.alias_or_name]
            seen.add(node.alias_or_name)
            order.append(node.alias_or_name)
        return joined(first, node.args.get("joins") or [])

    def joined(current: list[str | _Merge], joins: list[exp.Join]) -> list[str | _Merge]:
        """The columns after each join in turn. The merged column of a USING join is the join's: an
        inner or LEFT join's the left input's, a RIGHT join's the right input's, a FULL join's
        coalesced from both (measured by the ui lane on Postgres and DuckDB). `SELECT doc FROM t
        RIGHT JOIN t q(region, doc) USING (doc)` compares `q.doc`, `t.region`, where the walk
        discarded q and took `t.doc` as the sole owner (Codex's second review of #178); a FULL
        join's the same with the literal on q's side (the ui lane). A NATURAL join is a USING join
        on every name its sides share, so it merges the name only where both sides may hold it: a
        name the right side cannot hold the join merges nothing of, and one the left side cannot
        hold is the right side's (Codex's third review: `t NATURAL RIGHT JOIN (SELECT region AS
        doc FROM t) q` compares `t.region`; the seventh: `EXISTS (SELECT 1 FROM w NATURAL JOIN w b
        WHERE doc = 'west')` compares the outer `t.doc`; the eighth: `w NATURAL JOIN w b, t`, `t
        NATURAL JOIN t b` and five more compare `t.doc`, where a skip on any NATURAL join in the
        scope let an absent value through). A join's left input is the FROM item and the joins
        before it, not every other source of the scope: read as the latter, a source joined LATER
        that held the name made an earlier NATURAL join a merge, and `w NATURAL JOIN (SELECT region
        AS doc, 'west' AS k FROM t) b NATURAL LEFT JOIN t c WHERE doc = 'west'` refused a present
        value, `b.doc`, against `t.doc` (the review gate on round 13). A comma binds more loosely
        than JOIN, so it begins a new input: `v, t FULL JOIN t b USING (doc)` compares
        `coalesce(t.doc, b.doc)`, and with `v` among the join's left input the owners differed and
        the scope skipped where main refused (the review gate on round 15); under a star the
        earlier input's columns stay listed before the join's, so `(SELECT region AS doc FROM t)
        b, t RIGHT JOIN t c USING (doc)` lists `b`'s `doc` first and the engine binds the bare
        name there, where the join's narrowing to `c` refused a present value against `t.doc`
        (Codex's sixteenth review). A merged column followed by another holder is the first
        listed (`t JOIN t c USING (doc) CROSS JOIN (SELECT region AS doc FROM t) b` is `t.doc`),
        where a guard on any narrowing skipped the absent value main refused (the same review)."""
        nonlocal narrowed
        inputs: list[list[str | _Merge]] = []
        for join in joins:
            right = columns(join.this)
            if comma_starts_an_input and not any(join.args.get(k) for k in ("side", "kind", "method", "on", "using")):
                inputs.append(current)
                current = right
                continue
            if (join.method or "").upper() == "NATURAL":
                left_may, right_may = any(map(may_hold, current)), any(map(may_hold, right))
                if not (left_may and right_may):
                    narrowed = narrowed or (right_may and not left_may)
                    current = current + right
                    continue
            elif not any(_written(key, dialect) == wanted for key in (join.args.get("using") or ())):
                current = current + right
                continue
            narrowed = True
            current = [_Merge((join.side or "INNER").upper(), current, right)]
        inputs.append(current)
        return [col for group in inputs for col in group]

    def bare(cols: list[str | _Merge]) -> object:
        """What a bare name among these columns reads, as a USING key is read on each input: one
        owner, also beside a source the walk cannot name (the engine raises where that source
        holds it too: `(VALUES ('west')) AS x(doc), t` is `t.doc` or the error); `_HELD`;
        `_UNKNOWN`; None where no column is the name's; `_WIDE` where two lineages are listed,
        the engine's ambiguity error. On DuckDB a merged column among them takes the name before
        every other source's column (`_BINDS_A_MERGE_FIRST`), inside a join's input as at the top:
        `t RIGHT JOIN (t b JOIN t c USING (doc) CROSS JOIN (SELECT region AS doc FROM t) d) USING
        (doc)` binds the right input's key to the merged `doc`, `t.doc`, where the merged column
        and `d`'s counted as two lineages and the absent value main refused was skipped (Codex's
        seventeenth review of #178); two merged columns are the ambiguity error there too."""
        if (dialect or "").lower() in _BINDS_A_MERGE_FIRST:
            merges = [col for col in cols if isinstance(col, _Merge)]
            if merges:
                return resolve(merges[0]) if len(merges) == 1 else _WIDE
        found = [f for f in (resolve(col) for col in cols) if f is not None]
        if not found:
            return None
        if any(f is _HELD for f in found):
            return _HELD
        owners = {f for f in found if isinstance(f, tuple)}
        if len(owners) == 1:
            return next(iter(owners))
        return _WIDE if owners else _UNKNOWN

    def resolve(col: str | _Merge) -> object:
        """A column's lineage: a source's presence, or a merged column's by its join's side. A FULL
        join's is coalesced from both inputs: each must add an owner, and every owner must be the
        one base column (`t NATURAL FULL JOIN t b` compares `coalesce(t.doc, b.doc)`, both `t.doc`:
        Codex's ninth review), else nothing; an input that adds none, a table lacking the name
        under the grant or a source the walk cannot name, is one whose values the coalesced column
        still carries (the review gate on round 14). A merged column is one side's, so where that
        side's output the walk cannot name the scope answers nothing rather than the statement-wide
        reading, which resurrected the discarded side: `t NATURAL RIGHT JOIN (SELECT * FROM (SELECT
        region AS doc FROM t) z CROSS JOIN u) q WHERE doc = 'west'` compares `q.doc`, `t.region`, and
        was refused against `t.doc` (the same review)."""
        if not isinstance(col, _Merge):
            return present(col)
        # The merged column is there whatever the walk can say of its inputs (USING demands the
        # name on both), so where its lineage cannot be told the scope holds the name: `_HELD`.
        if col.side == "FULL":
            left, right = bare(col.left), bare(col.right)
            return left if isinstance(left, tuple) and left == right else _HELD
        side = bare(col.right if col.side == "RIGHT" else col.left)
        if col.side == "INNER" and not isinstance(side, tuple) and side is not _WIDE:
            # An inner join's merged values are both inputs' on every row, so the other input's
            # lineage reads the same values where this one cannot be read under the grant (`v FULL
            # JOIN v b USING (doc) NATURAL JOIN (SELECT doc FROM t) q` with `v.doc` hidden is `t.doc`:
            # the review gate on round 15); not where this input lists two lineages, the engine's
            # ambiguity error on the key.
            other = bare(col.right)
            if isinstance(other, tuple):
                return other
        return side if isinstance(side, tuple) else _HELD

    from_ = scope.expression.args.get("from_")
    cols = joined(columns(from_.this) if from_ is not None else [], scope.expression.args.get("joins") or [])
    # The star's columns of the name in the engine's order (measured on DuckDB 1.5.4, Codex's twenty-first
    # review of #178 and the review gates on rounds 29 and 30): a merged column sits where its first
    # surviving holder sits, keeps its lineage however many holders a qualified EXCLUDE names, and
    # goes with its last; an input's other columns of the name keep their own places. The holders
    # of a merge are the sources its USING key binds on each input: a nested merge's own holders
    # where the input holds one (the key binds to the merged column before any other: `(t JOIN t b
    # USING (doc) CROSS JOIN (SELECT region AS doc FROM t) d) JOIN t c USING (doc)` under `EXCLUDE
    # (t.doc, b.doc, c.doc)` lists `d`'s `doc` alone, where `d` counted as a holder kept the outer
    # merge alive and refused `t.region`'s present value), else the input's holders the grant
    # shows, else, the grant hiding the key USING demands, every source of the input (`EXCLUDE
    # (t.doc)` over `t LEFT JOIN v USING (doc)` with `v.doc` hidden keeps the merge: round 30's
    # gate). `EXCLUDE (t.doc, b.doc)` over the nested shape lists `d`'s `doc` before the merge,
    # now at `c`'s place, and `EXCLUDE (t.doc)` over `t LEFT JOIN b USING (doc) CROSS JOIN d` the
    # merge at `b`'s place, before `d`.

    def lacks(alias: str) -> bool:
        """Whether a source's own output provably lacks the name: a local source whose projections,
        none of them a star, name it nowhere (`_names` False). A base table may hold it in a column
        the grant hides, and so may a star over one, which `_names` reads by the grant alone (`(SELECT
        * FROM v) z` with `v.doc` withheld: the review gate on round 32, where the wrapped holder was
        dropped as a stand-in and the merge it held went with the visible input's exclusion)."""
        source = scope.selected_sources.get(alias, (None, None))[1]
        return (isinstance(source, Scope) and isinstance(source.expression, exp.Select)
                and not any(_projects_a_star(p) for p in source.expression.selects)
                and _names(source, wanted, dialect, visible, depth + 1) is False)

    def holders(col: _Merge) -> list[str]:
        out: list[str] = []
        for side in (col.left, col.right):
            merges = [c for c in side if isinstance(c, _Merge)]
            if merges:
                for m in merges:
                    out += holders(m)
                continue
            visible_holders = [c for c in side if present(c) is not None]
            if visible_holders:
                out += visible_holders
                continue
            # The input's sources stand in for the holder the grant hides, save one whose own output
            # provably lacks the name (a derived table projecting other columns): counted, `z` in `t
            # LEFT JOIN (v CROSS JOIN (SELECT name FROM u) z) USING (doc)` with `v.doc` withheld kept
            # the merge alive at `z`'s place once `t.doc` and `v.doc` were excluded, and `d`'s present
            # value was refused against `t.doc` (Codex's twenty-second review of #178). A stand-in a
            # qualified EXCLUDE names holds the column, the engine refusing an EXCLUDE of a column a
            # source lacks, and is the input's only holder, two being the ambiguity error on the key,
            # so it alone stands in and the merge goes with it (Codex's twenty-third review: `u` beside
            # the excluded `v` kept the merge alive and `d`'s present value was refused; the review gate
            # on round 33: the exclusion settles which stand-in held the key, no unsure case is left).
            candidates = [c for c in side if not isinstance(c, _Merge) and not lacks(c)]
            named = [c for c in candidates if c in without]
            out += named or candidates
        return out

    merge_of: dict[str, _Merge] = {}
    for col in cols:
        if isinstance(col, _Merge):
            for alias in holders(col):
                merge_of[alias] = col
    emitted: set[int] = set()
    star: list[str | _Merge] = []
    for alias in order + [alias for alias in scope.selected_sources if alias not in seen]:
        if alias in without:
            continue
        merge = merge_of.get(alias)
        if merge is None:
            star.append(alias)
        elif id(merge) not in emitted:
            emitted.add(id(merge))
            star.append(merge)
    cols = star
    if not listed:
        # The engine's binding of the bare name among the columns, as a USING key's on an input.
        answer = bare(cols)
        if answer is None:
            return _ABSENT
        if isinstance(answer, tuple) or answer is _HELD:
            return answer
        # Two lineages, the engine's ambiguity error, or sources the walk cannot name alone:
        # nothing where a join made the name one side's, the statement-wide reading's turn else.
        return None if narrowed else _WIDE
    # Under a star, the name's columns in the order the engine lists them.
    found = [f for f in (resolve(col) for col in cols) if f is not None]
    if not found:
        return _ABSENT
    if (dialect or "").lower() in _LISTS_DUPLICATE_STAR_COLUMNS:
        # DuckDB lists every column and binds the bare name to the first, whatever follows (`t CROSS
        # JOIN (SELECT region AS doc FROM t) b` lists `doc` then `doc_1`: Codex's fifteenth review,
        # where the reversed order refused `t.region`'s present value against `t.doc`), so a known
        # first column is the reading even where a later one cannot be placed: `t CROSS JOIN (t b
        # FULL JOIN (SELECT region AS doc FROM t) c USING (doc))` is `t.doc`, and a guard on any
        # unplaced column skipped the absent value main refused (the eighteenth review). A first
        # column the walk cannot name may be the engine's binding (`(VALUES ('west')) AS x(doc),
        # t`: the review gate on round 18, where `t` alone was counted and a present value
        # refused), so nothing is placed; a later owner, or a later column held without a place,
        # says the scope holds the name (the review gate on round 27: read as unknown, the
        # enclosing scope fell to the statement-wide reading and refused a present value).
        first = found[0]
        if isinstance(first, tuple) or first is _HELD:
            return first
        return _HELD if any(isinstance(f, tuple) or f is _HELD for f in found) else None
    # The other engines refuse a bare name two listed columns hold, or the derived table itself
    # (Postgres 16 and MySQL 8.4, measured), so two holders of one lineage are that column (`t
    # CROSS JOIN t tt`: Codex's fourteenth review) and any other pair holds the name unplaced.
    if len(set(found)) == 1 and isinstance(found[0], tuple):
        return found[0]
    if any(isinstance(f, tuple) or f is _HELD for f in found):
        return _HELD
    return None  # sources the walk cannot name alone: the scope may hold the name


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
    resolve = _resolver(ast, visible, dialect)

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


def _resolver(ast: exp.Expression, visible: dict[str, set[str]],
              dialect: str | None) -> Callable[[exp.Column], tuple[str, str] | None]:
    """The base table and column a column node reads, as `check_values` places it: the scope's own
    answer, then each enclosing scope's, then the statement-wide reading. One resolver, so the
    check refusing a value before the run and the grounding an empty result rests on after it
    (`ground_values`, M146) cannot place the same column on two different base columns."""
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
                return None if answer is _HELD else answer  # held and unplaced: the scope's own skip
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

    return resolve


# --- M146: what an empty result rests on ----------------------------------------------------------
#
# An empty result is the answer when the question's true answer is "none", and it is also what a
# misspelled filter value produces. The no-rows rule declined both. The value index holds the
# evidence that tells them apart, so an empty result is answered only when every string value it
# filters on was checked against its column's indexed values and found: an allow-list. A value
# nothing could check is UNCHECKED, which is unknown and not absent, and declines exactly as an
# ABSENT one does -- "decline when a value is absent" would pass a misspelling on a column the
# index does not cover, and absent values on indexed columns are already refused before the run.

GROUNDED = "grounded"  # its column is indexed, and holds the value
ABSENT = "absent"  # its column is indexed, and does not hold it
UNCHECKED = "unchecked"  # nothing could check it: unknown, which is not absent
TYPED = "typed"  # a date-only ISO string read as a date, or an interval: a bound with no spelling

_EQUALITIES = (exp.EQ, exp.NEQ, exp.NullSafeEQ, exp.NullSafeNEQ)
_RANGES = (exp.GT, exp.GTE, exp.LT, exp.LTE)
_PATTERNS = (exp.Like, exp.ILike, exp.SimilarTo, exp.Glob, exp.RegexpLike, exp.RegexpILike)

# The engines that read a well-formed ISO date string as the date it spells wherever a date is
# compared or cast: DuckDB, measured on 1.5.4 (`d >= '2031-01-01'` against a DATE column compares
# dates); Postgres documents ISO 8601 input as accepted under every DateStyle. Not SQLite, whose
# DATE is an affinity over text, so a date string there is a text comparison and a format mismatch
# returns nothing (a BIRD world comparing '2013/09' with '2013-09'), nor an engine whose string to
# date conversion follows a session format (Oracle's NLS_DATE_FORMAT). Elsewhere such a value stays
# unchecked. Validated here, not left to the engine: DuckDB compares no row against `d = 'garbage'`
# and `DATE 'garbage' < d` when an earlier filter has removed every row, and returns the empty
# result without an error (measured on 1.5.4) -- the empty case exactly.
#
# A proven DATE alone (Codex's reviews and the task reviews of ui-36, measured on 1.5.4): `ts =
# '2024-03-03'` on a TIMESTAMP column is midnight and returned no rows while 15 fell on that day,
# a same-day BETWEEN the same, and on a TIME column `t < '2024-03-03'` reads the string as 00:00;
# `CAST(ts AS TIMESTAMP) = DATE '2024-03-03'` and `greatest(ts, ts) = ...` are the same mistake
# behind an expression. So a date-only string, or a DATE literal, is a bound only where every
# operand it is directly compared with is a bare column the resolver places on a column the
# snapshot declares DATE; against an expression, a cast, a function, or a column of any other or
# unknown type it is unchecked. One exception, since: a plain date-only string that is the
# inclusive start (`ts >= '2031-01-01'`) or the exclusive end (`ts < '2031-01-02'`) of a range on a
# bare naive TIMESTAMP column is a bound, because midnight is what a half-open day range means
# (`_half_open`). The cost, accepted: a true "none" filtered by any other date form on a TIMESTAMP
# column, or through an expression, is declined, naming the date.
#
# A timestamp with its time of day is the TIMESTAMP column's own precision, as safe a bound as a
# number (the controller's addition to fix round 2): a week written to the second on a timestamp
# column is a common genuine "none", declined without this. So a full ISO timestamp, bare or as a
# TIMESTAMP literal, is a bound against bare columns of its own zone-awareness, and on the same
# terms as a date otherwise: unchecked against a DATE or TIME column, an expression, a cast, or a
# column of unknown type, and a TIMESTAMP literal without a time of day is no bound. Two limits
# from Codex's pass on fix round 2 (fix round 3): a naive value is a bound against a naive
# TIMESTAMP column and an offset one against TIMESTAMPTZ, never across -- a naive range on a
# TIMESTAMPTZ column gave 0 rows under UTC and 1 under New York for the same row, measured on
# 1.5.4 -- and only as a range bound (`>`, `>=`, `<`, `<=`, BETWEEN), never `=`, `<>` or IN:
# `ts = '2024-03-03 00:00:00'` is a disguised day filter that a midday row does not meet.
_ISO_DATES_READ_AS_DATES = frozenset({"duckdb", "postgres", "postgresql"})
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?"
                           r"(Z|[+-]\d{2}(:?\d{2})?)?")
_INTERVAL_TEXT = re.compile(r"\s*\d+(\.\d+)?(\s+(microsecond|millisecond|second|minute|hour|day|week"
                            r"|month|quarter|year|decade|century|millennium)s?)?\s*", re.IGNORECASE)


@dataclass(frozen=True)
class FilterValue:
    """One string literal of a statement and what checked it."""

    text: str
    outcome: str  # GROUNDED | ABSENT | UNCHECKED | TYPED
    # Why nothing could check it, for the sentence that names it; "" unless UNCHECKED. One of
    # not_indexed (its column has no indexed values: high-cardinality, not a string, or no index),
    # masked, row_filtered (a table this identity reads through a row filter, whose unfiltered
    # index is never consulted), pattern, range (a text bound), date_on_time (a date-only value
    # against a column holding times), not_on_a_date_column (a date compared with anything but a
    # bare column proven DATE), not_on_a_timestamp_column (a full timestamp compared with anything
    # but a bare column proven TIMESTAMP), not_a_date and not_a_timestamp (a DATE literal naming no
    # real day; a TIMESTAMP literal without a well-formed time of day), typed_other (a typed literal
    # of another type), interval (one not compared with a date or time column), timestamp_equality
    # (a full timestamp in `=`, `<>` or IN), zone_mismatch (a naive timestamp and a zone-aware one),
    # not_a_plain_string (a constant written other than as a plain literal, or a parameter: its
    # `text` is the constant as written), not_a_column (compared with an expression),
    # not_compared, unresolved (the resolver could not place it). "probed" on a value the source
    # itself grounded or ruled out (`confirm_by_source`).
    why: str
    column: str | None  # `table.column`, where the resolver placed it
    # (table, column) for a not_indexed value compared by equality or IN with a bare column, so the
    # source can be asked whether the column holds it; None for every other value.
    placed: tuple[str, str] | None = None


@dataclass(frozen=True)
class ValueGrounding:
    """What an empty result rests on, recorded by the decider on the statement as asked (M146).

    Recorded before the row policy's rewrite, so neither the policy's own literals nor its
    predicate are counted, named or rendered. The value index is built without the filter (M5),
    so a grounded value is one the table holds, not one this caller's rows hold: exactly the case
    where an empty result is honest."""

    values: tuple[FilterValue, ...]
    never_true: str | None  # a predicate that can never hold (`1 = 0`, `x = NULL`, LIMIT 0), as written
    # Each WHERE, HAVING and QUALIFY condition outside another such condition, as asked: the CTEs',
    # the derived tables' and a projected subquery's too, in the order met from the top.
    filters: tuple[str, ...]
    # An output column that is NULL as written (`SELECT NULL, NULL`, `CAST(NULL AS INT) AS x`), so a
    # NULL row is the query's own and no aggregate over nothing. Measured on the saved BIRD runs: a
    # model saying "no answer" in SQL, which with no value to ground passed as a "none".
    null_as_written: str | None = None

    def blocking(self) -> list[FilterValue]:
        """The values an answered result cannot rest on: every one neither grounded nor typed,
        wherever it sits. No projected value is exempt (fix round 2): fix round 1 exempted one that
        provably removed no rows, and each review found that exemption bypassed one level deeper --
        a set-returning call, an alias WHERE reads, an alias a sibling item reads that WHERE,
        QUALIFY or a set-returning call reads. An exemption that must model every way a projection
        reaches the rows does not converge. The cost, accepted: a true "none" whose projection holds
        a string constant (`'big claim' AS kind`) is declined, naming it."""
        return [v for v in self.values if v.outcome not in (GROUNDED, TYPED)]


def _result_queries(node: exp.Expression) -> list[exp.Expression]:
    """The queries whose rows are the statement's result, for `_null_as_written`: the statement,
    and each arm of a UNION, whose projections each give the result's columns."""
    while isinstance(node, exp.Subquery):
        node = node.this
    if isinstance(node, exp.Union):
        return [node, *_result_queries(node.left), *_result_queries(node.right)]
    if isinstance(node, (exp.Select, exp.SetOperation)):
        return [node]
    return []


def _iso_date(text: str) -> bool:
    """A date-only ISO string naming a day that exists (`2024-02-30` does not)."""
    if not _ISO_DATE.fullmatch(text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _moment(text: str) -> str | None:
    """An ISO timestamp with its time of day, naming a moment that exists: "timestamptz" where it
    carries an explicit offset or zone, "timestamp" where it is naive, else None."""
    if not _ISO_DATETIME.fullmatch(text):
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return "timestamp" if parsed.tzinfo is None else "timestamptz"


_T = exp.DataType.Type
_NAIVE_TIMESTAMPS = tuple(getattr(_T, n) for n in (
    "TIMESTAMP", "TIMESTAMPNTZ", "TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP_NS", "DATETIME",
    "DATETIME2", "DATETIME64", "SMALLDATETIME") if hasattr(_T, n))
_ZONED_TIMESTAMPS = tuple(getattr(_T, n) for n in ("TIMESTAMPTZ", "TIMESTAMPLTZ") if hasattr(_T, n))
_TIMESTAMP_KINDS = ("timestamp", "timestamptz")


def _temporal_kind(data_type, dialect: str | None) -> str | None:
    """"date" for a DATE; "timestamp" for a naive timestamp of any precision and "timestamptz" for
    one that is zone-aware, kept apart because a naive value against a zone-aware column reads in
    the session's zone (`_ISO_DATES_READ_AS_DATES`); "time" for a time of day; else None."""
    try:
        built = data_type if isinstance(data_type, exp.DataType) else exp.DataType.build(
            str(data_type), dialect=dialect, udt=True)
    except (SqlglotError, ValueError):
        return None
    if built.is_type(_T.DATE, _T.DATE32):
        return "date"
    if built.is_type(*_NAIVE_TIMESTAMPS):
        return "timestamp"
    if built.is_type(*_ZONED_TIMESTAMPS):
        return "timestamptz"
    return "time" if built.is_type(*exp.DataType.TEMPORAL_TYPES) else None


def _written_as(node: exp.Expression, dialect: str | None) -> str:
    """A constant as the caller wrote it, for the sentence naming it; sqlglot prints no JSON union
    path for DuckDB, and a sentence is no reason to fail the decision."""
    for rendering in (dialect, None):
        try:
            return node.sql(dialect=rendering)
        except (ValueError, SqlglotError):
            continue
    return "a constant that could not be printed"


def _opaque_constant(node: exp.Expression) -> bool:
    """A constant whose value the scan cannot read as a plain string (fix round 3): any leaf that
    carries text and is not a plain literal, a name or a keyword -- sqlglot's E'..' (ByteString),
    N'..' (National), $$..$$ (RawString), a JSON path key, and any type a later sqlglot adds -- and
    every parameter or placeholder, whose value is not in the statement at all. Found by what it
    is not, so a new kind of constant fails closed rather than slipping past as no value."""
    if isinstance(node, (exp.Parameter, exp.Placeholder, exp.SessionParameter)):
        return True
    if isinstance(node, (exp.Literal, exp.Identifier, exp.Var)):
        return False
    values = node.args.values()
    if any(isinstance(v, exp.Expression)
           or (isinstance(v, list) and any(isinstance(x, exp.Expression) for x in v)) for v in values):
        return False
    return any(isinstance(v, str) for v in values)


def _is_condition(node: exp.Expression) -> bool:
    """Whether the node is a whole condition, or one of its conjuncts: a filter's own truth value."""
    child, parent = node, node.parent
    while isinstance(parent, (exp.And, exp.Paren)):
        child, parent = parent, parent.parent
    if isinstance(parent, (exp.Where, exp.Having, exp.Qualify)):
        return True
    return isinstance(parent, exp.Join) and child.arg_key == "on"


def _numeric(node: exp.Expression | None) -> float | None:
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return float(node.this)
        except ValueError:
            return None
    return None


def _half_open(comparison: exp.Expression, literal: exp.Expression) -> bool:
    """Whether `literal` is the inclusive start or the exclusive end of a range on the other
    operand, on whichever side it is written: `col >= x`, `x <= col`, `col < x`, `x > col`."""
    if comparison.right is literal:
        return isinstance(comparison, (exp.GTE, exp.LT))
    return isinstance(comparison, (exp.LTE, exp.GT))


_HOLDS = {exp.EQ: lambda a, b: a == b, exp.NEQ: lambda a, b: a != b, exp.GT: lambda a, b: a > b,
          exp.GTE: lambda a, b: a >= b, exp.LT: lambda a, b: a < b, exp.LTE: lambda a, b: a <= b}


def _never_true(ast: exp.Expression, render_as: str | None) -> str | None:
    """The first predicate that can never hold, as written: the model's way of saying "no answer" in
    SQL (`WHERE 1 = 0`), a comparison with NULL, or a LIMIT of nothing. An empty result there says
    nothing about the data. Read conservatively: such a predicate under an OR counts too, and so
    does one in the projection, which an alias can carry into a filter (fix round 2)."""
    for node in ast.find_all(exp.Boolean, exp.Null, exp.Literal, exp.Limit, exp.Fetch,
                             *_EQUALITIES, *_RANGES):
        if isinstance(node, (exp.Limit, exp.Fetch)):
            count = node.args.get("count") if isinstance(node, exp.Fetch) else node.expression
            if _numeric(count) == 0:
                return _written_as(node, render_as).strip()
            continue
        if isinstance(node, (exp.Boolean, exp.Null, exp.Literal)):
            false = ((isinstance(node, exp.Boolean) and not node.this)
                     or isinstance(node, exp.Null) or _numeric(node) == 0)
            if false and _is_condition(node):
                return _written_as(node, render_as)
            continue
        left, right = node.left, node.right
        if not isinstance(node, (exp.NullSafeEQ, exp.NullSafeNEQ)) and (
                isinstance(left, exp.Null) or isinstance(right, exp.Null)):
            return _written_as(node, render_as)
        a, b = _numeric(left), _numeric(right)
        if a is not None and b is not None and type(node) in _HOLDS and not _HOLDS[type(node)](a, b):
            return _written_as(node, render_as)
    return None


def ground_values(
    ast: exp.Expression,
    visible: dict[str, set[str]],
    values,
    masks: Callable[[str, str], bool] | None = None,
    dialect: str | None = None,
    types: dict[str, dict[str, str]] | None = None,
    render_as: str | None = None,
    row_filtered: set[str] | None = None,
) -> ValueGrounding:
    """Every string literal of the statement, wherever it sits, each with what checked it (M146).

    GROUNDED and ABSENT only where an equality or IN list compares it with a column the resolver
    places on an indexed base column -- `check_values`'s own resolver (M141, M174), so a CTE that
    renames by position is read as the engine reads it -- and that column is neither masked for
    this identity nor on a table it reads through a row filter. Neither is ever looked up: "does
    it exist" is what a mask withholds, and the index is built without the row filter (M5), so on
    such a table it would answer for rows the caller cannot read -- a value present only in hidden
    rows read GROUNDED and its empty result was answered, an absent guess read ABSENT, a membership
    test through `<>` (Codex's review of ui-36). `row_filtered` is folded as `check_values` folds
    it (`names_one_object`). TYPED only where a date-only ISO string or a DATE literal is directly
    compared with bare columns all proven DATE (`_ISO_DATES_READ_AS_DATES`, `types`), a plain
    date-only string is the half-open start or end of a range on a bare naive TIMESTAMP column
    (`_half_open`), or a well-formed interval with a bare date or time column. UNCHECKED everywhere else: a pattern, a
    text range, a date against anything but a proven DATE, another typed literal, a value compared
    with an expression or with nothing, a column the resolver cannot place, one with no indexed
    values, or no index at all (`values` None).

    `dialect` resolves names, as `check_values` is given the target; `render_as` writes the
    filters and the never-true predicate in the dialect the caller is shown (the plan's).
    """
    resolve = _resolver(ast, visible, dialect)
    dates = (dialect or "").lower() in _ISO_DATES_READ_AS_DATES
    render_as = dialect if render_as is None else render_as
    hidden_rows = row_filtered or set()

    def kind(table: str, name: str) -> str | None:
        declared = (types or {}).get(table, {}).get(name)
        return None if declared is None else _temporal_kind(declared, dialect)

    placements: dict[int, tuple[str, str] | None] = {}  # an IN list's column is placed once

    def place(column: exp.Column) -> tuple[str, str] | None:
        if id(column) not in placements:
            placements[id(column)] = resolve(column)
        return placements[id(column)]

    def withheld(table: str, name: str) -> str | None:
        """Why this column's values may not be consulted for this identity, or None."""
        if names_one_object(table, hidden_rows):
            return "row_filtered"
        if masks is not None and masks(table, name):
            return "masked"
        return None

    def against(other: exp.Expression, text: str, ranged: bool,
                half_open: bool = False) -> tuple[str, str, str | None]:
        if not isinstance(other, exp.Column):
            return UNCHECKED, "not_a_column", None
        placed = place(other)
        if placed is None:
            return UNCHECKED, "unresolved", None
        table, name = placed
        column = f"{table}.{name}"
        why = withheld(table, name)
        if why is not None:
            return UNCHECKED, why, column
        held = kind(table, name)
        if dates and _iso_date(text):
            if held == "date":
                return TYPED, "", column
            # Midnight is what a half-open day range means on a naive TIMESTAMP column, so the
            # range's start (`>=`) or end (`<`) counts the day whole and an empty result there is
            # honest. Measured on saved local-model runs: every right answer this rule declined
            # bounded its day this way. Not on TIMESTAMPTZ, whose midnight is the session's
            # zone's.
            if held == "timestamp" and half_open:
                return TYPED, "", column
            if held in (*_TIMESTAMP_KINDS, "time"):
                return UNCHECKED, "date_on_time", column
        moment = _moment(text) if dates else None
        if moment is not None and held in (*_TIMESTAMP_KINDS, "date", "time"):
            if held not in _TIMESTAMP_KINDS:
                return UNCHECKED, "not_on_a_timestamp_column", column
            if not ranged:
                return UNCHECKED, "timestamp_equality", column
            return (TYPED, "", column) if held == moment else (UNCHECKED, "zone_mismatch", column)
        if ranged:
            return UNCHECKED, "range", column
        if values is None or not values.has(table, name):
            return UNCHECKED, "not_indexed", column
        return (GROUNDED if values.contains(table, name, text) else ABSENT), "", column

    def compared_with(node: exp.Expression) -> list | None:
        """The operands a node is directly compared with: the other side, the bounds of a BETWEEN
        it is the subject of (`DATE '2024-03-03' BETWEEN s AND e`), or the list of an IN."""
        parent = node.parent
        if isinstance(parent, _EQUALITIES + _RANGES):
            return [parent.right if parent.left is node else parent.left]
        if isinstance(parent, exp.Between):
            return ([parent.args.get("low"), parent.args.get("high")] if parent.this is node
                    else [parent.this])
        if isinstance(parent, exp.In):
            return list(parent.expressions) if parent.this is node else [parent.this]
        return None

    def as_bound(node: exp.Expression, want: str) -> tuple[str, str, str | None]:
        """A date-only value (`want` "date") or a full timestamp ("timestamp" naive, "timestamptz"
        with an offset) is a bound only where every operand it is directly compared with is a bare
        column placed on a column declared that kind -- and a timestamp only as a range bound:
        `ts = '2024-03-03 00:00:00'` is a disguised day filter (fix round 3). Anything else is
        unchecked."""
        unproven = "not_on_a_date_column" if want == "date" else "not_on_a_timestamp_column"
        partners = compared_with(node) or []
        placed = [place(p) if isinstance(p, exp.Column) else None for p in partners]
        if not partners or None in placed:
            return UNCHECKED, unproven, None
        for spot in placed:
            why = withheld(*spot)
            if why is not None:
                return UNCHECKED, why, f"{spot[0]}.{spot[1]}"
        column = f"{placed[0][0]}.{placed[0][1]}"
        if want == "date":
            timed = next((s for s in placed if kind(*s) in (*_TIMESTAMP_KINDS, "time")), None)
            if timed is not None:
                return UNCHECKED, "date_on_time", f"{timed[0]}.{timed[1]}"
        else:
            if not isinstance(node.parent, _RANGES + (exp.Between,)):
                return UNCHECKED, "timestamp_equality", column
            if all(kind(*s) in _TIMESTAMP_KINDS for s in placed) and any(kind(*s) != want
                                                                        for s in placed):
                return UNCHECKED, "zone_mismatch", column
        if dates and all(kind(*s) == want for s in placed):
            return TYPED, "", column
        return UNCHECKED, unproven, column

    def interval(literal: exp.Literal, text: str) -> tuple[str, str, str | None]:
        """A well-formed interval is a bound where the arithmetic holding it is compared with a
        bare date or time column (`filed > current_date - INTERVAL '30 days'`)."""
        child, parent = literal.parent, literal.parent.parent
        while parent is not None and not isinstance(parent, _EQUALITIES + _RANGES + (exp.Between,)):
            if not isinstance(parent, (exp.Add, exp.Sub, exp.Paren, exp.Cast)):
                return UNCHECKED, "interval", None
            child, parent = parent, parent.parent
        partners = compared_with(child) if parent is not None else None
        spot = (place(partners[0]) if partners and len(partners) == 1
                and isinstance(partners[0], exp.Column) else None)
        if spot is None:
            return UNCHECKED, "interval", None
        column = f"{spot[0]}.{spot[1]}"
        why = withheld(*spot)
        if why is not None:
            return UNCHECKED, why, column
        if dates and _INTERVAL_TEXT.fullmatch(text) and kind(*spot) in ("date", *_TIMESTAMP_KINDS,
                                                                         "time"):
            return TYPED, "", column
        return UNCHECKED, "interval", column

    def classify(literal: exp.Literal) -> tuple[str, str, str | None]:
        parent, text = literal.parent, literal.this
        if isinstance(parent, exp.Cast) and parent.this is literal:
            to = _temporal_kind(parent.to, dialect)
            if to == "date":
                return as_bound(parent, "date") if _iso_date(text) else (UNCHECKED, "not_a_date", None)
            if to in _TIMESTAMP_KINDS:
                moment = _moment(text)
                if moment is None:
                    return UNCHECKED, "not_a_timestamp", None
                # TIMESTAMPTZ '..' of naive text takes the session's zone, and TIMESTAMP '..+02'
                # drops or shifts an offset: the text's zone-awareness must be the type's.
                return as_bound(parent, to) if moment == to else (UNCHECKED, "zone_mismatch", None)
            return UNCHECKED, "typed_other", None
        if isinstance(parent, exp.Interval) and parent.this is literal:
            return interval(literal, text)
        if isinstance(parent, _EQUALITIES + _RANGES):
            other = parent.right if parent.left is literal else parent.left
            return against(other, text, isinstance(parent, _RANGES), _half_open(parent, literal))
        if isinstance(parent, exp.In) and any(e is literal for e in parent.expressions):
            return against(parent.this, text, False)
        if isinstance(parent, exp.Between) and parent.this is not literal:
            return against(parent.this, text, True)
        if isinstance(parent, (exp.In, exp.Between)) and _iso_date(text):
            return as_bound(literal, "date")  # a date as the subject: read its bounds or its list
        if isinstance(parent, (exp.In, exp.Between)) and _moment(text) is not None:
            return as_bound(literal, _moment(text))
        if isinstance(parent, _PATTERNS):
            other = parent.expression if parent.this is literal else parent.this
            placed = place(other) if isinstance(other, exp.Column) else None
            column = f"{placed[0]}.{placed[1]}" if placed else None
            why = withheld(*placed) if placed else None
            return UNCHECKED, why or "pattern", column
        return UNCHECKED, "not_compared", None

    def equality_spot(literal: exp.Literal) -> tuple[str, str] | None:
        """The column a not_indexed value is compared with, as `against` placed it."""
        parent = literal.parent
        if isinstance(parent, _EQUALITIES):
            other = parent.right if parent.left is literal else parent.left
        elif isinstance(parent, exp.In) and any(e is literal for e in parent.expressions):
            other = parent.this
        else:
            return None
        return place(other) if isinstance(other, exp.Column) else None

    found = []
    for node in ast.walk(bfs=False):  # as written, so a decline names them in order
        if node.find_ancestor(exp.JSONPath) is not None:
            continue  # a part of a path, which is classified whole
        if isinstance(node, (exp.JSONPath, exp.JSONPathPart)):
            # A JSON path is a constant whatever its leaves (fix round 4): a root, an integer
            # subscript, a wildcard, a slice or a union carry no text, and `(j->'$[9]') IS NOT NULL`
            # over `[42]` passed with no value to name. The whole path is one unchecked value.
            found.append(FilterValue(_written_as(node, render_as), UNCHECKED, "json_path", None))
        elif isinstance(node, exp.Literal) and node.is_string:
            outcome, why, column = classify(node)
            found.append(FilterValue(node.this, outcome, why, column,
                                     equality_spot(node) if why == "not_indexed" else None))
        elif _opaque_constant(node):
            found.append(FilterValue(_written_as(node, render_as), UNCHECKED, "not_a_plain_string",
                                     None))
    filters: list[str] = []
    clauses = (exp.Where, exp.Having, exp.Qualify)
    for clause in ast.find_all(*clauses):
        # One inside another's condition (a subquery's WHERE in an IN) is written out by it.
        if clause.find_ancestor(*clauses) is None:
            text = _written_as(clause.this, render_as)
            if text not in filters:
                filters.append(text)
    return ValueGrounding(tuple(found), _never_true(ast, render_as), tuple(filters),
                          _null_as_written(ast, render_as))



# --- a value the index could not check, asked of the source -------------------------------------
#
# The value index holds a column's values only where it can hold them all (`build_value_index`), so
# a high-cardinality column -- an id, a lot number -- has none, and every value on it was UNCHECKED:
# a true "none" filtered on one was declined. Measured on saved local-model runs: a third of the
# right answers the sanity layer declined. The source can answer for such a value itself, so once a
# result is empty each not_indexed value is asked of it: found is grounded, missing is absent, and
# a probe that fails or times out leaves the value unchecked, the empty result declined as before.
# Only a not_indexed value carries a column to ask (`FilterValue.placed`): a masked column or a
# row-filtered table never does, because whether a value exists is what those withhold.

PROBE_LIMIT = 8  # distinct values asked per grounding; any beyond stay unchecked


def confirm_by_source(grounding: ValueGrounding, holds: Callable[[str, str, str], bool | None],
                      limit: int = PROBE_LIMIT) -> ValueGrounding:
    """`grounding` with each not_indexed value the source was asked about grounded or ruled out.
    `holds(table, column, text)` answers True, False, or None for a source that said nothing."""
    asked: dict[tuple[str, str, str], bool | None] = {}
    values = []
    for value in grounding.values:
        if value.outcome != UNCHECKED or value.why != "not_indexed" or value.placed is None:
            values.append(value)
            continue
        key = (*value.placed, value.text)
        if key not in asked:
            if len(asked) >= limit:
                values.append(value)
                continue
            asked[key] = holds(*key)
        found = asked[key]
        values.append(value if found is None
                      else replace(value, outcome=GROUNDED if found else ABSENT, why="probed"))
    return replace(grounding, values=tuple(values))


def probe_sql(table: str, column: str, text: str, dialect: str | None) -> str:
    """Whether `table.column` holds `text`, in the source's dialect: the names quoted in their stored
    spelling, as the value index reads them, and the value a string literal sqlglot escapes. The
    table is one quoted name even when it holds a dot: split, a table stored as `a.b` would be read
    as table `b` in schema `a`, an object nobody checked for this caller. So a federated source's
    `catalog.schema.table` does not resolve, and its probe only says nothing, as its value index
    holds nothing (the e2e lane's read of 9a0ed29c)."""
    return (exp.select(exp.Literal.number(1))
            .from_(exp.Table(this=exp.to_identifier(table, quoted=True)))
            .where(exp.EQ(this=exp.Column(this=exp.to_identifier(column, quoted=True)),
                          expression=exp.Literal.string(text)))
            .limit(1)
            .sql(dialect=dialect))


def source_holds(adapter, dialect: str | None, physical: dict[str, str],
                 timeout_s: float) -> Callable[[str, str, str], bool | None]:
    """`holds` for `confirm_by_source`, asking `adapter`. `physical` maps a snapshot table to the
    source's own name for it (`Snapshot.source_bindings`), as the value index does."""
    def holds(table: str, column: str, text: str) -> bool | None:
        sql = probe_sql(physical.get(table, table), column, text, dialect)
        try:
            return adapter.execute_arrow(sql, timeout_s=timeout_s).num_rows > 0
        except Exception:  # noqa: BLE001 -- unanswered, the value stays unchecked and declines
            logger.info("the source did not answer whether %s.%s holds a value; it stays unchecked",
                        table, column, exc_info=True)
            return None
    return holds


def _null_as_written(ast: exp.Expression, render_as: str | None) -> str | None:
    """The first output column of the result's queries that is NULL as written, rendered."""
    for query in _result_queries(ast):
        for projection in query.expressions if isinstance(query, exp.Select) else ():
            inner = projection.unalias()
            while isinstance(inner, (exp.Cast, exp.Paren)):
                inner = inner.this
            if isinstance(inner, exp.Null):
                return _written_as(projection, render_as)
    return None
