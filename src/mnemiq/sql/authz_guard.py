from __future__ import annotations

import re

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect

from mnemiq.sql.functions import FunctionInventory, UnreadableCalls, called_names
from mnemiq.sql.identifiers import resolve, resolve_identifier, resolve_stored, rules_for
from mnemiq.sql.qualify import object_key
from mnemiq.sql.scope import GENERATED_NAME, base_tables, candidate_tables, column_tables
from mnemiq.sql.verdict import Refusal, RefusalCode


def check_unmodelled_source(ast: exp.Expression) -> Refusal | None:
    """Refuse a statement that reads through a source this engine does not model (M65).

    `check_access` walks `base_tables`, and `SELECT x FROM LATERAL (VALUES ((SELECT max(store_id)
    FROM customer))) AS lv(x)` gives it `[]` while scope resolution reports success -- so the
    decider APPROVED a read of `customer` with `customer` absent from `visible`, measured on DuckDB
    and Postgres. An empty list is a normal answer, and nothing downstream can tell "read nothing"
    from "could not see what was read". `views.unrecognised_source` is the whitelist built for
    this, and it was applied to view bodies and to the audit record but not to the statement the
    decider approves. Same rule, one level out: what the engine cannot model, it does not approve.
    """
    from mnemiq.sql.views import unrecognised_source

    shape = unrecognised_source(ast)
    if shape is None:
        return None
    return Refusal(
        code=RefusalCode.UNMODELLED_SOURCE,
        message=(
            f"The query reads through a {shape.lower()} rather than a named table or a subquery "
            "over one, so this engine cannot tell which tables it reads. Write it with tables, "
            "joins and subqueries instead."
        ),
        subject=shape.lower(),
    )


def _pseudo_column(ast: exp.Expression, dialect: str | None) -> Refusal | None:
    """Every reference on Oracle to an unquoted ROWNUM, or to a name Oracle reads as ROWID (M163).

    Read before the column checks, not inside them, because those exempt three places where a
    name is not looked up -- a select alias, a qualifier naming a CTE or subquery, a statement
    with no base table -- and Oracle reads the pseudo-column in each. Measured on Oracle 23, all
    approved before this: `SELECT s.ROWID FROM (SELECT id FROM claim) s` and its CTE form ran and
    returned row addresses, and `s.ROWNUM` and `SELECT id AS rownum FROM claim WHERE rownum <= 5`
    were left for Oracle to reject (ORA-03050, ORA-00923), with the source's error where this
    advice belongs (Codex's review of #161).

    Which spellings, also measured: Oracle folds an unquoted name to upper case and keeps a quoted
    one as written, and reads ROWID wherever the name comes out as ROWID -- `"ROWID"` returns row
    addresses too, and no table can have that column (ORA-03050). ROWNUM it reads unquoted only:
    quoted, `"ROWNUM"` is an ordinary name a table may have, decided below like any other.

    A field too, by the same names (M165). `c.payload."ROWID"` reaches here as a field of
    `payload`, and Oracle reads that field as ROWID under the bound the row cap writes: measured on
    23, `SELECT c.payload."ROWID" FROM claim c` returns the key's value, and the row addresses
    with `FETCH FIRST n ROWS ONLY`. With an alias, or a deeper path, it returns the key. Those
    are refused all the same, so that one rule covers every spelling: telling which ones Oracle
    reads as ROWID would mean modelling how it names the columns of its FETCH FIRST rewrite.
    `JSON_VALUE` reads the key under the bound, so the refusal says to use it.
    """
    if (dialect or "").lower() != "oracle":
        return None
    for node in ast.find_all(exp.Column, exp.Dot):
        identifier = node.this if isinstance(node, exp.Column) else node.expression
        if not isinstance(identifier, exp.Identifier):
            continue
        folded = identifier.name if identifier.quoted else identifier.name.upper()
        if folded == "ROWID" or (folded == "ROWNUM" and not identifier.quoted):
            if isinstance(node, exp.Dot):
                return _pseudo_field_refusal(node, folded)
            return _pseudo_column_refusal(node.name)
    return None


def _pseudo_field_refusal(field: exp.Dot, pseudo: str) -> Refusal:
    """A field named as a pseudo-column, refused saying to read it with JSON_VALUE (M165)."""
    name = field.expression.name
    # The path the statement read, which may go on past the name: `c.payload.a.rowid.b` is `b`.
    while isinstance(field.parent, exp.Dot) and field.parent.this is field:
        field = field.parent
    path, node = [], field
    while isinstance(node, exp.Dot) and isinstance(node.expression, exp.Identifier):
        path.insert(0, node.expression.name)
        node = node.this
    read = "JSON_VALUE on the column, with a path naming the field"
    # A path is written out only for a column behind an alias, the one form Oracle reads a field
    # through, only with plain names, which need no quoting in the path, and only where the path
    # ends: an array step after it (`x[0]`) is not one this writes.
    if (isinstance(node, exp.Column) and node.table and not node.args.get("db")
            and not isinstance(field.parent, exp.Bracket)
            and all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", step) for step in path)):
        read = f"JSON_VALUE({node.sql(dialect='oracle')}, '$.{'.'.join(path)}')"
    return Refusal(
        code=RefusalCode.UNKNOWN_COLUMN,
        message=(f"{name!r} names a field here, and Oracle can read that name as its {pseudo} "
                 f"pseudo-column instead. Read the field with {read}."),
        subject=name,
    )


def _pseudo_column_refusal(name: str) -> Refusal:
    """ROWNUM or ROWID refused saying what to write instead, not as a column the table lacks."""
    if name.lower() == "rownum":
        # Refused, not admitted. ROWNUM is numbered before the ORDER BY of its own block, so
        # `WHERE ROWNUM <= 5 ORDER BY x` sorts five arbitrary rows, and `ROWNUM > 5` is always
        # empty -- both ran and answered wrong once it was admitted (the ui lane, measured on
        # Oracle 23). FETCH FIRST is the one bound that is right in either place, and the row cap
        # reads it (M162). NULLS LAST, on the FIRST key, because Oracle puts NULLs first on a
        # descending key and the logic lint refuses that on a written limit, reading the first
        # key only (M161): one repair, not two.
        return Refusal(
            code=RefusalCode.UNKNOWN_COLUMN,
            message=("ROWNUM is numbered before the ORDER BY of its own query block, so what it "
                     "returns depends on where it is written, and it cannot be checked against "
                     "the row bound. For the first n rows in an order, write ORDER BY with "
                     "NULLS LAST on the first sort key, then FETCH FIRST n ROWS ONLY."),
            subject=name,
        )
    # Not admitted either: a physical row address is not data an answer reports, and Oracle
    # cannot read it through the view an access policy puts over a table (ORA-01446, measured),
    # so admitting it would fail every governed statement that names it.
    return Refusal(
        code=RefusalCode.UNKNOWN_COLUMN,
        message=("ROWID is a row's physical address, not data an answer reports, and it "
                 "cannot be read through an access policy. Leave it out."),
        subject=name,
    )


def _unknown_column(name: str) -> Refusal:
    return Refusal(
        code=RefusalCode.UNKNOWN_COLUMN,
        message=(
            f"No table in this query has a column {name!r}. "
            "Use only the columns listed on the schema cards."
        ),
        subject=name,
    )


# The clauses of its own SELECT in which each engine reads a select alias, as a bare item
# (`ORDER BY x`, `GROUP BY x`) and inside an expression (`ORDER BY x + 0`, `WHERE x = 1`),
# measured on DuckDB and SQLite (M164's fix: DuckDB reads one in a later projection, in WHERE,
# GROUP BY, HAVING, QUALIFY, ORDER BY and inside a window's ORDER BY or PARTITION BY, not in
# a JOIN's ON, and inside an expression everywhere but GROUP BY, where it fails to bind; SQLite
# in WHERE, GROUP BY, HAVING, ORDER BY and an ON, bare and inside an expression alike, not in
# a projection or a window), on Oracle 23 for a bare ORDER BY item (the register's measurement;
# GROUP BY and HAVING there not measured, so not admitted), and as documented for Postgres
# (GROUP BY and ORDER BY, where an output name is read as a bare item alone and inside an
# expression as an input column: `ORDER BY xmin + 0` reads the system column, the review gate)
# and MySQL (GROUP BY, HAVING and ORDER BY; inside an expression not measured, so not
# admitted). An engine not listed reads a bare alias in ORDER BY alone, which every engine
# does: fail-closed, since an alias admitted where the engine reads a column, a pseudo-column
# or a function instead is M164's leak, and an alias refused where the engine would read it
# costs a retry. Where a column and an alias share the name the engine reads the column
# (measured on both), which is checked by its name.
_ALIAS_BARE: dict[str | None, frozenset[str]] = {
    "duckdb": frozenset({"select", "where", "group", "having", "order", "qualify", "window", "distinct"}),
    "sqlite": frozenset({"where", "group", "having", "order", "join"}),
    "postgres": frozenset({"group", "order", "distinct"}),  # DISTINCT ON reads a bare alias as ORDER BY does, measured on Postgres 16 (Codex's fifth review of #177)
    "postgresql": frozenset({"group", "order", "distinct"}),
    "mysql": frozenset({"group", "having", "order"}),
    "oracle": frozenset({"order"}),
    None: frozenset({"order"}),
}
_ALIAS_IN_EXPRESSION: dict[str | None, frozenset[str]] = {
    "duckdb": frozenset({"select", "where", "having", "order", "qualify", "window"}),
    "sqlite": frozenset({"where", "group", "having", "order", "join"}),
    None: frozenset(),
}


def _is_distinct_on(node: exp.Expression | None) -> bool:
    """A SELECT's own `DISTINCT ON (...)`: sqlglot wraps an aggregate's DISTINCT in the same node
    (`count(DISTINCT x)`), which is part of its clause's expression (the review gate on the
    fourth commit of #177, where every alias inside one was read as a DISTINCT ON item and
    `HAVING count(DISTINCT x) > 0` refused on SQLite, which runs it)."""
    return isinstance(node, exp.Distinct) and isinstance(node.parent, exp.Select) and node.args.get("on") is not None


# The engines measured reading redundant parentheses around a bare item through: an engine not
# listed keeps them an expression, fail-closed, as the table above admits nothing unmeasured
# (the review gate on the sixth commit: Oracle reads `ORDER BY ((alias))` as the bare item, measured on Oracle 23 (its one clause with an alias)).
_READS_THROUGH_PARENTHESES = frozenset({"duckdb", "postgres", "postgresql", "mysql", "sqlite", "oracle"})


def _is_wrapper(node: exp.Expression, dialect: str | None) -> bool:
    """Parentheses around a bare item that the engine reads through: a grouping set's, a
    DISTINCT ON list's, a one-item set among them (`GROUPING SETS ((x), ())` parses `(x)` as a
    Paren), structure on every engine; and, on the engines measured (`_READS_THROUGH_PARENTHESES`),
    redundant ones around a GROUP BY, ROLLUP, CUBE, GROUPING SETS, DISTINCT ON or ORDER BY item
    to any depth: `GROUP BY ((secret))` reads the alias as `GROUP BY secret` does on DuckDB, Postgres, MySQL, SQLite and Oracle 23,
    measured, where the parentheses made the item an expression and the table refused it
    (Codex's fourth review of #177). A list in GROUP BY (`GROUP BY (secret, id)`) is its items on
    DuckDB and Postgres and an error on MySQL and SQLite, measured; in ORDER BY it is a row
    constructor, an expression, which Postgres reads as input columns (measured: no alias
    bound), so it stays one there. Parentheses beside anything else (`(secret) + 0`) are an
    expression still."""
    if not isinstance(node, (exp.Tuple, exp.Paren)):
        return False
    parent = node.parent
    if isinstance(parent, (exp.Rollup, exp.Cube, exp.GroupingSets)) or _is_distinct_on(parent):
        return True
    if (dialect or "").lower() not in _READS_THROUGH_PARENTHESES:
        return False
    if isinstance(parent, exp.Group):
        return True
    if isinstance(node, exp.Tuple):
        # A list is structure directly under its clause alone: wrapped in parentheses it is a row
        # constructor, an expression, and the parentheses above it do not make it bare (Codex's
        # fifth review of #177: `ORDER BY ((ctid, id))` inherited the outer parentheses' bare
        # reading through the recursion, where Postgres reads the system column inside the row).
        return False
    if isinstance(parent, exp.Ordered):
        return True
    return _is_wrapper(parent, dialect)


def _windowed(node: exp.Expression) -> bool:
    """Whether an aggregate (a Func or a WITHIN GROUP, or the FILTER over one) is a window
    function's, whose FILTER predicate and ORDER BY DuckDB binds an output alias in, measured
    (`count(*) FILTER (WHERE secret = 1) OVER ()`, `string_agg(id ORDER BY secret) OVER ()`),
    where a plain aggregate's bind none (Codex's fourth review of #177); Postgres binds none in
    either, and its table admits no `window`."""
    if isinstance(node.parent, exp.Filter):
        node = node.parent
    return isinstance(node.parent, exp.Window)


def _clause_of(column: exp.Column, dialect: str | None) -> tuple[str, bool]:
    """The clause of its own SELECT a column sits in, and whether it is the whole item there:
    `where`, `group`, `having`, `order`, `qualify`, `join` (an ON), `window` (an OVER's own
    ORDER BY or PARTITION BY, whatever clause holds it, and a window function's own FILTER
    predicate and ORDER BY, which DuckDB binds an alias in, measured: Codex's fourth review of
    #177), `aggregate` (a plain aggregate's own ORDER
    BY or FILTER predicate, which bind to source columns alone: DuckDB rejects an alias there,
    measured, and with a hidden column of the alias's name read it, Codex's first and third
    reviews of #177), `limit` (a
    LIMIT or OFFSET, where no engine measured reads an alias), or `select`, the projection
    list, or `distinct` (the SELECT's own DISTINCT ON item, `_is_distinct_on`: DuckDB reads a
    bare alias there and the source column inside an expression, measured, where `DISTINCT ON
    (secret + 0)` beside an alias `secret` deduplicated on the hidden column and was admitted
    as the projection list's expression, Codex's second review of #177; an aggregate's own
    DISTINCT is its clause's expression); a set operation's ORDER BY is `order`,
    sqlglot holding it under the operation. A ROLLUP, CUBE or GROUPING SETS wrapper, a
    grouping set's parentheses and a DISTINCT ON list's are structure, not an expression:
    `GROUP BY ROLLUP(x)` is a bare item (the same review, where the wrapper made it an
    expression and DuckDB's table refused it). Redundant parentheses around a bare item are
    structure on the engines measured alone (`_is_wrapper`, `_READS_THROUGH_PARENTHESES`):
    `GROUP BY ((x))` is a bare item there (the fourth) and an expression on any other engine,
    fail-closed. Where an alias and a source column
    share a name the engine reads the column first everywhere but a bare ORDER BY item (and
    HAVING on DuckDB), measured, which the resolver binds under the complete schema and the
    grant alone cannot see: the register's residual, recorded at #167."""
    clause, node, bare = None, column.parent, True
    while node is not None and not isinstance(node, (exp.Select, exp.SetOperation)):
        if isinstance(node, (exp.Window, exp.WindowSpec)):
            return "window", bare
        if isinstance(node, exp.Order) and isinstance(node.parent, (exp.Func, exp.WithinGroup)):
            # WITHIN GROUP (ORDER BY ...) is no Func to sqlglot (the gate on the second commit),
            # and under OVER it is not measured on DuckDB, so it stays the aggregate's, fail-closed
            # (the gate on the sixth commit); a Func's own ORDER BY under OVER is the window's.
            windowed = isinstance(node.parent, exp.Func) and _windowed(node.parent)
            return ("window" if windowed else "aggregate"), False
        if isinstance(node, exp.Where) and isinstance(node.parent, exp.Filter):
            # An aggregate's FILTER (WHERE ...) binds to source columns alone, as its ORDER BY
            # does: read as the SELECT's WHERE, where DuckDB admits an alias, `count(*) FILTER
            # (WHERE secret = 1)` beside an alias `secret` counted the hidden column's rows, and
            # DuckDB rejects the alias there with no such column (Codex's third review of #177).
            # A window function's FILTER is the window's, where DuckDB binds the alias (the fourth).
            return ("window" if _windowed(node.parent) else "aggregate"), False
        if isinstance(node, (exp.Limit, exp.Offset)):
            return "limit", False
        kind = next((name for cls, name in ((exp.Where, "where"), (exp.Group, "group"), (exp.Having, "having"),
                                            (exp.Order, "order"), (exp.Qualify, "qualify"), (exp.Join, "join"))
                     if isinstance(node, cls)), None)
        if kind is None and _is_distinct_on(node):
            kind = "distinct"  # the SELECT's own DISTINCT ON, not an aggregate's DISTINCT (`count(DISTINCT x)`), the same node to sqlglot (the gate)
        if kind is not None:
            if clause is None:
                clause = kind
        elif _is_wrapper(node, dialect):
            pass  # parentheses the engine reads through: a grouping set's, a DISTINCT ON list's, redundant ones around a bare item
        elif not isinstance(node, (exp.Ordered, exp.Alias, exp.Rollup, exp.Cube, exp.GroupingSets)):
            bare = False  # the column sits inside an expression of the clause
        node = node.parent
    if clause in ("where", "having", "qualify", "join"):
        bare = True  # a predicate: an output name there is always inside an expression
    return clause or "select", bare


def check_access(ast: exp.Expression, visible: dict[str, set[str]],
                 dialect: str | None = None,
                 columns: dict[str, set[str]] | None = None) -> Refusal | None:
    """Re-check every table and column the query touches against what the identity may see.

    `columns` is the schema the resolver binds a derived table's star through (M181): the grant's
    and the policy's columns together, from `known_columns`, so a column hidden from this
    identity still binds to the table that has it and is refused as unknown there; without it,
    the grant's alone.

    `dialect` is the ENGINE that will run this, not the one it was parsed as. Identifier folding differs between them -- Oracle folds unquoted names up, the rest fold them down, and quoting is significant in Postgres and Oracle but not DuckDB -- so a resolver keyed on the wrong one answers for the wrong engine. Threading it to SOME call sites and not others is worse than either: `check_access` said a name was a CTE while the audit list said it was a base read, in the same statement.

    Retrieval scoping (the semantic store) means the model was never *shown* a forbidden
    table. It can still *name* one -- `users`, `employees`, `salaries` are in every schema it
    was trained on. This is the lock that makes naming it useless.
    """
    # tables, and the aliases that stand for them. `base_tables` -- not `find_all` minus a flat
    # set of CTE names -- because a reference inside a CTE body naming that same CTE reads the
    # base table, and skipping it let an ungranted table through (M31).
    alias_to_table: dict[str, str] = {}
    named: set[int] = set()
    resolved = column_tables(ast, dialect, columns if columns is not None else visible, named=named)
    for table in base_tables(ast, dialect):
        name = object_key(table)
        if name not in visible:
            return Refusal(
                code=RefusalCode.UNAUTHORIZED_TABLE,
                message=f"You may not query {name!r}. Answer using only the tables provided.",
                subject=name,
            )
        alias_to_table[table.alias_or_name] = name

    pseudo = _pseudo_column(ast, dialect)
    if pseudo is not None:
        return pseudo

    if not alias_to_table:
        return None  # a query over CTEs alone; their sources were checked above

    # Names are compared as the engine resolves them where that is exact (M170): DuckDB and
    # SQLite keep a name's case without telling names apart by it, so `SECRET` and a stored
    # `Secret` are one column, and both sides fold. Where quoting keeps case (Postgres, Oracle)
    # both sides stay as typed, as before: folding the stored side would let an unquoted
    # spelling meet a column created with quotes, which the engine reads as ANOTHER column (the
    # review gate's case, `"Amount"` beside `amount`), and the catalog boundary there -- table
    # keys and row filters are as typed too -- is one decision, deferred to the register. One
    # exception, in the HIDDEN direction alone: a USING key is also read as the engine folds it,
    # against the hidden names folded the same way (below), since with the complete schema the
    # key carries no positive check and a spelling the engine folds onto a hidden name passed
    # as typed; that refuses more, never less. The grant side stays as typed, as here.
    folds = not rules_for(dialect).quoting_preserves_case

    def as_engine(names) -> set[str]:
        return {resolve_stored(n, dialect) for n in names} | set(names) if folds else set(names)

    def written(identifier: exp.Expression | None, text: str) -> str:
        if folds and isinstance(identifier, exp.Identifier):
            return resolve_identifier(identifier, dialect) or text
        return text

    referenced = set(alias_to_table.values())
    known_columns = as_engine({c for t in referenced for c in visible[t]})

    # A JOIN KEY is not an `exp.Column`, and it reads the column all the same (M103 closed the
    # denied and masked key in `check_cls`); a key HIDDEN from the grant was read by nothing
    # (M200): `claim JOIN (SELECT 1 AS id, 'Z' AS secret) x USING (id, secret)` with `secret`
    # absent from the grant was approved, and whether rows match says whether the hidden value
    # is the guess, through a derived star the same. A USING key is refused where a base table
    # the join's SELECT reaches (through a derived table or a CTE too) has it hidden, or is not
    # listed in full so that hiding cannot be ruled out; a NATURAL join names nothing, so one
    # over any such table with a hidden column, or not listed in full, is refused.
    complete = getattr(columns, "complete", frozenset()) if columns is not None else frozenset()
    listed = columns if columns is not None else visible
    def cte_body(table: exp.Table) -> exp.Expression | None:
        """The body of the CTE this reference names, from the nearest enclosing WITH that defines
        the name outward (a name defined in two scopes is the inner one's: the review gate), as
        the engine scopes it: a reference inside a CTE's own body, or inside a sibling's, sees
        only the siblings declared before that CTE, so `WITH pair AS (SELECT * FROM pair)` reads
        the base table inside its body (the review gate's third round)."""
        # Names meet as the engine resolves them (M170): `WITH W AS (...) SELECT ... FROM w` reads
        # the CTE on DuckDB, where a granted base table `w` would otherwise be read instead
        # (Codex's third review of #170). A qualified reference (`pg.claim`) never names a CTE,
        # whatever a CTE is called: it is the base table (the review gate on that fix, where
        # folding the bare name alone let a CTE named `claim` capture it).
        if table.args.get("db") or table.args.get("catalog"):
            return None
        # Resolved by the shared identifier rules for every engine, not folded only where the
        # engine folds: Oracle keeps a quoted `"claim"` and upper-folds an unquoted `claim`, so
        # the two are different names there (Codex's fourth review of #170).
        name = resolve_identifier(table.this, dialect) or object_key(table)
        enclosing, above = [], table.parent
        while above is not None:
            if isinstance(above, exp.CTE):
                enclosing.append(above)
            above = above.parent
        node: exp.Expression | None = table
        while node is not None:
            with_ = node.args.get("with_") if isinstance(node, exp.Expression) else None
            if isinstance(with_, exp.With):
                ctes = list(with_.expressions)
                own = next((c for c in enclosing if c.parent is with_), None)
                candidates = ctes[:ctes.index(own)] if own is not None else ctes
                for cte in candidates:
                    declared = resolve_identifier(cte.args.get("alias"), dialect) or cte.alias_or_name
                    if declared == name:
                        return cte.this
            node = node.parent
        return None

    def bases(node: exp.Expression, following: frozenset[int] = frozenset()) -> set[str]:
        """The base tables an input reads, a CTE name followed to its body (a CTE defined in an
        outer scope reads as a base table inside the join's own SELECT: the review gate); a
        name whose body is already being followed is read as a base table of that name, never
        dropped (fail-closed). Every table reference of the input, not the scope walk's base
        reads alone: that walk, over the input on its own, dropped a reference the engine
        binds to a CTE of an OUTER scope (`WITH R AS (SELECT * FROM claim) ... (WITH r AS
        (SELECT * FROM r) SELECT * FROM r) d JOIN ... USING (secret)`, where the inner body's
        `r` is the outer `R` folded), whose scope the input's tree does not hold, so the
        input read no base table and the hidden key went unchecked (Codex's third review of
        #174). `cte_body` reads the enclosing WITHs outward from the reference, the outer
        one included, and a name no WITH declares is the base table."""
        found: set[str] = set()
        for table in node.find_all(exp.Table):
            body = cte_body(table)
            if body is None or id(body) in following:
                found.add(object_key(table))
            else:
                found |= bases(body, following | {id(body)})
        return found

    for join in ast.find_all(exp.Join):
        # The join's own inputs: what stands left of it in its SELECT (the FROM and the joins
        # before it) and its right side, each followed to the base tables it reads; not every
        # table the SELECT mentions, a WHERE's subquery among them (the review gate).
        select = join.parent if isinstance(join.parent, exp.Select) else None
        if select is not None and select.args.get("from_") is not None:
            earlier = select.args.get("joins") or []
            before = next(i for i, j in enumerate(earlier) if j is join)  # by identity: two
            inputs = [select.args["from_"].this, *(j.this for j in earlier[:before]), join.this]  # equal joins
            reached = set().union(*(bases(i) for i in inputs))
        else:
            reached = bases(ast)
        # Every base table here passed the grant above, whatever alias it carries: the alias map
        # is one name per alias and a reused alias in another scope would drop a table from it
        # (Codex's second review: `claim t JOIN ... USING (secret) WHERE EXISTS (SELECT 1 FROM
        # sink t)` dropped claim), so the reached set is not narrowed by it.
        reached = {t for t in reached if t in visible}
        hidden_on = {t: as_engine(set(listed.get(t, ())) - set(visible[t])) for t in reached}

        def folded(names) -> set[str]:
            # Every hidden name as the engine would fold it typed unquoted: on Oracle a grant
            # spells a column `secret` or `SECRET` as its author did (the catalog boundary
            # above), and both are the one column an unquoted `USING (secret)` names. For the
            # hidden set only: folding a stored name is fail-closed here and fail-open on the
            # grant side (a quoted-case `"Amount"` granted would let `USING (amount)` through
            # where the engine reads another column: the review gate).
            return {resolve(n, quoted=False, dialect=dialect) for n in names}

        for key in join.args.get("using") or ():
            # Hidden under EITHER reading is refused: as the column check reads a column (typed,
            # folded only where quoting does not preserve case), or as the engine resolves it,
            # against the hidden names folded the same way. As typed alone, `USING (secret)`
            # passed a hidden `SECRET` on Oracle with the complete schema, where `USING
            # (SECRET)` was refused (Codex's fifth review); folded alone, a hidden `secret`
            # spelled so was missed (the review gate). Without the complete schema the key is
            # held to the grant as typed, as a column is.
            spelled = key.name if hasattr(key, "name") else str(key)
            typed = written(key, spelled)
            as_engine_reads = resolve_identifier(key, dialect) or spelled
            for table in sorted(reached):
                hidden = typed in hidden_on[table] or as_engine_reads in folded(hidden_on[table])
                granted = typed in as_engine(visible[table])
                if hidden or (table not in complete and not granted):
                    return Refusal(
                        code=RefusalCode.UNKNOWN_COLUMN,
                        message=(f"The join key {spelled!r} is not a column of {table!r} you may read. "
                                 "Join with an explicit ON over the columns provided."),
                        subject=spelled,
                    )
        if (join.args.get("method") or "").upper() == "NATURAL":
            for table in sorted(reached):
                if hidden_on[table] or table not in complete:
                    return Refusal(
                        code=RefusalCode.UNKNOWN_COLUMN,
                        message=(f"A NATURAL join reads whatever columns these tables share, and "
                                 f"{table!r} has columns you may not read, or is not listed in "
                                 "full. Join with an explicit ON or USING naming the columns."),
                        subject=table,
                    )

    # A SELECT alias is a name the query invents for an expression, and GROUP BY / ORDER BY /
    # HAVING may refer to it: `SELECT year(d) AS y, count(*) FROM t GROUP BY y` is ordinary,
    # correct SQL -- and without this the engine could not answer "count X by year" at all,
    # on the engines that read an alias there (`_ALIAS_BARE`, `_ALIAS_IN_EXPRESSION`: on Oracle
    # a bare name in WHERE is a column, a pseudo-column or a function, M164). The
    # aliases of the reference's OWN SELECT, since no engine resolves a name to another scope's
    # alias (the ui lane's replay of #167: statement-wide, `SELECT (SELECT max(secret) FROM
    # claim) AS secret FROM wide` let the inner hidden `secret` through on the outer alias). In
    # ORDER BY and HAVING, which sqlglot resolves to the projections before listing a scope's
    # columns so the resolver never visits them, a name is also any column the SELECT projects
    # by name (`ORDER BY d_week_seq1` over `SELECT d_week_seq1 ... FROM (...) y`), itself
    # checked as a projection.
    # Where the scopes could not be resolved at all, the statement-wide set stands, the rule the
    # fallback has always had (TPC-DS q23 and q24, and a WITH inside an UPDATE's SET).
    statement_aliases = {written(alias.args.get("alias"), alias.alias)
                         for alias in ast.find_all(exp.Alias) if alias.alias}
    own_aliases: dict[int, set[str]] = {}
    own_outputs: dict[int, set[str]] = {}

    def aliases_of(column: exp.Column) -> set[str]:
        if resolved is None:
            return statement_aliases
        clause, bare = _clause_of(column, dialect)
        reads = _ALIAS_BARE if bare else _ALIAS_IN_EXPRESSION
        if clause not in reads.get((dialect or "").lower(), reads[None]):
            # The engine reads no select alias in this clause, so the name is a column, a
            # pseudo-column or a no-argument function to it: on Oracle, `SELECT id AS mq_leak
            # FROM claim WHERE mq_leak = 1` ran a function of that name and returned every row
            # (M164, measured on Oracle 23), and `AS ORA_ROWSCN ... WHERE ORA_ROWSCN IS NOT NULL`
            # filtered on the pseudo-column, where the alias had let the bare name through.
            return set()
        select = column.find_ancestor(exp.Select, exp.SetOperation)
        while isinstance(select, (exp.SetOperation, exp.Subquery, exp.Paren)):
            # A set operation's ORDER BY names its output, which is the first branch's (the
            # review gate: `SELECT id AS x FROM claim UNION SELECT id FROM other ORDER BY x`),
            # a parenthesised branch unwrapped (sqlglot reads it as a subquery).
            select = select.left if isinstance(select, exp.SetOperation) else select.this
        if not isinstance(select, exp.Select):
            return set()
        clause = column.find_ancestor(exp.Order, exp.Having, exp.Select, exp.SetOperation)
        # A window's ORDER BY inside OVER (...) binds to source columns, not the output.
        ordering = isinstance(clause, (exp.Order, exp.Having, exp.SetOperation)) and not (
            isinstance(clause, exp.Order) and isinstance(clause.parent, (exp.Window, exp.WindowSpec)))
        store = own_outputs if ordering else own_aliases
        names = store.get(id(select))
        if names is None:
            names = set()
            for projection in select.selects:
                if isinstance(projection, exp.Alias) and projection.alias:
                    names.add(written(projection.args.get("alias"), projection.alias))
                elif store is own_outputs and isinstance(projection, exp.Column):
                    names.add(written(projection.this, projection.name))
            store[id(select)] = names
        return names

    for column in ast.find_all(exp.Column):
        name = written(column.this, column.name)  # the refusal keeps the spelling written
        qualifier = column.table
        # Resolved per scope, not from one flat map: an alias can name two different tables in
        # one statement, and keeping only the last let a column be checked against the wrong
        # one (Codex review, 2026-08-12). An unqualified column resolves too where its scope's
        # only source is a derived table (M181), so both spellings of a read through a star
        # meet the same table here.
        table = resolved.get(id(column)) if resolved is not None else None
        if table is None and id(column) in named:
            # A named projection of its own derived source, or a CTE's or LATERAL's column-list
            # name (`WITH w(n) AS (SELECT 1) ... SELECT n FROM w`): the nodes it stands for are
            # checked in that scope, and the name says nothing about a base table here.
            continue
        if table == GENERATED_NAME:
            return Refusal(
                code=RefusalCode.UNKNOWN_COLUMN,
                message=(f"{column.name!r} is a name the engine makes up for a duplicate column "
                         "of a subquery's output, which says nothing about which column is "
                         "read. Alias each column of the subquery distinctly and read it by "
                         "that name."),
                subject=column.name,
            )
        if table is not None:
            if name not in as_engine(visible[table]):
                return Refusal(
                    code=RefusalCode.UNKNOWN_COLUMN,
                    message=f"{table!r} has no column {column.name!r}. Use only listed columns.",
                    subject=column.name,
                )
        elif qualifier:
            if resolved is None:
                # Unreadable scopes: fall back to the rule for an unqualified column rather
                # than to the flat map that was the defect. Same guarantee, no worse.
                if name not in known_columns and name not in aliases_of(column):
                    return _unknown_column(column.name)
            # Else the qualifier names a CTE or a subquery alias whose own projection names the
            # column, checked in that scope.
        elif name not in known_columns and name not in aliases_of(column):
            return _unknown_column(column.name)

    return None





def _known_function_names() -> frozenset[str]:
    """Every function name sqlglot can model, in ANY dialect.

    Not `exp.Func` subclasses alone: those carry `sql_names()` for 623 names and miss `now`
    and `date_part`, which live in each dialect's PARSER table rather than on a class. Missing
    them is not academic -- see the guard below for what it cost.
    """
    names: set[str] = set()

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            try:
                names.update(n.upper() for n in sub.sql_names())
            except Exception:  # a class that does not declare names is simply not a source
                pass
            walk(sub)

    walk(exp.Func)
    for dialect in Dialect.classes.values():
        try:
            names.update(k.upper() for k in dialect.parser_class.FUNCTIONS)
        except Exception:
            continue
    return frozenset(names)


_KNOWN_FUNCTIONS = _known_function_names()


def check_unmodelled_calls(
    ast: exp.Expression,
    inventory: FunctionInventory | None = None,
    *dialects: str,
) -> Refusal | None:
    """Refuse a call this engine cannot model, because it cannot say what such a call reads.

    `check_access` re-checks every `exp.Table` against the identity's visible set, and the RLS
    rewrite wraps each one. A function in PROJECTION position produces no `exp.Table` at all,
    so both walk past it: measured, `SELECT customer_rows() AS x` and
    `SELECT pg_read_file('/etc/passwd')` were APPROVED with `tables=[]` for a caller granted
    only `claim`, while `SELECT id FROM customer` was correctly refused in the same run (M43).

    A **whitelist**, the same inversion `unrecognised_source` took for source shapes, and for
    the reason its docstring gives: enumerate dangerous functions and an unlisted one passes,
    with every review round finding another nobody thought of.

    The allowlist is sqlglot's whole function vocabulary, ACROSS DIALECTS, and that detail is
    the guard. The first version tested `isinstance(node, exp.Anonymous)`, which asks whether
    the ONE dialect being parsed happens to model the name. Production parses as `duckdb`
    (`Agent.dialect` from the adapter), sqlglot models `now` and `date_part` only under
    postgres, and so the first version refused `SELECT id FROM claim WHERE created_at < now()`
    -- measured through `decide`, on the shipped default. Its control test asserted the
    opposite while pinning `dialect="postgres"`, the one dialect where the claim held.

    Not a claim that an unknown call reads data. It is a claim that the decider cannot tell,
    and the decider's premise is that it sees every table a query reads. An opaque call makes
    that premise false, so the honest verdict is that this query cannot be decided.

    The allowlist alone was names, not identities: a UDF named `median` is modelled by sqlglot
    and passed. Measured through this function on the shipped default, with a macro in an
    attached read-only DuckDB file, `SELECT median(id) FROM claim` was APPROVED with
    `tables=['claim']` and returned an SSN from a table the identity was never granted. So did
    `SELECT count(*) FROM claim`, against a macro named `count_star` -- a name that appears in
    no spelling of that query, because it is DuckDB's binder name for `COUNT(*)`.

    `inventory` closes that, in the two shapes the leak comes in. A call reachable under its
    OWN name must be spelled, so `called_names` finds it however sqlglot rewrote the node, and
    the refusal names it. A call reachable under a BUILTIN's name is not derivable at all, and
    cannot happen unless the source defines a name a builtin also has -- so that case condemns
    the whole SOURCE, every statement against it, whether or not anything here looks like a
    call. `SELECT id + 1 FROM claim` was the measurement: no `exp.Func` node in it, `+` shadowed
    by a macro, and an SSN in the result.

    A third arrival, and it is the one that has to be written down rather than inferred: an
    inventory that was ASKED and could not answer carries the same empty `names` as one that
    answered "none", and reaches `_cannot_resolve` for it. `never_asked` does not, because that
    is every fixture and the state the engine shipped in -- without an inventory this is the
    allowlist alone, which is where it started.

    **WHICH SOURCES THAT ACTUALLY COVERS, because the paragraphs above read as if it were all of
    them.** Only an adapter implementing `user_functions` is asked -- that method alone decides
    `never_asked`; `virtual_columns` is a separate path, and an adapter missing it yields an
    empty opaque set rather than landing here. When this paragraph was written the DuckDB
    family was the only one asked, and it named `OracleAdapter` as the live adapter
    implementing neither, so an Oracle schema function called `median` passed here exactly as
    every source did before the inventory existed (M99).

    **Oracle answers both now**, at different times: `user_functions` at `ce4e16a`,
    `virtual_columns` three commits later at `77d24b0`. So every `names`-driven rule below
    applies to it. `binder_prefers_builtins` is True there, which rules out the coarse shadow
    arm and leaves two: `called & names`, and -- for any statement the scan cannot read at all
    -- the `UnreadableCalls` refusal, which fires whatever the scan returns. `adapters/oracle.py`
    records that second arm as the correction to a note of its own that framed the first as the
    only one.

    An adapter still implementing `user_functions` gets asked; one that does not gets
    `never_asked` and the bare allowlist, which remains the reason this is not fail-closed by
    default: that would refuse every call-bearing query on the adapters that cannot answer. The
    older Oracle exposure is filed as deployment preconditions M66 and M71, which
    `adapters/oracle.py` documents at length.
    """
    for call in ast.find_all(exp.Anonymous):
        name = str(call.this)
        if name.upper() in _KNOWN_FUNCTIONS:
            continue
        return Refusal(
            code=RefusalCode.UNMODELLED_CALL,
            message=(
                f"{name}() is a function this engine cannot model, so it cannot confirm what "
                "the query reads. Answer using only the listed tables and columns and "
                "standard SQL functions."
            ),
            subject=name,
        )

    inventory = inventory if inventory is not None else FunctionInventory.never_asked()
    if inventory.asked and not inventory.available:
        # Asked and could not answer, which is not the same as answering "none" -- and the
        # empty `names` an `unavailable` inventory carries would otherwise fall through every
        # test below and clear the statement. The collapse this type exists to prevent, in the
        # guard written to use it: measured, a raising `user_functions()` on a source holding a
        # `median` macro left `SELECT median(id) FROM claim` approved.
        return _cannot_resolve(inventory)
    # A whole-source answer naming nothing clears every statement. A narrow one (M101) says only
    # that none of the names it was ASKED about is defined, so it clears nothing until the
    # statement's own calls are known to be among them.
    if inventory.asked_names is None and not (inventory.names or inventory.unresolvable):
        return None

    if inventory.may_shadow_a_builtin:
        # No `exp.Func` test in front of this, deliberately, and that absence is the fix for a
        # leak this function's first version had. `exp.Add`, `exp.DPipe` and `exp.AtTimeZone`
        # are not `exp.Func` subclasses, while DuckDB lists `+`, `||` and `timezone` as
        # internal functions a macro can shadow -- measured, `SELECT id + 1 FROM claim` was
        # APPROVED and returned an SSN with a `"+"` macro in the source. Enumerating the node
        # types that bind to a catalogue entry is the blocklist this codebase keeps refusing to
        # write; proving a statement CALL-FREE is as hard as naming its calls, so a source
        # whose names cannot be trusted answers nothing.
        return _cannot_resolve(inventory)

    try:
        called = called_names(ast, *dialects)
    except UnreadableCalls:
        # Cannot enumerate, so cannot clear. The same verdict as shadowing, and for the same
        # reason -- an empty set read as "no calls" would clear all of them -- but its own
        # sentence: this one is about THIS statement, and blaming the adapter for it would
        # send the deployer to fix a catalogue method that is working.
        return _cannot_resolve(inventory, unreadable=True)
    if not inventory.answers_for(called):
        # The answer was given for other names than this statement calls -- the decider asks
        # about exactly these, so this is two renderings disagreeing, and an unasked name is
        # nothing ruled out. The same verdict as an unreadable statement, for the same reason.
        return _cannot_resolve(inventory, unreadable=True)

    # UNRESOLVABLE FIRST, and with its own sentence. "Defined by this source itself" is false
    # of a synonym over a database link -- the source cannot see the target either -- and a
    # deployer told to rename their function will look for one that is not there.
    for name in sorted(called & inventory.unresolvable):
        return Refusal(
            code=RefusalCode.UNMODELLED_CALL,
            message=(
                f"{name}() reaches this source through a synonym it cannot resolve, so this "
                "engine cannot confirm what the query reads. Answer using only the listed "
                "tables and columns and standard SQL functions."
            ),
            subject=name,
        )

    for name in sorted(called & inventory.names):
        return Refusal(
            code=RefusalCode.UNMODELLED_CALL,
            message=(
                f"{name}() is defined by this source itself, so this engine cannot confirm "
                "what the query reads. Answer using only the listed tables and columns and "
                "standard SQL functions."
            ),
            subject=name,
        )
    return None


def check_opaque_columns(ast: exp.Expression, opaque, dialect: str | None = None,
                         columns: dict[str, set[str]] | None = None) -> Refusal | None:
    """Refuse a column whose stored expression this engine cannot attribute.

    A virtual column runs user code on read, and the statement never names it. MEASURED on
    Oracle: `leaked AS (vc_udf(id))` over a function reading another table, and
    `SELECT id, leaked FROM vc_t` returned an SSN from a table the identity was never granted.
    `check_unmodelled_calls` walks call nodes and sees none; `called_names` reads the rendered
    text and sees none; `check_access` sees a column the snapshot lists and passes it.

    Third time this codebase has met code reached with no call in the text -- after `count(*)`
    binding a macro named `count_star`, and `SELECT id + 1` binding one named `+`. The first two
    were answered by asking the source a different question rather than enumerating shapes, and
    so is this: `opaque` comes from the source's own dictionary.

    REPAIRABLE, unlike the other arrivals at this code. Selecting a different column is a real
    rewrite, and the message names the one to avoid.
    """
    if opaque is None:
        # ASKED and could not answer. Every column could be computed by an expression nobody can
        # read, and `check_access` passes a listed column, so there is nothing else looking.
        # Retried rather than final: `decide` re-reads the catalogue on every attempt.
        return Refusal(
            code=RefusalCode.UNRESOLVABLE_CALLS,
            message=(
                "This source could not say which of its columns are computed, so this engine "
                "cannot confirm what reading them executes. Try again."
            ),
            repairable_override=True,
        )
    if not opaque:
        return None
    referenced = {object_key(t) for t in base_tables(ast, dialect)}
    # Scope-aware, like `check_access` and `check_cls`. A flat alias map keyed on the written
    # case let `SELECT Q.leaked FROM vc_t Q` through while refusing the lowercase spelling, and
    # one alias meaning two tables across UNION branches changed the verdict with branch order.
    # The schema a derived star is read through (M181): the caller's (the grant's and the
    # policy's columns, from `known_columns`) with the catalogue's pairs beside it, so a virtual
    # column reached through `(SELECT * FROM plain JOIN vc_t) d` binds to the table that
    # computes it rather than to a table the schema does not list, which would take any name.
    schema: dict[str, set[str]] = {table: set(names) for table, names in (columns or {}).items()}
    for table, name in opaque:
        schema.setdefault(table, set()).add(name)
    resolved = column_tables(ast, dialect, schema)
    local = {cte.alias_or_name for cte in ast.find_all(exp.CTE)}
    aliased = local | {s.alias_or_name for s in ast.find_all(exp.Subquery) if s.alias_or_name}

    def refuse(name: str, why: str, repair: str) -> Refusal:
        return Refusal(
            code=RefusalCode.UNRESOLVABLE_CALLS,
            message=(
                f"{name!r} is computed by an expression this source stores and this engine "
                f"cannot attribute, so it cannot confirm what {why} executes. {repair}"
            ),
            subject=name,
            repairable_override=True,
        )

    # A JOIN KEY is not an `exp.Column`, and it reads the column all the same. `USING (leaked)`
    # names it as a bare identifier, and a NATURAL join names nothing at all -- both evaluate the
    # expression per row, and whether rows match leaks its value a bit at a time. Measured: both
    # passed a guard that walked only column nodes, while `WHERE leaked = 'x'` was refused, which
    # is the same channel one syntax over.
    for join in ast.find_all(exp.Join):
        for key in join.args.get("using") or ():
            name = key.name.lower() if hasattr(key, "name") else str(key).lower()
            for table in referenced:
                if (table.lower(), name) in opaque:
                    return refuse(name, "joining on it", "Join on a different column.")
        if (join.args.get("method") or "").upper() == "NATURAL":
            # It names no column, so there is nothing to check against: the keys are whatever the
            # two tables share. Any opaque column on a table in this statement could be one.
            # Lowercased, like the USING arm and the column loop. `opaque` keys arrive
            # lowercase from the adapter and an Oracle query naturally writes `FROM VC_T`, so
            # comparing raw let the uppercase spelling through -- the case that source actually
            # produces, while the lowercase one was refused.
            lowered = {t.lower() for t in referenced}
            for table, column in sorted(opaque):
                if table.lower() in lowered:
                    return refuse(column, "a NATURAL join on this table",
                                  "Join with an explicit ON or USING naming the columns.")

    for column in ast.find_all(exp.Column):
        name = column.name.lower()
        # A qualifier names the table directly; without one, any table in scope could own it,
        # and an opaque column anywhere in scope is one this statement may be reading.
        owners = candidate_tables(column, resolved, referenced, aliased)
        for table in owners:
            if (table.lower(), name) in opaque:
                return refuse(name, "reading it", "Answer without that column.")
    return None


def _cannot_resolve(inventory: FunctionInventory, *, unreadable: bool = False) -> Refusal:
    """Nothing here can be attributed, so nothing here can be decided.

    Several ways to arrive, and each gets its OWN SENTENCE, because a refusal that sends the
    reader to the wrong place is worse than a vague one. Sharing an owner is not enough to
    share a sentence: two of these are the source failing, and "it would not name its builtins"
    and "it would not name its own functions" are found by looking in different places.

      * the source defines a name it also lists as a builtin -- the DEPLOYER renames it
      * the adapter has no `builtin_functions` -- its AUTHOR implements it
      * it has one and the call raised -- the SOURCE is the problem, and it may be transient
      * `user_functions` itself raised -- likewise, and earlier
      * this one statement would not render -- about the STATEMENT, not the source at all

    Worth RETRYING, and saying so with `repairable_override`: the statement that would not
    render, since a rewrite can fix it, and both catalogue calls that RAISED, since `decide`
    re-asks the source on every attempt and a blip resolves itself. Not worth retrying: a source
    that redefines a builtin's name, and an adapter with no `builtin_functions` at all. Neither
    is a thing an attempt of ours changes, and the loop stops on them (M98).

    Two of them were caught borrowing another's sentence, and both times the effect was to send
    someone to fix working code. That is the failure this list is arranged against.
    """
    if unreadable:
        # "defines functions of its own" is FALSE where only `unresolvable` is populated -- a
        # synonym over a database link is not the source's function, and the source cannot see
        # its target either. Widening the gate to `names or unresolvable` made this arm
        # reachable in that state, and a deployer reading it goes looking for a UDF that does
        # not exist. The same argument the sibling arm's own sentence rests on.
        has_own = bool(inventory.names)
        why = (
            "this source defines functions of its own"
            if has_own
            else "this source has names this engine cannot resolve"
        )
        return Refusal(
            code=RefusalCode.UNRESOLVABLE_CALLS,
            message=(
                "This query could not be rendered, so this engine cannot confirm which "
                f"functions it asks the source for, and {why}. "
                "Answer using only the listed tables and columns and standard SQL."
            ),
            # The one arrival that is about the STATEMENT, so the one a rewrite can fix -- and
            # the message asks for one. The code alone would send it to the no-retry path with
            # the other arrivals, under a card saying rephrasing will not help.
            repairable_override=True,
        )
    if not inventory.available:
        # Retried, because it may be transient and `decide` re-asks the source on every attempt
        # (`inventory_from` reads LIVE, not from the snapshot). A single blip should not tell a
        # caller to page an operator. If it is not a blip the attempts run out and it says so.
        return Refusal(
            code=RefusalCode.UNRESOLVABLE_CALLS,
            message=(
                "This source could not say which functions it defines, so this engine cannot "
                "confirm what this query executes against it."
            ),
            repairable_override=True,
        )
    shadowed = sorted(inventory.names & inventory.builtins) if inventory.builtins else []
    if shadowed:
        # The source's own configuration. Every attempt lands here until someone renames it.
        detail = f"This source defines {shadowed[0]!r} under a name it also lists as a builtin"
        retry = False
    elif inventory.builtins_asked:
        # ASKED and failed, so the same call that raised is made again next attempt -- the same
        # argument as the `user_functions` arrival above, which is why leaving this one out was
        # a contradiction rather than a nuance.
        detail = ("This source defines functions of its own and could not say which names are "
                  "its builtins")
        retry = True
    else:
        # The adapter has no such method, which is not a thing an attempt changes.
        detail = ("This source defines functions of its own and was never asked which names are "
                  "its builtins")
        retry = False
    return Refusal(
        code=RefusalCode.UNRESOLVABLE_CALLS,
        message=(
            f"{detail}, so this engine cannot confirm what this query executes against it. "
            + ("Try again." if retry else "No query can be decided against this source.")
        ),
        subject=shadowed[0] if shadowed else None,
        repairable_override=True if retry else None,
    )
