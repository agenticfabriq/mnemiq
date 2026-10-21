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
    unnamed = unnamed_read(ast, executes_as or dialect)
    if unnamed is not None:
        return unnamed
    misgrouped = misgrouped_set_chain(ast, executes_as or dialect)
    if misgrouped is not None:
        return misgrouped
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

    positional = bound_by_position(ast, executes_as or dialect, columns)
    if positional is not None:
        return positional

    # The schema said which of these are columns; qualifying them is what makes that answer
    # checkable by the engine rather than taken on trust (M90). Before `_with_limit` so the
    # returned tree is the one that runs, and before `check_cls` ever sees it -- a qualified
    # reference is one that grant checking can rule on.
    # Read before qualifying, which rebuilds columns without their source positions.
    null_order_written = _first_key_writes_null_order(ast, sql, dialect)
    _qualify_schema_resolved(ast, columns)
    shaped = _with_limit(ast, max_rows, executes_as or dialect)
    if not isinstance(shaped, Refusal):
        shaped.meta["null_order_written"] = null_order_written
    return shaped


def bound_by_position(ast: exp.Expression, dialect: str,
                       columns: dict[str, set[str]] | None) -> Refusal | None:
    """A column the statement reaches by POSITION rather than by name, which no name-based check
    can read: refused as unnamed, before any grant is consulted (M183 to M186).

    The resolver binds a qualified column through a derived table by what the derived table
    EXPOSES under the name. Three shapes expose columns by position instead, measured on DuckDB:
    a star in a branch of a set operation inside a derived table or CTE lines the branches up by
    position, so `(SELECT * FROM claim UNION ALL SELECT * FROM wide) d` serves wide's second
    column under claim's name `secret` (M183); a column list over a star renames by position, so
    `(SELECT * FROM claim) d(x, y)` serves `secret` as `y` (M184); and DuckDB's `#2` is the second
    column whatever its name (M185). A PIVOT or UNPIVOT source has no projection the resolver
    models, and traversing it was an AssertionError rather than a verdict (M186). Each is refused
    with the advice to name the columns; a set operation of named branches, a column list over
    named projections, and VALUES under either (the ui lane's 42 recorded statements) are not
    these shapes and pass.
    """
    for positional in ast.find_all(exp.PositionalColumn):
        return Refusal(
            code=RefusalCode.SELECT_STAR,
            message=(f"`#{positional.this.name}` names a column by its position, which the engine "
                     "cannot check by name. Name the column instead."),
            subject=f"#{positional.this.name}",
        )
    for pivot in ast.find_all(exp.Pivot):
        kind = "UNPIVOT" if pivot.args.get("unpivot") else "PIVOT"
        return Refusal(
            code=RefusalCode.UNMODELLED_SOURCE,
            message=(f"A {kind} source reshapes a table in a way the engine cannot attribute "
                     "column by column. Write the reshaping as a query over named columns "
                     "(a UNION ALL of one SELECT per column, or CASE expressions) instead."),
            subject=kind,
        )
    for alias in ast.find_all(exp.Alias):
        if isinstance(alias.this, exp.Columns):
            # `COLUMNS(*) AS "x_\0"` renames every column it expands by a pattern, so the names
            # the outer query reads are derived and no node names the column read (Codex's
            # review of #167: `d.x_secret` over it returned a withheld secret).
            return Refusal(
                code=RefusalCode.SELECT_STAR,
                message=("`COLUMNS(...)` with an alias template renames the columns it expands "
                         "by a pattern, so the engine cannot tell which column a derived name "
                         "reads. Name the columns, and alias each one."),
                subject=alias.alias,
            )
    ctes = _visible_ctes(ast, {}, dialect)
    memo: dict[int, bool] = {}
    for table in ast.find_all(exp.Table):
        alias = table.args.get("alias")
        if alias is None or not alias.args.get("columns"):
            continue
        # A column list on a table REFERENCE renames by position too (Codex's review of #167):
        # over a base table, `FROM claim d(secret, id)` serves secret as `d.id`; over a CTE
        # whose body expands a star, the same; over a table function, its columns. Refused
        # whatever stands behind the name: telling a CTE of named projections apart would need
        # the CTE visible at the reference's own scope, and a lookup at the root let an inner
        # WITH shadowing an outer name through (the review gate); no recorded statement lists
        # columns on a reference (the ui lane's count).
        return Refusal(
            code=RefusalCode.SELECT_STAR,
            message=(f"The column list on {table.alias_or_name!r} renames what it reads by "
                     "position, so the new names say nothing about which columns are read. "
                     "Name the columns in a SELECT instead of listing them on the alias."),
            subject=table.alias_or_name or None,
        )
    for node in ast.find_all(exp.Subquery, exp.CTE, exp.Lateral):
        body = node.this
        if isinstance(node, exp.Lateral):
            # A LATERAL's column list sits on the LATERAL's alias, over its subquery (the ui
            # lane's replay of #167: `LATERAL (SELECT * FROM claim) l(x, y)` served secret as
            # `l.y` under hidden, denied and masked alike).
            body = body.this if isinstance(body, exp.Subquery) else body
        while isinstance(body, exp.Paren):
            body = body.this
        if not isinstance(body, exp.Query):
            continue
        if body.find(exp.Table) is None:
            # No table or CTE is named under it, so no star in it reaches a base table: `WITH
            # t(a, b) AS (SELECT * FROM (VALUES (1, 2)))`, the shape of every recorded statement
            # a column list appears in (the ui lane's count), where the star walk's fail-closed
            # reading of VALUES would refuse it.
            continue
        alias = node.args.get("alias")
        if isinstance(body, exp.SetOperation):
            # The branches' stars, each asked whether it reaches a base table in its own scope.
            for select in _output_selects(body):
                inner = _visible_ctes(select, ctes, dialect)
                for projection in select.expressions:
                    if _is_star(projection) and _expands_a_base_table(
                            select, projection, inner, columns, memo, frozenset(), dialect):
                        return Refusal(
                            code=RefusalCode.SELECT_STAR,
                            message=("A `*` in a branch of a UNION, INTERSECT or EXCEPT inside a "
                                     "subquery or CTE lines the branches' columns up by position, "
                                     "so a name outside it can read another table's column. Name "
                                     "the columns in every branch."),
                            subject=node.alias_or_name or None,
                        )
        if alias is not None and alias.args.get("columns") and _star_reaches_base(
                body, ctes, memo, columns, frozenset(), dialect):
            return Refusal(
                code=RefusalCode.SELECT_STAR,
                message=(f"The column list on {node.alias_or_name!r} renames a `*` by position, "
                         "so the new names say nothing about which columns are read. Name the "
                         "columns in the SELECT instead of listing them on the alias."),
                subject=node.alias_or_name or None,
            )
    return None


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


# The engines that bind INTERSECT tighter than UNION and EXCEPT, against the parser's left fold.
_INTERSECT_BINDS_TIGHTER = frozenset({"duckdb", "postgres", "postgresql", "mysql"})


# The niladic names sqlglot reads as a function with no Column node, which DuckDB binds to a
# same-named column of the query's own FROM before the function (M194).
_NILADIC = (exp.CurrentDate, exp.CurrentTimestamp, exp.CurrentTime, exp.Localtime,
            exp.Localtimestamp, exp.SessionUser, exp.CurrentCatalog, exp.CurrentUser)


# The call spelling each dialect keeps as a call, for the repair, in the dialect the model
# writes in (the statement it retries with is parsed as `dialect` and transpiled to the
# target, so a target's spelling could be a column to the parser: the review gate). DuckDB
# prints `current_date()` bare again, so a function with no bare form is named; Oracle reads
# every bare name as the function and has no `now()`; Postgres and SQLite reject
# `current_date()` outright (the gate), so each gets its own. An engine not listed is told to
# write a call with no example (the ui lane's read of the repair loop: the message is what
# the model retries with).
_FUNCTION_CALL_ADVICE = {
    "duckdb": "now() or get_current_timestamp()",
    "postgres": "now() or clock_timestamp()",
    "postgresql": "now() or clock_timestamp()",
    "mysql": "now() or curdate()",
    "sqlite": "datetime('now') or date('now')",
    "oracle": "SYSDATE or SYSTIMESTAMP",
}


def niladic_names(ast: exp.Expression, columns, dialect: str | None = None) -> Refusal | None:
    """A column named like a niladic SQL function is read through the function's node, which
    no grant, deny or mask check sees: DuckDB binds a bare `current_date` to a same-named
    column of the query's own FROM before the function, and sqlglot reads the name as
    `CurrentDate` with no Column node, so bare `current_date`, `current_timestamp`,
    `current_time`, `localtime`, `localtimestamp`, `session_user` and `current_catalog`, in a
    projection, an aggregate, a WHERE, an ORDER BY, a derived table, a CTE and a write, each
    returned a withheld column's value or an oracle on it (M194, the ui lane's replay of #169).
    The call form is no safer: `current_date()` and `today()` parse to the same node and the
    target SQL prints the bare name, which the engine binds to the column. Refused where any
    table the decider knows has a column so named, the broad form (DuckDB binds the column only
    where that table is in the query's own FROM, through a derived star, a CTE star, USING or
    LATERAL too; measured cost of the broad form: 0 of 2,803 corpus statements using
    CURRENT_DATE or CURRENT_TIMESTAMP, in the register), fail-closed; where no known table has
    one the name is the function to every engine measured. With no schema beyond the grant a
    withheld column is unknown here, a residual the register records. The repair names the
    column with its table, and a call this engine keeps as a call, in the engine's own
    spelling (`_FUNCTION_CALL_ADVICE`)."""
    calls = _FUNCTION_CALL_ADVICE.get((dialect or "").lower())
    advice = f"such as {calls}" if calls else "with its own name and parentheses"
    for node in ast.find_all(*_NILADIC):
        name = node.sql(dialect="duckdb").split("(")[0].strip().lower()
        owners = sorted(table for table, names in (columns or {}).items()
                        if any(n.lower() == name for n in names))
        if owners:
            return Refusal(
                code=RefusalCode.AMBIGUOUS_NILADIC_NAME,
                message=(f"{name!r} names a column of {owners[0]!r} and a SQL function, and this "
                         f"engine reads the column where that table is in the FROM. Write "
                         f"{owners[0]}.{name} for the column, or for the function a call this engine "
                         f"keeps as a call, {advice}."),
                subject=name,
            )
    return None


def misgrouped_set_chain(ast: exp.Expression, engine: str) -> Refusal | None:
    """Refuse a set-operation chain the parser groups differently from the engine: sqlglot folds
    an unparenthesised chain left to right, so `A UNION ALL B INTERSECT C` is an INTERSECT over a
    UNION, where DuckDB binds INTERSECT tighter and reads `A UNION ALL (B INTERSECT C)` (M203,
    measured; Postgres documents the same precedence and MySQL has it from 8.0.31, so both are
    held to it unmeasured, the review gate on the ninth commit; Oracle and SQLite group left to
    right as the parser does, and are not refused). Every rule
    over the tree then reads a shape the engine does not run. Not a cost alone: in a RECURSIVE
    body the resolver found no recursive part and read the CTE's own references as the base
    table, so a row filter on it was injected into them, and the approved statement answered
    `[1]` where the original answers `[1, 2, 3]` (Codex's sixth review of #170). The shape is
    an INTERSECT whose bare left operand is a UNION or an EXCEPT; a parenthesised operand is a
    subquery to the parser and grouped as written. The corpus holds no such chain (the ui lane's
    count, 2026-10-19)."""
    if (engine or "").lower() not in _INTERSECT_BINDS_TIGHTER:
        return None
    for node in ast.find_all(exp.Intersect):
        if isinstance(node.this, (exp.Union, exp.Except)):
            return Refusal(
                code=RefusalCode.PARSE_ERROR,
                message=("A UNION or EXCEPT followed by INTERSECT without parentheses is grouped "
                         "one way by the parser and another by the engine, which binds INTERSECT "
                         "tighter, so the statement cannot be checked as it would run. Write the "
                         "parentheses: `A UNION ALL (B INTERSECT C)` or `(A UNION ALL B) INTERSECT C`."),
                subject="INTERSECT",
            )
    return None


def unnamed_read(ast: exp.Expression, engine: str, write: bool = False) -> Refusal | None:
    """Two DuckDB spellings that read a column no node names, so no grant, deny or mask check
    sees the read; refused before any check runs, by both deciders (M192, M193: measured on
    DuckDB 1.5.4, each returned a withheld value under deny, under mask and hidden with the
    schema listing it; Codex's first review of #169 copied the denied value into a writable
    table through the write decider's own shape check, which had not run this).

    Method-call syntax: `x.f()` is `f(x)` to DuckDB, and sqlglot reads `secret.lower()` as a Dot
    over an identifier and a call, with no Column node; `secret.max()`, `claim.secret.lower()`,
    `WHERE secret.starts_with('Z')`, `ORDER BY secret.lower()` and `HAVING secret.max() LIKE
    'Z%'` the same. Whether the left side is a column is not for the decider to resolve: the
    grant lists only the visible columns, so a hidden one is exactly the name a visible-schema
    lookup misses. Executed as DuckDB, every Dot whose right side is a call is refused, the
    schema-qualified `main.lower(note)` with it (the plain spelling exists, and the advice names
    it). Executed as Oracle, which has no method syntax on a bare column (a method of an
    object-typed column is reached only through a table alias, `c.secret.f()`), anything but a
    bare identifier on the left is refused (an alias-led Dot, a Column, a parenthesised
    column), and a bare identifier is a package or schema name (`DBMS_LOB.SUBSTR(note, 10,
    1)`), whose column sits in the arguments and is checked, and whose function the inventory
    judges. The walk visits every Dot, so a chain (`secret.lower().upper()`) is refused at
    its outer call; the left side's spelling (quoted, upper-cased, parenthesised, a struct
    column, a derived table's alias) never matters.

    `COLUMNS(...)` inside an expression: as a projection on its own the resolver reads it, and
    among the outermost projections the star check refuses it, but inside a call, a WHERE, an
    ORDER BY, a struct or a derived table's expression (`upper(COLUMNS('e'))`, `WHERE
    COLUMNS('ecr') LIKE 'Z%'`, `struct_pack(COLUMNS('e'))`, `concat(*COLUMNS('e'))`) the engine
    expands it into reads of columns that appear in no node. Every COLUMNS that is not a bare
    projection is refused, the qualified `x.COLUMNS(...)` spelling (a Dot over a call) with it.

    A star, or a bare COLUMNS, is read by the outermost star check (`_has_projection_star`), by
    the resolver for a derived table or a CTE the outer query reads by name and whose own Select
    does not consume it (a DISTINCT, a GROUP BY, an ORDER BY ALL or by position reads every
    column: `SELECT count(*) FROM (SELECT DISTINCT * FROM claim) d` answers by whether two
    withheld values are equal, M198), and by nothing else; `COUNT(*)` and an EXISTS's projection
    read no value, unless an INTERSECT or EXCEPT under the EXISTS compares it, and a PIVOT or
    UNPIVOT source is refused as unmodelled with its own code (`_projection_is_read` is that
    whole list). A star in a scalar subquery, an IN list or a RETURNING is read by none of them,
    and its Select names no Column: `SELECT (SELECT * FROM one LIMIT 1) AS s FROM pub` over a one-column table
    returned the withheld value hidden, denied and masked, and `WHERE (SELECT * FROM one LIMIT
    1) LIKE 'Z%'` was an oracle (M196). Every star whose query is not the outermost one, a
    derived table, a CTE, an EXISTS or `COUNT(*)` is refused. In a write (`write=True`, the
    write decider's shape check) the outermost, derived and CTE readings do not exist either:
    the INSERT's source has no outermost check, so `INSERT INTO sink SELECT * FROM claim` and
    `INSERT INTO sink SELECT COLUMNS('secret') FROM claim` were approved and copied the
    withheld value into a table the same grant reads (Codex's second review of #169; M195).
    Every star and every COLUMNS but `COUNT(*)`'s and an EXISTS's is refused there: a write names
    every column it reads. A column reached by position (`#2`) is the same read under another
    spelling, refused on the read side by `bound_by_position` and, since M197, in the write
    decider's shape check as well.
    """
    for dot in ast.find_all(exp.Dot):
        if isinstance(dot.expression, exp.Anonymous) and str(dot.expression.this).upper() == "COLUMNS":
            continue  # the qualified star spelling, refused below with the star's own code
        if not isinstance(dot.expression, (exp.Anonymous, exp.Func)):
            continue
        left = dot.this
        if engine == "duckdb" or not isinstance(left, exp.Identifier):
            # The same shape as a schema-qualified call (`public.customer_rows()`), which the
            # function inventory refuses as an unmodelled call: one code for both readings,
            # since the decider cannot tell what either reads.
            return Refusal(
                code=RefusalCode.UNMODELLED_CALL,
                message=(f"`{dot.sql()}` is a call the engine cannot see into: DuckDB's method "
                         "syntax `x.f()` names the column where no check can read it, and a "
                         "qualified call names a function the engine does not model. Write the "
                         "call as `f(x)` over a plain column."),
                subject=dot.this.sql(),
            )
    for expansion in ast.find_all(exp.Columns):
        if expansion.find_ancestor(exp.Pivot) is not None or (
                isinstance(expansion.parent, exp.Select) and _projection_is_read(expansion, write)):
            continue  # a PIVOT's own refusal; a bare projection the resolver or star check reads
        return Refusal(
            code=RefusalCode.SELECT_STAR,
            message=("`COLUMNS(...)` inside an expression expands to columns no check can "
                     "see. Use it only as a projection on its own, or name the columns."),
            subject=expansion.sql(),
        )
    for call in ast.find_all(exp.Anonymous):
        if str(call.this).upper() == "COLUMNS":
            return Refusal(
                code=RefusalCode.SELECT_STAR,
                message=("`COLUMNS(...)` inside an expression expands to columns no check can "
                         "see. Use it only as a projection on its own, or name the columns."),
                subject=call.sql(),
            )
    for star in ast.find_all(exp.Star):
        if isinstance(star.parent, exp.Columns):
            continue  # `COLUMNS(*)`: the COLUMNS loop above ruled on it (Codex's fifth review)
        if _projection_is_read(star, write):
            continue
        return Refusal(
            code=RefusalCode.SELECT_STAR,
            message=("`*` here expands to columns no check can see: a star is read only as the "
                     "outermost projection, in a derived table or a CTE the query reads by "
                     "name, under EXISTS or as COUNT(*), and in a write only as the last two. "
                     "Name the columns instead."),
            subject=star.parent.sql() if isinstance(star.parent, exp.Column) else star.sql(),
        )
    return None


def _projection_is_read(node: exp.Expression, write: bool) -> bool:
    """Whether some check reads this star or bare COLUMNS: `COUNT(*)`; the projection of an
    EXISTS, unless an INTERSECT or EXCEPT on the way compares the values (`EXISTS (SELECT *
    FROM one INTERSECT SELECT 'Z')` is true only on the withheld value: Codex's third review of
    #169) or a DISTINCT, a GROUP BY, an ORDER BY by position or ALL, or a UNION without ALL
    consumes them (`... UNION SELECT 'Z' OFFSET 1`: the fourth); and, on the read side only, the outermost query's (the star check), a derived
    table's or a CTE's (the resolver, where the outer query names what it reads). A star or
    COLUMNS anywhere else, a scalar subquery (`WHERE (SELECT COLUMNS('secret') FROM one LIMIT
    1) = 'Z'`, the same review), an IN list, a value in parentheses, an INSERT's source, is read
    by nothing."""
    holder = node.parent
    if isinstance(holder, exp.Column):
        holder = holder.parent  # `t.*`
    if isinstance(holder, exp.Count):
        return True
    if node.find_ancestor(exp.Pivot) is not None:
        return True  # a PIVOT or UNPIVOT source is refused as unmodelled, with its own code
    if not isinstance(holder, exp.Select):
        return False
    query, compared, wrappers = holder, False, [holder]
    while isinstance(query.parent, (exp.SetOperation, exp.Paren)):
        query = query.parent
        wrappers.append(query)
        compared = compared or isinstance(query, (exp.Intersect, exp.Except))
    outer = query.parent
    if isinstance(outer, exp.Exists):
        # Existence reads the values where an INTERSECT or EXCEPT compares them, or where a
        # deduplication or a grouping decides how many rows there are and an OFFSET or a
        # HAVING then asks: `EXISTS (SELECT * FROM one UNION SELECT 'Z' OFFSET 1)` is true only
        # where the withheld value differs from the guess (Codex's fourth review of #169). The
        # rule is broader than the oracle, fail-closed: any consumer on the climb (a DISTINCT, a
        # GROUP BY, an ORDER BY by position or ALL, a UNION without ALL) refuses, with or
        # without the OFFSET, since a LIMIT, a QUALIFY or a HAVING can ask the same question.
        return not compared and not any(_consumes_its_projection(wrapper) or (
            isinstance(wrapper, exp.SetOperation) and bool(wrapper.args.get("distinct")))
            for wrapper in wrappers)
    if write:
        return False
    if outer is None:
        return True  # the outermost projection: `_has_projection_star` rules on it
    if isinstance(outer, exp.CTE):
        return not any(_consumes_its_projection(wrapper) for wrapper in wrappers)
    while isinstance(outer, exp.Subquery) and isinstance(outer.parent, exp.Subquery):
        wrappers.append(outer)
        outer = outer.parent  # `((SELECT ...)) d` nests Subquery in Subquery
    if not isinstance(outer, exp.Subquery) or not isinstance(outer.parent, (exp.From, exp.Join, exp.Lateral)):
        return False
    wrappers.append(outer)
    return not any(_consumes_its_projection(wrapper) for wrapper in wrappers)  # M198


_WRAPPERS = (exp.Select, exp.Subquery, exp.SetOperation, exp.Paren)


def _consumes_its_projection(node: exp.Expression) -> bool:
    """Whether the query, or a wrapper around it, reads every column of its projection, star
    included: a DISTINCT (plain or ON) compares them, a GROUP BY (ALL, or by position: a star
    under any other GROUP BY is DuckDB's own error) groups by them, an ORDER BY ALL or by
    position orders by them, in the Select's own ORDER BY, a window's, or the ORDER BY sqlglot
    hangs on a wrapping Subquery (`((SELECT * FROM pair) ORDER BY 2 LIMIT 1) d`: the review gate
    on the fourth round) or on a set operation (`(SELECT * FROM pair UNION ALL SELECT * FROM
    pair ORDER BY 2 LIMIT 1) d`, whose branches the positional check refuses as well). Measured on DuckDB 1.5.4: `SELECT count(*) FROM (SELECT DISTINCT *
    FROM claim) d`, the GROUP BY ALL twin and the CTE spelling answer 1 or 2 by whether two
    rows' withheld values are equal, and `(SELECT * FROM claim ORDER BY 2 LIMIT 1) d` serves the
    id of the smallest (the ui lane's replay of #169; M198). A name in any of those clauses is a
    Column the checks read; a position or ALL is not."""
    if isinstance(node, exp.Select) and (node.args.get("distinct") is not None
                                         or node.args.get("group") is not None):
        return True  # a set operation's own DISTINCT is M183's: its branches bind by position
    for ordered in node.find_all(exp.Ordered):
        if ordered.find_ancestor(*_WRAPPERS) is not node:
            continue  # a subquery's own ORDER BY, judged with that subquery's projection
        key = ordered.this
        while isinstance(key, exp.Paren):
            key = key.this
        if not isinstance(key, exp.Column):
            # A key is left alone only as a bare column name, which the checks read. Every
            # other spelling consumes the projection, fail-closed: a position (`2`, `(2)`,
            # `-(-2)`, `2 COLLATE nocase`, each the position to DuckDB: the ui lane, Codex's
            # sixth review and the gate found one after another), ALL, and a constant or an
            # expression (`2::INT`, `id + 1`), which DuckDB reads otherwise and are refused
            # all the same rather than told apart spelling by spelling.
            return True
    return False


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




def _resolve_bare(column: exp.Column,
                  columns: dict[str, set[str]] | None) -> tuple[str, exp.Expression | None]:
    """What a bare name binds to: ``("column", source | None)``, ``("row", source)`` or ``("", None)``.

    **One resolver, two callers.** `_binds_to_a_row` asks whether to refuse and `_schema_qualifier`
    asks what to qualify with, and every time this file has answered one question in two places the
    two have drifted -- four findings this week, three of them leaks. Both read this.

    Innermost scope decides, as SQL does. At each scope:

    * a SELECT ALIAS of the same name wins over any table column, so the reference is neither a row
      nor a column of a table and nothing may be rewritten. `ORDER BY claim_amount` against
      `amount AS claim_amount` was qualified to `claim_amount.claim_amount` and repointed the sort;
    * exactly ONE source claiming the name resolves it to that column;
    * a CTE reference is skipped when reading the schema: it is an `exp.Table`, so the lookup
      would hand it the columns of whatever BASE table it shadows, and `WITH other AS (...)`
      borrowed the granted `other`'s column list and returned a struct with a denied field in it;
    * two or more claiming it is unresolvable, and so is a name that collides with a source no
      snapshot claims -- both answer "row", which refuses. Leaving an ambiguous one bare put the
      M90 leak back verbatim: measured, a snapshot claiming the name on two tables that do not have
      it returned the whole struct with a denied column in it.
    """
    lookup = (
        {key.lower(): {name.lower() for name in names} for key, names in columns.items()}
        if columns else {}
    )
    name = column.name.lower()
    root = column
    while root.parent is not None:
        root = root.parent
    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    scope = column.parent
    while scope is not None:
        if isinstance(scope, exp.Select):
            sources = {key.lower(): source for key, source in _sources(scope).items()}
            holders = [
                source for source in sources.values()
                if isinstance(source, exp.Table)
                and source.name.lower() not in cte_names
                and name in lookup.get(object_key(source).lower(), frozenset())
            ]
            if name in sources:
                return ("column", holders[0]) if len(holders) == 1 else ("row", sources[name])
            if holders:
                return "column", None  # an ordinary column; no collision to arbitrate
        scope = scope.parent
    return "", None


def _schema_qualifier(column: exp.Column, columns: dict[str, set[str]] | None) -> str | None:
    """The source to qualify this bare name with, when the SCHEMA is what resolved it to a column.

    **Register M90.** Letting the schema decide "column or row" (M88) put a stale snapshot in the
    fail-OPEN direction: one claiming a column the table no longer has makes `SELECT claim FROM
    claim` pass as a column reference while the engine binds it to the row STRUCT and hands back
    every field, a denied one included. `prove`'s EXPLAIN succeeds either way, because a query that
    resolves to a row is perfectly valid SQL, so nothing downstream separates them.

    A freshness promise is not available -- nothing in this process knows when the catalog was
    built. So the query stops RELYING on the claim: the reference is QUALIFIED, which makes the
    schema's answer checkable by the engine. Right, and the query means exactly what it did; stale,
    and the binder refuses it. `SELECT claim.claim FROM claim` is a Binder Error on DuckDB where the
    bare form returns the struct.

    The qualifier keeps the source's own quoting -- rebuilding it bare folded `"Claim"` to `claim`,
    which names no source against postgres and turned a legitimate query into `EXPLAIN_FAILED`.

    **What this does NOT cover, stated because deleting the sentence that said so was itself a
    finding.** Only a name that COLLIDES with a source is qualified, since that is the case the
    schema had to arbitrate. A stale entry claiming a column on a source the name does not collide
    with is still left bare and can still bind outward to an outer table's row --
    `SELECT (SELECT claim['ssn'] FROM line_item LIMIT 1) AS x FROM claim` with `line_item`
    stale-claiming `claim` is approved and returns the denied value. That hole predates this and is
    not closed by it; the guarantee is narrower than "the query stops relying on the claim", and
    saying the broad version would put the file back to trusting something it cannot check.
    """
    if column.table:
        return None
    kind, source = _resolve_bare(column, columns)
    return _qualifier_of(source) if kind == "column" and source is not None else None


def _qualifier_of(source: exp.Table) -> str:
    """How to spell this source in a qualifier, keeping the quoting it was written with."""
    alias = source.args.get("alias")
    identifier = alias.this if alias is not None and alias.this is not None else source.this
    return identifier.sql(dialect="duckdb") if isinstance(identifier, exp.Identifier) else str(identifier)


def _qualify_schema_resolved(ast: exp.Expression, columns: dict[str, set[str]] | None) -> None:
    """Qualify the bare PROJECTION names the schema resolved to a column, in place.

    **Projections only, and the narrowing is the finding.** Four versions of an exemption for
    select aliases each reopened something: scope-wide disarmed the projection check entirely,
    outside-the-projection reopened M86's oracle in WHERE and a frequency count in GROUP BY, and
    ORDER-BY-and-HAVING still exempted `ORDER BY claim['ssn']`, because only the BARE name binds to
    an alias and a subscript on it is the struct again. Every version was a rule about WHERE a
    reference sits rather than what the engine does with it, and every one was wrong towards the
    leak.

    There is no alias question in the projection: a lateral alias sees only EARLIER items, so a
    bare name there binds to the table. Confining the rewrite removes the exemption and everything
    that kept going wrong inside it, and `_binds_to_a_row` keeps its full strength in every clause
    -- an aliased `ORDER BY claim_amount` is neither rewritten nor refused, because the schema
    resolves it to a column and nothing needs to touch it.

    The cost is that staleness is only caught where the rewrite reaches. A snapshot claiming a
    column the table lacks still lets a WHERE-clause row use through, which is an oracle rather
    than an escape -- the projection is where a struct actually leaves in the answer, and that is
    the half now checked by the engine instead of trusted.
    """
    for select in ast.find_all(exp.Select):
        for projection in select.expressions:
            _qualify_in(projection, columns)


def _qualify_in(node: exp.Expression, columns: dict[str, set[str]] | None) -> None:
    for column in node.find_all(exp.Column):
        qualifier = _schema_qualifier(column, columns)
        if qualifier:
            # Parsed rather than constructed, so a quoted qualifier stays quoted.
            column.set("table", sqlglot.parse_one(qualifier, read="duckdb", into=exp.Identifier))

def _binds_to_a_row(column: exp.Column,
                    columns: dict[str, set[str]] | None) -> exp.Expression | None:
    """The source this bare name binds to as a WHOLE ROW, or None if it is a column.

    The refusing half of `_resolve_bare`, which carries the scope rules and the measurements
    behind them.
    """
    kind, source = _resolve_bare(column, columns)
    return source if kind == "row" else None


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
# A key no CTE can be named, an object and not a string (a quoted identifier can spell any
# string, and a CTE so named would have overwritten a string key: the review gate): under it a
# scope carries the recursive parts of its RECURSIVE CTEs, (id of the recursive part's node, the
# CTE's name, its entry), which `_scope_within` binds for a SELECT sitting in that part as the
# star resolver descends into the body.
_RECURSIVE_PARTS: object = object()


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
        # siblings, plus itself only when the WITH is RECURSIVE, and then only in its RECURSIVE
        # PART, the right side of the body's UNION: in the anchor the name is the base table
        # (DuckDB, measured: `WITH RECURSIVE claim AS (SELECT * FROM claim UNION ALL ...)
        # SELECT * FROM claim` returned the base table's hidden column, the anchor's star bound
        # to the CTE and vouched for by nothing: M201). Only a UNION makes a CTE recursive: an
        # INTERSECT or EXCEPT body is a plain CTE, its own name the base table everywhere in it
        # (`... INTERSECT SELECT * FROM claim` expanded the CTE and returned the hidden column:
        # the ui lane's replay of #170 at 3e98092), and a body with no UNION has no recursive
        # part. The rule `_in_recursive_part` in `scope` states for the name resolver.
        body_scope: Scope = dict(scope)
        entry = (cte.this, body_scope)
        body = cte.this
        while isinstance(body, (exp.Subquery, exp.Paren)):
            body = body.this
        if recursive and isinstance(body, exp.Union):
            body_scope[_RECURSIVE_PARTS] = (*body_scope.get(_RECURSIVE_PARTS, ()), (id(body.right), key, entry))
        scope[key] = entry
    return scope


def _scope_within(select: exp.Expression, scope: Scope) -> Scope:
    """The scope a SELECT sees: `scope`, plus a RECURSIVE CTE's own name where the SELECT sits in
    that CTE's recursive part."""
    parts = scope.get(_RECURSIVE_PARTS, ())
    if not parts:
        return scope
    above = {id(n) for n in [select, *_ancestors(select)]}
    bound = [(key, entry) for part, key, entry in parts if part in above]
    if not bound:
        return scope
    seen: Scope = dict(scope)
    for key, entry in bound:
        seen[key] = entry
    return seen


def _ancestors(node: exp.Expression) -> list[exp.Expression]:
    out, node = [], node.parent
    while node is not None:
        out.append(node)
        node = node.parent
    return out


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
            _expands_a_base_table(select, projection, _scope_within(select, inner), columns, memo, within, dialect)
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


def _with_limit(ast: exp.Expression, max_rows: int, engine: str = "duckdb") -> exp.Expression | Refusal:
    """Bound the result, and mark the tree when the bound is ours (M118).

    A limit the query set itself is the query's business: `LIMIT 10` for a top ten returns ten
    rows on purpose. One the guard imposed is a cut nobody asked for, and a result that fills it
    may have been cut -- so the tree carries `meta["row_cap"]` for the caller to say so.

    It also carries `meta["wrote_limit"]`: whether the statement bounded itself, kept or clamped. A
    reader asking whether the statement set a LIMIT cannot ask the tree, whose LIMIT may be the cap's,
    nor `row_cap`, which also marks a clamped one (M161: the logic lint refused for the cap's LIMIT).
    """
    existing = ast.args.get("limit")
    wrote = existing is not None and not _states_no_bound(existing, engine)
    if wrote:
        count = _written_count(existing)
        if count is None:
            # Replacing a bound we cannot read changes the answer: a top-5 WITH TIES became a
            # thousand rows, a percentage became a fixed count (M162). Keeping it uncapped is no
            # better -- ties and percentages are unbounded, and the adapters fetch every row. So
            # the statement states its bound as a count, which the cap can then be checked against.
            #
            # The share is counted in rows, not read off CUME_DIST: CUME_DIST gives a whole peer
            # group its last row's share, so ten tied rows filtered at 50% kept none (Codex, #156).
            # CEIL is Oracle's rounding of FETCH ... PERCENT, and it never empties a non-empty
            # result, where DuckDB's own `LIMIT 10%` of 7 rows returns none. Multiplied before
            # divided: in DuckDB's doubles `7 / 100.0 * 100` rounds up to 8.
            return Refusal(
                code=RefusalCode.UNREADABLE_LIMIT,
                message=(f"State the row limit as a plain number of rows, at most {max_rows:,}: "
                         "`LIMIT n`, or `FETCH FIRST n ROWS ONLY`. A limit given as a percentage, "
                         "WITH TIES, or an expression cannot be checked against this engine's "
                         f"{max_rows:,}-row bound. To keep ties, rank in a subquery and filter on "
                         "`RANK() OVER (ORDER BY ...) <= n`. For a share of the rows, number "
                         "them in a subquery with `ROW_NUMBER() OVER (ORDER BY ...) AS rn` and "
                         "`COUNT(*) OVER () AS total`, and filter on "
                         "`rn <= CEIL(total * p / 100.0)`; use RANK() in place of ROW_NUMBER() to "
                         "keep ties as well."),
            )
        if count <= max_rows:
            ast.meta["wrote_limit"] = True
            return ast
    capped = ast.limit(max_rows)
    capped.meta["row_cap"] = max_rows
    capped.meta["wrote_limit"] = wrote
    return capped


def _first_key_writes_null_order(ast: exp.Expression, sql: str, dialect: str) -> bool:
    """Is the root's first sort key followed, in the text, by NULLS FIRST or NULLS LAST?

    sqlglot records every key's null order, written or the dialect's default, and keeps no mark of
    which; the logic lint leaves a written one alone (M161). So the text decides, at the key's own
    position: a NULLS clause in a window or a subquery says nothing about the root's top-N. Only a
    column key is read, because only a column key is linted."""
    order = ast.args.get("order")
    if order is None or not order.expressions:
        return False
    key = order.expressions[0].this
    ident = key.this if isinstance(key, exp.Column) else None
    end = ident.meta.get("end") if isinstance(ident, exp.Identifier) else None
    if end is None:
        return False
    try:
        tokens = sqlglot.tokenize(sql, read=dialect)
    except Exception:
        return False
    words = [t.text.upper() for t in tokens if t.start > end][:3]
    if words[:1] in (["ASC"], ["DESC"]):
        words = words[1:]
    return words[:2] in (["NULLS", "FIRST"], ["NULLS", "LAST"])


def _states_no_bound(limit: exp.Expression, engine: str) -> bool:
    """`LIMIT NULL` means no bound, and so does a negative number in SQLite, so the cap applies exactly
    as if no LIMIT were written. On main (`70188c4`, before #156) SQLite's `LIMIT -1` was kept as
    written and ran uncapped (the ui lane's review of #156). Only a negative PLAIN NUMBER: `-(-5)` is
    five rows and `-0` none, so anything else under the minus is left to `_written_count`, which
    refuses it. Keyed on the engine that runs the statement, like the rest of this module; elsewhere
    a negative LIMIT is an error, and it is refused there too. `LIMIT ALL` parses as no limit."""
    if isinstance(limit, exp.Fetch):
        return False
    if isinstance(limit.expression, exp.Null):
        return True
    minus = limit.expression
    if not isinstance(minus, exp.Neg) or engine.lower() != "sqlite":
        return False
    number = minus.this
    if not isinstance(number, exp.Literal) or number.is_string:
        return False
    try:
        return int(number.name) > 0
    except ValueError:
        return False


def _written_count(limit: exp.Expression) -> int | None:
    """The number of rows a LIMIT or FETCH clause states, or None where it states no plain count.

    sqlglot keeps LIMIT's count in `expression` and FETCH's in `count`. Reading only the first, an
    Oracle `FETCH FIRST 5 ROWS ONLY` counted as unreadable and ran as the 1,000-row cap (M162). A
    percentage or WITH TIES is not a row count: DuckDB's `LIMIT 10%` read as ten rows and ran
    uncapped. `FETCH FIRST ROW ONLY` states no count, and means one row."""
    options = limit.args.get("limit_options")
    if options is not None and (options.args.get("percent") or options.args.get("with_ties")):
        return None
    node = limit.args.get("count") if isinstance(limit, exp.Fetch) else limit.expression
    if node is None:
        return 1 if isinstance(limit, exp.Fetch) else None
    if not isinstance(node, exp.Literal) or node.is_string:
        return None
    try:
        return int(node.name)
    except ValueError:
        return None
