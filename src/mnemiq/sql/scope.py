from __future__ import annotations

from sqlglot import exp
from sqlglot.optimizer.scope import build_scope

from mnemiq.sql.identifiers import resolve_identifier, resolve_name
from mnemiq.sql.qualify import object_key


def _unscoped_ctes(ast: exp.Expression, root) -> bool:
    """True when the statement holds CTEs the scope builder will not descend into.

    A write root parks its CTEs in its own WITH arg, OUTSIDE the query `build_scope` roots
    itself at, so `WITH x AS (SELECT ... FROM claim) INSERT INTO t SELECT ... FROM x` yields a
    perfectly healthy `Scope<SELECT ... FROM x>` and the body holding `claim` is never visited.

    That is worse than an outright resolver failure, because it does not look like one. The
    caller gets `[]` -- and `[]` means "nothing here reads a real object", which is what
    `check_access` then acts on: its `if not alias_to_table: return None` early-out is justified
    by the comment *"their sources were checked above"*, and above, nothing was. Ninth instance
    in this codebase of an absence and a failure sharing one value (M2, M6, M34, M45, ...).

    DEFENCE IN DEPTH, and not the live control: `check_write_shape` refuses this shape outright
    with UNSCOPED_CTE, so nothing reaches here today carrying a leading WITH. Kept because the
    refusal is ONE guard's decision while `base_tables` is the premise `check_access`, `check_cls`
    and `apply_row_and_mask` all share, and a resolver must not answer `[]` for a statement it
    never read. Measured with the shape refusal out of the way: these return `['claim', 'scratch',
    'x']` and are refused on `x`. So this floor declines; it does not govern.

    The predicate is STRUCTURAL -- an `exp.With` among the root's own args -- and never an arg
    name. `pyproject` declares `sqlglot>=25`; 25.x spells that arg `with` where 30.x spells it
    `with_`, so a name-keyed lookup returns None across much of the declared range and fails OPEN
    in the one place built to fail closed. The list branch mirrors `_has_unscoped_with`, the guard
    this floors, so the two cannot drift; no 30.12 root puts a `With` in a list arg, so that
    branch is shape parity rather than a reachable case, and no test pins it.
    """
    in_scope = list(root.expression.find_all(exp.With))
    for value in ast.args.values():
        found = [value] if isinstance(value, exp.With) else (
            [v for v in value if isinstance(v, exp.With)] if isinstance(value, list) else [])
        if any(not any(n is f for n in in_scope) for f in found):
            return True
    return False


def _target_reads(ast: exp.Expression) -> list[exp.Table]:
    """Every table a WRITE reads through its own TARGET position; empty when it reads none.

    `build_scope` models the query a statement runs, not the object it mutates, so a write
    target is in no `scope.sources` and every guard built on `base_tables` was blind to it.
    Measured before this: `UPDATE claim SET amount = 0 WHERE ssn = '1' AND id IN (SELECT id FROM
    other)` resolved to ['other'] alone, so a DENIED column on `claim` was checked against
    `other` and the write was approved carrying it -- while the same statement without the
    subquery was refused, because `build_scope` fails there and the `find_all` fallback put the
    target back. The guard was doing its job only when the resolver gave up.

    Which shapes READ their target is the whole question, and it is not the verb:

      UPDATE   yes -- it reads the rows it is about to change
      DELETE   yes -- same, through its WHERE
      INSERT   no -- the target is written and never read
      INSERT ... ON CONFLICT DO UPDATE   YES: the SET clause reads the existing row

    That last row is why this is a predicate and not `isinstance(ast, (Update, Delete))`. An
    upsert is an INSERT by node type and a read of its target in fact, and it was approved
    against a target the caller could not see, whose column was denied.
    """
    if isinstance(ast, (exp.Update, exp.Delete)):
        # A multi-target DELETE puts its targets in `tables` and leaves a JOIN in `this`, so
        # reading `this` alone names ONE of them and drops the rest. `_target_node` in
        # `decide_write` already guards this and refuses AMBIGUOUS_WRITE_TARGET -- but a
        # resolver whose correctness depends on a guard downstream of it is the bug this
        # whole change is about, so it is handled here too. All targets, not a pick.
        targets = ast.args.get("tables")
        nodes = list(targets) if targets else [ast.this]
    elif isinstance(ast, exp.Insert):
        conflict = ast.args.get("conflict")
        if conflict is None or not conflict.args.get("expressions"):
            return []  # no ON CONFLICT, or DO NOTHING: nothing reads the target
        nodes = [ast.this]
    else:
        return []
    out: list[exp.Table] = []
    for node in nodes:
        if isinstance(node, exp.Schema):  # INSERT INTO t (cols)
            node = node.this
        if isinstance(node, exp.Table):
            out.append(node)
    return out


def _root_scope(ast: exp.Expression):
    """The resolved root scope, or None. ONE place builds it, because `base_tables` and
    `scope_resolved` must never disagree about whether resolution succeeded -- that is M7's shape,
    two implementations of one question."""
    try:
        return build_scope(ast)
    except Exception:
        return None


def scope_resolved(ast: exp.Expression) -> bool:
    """Did the scope resolve, so that `base_tables` is a resolved answer rather than a fallback?

    When it did not, `base_tables` returns `find_all(exp.Table)` -- CTE aliases included -- which
    OVER-reports. That is the right direction for a guard, because an over-reported name is not in
    `visible` and the query is refused. It is the wrong direction for an audit record, which would
    name an object the query never read. So the record needs to know, and the guard does not.
    """
    ast_root = _root_scope(ast)
    return ast_root is not None and not _unscoped_ctes(ast, ast_root)


def _defining_identifier(source) -> exp.Expression | None:
    """The identifier that NAMES a local source, with its quoting intact.

    `scope.sources` is keyed by bare text, so the quoting is gone by the time a name is looked up
    there. The defining node still has it, so this walks UP from the source's expression to the
    nearest node that defines a name.

    Upward, not by matching the expression against every CTE: a RECURSIVE CTE's self-reference
    binds to a scope whose expression is one BRANCH of the union while the CTE's own body is the
    whole union, so identity matched nothing, `_engine_shadows` fell through to its fail-open
    answer, and `WITH RECURSIVE "Claim" AS (... UNION ALL SELECT id FROM Claim) SELECT id FROM
    "Claim"` came back Approved with the base table dropped from the read list -- the M79 leak, in
    the one shape a note in this file claimed could not reach that branch.
    """
    node = getattr(source, "expression", None)
    while node is not None:
        if isinstance(node, exp.CTE | exp.Subquery):
            alias = node.args.get("alias")
            if alias is not None:
                return alias
            # An alias-less definer NAMES nothing, so stopping here answers "no definer" for a
            # source that plainly has one. A parenthesised CTE body -- `WITH "Claim" AS ((SELECT
            # ...))` -- wraps the body in exactly that, and one pair of parentheses turned the M79
            # refusal back into an Approved. Keep walking to the node that does name it.
        node = node.parent
    return None


def _engine_shadows(table: exp.Table, source, dialect: str | None) -> bool:
    """Would the EXECUTING engine resolve this reference to that local source?

    sqlglot matches a source by its bare name and the engine does not, so a CTE could hide a base
    table from the grant check that shares nothing with it but spelling. Measured: `WITH "Claim" AS
    (SELECT id FROM policy) SELECT id FROM Claim` was Approved with `tables=['policy']`, while
    `SELECT id FROM Claim` alone is refused `unauthorized_table` -- the CTE's mere presence was
    what unlocked the base table. Postgres folds the unquoted reference to `claim` and preserves
    `"Claim"`, so the two are different objects there and the read was never authorized. (M79.)

    When no defining identifier can be found, this now answers that the engine does NOT shadow --
    so the reference is treated as a real read and has to be found in `visible`. That is the
    opposite of what it did, and the reason is a record rather than a principle: THREE shapes
    reached the old fail-open answer within a day, each time under a note claiming none could.
    A recursive CTE, whose self-reference binds to one BRANCH of the union; a parenthesised body,
    which wraps the query in a `Subquery` that names nothing; and a `VALUES` alias, where the
    source is not reached by walking at all. Each was the M79 leak again, and each was closed by
    teaching the walk one more shape.

    Failing closed retires the class instead of the instance: a source whose definer cannot be
    identified is one this cannot reason about, and the safe answer for a governance guard is to
    check the name rather than to assume it is local.

    Its cost is measured, not assumed, because a guard that refuses legitimate analytics SQL is
    broken rather than safe. The branch is consulted ONLY for a table NODE whose source is not an
    `exp.Table` -- so a derived table, a CTE, a LATERAL or a UNION branch never reaches it unless a
    reference COLLIDES with its name. Thirteen ordinary shapes produce zero hits on sqlglot
    25.34.1, 26.16.4, 28.0.0 and 30.12.0, and the suite is unchanged.

    Four versions are a SAMPLE, not the range: `pyproject.toml` declares `>=25.34.1` with no
    ceiling, so any later release is supported and unmeasured -- and this file already notes that
    scope internals differ across that span. The probe was run by hand against each install and
    nothing in the repo re-runs it; what the tests DO assert is the property the number rests on,
    that ordinary shapes never reach the branch at all.
    """
    alias = _defining_identifier(source)
    if alias is None:
        return False
    return resolve_identifier(alias, dialect) == resolve_name(table, dialect)


def _is_base_read(table: exp.Table, scope, dialect: str | None) -> bool:
    """Is this reference a real read, or does a local source stand in front of it?

    ONE predicate, shared by `base_tables` and `column_tables`, because the two disagreeing inside
    a single statement is the defect rather than a detail of it: with only `base_tables` taught the
    dialect, the audit list named a base table while `check_cls` still believed the qualifier was a
    CTE, and a DENIED column came back Approved on Oracle.
    """
    source = scope.sources.get(table.alias_or_name)
    if isinstance(source, exp.Table):
        return True
    return source is not None and not _engine_shadows(table, source, dialect)


def base_tables(ast: exp.Expression, dialect: str | None = None) -> list[exp.Table]:
    """Every `exp.Table` node that reads a real object in the source.

    A CTE alias is reported as a Table by `find_all`, and must not be authorized as a source
    object -- authorizing it would be a category error, and refusing it would break every
    valid WITH clause. But a name is a CTE only *where that CTE is in scope*: a reference
    inside the CTE's own body reads the base table it was named after, and so does one in a
    sibling CTE declared before it.

    M31: three guards each asked only whether the name appeared in a flat set of CTE names --
    `check_access`, `check_cls` and `apply_row_and_mask` -- and so each skipped both halves of
    `WITH t AS (SELECT ... FROM t) SELECT ... FROM t`. Table authorization, column deny/mask
    and row filters fell together, from one blind spot shared three ways. This resolver is
    deliberately the single answer to "is this node a real read", because three copies of a
    scope rule is how they came to disagree in the first place.
    """
    root = _root_scope(ast)
    if root is None or _unscoped_ctes(ast, root):
        # Unresolvable scopes mean we cannot say which names are local, so every table node
        # is treated as a real read. That over-reports -- a CTE reference will not be in
        # `visible` and the query is refused -- which is the direction to fail in: the
        # alternative, falling back to the flat name set, is exactly the M31 bypass.
        return list(ast.find_all(exp.Table))

    out: list[exp.Table] = []
    for scope in root.traverse():
        for table in scope.tables:
            if _is_base_read(table, scope, dialect):
                out.append(table)
    # The write target, which no scope names. Added here rather than in each guard so the three
    # of them keep sharing one premise -- the M31 lesson, and the reason M48 sits in this file.
    for target in _target_reads(ast):
        if not any(t is target for t in out):
            out.append(target)
    return out


def column_tables(ast: exp.Expression, dialect: str | None = None) -> dict[int, str] | None:
    """`id(column node)` -> the base table its qualifier names **in that column's own scope**,
    or **None** when the scopes could not be resolved at all.

    None and `{}` are different answers and the difference is the finding. An empty map used to
    mean both "resolved, and no column resolves to a base table" and "the resolver failed", so
    a failure was read as the former: `cls` concluded a denied column belonged to no table and
    let it through. That is the same collapse as M2's outage-versus-empty-policy and M34's
    missing-versus-unreadable baseline -- an absence and a failure wearing one value.

    A missing entry within a returned map still means only "this qualifier names a CTE or a
    derived table".

    M31 made `base_tables` scope-aware and stopped there. Both consumers went on building ONE
    alias->table dictionary across every scope, so in

        SELECT q.secret FROM inner_t q UNION ALL SELECT q.secret FROM outer_t q

    the second `q` overwrote the first and a denied column was resolved against the permitted
    table. The resolver was right; the map that consumed it was still flat.
    """
    root = _root_scope(ast)
    if root is None or _unscoped_ctes(ast, root):
        return None

    out: dict[int, str] = {}
    for scope in root.traverse():
        # Built from `_is_base_read`, not from `sources` alone: a reference sqlglot bound to a
        # same-named CTE that the ENGINE would not shadow is a base table, and reading `sources`
        # here left it unattributed -- so `check_cls` treated a denied column's qualifier as a CTE
        # and let it through, on the very statement `base_tables` had just called a real read.
        local = {
            table.alias_or_name: object_key(table)
            for table in scope.tables
            if _is_base_read(table, scope, dialect)
        }
        for column in getattr(scope, "columns", ()):
            table = local.get(column.table)
            if column.table and table is not None:
                out[id(column)] = table

    # Columns qualified by the write target resolve to it. `ON CONFLICT DO UPDATE SET amount =
    # scratch.amount + 1` reads `scratch.amount` from a table in no scope, so without this the
    # map has no entry and `check_cls` falls back to the referenced set -- checking a denied
    # column against the wrong table. Only fills entries the scope walk did not, so a real
    # source or CTE of the same name keeps its scoped answer.
    for target in _target_reads(ast):
        key = object_key(target)
        for column in ast.find_all(exp.Column):
            if column.table == target.alias_or_name and id(column) not in out:
                out[id(column)] = key
    return out


def candidate_tables(
    column: exp.Column, resolved: dict[int, str] | None, referenced: set[str], aliased: set[str]
) -> set[str]:
    """The base table(s) a column could belong to.

    Lives here, with the other scope resolution, because two guards need the same answer --
    `check_cls` and `check_opaque_columns` -- and the second had its own flat alias map for one
    commit, keyed on the written case, which let `SELECT Q.leaked FROM vc_t Q` through while
    refusing the lowercase spelling. `check_access` resolves columns from `column_tables`
    directly rather than through this, which is a third copy of the same idea and worth knowing
    about before a fourth is written.

    Qualified -> the table its alias names IN ITS OWN SCOPE, which is why this takes a
    per-node map rather than a name->table dictionary: one alias can mean two tables in one
    statement, and a flat map kept whichever was seen last (Codex review, 2026-08-12).

    Unqualified -> every referenced base table, fail-closed. An alias the resolver could not
    place is treated the same way rather than as "no table": unresolvable is not permission.
    """
    if not column.table or resolved is None:
        # `resolved is None` -> the scopes could not be read, so nothing here is known to be a
        # local alias. Every referenced base table is a candidate; an unreadable statement is
        # not an argument for permission.
        return referenced
    table = resolved.get(id(column))
    if table is not None:
        return {table}
    # A qualifier naming a CTE or derived table owns no base column -- but only when we are
    # sure that is what it is. Otherwise fall back to every table.
    return set() if column.table in aliased else referenced
