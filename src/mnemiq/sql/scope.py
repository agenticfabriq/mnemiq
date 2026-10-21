from __future__ import annotations

import re
import threading

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.scope import build_scope

from mnemiq.sql.identifiers import (resolve, resolve_identifier, resolve_name, resolve_stored,
                                    rules_for)
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

    DEFENCE IN DEPTH, and not the live control: `check_write_shape` moves a leading WITH into the
    one query that uses it, or refuses it with UNSCOPED_CTE, so no write reaches here from the
    write decider carrying one (M30). Kept because that is ONE guard's decision while
    `base_tables` is the premise `check_access`, `check_cls` and `apply_row_and_mask` all share,
    and a resolver must not answer `[]` for a statement it never read. Measured with the shape refusal out of the way: these return `['claim', 'scratch',
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


def _bound_root(ast: exp.Expression, dialect: str | None):
    """`_root_scope`, bound as the engine binds it before any consumer reads it
    (`_bind_folded_ctes`); every consumer of the tree starts here."""
    root = _root_scope(ast)
    if root is not None:
        _bind_folded_ctes(root, dialect)
    return root


def _bind_folded_ctes(root, dialect: str | None) -> None:
    """Re-bind, in every scope of the tree, each table reference to the CTE the engine folds
    its name to (`_cte_binding`) and whose scope the tree holds (`_cte_scope`: a CTE above
    the tree, as the join check reads a CTE body on its own, is bound by nothing here), so
    every consumer of the tree reads that CTE's scope as it does for a CTE sqlglot matched
    by spelling: the base-read predicate, the column path, the star expansion and the
    output names alike. sqlglot keys a CTE's scope by its spelling as written and
    matches a reference by that spelling, where DuckDB and Oracle fold both (M202: with
    `w(id, secret)` granted and `claim.secret` withheld, `WITH W AS (SELECT * FROM claim)
    SELECT w.secret FROM w` read the granted base table here and the CTE there). Bound in
    two consumers by hand first, a star over the folded reference (`(SELECT * FROM w) d`, a
    second CTE over it) still expanded from the granted table's columns (the review gate on
    the first commit): the tree is bound once instead. sqlglot computes `selected_sources`
    from `sources` on first access, which no consumer makes before this.

    Every table reference of the scope is re-bound, not only one sqlglot left as a table: a
    reference it matched by spelling to a CTE of an OUTER scope, where a nearer WITH declares
    the name in another case, binds to the nearer one as the engine binds it (Codex's first
    review of #174: `WITH w AS (SELECT * FROM safe) SELECT d.secret FROM (WITH W AS (SELECT *
    FROM claim) SELECT * FROM w) d` read the outer `w` here, claim's secret on DuckDB, hidden,
    denied and masked alike). Where `_cte_binding` names no CTE the binding sqlglot made
    stands, as before."""
    for scope in root.traverse():
        for table in scope.tables:
            name = table.alias_or_name
            if name not in scope.sources:
                continue
            cte = _cte_binding(table, dialect)
            bound = _cte_scope(scope, cte) if cte is not None else None
            if bound is None or scope.sources[name] is bound:
                continue
            if any(node is table for node in bound.expression.walk()):
                continue  # a scope holding the reference itself: a loop no consumer may walk (Codex's second review of #174)
            scope.sources[name] = bound


def scope_resolved(ast: exp.Expression, dialect: str | None = None) -> bool:
    """Did the scope resolve, so that `base_tables` is a resolved answer rather than a fallback?

    When it did not, `base_tables` returns `find_all(exp.Table)` -- CTE aliases included -- which
    OVER-reports. That is the right direction for a guard, because an over-reported name is not in
    `visible` and the query is refused. It is the wrong direction for an audit record, which would
    name an object the query never read. So the record needs to know, and the guard does not.
    """
    ast_root = _bound_root(ast, dialect)
    return (ast_root is not None and not _unscoped_ctes(ast, ast_root)
            and not _scope_missed_a_read(ast, _walked_reads(ast, ast_root, dialect), dialect))


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
        return True  # bound as the engine binds, a CTE the name folds to among them (`_bind_folded_ctes`)
    if source is None:
        return False
    if not _engine_shadows(table, source, dialect):
        return True
    # The spellings agree; WHERE the name binds may not. sqlglot binds a RECURSIVE CTE's own
    # name across its whole body, the engine in its recursive part alone (`_cte_in_scope`,
    # DuckDB's rule), so a reference in the anchor, in a UNION ALL branch short of the last or
    # anywhere in an INTERSECT or EXCEPT body reads the base table. Left to sqlglot, the walk
    # missed that read, and the column map's fallback held the name to the statement-wide
    # aliases and known names instead of to the table's grant: the middle branch of three
    # returned claim.secret hidden (the ui lane's replay of #170 at 3e98092).
    return not _cte_in_scope(table, dialect)


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
    root = _bound_root(ast, dialect)
    if root is None or _unscoped_ctes(ast, root):
        # Unresolvable scopes mean we cannot say which names are local, so every table node
        # is treated as a real read. That over-reports -- a CTE reference will not be in
        # `visible` and the query is refused -- which is the direction to fail in: the
        # alternative, falling back to the flat name set, is exactly the M31 bypass.
        return list(ast.find_all(exp.Table))

    out = _walked_reads(ast, root, dialect)
    missed = _missed_reads(ast, out, dialect)
    if missed:
        # Every table node, as an unresolvable scope gives -- but the reads the walk MISSED first,
        # so a guard refusing at the first name not granted names the table the query hid, not a
        # CTE it defined (review gate on Codex's review of #109: the model was told it may not
        # query `t`, its own CTE).
        return missed + [t for t in ast.find_all(exp.Table) if not any(m is t for m in missed)]
    return out


def _walked_reads(ast: exp.Expression, root, dialect: str | None) -> list[exp.Table]:
    """The real reads the scope walk names, plus any write target the write reads."""
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


def _scope_missed_a_read(ast: exp.Expression, walked: list[exp.Table], dialect: str | None) -> bool:
    """Did the scope walk leave out a table the statement reads? ONE predicate for `base_tables`,
    `column_tables` and `scope_resolved`: a resolver that falls back for tables but still trusts
    the same incomplete scope for columns let a denied column through (Codex review of #66:
    `claim.secret` copied into an ungoverned table on the locked sqlglot).

    On a WRITE, a table the walk left out that no CTE could stand in front of is a table the
    scope model MISSED, and an unreported read is one no guard sees: fail toward over-reporting,
    exactly as an unresolvable scope does. Not hypothetical (M121): sqlglot 30.17 began scoping
    UPDATE, its scope lists none of the FROM tables, and `UPDATE s SET a = c.a FROM c` reported
    `s` alone -- so `c` was neither grant-checked nor row-filtered. A name no CTE on an enclosing
    query defines (`_cte_in_scope`) can only be a real read, whatever this sqlglot's scope model
    thinks. The write's own TARGET is excused (an INSERT writes it without reading it; a target
    that is read is in `walked` through `_target_reads`).

    **Reads too, since Codex's review of M65 (#109).** This was writes-only, on the reasoning that
    an under-report on the read path was lineage's to classify and no read-path test failed. But
    the read decider's `check_access` reads the same list: a scalar subquery inside a VALUES that
    sits outside every FROM/JOIN position -- a set-operation branch, a CTE body, a LATERAL body --
    is outside the walk, so `WITH t(x) AS (VALUES ((SELECT ssn FROM customer)) UNION ALL SELECT 1)
    SELECT x FROM t` was APPROVED, measured on DuckDB and Postgres, and returned the ungranted SSN.
    A source whitelist cannot see a VALUES that is not a source. Coverage can: whatever the walk
    missed falls back to every table node. Measured cost on reads: none of 9,687 gold queries
    (BIRD mini-dev in SQLite, Postgres and MySQL, KaggleDBQA, BEAVER, Spider2-lite local).
    """
    return bool(_missed_reads(ast, walked, dialect))


def cte_references(ast: exp.Expression, dialect: str | None = None, *,
                   engine: str | None = None) -> list[exp.Table]:
    """The table nodes that name a CTE in scope where they stand -- what the fallback over-reports
    as reads when the walk missed one. `engine` is whose binding to model when it is not the
    executor's: a view body is bound by its own source engine (Codex's review of #138). None is
    DuckDB's rule, the one every decider needs."""
    return [table for table in ast.find_all(exp.Table) if _cte_in_scope(table, dialect, engine)]


def _missed_reads(ast: exp.Expression, walked: list[exp.Table], dialect: str | None) -> list[exp.Table]:
    """The table nodes `_scope_missed_a_read` counts: left out by the walk, not written, and not
    a CTE in scope -- reads the scope model missed."""
    written = _written_targets(ast)
    return [table for table in ast.find_all(exp.Table)
            if not any(t is table for t in walked) and not any(w is table for w in written)
            and not _cte_in_scope(table, dialect)]


# The engines a row or column policy may be decided for (M150). `_cte_in_scope` models DuckDB's
# binding of a name in a WITH unless told another engine's: earlier siblings in scope, the CTE itself only under RECURSIVE, a
# later sibling never. Postgres binds a later sibling under RECURSIVE and SQLite always, so there
# the policy rewrite would wrap a CTE's output as if it were the table -- measured on Postgres 16,
# `[40]` for `[1, 40]`. Oracle is on the list by measurement, not by modelling: it refuses a
# forward reference (ORA-32031) and rejects a rewritten self-reference (ORA-32042), so a mismatch
# there errors rather than answers (mnemiq-internal, 2026-10-14-m150-oracle-forward-cte-binding).
POLICY_TARGETS = frozenset({"duckdb", "oracle"})


def check_policy_target(ast: exp.Expression, policy, target: str | None):
    """Refuse a statement that names a CTE, under a row or column policy, decided for an engine
    outside POLICY_TARGETS. Production always decides as the adapter's own dialect -- duckdb, or
    oracle for an Oracle source -- so this is the precondition made a check: a deployment that ever
    executes model SQL elsewhere under a policy is refused, never answered wrong.

    Only a statement with a WITH: the engines differ in how they bind a CTE's NAME, so a statement
    naming no CTE binds the same everywhere, and refusing it would refuse every governed read on
    such a target for a mismatch it cannot have."""
    from mnemiq.sql.verdict import Refusal, RefusalCode

    if policy is None or policy.empty or (target or "").lower() in POLICY_TARGETS:
        return None
    if ast.find(exp.CTE) is None:
        return None
    return Refusal(
        code=RefusalCode.POLICY_TARGET_UNSUPPORTED,
        message=(
            f"This engine applies row and column policies to a statement with a WITH clause only "
            f"when it executes as one of {', '.join(sorted(POLICY_TARGETS))}, and this source is "
            f"decided as {target!r}. This is a deployment setting, not a limit on your grants."
        ),
        subject=None,
    )


# Engines that bind a name to ANY CTE of the WITH it is read inside: a later sibling, and the CTE
# itself, RECURSIVE or not. Measured on SQLite 3.50, with a table `b` holding 40: `WITH a AS
# (SELECT id FROM b), b AS (SELECT 1 AS id) SELECT id FROM a` read 1, from a subquery nested in
# `a` too, and a self-reference was refused as circular. Not Postgres: it binds a later sibling
# only under RECURSIVE, and `pg_get_viewdef`, where its view bodies come from, writes a RECURSIVE
# WITH in dependency order (measured on 16), which the earlier-siblings rule already reads right.
_EVERY_SIBLING = frozenset({"sqlite"})


def _in_recursive_part(table: exp.Table, cte: exp.CTE | None) -> bool:
    """Whether the reference sits in the recursive part of the CTE's body: the right side of
    the body's top-level UNION, ALL or not. Only a UNION makes a CTE recursive; an INTERSECT or
    EXCEPT body is a plain CTE to the engine, its own name the base table everywhere in it, and
    of three UNION ALL branches only the LAST is the recursive part (DuckDB, measured on the ui
    lane's replay of #170 at 3e98092: `WITH RECURSIVE claim AS (... INTERSECT SELECT id, secret
    FROM claim)` and the middle branch of three each returned the hidden column). A body with
    no UNION has no recursive part."""
    body = cte.this if cte is not None else None
    while isinstance(body, (exp.Subquery, exp.Paren)):
        body = body.this
    if not isinstance(body, exp.Union):
        return False
    node = table.parent
    while node is not None and node is not body:
        if node is body.right:
            return True
        node = node.parent
    return False


def _cte_in_scope(table: exp.Table, dialect: str | None = None,
                  engine: str | None = None) -> bool:
    """Is a CTE of this table's name in scope where the table is read? (`_cte_binding`.)"""
    return _cte_binding(table, dialect, engine) is not None


def _cte_scope(scope, cte: exp.CTE):
    """The sqlglot scope built for `cte`'s body, found from `scope` outward: the scopes sqlglot
    keys by the CTE's spelling as written, which is why a reference the engine folds to it
    was never matched to it (M202). None where no enclosing scope holds it. A RECURSIVE CTE's
    own reference binds to the scope over its body's left operand, one step down (`A UNION
    ALL B UNION ALL C` is `(A UNION ALL B) UNION ALL C`, and the reference binds to the scope
    over `A UNION ALL B`), as sqlglot binds the reference spelled as written: bound to the
    scope over the whole body, which holds the reference, the scope graph looped back on
    itself, and a consumer walking sources with no visited set would never end (the review
    gate on the second commit of #174); taken to the leftmost branch, a three-branch body
    matched nothing nearer and fell back to the whole body (the gate's third round). A
    parenthesised anchor, `((SELECT 1 AS id) UNION ALL ...)`, is read through its
    parentheses, as sqlglot's scope is over the SELECT inside them (Codex's second review of
    #174, where the wrapper matched nothing and the whole body was taken again, and
    `fields_as_members` recursed without end). Never a scope whose expression holds the
    reference itself (`_bind_folded_ctes`)."""
    body = cte.this
    anchor = body.left if isinstance(body, exp.SetOperation) else body
    while isinstance(anchor, (exp.Subquery, exp.Paren)) and isinstance(anchor.this, exp.Expression):
        anchor = anchor.this
    anc = scope
    while anc is not None:
        found = anc.cte_sources.get(cte.alias_or_name)
        if found is not None and (found.expression is body or found.expression is anchor):
            return found
        anc = anc.parent
    return None


def _cte_binding(table: exp.Table, dialect: str | None = None,
                 engine: str | None = None) -> exp.CTE | None:
    """The CTE in scope where the table is read that the engine binds this name to, or None.

    Only a WITH on one of the table's ENCLOSING queries can stand in front of it. A name matched
    anywhere in the statement is not enough: `UPDATE t SET a = (WITH claim AS (...) SELECT a FROM
    claim) FROM claim` declares `claim` inside a scalar subquery, and the outer `FROM claim` is
    the real table -- matched by name, it was skipped, never grant-checked, and the write was
    approved with `claim` ungranted on the locked sqlglot (M122). Inside a WITH's own CTE bodies
    only earlier siblings are visible, plus the CTE itself when the WITH is RECURSIVE (M31's rule,
    DuckDB's) -- unless `engine` is one in `_EVERY_SIBLING`, which binds every CTE of the WITH there,
    the CTE itself included: how a view body is bound when its source engine runs it.
    A qualified reference (`main.claim`) is never a CTE -- a CTE has no schema -- and names are
    resolved by `mnemiq.sql.identifiers`, the shared resolver (M55) -- which `authz_guard`'s and
    `cls`'s own CTE name sets do not use yet, so they may still disagree with this one: matching lowercased bare names excused `FROM
    main.claim` behind a CTE `claim`, and Postgres's `claim` behind `"Claim"`.
    Structural, never by arg name. In doubt the answer is None, which over-reports: a refusal.
    """
    if table.db or table.catalog:
        return None
    name = resolve_name(table, dialect)
    if name is None:
        return None
    child, node, last_cte = table, table.parent, None
    while node is not None:
        if isinstance(node, exp.CTE):
            last_cte = node
        for with_ in (v for v in node.args.values() if isinstance(v, exp.With)):
            names = [(resolve_name(cte, dialect), cte) for cte in with_.expressions]
            if child is with_ and (engine or "").lower() in _EVERY_SIBLING:
                visible = names
            elif child is with_:  # the table sits inside one of this WITH's own CTE bodies
                at = next((i for i, cte in enumerate(with_.expressions) if cte is last_cte), 0)
                # Under RECURSIVE the CTE's own name binds in its RECURSIVE part alone, the
                # right side of its body's UNION (`_in_recursive_part`); in the anchor, in an
                # earlier UNION ALL branch and anywhere in an INTERSECT or EXCEPT body it is the
                # base table (DuckDB, measured: `WITH RECURSIVE claim AS (SELECT ... FROM claim
                # UNION ALL ...)` read the base `claim` in the anchor, and a hidden join key
                # went unread when the whole body was taken as the CTE: Codex's fourth review
                # of #170; the other two shapes, the ui lane's replay at 3e98092).
                itself = names[at:at + 1] if with_.args.get("recursive") and _in_recursive_part(table, last_cte) else []
                visible = names[:at] + itself
            else:
                visible = names
            for visible_name, cte in visible:
                if visible_name == name:
                    return cte
        child, node = node, node.parent
    return None


def _written_targets(ast: exp.Expression) -> list[exp.Table]:
    """The table nodes a write statement writes, located by position: INSERT's `this` (inside the
    column-list Schema when there is one), UPDATE's `this`, DELETE's `this` or its `tables`. Empty
    otherwise -- MERGE included: it reads its target and `_target_reads` does not model it, so its
    target must stay a missed read, and fail closed, rather than be excused."""
    if not isinstance(ast, (exp.Insert, exp.Update, exp.Delete)):
        return []
    found: list[exp.Table] = []
    for node in [ast.this, *(ast.args.get("tables") or [])]:
        if isinstance(node, exp.Schema):
            node = node.this
        if isinstance(node, exp.Table):
            found.append(node)
    return found


def fields_as_members(ast: exp.Expression, dialect: str | None = None,
                      columns: dict[str, set[str]] | None = None):
    """Rewrite, in place, every column reference that reads a FIELD of another column into a
    field access over that column, so each column rule reads the column (M165).

    sqlglot reads a dotted name left to right as catalog, db, table and column, so
    `c.secret.status` is `Column(this=status, table=secret, db=c)` and `secret.status` is
    `Column(this=status, table=secret)`: the qualifier is the column and the name is its field.
    Every guard then saw a column called `status` qualified by something no scope defines, which
    `check_access` reads as a CTE or subquery and skips, and `check_cls` and the mask rewrite never
    saw `secret` at all. Measured on main `ec0ad7d`: with `secret` a DuckDB STRUCT or JSON column,
    or an Oracle JSON one, `SELECT c.secret.status FROM claim c` was approved and returned the
    field whether `secret` was left out of the grant, denied or masked, as did `secret.status`
    and a WHERE on it -- while `SELECT c.secret` was refused.

    So a reference whose qualifier names no source in its scope or an enclosing one becomes what
    sqlglot's own optimizer makes of it (`qualify_columns` turns such a column into a `Dot`):
    `c.secret.status` -> `Dot(Column(secret, table=c), status)` where `c` is a source, and
    `secret.status` -> `Dot(Column(secret), status)` where it is not. The base column then meets
    every rule a column meets, and a qualifier that names neither a source nor a column is an
    unknown column, refused, where before it was skipped. Renders to the same SQL.

    Names are compared as `dialect`, the engine, resolves them, so `C.id` against `claim c` is the
    source it names on Oracle and DuckDB alike. And on Oracle only the first form is a field:
    Oracle reads one only after a table alias -- `secret.status` and `claim.secret.status` are
    ORA-00904 without one, and `seq.NEXTVAL` is a sequence (all measured on 23) -- so there a
    reference not led by a source is left as written, as it was.

    Scope by scope, the sources each scope selects (a CTE no FROM names is no source there), the
    enclosing scopes' included for a correlated reference, and a write's targets in every scope: `build_scope` models the subqueries of a write, not the write,
    so `UPDATE scratch ... WHERE EXISTS (SELECT 1 FROM claim WHERE claim.id = scratch.id)` gives
    one scope, the subquery's, where `scratch` is no source. A column outside every scope, and
    every column of a statement whose scopes cannot be read, is read against every table, alias,
    CTE and subquery name in the statement, with `excluded` inside an ON CONFLICT clause.

    A reference led by a source with a source named further along it is refused instead, and
    returned, since the names do not settle which one the engine reads it through (Codex's review
    of #162). Both engines read `c.secret.status` as a field of `c.secret` with a source `secret`
    in the query, so leaving it a column of `secret` let a withheld `secret` be read; and DuckDB
    reads `s.claim.id` as schema `s`, table `claim`, with a source aliased `s` in the query, which
    Oracle calls ambiguous (ORA-00918; all measured). Two parts stay the source's column: there
    both engines read `secret.status` as the column of the source `secret`.

    On DuckDB a qualifier naming only an outer source, or a write's target, is refused as well
    when a nearer source has a column of that name (M167). DuckDB binds a qualifier scope by
    scope -- a source of that name, then a column of that name among the scope's sources -- before
    it looks outward, so in `SELECT (SELECT secret.status FROM claim) FROM other secret` it reads
    `claim.secret`, not `other.status` (measured, a write's subquery too); Oracle reads the outer
    source (measured on 23), so there nothing changes. `columns` is what the decider is told each
    table has (`known_columns`), and a subquery's or CTE's columns are its projection: `*` read
    through to every source its scope selects, `t.*` to `t` alone, EXCLUDE and RENAME applied
    (DuckDB folds their names too, measured), and a set operation through each branch. A scope
    whose sources share a name is refused on every engine: sqlglot cannot say which one a
    qualifier there names, and DuckDB runs one (`FROM claim c, (SELECT 1) c`).
    """
    # Each name a source can be known by, as written -> as the engine resolves it. A scope keys
    # its sources by the written text, so this is how a key is resolved.
    spelled: dict[str, set[str]] = {}
    for node in [*(a.this for a in ast.find_all(exp.TableAlias)),
                 *(t.this for t in ast.find_all(exp.Table))]:
        if isinstance(node, exp.Identifier) and node.name:
            spelled.setdefault(node.name, set()).add(resolve_identifier(node, dialect))

    def resolved(names) -> set[str]:
        return {r for name in names
                for r in spelled.get(name, {resolve(name, quoted=False, dialect=dialect)})}

    # Every name a statement selects from; a CTE's own name only where some FROM names it.
    every = {resolve_identifier(node, dialect)
             for node in [*(t.this for t in ast.find_all(exp.Table)),
                          *(a.this for a in ast.find_all(exp.TableAlias)
                            if not isinstance(a.parent, exp.CTE))]
             if isinstance(node, exp.Identifier) and node.name}
    upsert = every | {resolve("excluded", quoted=False, dialect=dialect)}
    targets = {t.alias_or_name for t in [*_target_reads(ast), *_written_targets(ast)]}
    if isinstance(ast, exp.Merge):
        targets |= {node.alias_or_name for node in (ast.this, ast.args.get("using")) if node}
    targets = resolved(targets)
    known = {table: {resolve(name, quoted=False, dialect=dialect) for name in names}
             for table, names in (columns or {}).items()}

    def columns_of(source) -> set[str]:
        if isinstance(source, exp.Table):
            return set(known.get(object_key(source), ()))  # a copy: EXCLUDE edits it per source
        if source.union_scopes:
            return set().union(*(columns_of(branch) for branch in source.union_scopes))
        found: set[str] = set()
        for select in source.expression.selects:
            star = select if isinstance(select, exp.Star) else (
                select.this if isinstance(select, exp.Column) and isinstance(select.this, exp.Star)
                else None)
            if star is None and isinstance(select, exp.Columns):
                star = exp.Star()  # COLUMNS(...) picks by pattern: read as everything (M169)
            if star is not None:
                # `t.*` is t's columns; `*` and a `t` naming nothing are every source's (M169).
                def named(qualifier) -> list[str]:
                    return [name for name in source.selected_sources
                            if not qualifier or resolved({name}) == resolved({qualifier})]

                picked = named(getattr(select, "table", None)) or list(source.selected_sources)
                exposed = {name: columns_of(source.selected_sources[name][1]) for name in picked}
                # An EXCLUDE or RENAME item applies to the source its qualifier names, else to
                # every source (measured: `EXCLUDE (c.secret) FROM claim c, claim k` keeps
                # k.secret, `EXCLUDE (secret)` drops both). REPLACE keeps every name.
                for item in star.args.get("except_") or ():
                    column = item if isinstance(item, exp.Column) else exp.Column(this=item)
                    for name in named(column.table):
                        exposed.get(name, set()).discard(resolve_identifier(column.this, dialect))
                # RENAME items (`old AS new`) all apply to the original names at once, so
                # `RENAME (id AS secret, secret AS s2)` still exposes `secret`; one naming a column
                # the picked source lacks is ignored, so `p.* RENAME (amount AS secret)` over
                # plain p exposes plain's `id` alone (both measured; Codex's reviews of #165).
                for name, columns in exposed.items():
                    renames = [(resolve(item.this.name, quoted=False, dialect=dialect),
                                resolve(item.alias, quoted=False, dialect=dialect))
                               for item in star.args.get("rename") or ()
                               if name in named(item.this.table)]
                    renames = [(old, new) for old, new in renames if old in columns]
                    exposed[name] = ((columns - {old for old, _ in renames})
                                     | {new for _, new in renames})
                found |= set().union(*exposed.values())
            elif select.alias_or_name:
                found.add(resolve(select.alias_or_name, quoted=False, dialect=dialect))
        return found

    binds_nearer = (dialect or "").lower() != "oracle"
    seen: set[int] = set()
    root = _bound_root(ast, dialect)
    if root is not None and not _unscoped_ctes(ast, root):
        scopes = list(root.traverse())
        try:
            selected = {id(scope): resolved(scope.selected_sources) for scope in scopes}
        except OptimizeError:
            # Two sources of one scope go by one name, which DuckDB runs (`FROM claim, claim`), so
            # which one a qualifier reads cannot be told: refused, not read as either.
            return _sources_share_a_name()
        for scope in scopes:
            chain = []
            enclosing = scope
            while enclosing is not None:
                chain.append(enclosing)
                enclosing = enclosing.parent
            levels = [selected[id(level)] for level in chain]
            names = set(targets).union(*levels)
            for column in list(getattr(scope, "columns", ())):
                # Once, in the nearest scope that lists it. sqlglot lists a subquery's external
                # columns in the enclosing scope too, and calls `D.secret` external when the
                # source is `d`, matching the alias as written; read again against the outer
                # sources it became a field of a column `D` (M166).
                if id(column) in seen:
                    continue
                seen.add(id(column))
                refused = (_nearer_column(column, chain, levels, targets, columns_of, dialect)
                           if binds_nearer else None) or _as_member(column, names, dialect)
                if refused is not None:
                    return refused
    for column in list(ast.find_all(exp.Column)):
        if id(column) not in seen:
            in_upsert = column.find_ancestor(exp.OnConflict) is not None
            refused = _as_member(column, upsert if in_upsert else every, dialect)
            if refused is not None:
                return refused
    return None


def _as_member(column: exp.Column, sources: set[str], dialect: str | None):
    if not column.table:
        return None
    parts = [column.args[key] for key in ("catalog", "db", "table") if column.args.get(key)]
    named = [resolve_identifier(part, dialect) in sources for part in parts]
    if named[0] and len(parts) > 1:
        if any(named[1:]):
            return _two_readings(column, parts[named.index(True, 1)].name, dialect)
        base = exp.Column(this=parts[1].copy(), table=parts[0].copy())  # alias.column.field
        fields = parts[2:]
    elif named[-1]:
        return None  # a column of the source its qualifier names
    elif (dialect or "").lower() == "oracle":
        return None  # not a field on Oracle, see `fields_as_members`
    else:
        base = exp.Column(this=parts[0].copy())  # column.field
        fields = parts[1:]
    member: exp.Expression = base
    for field in [*fields, column.this]:
        member = exp.Dot(this=member, expression=field.copy())
    column.replace(member)
    return None


def _nearer_column(column: exp.Column, chain, levels, targets, columns_of, dialect: str | None):
    """A qualifier naming only a source beyond the nearest scope, where a nearer source has a
    column of that name, refused: DuckDB reads it as that column (M167, `fields_as_members`)."""
    if not column.table:
        return None
    lead = column.parts[0]
    name = resolve_identifier(lead, dialect)
    for depth, level in enumerate(levels):
        if name in level:
            nearer = chain[:depth]
            break
    else:
        if name not in targets:
            return None  # names no source: a field of a column, rewritten as one
        nearer = chain  # a write's target, outside every scope `build_scope` models
    for scope in nearer:
        for _, source in scope.selected_sources.values():
            if name in columns_of(source):
                from mnemiq.sql.verdict import Refusal, RefusalCode

                return Refusal(
                    code=RefusalCode.UNKNOWN_COLUMN,
                    message=(f"{column.sql(dialect=dialect)!r} can be read two ways here: through "
                             f"the source {lead.name!r} outside this subquery, or as a field of a "
                             f"column {lead.name!r} of a table inside it. Give that source "
                             "another alias, so that it reads one way."),
                    subject=lead.name,
                )
    return None


def _sources_share_a_name():
    from mnemiq.sql.verdict import Refusal, RefusalCode

    return Refusal(
        code=RefusalCode.UNMODELLED_SOURCE,
        message=("Two sources in one FROM go by the same name here, so which one a reference "
                 "reads cannot be told. Give each source its own alias."),
        subject=None,
    )


class KnownColumns(dict):
    """A schema, table key to column names, that also says which tables it lists in full:
    `complete` holds the policy schema's tables, every column of the snapshot's (M181). A table
    listed in full that lacks a name does not have it; one listed from the grant alone may hide
    it."""

    complete: frozenset[str] = frozenset()


def known_columns(visible, policy=None) -> KnownColumns:
    """Every column the decider is told a table has: the grant's, and the policy's (M167)."""
    known = KnownColumns({table: set(names) for table, names in (visible or {}).items()})
    if policy is not None:
        for table, names in (policy.policy_schema or {}).items():
            known.setdefault(table, set()).update(names)
        for table, name in (*policy.denied, *policy.masked):
            known.setdefault(table, set()).add(name)
        known.complete = frozenset(policy.policy_schema or {})
    return known


def _two_readings(column: exp.Column, other: str, dialect: str | None):
    """A reference led by a source with another source named further along it, refused (Codex's
    review of #162): which one the engine reads it through is not settled by the names."""
    from mnemiq.sql.verdict import Refusal, RefusalCode

    return Refusal(
        code=RefusalCode.UNKNOWN_COLUMN,
        message=(f"{column.sql(dialect=dialect)!r} can be read two ways here: as a field of a column of "
                 f"{column.parts[0].name!r}, or through {other!r}, which this query also names. "
                 f"Give {other!r} another alias, so that it reads one way."),
        subject=other,
    )


# What a derived source exposes under a name: ("table", key) binds the column to a base table,
# `_NAMED` says the source's own projection names it (checked in that scope), ("first", key) is
# the fail-closed answer for a name no schema lists under a star (that table's check refuses it
# as unknown), and None says nothing here exposes it, so the next scope out may (M182).
_NAMED = ("named",)


def _star_of(projection: exp.Expression) -> exp.Star | None:
    """The star a projection is, qualified or not, COLUMNS(...) read as one (M169), an alias
    template over it too (`check_shape` refuses that one; here it still expands the sources)."""
    if isinstance(projection, exp.Alias) and isinstance(projection.this, (exp.Columns, exp.Star)):
        projection = projection.this
    if isinstance(projection, exp.Star):
        return projection
    if isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
        return projection.this
    if isinstance(projection, exp.Columns):
        return projection.this if isinstance(projection.this, exp.Star) else exp.Star()
    return None


def _columns_pattern(projection: exp.Expression) -> str | None:
    """The regular expression a `COLUMNS('...')` projection picks columns by, else None (a
    `COLUMNS(*)`, a lambda, or no COLUMNS at all)."""
    if isinstance(projection, exp.Alias):
        projection = projection.this
    if isinstance(projection, exp.Columns) and isinstance(projection.this, exp.Literal) \
            and projection.this.is_string:
        return projection.this.this
    return None


_REGEX_LOCK = threading.Lock()
_REGEX_READER = None


def _pattern_picks(pattern: str, candidates) -> bool:
    """Whether DuckDB's `COLUMNS('regex')` picks a column of one of these stored spellings,
    asked of DuckDB's own regular expressions (RE2, through `regexp_matches`, which `COLUMNS`
    uses: a search, case kept; measured: `COLUMNS('i')` picks `id`, `COLUMNS('ID')` nothing),
    never of Python's, which read some patterns differently (the review gate on #167). Any
    error, DuckDB's refusal of the pattern among them, reads as picking, fail-closed. Only a
    stored spelling can be asked: a derived source's output names are known folded alone, so a
    pattern over one keeps the whole-star reading (the gate: a case-folding stand-in can be
    switched off by an inline flag in the pattern)."""
    global _REGEX_READER
    try:
        from mnemiq.duckdb_extensions import connect_duckdb

        with _REGEX_LOCK:
            if _REGEX_READER is None:
                _REGEX_READER = connect_duckdb(local_only=True)
            for candidate in candidates:
                row = _REGEX_READER.execute("SELECT regexp_matches($1, $2)",
                                            [candidate, pattern]).fetchone()
                if row is not None and row[0]:
                    return True
        return False
    except Exception:  # noqa: BLE001 -- unsure: the pattern may pick it
        return True


def _stored_matches(stored: str, name: str, dialect: str | None) -> bool:
    """Whether a stored column spelling is the one a resolved reference name reads."""
    if rules_for(dialect).quoting_preserves_case:
        return stored == name
    return resolve_stored(stored, dialect) == name or stored == name


def _schema_lists(columns, key: str, column: exp.Column, dialect: str | None) -> bool:
    """Whether the schema lists table `key` with the column, compared as the engine compares names
    (`check_access`'s rule, M170): both sides fold where the engine folds, as typed where quoting
    keeps case."""
    if columns is None or key not in columns:
        return False
    stored = columns[key]
    if rules_for(dialect).quoting_preserves_case:
        return column.name in stored
    name = resolve_identifier(column.this, dialect)
    return name in ({resolve_stored(n, dialect) for n in stored} | set(stored))


def _names_item(item: exp.Expression, name: str, dialect: str | None) -> bool:
    identifier = item.this if isinstance(item, exp.Column) else item
    return resolve_identifier(identifier, dialect) == name


def _item_sources(item: exp.Expression, picked: list[str], dialect: str | None) -> list[str]:
    """The picked sources an EXCLUDE, RENAME or REPLACE item applies to: its qualifier's, else
    every one (measured for EXCLUDE: `EXCLUDE (c.secret) FROM claim c, claim k` keeps k.secret)."""
    qualifier = (resolve_identifier(item.args.get("table"), dialect)
                 if isinstance(item, exp.Column) else None)
    if not qualifier:
        return picked
    return [source for source in picked
            if resolve(source, quoted=False, dialect=dialect) == qualifier]


# The owner `column_tables` reports for a column the engine named itself, a duplicate output
# renamed `name_1`: no table, a refusal. `check_access` refuses it outright; no listed column
# can carry it, and a table binding is not fail-closed where some table happens to have a column
# of the generated name (Codex's fifth review of #167: `wide.secret_1`, granted, let `d.secret_1`
# over `(SELECT 'SAFE' AS secret, c.* FROM wide w CROSS JOIN claim c) d` read claim's secret).
GENERATED_NAME = "\0generated-name"


def _exposed(source, column: exp.Column, columns, dialect: str | None):
    """What a derived source exposes under a column's name, DuckDB's renaming of duplicate
    output names allowed for. A later column of a name an earlier output already took is
    renamed `name_1`, `name_2`, ... by rules of collision no name-based reading follows exactly
    (measured: `(SELECT 'SAFE' AS secret, * FROM claim) d` exposes claim's secret as `secret_1`;
    an explicit `secret_1` after it is itself renamed; a taken `secret_1` moves the next to
    `secret_2`; Codex's fourth review of #167 and the gate read a withheld column through each).
    So a name of that form is read as the engine's own wherever the derived output carries its
    base name twice (the engine renames only a duplicate), or where the output's names cannot
    be told: `GENERATED_NAME`, which is refused. A name of that form over an output that
    carries the base once or never is an honest column, read directly (the ui lane's replay:
    `end_hour_24` beside one `end_hour` was refused, and nothing was renamed)."""
    suffixed = re.fullmatch(r"(.+)_([1-9]\d*)", column.name)
    if suffixed is not None:
        names = _output_names(source, columns, dialect, frozenset())
        base = resolve(suffixed[1], quoted=bool(column.this.args.get("quoted")), dialect=dialect)
        if names is None or names.count(base) > 1:
            return ("table", GENERATED_NAME)
    name = resolve_identifier(column.this, dialect)
    return _exposure(source, name, column, columns, dialect, frozenset())


def _exposure(scope, name: str, column: exp.Column, columns, dialect: str | None,
              seen: frozenset[int], listed: tuple[str, ...] | None = None):
    """What `scope` (a derived table's or CTE's) exposes under `name`, in projection order (M181).

    The first projection that can expose the name settles it, as DuckDB binds the first column
    of a name in projection order (measured: `(SELECT *, 'NAMED' AS secret FROM claim) d` reads
    claim's secret, `(SELECT 'NAMED' AS secret, * FROM claim) d` the literal). A named
    projection settles it in its own scope. A star settles it on the first source under it, in
    FROM order, that the schema lists with the name (DuckDB binds the first source that has it,
    measured on a join of two tables both holding it), unless an EXCLUDE drops it or a RENAME
    moves it away from that source; a RENAME to the name or a REPLACE of it is a named column,
    its own nodes checked. A star whose sources are listed with the name nowhere still settles
    it, on a base table under it: the first the schema does not list, which may well have it (a
    partial schema, the eval criterion's or a policy's alone), else the first listed without it,
    where the name is hidden or absent and that table's check refuses it as unknown. Read as
    unexposed instead, the column would bind outward to a table that grants the name (the ui
    lane's edges on M181), and an unlisted table taking the name FIRST let `wide` own `secret`
    over a join with claim and dropped claim's mask (the review gate).
    """
    if not hasattr(scope, "expression"):
        return _NAMED  # a table function or another source whose columns are its own
    if id(scope) in seen:
        return None  # a recursive CTE naming itself
    if isinstance(scope.expression, exp.Lateral):
        # A LATERAL source is its query, read as any derived source, under the alias's column
        # list where there is one (the ui lane's replay of #167: `LATERAL (SELECT * FROM claim)
        # l(x, y)` served secret as `l.y`, and `l.secret` through the star went unbound).
        inner = _lateral_inner(scope)
        if inner is None:
            return None
        listed = _column_list(scope, dialect) or listed
        if not isinstance(getattr(inner, "expression", None), (exp.Select, exp.SetOperation)):
            # VALUES or a table function under it: no table behind the list's names.
            if listed and name in listed and inner.expression.find(exp.Table) is None:
                return _NAMED
            return None
        return _exposure(inner, name, column, columns, dialect, seen, listed)
    if listed is None:
        listed = _column_list(scope, dialect)
    if listed and name in listed:
        # The list names the projection at the name's position (the ui lane's replay: `WITH w(n)
        # AS (SELECT 1)` read bare and qualified went unexposed and fell closed outward); a star
        # there is read by position, which `bound_by_position` refuses where it reaches a base
        # table, so nothing is exposed under the name here.
        if scope.union_scopes:
            return _exposure(scope.union_scopes[0], name, column, columns, dialect, seen, listed)
        index = listed.index(name)
        selects = scope.expression.selects
        if index < len(selects) and _star_of(selects[index]) is None:
            return _NAMED
        return None
    # The positions a list renames expose nothing under their old names: `WITH w(x) AS (SELECT 1
    # AS id FROM wide)` has no `id`, and `w.id` inside a subquery over it binds outward, to an
    # outer source called w (Codex's eighth review of #167: read as the named projection, it let
    # claim's id through under the complete schema). A star among those positions is read by
    # position, which `bound_by_position` refuses where it reaches a base table.
    renamed = len(listed) if listed else 0
    seen = seen | {id(scope)}
    if scope.union_scopes:
        # The branches' names are the first branch's. A star in a branch binds by position,
        # which no name can read: `check_shape` refuses it (M183); here, fail-closed, the first
        # base table under the branches.
        for branch in scope.union_scopes:
            for projection in branch.expression.selects:
                if _star_of(projection) is not None:
                    first = next((object_key(source)
                                  for _, source in branch.selected_sources.values()
                                  if isinstance(source, exp.Table)), None)
                    return ("first", first) if first else _NAMED
        return _exposure(scope.union_scopes[0], name, column, columns, dialect, seen)
    for position, projection in enumerate(scope.expression.selects):
        star = _star_of(projection)
        if position < renamed:
            if star is not None:
                return None
            continue
        if star is None:
            alias = projection.alias_or_name
            if alias and resolve(alias, quoted=False, dialect=dialect) == name:
                return _NAMED
            continue
        # `COLUMNS('id')` expands only the names its pattern picks, so a later projection of a
        # name it does not pick is read as the engine reads it (Codex's fourteenth review of
        # #167: `(SELECT COLUMNS('id'), 7 AS secret FROM claim) d` read `d.secret` through the
        # expansion and refused the literal on a hidden claim.secret); asked per source below,
        # against the stored spellings the schema lists.
        pattern = _columns_pattern(projection)
        qualifier = (resolve_identifier(projection.args.get("table"), dialect)
                     if isinstance(projection, exp.Column) else None)
        picked = [source for source in scope.selected_sources
                  if not qualifier or resolve(source, quoted=False, dialect=dialect) == qualifier]
        if any(item.alias and resolve(item.alias, quoted=False, dialect=dialect) == name
               for item in star.args.get("replace") or ()):
            return _NAMED  # `* REPLACE (id AS secret)`: the name holds the expression's nodes
        unlisted = listed_without = None
        for source in picked:
            excluded = any(_names_item(item, name, dialect)
                           for item in star.args.get("except_") or ()
                           if source in _item_sources(item, picked, dialect))
            renames = [item for item in star.args.get("rename") or ()
                       if source in _item_sources(item.this, picked, dialect)]
            if any(resolve(item.alias, quoted=False, dialect=dialect) == name for item in renames):
                return _NAMED  # `RENAME (secret AS s2)`: `s2` is the column node `secret`, checked
            if excluded or any(_names_item(item.this, name, dialect) for item in renames):
                continue  # dropped, or moved away from this source
            node, inner = scope.selected_sources[source]
            if isinstance(inner, exp.Table) or (
                    isinstance(node, exp.Table) and _is_base_read(node, scope, dialect)):
                key = object_key(inner if isinstance(inner, exp.Table) else node)
                if pattern is not None and columns is not None and key in columns:
                    stored = [n for n in columns[key] if _stored_matches(n, name, dialect)]
                    if stored and not _pattern_picks(pattern, stored):
                        continue  # listed, and the pattern picks none of its spellings
                if _schema_lists(columns, key, column, dialect):
                    return ("table", key)
                if columns is not None and key in columns:
                    # Listed in full (the policy schema's, every column of the snapshot's) and
                    # lacking the name: the table does not have it, and the engine reads on to a
                    # later projection of that name (`SELECT d.rn FROM (SELECT *, 1 AS rn FROM
                    # claim) d` reads the literal; the ui lane's replay, three statements refused
                    # when this fell closed). Listed from the grant alone: it may hide the name.
                    if key not in getattr(columns, "complete", ()):
                        listed_without = listed_without or key
                else:
                    unlisted = unlisted or key
                continue
            found = _exposure(inner, name, column, columns, dialect, seen)
            if found is None:
                continue
            if found[0] == "first":
                listed_without = listed_without or found[1]
                continue
            return found
        if unlisted or listed_without:
            return ("first", unlisted or listed_without)  # the star may expose it: settled here
    return None


def _scope_names(scope, dialect: str | None):
    """What a scope's qualifiers resolve to: `local`, the base tables it reads, by the alias's
    resolved spelling; `by_alias`, every other source it selects from (a CTE's or derived
    table's scope, a table function), the same way."""
    # Built from `_is_base_read`, not from `sources` alone: a reference sqlglot bound to a
    # same-named CTE that the ENGINE would not shadow is a base table, and reading `sources`
    # here left it unattributed -- so `check_cls` treated a denied column's qualifier as a CTE
    # and let it through, on the very statement `base_tables` had just called a real read.
    # Keyed by what the alias RESOLVES to, and looked up the same way (M166): `C.secret FROM
    # claim c` is `c.secret` to every engine (quoted `"C"` too, on Oracle), and matched as
    # written it named nothing, which read as a derived table's alias and went unchecked.
    local = {
        resolve_identifier(table.args.get("alias") or table.this, dialect): object_key(table)
        for table in scope.tables
        if _is_base_read(table, scope, dialect)
    }
    by_alias = {}
    for node in [*scope.tables, *scope.derived_tables, *scope.udtfs]:
        alias = resolve_identifier(
            node.args.get("alias") or (node.this if isinstance(node, exp.Table) else None),
            dialect)
        if alias is None and isinstance(node, exp.Subquery):
            # An unaliased derived table (`FROM (SELECT ...)`, which DuckDB, SQLite and Oracle
            # accept) is a source no qualifier can name; sqlglot keys it by the empty name, and
            # read as no source at all its bare names bound outward (the ui lane's replay of
            # #167: a false refusal from a CTE body, and `SELECT secret FROM (SELECT *, 'SAFE'
            # AS secret FROM claim)` reading claim's secret, which the aliased twin refused).
            alias = ""
        if alias is not None and alias not in local:
            by_alias[alias] = scope.sources.get(node.alias_or_name)
    return local, by_alias


def _column_list(scope, dialect: str | None) -> tuple[str, ...] | None:
    """The column list a CTE's, a LATERAL's or a derived table's alias puts on the scope's
    output, by position (`WITH w(n) AS (...)`, `LATERAL (...) l(x, y)`, `(...) d(x, y)`),
    resolved as the engine resolves names."""
    expression = getattr(scope, "expression", None)
    holder = expression if isinstance(expression, exp.Lateral) else getattr(expression, "parent", None)
    while isinstance(holder, exp.SetOperation) or (
            isinstance(holder, exp.Subquery) and not _alias_columns(holder)):
        # A recursive CTE's anchor, which sqlglot hands a self-reference inside the recursive
        # branch (the ui lane's replay: `WITH RECURSIVE m(a) AS (SELECT 1 UNION ALL SELECT a + 1
        # FROM m WHERE a < 3)` read nowhere else was refused on `a`); a parenthesised body,
        # branch or derived table, which sqlglot wraps in a subquery of its own, the list on the
        # outermost (the replay at cb90806: `WITH w(x) AS ((SELECT 'SAFE' AS secret FROM wide))`
        # kept `secret` exposed; the review gate: `((SELECT 1 AS secret)) d(x)` the same).
        holder = holder.parent
    # A derived table's alias carries a list too (`(SELECT 1 AS secret) d(x)`; Codex's ninth
    # review of #167: left unread, the old name stayed exposed and `d.secret` passed where the
    # engine binds it outward to a hidden column).
    if not isinstance(holder, (exp.CTE, exp.Lateral, exp.Subquery)):
        return None
    alias = holder.args.get("alias")
    listed = alias.args.get("columns") if alias is not None else None
    if not listed:
        return None
    return tuple(resolve_identifier(identifier, dialect) for identifier in listed)


def _alias_columns(holder) -> list:
    alias = holder.args.get("alias")
    return (alias.args.get("columns") or []) if alias is not None else []


def _lateral_inner(scope):
    """The scope of the query under a LATERAL source (sqlglot keys it by an empty name)."""
    subquery = scope.expression.this
    return next((inner for inner in scope.sources.values()
                 if getattr(inner, "expression", None) is not None
                 and inner.expression.parent is subquery), None)


def _output_names(scope, columns, dialect: str | None, seen: frozenset[int]) -> list[str] | None:
    """The names a derived scope's output carries, in order, duplicates kept, or None where a
    source's names cannot be told (a table the schema does not list, a COLUMNS pattern). A star
    expands to each picked source's schema names less its EXCLUDEs and with its RENAMEs; a named
    projection is its alias; a CTE's, a LATERAL's or a derived table's column list renames by
    position."""
    if not hasattr(scope, "expression"):
        return None
    listed = _column_list(scope, dialect)
    if isinstance(scope.expression, exp.Lateral):
        inner = _lateral_inner(scope)
        names = _output_names(inner, columns, dialect, seen) if inner is not None else None
    else:
        names = _projected_names(scope, columns, dialect, seen)
    if names is not None and listed:
        names[:len(listed)] = listed
    return names


def _projected_names(scope, columns, dialect: str | None, seen: frozenset[int]) -> list[str] | None:
    if id(scope) in seen:
        return None
    seen = seen | {id(scope)}
    if scope.union_scopes:
        return _projected_names(scope.union_scopes[0], columns, dialect, seen)
    names: list[str] = []
    for projection in scope.expression.selects:
        star = _star_of(projection)
        if star is None:
            alias = projection.alias_or_name
            if not alias:
                return None
            names.append(resolve(alias, quoted=False, dialect=dialect))
            continue
        pattern = _columns_pattern(projection)
        if pattern is None and (isinstance(projection, exp.Columns) or (
                isinstance(projection, exp.Alias) and isinstance(projection.this, exp.Columns))):
            return None  # a pattern that is no regular expression picks an unknown subset
        qualifier = (resolve_identifier(projection.args.get("table"), dialect)
                     if isinstance(projection, exp.Column) else None)
        picked = [source for source in scope.selected_sources
                  if not qualifier or resolve(source, quoted=False, dialect=dialect) == qualifier]
        for source in picked:
            node, inner = scope.selected_sources[source]
            if isinstance(inner, exp.Table) or (
                    isinstance(node, exp.Table) and _is_base_read(node, scope, dialect)):
                key = object_key(inner if isinstance(inner, exp.Table) else node)
                if columns is None or key not in columns:
                    return None
                folds = not rules_for(dialect).quoting_preserves_case
                stored = list(columns[key])
                if pattern is not None:
                    stored = [n for n in stored if _pattern_picks(pattern, [n])]
                exposed = [resolve_stored(n, dialect) if folds else n for n in stored]
            else:
                exposed = _output_names(inner, columns, dialect, seen)  # a nested list applies
                if exposed is None or pattern is not None:
                    return None  # a pattern over folded names picks an unknown subset
            dropped = {resolve_identifier(item.this if isinstance(item, exp.Column) else item,
                                          dialect)
                       for item in star.args.get("except_") or ()
                       if source in _item_sources(item, picked, dialect)}
            renamed = {resolve(item.this.name, quoted=False, dialect=dialect):
                       resolve(item.alias, quoted=False, dialect=dialect)
                       for item in star.args.get("rename") or ()
                       if source in _item_sources(item.this, picked, dialect)}
            names.extend(renamed.get(n, n) for n in exposed if n not in dropped)
    return names


def _bound_outward_bare(scope, column: exp.Column, columns, dialect: str | None):
    """Where the engine binds a bare name no source of its own scope exposes: the nearest
    enclosing scope with a source that has it (`SELECT (SELECT secret FROM (SELECT id FROM
    claim) d) FROM other d` reads other's secret; the review gate on #167)."""
    ancestor = scope.parent
    while ancestor is not None:
        local, by_alias = _scope_names(ancestor, dialect)
        fallback = None
        for key in local.values():
            if _schema_lists(columns, key, column, dialect):
                return ("table", key)
            # As under a star: a table the schema lists in full and lacking the name does not
            # have it; one listed from the grant alone, or not listed, may, and falls closed.
            if columns is None or key not in columns or key not in getattr(columns, "complete", ()):
                fallback = fallback or key
        for source in by_alias.values():
            found = _exposed(source, column, columns, dialect)
            if found is not None:
                return found
        if fallback is not None:
            return ("first", fallback)
        ancestor = ancestor.parent
    return None


def _bound_outward(scope, qualifier: str, column: exp.Column, columns, dialect: str | None):
    """Where the engine binds a qualifier its own scope names but whose source exposes nothing
    under the column's name: the nearest enclosing scope naming it (M182). sqlglot lists such a
    column in no enclosing scope, since the qualifier IS a source of its own, so the walk up is
    made here: `SELECT (SELECT d.secret FROM (SELECT id FROM claim) d) FROM other d` reads
    other's secret on DuckDB (measured), and read as unbound it reached no check."""
    ancestor = scope.parent
    while ancestor is not None:
        local, by_alias = _scope_names(ancestor, dialect)
        if qualifier in local:
            return ("table", local[qualifier])
        if qualifier in by_alias:
            found = _exposed(by_alias[qualifier], column, columns, dialect)
            if found is not None:
                return found
        ancestor = ancestor.parent
    return None


def column_tables(ast: exp.Expression, dialect: str | None = None,
                  columns: dict[str, set[str]] | None = None,
                  named: set[int] | None = None) -> dict[int, str] | None:
    """`id(column node)` -> the base table its qualifier names **in that column's own scope**,
    or **None** when the scopes could not be resolved at all. `named`, when given, collects the
    ids of columns that resolve to a named projection of a derived source or a CTE's or
    LATERAL's column-list name, whose own nodes are checked in that scope: `check_access` lets
    those by and holds every other unbound bare name to the grant's columns and the query's own
    aliases (the review gate on #167: a CTE's list names made known statement-wide let `WITH
    w(secret) AS (SELECT 1) SELECT secret FROM claim` read a hidden column).

    None and `{}` are different answers and the difference is the finding. An empty map used to
    mean both "resolved, and no column resolves to a base table" and "the resolver failed", so
    a failure was read as the former: `cls` concluded a denied column belonged to no table and
    let it through. That is the same collapse as M2's outage-versus-empty-policy and M34's
    missing-versus-unreadable baseline -- an absence and a failure wearing one value.

    A missing entry within a returned map means "this qualifier names a CTE or a derived table
    whose own projection names the column", or a column no scope settled.

    `columns` is the schema the exposure walk reads, table key to column names: a caller passes
    what it knows (the grant's and the policy's columns together, in `decide`); a table the
    schema does not list owns a name only where no listed table under the same star has it. A
    qualifier naming a derived table or CTE is bound
    through what that source EXPOSES under the column's name (M181): the first star reaching a
    base table that has the column binds it to that table, as `SELECT d.secret FROM (SELECT *
    FROM claim) d` reads claim's secret, which a qualifier answered "derived, no table" let
    through hidden, denied or masked; a source exposing nothing under the name leaves the column
    for the next scope out, as the engine binds `SELECT (SELECT d.secret FROM (SELECT id FROM
    claim) d) FROM other d` to other (M182), where an unbound column was read by no check.

    M31 made `base_tables` scope-aware and stopped there. Both consumers went on building ONE
    alias->table dictionary across every scope, so in

        SELECT q.secret FROM inner_t q UNION ALL SELECT q.secret FROM outer_t q

    the second `q` overwrote the first and a denied column was resolved against the permitted
    table. The resolver was right; the map that consumed it was still flat.
    """
    root = _bound_root(ast, dialect)
    if root is None or _unscoped_ctes(ast, root):
        return None
    if _scope_missed_a_read(ast, _walked_reads(ast, root, dialect), dialect):
        return None  # the coverage `base_tables` falls back on is not trusted for columns either

    out: dict[int, str] = {}
    settled: set[int] = set()  # columns a scope has answered for, with a base table or without
    for scope in root.traverse():
        # The base tables the scope reads, and every other source it selects from (a CTE's or a
        # derived table's), each by its alias's resolved spelling: a qualifier naming one of the
        # latter is answered by what that source exposes under the column's name (`_exposure`).
        local, by_alias = _scope_names(scope, dialect)
        # sqlglot lists no unqualified HAVING or QUALIFY column in a scope (it reads them all as
        # the query's own names), no ORDER BY column matching one, and of a window only its own
        # ORDER BY columns, where DuckDB reads an output alias for a bare ORDER BY item, or a
        # bare HAVING name outside every call, alone (measured: `HAVING secret > 0` beside `id
        # AS secret` filters by the alias, `ORDER BY secret` sorts by it) and the source column
        # everywhere else: inside any call, aggregate, FILTER or window, or an ORDER BY
        # expression (`HAVING max(secret) = 'x'`, `ORDER BY lower(secret)`, `ORDER BY
        # first_value(secret) OVER (...)`, `HAVING count(*) FILTER (WHERE secret LIKE 'Z%') > 0`,
        # `HAVING arbitrary(secret) ...`, `QUALIFY first_value(secret) OVER (...) ...` all read
        # claim's secret, M189 and the ui lane's replay at cb90806; `HAVING sum(netpaid) > ...`
        # over a CTE exposing netpaid went unvisited here and fell to the flat rule: TPC-DS q23
        # and q24). Those are read with the scope's columns, fail-closed: where the engine reads
        # the alias and this reads the source, the cost is a refusal, never a reading.
        in_having: list = []
        if isinstance(scope.expression, exp.Select):
            own = {resolve(projection.alias, quoted=False, dialect=dialect)
                   for projection in scope.expression.selects
                   if isinstance(projection, exp.Alias) and projection.alias}
            listed_ids = {id(column) for column in getattr(scope, "columns", ())}

            def reads_the_alias(column: exp.Column, clause) -> bool:
                name = resolve_identifier(column.this, dialect)
                if name not in own:
                    return False
                if isinstance(clause, exp.Order):
                    # The bare alias, parentheses aside: `ORDER BY (secret) DESC` sorts by the
                    # alias as `ORDER BY secret DESC` does (Codex's thirteenth review of #167).
                    ordered = column.parent
                    while isinstance(ordered, exp.Paren):
                        ordered = ordered.parent
                    return isinstance(ordered, exp.Ordered) and ordered.parent is clause
                if isinstance(clause, exp.Having):
                    # In HAVING DuckDB reads the alias through scalar expressions, connectors,
                    # CASE and COALESCE (measured: `HAVING abs(secret) = 1`, `HAVING secret > 0
                    # OR secret = 1` beside `id AS secret` run on the alias; Codex's eleventh
                    # review of #167 and the ui lane's replay at 3f01ef8, where sqlglot's Or
                    # and Case being Funcs refused them) and the source column inside an
                    # aggregate, a FILTER, a window, or a call it does not know (`arbitrary`),
                    # fail-closed; a grouped name is the source too (`GROUP BY id, secret
                    # HAVING lower(secret) = ...`, the review gate on 4fc447f), and is checked
                    # at the GROUP BY, which lists it among the scope's columns. Oracle reads
                    # no alias in HAVING at all. The walk stops at the clause.
                    if dialect != "duckdb":
                        return False
                    node = column.parent
                    while node is not None and node is not clause:
                        if isinstance(node, (exp.AggFunc, exp.Filter, exp.Window, exp.WindowSpec,
                                             exp.Anonymous)):
                            return False
                        node = node.parent
                    return True
                return False

            for clause in (scope.expression.args.get("having"), scope.expression.args.get("order"),
                           scope.expression.args.get("qualify")):
                if clause is None:
                    continue
                in_having += [
                    column for column in clause.find_all(exp.Column)
                    if not column.table and id(column) not in listed_ids
                    and column.find_ancestor(exp.Select) is scope.expression
                    and not reads_the_alias(column, clause)]
        for column in [*getattr(scope, "columns", ()), *in_having]:
            # Bound once, in the nearest scope that names the qualifier, as every engine binds it
            # (`traverse` yields a scope after the scopes inside it). sqlglot lists a subquery's
            # external columns in the enclosing scope as well, and calls `D.secret` external when
            # the source is `d`, so `SELECT (SELECT D.secret FROM claim d) FROM other D` was bound
            # to `claim` here and then, a scope up, to `other`, whose grant of `secret` approved a
            # read of claim's, deny and mask included (Codex's review of #167).
            if id(column) in settled:
                continue
            qualifier = resolve_identifier(column.args.get("table"), dialect)
            if not qualifier:
                # An unqualified column in a scope whose only source is a derived table or CTE
                # reads it as a qualified one would, so both spellings resolve the same way:
                # `SELECT secret FROM (SELECT *, 'NAMED' AS secret FROM claim) d` reads the
                # star's column first, as `d.secret` does (the ui lane's replay of #167). With
                # a base table among the sources the old rule stands: the grant's columns and
                # the statement's own aliases, in `check_access`.
                if column.find_ancestor(exp.Select, exp.SetOperation) is not scope.expression:
                    # sqlglot lists an inner scope's external columns in the enclosing scope too
                    # (a RENAME item's column inside a derived table's star among them); a bare
                    # column is bound in the scope whose SELECT holds it, walking outward from
                    # there, not where it is relisted.
                    continue
                fallback = None
                if local:
                    # A base table listed with the name binds it, in every scope: the engine
                    # binds a source column before the query's own alias of the same name in
                    # the SELECT list, WHERE and GROUP BY, so `SELECT id AS secret, secret FROM
                    # claim` reads claim's secret, hidden or not (Codex's seventh review of
                    # #167, under the complete schema; `check_access`'s alias allowance passed
                    # it). Checked before any derived source (Codex's fifth review: `FROM wide w
                    # CROSS JOIN (SELECT *, 'SAFE' AS secret FROM claim) d` let a bare `secret`
                    # through on the inner alias), which is harmless since the engine refuses a
                    # name two sources expose as ambiguous. As under a star, a table listed in
                    # full and lacking the name does not have it; one listed from the grant
                    # alone, or not listed, may, and beside a derived source falls closed below
                    # once none exposes the name (the review gate: bound outward instead,
                    # `SELECT (SELECT secret FROM claim c CROSS JOIN (SELECT id AS x FROM wide)
                    # d LIMIT 1) FROM other o` read claim's secret through other's grant). Over
                    # base tables alone, a name the grant lacks keeps `check_access`'s reading,
                    # the grant's columns and the query's own aliases: refusing every lateral
                    # alias for a caller deciding from the grant alone is a cost taken only
                    # beside a derived source. The deciding paths that pass the policy decide
                    # under its complete schema; the enrichment examples' validation passes
                    # every column of the snapshot and no policy. The residual over base tables
                    # alone from the grant alone is recorded in the internal register.
                    listed_with = []
                    for source in scope.selected_sources:
                        if source in by_alias:
                            continue
                        key = local.get(resolve(source, quoted=False, dialect=dialect))
                        if key is None:
                            continue
                        if _schema_lists(columns, key, column, dialect):
                            listed_with.append(key)
                        elif by_alias and (columns is None or key not in columns
                                           or key not in getattr(columns, "complete", ())):
                            fallback = fallback or key
                    if len(listed_with) == 1:
                        out[id(column)] = listed_with[0]
                        settled.add(id(column))
                        continue
                    if listed_with:
                        # Two base tables listed with the name: the engine refuses the reference
                        # as ambiguous and reads neither, and a binding here would be a guess
                        # (eval's mask attribution keeps such a reference unmeasured).
                        continue
                    if not by_alias:
                        # Over base tables alone and listed with the name nowhere: the flat rule,
                        # not the outward walk, whose grant-only fallback refused `SELECT (SELECT
                        # id AS k FROM wide GROUP BY k LIMIT 1) FROM claim` from the grant alone
                        # (the review gate). A scope with no source at all binds outward below,
                        # as the engine does.
                        continue
                # A bare name in ORDER BY or HAVING that is the query's own alias never reaches
                # here: sqlglot resolves it to the alias before listing the scope's columns, as
                # the engine reads it (measured, a source column of the same name included). In
                # the projection, a WHERE or a GROUP BY the engine reads the source's column
                # whatever an alias is called (DuckDB groups `GROUP BY secret` by the source's
                # secret beside `id AS secret`), so those bind below; a GROUP BY alias the source
                # cannot be shown to lack falls closed, and the complete schema shows it. With
                # several derived sources the engine binds the bare name to the one that exposes
                # it (measured; two exposing it is its own error), read here in FROM order
                # (Codex's fourth review of #167: a second source had switched this binding off).
                found = next((hit for source in by_alias.values()
                              if (hit := _exposed(source, column, columns, dialect)) is not None),
                             None)
                if found is None and fallback is not None:
                    # Also where the name is the query's own projection alias, which the engine
                    # reads laterally in the SELECT list, WHERE and GROUP BY only once no source
                    # has it: from the grant alone the resolver cannot tell whether the table
                    # has the name, and the engine binds the source column first (Codex's sixth
                    # review of #167: `SELECT c.id AS secret, secret FROM claim c CROSS JOIN
                    # (SELECT id AS x FROM wide) d` read claim's hidden secret with claim granted
                    # without it, once the alias was exempted). So `SELECT c.id AS k, k + 1 FROM
                    # claim c CROSS JOIN (SELECT id AS x FROM other) d` is refused from the grant
                    # alone and approved under the complete schema: `build_access_policy`'s
                    # `policy_schema`, every column of the snapshot, which both shipped deciding
                    # paths pass (`visible_schema` alone marks no table complete; a `decide`
                    # with no policy reads from the grant alone, the review gate).
                    found = ("first", fallback)
                if found is None:
                    found = _bound_outward_bare(scope, column, columns, dialect)
                if found is None:
                    continue
                settled.add(id(column))
                if found is not _NAMED:
                    out[id(column)] = found[1]
                elif named is not None:
                    named.add(id(column))
                continue
            table = local.get(qualifier)
            if table is not None:
                out[id(column)] = table
                settled.add(id(column))
            elif qualifier in by_alias:
                found = _exposed(by_alias[qualifier], column, columns, dialect)
                if found is None:
                    found = _bound_outward(scope, qualifier, column, columns, dialect)
                if found is None:
                    continue  # no scope exposes it: the engine would refuse the reference
                settled.add(id(column))
                if found is not _NAMED:
                    out[id(column)] = found[1]
                elif named is not None:
                    named.add(id(column))

    # Columns qualified by the write target resolve to it. `ON CONFLICT DO UPDATE SET amount =
    # scratch.amount + 1` reads `scratch.amount` from a table in no scope, so without this the
    # map has no entry and `check_cls` falls back to the referenced set -- checking a denied
    # column against the wrong table. Only fills entries the scope walk did not settle, so a real
    # source or CTE of the same name keeps its scoped answer.
    for target in _target_reads(ast):
        key = object_key(target)
        known_as = resolve_identifier(target.args.get("alias") or target.this, dialect)
        for column in ast.find_all(exp.Column):
            if (known_as is not None and id(column) not in settled
                    and resolve_identifier(column.args.get("table"), dialect) == known_as):
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
