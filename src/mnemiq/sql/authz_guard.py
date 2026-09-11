from __future__ import annotations

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect

from mnemiq.sql.functions import FunctionInventory, UnreadableCalls, called_names
from mnemiq.sql.qualify import object_key
from mnemiq.sql.scope import base_tables, column_tables
from mnemiq.sql.verdict import Refusal, RefusalCode


def check_access(ast: exp.Expression, visible: dict[str, set[str]],
                 dialect: str | None = None) -> Refusal | None:
    """Re-check every table and column the query touches against what the identity may see.

    `dialect` is the ENGINE that will run this, not the one it was parsed as. Identifier folding differs between them -- Oracle folds unquoted names up, the rest fold them down, and quoting is significant in Postgres and Oracle but not DuckDB -- so a resolver keyed on the wrong one answers for the wrong engine. Threading it to SOME call sites and not others is worse than either: `check_access` said a name was a CTE while the audit list said it was a base read, in the same statement.

    Retrieval scoping (the semantic store) means the model was never *shown* a forbidden
    table. It can still *name* one -- `users`, `employees`, `salaries` are in every schema it
    was trained on. This is the lock that makes naming it useless.
    """
    # tables, and the aliases that stand for them. `base_tables` -- not `find_all` minus a flat
    # set of CTE names -- because a reference inside a CTE body naming that same CTE reads the
    # base table, and skipping it let an ungranted table through (M31).
    alias_to_table: dict[str, str] = {}
    resolved = column_tables(ast, dialect)
    for table in base_tables(ast, dialect):
        name = object_key(table)
        if name not in visible:
            return Refusal(
                code=RefusalCode.UNAUTHORIZED_TABLE,
                message=f"You may not query {name!r}. Answer using only the tables provided.",
                subject=name,
            )
        alias_to_table[table.alias_or_name] = name

    if not alias_to_table:
        return None  # a query over CTEs alone; their sources were checked above

    referenced = set(alias_to_table.values())
    known_columns = {c for t in referenced for c in visible[t]}

    # A SELECT alias is a name the query invents for an expression, and GROUP BY / ORDER BY /
    # HAVING may refer to it: `SELECT year(d) AS y, count(*) FROM t GROUP BY y` is ordinary,
    # correct SQL -- and without this the engine could not answer "count X by year" at all.
    # It is safe: an alias can only be defined from columns that were themselves checked.
    known_columns |= {alias.alias for alias in ast.find_all(exp.Alias) if alias.alias}

    for column in ast.find_all(exp.Column):
        qualifier = column.table
        if qualifier:
            # Resolved per scope, not from one flat map: an alias can name two different
            # tables in one statement, and keeping only the last let a column be checked
            # against the wrong one (Codex review, 2026-08-12).
            if resolved is None:
                # Unreadable scopes: fall back to the rule for an unqualified column rather
                # than to the flat map that was the defect. Same guarantee, no worse.
                if column.name not in known_columns:
                    return Refusal(
                        code=RefusalCode.UNKNOWN_COLUMN,
                        message=(
                            f"No table in this query has a column {column.name!r}. "
                            "Use only the columns listed on the schema cards."
                        ),
                        subject=column.name,
                    )
                continue
            table = resolved.get(id(column))
            if table is None:
                continue  # qualifier belongs to a CTE or a subquery alias
            if column.name not in visible[table]:
                return Refusal(
                    code=RefusalCode.UNKNOWN_COLUMN,
                    message=f"{table!r} has no column {column.name!r}. Use only listed columns.",
                    subject=column.name,
                )
        elif column.name not in known_columns:
            return Refusal(
                code=RefusalCode.UNKNOWN_COLUMN,
                message=(
                    f"No table in this query has a column {column.name!r}. "
                    "Use only the columns listed on the schema cards."
                ),
                subject=column.name,
            )

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
    OWN name must be spelled, so `called_names` finds it however sqlglot rewrote the node. A
    call reachable under a BUILTIN's name is not derivable at all, and cannot happen unless the
    source defines something a builtin is also called -- so that case condemns every call here
    rather than pretending to pick one. Without an inventory this is the allowlist alone, which
    is where it started.
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
    if not inventory.names or next(ast.find_all(exp.Func), None) is None:
        # Nothing defined, or nothing called. The second half is what keeps a source with one
        # helper from refusing `SELECT id FROM claim`.
        return None

    if inventory.may_shadow_a_builtin:
        return _shadowed(inventory)

    try:
        called = called_names(ast, *dialects)
    except UnreadableCalls:
        # Cannot enumerate, so cannot clear. The same answer as shadowing, and for the same
        # reason: an empty set read as "no calls" would clear all of them.
        return _shadowed(inventory)

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


def _shadowed(inventory: FunctionInventory) -> Refusal:
    """No call on this source can be read off the text, so none of them can be decided.

    Two ways to arrive, kept apart in the message because they need different fixes. The source
    defines a name it also calls a builtin, which the deployer resolves by renaming it. Or the
    source never said what its builtins are, which the ADAPTER resolves by implementing
    `builtin_functions` -- a missing method should not read in a trace like a hostile database.
    """
    shadowed = sorted(inventory.names & inventory.builtins) if inventory.builtins else []
    if shadowed:
        detail = (
            f"This source defines {shadowed[0]!r} under a name it also lists as a builtin"
        )
    else:
        detail = (
            "This source defines functions of its own and did not report which names are "
            "builtins"
        )
    return Refusal(
        code=RefusalCode.SHADOWED_FUNCTION,
        message=(
            f"{detail}, so this engine cannot confirm what any call in this query executes. "
            "No query using a function can be decided against this source."
        ),
        subject=shadowed[0] if shadowed else None,
    )
