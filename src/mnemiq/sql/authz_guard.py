from __future__ import annotations

from sqlglot import exp

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
    resolved = column_tables(ast)
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
