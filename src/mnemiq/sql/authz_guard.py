from __future__ import annotations

from sqlglot import exp

from mnemiq.sql.verdict import Refusal, RefusalCode


def _cte_names(ast: exp.Expression) -> set[str]:
    # A CTE alias is reported as a Table by find_all(). It is a name the query defines for
    # itself, not an object in the source -- authorizing it would be a category error, and
    # rejecting it would break every valid WITH clause.
    return {cte.alias_or_name for cte in ast.find_all(exp.CTE)}


def check_access(ast: exp.Expression, visible: dict[str, set[str]]) -> Refusal | None:
    """Re-check every table and column the query touches against what the identity may see.

    Retrieval scoping (the semantic store) means the model was never *shown* a forbidden
    table. It can still *name* one -- `users`, `employees`, `salaries` are in every schema it
    was trained on. This is the lock that makes naming it useless.
    """
    local = _cte_names(ast)

    # tables, and the aliases that stand for them
    alias_to_table: dict[str, str] = {}
    for table in ast.find_all(exp.Table):
        name = table.name
        if name in local:
            continue
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
            if qualifier in local:
                continue  # a column of a CTE: its source columns were checked at the source
            table = alias_to_table.get(qualifier)
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
