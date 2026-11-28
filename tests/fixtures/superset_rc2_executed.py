"""Writes `superset_rc2_executed.json`: what Superset 7.0.0rc2 would run for statements mnemiq sends it on Oracle, made
by rc2's own code, for `tests/test_superset_executed.py`.

Each statement is prepared as rc2's executor prepares one (`SQLExecutor._prepare_sql` in
`superset/sql/execution/executor.py`): read by `SQLScript(sql, "oracle")`; each table its statement reads given the
case's row rule, qualified by the session schema as `superset.utils.rls.apply_rls` qualifies it, the clause read by the
statement's own `parse_predicate`, and applied by `SQLStatement.apply_rls` with `RLSMethod.AS_SUBQUERY` (Oracle's
`get_rls_method`); the limit, when the case asks one, set by `set_limit_value` with `LimitMethod.FORCE_LIMIT` (Oracle's
`limit_method`); then each statement's `.format()`, the text a dry run answers as `executed_sql` and a run sends.

Each case also carries its `placeholder` rendering: the same, with the placeholder rule the extension's `tables_read`
puts on every table the statement reads (`PLACEHOLDER`, `reads.PLACEHOLDER_RULE`) and no limit, the text that tool
answers as `ruled`. `sizes` records how many bytes `tables_read`'s answer takes, as the tool measures it (its JSON
text), for a probe naming `n` tables of names `width` characters long, which mnemiq sizes its grant probes from.

Run it again with rc2's own interpreter after an upgrade of Superset or of its sqlglot, and read the diff:

    <Superset's venv>/bin/python tests/fixtures/superset_rc2_executed.py
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import sqlglot
from superset.sql.parse import LimitMethod, RLSMethod, SQLScript

SCHEMA = "shop"
CLAUSE = "region = 'west'"
PLACEHOLDER = "1 = 1"

# Each case: the text mnemiq sends (in Superset's spelling, as `names.for_superset` writes it), the tables a rule is
# put on (by name as rc2 reads them; "*" for every table), and the limit asked, if any.
CASES = {
    "one table under its rule": ('SELECT o.id, o.status FROM orders "O" WHERE o.status = \'open\'', ["*"], None),
    "one table with no rule": ('SELECT o.id, o.status FROM orders "O" WHERE o.status = \'open\'', [], None),
    "a join, one table ruled": (
        'SELECT o.id, c.name FROM orders "O" JOIN customers "C" ON c.id = o.customer_id', ["orders"], None),
    "a CTE, a subquery and a schema": (
        'WITH recent AS (SELECT o.id FROM shop.orders "O" WHERE o.placed > TO_DATE(\'2024-01-01\', \'YYYY-MM-DD\')) '
        'SELECT r.id FROM recent r WHERE EXISTS (SELECT 1 FROM lines "L" WHERE l.order_id = r.id)', ["*"], None),
    "a column qualified by its table's name": ('SELECT orders.id FROM orders "ORDERS" WHERE orders.id > 0', ["*"], None),
    "a sampled table": ('SELECT orders.id FROM orders SAMPLE (10) "ORDERS" WHERE orders.id > 0', ["*"], None),
    "a table read twice": (
        'SELECT a.id FROM orders "A" JOIN orders "B" ON b.parent_id = a.id', ["*"], None),
    "a quoted table": ('SELECT i.qty FROM "OrderItems" "I"', ["*"], None),
    "Superset's limit": ('SELECT o.id FROM orders "O" ORDER BY o.id FETCH FIRST 1000 ROWS ONLY', ["*"], 7),
    "Superset's limit after an offset": (
        'SELECT o.id FROM orders "O" ORDER BY o.id OFFSET 5 ROWS FETCH NEXT 10 ROWS ONLY', [], 7),
    "a subquery's own limit": (
        'SELECT x.id FROM (SELECT o.id FROM orders "O" ORDER BY o.id FETCH FIRST 5 ROWS ONLY) x '
        'FETCH FIRST 10 ROWS ONLY', ["*"], 7),
    "a statement's own limit": ('SELECT o.id FROM orders "O" ORDER BY o.id FETCH FIRST 10 ROWS ONLY', ["*"], None),
    "the proof's probe": (
        'SELECT * FROM (SELECT o.region, COUNT(*) AS n FROM orders "O" GROUP BY o.region) mnemiq_probe WHERE 1 = 0',
        ["*"], None),
    # Shapes rc2 may render otherwise than they were sent: each is refused, or shown to read alike.
    "CROSS APPLY": ('SELECT o.id, x.n FROM orders "O" CROSS APPLY '
                    '(SELECT COUNT(*) AS n FROM lines "L" WHERE l.order_id = o.id) x', [], None),
    "OUTER APPLY": ('SELECT o.id, x.n FROM orders "O" OUTER APPLY '
                    '(SELECT COUNT(*) AS n FROM lines "L" WHERE l.order_id = o.id) x', [], None),
    "TO_DATE with its NLS argument": (
        "SELECT o.id FROM orders \"O\" WHERE o.placed > "
        "TO_DATE('01-JAN-2024', 'DD-MON-YYYY', 'NLS_DATE_LANGUAGE = AMERICAN')", [], None),
    "TO_TIMESTAMP with its NLS argument": (
        "SELECT o.id FROM orders \"O\" WHERE o.placed > "
        "TO_TIMESTAMP('01-JAN-2024', 'DD-MON-YYYY', 'NLS_DATE_LANGUAGE = AMERICAN')", [], None),
    "GROUP BY a column named sample": ('SELECT o.region FROM orders "O" GROUP BY o.region, sample', [], None),
    "IS NOT NULL compared": ('SELECT o.id FROM orders "O" WHERE o.shipped IS NOT NULL = 1', [], None),
    "IS NOT NULL compared, as mnemiq prints it": (
        'SELECT o.id FROM orders "O" WHERE (NOT o.shipped IS NULL) = 1', [], None),
    "a subquery's outer-join mark": (
        'SELECT o.id FROM orders "O" WHERE o.id = (SELECT MAX(l.order_id) FROM lines "L")(+)', ["*"], None),
    "a qualifier two tables share": ("SELECT ORDERS.ID FROM shop.orders, orders", ["*"], None),
    "a qualifier two tables share, no rule": ("SELECT ORDERS.ID FROM shop.orders, orders", [], None),
    "a qualifier that would name an outer table": (
        'SELECT 1 AS one FROM orders "T" WHERE EXISTS (SELECT 1 FROM t WHERE t.x = 1)', ["*"], None),
    "a table read in a query and its subquery": (
        'SELECT o.id FROM orders "O" WHERE o.id IN (SELECT p.parent_id FROM orders "P")', [], None),
}


def executed(sent: str, ruled: list[str], limit: int | None, clause: str = CLAUSE) -> str:
    script = SQLScript(sent, "oracle")
    [statement] = script.statements
    predicates = {table.qualify(catalog=None, schema=SCHEMA): [statement.parse_predicate(clause)]
                  for table in statement.tables if "*" in ruled or table.table in ruled}
    statement.apply_rls(None, SCHEMA, predicates, RLSMethod.AS_SUBQUERY)
    if limit is not None and not script.has_mutation():
        statement.set_limit_value(limit, LimitMethod.FORCE_LIMIT)
    return statement.format()


def probe_names(n: int, width: int) -> list[str]:
    """The `n` stored names of `width` characters a size is measured for: `T0000`, `T0001`, ... padded with `X`."""
    return [(f"T{i:04d}" + "X" * width)[:width] for i in range(n)]


def sizes() -> list[dict]:
    """The bytes of `tables_read`'s answer, `{"tables": [...], "ruled": ...}` as the tool measures it (its JSON text,
    `reads._size`), for a grant probe naming `n` tables (`probe_names`), as mnemiq sends one on Oracle:
    `SELECT 1 FROM name "NAME", ... WHERE 1 = 0`, each name in Superset's spelling aliased by its stored one."""
    found = []
    for n, width in ((1, 5), (1, 30), (100, 8), (300, 8), (100, 30), (100, 128), (50, 128)):
        probe = "SELECT 1 FROM " + ", ".join(f'{name.lower()} "{name}"' for name in probe_names(n, width)) \
            + " WHERE 1 = 0"
        [statement] = SQLScript(probe, "oracle").statements
        tables = sorted(statement.tables, key=lambda t: t.table)
        answer = {"tables": [{"catalog": None, "schema": SCHEMA, "table": t.table} for t in tables],
                  "ruled": executed(probe, ["*"], None, PLACEHOLDER)}
        found.append({"n": n, "width": width, "bytes": len(json.dumps(answer).encode())})
    return found


def main() -> None:
    cases = {}
    for name, (sent, ruled, limit) in CASES.items():
        script = SQLScript(sent, "oracle")
        [statement] = script.statements
        cases[name] = {"sent": sent, "ruled": sorted(t.table for t in statement.tables if "*" in ruled
                                                     or t.table in ruled),
                       "limit": limit, "executed": executed(sent, ruled, limit),
                       "placeholder": executed(sent, ["*"], None, PLACEHOLDER)}
    out = {"superset": version("apache-superset"), "sqlglot": sqlglot.__version__, "schema": SCHEMA,
           "clause": CLAUSE, "placeholder": PLACEHOLDER, "cases": cases, "sizes": sizes()}
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
