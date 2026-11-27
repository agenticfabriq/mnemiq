"""The tables Superset reads in a statement mnemiq sends it, found as Superset finds them, so `names.for_superset`
sends a statement only when Superset reads the same tables in it as mnemiq does.

`names` holds each reference to this copy: each table mnemiq reads must be one this reading finds at its own place in
the statement, by the name Superset gives it. On Oracle the adapter also asks Superset's own reading, the extension's
`tables_read`, and sends a statement, a virtual dataset's SQL the build makes among them, only when every distinct
table mnemiq reads in it is one Superset's reading names (`executed.check_tables_read`): that confirms each table,
not each reference.

A copy of Superset 7.0.0rc2's own table extraction (`superset/sql/parse.py`), the one its access check
(`extract_tables_from_statement`, through `_find_table_sources`) and its row rules (`SQLStatement.apply_rls`) share:
each `exp.Table` among the sources of each scope sqlglot's `traverse_scope` gives (`Scope.sources`), except one rc2's
`is_cte` takes for a CTE of that scope. It reads mnemiq's own parse of the text sent, with mnemiq's sqlglot.

Checked against rc2 7.0.0rc2, run with its own sqlglot (30.17.0), from mnemiq's sqlglot 30.29.0: over the one-statement
queries among the SQL literals of mnemiq's source and tests, as `for_superset` sends them, each table mnemiq reads in a
statement is among the tables rc2's own extraction reads and is one rc2's `apply_rls` applies a row rule to. Where the
two versions read a statement apart, the reading here is not rc2's. They are known to on a partition or subpartition
clause, on a flashback clause (AS OF), on the sources a scope lists for a pivoted or unpivoted reference, and on
Oracle's q-quoted literals: `names` refuses a statement with any of them (`_CONFIRMED_ARGS`, `_refusal`). A change of
either version is to be checked again.
Not modelled: rc2's access check reads a SHOW or DESCRIBE statement, and one sqlglot keeps as a Command, otherwise;
mnemiq sends none.
"""

from __future__ import annotations

from sqlglot import exp
from sqlglot.optimizer.scope import Scope, ScopeType, traverse_scope


def superset_reads(statement: exp.Expression) -> list[exp.Table]:
    """Each table node Superset reads in `statement`, a parse of the text sent: every `exp.Table` source of every scope
    `traverse_scope` gives but a CTE's (`_is_cte`), as rc2's `_find_table_sources` lists them, a node once per scope
    that has it among its sources. Raises sqlglot's `OptimizeError` where `traverse_scope` does."""
    return [source for scope in traverse_scope(statement) for source in scope.sources.values()
            if isinstance(source, exp.Table) and not _is_cte(source, scope)]


def _is_cte(source: exp.Table, scope: Scope) -> bool:
    """rc2's `is_cte`: whether `scope` takes the table `source` for one of its CTEs."""
    if source.db or source.catalog:
        return False
    resolved = scope.cte_sources.get(source.name)
    return isinstance(resolved, Scope) and resolved.scope_type == ScopeType.CTE
