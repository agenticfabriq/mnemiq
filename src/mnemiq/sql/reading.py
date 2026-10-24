"""Where sqlglot's tree and DuckDB's reading part, the tree is re-read as DuckDB reads it, once,
before anything reads it: the guards, the typing rules, and the rendering of what runs.

A MAP literal's bare key is an expression to DuckDB, a column or a lambda's parameter, and
sqlglot parses it as a struct's key, an Identifier, which it writes back as a string. So
`SELECT map_keys(MAP {secret: 1})[1] FROM claim` read a hidden column past the grant check,
which walks Columns, and the plan's round trip to the target wrote `MAP {'secret': 1}`, a
string, so the statement run was not the statement checked (nothing leaked by that accident,
and `MAP {id: 1}[id]` answered for the string where the author asked for the column); a row
filter `list_transform(tags, x -> MAP {X: x}[x] IS NULL) = [true]` removed every row and
installed as `MAP {'X': x}[x] IS NULL` kept every row, the policy reversed (Codex's forty-third
review of #160, measured on DuckDB 1.5.4). Read as the Column it is, the key renders as
written, quoted or not, the grant check sees the read, the row filter's case guard reads a
lambda's parameter in another case as the reference it is, and the typing rules read the
column. A struct literal's key (`{X: tags}`) is a name to DuckDB and stays one.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp


def map_struct(node: exp.ToMap) -> exp.Struct | None:
    """The struct a MAP literal is built from, through the bracket or the parentheses sqlglot
    wraps it in (`MAP {...}['b']`, `(MAP {...})`), or None where the map is spelled otherwise."""
    struct = node.this
    while isinstance(struct, (exp.Bracket, exp.Paren)):
        struct = struct.this
    return struct if isinstance(struct, exp.Struct) else None


def as_duckdb_reads(ast: exp.Expression) -> exp.Expression:
    """The tree with every MAP literal's bare key read as the Column it is, in place."""
    for node in list(ast.find_all(exp.ToMap)):
        struct = map_struct(node)
        for item in struct.expressions if struct is not None else ():
            if isinstance(item, exp.PropertyEQ) and isinstance(item.this, exp.Identifier):
                item.set("this", exp.Column(this=item.this.copy()))
    return ast


def rendered(sql: str, read: str, write: str) -> str:
    """`sql`'s round trip into the target dialect, sqlglot's `transpile` with the re-reading
    between the parse and the render, so the statement run is the statement read."""
    return as_duckdb_reads(sqlglot.parse_one(sql, read=read)).sql(dialect=write)
