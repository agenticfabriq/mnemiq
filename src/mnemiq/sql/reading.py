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
column. A struct literal's key (`{X: tags}`) is a name to DuckDB and stays one. A key spelled
exactly as an enclosing lambda's parameter is that parameter, which DuckDB binds before any
column (measured: `list_transform([7], x -> MAP {x: 1}[7])` over a table with a column `x` reads
the parameter), so it becomes a Var, bare in the render and no Column to any guard; a parameter
spelled in another case stays a Column, and is refused as an unknown one, since DuckDB binds
it to a column of the folded name in a SELECT list where one is in scope.

The arrow parts the two readings the same way: sqlglot reads `a -> b` as a lambda inside any
call, DuckDB only inside the calls it hands a lambda to (`lambda_takers`) and as the JSON
extract anywhere else, so the arrow's left side is a column to the engine and a lambda's
parameter to every reader of the tree: with `secret` hidden, `SELECT concat(secret -> 1) FROM
claim` passed the grant check, which walks Columns, and DuckDB ran `json_extract(secret, 1)`,
raising with the hidden value quoted (M213, the review gate on round 54 of #160). Such a
Lambda becomes the `exp.JSONExtract` it is, its left side the Column, or the row of Columns,
DuckDB reads, so the grant check sees the read and the typing rules read the extract as they
read `json_extract(value, path)`.

One divergence no tree can undo: sqlglot reads a dotted MAP key, `MAP {x.region: 1}`, as its
last part alone, `region`, before any node exists, so the re-reading made it the base column
and `list_transform([{region: 'south'}], x -> MAP {x.region: 1}[region] IS NULL) = [false]`,
which removes every row (the key is the element's field), was installed as `MAP {region: 1}`
and kept every row, the policy reversed (Codex's forty-fourth review of #160). Such a key is
found in the text's tokens (`dotted_map_keys`) and the statement refused, with the spellings
sqlglot keeps in the advice: `MAP {(x.region): 1}`, or `MAP([x.region], [1])`.

sqlglot builds `exp.ToMap` over a Struct from DuckDB's `MAP {...}` syntax alone, so for a tree
parsed in any other dialect this is a no-op; the name says whose reading it restores.
"""

from __future__ import annotations

import logging
import re
import threading

import sqlglot
from sqlglot import exp
from sqlglot.tokens import TokenType

logger = logging.getLogger(__name__)

# The calls DuckDB hands a lambda to. Anywhere else `a -> b` is DuckDB's JSON extract, which
# sqlglot reads as a lambda inside a call's arguments or beside an operator (`list_contains(
# [region], attrs -> '$.a')`, `region = attrs -> '$.a'`: Codex's nineteenth review of #160), so a
# reader trusting the parse sees a statement the engine does not run, and the arrow's left side,
# a column to DuckDB, is a lambda's parameter to sqlglot: `concat(secret -> map_keys(MAP {secret:
# 1})[1])` read a hidden column through a key this module had made a Var (the review gate on
# round 54 of #160; DuckDB raises quoting the value).
LAMBDA_TAKERS_KNOWN = frozenset({"list_transform", "array_transform", "apply", "list_apply",
                                 "array_apply", "list_filter", "array_filter", "filter",
                                 "list_reduce", "array_reduce", "reduce"})
LAMBDA_TAKER_NODES = (exp.Transform, exp.ArrayFilter, exp.Reduce)
_TAKERS: frozenset[str] | None = None
_TAKERS_LOCK = threading.Lock()


def lambda_takers() -> frozenset[str]:
    """The calls DuckDB hands a lambda to, asked of the local reader's own catalogue
    (`duckdb_functions()`, a LAMBDA among the parameter types: eleven names on 1.5.4), the
    list above where the catalogue cannot be asked. A typed list missed `array_apply` and
    listed `list_sort`, `list_any_value` and `list_grade_up`, which take none (the review gate
    on the twentieth and twenty-first rounds of #160)."""
    global _TAKERS
    with _TAKERS_LOCK:
        if _TAKERS is None:
            try:
                from mnemiq.duckdb_extensions import connect_duckdb

                rows = connect_duckdb(local_only=True).execute(
                    "SELECT DISTINCT lower(function_name) FROM duckdb_functions() "
                    "WHERE list_contains(parameter_types, 'LAMBDA')").fetchall()
                _TAKERS = frozenset(name for (name,) in rows) or LAMBDA_TAKERS_KNOWN
            except Exception:  # noqa: BLE001 -- the catalogue could not be asked: the typed list
                logger.warning("DuckDB's catalogue could not be asked which calls take a lambda; "
                               "the typed list stands in (M159)")
                _TAKERS = LAMBDA_TAKERS_KNOWN
        return _TAKERS


def arrow_read_as_a_lambda(node: exp.Expression) -> bool:
    """Whether sqlglot's Lambda is DuckDB's JSON arrow: a Lambda under anything but a call DuckDB
    hands a lambda to, or in any slot of such a call but the lambda's own, the second argument
    (`list_transform(secret -> ['$'], x -> x)` reads `secret` through the arrow in the list's
    slot, where the parent alone said lambda: the review gate on round 55 of #160)."""
    if not isinstance(node, exp.Lambda):
        return False
    parent = node.parent
    if isinstance(parent, LAMBDA_TAKER_NODES):
        return node is not parent.args.get("expression") and node is not parent.args.get("merge")
    if isinstance(parent, exp.Anonymous) and parent.name.lower() in lambda_takers():
        return len(parent.expressions) < 2 or parent.expressions[1] is not node
    return True


_OPENERS = (TokenType.L_PAREN, TokenType.L_BRACKET, TokenType.L_BRACE)
_CLOSERS = (TokenType.R_PAREN, TokenType.R_BRACKET, TokenType.R_BRACE)


def dotted_map_keys(sql: str, dialect: str | None) -> list[str]:
    """The dotted keys of the MAP literals in `sql` that sqlglot's parse loses, read off the text's
    tokens: after `MAP {` or a comma at the map's own depth, up to the colon, a key that is a
    chain of identifiers joined by dots and nothing else (`x.region`, `a.b.c`, `"x".region`).
    Any other key sqlglot keeps whole, `x[1].f`, `x.f(1)`, `claim.region::VARCHAR`, `(x.region)`,
    a CASE over a dotted name (the review gate on round 57 of #160, where a dot anywhere at the
    map's depth was flagged and each of these was refused)."""
    try:
        tokens = sqlglot.Dialect.get_or_raise(dialect or "duckdb").tokenize(sql)
    except Exception:  # noqa: BLE001 -- text the tokenizer refuses: the parser will say so
        return []
    found: list[str] = []
    stack: list[list] = []  # one frame per open bracket: [is a MAP brace, the key's tokens or None]
    for position, token in enumerate(tokens):
        kind = token.token_type
        if kind in _OPENERS:
            if stack and stack[-1][0] and stack[-1][1] is not None:
                stack[-1][1].append(token)  # a bracket or a parenthesis in the key: no chain of names
            is_map = kind == TokenType.L_BRACE and position > 0 and tokens[position - 1].token_type == TokenType.MAP
            stack.append([is_map, [] if is_map else None])
            continue
        if kind in _CLOSERS:
            if stack:
                stack.pop()
            continue
        if not stack or not stack[-1][0]:
            continue
        frame = stack[-1]
        if kind == TokenType.COLON and frame[1] is not None:
            if _dotted_chain(frame[1]):
                found.append(".".join(k.text for k in frame[1][::2]))
            frame[1] = None
        elif kind == TokenType.COMMA:
            frame[1] = []
        elif frame[1] is not None:
            frame[1].append(token)
    return found


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _dotted_chain(key: list) -> bool:
    """Whether a key's tokens are identifiers joined by dots and nothing else, the one shape sqlglot
    reduces to its last identifier: `x.region`, `a.b.c`, a quoted name among them."""
    if len(key) < 3 or len(key) % 2 == 0:
        return False
    names, dots = key[::2], key[1::2]
    return all(d.token_type == TokenType.DOT for d in dots) and all(
        n.token_type != TokenType.DOT and (n.token_type == TokenType.IDENTIFIER or _NAME.match(n.text)) for n in names)


def map_struct(node: exp.ToMap) -> exp.Struct | None:
    """The struct a MAP literal is built from, through the bracket or the parentheses sqlglot
    wraps it in (`MAP {...}['b']`, `(MAP {...})`), or None where the map is spelled otherwise."""
    struct = node.this
    while isinstance(struct, (exp.Bracket, exp.Paren)):
        struct = struct.this
    return struct if isinstance(struct, exp.Struct) else None


def is_lambda_parameter(node: exp.Expression) -> bool:
    """Whether `node` references an enclosing lambda's parameter in its exact spelling: an
    Identifier (sqlglot's reading of a bare parameter), an unqualified Column, or the Var a MAP
    literal's key becomes here. A lambda sqlglot read out of DuckDB's JSON arrow declares
    nothing (`arrow_read_as_a_lambda`): its left side is a column to the engine."""
    if not isinstance(node, (exp.Identifier, exp.Column, exp.Var)) or node.args.get("table"):
        return False
    lam = node.find_ancestor(exp.Lambda)
    while lam is not None:
        if not arrow_read_as_a_lambda(lam) and node.name in {p.name for p in lam.expressions}:
            return True
        lam = lam.find_ancestor(exp.Lambda)
    return False


def as_duckdb_reads(ast: exp.Expression) -> exp.Expression:
    """The tree re-read as DuckDB reads it, in place: a Lambda outside a lambda taker the JSON
    extract over Columns, innermost first, and then every MAP literal's bare key the Column
    it is, or the Var of an enclosing lambda's parameter spelled exactly."""
    for lam in reversed(list(ast.find_all(exp.Lambda))):
        if not arrow_read_as_a_lambda(lam):
            continue
        columns = [exp.Column(this=p.copy()) if isinstance(p, exp.Identifier) else p.copy() for p in lam.expressions]
        left = columns[0] if len(columns) == 1 else exp.Tuple(expressions=columns)
        lam.replace(exp.JSONExtract(this=left, expression=lam.this.copy()))
    for node in list(ast.find_all(exp.ToMap)):
        struct = map_struct(node)
        for item in struct.expressions if struct is not None else ():
            if isinstance(item, exp.PropertyEQ) and isinstance(item.this, exp.Identifier):
                key = item.this
                item.set("this", exp.Var(this=key.name) if is_lambda_parameter(key) else exp.Column(this=key.copy()))
    return ast


def rendered(sql: str, read: str, write: str) -> str:
    """`sql`'s round trip into the target dialect, sqlglot's `transpile` with the re-reading
    between the parse and the render, so the statement run is the statement read."""
    return as_duckdb_reads(sqlglot.parse_one(sql, read=read)).sql(dialect=write)
