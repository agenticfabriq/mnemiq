from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp


@dataclass
class Violation:
    code: str
    message: str  # written for the corrector to act on, not for a log


def _has_not_null_guard(root: exp.Expression, col: str) -> bool:
    for is_node in root.find_all(exp.Is):
        left = is_node.this
        if (
            isinstance(is_node.expression, exp.Null)
            and isinstance(left, exp.Column)
            and left.name == col
            and isinstance(is_node.parent, exp.Not)
        ):
            return True
    return False


def _wrote_limit(root: exp.Expression) -> bool:
    """Did the statement bound itself? Not "is there a LIMIT": the shape check appends the row cap to
    every unbounded statement, and this rule refused 13 of a pilot customer's 18 gold statements and 40 of
    BEAVER's scored 200 for a LIMIT they never wrote (M161). The shape check says which it was; a tree
    that never passed through it is read as written."""
    return root.meta.get("wrote_limit", root.args.get("limit") is not None)


def _null_before_ordered_limit(root: exp.Expression) -> Violation | None:
    """A top-N whose first sort key puts NULLs first returns NULLs instead of values.

    Where NULLs go is the ENGINE's, and the old rule assumed one answer -- ascending first -- that
    neither engine production runs gives: DuckDB 1.5.4 sorts them last both ways, and Oracle 23 last
    ascending and FIRST descending. SQLite does put them first ascending (measured, M161;
    `test_lint_null_order.py` pins all three; Postgres, first descending, is not pinned). sqlglot
    records each key's order as `nulls_first`, from a written NULLS clause or the parse dialect's
    default, and a transpile writes the clause out where the target's default differs, so it is the
    order the statement runs with. A statement that writes a null order chose it, and is left alone.
    """
    if not isinstance(root, exp.Select) or not _wrote_limit(root):
        return None
    order = root.args.get("order")
    if order is None or not order.expressions:
        return None
    first = order.expressions[0]
    if not first.args.get("nulls_first") or root.meta.get("null_order_written"):
        return None
    col = first.this
    if not isinstance(col, exp.Column) or _has_not_null_guard(root, col.name):
        return None
    key = f"{col.name} DESC" if first.args.get("desc") else col.name
    # NULLS LAST, not `WHERE col IS NOT NULL`: the old repair dropped the NULL rows, and on BEAVER
    # it turned 4 correct gold statements of 40 into wrong ones. Reordering keeps every row.
    return Violation(
        "null_before_ordered_limit",
        f"ORDER BY {key} puts NULLs first on this engine, so the row limit can return NULLs "
        f"instead of values; write ORDER BY {key} NULLS LAST, which keeps every row and puts the "
        "values first.",
    )


def _join_or_fanout(root: exp.Expression) -> Violation | None:
    for join in root.find_all(exp.Join):
        on = join.args.get("on")
        if on is not None and on.find(exp.Or) is not None:
            return Violation(
                "join_or_fanout",
                "the JOIN ON condition uses OR, which multiplies rows and corrupts counts "
                "and sums; join on the single intended equality instead.",
            )
    return None


_DETECTORS = (_null_before_ordered_limit, _join_or_fanout)


def lint(root: exp.Expression) -> Violation | None:
    """First high-precision antipattern found, or None. Deterministic; no LLM, no execution."""
    for detector in _DETECTORS:
        violation = detector(root)
        if violation is not None:
            return violation
    return None
