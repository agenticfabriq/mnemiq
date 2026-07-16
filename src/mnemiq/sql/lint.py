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


def _null_before_ordered_limit(root: exp.Expression) -> Violation | None:
    if not isinstance(root, exp.Select) or root.args.get("limit") is None:
        return None
    order = root.args.get("order")
    if order is None or not order.expressions:
        return None
    first = order.expressions[0]
    if bool(first.args.get("desc")):  # DESC -> NULLs last, not the trap
        return None
    col = first.this
    if not isinstance(col, exp.Column) or _has_not_null_guard(root, col.name):
        return None
    return Violation(
        "null_before_ordered_limit",
        f"ORDER BY {col.name} ascending with LIMIT can return NULLs first; add "
        f"WHERE {col.name} IS NOT NULL so the limit picks a real value.",
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
