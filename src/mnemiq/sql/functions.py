"""What the source says about its own functions, and whether it could be asked.

The engine cannot tell a builtin from a same-named user-defined function by parsing. sqlglot
types `log` as a builtin before the source binds it, so a UDF called `log` is invisible, and two
separate controls settle for less because of it: `lineage` marks ANY call unconfirmable (issue
#5, which makes the marker fire on `count(*)` and so on nearly every real query), and
`check_unmodelled_calls` allows any name sqlglot models, so a UDF called `median` passes (M43's
open residual).

Both want the same fact from the same place: which names on THIS source are not builtins.
"""

from __future__ import annotations


class FunctionInventory(frozenset):
    """User-defined function names on a source, plus whether the source could be ASKED.

    A plain empty set asserts "this source defines no functions of its own". A failed lookup
    produces the same empty set and means "we do not know what it defines", and no control may
    read the second as the first. That distinction is the whole reason this is a type rather
    than a `set[str]`, and it is the twelfth instance in this codebase of an absence and a
    failure sharing one value -- `ViewInventory` and `authz/grants.py`'s EMPTY/UNAVAILABLE are
    the same shape, deliberately, so the three read alike.

    `available` is whether the source answered. `asked` is whether anyone put the question,
    which `available` cannot carry: an adapter that has no inventory method and one whose query
    failed both produce an empty set that is available by default, and the two want different
    responses. Refusing every function because a hand-built fixture never asked would be harsh;
    declining to CERTIFY lineage in the same case costs nothing.

    A frozenset subclass so a caller holding a bare set keeps meaning what it meant.
    """

    available: bool
    asked: bool

    def __new__(cls, names=None, available: bool = True, asked: bool = True):
        self = super().__new__(cls, {n.lower() for n in (names or ())})
        self.available = available
        self.asked = asked
        return self

    @classmethod
    def unavailable(cls, reason: str = "") -> "FunctionInventory":
        """The source was asked and could not answer. Denies certification, never an answer."""
        inv = cls((), available=False, asked=True)
        inv.reason = reason
        return inv

    @classmethod
    def never_asked(cls) -> "FunctionInventory":
        """No one put the question. Distinct from an empty answer, per the class docstring."""
        return cls((), available=True, asked=False)

    def defines(self, name: str) -> bool:
        """True when THIS source is known to define `name` itself."""
        return name.lower() in self

    @property
    def certain(self) -> bool:
        """Whether a caller may treat a name absent from this set as a builtin."""
        return self.available and self.asked
