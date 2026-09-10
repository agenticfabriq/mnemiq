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

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FunctionInventory:
    """User-defined function names on a source, plus whether the source could be ASKED.

    An empty answer asserts "this source defines no functions of its own". A failed lookup
    produces the same empty result and means "we do not know what it defines", and no control
    may read the second as the first. That is another instance of the absence-and-failure
    collapse this codebase keeps finding (see `ViewInventory` and `authz/grants.py`'s
    EMPTY/UNAVAILABLE, and the register for the running list -- no count is written here,
    because a typed count is one more thing to go stale).

    `available` is whether the source answered. `asked` is whether anyone put the question,
    which `available` cannot carry: an adapter with no inventory method and one whose query
    failed both produce nothing, and they deserve different responses. Declining to CERTIFY
    lineage costs nothing in either case; refusing every function because a fixture never asked
    would be harsh.

    Deliberately NOT a `frozenset` subclass, unlike `ViewInventory`'s `dict`. That inheritance
    buys back-compat for callers holding a bare mapping, and this type has no such callers to
    keep. It costs the thing the type exists for: measured on the subclassing version,
    `FunctionInventory(()) == FunctionInventory.unavailable()` was True with equal hashes, so a
    memo keyed on one returned the other, and `unavailable | other` produced a plain frozenset
    whose missing flags read as available under the codebase's `getattr(x, "available", True)`
    idiom. A federated caller unioning per-source inventories would have certified lineage over
    a source whose lookup failed, and the only sign would be a reason absent from an audit
    record. Names are reached through `defines()`, which is what both callers actually want.
    """

    names: frozenset[str] = field(default_factory=frozenset)
    available: bool = True
    asked: bool = True
    reason: str = ""   # declared for every constructor, not only the failing one

    @classmethod
    def of(cls, names, **kw) -> "FunctionInventory":
        """Build from any iterable of names, folded to lower case.

        SQL folds case and catalogues disagree across engines, so raw strings would answer
        `defines('MEDIAN')` differently from `defines('median')`.
        """
        return cls(frozenset(n.lower() for n in (names or ())), **kw)

    @classmethod
    def unavailable(cls, reason: str = "") -> "FunctionInventory":
        """Asked and could not answer. Denies certification, never an answer."""
        return cls(available=False, asked=True, reason=reason)

    @classmethod
    def never_asked(cls) -> "FunctionInventory":
        """Nobody put the question -- an adapter without the method, or a bare fixture."""
        return cls(available=True, asked=False)

    def defines(self, name: str) -> bool:
        """True when THIS source is known to define `name` itself."""
        return name.lower() in self.names

    @property
    def certain(self) -> bool:
        """Whether a caller may treat a name absent from `names` as a builtin."""
        return self.available and self.asked
