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
    EMPTY/UNAVAILABLE for the same shape). No ordinal is written here: this codebase keeps
    finding new ones and a typed count is one more thing to go stale.

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
    # Whether this answer also covers what a VIEW BODY on this source may invoke. It does not,
    # for a Postgres attachment: measured, `SELECT median(1)` binds to DuckDB's aggregate and a
    # Postgres UDF is unreachable from a generated query, but reading a Postgres view whose body
    # calls that same UDF returns its value -- it ran server-side, where DuckDB's binder never
    # looked. So `duckdb_functions()` answers the top-level question completely and the
    # view-body question not at all, and a consumer walking view bodies must know which it has.
    # Without this field the Postgres path returns an empty list that reads as `certain`, which
    # would certify a server-side UDF as a builtin. Prose in a docstring cannot carry that.
    covers_view_bodies: bool = True

    def __post_init__(self) -> None:
        # Normalised HERE, not only in `of()`. A dataclass hands out its plain constructor
        # whether or not you meant to, and that one built a broken instance:
        # `FunctionInventory({"MEDIAN"}).defines("median")` was False and `hash()` raised on the
        # unfrozen set, while `of()` a few lines down did the right thing. Two constructors with
        # different semantics is the shape of a bug nobody looks for, and the case-sensitive one
        # fails in the direction that matters -- a UDF named like a builtin read as the builtin.
        names = self.names
        # A bare string is an iterable of characters, so `FunctionInventory("median")` produced
        # {a,d,e,i,m,n} and `defines("median")` was False -- failing in the direction that
        # matters, a UDF named like a builtin read as the builtin. The guard was in `of()` only,
        # which left the two constructors disagreeing about the same input.
        if isinstance(names, str):
            names = [names]
        object.__setattr__(self, "names", frozenset(n.lower() for n in names))

    @classmethod
    def of(cls, names, **kw) -> "FunctionInventory":
        """Build from any iterable of names, folded to lower case.

        A convenience over the constructor for the common "I have an iterable" case. Both
        normalise identically, in `__post_init__`, so neither can be the trap.
        """
        # Passed straight through: freezing here would splay a bare string into characters
        # BEFORE `__post_init__` could guard it, which is how the two constructors kept
        # disagreeing after the guard moved.
        return cls(names or (), **kw)

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
        """Whether a caller may treat a name absent from `names` as a builtin.

        Top-level calls only. A consumer walking view bodies wants `certain_for_view_bodies`.
        """
        return self.available and self.asked

    @property
    def certain_for_view_bodies(self) -> bool:
        """The same licence, for a call found inside a view body rather than the query."""
        return self.certain and self.covers_view_bodies
