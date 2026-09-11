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
    #
    # Defaults FALSE. The first version defaulted True, so the obvious wiring --
    # `FunctionInventory.of(adapter.user_functions())` -- produced exactly the certification the
    # field was added to prevent, and the guard failed open on arrival. A producer that knows
    # its catalogue covers view bodies says so; forgetting yields the conservative answer, which
    # is the inversion `writes_enabled` took in M3 and `unrecognised_source` took for shapes.
    covers_view_bodies: bool = False

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
        normalise the NAMES identically, in `__post_init__`, so neither can be the trap that
        one of them was. They still differ on empty input: `of()` maps a falsey argument to
        `()`, so `of(None)` is an empty inventory where the constructor would raise.
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


def inventory_from(adapter) -> FunctionInventory:
    """Ask an adapter what it defines, in the one place, so two callers cannot drift.

    LIVE, not from the snapshot, which is where `inventory_for(views)` reads. The two differ
    because their staleness fails in opposite directions. A view body is policy content: the
    snapshot holds the body governance was reasoned about, and a body that drifted since is a
    governance change worth noticing. A function list is not policy, it is what names mean --
    and a UDF created since the snapshot would read as a builtin, failing open in exactly the
    direction M43 is about. Measured at ~6ms against queries that take seconds.

    Three answers, matching the three the type keeps apart. An adapter with no `user_functions`
    was never asked. One that raises was asked and could not answer. Anything else is an answer,
    including an empty one.
    """
    if adapter is None or not hasattr(adapter, "user_functions"):
        return FunctionInventory.never_asked()
    try:
        names = adapter.user_functions()
    except Exception as exc:
        # The TYPE, not the message. A source's exception text is made of the caller's schema
        # and can carry a DSN, and this reason reaches an audit record (M94).
        return FunctionInventory.unavailable(type(exc).__name__)
    return FunctionInventory.of(
        names, covers_view_bodies=getattr(adapter, "functions_cover_view_bodies", False)
    )


def calls_are_confirmable(inventory: FunctionInventory, *, in_view_body: bool) -> bool:
    """Whether a call in this statement can be taken for a builtin.

    ONE question about the SOURCE, not one per call, and that is the whole design. Three earlier
    attempts tried to name each call and match it against the catalogue. Each leaked, because a
    call's bound name is not recoverable from the text or the tree:

      * as WRITTEN misses what executes -- `len(name)` reaches the source as `LENGTH(name)` and
        bound a macro named `length`
      * as RENDERED misses what the engine itself renames -- DuckDB binds `count(*)` to
        `count_star`, `EXTRACT(...)` to `date_part` and `CURRENT_DATE` to `current_date()`, none
        of which appear in sqlglot's output
      * and a VIEW BODY binds as WRITTEN, since the source stored that text, so the rendering
        argument inverts for bodies

    Each returned COMPLETE over a macro reading another table. M56, three times, from one wrong
    assumption -- which is why this stopped being patched and changed shape instead.

    The question is not "which name is this call" but "could any call here be a UDF". If the
    source defines no functions of its own, no call can be one, however it is spelled. If it
    defines any, this engine cannot say which call is which, and says so.

    Cruder than per-call matching, and sound, which the precise version was not. The cost is
    that a source defining one unrelated helper downgrades every answer. The benefit is that no
    spelling, rename or dialect quirk can make a UDF read as a builtin.
    """
    licensed = inventory.certain_for_view_bodies if in_view_body else inventory.certain
    return licensed and not inventory.names
