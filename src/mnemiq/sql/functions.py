"""What the source says about its own functions, and whether it could be asked.

The engine cannot tell a builtin from a same-named user-defined function by parsing. sqlglot
types `log` as a builtin before the source binds it, so a UDF called `log` is invisible, and two
separate controls settled for less because of it: `lineage` marked ANY call unconfirmable (issue
#5, which made the marker fire on `count(*)` and so on nearly every real query), and
`check_unmodelled_calls` allowed any name sqlglot models, so a UDF called `median` passed and
returned an SSN from an ungranted table (M43's residual).

Both want the same fact from the same place: which names on THIS source are not builtins. They
then ask it differently, and the difference is not an inconsistency. Lineage REPORTS, so the
cheapest sound rule is right for it: any user function at all means no call is confirmable.
The decider REFUSES, so the same rule would end the product on a source with one macro -- it
splits the question instead, into the names a statement is written to ask for
(`called_names`) and the names a binder can substitute without being asked
(`may_shadow_a_builtin`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlglot import exp


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
    # What the source calls a BUILTIN, or None for "never asked". Only used to answer one
    # question -- whether any name in `names` shadows one -- and that question is what makes a
    # call's true binding underivable from the text. Measured: with a macro named `count_star`
    # in an attached read-only DuckDB file, `SELECT count(*) FROM claim` returned an SSN from an
    # ungranted table. `count_star` appears in no spelling of that query, in no dialect, because
    # it is the BINDER's name for `COUNT(*)`. A macro can only be reached that way if its name
    # is a builtin, so the intersection is exactly the set of sources where names cannot be
    # trusted at all.
    #
    # `None`, not an empty frozenset, for the reason the type exists: an empty answer would
    # assert "this source shadows nothing", which is the absence-and-failure collapse in the
    # one place it would fail open. Defaulting to None makes forgetting conservative.
    builtins: frozenset[str] | None = None
    # Whether anyone PUT the builtin question, which `builtins is None` cannot carry: an
    # adapter with no `builtin_functions` and one whose call raised both leave it None. The
    # same split as `available` and `asked` above, one level down, and it exists for the same
    # reason -- the refusal names a different owner for each, and telling an adapter author to
    # implement a method they already implemented sends them to fix working code.
    builtins_asked: bool = False
    # The subset of `names` a query can call WITHOUT qualifying it, or None for "not known".
    # Only `may_shadow_a_builtin` reads it, and only because that rule is the coarse one: a
    # binder rename can only reach a function the query could have called unqualified, so a
    # macro in a schema off the search path cannot collect `COUNT(*)`. `names` stays whole,
    # because a query may still qualify and the bare name read off the rendered text catches it.
    #
    # None falls back to `names`, which is what an adapter that cannot scope its catalogue gets
    # -- the same shape as `builtins`, and conservative in the same direction.
    reachable: frozenset[str] | None = None

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
        reachable = self.reachable
        if reachable is not None:
            if isinstance(reachable, str):
                reachable = [reachable]
            object.__setattr__(self, "reachable", frozenset(r.lower() for r in reachable))
        builtins = self.builtins
        if builtins is not None:
            # An answer implies the question. Without this the constructor can build "the
            # catalogue answered, but nobody asked", which the field's own comment forbids and
            # which a later reader would take at face value.
            object.__setattr__(self, "builtins_asked", True)
            if isinstance(builtins, str):
                builtins = [builtins]
            object.__setattr__(self, "builtins", frozenset(b.lower() for b in builtins))

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

    @property
    def may_shadow_a_builtin(self) -> bool:
        """Whether some call here could bind to a user function under a name the SQL never says.

        A user function is reachable in two ways. Under its own name, which the statement must
        then spell, so enumerating the names a statement calls finds it. Or under a builtin's
        name, where the engine's binder supplies a name the statement never contains -- DuckDB
        turns `COUNT(*)` into `count_star`, and a macro called `count_star` collects the call.
        Only the second is underivable, and only a name that IS a builtin can be reached that
        way. So this is the whole question of whether names can be trusted on this source.

        Three answers rather than two. Nothing defined, nothing to shadow. Builtins never
        asked, so assume the worst -- but only for a source that defines something, which keeps
        the conservative default off every source that has no user functions at all.

        And the question is asked of the REACHABLE names, not all of them, because a binder
        rename can only ever land on a function the query could have called unqualified. A
        macro in a schema off the search path is invisible to `COUNT(*)`; counting it condemned
        whole sources that were answering correctly.
        """
        # The names a query could have called unqualified, which is the only way a BINDER
        # rename can land on a user function. Scoped, because the unscoped version condemned a
        # whole source over a macro in a schema nothing on the search path can reach.
        candidates = self.names if self.reachable is None else self.reachable
        if not candidates:
            return False
        if self.builtins is None:
            return True
        return bool(candidates & self.builtins)


def inventory_from(adapter) -> FunctionInventory:
    """Ask an adapter what it defines, in the one place, so two callers cannot drift.

    LIVE, not from the snapshot, which is where `inventory_for(views)` reads. The two differ
    because their staleness fails in opposite directions. A view body is policy content: the
    snapshot holds the body governance was reasoned about, and a body that drifted since is a
    governance change worth noticing. A function list is not policy, it is what names mean --
    and a UDF created since the snapshot would read as a builtin, failing open in exactly the
    direction M43 is about. Measured at ~13ms against queries that take seconds -- two scans of
    `duckdb_functions()`, one per half.

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
    builtins, builtins_asked = _builtins_from(adapter)
    return FunctionInventory.of(
        names,
        covers_view_bodies=getattr(adapter, "functions_cover_view_bodies", False),
        builtins=builtins,
        builtins_asked=builtins_asked,
        reachable=_reachable_from(adapter),
    )


def _reachable_from(adapter) -> frozenset[str] | None:
    """Which of the source's own functions answer to an unqualified call, or None if unasked.

    Optional like `builtin_functions`, and it costs precision rather than soundness in the same
    direction: without it every user function counts as reachable, which is where this started.
    """
    if adapter is None or not hasattr(adapter, "reachable_user_functions"):
        return None
    try:
        return frozenset(n.lower() for n in adapter.reachable_user_functions())
    except Exception:
        return None


def _builtins_from(adapter) -> tuple[frozenset[str] | None, bool]:
    """The source's own builtin catalogue and whether the question was put at all.

    Separate from `user_functions` because a consumer that never asks still gets the
    conservative answer out of `may_shadow_a_builtin` -- soundness does not depend on it. What
    depends on it is whether the source can answer anything at all: without this, a source
    holding one macro has every read and every write against it refused. Optional in the sense
    that omitting it cannot open a hole, not in the sense that omitting it is cheap -- the
    refusal is classified unrepairable, so the caller gets it at once (M98) rather than after a
    round of retries that cannot succeed -- one clean refusal naming what to change.
    """
    if adapter is None or not hasattr(adapter, "builtin_functions"):
        return None, False
    try:
        return frozenset(n.lower() for n in adapter.builtin_functions()), True
    except Exception:
        # No reason string, unlike the `user_functions` failure -- that one changes what the
        # inventory MEANS, while this lands on the default the field already has. The fact that
        # it was ASKED is kept, though: it is the difference between an adapter that has not
        # implemented this and one whose source would not answer, and the refusal says which.
        return None, True


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


# An identifier immediately before an open paren. Deliberately crude: it over-collects, and
# over-collecting is the safe direction for the one use it has. `FROM customer AS c(id, name)`
# yields `c`, which costs nothing unless the source also defines a function called `c` -- in
# which case refusing is right anyway. An earlier attempt used a scan like this to CLEAR calls
# by counting them, where the same over-collection was a leak (M56).
_CALLED = re.compile(r"([A-Za-z_][A-Za-z_0-9$]*)\s*\(")


class UnreadableCalls(Exception):
    """The statement could not be rendered, so its call names could not be enumerated."""


def called_names(ast: exp.Expression, *dialects: str) -> frozenset[str]:
    """Every function name the SOURCE will see when this statement runs.

    Read off the RENDERED statement, not the tree and not the model's text, because rendering
    is what the source is handed: `decide` approves an AST and `target_sql` is generated from
    it. Three earlier attempts got this wrong in the other direction and each leaked an SSN --
    the written spelling is not what executes, `len(name)` arrives as `LENGTH(name)`.

    Not the tree, because a node loses the word. `date_trunc(...)` parses to
    `exp.TimestampTrunc`, whose declared names are `timestamp_trunc` and `trunc`, and renders
    back to `DATE_TRUNC(...)`. Only the text has the name the binder will resolve.

    A weaker question than the one those attempts failed at, and that is what makes it safe.
    They asked which name a call BINDS to, so a missing spelling cleared a UDF. This asks which
    names the source is asked for, so a missing spelling can only over-refuse -- and the one
    thing text cannot show, the binder's own renames (`COUNT(*)` to `count_star`, `EXTRACT` to
    `date_part`), is covered by `may_shadow_a_builtin` instead, because a rename can only ever
    land on a builtin's name.

    Rendered once per dialect given: the parse dialect and the execution target can spell the
    same node differently, and more names can only over-refuse. Raises `UnreadableCalls` when
    none of them renders, rather than returning an empty set -- a caller reading that as "no
    calls" would clear every one of them.
    """
    found: set[str] = set()
    rendered = False
    for dialect in dialects or (None,):
        try:
            text = ast.sql(dialect=dialect) if dialect else ast.sql()
        except Exception:
            continue
        rendered = True
        found.update(m.lower() for m in _CALLED.findall(text))
    if not rendered:
        raise UnreadableCalls(type(ast).__name__)
    return frozenset(found)
