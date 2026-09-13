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
    # Whether this SOURCE'S BINDER resolves a builtin before a schema object of the same name.
    # Measured per engine, because the two live engines disagree and the disagreement decides
    # whether the coarse rule is needed at all:
    #
    #   DuckDB  -- `count(*)` binds a macro named `count_star`, and `SELECT length('abc')` binds
    #             a macro named `length`. The builtin loses, so a name the query never says can
    #             collect the call, and only the coarse rule can catch that.
    #   Oracle  -- with a `LENGTH` function owned by the caller, `SELECT length('abc')` returns
    #             3, the builtin. The UDF answers only to `appuser.length(...)`. No FUNCTION
    #             name the query omits can reach it, so there is nothing for the coarse rule to
    #             catch and refusing the whole source would be pure cost.
    #
    # Read narrowly: this is about names a BINDER substitutes, not about every way user code
    # runs. An Oracle virtual column executes a UDF that the statement never names at all, which
    # no setting of this flag affects and this guard does not see (M100).
    #
    # Defaults FALSE: an engine nobody measured is assumed to behave like DuckDB, which is the
    # conservative half. Forgetting yields a refusal, not a leak.
    binder_prefers_builtins: bool = False
    # Names that must REFUSE A CALL without being evidence this source defines functions.
    # The third category, and `names` was doing both jobs: `calls_are_confirmable` is
    # `licensed and not names`, so ANY name downgrades the whole source's completeness.
    #
    # M104, measured: an Oracle schema defining nothing of its own, plus one
    # `PUBLIC orders FOR orders@ERP_LINK` over a remote TABLE, put `orders` in `names` and
    # flipped `calls_are_confirmable` from True to False -- `completeness='unknown'` on every
    # answer carrying any call. A link to a table is the common enterprise shape and nothing
    # local can tell it from a link to a function, so the alias must still refuse a call; what
    # it must not do is assert the source has functions, which is a different claim.
    #
    # Not read by `may_shadow_a_builtin`: an unresolvable alias cannot be reached by a binder
    # rename, because the binder would have to resolve it to do that, and the one engine that
    # produces these sets `binder_prefers_builtins` anyway.
    unresolvable: frozenset[str] = field(default_factory=frozenset)

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
        # Same normalisation, same reasons: a bare string splayed into characters, and a
        # case-sensitive set reading a UDF as a builtin.
        unresolvable = self.unresolvable
        if isinstance(unresolvable, str):
            unresolvable = [unresolvable]
        object.__setattr__(self, "unresolvable", frozenset(n.lower() for n in unresolvable))
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
        if self.binder_prefers_builtins:
            # Nothing to catch: this engine hands an unqualified call to its own builtin, so a
            # user function of that name is reachable only by a query that SPELLS the
            # qualification -- which `called_names` reads off the rendered text. Checked before
            # `builtins`, so an engine whose catalogue of builtins is unreadable still gets this
            # answer instead of a whole-source refusal it does not need.
            return False
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
        # SAME try, so one outage decides one way. Both are live dictionary reads carrying the
        # same security property, and handling them separately let a raise on this one fall
        # through to an empty set while the same raise on the other refused every statement.
        unresolvable = _unresolvable_from(adapter)
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
        binder_prefers_builtins=getattr(adapter, "binder_prefers_builtins", False),
        unresolvable=unresolvable,
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


def _unresolvable_from(adapter) -> frozenset[str]:
    """Aliases this source can neither resolve nor rule out. RAISES if the source cannot say.

    Absence and FAILURE are opposite here, and an earlier version of this collapsed them into
    an empty set with the justification that "forgetting it only means such names stay in
    `names`, where they already refuse a call". That was disproven by the same diff: the
    Oracle adapter stopped putting link aliases in `user_functions`, so an empty return leaves
    such a name in NEITHER set and the guard clears the call. Measured with a stub whose
    `unresolvable_aliases()` raised ORA-03113: `decide` APPROVED
    `SELECT add_days(1) FROM dbl_claim` -- the exact statement M102 measured returning an
    ungranted SSN, re-approved through the new door.

    So a raise PROPAGATES, and `inventory_from` turns it into `unavailable` exactly as it does
    for `user_functions`. Two live dictionary reads deciding opposite ways on one outage was
    the asymmetry that made this unsafe.

    ABSENCE still means empty, and that stays safe: an adapter without the method has no
    concept of an unresolvable alias, so whatever it knows is already in `names`.
    """
    if adapter is None or not hasattr(adapter, "unresolvable_aliases"):
        return frozenset()
    return frozenset(n.lower() for n in adapter.unresolvable_aliases())


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


def opaque_columns(adapter, inventory: FunctionInventory) -> frozenset[tuple[str, str]] | None:
    """(table, column) pairs whose stored expression this engine cannot attribute.

    A virtual column is a route to user code with no call in the statement, which is the third
    shape of that this codebase has found -- after `count(*)` binding a macro named `count_star`
    and `SELECT id + 1` binding one named `+`. Each time the answer was to stop enumerating
    shapes and ask the source a different question; this is that question for columns.

    Opaque means the expression NAMES something this source defines. Arithmetic does not --
    `"QTY"*"PRICE"` calls nothing -- and refusing every virtual column would cost a legitimate
    modelling feature to close a route that needs a function to be a route at all.

    Three answers, not two. An adapter that cannot report virtual columns yields an empty set,
    which is the state every adapter shipped in. One that was ASKED and could not answer yields
    **None**, and the guard refuses on it.

    The first version returned an empty set for both, with a comment arguing that the permissive
    reading was safe because `check_access` still stands. That is exactly backwards:
    `check_access` sees a column the snapshot lists and passes it, which is the whole reason
    this function exists. Measured on the first version -- with `virtual_columns()` raising,
    `SELECT leaked FROM vc_t` was APPROVED, so a catalogue outage switched the control off. The
    absence-and-failure collapse, argued for in a comment.

    An inventory that is not `certain` yields every virtual column, because then no name in an
    expression can be cleared.
    """
    if adapter is None or not hasattr(adapter, "virtual_columns"):
        return frozenset()
    try:
        columns = adapter.virtual_columns()
    except Exception:
        return None
    if not inventory.certain:
        return frozenset((t, c) for t, c, _ in columns)
    return frozenset(
        (table, column)
        for table, column, expression in columns
        # BOTH sets. A link alias left `names` when M104 split the two categories, so a
        # virtual column whose stored expression reaches one stopped being flagged -- the
        # M100 route back, for that one shape, in the same diff that moved the name.
        #
        # MEASURED: Oracle refuses `GENERATED ALWAYS AS (link_syn(id))` with ORA-02069,
        # "global_names parameter must be set to TRUE for this operation", while the same DDL
        # over a LOCAL function is accepted -- so the refusal is about the link and no such
        # column can exist on a default instance. But that error names a SETTING, so with
        # `global_names=TRUE` it may be permitted, and the union costs nothing where it is not.
        if names_called_in(expression) & (inventory.names | inventory.unresolvable)
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


# An identifier immediately before an open paren. Deliberately crude: it over-collects, and
# over-collecting is the safe direction for the one use it has. `FROM customer AS c(id, name)`
# yields `c`, which costs nothing unless the source also defines a function called `c` -- in
# which case refusing is right anyway. An earlier attempt used a scan like this to CLEAR calls
# by counting them, where the same over-collection was a leak (M56).
_CALLED = re.compile(r"([A-Za-z_][A-Za-z_0-9$]*)\s*\(")


class UnreadableCalls(Exception):
    """The statement could not be rendered, so its call names could not be enumerated."""


def names_called_in(expression: str) -> frozenset[str]:
    """Every name called in a stored expression, over-collected on purpose.

    The same crude scan `called_names` runs over a rendered statement, on a fragment the source
    stored rather than one this engine rendered -- a virtual column's `DATA_DEFAULT`, which
    arrives quoted: `"APPUSER"."VC_UDF"("ID")`. The quotes are stripped first, because an
    identifier followed by `"` followed by `(` matches nothing otherwise, and a scan that
    silently finds no calls in a fragment full of them is worse than no scan.
    """
    return frozenset(m.lower() for m in _CALLED.findall(expression.replace('"', "")))


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
