from __future__ import annotations

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect

from mnemiq.sql.functions import FunctionInventory, UnreadableCalls, called_names
from mnemiq.sql.qualify import object_key
from mnemiq.sql.scope import base_tables, candidate_tables, column_tables
from mnemiq.sql.verdict import Refusal, RefusalCode


def check_access(ast: exp.Expression, visible: dict[str, set[str]],
                 dialect: str | None = None) -> Refusal | None:
    """Re-check every table and column the query touches against what the identity may see.

    `dialect` is the ENGINE that will run this, not the one it was parsed as. Identifier folding differs between them -- Oracle folds unquoted names up, the rest fold them down, and quoting is significant in Postgres and Oracle but not DuckDB -- so a resolver keyed on the wrong one answers for the wrong engine. Threading it to SOME call sites and not others is worse than either: `check_access` said a name was a CTE while the audit list said it was a base read, in the same statement.

    Retrieval scoping (the semantic store) means the model was never *shown* a forbidden
    table. It can still *name* one -- `users`, `employees`, `salaries` are in every schema it
    was trained on. This is the lock that makes naming it useless.
    """
    # tables, and the aliases that stand for them. `base_tables` -- not `find_all` minus a flat
    # set of CTE names -- because a reference inside a CTE body naming that same CTE reads the
    # base table, and skipping it let an ungranted table through (M31).
    alias_to_table: dict[str, str] = {}
    resolved = column_tables(ast, dialect)
    for table in base_tables(ast, dialect):
        name = object_key(table)
        if name not in visible:
            return Refusal(
                code=RefusalCode.UNAUTHORIZED_TABLE,
                message=f"You may not query {name!r}. Answer using only the tables provided.",
                subject=name,
            )
        alias_to_table[table.alias_or_name] = name

    if not alias_to_table:
        return None  # a query over CTEs alone; their sources were checked above

    referenced = set(alias_to_table.values())
    known_columns = {c for t in referenced for c in visible[t]}

    # A SELECT alias is a name the query invents for an expression, and GROUP BY / ORDER BY /
    # HAVING may refer to it: `SELECT year(d) AS y, count(*) FROM t GROUP BY y` is ordinary,
    # correct SQL -- and without this the engine could not answer "count X by year" at all.
    # It is safe: an alias can only be defined from columns that were themselves checked.
    known_columns |= {alias.alias for alias in ast.find_all(exp.Alias) if alias.alias}

    for column in ast.find_all(exp.Column):
        qualifier = column.table
        if qualifier:
            # Resolved per scope, not from one flat map: an alias can name two different
            # tables in one statement, and keeping only the last let a column be checked
            # against the wrong one (Codex review, 2026-08-12).
            if resolved is None:
                # Unreadable scopes: fall back to the rule for an unqualified column rather
                # than to the flat map that was the defect. Same guarantee, no worse.
                if column.name not in known_columns:
                    return Refusal(
                        code=RefusalCode.UNKNOWN_COLUMN,
                        message=(
                            f"No table in this query has a column {column.name!r}. "
                            "Use only the columns listed on the schema cards."
                        ),
                        subject=column.name,
                    )
                continue
            table = resolved.get(id(column))
            if table is None:
                continue  # qualifier belongs to a CTE or a subquery alias
            if column.name not in visible[table]:
                return Refusal(
                    code=RefusalCode.UNKNOWN_COLUMN,
                    message=f"{table!r} has no column {column.name!r}. Use only listed columns.",
                    subject=column.name,
                )
        elif column.name not in known_columns:
            return Refusal(
                code=RefusalCode.UNKNOWN_COLUMN,
                message=(
                    f"No table in this query has a column {column.name!r}. "
                    "Use only the columns listed on the schema cards."
                ),
                subject=column.name,
            )

    return None





def _known_function_names() -> frozenset[str]:
    """Every function name sqlglot can model, in ANY dialect.

    Not `exp.Func` subclasses alone: those carry `sql_names()` for 623 names and miss `now`
    and `date_part`, which live in each dialect's PARSER table rather than on a class. Missing
    them is not academic -- see the guard below for what it cost.
    """
    names: set[str] = set()

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            try:
                names.update(n.upper() for n in sub.sql_names())
            except Exception:  # a class that does not declare names is simply not a source
                pass
            walk(sub)

    walk(exp.Func)
    for dialect in Dialect.classes.values():
        try:
            names.update(k.upper() for k in dialect.parser_class.FUNCTIONS)
        except Exception:
            continue
    return frozenset(names)


_KNOWN_FUNCTIONS = _known_function_names()


def check_unmodelled_calls(
    ast: exp.Expression,
    inventory: FunctionInventory | None = None,
    *dialects: str,
) -> Refusal | None:
    """Refuse a call this engine cannot model, because it cannot say what such a call reads.

    `check_access` re-checks every `exp.Table` against the identity's visible set, and the RLS
    rewrite wraps each one. A function in PROJECTION position produces no `exp.Table` at all,
    so both walk past it: measured, `SELECT customer_rows() AS x` and
    `SELECT pg_read_file('/etc/passwd')` were APPROVED with `tables=[]` for a caller granted
    only `claim`, while `SELECT id FROM customer` was correctly refused in the same run (M43).

    A **whitelist**, the same inversion `unrecognised_source` took for source shapes, and for
    the reason its docstring gives: enumerate dangerous functions and an unlisted one passes,
    with every review round finding another nobody thought of.

    The allowlist is sqlglot's whole function vocabulary, ACROSS DIALECTS, and that detail is
    the guard. The first version tested `isinstance(node, exp.Anonymous)`, which asks whether
    the ONE dialect being parsed happens to model the name. Production parses as `duckdb`
    (`Agent.dialect` from the adapter), sqlglot models `now` and `date_part` only under
    postgres, and so the first version refused `SELECT id FROM claim WHERE created_at < now()`
    -- measured through `decide`, on the shipped default. Its control test asserted the
    opposite while pinning `dialect="postgres"`, the one dialect where the claim held.

    Not a claim that an unknown call reads data. It is a claim that the decider cannot tell,
    and the decider's premise is that it sees every table a query reads. An opaque call makes
    that premise false, so the honest verdict is that this query cannot be decided.

    The allowlist alone was names, not identities: a UDF named `median` is modelled by sqlglot
    and passed. Measured through this function on the shipped default, with a macro in an
    attached read-only DuckDB file, `SELECT median(id) FROM claim` was APPROVED with
    `tables=['claim']` and returned an SSN from a table the identity was never granted. So did
    `SELECT count(*) FROM claim`, against a macro named `count_star` -- a name that appears in
    no spelling of that query, because it is DuckDB's binder name for `COUNT(*)`.

    `inventory` closes that, in the two shapes the leak comes in. A call reachable under its
    OWN name must be spelled, so `called_names` finds it however sqlglot rewrote the node, and
    the refusal names it. A call reachable under a BUILTIN's name is not derivable at all, and
    cannot happen unless the source defines a name a builtin also has -- so that case condemns
    the whole SOURCE, every statement against it, whether or not anything here looks like a
    call. `SELECT id + 1 FROM claim` was the measurement: no `exp.Func` node in it, `+` shadowed
    by a macro, and an SSN in the result.

    A third arrival, and it is the one that has to be written down rather than inferred: an
    inventory that was ASKED and could not answer carries the same empty `names` as one that
    answered "none", and reaches `_cannot_resolve` for it. `never_asked` does not, because that
    is every fixture and the state the engine shipped in -- without an inventory this is the
    allowlist alone, which is where it started.

    **WHICH SOURCES THAT ACTUALLY COVERS, because the paragraphs above read as if it were all of
    them.** Only an adapter implementing `user_functions` is asked -- that method alone decides
    `never_asked`; `virtual_columns` is a separate path, and an adapter missing it yields an
    empty opaque set rather than landing here. When this paragraph was written the DuckDB
    family was the only one asked, and it named `OracleAdapter` as the live adapter
    implementing neither, so an Oracle schema function called `median` passed here exactly as
    every source did before the inventory existed (M99).

    **Oracle answers both now**, at different times: `user_functions` at `ce4e16a`,
    `virtual_columns` three commits later at `77d24b0`. So every `names`-driven rule below
    applies to it. `binder_prefers_builtins` is True there, which rules out the coarse shadow
    arm and leaves two: `called & names`, and -- for any statement the scan cannot read at all
    -- the `UnreadableCalls` refusal, which fires whatever the scan returns. `adapters/oracle.py`
    records that second arm as the correction to a note of its own that framed the first as the
    only one.

    An adapter still implementing `user_functions` gets asked; one that does not gets
    `never_asked` and the bare allowlist, which remains the reason this is not fail-closed by
    default: that would refuse every call-bearing query on the adapters that cannot answer. The
    older Oracle exposure is filed as deployment preconditions M66 and M71, which
    `adapters/oracle.py` documents at length.
    """
    for call in ast.find_all(exp.Anonymous):
        name = str(call.this)
        if name.upper() in _KNOWN_FUNCTIONS:
            continue
        return Refusal(
            code=RefusalCode.UNMODELLED_CALL,
            message=(
                f"{name}() is a function this engine cannot model, so it cannot confirm what "
                "the query reads. Answer using only the listed tables and columns and "
                "standard SQL functions."
            ),
            subject=name,
        )

    inventory = inventory if inventory is not None else FunctionInventory.never_asked()
    if inventory.asked and not inventory.available:
        # Asked and could not answer, which is not the same as answering "none" -- and the
        # empty `names` an `unavailable` inventory carries would otherwise fall through every
        # test below and clear the statement. The collapse this type exists to prevent, in the
        # guard written to use it: measured, a raising `user_functions()` on a source holding a
        # `median` macro left `SELECT median(id) FROM claim` approved.
        return _cannot_resolve(inventory)
    if not (inventory.names or inventory.unresolvable):
        return None

    if inventory.may_shadow_a_builtin:
        # No `exp.Func` test in front of this, deliberately, and that absence is the fix for a
        # leak this function's first version had. `exp.Add`, `exp.DPipe` and `exp.AtTimeZone`
        # are not `exp.Func` subclasses, while DuckDB lists `+`, `||` and `timezone` as
        # internal functions a macro can shadow -- measured, `SELECT id + 1 FROM claim` was
        # APPROVED and returned an SSN with a `"+"` macro in the source. Enumerating the node
        # types that bind to a catalogue entry is the blocklist this codebase keeps refusing to
        # write; proving a statement CALL-FREE is as hard as naming its calls, so a source
        # whose names cannot be trusted answers nothing.
        return _cannot_resolve(inventory)

    try:
        called = called_names(ast, *dialects)
    except UnreadableCalls:
        # Cannot enumerate, so cannot clear. The same verdict as shadowing, and for the same
        # reason -- an empty set read as "no calls" would clear all of them -- but its own
        # sentence: this one is about THIS statement, and blaming the adapter for it would
        # send the deployer to fix a catalogue method that is working.
        return _cannot_resolve(inventory, unreadable=True)

    # UNRESOLVABLE FIRST, and with its own sentence. "Defined by this source itself" is false
    # of a synonym over a database link -- the source cannot see the target either -- and a
    # deployer told to rename their function will look for one that is not there.
    for name in sorted(called & inventory.unresolvable):
        return Refusal(
            code=RefusalCode.UNMODELLED_CALL,
            message=(
                f"{name}() reaches this source through a synonym it cannot resolve, so this "
                "engine cannot confirm what the query reads. Answer using only the listed "
                "tables and columns and standard SQL functions."
            ),
            subject=name,
        )

    for name in sorted(called & inventory.names):
        return Refusal(
            code=RefusalCode.UNMODELLED_CALL,
            message=(
                f"{name}() is defined by this source itself, so this engine cannot confirm "
                "what the query reads. Answer using only the listed tables and columns and "
                "standard SQL functions."
            ),
            subject=name,
        )
    return None


def check_opaque_columns(ast: exp.Expression, opaque, dialect: str | None = None) -> Refusal | None:
    """Refuse a column whose stored expression this engine cannot attribute.

    A virtual column runs user code on read, and the statement never names it. MEASURED on
    Oracle: `leaked AS (vc_udf(id))` over a function reading another table, and
    `SELECT id, leaked FROM vc_t` returned an SSN from a table the identity was never granted.
    `check_unmodelled_calls` walks call nodes and sees none; `called_names` reads the rendered
    text and sees none; `check_access` sees a column the snapshot lists and passes it.

    Third time this codebase has met code reached with no call in the text -- after `count(*)`
    binding a macro named `count_star`, and `SELECT id + 1` binding one named `+`. The first two
    were answered by asking the source a different question rather than enumerating shapes, and
    so is this: `opaque` comes from the source's own dictionary.

    REPAIRABLE, unlike the other arrivals at this code. Selecting a different column is a real
    rewrite, and the message names the one to avoid.
    """
    if opaque is None:
        # ASKED and could not answer. Every column could be computed by an expression nobody can
        # read, and `check_access` passes a listed column, so there is nothing else looking.
        # Retried rather than final: `decide` re-reads the catalogue on every attempt.
        return Refusal(
            code=RefusalCode.UNRESOLVABLE_CALLS,
            message=(
                "This source could not say which of its columns are computed, so this engine "
                "cannot confirm what reading them executes. Try again."
            ),
            repairable_override=True,
        )
    if not opaque:
        return None
    referenced = {object_key(t) for t in base_tables(ast, dialect)}
    # Scope-aware, like `check_access` and `check_cls`. A flat alias map keyed on the written
    # case let `SELECT Q.leaked FROM vc_t Q` through while refusing the lowercase spelling, and
    # one alias meaning two tables across UNION branches changed the verdict with branch order.
    resolved = column_tables(ast, dialect)
    local = {cte.alias_or_name for cte in ast.find_all(exp.CTE)}
    aliased = local | {s.alias_or_name for s in ast.find_all(exp.Subquery) if s.alias_or_name}

    def refuse(name: str, why: str, repair: str) -> Refusal:
        return Refusal(
            code=RefusalCode.UNRESOLVABLE_CALLS,
            message=(
                f"{name!r} is computed by an expression this source stores and this engine "
                f"cannot attribute, so it cannot confirm what {why} executes. {repair}"
            ),
            subject=name,
            repairable_override=True,
        )

    # A JOIN KEY is not an `exp.Column`, and it reads the column all the same. `USING (leaked)`
    # names it as a bare identifier, and a NATURAL join names nothing at all -- both evaluate the
    # expression per row, and whether rows match leaks its value a bit at a time. Measured: both
    # passed a guard that walked only column nodes, while `WHERE leaked = 'x'` was refused, which
    # is the same channel one syntax over.
    for join in ast.find_all(exp.Join):
        for key in join.args.get("using") or ():
            name = key.name.lower() if hasattr(key, "name") else str(key).lower()
            for table in referenced:
                if (table.lower(), name) in opaque:
                    return refuse(name, "joining on it", "Join on a different column.")
        if (join.args.get("method") or "").upper() == "NATURAL":
            # It names no column, so there is nothing to check against: the keys are whatever the
            # two tables share. Any opaque column on a table in this statement could be one.
            # Lowercased, like the USING arm and the column loop. `opaque` keys arrive
            # lowercase from the adapter and an Oracle query naturally writes `FROM VC_T`, so
            # comparing raw let the uppercase spelling through -- the case that source actually
            # produces, while the lowercase one was refused.
            lowered = {t.lower() for t in referenced}
            for table, column in sorted(opaque):
                if table.lower() in lowered:
                    return refuse(column, "a NATURAL join on this table",
                                  "Join with an explicit ON or USING naming the columns.")

    for column in ast.find_all(exp.Column):
        name = column.name.lower()
        # A qualifier names the table directly; without one, any table in scope could own it,
        # and an opaque column anywhere in scope is one this statement may be reading.
        owners = candidate_tables(column, resolved, referenced, aliased)
        for table in owners:
            if (table.lower(), name) in opaque:
                return refuse(name, "reading it", "Answer without that column.")
    return None


def _cannot_resolve(inventory: FunctionInventory, *, unreadable: bool = False) -> Refusal:
    """Nothing here can be attributed, so nothing here can be decided.

    Several ways to arrive, and each gets its OWN SENTENCE, because a refusal that sends the
    reader to the wrong place is worse than a vague one. Sharing an owner is not enough to
    share a sentence: two of these are the source failing, and "it would not name its builtins"
    and "it would not name its own functions" are found by looking in different places.

      * the source defines a name it also lists as a builtin -- the DEPLOYER renames it
      * the adapter has no `builtin_functions` -- its AUTHOR implements it
      * it has one and the call raised -- the SOURCE is the problem, and it may be transient
      * `user_functions` itself raised -- likewise, and earlier
      * this one statement would not render -- about the STATEMENT, not the source at all

    Worth RETRYING, and saying so with `repairable_override`: the statement that would not
    render, since a rewrite can fix it, and both catalogue calls that RAISED, since `decide`
    re-asks the source on every attempt and a blip resolves itself. Not worth retrying: a source
    that redefines a builtin's name, and an adapter with no `builtin_functions` at all. Neither
    is a thing an attempt of ours changes, and the loop stops on them (M98).

    Two of them were caught borrowing another's sentence, and both times the effect was to send
    someone to fix working code. That is the failure this list is arranged against.
    """
    if unreadable:
        # "defines functions of its own" is FALSE where only `unresolvable` is populated -- a
        # synonym over a database link is not the source's function, and the source cannot see
        # its target either. Widening the gate to `names or unresolvable` made this arm
        # reachable in that state, and a deployer reading it goes looking for a UDF that does
        # not exist. The same argument the sibling arm's own sentence rests on.
        has_own = bool(inventory.names)
        why = (
            "this source defines functions of its own"
            if has_own
            else "this source has names this engine cannot resolve"
        )
        return Refusal(
            code=RefusalCode.UNRESOLVABLE_CALLS,
            message=(
                "This query could not be rendered, so this engine cannot confirm which "
                f"functions it asks the source for, and {why}. "
                "Answer using only the listed tables and columns and standard SQL."
            ),
            # The one arrival that is about the STATEMENT, so the one a rewrite can fix -- and
            # the message asks for one. The code alone would send it to the no-retry path with
            # the other arrivals, under a card saying rephrasing will not help.
            repairable_override=True,
        )
    if not inventory.available:
        # Retried, because it may be transient and `decide` re-asks the source on every attempt
        # (`inventory_from` reads LIVE, not from the snapshot). A single blip should not tell a
        # caller to page an operator. If it is not a blip the attempts run out and it says so.
        return Refusal(
            code=RefusalCode.UNRESOLVABLE_CALLS,
            message=(
                "This source could not say which functions it defines, so this engine cannot "
                "confirm what this query executes against it."
            ),
            repairable_override=True,
        )
    shadowed = sorted(inventory.names & inventory.builtins) if inventory.builtins else []
    if shadowed:
        # The source's own configuration. Every attempt lands here until someone renames it.
        detail = f"This source defines {shadowed[0]!r} under a name it also lists as a builtin"
        retry = False
    elif inventory.builtins_asked:
        # ASKED and failed, so the same call that raised is made again next attempt -- the same
        # argument as the `user_functions` arrival above, which is why leaving this one out was
        # a contradiction rather than a nuance.
        detail = ("This source defines functions of its own and could not say which names are "
                  "its builtins")
        retry = True
    else:
        # The adapter has no such method, which is not a thing an attempt changes.
        detail = ("This source defines functions of its own and was never asked which names are "
                  "its builtins")
        retry = False
    return Refusal(
        code=RefusalCode.UNRESOLVABLE_CALLS,
        message=(
            f"{detail}, so this engine cannot confirm what this query executes against it. "
            + ("Try again." if retry else "No query can be decided against this source.")
        ),
        subject=shadowed[0] if shadowed else None,
        repairable_override=True if retry else None,
    )
