"""The pre-registered kill criterion: `touched(a) == disclosed(a)`, per answer.

NOT a rate. Aggregate rates cancel -- one over-fire and one silence in the same run sum to green,
and both directions are live at once (over-firing from a bare-name match, under-firing from a fold
miss), so cancellation is the expected shape rather than a corner case. Hence a set equality on
every governed answer, plus two run-level conditions that stop a vacuous or unmeasured run from
reporting as a pass.

`touched` is derived HERE, from the plan and the policy, and deliberately never from
`apply_row_and_mask`. Deriving it from the disclosure path would make the criterion the disclosure
code agreeing with itself, which is green by construction on exactly the defects it exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.qualify_columns import qualify_columns
from sqlglot.schema import MappingSchema

from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.qualify import object_key
from mnemiq.sql.scope import base_tables, column_tables

# ("filter", object) | ("mask", object) -- option (a), table-granular, matching `Narrowing`.
# The kind tag is not decoration: without it an implementation rendering a row-filtered object as
# "columns were masked for you" produces an identical set on both sides and passes green.
Entry = tuple[str, str]


@dataclass(frozen=True)
class Touched:
    """What the policy narrows in what the plan referenced, plus what could not be attributed."""

    entries: set[Entry] = field(default_factory=set)
    # Masked-column references `column_tables` could not attribute to a definite object. An answer
    # holding any of these is UNMEASURABLE for the mask half, not green.
    unattributable: tuple[str, ...] = ()


def _qualified(ast: exp.Expression, schema) -> exp.Expression:
    """The plan with every column attributed to its object, or the plan unchanged.

    Without this the criterion cannot pass an ordinary run. `column_tables` records an entry only
    for a column carrying a qualifier, the approved plan is MODEL-written, and nothing in the
    pipeline runs sqlglot's qualifier -- so a plain `SELECT ssn FROM claim` is unattributable, the
    answer is unmeasurable, and the zero threshold fails a run that was in fact measurable all
    along. The spec names this fix: qualify the approved plan before deriving `touched`.

    It is applied to a COPY and only to derive `touched`. The plan that ran is untouched, and the
    disclosure side keeps reading the tree the engine actually saw.

    Ambiguity is NOT an error the qualifier reports. Measured on the installed sqlglot, none of
    `SELECT bogus.ssn FROM claim`, an unqualified `ssn` across two tables that BOTH carry one, or
    an unknown column raises: each is left bare or left as written, so `column_tables` records
    nothing and the reference reports unattributable. That is the outcome this wants -- an
    ambiguous masked reference must not be silently assigned to one of its candidates -- and it is
    why the except clause below is defensive rather than the mechanism. No naturally-parsed shape
    was found to reach it.

    A failure there leaves the plan alone rather than guessing, which lands in the same place:
    unattributable, run invalid, loud. That is the direction that does not score an under-firing
    build green.
    """
    if not schema:
        return ast
    try:
        return qualify_columns(
            ast.copy(),
            MappingSchema({t: {c: "UNKNOWN" for c in cols} for t, cols in schema.items()}),
            infer_schema=True,
        )
    except (OptimizeError, KeyError, ValueError):
        return ast


def touched(ast: exp.Expression, policy: AccessPolicy, schema=None) -> Touched:
    """What the policy narrows in what the plan REFERENCED.

    `schema` is the caller's visible map -- the same one the disclosure path gets. Omitting it
    measures the plan exactly as written, which reports every unqualified masked reference as
    unattributable.

    "Referenced" is load-bearing and a draft of the spec dropped it, which inverted the criterion:
    without it `touched` becomes every masked pair whose table merely appears, so
    `SELECT a.email FROM a JOIN b` with only `b.email` masked marks `b` touched -- matching the
    un-tightened loop's over-fire while a correct build discloses nothing for `b`. That is the one
    mutation this criterion exists to catch, passing the defect and failing the fix.

    Object spellings diverge by construction, and each axis is folded the way the disclosure path
    itself folds it, so that both sides name an object identically and can then be compared EXACTLY:

      * FILTER -- `build_access_policy` copies grant keys through verbatim, so a policy keyed
        `CLAIM` over a snapshot keyed `claim` must still match. Matched case-insensitively, which is
        what `row_filter_for` does, rather than normalising a key to a snapshot id.
      * MASK -- `policy.masked` holds `c.object_id` verbatim while `masked_by_table` folds both
        halves, so membership is tested folded on BOTH axes.

    In every case the entry carries the PLAN's spelling (`object_key`), which is what the disclosed
    side reports, so the comparison itself stays exact on the object.
    """
    entries: set[Entry] = set()
    ast = _qualified(ast, schema)

    for name in {object_key(t) for t in base_tables(ast)}:
        if policy.row_filter_for(name) is not None:
            entries.add(("filter", name))

    masked_cols: dict[str, set[str]] = {}
    for obj, col in policy.masked:
        masked_cols.setdefault(obj.lower(), set()).add(col.lower())
    every_masked_name = {c for cols in masked_cols.values() for c in cols}

    # Attribution is keyed on `column_tables` AND ON NOTHING ELSE. `_candidate_tables` must not be
    # the instrument: it is a fail-CLOSED helper built for REFUSING, and fail-closed is the wrong
    # shape for measuring -- refusing wants a superset of what might be touched, measuring wants
    # exactly what was. On `SELECT bogus.ssn FROM claim` it answers `{claim}`, a definite single
    # object, where `column_tables` records nothing; keying on it would mark the reference
    # attributable and score the answer green unmeasured.
    owners = column_tables(ast)
    unattributable: list[str] = []
    for column in ast.find_all(exp.Column):
        name = column.name.lower()
        if name not in every_masked_name:
            continue
        owner = None if owners is None else owners.get(id(column))
        if owner is None:
            # NOT treated as touching every candidate, and NOT silently skipped. Skipping scores
            # an UNDER-firing build `∅ == ∅` green on this answer -- the more serious direction --
            # and widening it invents an entry the resolver cannot justify. It is a defect in the
            # MEASUREMENT, so it is reported and the threshold on it is zero.
            unattributable.append(column.sql())
            continue
        if name in masked_cols.get(owner.lower(), ()):
            entries.add(("mask", owner))

    return Touched(entries=entries, unattributable=tuple(unattributable))


def disclosed(narrowed) -> set[Entry]:
    """The same entries as the scoped disclosure actually names."""
    out: set[Entry] = set()
    for n in narrowed or []:
        if n.rows:
            out.add(("filter", n.object))
        if n.columns:
            out.add(("mask", n.object))
    return out


@dataclass(frozen=True)
class AnswerCheck:
    """One answer's result. `agrees` is meaningless while `unattributable` is non-empty."""

    agrees: bool
    missing: set[Entry]      # touched, not disclosed -- the silence the build exists to end
    spurious: set[Entry]     # disclosed, not touched -- the notice nobody reads
    unattributable: tuple[str, ...]
    touched: set[Entry]

    @property
    def measurable(self) -> bool:
        return not self.unattributable


def check(ast: exp.Expression, policy: AccessPolicy, narrowed, schema=None) -> AnswerCheck:
    """Both directions are reported rather than a bool alone: a caller that sees only `False`
    cannot tell an under-fire from an over-fire, and they are not the same defect."""
    t = touched(ast, policy, schema)
    d = disclosed(narrowed)
    return AnswerCheck(agrees=t.entries == d, missing=t.entries - d, spurious=d - t.entries,
                       unattributable=t.unattributable, touched=t.entries)


@dataclass(frozen=True)
class RunVerdict:
    """The run-level result. `passes` requires all three conditions, not just agreement."""

    disagreed: int
    unmeasurable: int
    filters_touched: int     # answers whose `touched` holds at least one filter entry
    masks_touched: int       # ...and at least one mask entry

    @property
    def vacuous(self) -> bool:
        """A run that touched no filter, or no mask, proves nothing about the half it missed.

        Without this floor a policy keyed at a table the band never queries -- or a `row_filters`
        key `build_access_policy` drops as unreachable -- yields `∅ == ∅` on every answer, the sets
        "agree", and the caller-facing half ships past a control that could not fail.
        """
        return self.filters_touched == 0 or self.masks_touched == 0

    @property
    def passes(self) -> bool:
        return self.disagreed == 0 and self.unmeasurable == 0 and not self.vacuous


def summarise(checks: list[AnswerCheck]) -> RunVerdict:
    """Aggregate WITHOUT averaging. The counts are reported beside the result so a failure names
    which of the three conditions it failed; a run reporting zero of either kind is not a pass."""
    return RunVerdict(
        disagreed=sum(1 for c in checks if c.measurable and not c.agrees),
        unmeasurable=sum(1 for c in checks if not c.measurable),
        filters_touched=sum(1 for c in checks if any(k == "filter" for k, *_ in c.touched)),
        masks_touched=sum(1 for c in checks if any(k == "mask" for k, *_ in c.touched)),
    )
