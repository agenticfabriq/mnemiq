"""`touched(a) == disclosed(a)` on every governed answer -- the pre-registered criterion.

Not a rate: aggregate rates cancel, one over-fire and one silence summing to green, and both
directions are live at once. So this is a set equality per answer, and the tests below are the
mutations the spec names as the ones it must catch.

The criterion derives `touched` from the plan and the policy, never from `apply_row_and_mask`.
Otherwise it is the disclosure code agreeing with itself, which passes by construction on exactly
the defects it exists to catch.
"""

import sqlglot

from mnemiq.contract.seams import Narrowed
from mnemiq.eval.criterion import AnswerCheck, check, disclosed, summarise, touched
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import apply_row_and_mask

_V = {"claim": {"id", "amount", "ssn"}, "person": {"id", "email"}}


def _end_to_end(sql, policy, visible=None):
    """Run the REAL disclosure path, then check it against an independently derived `touched`."""
    ast = sqlglot.parse_one(sql, read="duckdb")
    _out, narrowed = apply_row_and_mask(ast, policy, visible or _V, dialect="duckdb")
    # `touched` is derived from a FRESH parse of the original plan: `apply_row_and_mask` rewrites
    # in place, and deriving from the rewritten tree would read the fix back to itself.
    return check(sqlglot.parse_one(sql, read="duckdb"), policy, narrowed)


def test_a_filtered_table_agrees():
    c = _end_to_end("SELECT id FROM claim", AccessPolicy(row_filters={"claim": "amount > 0"}))
    assert c.agrees and c.measurable, (c.missing, c.spurious)


def test_a_referenced_masked_column_agrees():
    c = _end_to_end("SELECT claim.ssn FROM claim", AccessPolicy(masked={("claim", "ssn")}))
    assert c.agrees and c.measurable, (c.missing, c.spurious)


def test_both_kinds_at_once_agree():
    c = _end_to_end("SELECT claim.ssn FROM claim",
                    AccessPolicy(row_filters={"claim": "amount > 0"}, masked={("claim", "ssn")}))
    assert c.agrees and c.touched == {("filter", "claim"), ("mask", "claim")}


def test_an_UNREFERENCED_mask_is_touched_by_NEITHER_side():
    """"Referenced" is load-bearing and a draft of the spec dropped it. Without it,
    `SELECT claim.id FROM claim` with `claim.ssn` masked is "touched" while a correct
    implementation narrows nothing -- mismatching on every answer that selects no masked column."""
    policy = AccessPolicy(masked={("claim", "ssn")})
    assert touched(sqlglot.parse_one("SELECT id FROM claim", read="duckdb"), policy).entries == set()
    assert _end_to_end("SELECT id FROM claim", policy).agrees


# the collision shape: BOTH objects carry a column named `email`, only `claim`'s is masked
_COLLIDE = {"claim": {"id", "amount", "email"}, "person": {"id", "email"}}


def test_the_BARE_NAME_COLLISION_the_criterion_exists_to_catch():
    """`SELECT p.email FROM person p JOIN claim c` with only `claim.email` masked -- the spec's own
    example, and the one mutation this criterion exists for.

    A `touched` that matches on the bare NAME reports `claim`, agreeing with the un-tightened
    loop's over-fire and going GREEN on the defect, while a correct build discloses nothing for
    `claim` and goes RED. Attribution is what separates them, which is why `touched` must come from
    `column_tables` and why the name alone is never enough."""
    sql = "SELECT p.email FROM person p JOIN claim c ON p.id = c.id"
    policy = AccessPolicy(masked={("claim", "email")})
    t = touched(sqlglot.parse_one(sql, read="duckdb"), policy)
    assert t.entries == set(), "the masked column belongs to claim; the plan referenced person's"
    assert not t.unattributable, "and the reference is cleanly attributed, so this IS measured"
    c = _end_to_end(sql, policy, _COLLIDE)
    assert c.agrees and c.measurable, (c.missing, c.spurious)


def test_a_collision_on_a_table_that_IS_masked_elsewhere():
    """`SELECT claim.email FROM claim` where `claim.ssn` and `person.email` are masked. The object
    is masked and the name is masked, but not that name ON that object -- so a check that asks only
    "does this object have any mask?" over-fires."""
    sql = "SELECT claim.email FROM claim"
    policy = AccessPolicy(masked={("claim", "ssn"), ("person", "email")})
    assert touched(sqlglot.parse_one(sql, read="duckdb"), policy).entries == set()
    c = _end_to_end(sql, policy, _COLLIDE)
    assert c.agrees and c.measurable, (c.missing, c.spurious)


def test_the_SAME_query_DOES_touch_when_the_pair_really_is_masked():
    """The control for the two above: identical shape, `claim.email` genuinely masked, so a green
    result there is a real negative rather than a check that never fires."""
    sql = "SELECT claim.email FROM claim"
    policy = AccessPolicy(masked={("claim", "email")})
    assert touched(sqlglot.parse_one(sql, read="duckdb"), policy).entries == {("mask", "claim")}
    c = _end_to_end(sql, policy, _COLLIDE)
    assert c.agrees and c.touched == {("mask", "claim")}, (c.missing, c.spurious)


def test_the_KIND_TAG_is_not_decoration():
    """A row-filtered object rendered as a mask produces an identical set on both sides unless the
    kind is carried."""
    as_filter = disclosed([Narrowed(object="claim", rows=True, columns=False)])
    as_mask = disclosed([Narrowed(object="claim", rows=False, columns=True)])
    assert as_filter == {("filter", "claim")} and as_mask == {("mask", "claim")}
    assert as_filter != as_mask


def test_object_spelling_is_folded_for_FILTERS_because_grant_keys_are_verbatim():
    """`build_access_policy` copies grant keys through, so a policy keyed `CLAIM` over a snapshot
    keyed `claim` must still match -- otherwise a correct build is red and the caller-facing half
    is withheld forever."""
    ast = sqlglot.parse_one("SELECT id FROM claim", read="duckdb")
    t = touched(ast, AccessPolicy(row_filters={"CLAIM": "amount > 0"}))
    assert t.entries == {("filter", "claim")}   # the PLAN's spelling, which is what disclosed says


def test_column_case_does_not_split_a_mask():
    """`masked_by_table` folds columns while the policy stores them verbatim, so a snapshot column
    `SSN` would otherwise be red on a correct build."""
    ast = sqlglot.parse_one("SELECT claim.ssn FROM claim", read="duckdb")
    assert touched(ast, AccessPolicy(masked={("claim", "SSN")})).entries == {("mask", "claim")}


# --- attributability: the threshold is zero -------------------------------------------------

def test_an_UNATTRIBUTABLE_reference_is_unmeasurable_not_green():
    """The spec's live read instance. `column_tables` records no entry for `bogus.ssn`, so the
    reference cannot be attributed. Scoring it green would let an UNDER-firing build pass `0 == 0`
    on an answer it never measured -- the more serious direction."""
    ast = sqlglot.parse_one("SELECT bogus.ssn FROM claim", read="duckdb")
    t = touched(ast, AccessPolicy(masked={("claim", "ssn")}))
    assert t.unattributable, "the reference column_tables cannot attribute must be REPORTED"
    assert t.entries == set(), "and must not invent an entry the resolver cannot justify"


def test_candidate_tables_would_have_called_that_reference_attributable():
    """Keying on `_candidate_tables` instead is the mutation the classifier rules out: it is
    fail-CLOSED, built for refusing, and answers with a definite single object here."""
    from mnemiq.sql.cls import _candidate_tables
    from mnemiq.sql.scope import column_tables

    ast = sqlglot.parse_one("SELECT bogus.ssn FROM claim", read="duckdb")
    column = next(c for c in ast.find_all(sqlglot.exp.Column) if c.name == "ssn")
    # called exactly as `check_cls` calls it, so this measures the real helper
    cands = _candidate_tables(column, column_tables(ast), {"claim"}, set())

    assert cands == {"claim"}, "a definite single object -- which is why it looks attributable"
    assert column_tables(ast).get(id(column)) is None, "while the instrument records nothing"
    assert touched(ast, AccessPolicy(masked={("claim", "ssn")})).unattributable, (
        "so keying on _candidate_tables would score this answer green, unmeasured")


def test_an_unmeasurable_answer_does_not_count_as_agreeing():
    c = check(sqlglot.parse_one("SELECT bogus.ssn FROM claim", read="duckdb"),
              AccessPolicy(masked={("claim", "ssn")}), [])
    assert not c.measurable
    assert summarise([c]).unmeasurable == 1
    assert summarise([c]).disagreed == 0, "an unmeasurable answer is not ALSO counted as disagreeing"


def test_a_column_matching_no_mask_never_makes_an_answer_unmeasurable():
    """Only masked-column references matter; an unresolvable ordinary column is not the arm's
    problem, and treating it as one would make every run invalid."""
    t = touched(sqlglot.parse_one("SELECT bogus.id FROM claim", read="duckdb"),
                AccessPolicy(masked={("claim", "ssn")}))
    assert not t.unattributable and t.entries == set()


# --- run-level: the non-vacuity floor -------------------------------------------------------

def _ok(t):
    return AnswerCheck(agrees=True, missing=set(), spurious=set(), unattributable=(), touched=t)


def test_a_run_that_touched_no_mask_is_VACUOUS_not_a_pass():
    """A policy keyed at a table the band never queries yields 0 == 0 on every answer, the sets
    'agree', and the caller-facing half ships past a control that could not fail."""
    v = summarise([_ok({("filter", "claim")}), _ok({("filter", "claim")})])
    assert v.disagreed == 0 and v.unmeasurable == 0
    assert v.vacuous and not v.passes
    assert v.masks_touched == 0 and v.filters_touched == 2


def test_a_run_that_touched_no_filter_is_VACUOUS_too():
    v = summarise([_ok({("mask", "claim")})])
    assert v.vacuous and not v.passes


def test_a_run_touching_BOTH_kinds_and_agreeing_passes():
    v = summarise([_ok({("filter", "claim")}), _ok({("mask", "claim")})])
    assert not v.vacuous and v.passes


def test_ONE_unmeasurable_answer_fails_a_run_that_otherwise_passes():
    """The threshold is zero: an unmeasurable answer is a defect in the MEASUREMENT, and tolerating
    a share of them means tolerating a run scored green on answers it never measured."""
    good = [_ok({("filter", "claim")}), _ok({("mask", "claim")})]
    bad = AnswerCheck(True, set(), set(), ("bogus.ssn",), {("mask", "claim")})
    assert summarise(good).passes
    assert not summarise([*good, bad]).passes


def test_it_reports_WHICH_DIRECTION_failed():
    """Under-firing is the silence this build exists to end; over-firing is the notice nobody
    reads. A bare False cannot tell them apart."""
    ast = sqlglot.parse_one("SELECT id FROM claim", read="duckdb")
    under = check(ast, AccessPolicy(row_filters={"claim": "amount > 0"}), [])
    assert not under.agrees and under.missing == {("filter", "claim")} and under.spurious == set()

    over = check(ast, AccessPolicy(), [Narrowed(object="claim", rows=True, columns=False)])
    assert not over.agrees and over.spurious == {("filter", "claim")} and over.missing == set()
    assert summarise([under, over]).disagreed == 2, "two defects must not cancel to green"
