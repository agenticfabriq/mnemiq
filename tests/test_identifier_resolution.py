"""One dialect-aware identifier resolution, used by every resolver (M55/M79).

Every resolver in `sql/` has to decide whether two spellings name one object. Each one that
answered independently answered differently: `guard.py` had it wrong twice in consecutive commits,
and `scope.py` ignored quoting entirely, which let a quoted CTE hide a base table from the grant
check. `qualify.names_one_object` names this as the fix in its own docstring.

The through-line these tests exist for: a name matched more WIDELY than the engine matches it lets
a local alias stand in for a real table, and nothing downstream can catch that, because by then the
table is not in the list.
"""

import pytest
import sqlglot
from sqlglot import exp

from mnemiq.sql.decide import decide
from mnemiq.sql.identifiers import resolve, resolve_name, resolve_stored, rules_for
from mnemiq.sql.scope import base_tables

_VISIBLE = {"policy": {"id"}}

# quoted CTE / bare reference -- one object where quoting folds, two where it is preserved
_LEAK = 'WITH "Claim" AS (SELECT id FROM policy) SELECT id FROM Claim'
_MIRROR = 'WITH Claim AS (SELECT id FROM policy) SELECT id FROM "Claim"'
# already-lowercase quoted CTE: one object under a DOWN fold, two under Oracle's UP fold
_ORACLE_ONLY = 'WITH "claim" AS (SELECT id FROM policy) SELECT id FROM claim'


def _verdict(sql, engine):
    return type(decide(sql, _VISIBLE, target=engine, dialect=engine)).__name__


def test_a_quoted_CTE_no_longer_hides_a_base_table_from_the_grant_check():
    """M79. `decide` returned `Approved(tables=['policy'])` for a statement whose reference
    Postgres resolves to base table `claim` -- while `SELECT id FROM Claim` ALONE is refused
    `unauthorized_table`. The CTE's mere presence was what unlocked it, because `scope.sources` is
    keyed by bare text and the engine is not."""
    assert _verdict("SELECT id FROM Claim", "postgres") == "Refusal", "control: no CTE, no read"
    for engine in ("postgres", "oracle"):
        assert _verdict(_LEAK, engine) == "Refusal", engine
        assert _verdict(_MIRROR, engine) == "Refusal", engine


def test_it_is_not_refused_where_the_engine_really_does_shadow():
    """DuckDB folds quoted names, so the CTE genuinely shadows and the read never happens.
    Measured on duckdb 1.5.4: `WITH "Claim" AS (SELECT id FROM policy) SELECT id FROM Claim`
    returns the CTE's row. Refusing it there is a false refusal, which is what an
    engine-independent rule cost."""
    for engine in ("duckdb", "sqlite"):
        assert _verdict(_LEAK, engine) == "Approved", engine
        assert _verdict(_MIRROR, engine) == "Approved", engine


def test_the_fold_DIRECTION_changes_the_answer_on_one_engine_only():
    """The row no constant can produce. `"claim"` and `claim` are one object wherever unquoted
    names fold DOWN and two where they fold UP -- `adapters/oracle.py` records that Oracle folds up
    and stores schema names uppercased for exactly this reason."""
    assert _verdict(_ORACLE_ONLY, "oracle") == "Refusal"
    for engine in ("postgres", "duckdb", "sqlite"):
        assert _verdict(_ORACLE_ONLY, engine) == "Approved", engine


@pytest.mark.parametrize("sql", [
    'WITH claim AS (SELECT id FROM policy) SELECT id FROM claim',
    'WITH "claim" AS (SELECT id FROM policy) SELECT id FROM "claim"',
])
def test_an_ordinary_shadow_is_approved_everywhere(sql):
    """The controls. A guard that rejects legitimate SQL is broken, not safe."""
    for engine in ("postgres", "oracle", "duckdb", "sqlite"):
        assert _verdict(sql, engine) == "Approved", f"{engine}: {sql}"


def test_every_resolver_gets_the_SAME_answer():
    """Threading the dialect to SOME call sites is worse than to none, twice measured while
    building this. First: `check_access` called `base_tables` without it and said the name was a
    CTE while `decide`'s audit list said it was a base read -- Approved with `claim` in `tables`
    and no grant covering it. Then: `column_tables` had not learned the fold either, so
    `check_cls` treated a DENIED column's qualifier as a CTE on the statement `base_tables` had
    just called a real read, and the denied column came back Approved.

    So this exercises resolvers that do NOT share a call, which the first version of this test did
    not -- it asked `base_tables` and `check_access`, whose names both come from one
    `base_tables(ast, dialect)`, and stayed green under the mutation it was named for."""
    from mnemiq.sql.authz_guard import check_access
    from mnemiq.sql.cls import check_cls
    from mnemiq.sql.guard import check_shape
    from mnemiq.sql.policy import AccessPolicy
    from mnemiq.sql.scope import column_tables

    denied = AccessPolicy(denied={("claim", "ssn")})
    sql = 'WITH "claim" AS (SELECT id FROM policy) SELECT claim.ssn FROM claim'

    for engine, engine_reads_the_table in (("oracle", True), ("postgres", False), ("duckdb", False)):
        ast = check_shape(sql, dialect=engine, executes_as=engine)

        reads = {t.name for t in base_tables(ast, engine)}                      # audit list
        owners = column_tables(ast, engine) or {}                               # CLS attribution
        attributed = {v for v in owners.values()}
        refused_col = check_cls(ast, denied, engine) is not None                # column guard

        assert ("claim" in reads) is engine_reads_the_table, f"{engine}: base_tables"
        assert ("claim" in attributed) is engine_reads_the_table, f"{engine}: column_tables"
        assert refused_col is engine_reads_the_table, f"{engine}: check_cls disagrees"

        # and the table guard agrees with all three
        assert (check_access(ast, {"policy": {"id"}}, engine) is not None) is engine_reads_the_table


def test_the_WRITE_path_resolves_reads_the_same_way_the_read_path_does():
    """`apply_row_filters_to_write` took `executes_as` and forwarded it to the filter validator but
    not to `apply_row_and_mask`, so the write path governed its reads with the PARSE dialect while
    the read path used the executing one. The write half is the one that copies unfiltered rows
    somewhere durable."""
    import sqlglot

    from mnemiq.sql.decide_write import _target_node
    from mnemiq.sql.policy import AccessPolicy
    from mnemiq.sql.rls import apply_row_filters_to_write

    # `claim` is the CTE under a DOWN fold and the base table under Oracle's UP fold, so the row
    # filter must reach it there and must not here.
    sql = ('INSERT INTO scratch SELECT id, amount FROM '
           '(WITH "claim" AS (SELECT id, amount FROM src) SELECT id, amount FROM claim) t')
    visible = {"claim": {"id", "amount"}, "src": {"id", "amount"}, "scratch": {"id", "amount"}}
    policy = AccessPolicy(row_filters={"claim": "amount > 0"})

    def narrow(executes_as):
        ast = sqlglot.parse_one(sql, read="duckdb")
        out, narrowed = apply_row_filters_to_write(ast, policy, visible, _target_node(ast),
                                                   "duckdb", executes_as=executes_as)
        return {n.object for n in narrowed}, "amount > 0" in str(out)

    assert narrow("oracle") == ({"claim"}, True), "unfiltered rows would be copied into scratch"
    assert narrow("duckdb") == (set(), False), "the CTE really does shadow here"

    # asserted BEHAVIOURALLY rather than by reading the source: an earlier version of this test
    # checked that `executes_as=executes_as` appeared in the function's text, which is true of the
    # filter-validator call in the same function -- so it passed with the read-path call stripped.


def test_an_UNKNOWN_dialect_gets_the_strictest_reading_not_a_guess():
    """Treating quoting as significant can only make two spellings look like two objects, and every
    consumer fails closed on that -- an extra table node reaches the grant check rather than
    slipping past it."""
    assert rules_for("something-new").quoting_preserves_case
    assert resolve("Claim", quoted=True, dialect=None) == "Claim"
    ast = sqlglot.parse_one(_LEAK, read="duckdb")
    assert "Claim" in {t.name for t in base_tables(ast, None)}


def test_a_STORED_name_is_resolved_as_unquoted():
    """A catalog name has already been folded by the database that stored it, so re-folding is what
    makes it comparable to a resolved reference: `CREATE TABLE Claim` is `claim` in Postgres and
    `CLAIM` in Oracle."""
    assert resolve_stored("Claim", "postgres") == "claim"
    assert resolve_stored("Claim", "oracle") == "CLAIM"
    assert resolve_stored("Claim", "duckdb") == "claim"


def test_a_CTE_is_known_by_what_it_DEFINES_and_a_reference_by_what_it_READS():
    """`alias_or_name` on `claim AS c` is the alias `c`, so reading it for both sides made every
    aliased reference miss and fall through to the base-table branch."""
    ast = sqlglot.parse_one("WITH c AS (SELECT id FROM policy) SELECT id FROM c x", read="duckdb")
    cte = next(ast.find_all(exp.CTE))
    ref = next(t for t in ast.find_all(exp.Table) if t.name == "c")
    assert resolve_name(cte, "postgres") == "c"
    assert resolve_name(ref, "postgres") == "c", "the name it READS, not its own alias `x`"
    assert base_tables(ast, "postgres") == [t for t in ast.find_all(exp.Table) if t.name == "policy"]


def test_a_RECURSIVE_cte_self_reference_does_not_fall_through_to_fail_open():
    """The shape a note in `scope.py` claimed could not reach its fail-open branch, which reached
    it the same day.

    A recursive CTE's self-reference binds to a scope whose expression is one BRANCH of the union,
    while the CTE's own body is the whole union -- so matching the expression against every CTE
    node found nothing, `_engine_shadows` returned its fallback, and the base table was dropped
    from the read list. `_defining_identifier` walks UP to the nearest definer instead."""
    leak = ('WITH RECURSIVE "Claim" AS (SELECT id FROM policy UNION ALL SELECT id FROM Claim) '
            'SELECT id FROM "Claim"')
    for engine in ("postgres", "oracle"):
        assert _verdict(leak, engine) == "Refusal", engine
    assert _verdict(leak, "duckdb") == "Approved", "duckdb folds, so the CTE really does shadow"

    # the control: all bare, so every engine shadows it and a legitimate recursive CTE still runs
    control = ('WITH RECURSIVE claim AS (SELECT id FROM policy UNION ALL SELECT id FROM claim) '
               'SELECT id FROM claim')
    for engine in ("postgres", "oracle", "duckdb", "sqlite"):
        assert _verdict(control, engine) == "Approved", engine


def test_a_PARENTHESISED_body_does_not_hide_the_definer():
    """One pair of parentheses turned the M79 refusal back into an Approved.

    A parenthesised CTE body wraps the query in a `Subquery` that NAMES nothing, and the upward
    walk stopped at it and answered "no definer" for a source that plainly has one. Walking past a
    definer with no alias is the close; the shape is otherwise identical to `_LEAK`."""
    for sql in ('WITH "Claim" AS ((SELECT id FROM policy)) SELECT id FROM Claim',
                'WITH RECURSIVE "Claim" AS ((SELECT id FROM policy) UNION ALL '
                '(SELECT id FROM Claim)) SELECT id FROM "Claim"'):
        for engine in ("postgres", "oracle"):
            assert _verdict(sql, engine) == "Refusal", f"{engine}: {sql}"

    # and the parenthesised CONTROLS still run: an ordinary shadow, and a plain derived table
    assert _verdict('WITH claim AS ((SELECT id FROM policy)) SELECT id FROM claim',
                    "postgres") == "Approved"
    assert _verdict("SELECT id FROM ((SELECT id FROM policy)) t", "postgres") == "Approved"


def test_the_measuring_instrument_resolves_the_way_the_measured_path_does():
    """`touched` called `column_tables` with no dialect while the disclosure path called it with
    one, so the criterion disagreed with a CORRECT build: a quoted CTE sharing a masked table's
    name gave `touched={("mask","Claim")}` against an empty `disclosed`. Loud rather than silent,
    and still the defect this criterion exists to catch elsewhere."""
    from mnemiq.eval.criterion import check, summarise
    from mnemiq.sql.policy import AccessPolicy
    from mnemiq.sql.rls import apply_row_and_mask

    visible = {"claim": {"id", "ssn"}, "policy": {"id"}}
    policy = AccessPolicy(masked={("claim", "ssn")})
    sql = 'WITH "Claim" AS (SELECT id FROM policy) SELECT Claim.ssn FROM Claim'

    def agree(sql, policy, visible):
        ast = sqlglot.parse_one(sql, read="duckdb")
        _out, narrowed = apply_row_and_mask(ast, policy, visible,
                                            dialect="duckdb", executes_as="duckdb")
        return check(sqlglot.parse_one(sql, read="duckdb"), policy, narrowed, visible, "duckdb")

    # The MASK half: `touched` reads `column_tables`. What the threading fixed is the spurious
    # ENTRY -- `touched` was naming a mask on a table this answer never reads.
    mask = agree(sql, policy, visible)
    assert mask.touched == set(), mask.touched

    # It is still UNMEASURABLE, and saying only `agrees` here would misreport that: `AnswerCheck`
    # documents `agrees` as meaningless while `unattributable` is non-empty, and the run-level
    # verdict on this answer is red either way. The reason is the spec's own classifier -- a
    # qualifier naming a CTE records nothing in `column_tables`, and `ssn` matches a masked name,
    # so the reference cannot be attributed. That is a limit of the instrument, not a defect in
    # the product, and the spec's remedy for it is to fix the corpus or the harness.
    assert not mask.measurable and mask.unattributable == ("Claim.ssn",)
    assert summarise([mask]).unmeasurable == 1

    # the FILTER half reads `base_tables`, and needs its own case -- the mask case above stays
    # green with the dialect stripped from that call, so it is not a control for it. Without it
    # `touched` gains a spurious ("filter", "Claim") against an empty disclosure.
    filtered = agree('WITH "Claim" AS (SELECT id FROM policy) SELECT id FROM Claim',
                     AccessPolicy(row_filters={"claim": "id > 0"}),
                     {"claim": {"id"}, "policy": {"id"}})
    assert filtered.agrees and filtered.touched == set(), (filtered.missing, filtered.spurious)
    assert filtered.measurable, "the filter half has no masked-column reference to lose"


def test_sqlglot_binds_by_exact_text_which_over_reports_rather_than_under():
    """Recorded, not fixed. `scope.sources` is keyed by the name as typed, so a reference the
    ENGINE would fold onto a CTE -- `WITH "Claim" AS (...) ... FROM claim` on duckdb -- is bound to
    no source and reported as a real read.

    That direction only ever adds a table to the list the grant check walks, so it costs a refusal
    or a redundant mask, never a leak. Closing it means re-binding sources by resolved name instead
    of using sqlglot's, which is a deeper change than the one this file makes."""
    from mnemiq.sql.policy import AccessPolicy

    sql = 'WITH "Claim" AS (SELECT id FROM policy) SELECT claim.ssn FROM claim'
    result = decide(sql, {"claim": {"id", "ssn"}, "policy": {"id"}},
                    target="duckdb", dialect="duckdb",
                    policy=AccessPolicy(masked={("claim", "ssn")}))
    assert type(result).__name__ == "Approved"
    assert "claim" in result.tables, "reported as a read the engine would have shadowed"
    assert result.narrowed, "and masked accordingly -- redundant, not permissive"


def test_a_source_whose_definer_cannot_be_IDENTIFIED_is_checked_not_assumed_local():
    """The third shape to reach the fail-open branch, and the last one chased individually.

    A `VALUES` alias is not reached by walking up from the source's expression at all, so the
    definer came back None and the reference was assumed to be that local alias -- dropping base
    table `Claim` from the read list entirely, `tables=[]` on a statement that names it.

    The fallback answers "not shadowed" now, so an unidentifiable source costs a grant check rather
    than a bypass. Three shapes reached that branch in a day, each under a note saying none could;
    failing closed retires the class instead of the next instance."""
    leak = 'SELECT id FROM (VALUES (1)) AS "Claim"(id), Claim'
    for engine in ("postgres", "oracle"):
        assert _verdict(leak, engine) == "Refusal", engine

    # There is no APPROVING case for this branch to control against, and saying so is more
    # honest than offering one that is vacuous: `SELECT id FROM (VALUES (1)) AS v(id)` names no
    # table at all, so it is Approved under either answer and moves for neither. What the
    # fail-closed choice actually costs is measured by the test below instead.


def test_failing_closed_costs_no_ordinary_query():
    """The whole point of measuring it: a guard that refuses legitimate analytics SQL is broken,
    not safe. These are the shapes that DO resolve a definer, and none of them reaches the
    fallback."""
    for sql in ("SELECT id FROM policy",
                "SELECT id FROM (SELECT id FROM policy) t",
                "WITH c AS (SELECT id FROM policy) SELECT id FROM c",
                "WITH RECURSIVE r AS (SELECT id FROM policy UNION ALL SELECT id FROM r) "
                "SELECT id FROM r",
                "SELECT p.id FROM policy p JOIN (SELECT id FROM policy) q ON p.id = q.id",
                "SELECT id FROM policy, LATERAL (SELECT id FROM policy) y",
                "SELECT id FROM ((SELECT id FROM policy)) x",
                "SELECT id FROM (SELECT id FROM (SELECT id FROM policy) i) o",
                "WITH a AS (SELECT id FROM policy), b AS (SELECT id FROM a) SELECT id FROM b"):
        for engine in ("postgres", "oracle", "duckdb"):
            assert _verdict(sql, engine) == "Approved", f"{engine}: {sql}"

    # These are RUNNABLE queries, and what they measure is the cost: none is refused. The
    # property behind that -- that no ordinary shape reaches the fallback -- is read directly by
    # `test_no_ORDINARY_shape_reaches_the_fail_closed_fallback`, because this loop is not a
    # control for it: it stays green with the fail-open answer restored.

    # Run on ONE sqlglot -- whatever the lock pins -- so this proves nothing about other versions.
    # 25.34.1, 26.16.4, 28.0.0 and 30.12.0 were each installed and probed by hand and each gave
    # zero hits over thirteen shapes; that is a sample of a floor with no ceiling, not the range,
    # and nothing here re-runs it.


def test_no_ORDINARY_shape_reaches_the_fail_closed_fallback():
    """The property the "costs nothing" claim rests on, read directly rather than inferred from
    verdicts: every non-table source a reference binds to has a definer that NAMES it.

    The two halves are counted separately because they are reached by different SQL and were not
    equally covered: an earlier version listed five shapes of which three asserted nothing at all,
    every table node in them binding to an `exp.Table`, so dropping `exp.Subquery` from the walk
    left the whole suite green.

    It is the FALLBACK that ordinary shapes miss, not `_engine_shadows` -- a chained CTE reaches
    the function and gets a real answer from it. A first version conflated those and failed on
    exactly that shape.
    """
    from sqlglot.optimizer.scope import build_scope

    from mnemiq.sql.scope import _defining_identifier

    # NOT executable SQL: a derived-table alias is not a relation in its own FROM, so duckdb says
    # "Table with name x does not exist". They are here because they are the only way to make a
    # REFERENCE bind to a derived-table source, which is the half `exp.Subquery` covers. Their
    # cost is not what this measures -- the sibling test measures that, on runnable queries.
    derived = ("SELECT id FROM (SELECT id FROM policy) x, x",
               "SELECT id FROM ((SELECT id FROM policy)) x, x",
               "SELECT id FROM (SELECT id FROM (SELECT id FROM policy) i) o, o")
    ctes = ("WITH a AS (SELECT id FROM policy), b AS (SELECT id FROM a) SELECT id FROM b",
            "WITH RECURSIVE r AS (SELECT id FROM policy UNION ALL SELECT id FROM r) "
            "SELECT id FROM r")

    def bindings(shapes):
        seen = 0
        for sql in shapes:
            root = build_scope(sqlglot.parse_one(sql, read="postgres"))
            for scope in root.traverse():
                for table in scope.tables:
                    source = scope.sources.get(table.alias_or_name)
                    if source is None or isinstance(source, exp.Table):
                        continue
                    seen += 1
                    assert _defining_identifier(source) is not None, (
                        f"{table.sql()} in {sql} reaches the fail-closed fallback")
        return seen

    # per half, so losing one cannot hide behind the other -- and one shape each, so a deleted
    # `, o` shows up rather than being absorbed by slack
    assert bindings(derived) >= len(derived), "the derived-table half asserted on too few sources"
    assert bindings(ctes) >= len(ctes), "the CTE half asserted on too few sources"

    # Counted on the pinned sqlglot only. `pyproject` declares `>=25.34.1` with no ceiling and
    # this repo already records that scope internals differ across that span, so a version binding
    # these differently fails the count rather than the property it stands for.
