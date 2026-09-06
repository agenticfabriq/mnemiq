import pathlib

import pytest
import sqlglot
from sqlglot import exp

from mnemiq.sql.guard import MAX_ROWS, check_shape
from mnemiq.sql.verdict import Refusal, RefusalCode


def _ok(sql, **kw):
    result = check_shape(sql, **kw)
    assert not isinstance(result, Refusal), result
    return result


def _refused(sql) -> Refusal:
    result = check_shape(sql)
    assert isinstance(result, Refusal), f"should have been refused: {sql}"
    return result


def test_a_plain_select_passes():
    ast = _ok("SELECT claim_identifier FROM claim")
    assert isinstance(ast, exp.Select)


def test_a_missing_limit_is_injected_not_requested():
    ast = _ok("SELECT a FROM claim")
    assert ast.sql(dialect="duckdb").endswith(f"LIMIT {MAX_ROWS}")


def test_an_existing_smaller_limit_is_kept():
    ast = _ok("SELECT a FROM claim LIMIT 5")
    assert ast.sql(dialect="duckdb").endswith("LIMIT 5")


def test_an_oversized_limit_is_clamped():
    ast = _ok("SELECT a FROM claim LIMIT 999999")
    assert ast.sql(dialect="duckdb").endswith(f"LIMIT {MAX_ROWS}")


def test_ctes_and_unions_are_allowed():
    _ok("WITH x AS (SELECT a FROM claim) SELECT a FROM x")
    _ok("SELECT a FROM claim UNION SELECT b FROM policy")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE claim",
        "INSERT INTO claim VALUES (1)",
        "UPDATE claim SET a = 1",
        "DELETE FROM claim",
        "CREATE TABLE t (a INT)",
        "ATTACH 'evil.db'",
    ],
)
def test_anything_that_is_not_a_select_is_refused(sql):
    assert _refused(sql).code == RefusalCode.NOT_SELECT_ONLY


def test_a_trailing_statement_cannot_smuggle_a_drop():
    # sqlglot.parse_one does NOT raise here -- it returns a Block. "It parsed" is not safety.
    refusal = _refused("SELECT a FROM claim; DROP TABLE claim")
    assert refusal.code == RefusalCode.NOT_A_SINGLE_STATEMENT


def test_select_star_is_refused():
    # a star yields no Column nodes at all, so column-level authorization would silently
    # pass over it. The engine must know exactly which columns it returns.
    assert _refused("SELECT * FROM claim").code == RefusalCode.SELECT_STAR
    assert _refused("SELECT c.* FROM claim c").code == RefusalCode.SELECT_STAR


def test_an_inner_star_is_allowed_because_the_output_columns_are_still_explicit():
    # what must be knowable is the set of columns leaving the engine. An EXISTS discards its
    # projection, and a star inside a derived table is bounded by the explicit projection
    # that wraps it. Rejecting these would refuse 22 of the 99 TPC-DS queries.
    _ok("SELECT a FROM claim WHERE EXISTS (SELECT * FROM policy WHERE policy.a = claim.a)")
    _ok("SELECT a FROM (SELECT * FROM claim)")
    _ok("WITH x AS (SELECT * FROM claim) SELECT a FROM x")


def test_a_star_over_a_derived_table_is_allowed_its_columns_are_known():
    # the inner projection is explicit, so the output columns are fully determined
    _ok("SELECT * FROM (SELECT claim_identifier, status FROM claim)")
    _ok("WITH x AS (SELECT a FROM claim) SELECT * FROM x")


def test_a_star_over_a_base_table_is_refused_even_behind_a_join():
    # `claim` is a real table: its column set is unbounded, and we would not know what we
    # are returning -- nor could column-level authorization check it
    assert _refused("SELECT c.* FROM claim c JOIN (SELECT a FROM policy) x ON TRUE").code == (
        RefusalCode.SELECT_STAR
    )


def test_a_star_in_a_union_branch_is_still_refused():
    # every branch of a union returns columns to the caller
    assert _refused("SELECT a FROM claim UNION SELECT * FROM policy").code == (
        RefusalCode.SELECT_STAR
    )


def test_count_star_is_not_a_select_star():
    # COUNT(*) contains an exp.Star. A naive find_all(exp.Star) would reject the most common
    # analytics query there is, and the engine could not count anything.
    _ok("SELECT count(*) FROM claim")
    _ok("SELECT status, count(*) AS n FROM claim GROUP BY status")
    _ok("SELECT count(*) FROM claim WHERE status IN (SELECT status FROM policy)")


def test_unparseable_sql_is_refused_not_raised():
    assert _refused("this is not sql !!").code == RefusalCode.PARSE_ERROR


def test_a_refusal_message_tells_the_model_how_to_repair():
    assert "explicit" in _refused("SELECT * FROM claim").message.lower()


def test_a_WRAPPER_does_not_make_a_base_table_star_explicit():
    """The exemption for a derived table or CTE reads "its projection is explicit", and that is a
    claim about the inner projection -- true of `(SELECT a, b FROM t)`, false of
    `(SELECT * FROM t)`. Answering it by looking ONE level down let a base-table star through
    inside a wrapper, so the output column set was unbounded after all.

    The wrappers below are the ones measured to reach a base table on this sqlglot: a derived
    table, a CTE, either nested; a chain of CTEs; and every set operation, not only UNION --
    EXCEPT and INTERSECT are SIBLINGS of `exp.Union`, so keying on Union read them as having no
    output projection at all. Redundant parentheses nest `Subquery` inside `Subquery`, which the
    same reading skipped."""
    for sql in (
        "SELECT * FROM (SELECT * FROM claim) t",
        "WITH c AS (SELECT * FROM claim) SELECT * FROM c",
        "WITH c AS (SELECT * FROM claim) SELECT c.* FROM c",
        "SELECT * FROM (SELECT * FROM (SELECT * FROM claim) a) b",
        "WITH a AS (SELECT * FROM claim), b AS (SELECT * FROM a) SELECT * FROM b",
        "SELECT * FROM (SELECT * FROM claim UNION ALL SELECT * FROM claim) t",
        "SELECT * FROM (SELECT * FROM claim EXCEPT SELECT * FROM claim) t",
        "SELECT * FROM (SELECT * FROM claim INTERSECT SELECT * FROM claim) t",
        "SELECT * FROM ((SELECT * FROM claim)) t",
        "SELECT * FROM ((SELECT * FROM claim UNION ALL SELECT * FROM claim)) t",
    ):
        assert _refused(sql).code == RefusalCode.SELECT_STAR, sql


def test_an_INNER_cte_does_not_vouch_for_an_OUTER_star_of_the_same_name():
    """CTE bodies are resolved lexically. A flat `name -> body` map across the whole statement
    lets the inner explicit `c` overwrite the outer star `c`, so the outer star is vouched for by
    a projection that is not its own -- the defect `cls.py` records fixing in the CLS resolver,
    where one alias meant two tables and the flat map kept whichever was seen last."""
    assert _refused(
        "WITH c AS (SELECT * FROM claim) SELECT * FROM c "
        "WHERE id IN (WITH c AS (SELECT id FROM policy) SELECT id FROM c)"
    ).code == RefusalCode.SELECT_STAR

    # the control: same shape, outer body explicit, still permitted
    _ok("WITH c AS (SELECT id FROM claim) SELECT * FROM c "
        "WHERE id IN (WITH c AS (SELECT id FROM policy) SELECT id FROM c)")


def test_a_cte_body_is_read_in_the_scope_it_was_WRITTEN_in():
    """Merging the REFERENCE site's names into a body's scope is the same defect one level up.
    A body must be interpreted where it was written, and two rules follow that no reference-site
    map can express -- both measured returning every column of `claim`, `ssn` included."""
    # a non-recursive CTE cannot see a LATER sibling, so `claim` inside `a` is the base table
    assert _refused(
        "WITH a AS (SELECT * FROM claim), claim AS (SELECT id FROM policy) SELECT * FROM a"
    ).code == RefusalCode.SELECT_STAR

    # an inner WITH at the reference site cannot rebind a name inside an outer CTE's body
    assert _refused(
        "WITH a AS (SELECT * FROM claim) "
        "SELECT * FROM (WITH claim AS (SELECT id FROM policy) SELECT * FROM a) z"
    ).code == RefusalCode.SELECT_STAR

    # a PRECEDING sibling is visible, which is the half that must keep working
    _ok("WITH claim2 AS (SELECT id FROM policy), a AS (SELECT * FROM claim2) SELECT * FROM a")

    # and a NON-recursive CTE cannot see itself, so `c` inside it is a base table named `c`.
    # Binding it to itself would let the cycle guard vouch for the star: the walk would meet the
    # body already on its own stack, answer "no base table reached", and permit it.
    assert _refused("WITH c AS (SELECT * FROM c) SELECT * FROM c").code == (
        RefusalCode.SELECT_STAR)
    # WITH RECURSIVE genuinely does bind the name, and this shape reaches no base table at all
    _ok("WITH RECURSIVE c AS (SELECT * FROM c) SELECT * FROM c")


def test_a_QUALIFIED_name_is_never_a_cte_reference():
    """A CTE reference is a bare name. Matching on the bare name alone let a schema-qualified base
    table borrow an unrelated CTE's explicit projection -- the guard resolved `main.claim` to the
    CTE while the database resolved it to the table, and every column came back."""
    for sql in ("WITH claim AS (SELECT id FROM policy) SELECT * FROM main.claim",
                "WITH claim AS (SELECT id FROM policy) SELECT * FROM db.main.claim"):
        assert _refused(sql).code == RefusalCode.SELECT_STAR, sql

    # Neither case reaches the `catalog` half of that check: sqlglot parses a three-part name as
    # db='main', catalog='db', so `db` alone already refuses both. The only shape setting catalog
    # with an empty db is `db..claim`, which DuckDB rejects at parse time -- so the clause is
    # correct, unreachable today, and NOT covered here despite this test's name. (`db.""."claim"`
    # parses that way too; DuckDB rejects it as a zero-length delimited identifier.)
    parsed = sqlglot.parse_one("SELECT * FROM db.main.claim", read="duckdb")
    source = (parsed.args.get("from") or parsed.args.get("from_")).this
    assert (source.db, source.catalog) == ("main", "db")

    # the control: unqualified, so it really is the CTE, and its projection really is explicit
    _ok("WITH claim AS (SELECT id FROM policy) SELECT * FROM claim")


def test_a_cte_name_matches_what_it_RESOLVES_to_not_how_it_was_typed():
    """A CTE vouches for a star, so the name it is matched on must be the one the engine will
    resolve. Both directions of getting that wrong are live and only one of them is a leak.

    TOO WIDE: matching the text alone let `WITH "Claim" AS (...) SELECT * FROM Claim` match, while
    Postgres folds the reference to `claim` and resolves it to the base table -- whose columns then
    reached no grant check. `decide` approved exactly that with `tables=['policy']`.

    TOO NARROW: requiring identical QUOTING then refused `WITH "claim" AS (...) SELECT * FROM
    claim`, which is one object in Postgres and is the ordinary shape of a model quoting a
    definition but not its reference.

    Whether these are one object or two is the DIALECT's answer, not a constant -- Postgres and
    Oracle preserve a quoted name, DuckDB folds it. Measured on duckdb 1.5.4:
    `WITH "Claim" AS (SELECT id FROM policy) SELECT id FROM Claim` returns the CTE's row, so
    refusing it there was a false refusal, which is what an engine-independent rule cost."""
    for sql in ('WITH "Claim" AS (SELECT id FROM policy) SELECT * FROM Claim',
                'WITH Claim AS (SELECT id FROM policy) SELECT * FROM "Claim"'):
        for engine in ("postgres", "oracle"):
            assert isinstance(check_shape(sql, executes_as=engine), Refusal), f"{engine}: {sql}"
        _ok(sql)  # duckdb folds quoted names, so the CTE really does shadow

    # And it must not over-refuse, which is the other direction and equally live. Everything below
    # is ONE object under the default `duckdb` these run on, and under every DOWN-folding engine --
    # NOT under Oracle, where the mixed-quoting cases are two objects and are refused; the
    # dialect test below asserts that, and it is the rule rather than a bug. Refusing these would
    # reject ordinary model-written SQL, which is what makes a guard broken rather than safe:
    # quoting a definition but not its reference is the common LLM shape, and a bare case mismatch
    # was refused even before quoting entered the key.
    _ok("WITH claim AS (SELECT id FROM policy) SELECT * FROM claim")
    _ok('WITH "claim" AS (SELECT id FROM policy) SELECT * FROM "claim"')
    _ok('WITH "claim" AS (SELECT id FROM policy) SELECT * FROM claim')
    _ok('WITH claim AS (SELECT id FROM policy) SELECT * FROM "claim"')
    _ok("WITH claim AS (SELECT id FROM policy) SELECT * FROM CLAIM")
    _ok("WITH Sales AS (SELECT id FROM policy) SELECT * FROM sales")

    # and a CTE reference carrying its own alias still resolves by the name it READS: `c x` is a
    # reference to `c`, not to `x`, and reading `alias_or_name` for both sides missed every one
    _ok("WITH c AS (SELECT id FROM claim) SELECT * FROM c x")


def test_the_fold_direction_belongs_to_the_dialect_that_will_RUN_it():
    """Oracle folds unquoted identifiers UP where the others fold them down -- `adapters/oracle.py`
    states it, `oracle` is a shipped source kind, and `check_shape` is called with the adapter's
    own dialect. So one hardcoded direction is wrong for one of them in the direction that VOUCHES,
    and the same statement has to get opposite answers.

    `qualify.py` scopes its case rule to "all three engines". Oracle is the fourth."""
    quoted_lower = 'WITH "claim" AS (SELECT id FROM policy) SELECT * FROM claim'
    # down-folding engines: the bare reference lands on `claim`, which IS the quoted CTE
    for dialect in ("postgres", "duckdb", "sqlite"):
        _ok(quoted_lower, dialect=dialect)
    # Oracle folds it to `CLAIM`, a different object -- the base table, whose star must be refused
    assert isinstance(check_shape(quoted_lower, dialect="oracle"), Refusal)

    # The MIRROR case, and only the Oracle half of it is new -- the duckdb half is asserted in the
    # test above. It pins the behaviour on both sides of the fold rather than leaving the mirror
    # to be inferred from `quoted_lower`; several mutations flip it, and no claim is made here
    # about which of them it alone catches. Two earlier attempts at that claim were wrong.
    mirror = 'WITH claim AS (SELECT id FROM policy) SELECT * FROM "claim"'
    assert isinstance(check_shape(mirror, dialect="oracle"), Refusal)

    quoted_upper = 'WITH "CLAIM" AS (SELECT id FROM policy) SELECT * FROM claim'
    assert isinstance(check_shape(quoted_upper, dialect="postgres"), Refusal)
    _ok(quoted_upper, dialect="oracle")


def test_the_verdict_does_not_depend_on_UNION_BRANCH_ORDER():
    """The memo keys on node id, which is sound only because each node now sits in exactly one
    scope. While a body borrowed its caller's names, one branch's answer was cached for the other
    and the same two branches gave opposite verdicts depending on which came first."""
    rebound = "SELECT * FROM (WITH claim AS (SELECT id FROM policy) SELECT * FROM a) z"
    for query in (f"WITH a AS (SELECT * FROM claim) SELECT * FROM a UNION ALL {rebound}",
                  f"WITH a AS (SELECT * FROM claim) {rebound} UNION ALL SELECT * FROM a"):
        assert _refused(query).code == RefusalCode.SELECT_STAR, query


def test_a_shape_the_walker_cannot_decompose_is_refused_rather_than_permitted():
    """Fail-closed starts one level down. `_output_selects` returns `[]` for anything it does not
    recognise, and reading that as "no star reaches a base table" made every gap in it a silent
    hole -- which is exactly how EXCEPT and doubled parentheses got through."""
    from mnemiq.sql.guard import _star_reaches_base
    assert _star_reaches_base(
        exp.Anonymous(this="opaque"), {}, {}, None, frozenset(), "duckdb"
    ) is True


def test_the_recursion_stays_linear_in_wrapper_depth():
    """Each CTE here is referenced twice, so a walker that re-walks per path is exponential in
    depth. Measured before memoisation: 8s on ~1.2KB of SQL, spent in a pre-execution gate fed
    model-written queries."""
    import time

    # Each level must project a STAR over two references. A chain projecting `a.id` never enters
    # the recursion at all -- no star, nothing to follow -- so it times O(1) whatever the depth
    # and is not a control for this.
    # Depth is chosen so the FAILING run is affordable. Without the memo this doubles per level:
    # ~2s at 20 and ~8s at 22, but ~36 MINUTES at 30 -- which in a pre-commit suite reads as a
    # hang rather than as the failure this test names, so the assertion never gets to print.
    depth = 22
    ctes = ["c0 AS (SELECT id FROM claim)"] + [
        f"c{i} AS (SELECT * FROM c{i-1} a JOIN c{i-1} b ON a.id = b.id)"
        for i in range(1, depth)
    ]
    sql = "WITH " + ", ".join(ctes) + f" SELECT * FROM c{depth - 1}"
    started = time.perf_counter()
    _ok(sql)
    assert time.perf_counter() - started < 1.0, "exponential re-walk is back"


def test_the_wrapper_exemption_still_holds_when_the_projection_really_is_explicit():
    """The control for the above. Refusing these would refuse 22 of the 99 TPC-DS queries, so a
    guard that closed the hole by rejecting every wrapped star would be broken, not safe."""
    _ok("SELECT * FROM (SELECT id, amount FROM claim) t")
    _ok("WITH c AS (SELECT id FROM claim) SELECT * FROM c")
    _ok("SELECT a.* FROM (SELECT id, amount FROM claim) a JOIN policy p ON a.id = p.id")
    _ok("SELECT id FROM (SELECT * FROM claim) t")          # inner star, explicit output
    _ok("SELECT count(*) FROM claim")
    _ok("SELECT id FROM claim WHERE EXISTS (SELECT * FROM policy)")

    # The wrappers the walker learned to follow must not become blanket refusals. Fail-closed
    # covers a shape it cannot read, so WITHOUT these the safety tests above still pass while
    # ordinary analytics SQL is rejected -- the failure mode that makes a guard broken, not safe.
    _ok("SELECT * FROM ((SELECT id FROM claim)) t")
    _ok("SELECT * FROM (SELECT id FROM claim EXCEPT SELECT id FROM claim) t")
    _ok("SELECT * FROM (SELECT id FROM claim INTERSECT SELECT id FROM claim) t")
    _ok("SELECT * FROM (SELECT id FROM claim UNION ALL SELECT id FROM claim) t")


def test_a_recursive_cte_terminates():
    """The recursion follows CTE bodies, and a recursive CTE names itself.

    The shape has to contain a STAR over the self-reference to reach the cycle at all: a
    recursive CTE projecting named columns returns before recursing, so it is not a control.
    Without the cycle guard these raise RecursionError instead of answering."""
    _ok("WITH RECURSIVE r AS (SELECT id FROM claim UNION ALL SELECT * FROM r) SELECT * FROM r")
    _ok("WITH RECURSIVE r AS (SELECT * FROM r) SELECT * FROM r")


def test_a_DENIED_column_cannot_come_back_through_a_wrapped_star():
    """The bypass this closes, asserted where it matters: on what the caller receives.

    `check_cls` and `referenced_masked` both walk `find_all(exp.Column)`, and a star produces no
    `exp.Column` -- so reaching a column BY NAME was refused while reaching the same column BY
    STAR returned it in full. Masking behaved the same way, silently and with no disclosure.
    """
    from mnemiq.sql.cls import check_cls
    from mnemiq.sql.policy import AccessPolicy
    from mnemiq.sql.rls import apply_row_and_mask

    visible = {"claim": {"id", "amount", "ssn"}}
    named = "SELECT claim.ssn FROM claim"
    wrapped = "SELECT * FROM (SELECT * FROM claim) t"

    parsed = sqlglot.parse_one(wrapped, read="duckdb")

    for policy, expect in ((AccessPolicy(denied={("claim", "ssn")}), "denied"),
                           (AccessPolicy(masked={("claim", "ssn")}), "masked")):
        # NAMED: refused outright, or rewritten so the value cannot leave
        ast = _ok(named)
        if expect == "denied":
            assert check_cls(ast, policy) is not None
        else:
            out, narrowed = apply_row_and_mask(ast, policy, visible, dialect="duckdb")
            assert "NULL AS ssn" in out.sql() and narrowed

        # WRAPPED: every downstream control is blind to it, measured rather than assumed --
        # this is why the guard is the only thing standing between the caller and the value.
        assert check_cls(parsed, policy) is None, "check_cls sees no column to refuse"
        out, narrowed = apply_row_and_mask(parsed.copy(), policy, visible, dialect="duckdb")
        assert "NULL AS ssn" not in out.sql(), "no mask is applied"
        assert narrowed == [], "and nothing is disclosed, so the caller is not even told"

        # so the guard must refuse it, or those real `ssn` values reach the caller
        assert isinstance(check_shape(wrapped), Refusal), (
            f"a {expect} column came back through a star the guard permitted")


def test_COLUMNS_is_the_other_spelling_of_a_star():
    """DuckDB's `COLUMNS(*)` expands to a column set exactly as `*` does, and is not an
    `exp.Star`, so it parsed straight past a check that only knew that type.

    Measured on duckdb 1.5.4: `SELECT COLUMNS(*) FROM claim` returned every column including a
    DENIED one, and `SELECT COLUMNS('s.*') FROM claim` returned ONLY `ssn` -- the denied value
    exfiltrated without its name appearing anywhere in the query, which is what makes the regex
    form worse than the plain star rather than a variant of it.
    """
    for sql in ("SELECT COLUMNS(*) FROM claim",
                "SELECT columns(*) FROM claim",
                "SELECT COLUMNS('s.*') FROM claim",
                "SELECT min(COLUMNS(*)) FROM claim",           # still one column per column
                "SELECT claim.COLUMNS(*) FROM claim",          # Dot(Identifier, Anonymous)
                "SELECT claim.columns(*) FROM claim",           # sqlglot PRESERVES this case
                "SELECT claim.Columns(*) FROM claim",
                "SELECT c.COLUMNS('s.*') FROM claim c",
                "SELECT * FROM (SELECT COLUMNS(*) FROM claim) t",
                "WITH c AS (SELECT COLUMNS(*) FROM claim) SELECT * FROM c",
                "SELECT id FROM claim UNION ALL SELECT COLUMNS(*) FROM claim"):
        assert _refused(sql).code == RefusalCode.SELECT_STAR, sql


def test_the_qualified_form_carries_no_Columns_NODE_at_all():
    """Why the check matches on the name and not only on the type: an unqualified `COLUMNS(*)`
    parses to `exp.Columns`, a qualified one to `Dot(Identifier, Anonymous(this="COLUMNS"))` with
    no `exp.Columns` anywhere in it."""
    bare = sqlglot.parse_one("SELECT COLUMNS(*) FROM claim", read="duckdb").expressions[0]
    qualified = sqlglot.parse_one("SELECT claim.COLUMNS(*) FROM claim", read="duckdb").expressions[0]

    assert isinstance(bare, exp.Columns)
    assert not list(qualified.find_all(exp.Columns)), "a type check alone cannot see this one"
    assert any(isinstance(n, exp.Anonymous) and str(n.this).upper() == "COLUMNS"
               for n in qualified.walk())


def test_COUNT_STAR_is_not_an_expansion_and_stays_permitted():
    """The control. Searching for a bare `exp.Star` would be simpler and would refuse the most
    common analytics query there is -- `COUNT(*)` contains one and collapses it to ONE column.
    What is refused is the expansion, not the asterisk."""
    _ok("SELECT count(*) FROM claim")
    _ok("SELECT COUNT(*) FROM claim c")
    _ok("SELECT id, count(*) FROM claim GROUP BY id")
    _ok("SELECT id FROM claim WHERE EXISTS (SELECT * FROM policy)")


def test_naming_a_SOURCE_is_the_same_expansion_without_a_star():
    """DuckDB resolves a bare reference to a source name as the whole ROW. Measured on 1.5.4:
    `SELECT claim FROM claim` returns a struct holding every value including a denied `ssn`, and
    `SELECT UNNEST(claim) FROM claim` spreads it back into `id, amount, ssn`.

    There is no star anywhere in either, and `check_cls` finds no `exp.Column` for `ssn` -- the
    column is never spelled, exactly as with a star."""
    for sql in ("SELECT UNNEST(claim) FROM claim",
                "SELECT claim FROM claim",
                "SELECT unnest(c) FROM claim c",
                "SELECT c FROM claim c",
                "SELECT CLAIM FROM claim",                # duckdb folds; an exact match misses it
                "SELECT UNNEST(CLAIM) FROM claim",
                "SELECT * FROM (SELECT UNNEST(claim) FROM claim) t",
                # ALIASED and carried out through an ordinary-looking projection. An explicit
                # outer projection bounds the column COUNT and not the column SET once one item
                # is a whole row, so scanning only OUTPUT selects left every one of these live.
                "SELECT x FROM (SELECT claim AS x FROM claim) t",
                "SELECT UNNEST(x) FROM (SELECT claim AS x FROM claim) t",
                "SELECT (SELECT c FROM claim c LIMIT 1) AS x FROM policy",
                "WITH w AS (SELECT claim AS x FROM claim) SELECT x FROM w"):
        assert _refused(sql).code == RefusalCode.SELECT_STAR, sql


def test_unnesting_a_COLUMN_is_untouched():
    """The control, and the reason this is keyed on naming a SOURCE rather than on `UNNEST`:
    unnesting a LIST column expands ROWS and returns one column, which is ordinary SQL."""
    _ok("SELECT UNNEST(tags) FROM claim")
    _ok("SELECT a.id FROM claim a JOIN policy p ON a.id = p.id")
    # A DERIVED TABLE source is bounded by its OWN projection, so naming it returns known columns.
    # `SELECT id` is load-bearing here: with `SELECT *` inside, this same query must be refused,
    # which the test below asserts. One token separates the control from the leak.
    _ok("SELECT t FROM (SELECT id FROM claim) t")

    # A QUALIFIED reference names a column even when the column shares its table's name, so the
    # check keys on the reference being BARE. Without that it refuses ordinary SQL: a `policy`
    # table with a `policy` column is not an exotic schema.
    _ok("SELECT p.policy FROM policy p")
    _ok("SELECT policy.policy FROM policy")
    _ok("SELECT claim.claim FROM claim")


def test_naming_a_derived_table_is_bounded_by_ITS_projection_not_by_being_a_wrapper():
    """The row check SKIPS a derived-table source and the star walk handles it, because the row
    check looks at `exp.Column` nodes and a star inside the derived table yields none.

    Getting that wrong reopened a closed hole for one commit: with the row check claiming to cover
    subqueries and the star walk no longer entered for a row reference, `SELECT UNNEST(t) FROM
    (SELECT * FROM claim) t` returned `(1, 100, '999-11-2222')` and `check_cls` returned None. One
    token separates it from the permitted control above."""
    for sql in ("SELECT t FROM (SELECT * FROM claim) t",
                "SELECT UNNEST(t) FROM (SELECT * FROM claim) t",
                "SELECT t FROM ((SELECT * FROM claim)) t",
                "SELECT t FROM (SELECT COLUMNS(*) FROM claim) t",
                "SELECT t FROM (SELECT claim AS x FROM claim) t"):
        assert _refused(sql).code == RefusalCode.SELECT_STAR, sql

    # and the control, whose inner projection really is explicit
    _ok("SELECT t FROM (SELECT id FROM claim) t")
    _ok("SELECT UNNEST(t) FROM (SELECT id FROM claim) t")


def test_three_deliberate_false_refusals_of_this_check():
    """All fail-CLOSED, all recorded rather than discovered later.

    A CTE projected as a struct IS bounded by its own explicit projection, so refusing it is
    stricter than necessary -- taken because the alternative is resolving CTE scopes a second way
    inside this check, and this file has already had that rule wrong twice.

    A BARE reference cannot be told apart from a same-named COLUMN without a schema, which
    `check_shape` does not have: duckdb resolves `SELECT policy FROM policy` to the column when one
    exists, and this refuses it. The qualified spelling above is the one that keeps working.

    And scanning every select reaches DISCARDED projections, so a row reference inside an EXISTS
    is refused although nothing it projects leaves the engine. Scoping the scan to output selects
    is what let the aliased row-struct through, so this is the cost of reaching that."""
    assert _refused("WITH c AS (SELECT id FROM claim) SELECT c FROM c").code == (
        RefusalCode.SELECT_STAR)
    assert _refused("SELECT policy FROM policy").code == RefusalCode.SELECT_STAR
    # the same schema ambiguity wearing an ALIAS rather than a table name
    assert _refused("SELECT amount FROM claim amount").code == RefusalCode.SELECT_STAR
    # and the third: scanning EVERY select means a DISCARDED projection is scanned too, so a row
    # reference inside an EXISTS is refused although it never leaves the engine -- the one
    # position the star rule deliberately permits.
    assert _refused("SELECT id FROM claim WHERE EXISTS (SELECT policy FROM policy)").code == (
        RefusalCode.SELECT_STAR)
    _ok("SELECT id FROM claim WHERE EXISTS (SELECT * FROM policy)")  # the star form still passes


def test_the_declared_sqlglot_floor_carries_the_symbols_the_guard_USES():
    """`exp.Columns` and `exp.SetOperation` are both absent from sqlglot 25.0.0 and present from
    25.34.1, and `check_shape` only wraps `sqlglot.parse` in a try -- so at the old declared floor
    of `>=25` an `AttributeError` would escape it and EVERY select would CRASH rather than refuse,
    not only star-bearing ones: `_is_star` reaches `exp.Columns` for the first projection of every
    output select. uv.lock pins 30.12.0, so nothing resolved from the lock was affected; the
    declaration was."""
    import re
    import tomllib

    from packaging.version import Version

    root = pathlib.Path(__file__).resolve().parents[1]
    deps = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    spec = next(d for d in deps if d.startswith("sqlglot"))
    floor = Version(re.search(r">=\s*([0-9.]+)", spec).group(1))

    assert floor >= Version("25.34.1"), f"{spec} predates exp.Columns and exp.SetOperation"

    # The `hasattr` below passes on any modern sqlglot and so proves nothing about the FLOOR --
    # it guards the running environment, not the declaration. The floor itself was established by
    # installing 25.0.0 and 25.34.1 and reading both symbols: absent, then present.
    assert hasattr(exp, "Columns") and hasattr(exp, "SetOperation")


# --- A column that shares its table's name, inside an aggregate -------------------------------
#
# Found by the ACME accuracy gate, which is what the gate is for. Two golden cases --
# `claim-amount-total` and `policy-amount-total` -- were deferring, and the engine's three
# attempts were all refused by this guard. The gold SQL itself is refused:
#
#     SELECT sum(claim_amount) AS total FROM claim_amount     -> select_star
#     SELECT sum(c.claim_amount) AS total FROM claim_amount c -> allowed
#
# Two answerable cases, eight accuracy points, unreachable by any phrasing the model could pick
# except one that qualifies the column. It looked like model drift for exactly that reason: the
# baseline run happened to write the aliased form.


def _is_refused(sql: str) -> bool:
    """Bool form, for the cases below that only care THAT it refused.

    Named apart from this file's `_refused`, which returns the Refusal so a caller can assert on
    `.code`. Defining a second `_refused` silently replaced it for every test after this point --
    21 of them started failing on `'bool' object has no attribute 'code'`, which looked like a
    guard regression and was a name collision.
    """
    return isinstance(check_shape(sql), Refusal)


def test_a_collapsing_aggregate_over_a_table_named_column_is_allowed():
    """`sum` cannot carry a struct out of the engine, whichever way the name resolves.

    The rule this guard enforces is real -- a BARE reference to a source name is the whole row in
    DuckDB, and `check_cls` cannot see a denied column that is never spelled. It just does not
    reach here. If `claim_amount` resolves to the column, `sum` returns a number; if it resolves
    to the row, `sum(STRUCT)` is a type error and the query fails. Neither leaks a value.
    """
    assert not _is_refused("SELECT sum(claim_amount) AS total FROM claim_amount")
    assert not _is_refused("SELECT avg(claim_amount) FROM claim_amount")
    assert not _is_refused("SELECT count(claim_amount) FROM claim_amount")


def test_an_aggregate_that_returns_its_argument_is_still_refused():
    """The distinction the fix turns on, and the reason `exp.AggFunc` is the wrong test.

    `max` IS an aggregate and in DuckDB `max(claim)` over a struct returns a STRUCT -- every value
    in the row, including a denied one, through a projection that looks like an aggregate. Same
    for `min` and `any_value`. Only aggregates whose result type cannot be the argument's type
    are safe.
    """
    assert _is_refused("SELECT max(claim_amount) FROM claim_amount")
    assert _is_refused("SELECT min(claim_amount) FROM claim_amount")
    # The one the old justification would have admitted. `array_agg`'s result type is
    # LIST(argument), which is not the argument's type -- and on DuckDB it returns a list of whole
    # structs, every value in every row. It is the worst leak of the set, so it guards the tuple.
    assert _is_refused("SELECT array_agg(claim_amount) FROM claim_amount")
    assert _is_refused("SELECT any_value(claim_amount) FROM claim_amount")


def test_distinct_does_not_change_whether_the_argument_is_a_bare_struct():
    """M85. `count(DISTINCT claim_amount)` is what a model writes for "how many different claim
    amounts", and it deferred for a reason that has nothing to do with distinctness: sqlglot parses
    it as `Count(this=Distinct(...))`, so a node sits between the column and the aggregate.

    Fixed by giving the two callers their OWN predicate rather than sharing one, which is what the
    first attempt got wrong -- stepping over `Distinct` in the shared helper widened the star walk
    too, and `count(DISTINCT t) FROM (SELECT * FROM claim) t` stopped being examined at all. The
    projection rule can afford the step-over because it is deciding whether a VALUE leaves; the
    star walk cannot, because it is deciding whether to LOOK.
    """
    assert not _is_refused("SELECT count(DISTINCT claim_amount) FROM claim_amount")
    assert not _is_refused("SELECT sum(DISTINCT claim_amount) FROM claim_amount")
    # Stepping over DISTINCT must not step over an extraction underneath it.
    assert _is_refused("SELECT count(DISTINCT claim['salary']) FROM claim")
    assert _is_refused("SELECT max(DISTINCT claim_amount) FROM claim_amount")


def test_the_star_walk_still_examines_a_distinct_count_over_a_derived_star():
    """The shape that must never become allowed while fixing the one above.

    This is where the two callers differ. `count(DISTINCT t)` carries no value out, so the
    projection rule has nothing to object to -- but the star walk's job is to decide whether an
    unbounded column set is reached at all, and a projection it declines to walk is one it never
    examines.
    """
    assert _is_refused("SELECT count(DISTINCT t) FROM (SELECT * FROM claim) t")
    assert _is_refused("SELECT sum(DISTINCT t['salary']) FROM (SELECT * FROM claim) t")
    # One addend over. `_names_a_source` returns the FIRST match, so asking it alone resolved this
    # to the base table, took the exemption, and left the derived star unwalked.
    assert _is_refused(
        "SELECT count(DISTINCT claim_amount) + count(DISTINCT t) "
        "FROM claim_amount, (SELECT * FROM claim) t"
    )
    # Stricter than before this rule existed, and intended: `count(t)` over a derived star was
    # allowed because the exemption skipped the walk, so the inner star was never examined. Only a
    # cardinality ever left, so nothing leaked -- but "nobody looked" is not a property to keep.
    assert _is_refused("SELECT count(t) FROM (SELECT * FROM claim) t")
    assert _is_refused("SELECT sum(t) FROM (SELECT * FROM claim) t")
    # The CTE spelling of the same thing. A CTE reference is an `exp.Table`, so matching only
    # `exp.Subquery` left this star unwalked.
    assert _is_refused("WITH c AS (SELECT * FROM claim) SELECT count(DISTINCT c) FROM c")
    assert _is_refused(
        "WITH c AS (SELECT * FROM claim) "
        "SELECT count(DISTINCT claim_amount) + count(DISTINCT c) FROM claim_amount, c"
    )
    # ALIASED, which is where matching `alias_or_name` reopened it: `FROM c AS x` answers `x` and
    # misses a CTE named `c`. Every spelling that reaches the exemption, not just the bare one.
    for sql in (
        "WITH c AS (SELECT * FROM claim) SELECT count(DISTINCT x) FROM c AS x",
        "WITH c AS (SELECT * FROM claim) SELECT count(x) FROM c AS x",
        "WITH c AS (SELECT * FROM claim) SELECT sum(x) FROM c AS x",
        "WITH c AS (SELECT * FROM claim) "
        "SELECT count(DISTINCT claim_amount) + count(DISTINCT x) FROM claim_amount, c AS x",
    ):
        assert _is_refused(sql), sql
    # Quoted and mixed-case, since `resolve_name` is what folds these and a raw string compare
    # would pass the unquoted spellings above while missing these.
    assert _is_refused('WITH "c" AS (SELECT * FROM claim) SELECT count(DISTINCT x) FROM "c" AS x')
    assert _is_refused("WITH c AS (SELECT * FROM claim) SELECT count(DISTINCT x) FROM C AS x")

    # The other direction, which this change also moved: an ALIAS that collides with an unrelated
    # CTE name used to force the walk, and no longer does. The new verdict is the right one -- the
    # reference resolves to a BASE table, where `count(DISTINCT row)` is the M85 exemption -- but
    # nothing pinned it, so a future edit could re-tighten or further loosen it unnoticed.
    assert not _is_refused(
        "WITH c AS (SELECT id FROM policy) SELECT count(DISTINCT c) FROM claim_amount AS c"
    )


def test_a_bounded_derived_table_beside_a_base_table_is_a_known_false_refusal():
    """M88 again, in the star walk. Named here because it arrived as a side effect.

    `t` projects an explicit column list, so nothing about it is unbounded. But an UNQUALIFIED row
    reference makes `_expands_a_base_table` read the projection as a star over every source in the
    FROM, and the sibling base table answers yes. The qualified spelling is the rewrite, as it is
    everywhere else this collision shows up.
    """
    assert _is_refused(
        "SELECT count(claim_amount) + count(t) FROM claim_amount, (SELECT id FROM claim) t"
    )
    assert not _is_refused(
        "SELECT count(a.claim_amount) + count(t.id) FROM claim_amount a, (SELECT id FROM claim) t"
    )


def test_the_distinct_phrasing_works_outside_the_projection_too():
    """Three sites make this judgement, not two, and the third kept deferring.

    `count` returns a cardinality wherever it is written, so the reasoning that allows the
    projection spelling allows this one.
    """
    assert not _is_refused(
        "SELECT id FROM claim_amount GROUP BY id HAVING count(DISTINCT claim_amount) > 1"
    )
    assert not _is_refused(
        "SELECT id FROM claim_amount GROUP BY id HAVING count(claim_amount) > 1"
    )
    assert _is_refused(
        "SELECT id FROM claim_amount GROUP BY id HAVING max(DISTINCT claim_amount) > 1"
    )
    # The aggregate has to be the DIRECT parent here as much as in the projection. Asking whether
    # one appears anywhere above -- the "nearest enclosing function" shape that leaked twice
    # already -- would allow this, a denied column read through an extraction the CLS scan cannot
    # see, and it changes the verdict on nothing else in this file.
    assert _is_refused("SELECT id FROM claim GROUP BY id HAVING sum(claim['salary']) > 1")
    assert _is_refused("SELECT id FROM claim GROUP BY id HAVING count(DISTINCT claim['ssn']) > 1")


def test_a_spreading_function_over_a_table_named_column_is_still_refused():
    """`UNNEST` is the shape the original rule was written for and must not regress."""
    assert _is_refused("SELECT UNNEST(claim_amount) FROM claim_amount")
    assert _is_refused("SELECT sum(UNNEST(claim_amount)) FROM claim_amount")


def test_a_struct_extraction_inside_an_aggregate_is_still_refused():
    """The hole the first version of this exemption opened, and why it is narrow now.

    `exp.Bracket` and `exp.Dot` are not `exp.Func` subclasses, so a walk looking for the first
    enclosing FUNCTION strolls past the extraction to the `sum` above it. The only `exp.Column`
    in `sum(claim['salary'])` is `claim`, so the CLS scan never sees `salary` and never asks
    whether it is denied -- an exempted projection reading a column no grant check can see, which
    is the precise thing this guard exists to stop.
    """
    assert _is_refused("SELECT sum(claim['salary']) FROM claim")
    assert _is_refused("SELECT claim_identifier, sum(claim['salary']) FROM claim "
                       "GROUP BY claim_identifier")
    # NOT included: `sum(claim.salary)`. That names a column explicitly, `check_cls` sees it,
    # and it was allowed before this change as well -- a qualified reference is the shape the
    # whole-row rule is defined against, not an instance of it.


def test_the_exemption_does_not_stop_the_star_walk_reaching_a_base_table():
    """The second hole, and why the exemption is applied where it is.

    `_names_a_source` does double duty: `_projects_a_row` asks it, and the star walk uses it as
    an ENTRY condition. Exempting inside it removed these projections from the walk entirely, so
    the inner `SELECT *`'s unbounded column set reached no grant check -- the regression
    `_projects_a_row`'s own docstring warns about. The exemption lives at the `_projects_a_row`
    call site now, and `_names_a_source` answers exactly what it answered before.
    """
    assert _is_refused("SELECT sum(t['salary']) FROM (SELECT * FROM claim) t")
    assert _is_refused("WITH t AS (SELECT * FROM claim) SELECT sum(t['salary']) FROM t")


def test_a_bare_whole_row_projection_is_still_refused():
    """Nothing about this fix touches the case the guard exists for."""
    assert _is_refused("SELECT claim_amount FROM claim_amount")
    assert _is_refused("SELECT claim_amount AS everything FROM claim_amount")
    assert _is_refused("SELECT x FROM (SELECT claim_amount AS x FROM claim_amount) t")


# --- M86: a whole-row reference outside the projection -----------------------------------------
#
# `_projects_a_row` scanned `select.expressions` only, so every clause but the projection was
# unguarded. `check_cls` walks the whole statement, but it can only see columns that are SPELLED:
# `claim['ssn']` yields `Column(claim)` and never an `exp.Column` named `ssn`, so `policy.denies`
# is asked about `claim` and answers no. The row count then answers the predicate -- a binary
# oracle over a column the caller may not read.
#
# The rule is the SAME one the projection uses: a bare source reference is refused, with M84's
# collapsing-aggregate exemption and the derived-table skip. Two narrower versions leaked first --
# refusing only subscripts/dots/calls misses the struct-comparison oracle below, and resolving
# names against the innermost select only misses correlated references. See
# `_uses_a_row_outside_the_projection` for the measurements.


def test_a_field_read_from_a_whole_row_is_refused_in_every_clause():
    for sql in (
        "SELECT claim_identifier FROM claim WHERE claim['ssn'] = 'x'",
        "SELECT claim_identifier FROM claim ORDER BY claim['ssn']",
        "SELECT claim_identifier FROM claim GROUP BY claim['ssn']",
        "SELECT count(*) FROM claim HAVING max(claim['ssn']) > 'x'",
        "SELECT claim_identifier FROM claim c WHERE c['ssn'] = 'x'",
        "SELECT claim_identifier FROM claim WHERE claim['address']['city'] = 'x'",
        "SELECT claim_identifier FROM claim WHERE (claim).ssn = 'x'",
        # A function reaches a field without a subscript, and naming the dangerous ones would be a
        # list that leaks the moment it fell behind DuckDB.
        "SELECT claim_identifier FROM claim WHERE struct_extract(claim, 'ssn') = 'x'",
        "SELECT claim_identifier FROM claim WHERE claim IN (SELECT c FROM claim c)",
    ):
        assert _is_refused(sql), sql
    # Quoted and mixed-case, since `resolve_name` is what folds these and a raw string compare
    # would pass the unquoted spellings above while missing these.
    assert _is_refused('WITH "c" AS (SELECT * FROM claim) SELECT count(DISTINCT x) FROM "c" AS x')
    assert _is_refused("WITH c AS (SELECT * FROM claim) SELECT count(DISTINCT x) FROM C AS x")

    # The other direction, which this change also moved: an ALIAS that collides with an unrelated
    # CTE name used to force the walk, and no longer does. The new verdict is the right one -- the
    # reference resolves to a BASE table, where `count(DISTINCT row)` is the M85 exemption -- but
    # nothing pinned it, so a future edit could re-tighten or further loosen it unnoticed.
    assert not _is_refused(
        "WITH c AS (SELECT id FROM policy) SELECT count(DISTINCT c) FROM claim_amount AS c"
    )


def test_a_comparison_against_a_whole_row_is_refused_too():
    """The reason the rule outside the projection is blanket rather than clever.

    The first version refused subscripts, dots and calls, reasoning that only those can reach a
    field. True of extracting a value, false about the threat: DuckDB compares structs
    field-by-field, so this is a binary search over a denied column with no subscript, dot or call
    in it -- measured 1/0/0 against the real value on a live DuckDB.
    """
    assert _is_refused(
        "SELECT count(*) FROM claim WHERE claim > {'claim_identifier': 1, 'ssn': 'guess'}"
    )
    assert _is_refused("SELECT claim_identifier FROM claim ORDER BY claim")
    assert _is_refused("SELECT count(*) FROM claim GROUP BY claim")


def test_a_correlated_reference_to_an_outer_row_is_refused():
    """Names resolve OUTWARD, so the nearest select is not the whole answer.

    Resolving a bare reference against only the innermost select's sources missed an outer table
    entirely, and the oracle survived one `EXISTS (...)` deep -- verified returning the row for a
    correct guess and nothing for a wrong one.
    """
    assert _is_refused(
        "SELECT claim_identifier FROM claim WHERE EXISTS "
        "(SELECT 1 FROM policy WHERE claim['ssn'] = 'x')"
    )
    assert _is_refused(
        "SELECT p.id FROM policy p WHERE EXISTS (SELECT 1 FROM claim WHERE claim['ssn'] = 'x')"
    )


def test_a_table_named_column_in_a_predicate_is_a_known_false_refusal():
    """Pinned REFUSED on purpose, so the next person meets the decision rather than the bug.

    Nothing here can tell a column that shares its table's name from the row itself, and outside
    the projection every use of the row turned out to leak. So this pays the same price the
    projection rule has always paid. No ACME gold case filters or orders on such a column --
    checked, not assumed -- and `WHERE c.claim_amount > 10` names a column and is allowed, which
    is the rewrite that makes it work. Filed as M88.
    """
    assert _is_refused("SELECT id FROM claim_amount WHERE claim_amount > 10")
    assert _is_refused("SELECT id FROM claim_amount ORDER BY claim_amount")
    # The escape hatch, and the reason the cost is bounded.
    assert not _is_refused("SELECT c.id FROM claim_amount c WHERE c.claim_amount > 10")
    assert not _is_refused("SELECT c.id FROM claim_amount c ORDER BY c.claim_amount")


def test_a_bare_name_matching_an_outer_source_is_refused_in_any_scope():
    """Fail closed across scopes, which is a decision and cost M88, not an oversight.

    Honouring scope boundaries here was tried and reverted. Stopping the outward walk at a CTE
    looks right -- a CTE body should not see the outer FROM -- but DuckDB resolves outer columns
    into a CTE body nested in a correlated subquery, so the break re-armed the oracle: the FIRST
    assertion below returned the row for a correct guess and nothing for a wrong one. The second
    stayed refused throughout, which is what made the break look safe.
    """
    # The oracle, one CTE inside one correlated subquery deep.
    assert _is_refused(
        "SELECT claim_identifier FROM claim WHERE EXISTS "
        "(WITH y AS (SELECT 1 WHERE claim['ssn'] = 'x') SELECT * FROM y)"
    )
    assert _is_refused(
        "SELECT claim_identifier FROM claim WHERE EXISTS "
        "(SELECT 1 FROM policy WHERE claim['ssn'] = 'x')"
    )
    # The cost: a local column refused because an OUTER table shares its name. M88, across a
    # scope, with the same rewrite.
    assert _is_refused(
        "WITH x AS (SELECT id FROM policy WHERE claim_amount > 10) SELECT id FROM claim_amount"
    )
    assert not _is_refused(
        "WITH x AS (SELECT id FROM policy p WHERE p.claim_amount > 10) SELECT id FROM claim_amount"
    )


def test_indexing_a_list_column_is_not_a_row_read():
    """`tags[1]` names a COLUMN, which `check_cls` can see. Only a SOURCE name is a whole row."""
    assert not _is_refused("SELECT id FROM claim WHERE tags[1] = 'x'")
    assert not _is_refused("SELECT tags[1] FROM claim")


# --- M88: with a schema, the guard stops guessing ----------------------------------------------
#
# Every false refusal in this family -- M84's `sum(claim_amount)`, M85's distinct-count, M88's
# `WHERE claim_amount > 10` -- comes from one unanswerable question: is a bare name the COLUMN or
# the whole ROW? The guard had no schema, so it fails closed and pays for it.
#
# The decider has one. `decide` already takes `visible: dict[str, set[str]]`, table to columns, and
# passes it to `check_access` two lines later. Handed the same map, the question stops being a
# guess: DuckDB resolves a name to the COLUMN when one exists and to the row only when none does,
# measured both ways, so a name that is a known column is a column.


_ACME = {"claim_amount": {"claim_amount", "id"}, "claim": {"claim_identifier", "salary", "ssn"}}


def _with_schema(sql: str) -> bool:
    """Refused, when the guard is told what the columns are."""
    return isinstance(check_shape(sql, columns=_ACME), Refusal)


def test_a_known_column_sharing_its_table_name_is_a_column():
    """The M84/M88 family, ended rather than traded."""
    for sql in (
        "SELECT claim_amount FROM claim_amount",
        "SELECT id FROM claim_amount WHERE claim_amount > 10",
        # `decide` still refuses the next one, through `lint` rather than here: ascending order
        # with a LIMIT surfaces NULLs first. A different rule doing its job, repairable, and worth
        # knowing the shape check is no longer what stops it.
        "SELECT id FROM claim_amount ORDER BY claim_amount",
        "SELECT id FROM claim_amount GROUP BY id HAVING count(DISTINCT claim_amount) > 1",
        "SELECT abs(claim_amount) FROM claim_amount",
        "SELECT max(claim_amount) FROM claim_amount",
    ):
        assert not _with_schema(sql), sql


def test_a_name_with_no_such_column_is_still_the_whole_row():
    """`claim` has no column called `claim`, so it is the row -- and every leak stays closed."""
    for sql in (
        "SELECT claim FROM claim",
        "SELECT claim_identifier FROM claim WHERE claim['ssn'] = 'x'",
        "SELECT count(*) FROM claim WHERE claim > {'claim_identifier': 1, 'ssn': 'guess'}",
        "SELECT claim_identifier FROM claim ORDER BY claim",
        "SELECT UNNEST(claim) FROM claim",
        "SELECT max(claim) FROM claim",
        "SELECT claim_identifier FROM claim WHERE EXISTS "
        "(SELECT 1 FROM claim_amount WHERE claim['ssn'] = 'x')",
    ):
        assert _with_schema(sql), sql


def test_a_star_is_still_a_star_when_the_schema_is_known():
    """The schema answers "column or row". It says nothing about an unbounded projection."""
    for sql in (
        "SELECT * FROM claim_amount",
        "SELECT * FROM (SELECT * FROM claim) t",
        "SELECT COLUMNS(*) FROM claim_amount",
    ):
        assert _with_schema(sql), sql


def test_without_a_schema_the_guard_behaves_exactly_as_before():
    """The map is optional, and its absence must not quietly loosen anything.

    Every production path reaches this through `decide`, which forwards `visible` -- so the
    unmapped behaviour is what a direct caller and this test file get, and it must stay the strict
    one. Loosening on a missing map would make forgetting to pass it a silent grant.
    """
    assert _is_refused("SELECT id FROM claim_amount WHERE claim_amount > 10")
    assert not _with_schema("SELECT id FROM claim_amount WHERE claim_amount > 10")


def test_a_column_name_from_another_scope_does_not_exempt_a_row_here():
    """The schema has to be read per SCOPE, not unioned over the statement.

    A flat set of "every column name anywhere in this query" let a name that is a column
    SOMEWHERE exempt a whole-row reference WHERE IT IS NOT. Here `other.claim` is a column and
    `claim.claim` is not, so the outer `SELECT claim FROM claim` is the row -- and DuckDB returns
    the whole struct, `ssn` included, past a CLS check that sees no `exp.Column` named `ssn`.
    """
    schema = {"claim": {"id", "ssn"}, "other": {"claim", "id"}}
    assert isinstance(
        check_shape("SELECT claim FROM claim WHERE id IN (SELECT claim FROM other)",
                    columns=schema),
        Refusal,
    )
    # The extraction oracle rides back in on the same exemption.
    assert isinstance(
        check_shape("SELECT id FROM claim WHERE claim['ssn'] = 'x' "
                    "AND id IN (SELECT claim FROM other)", columns=schema),
        Refusal,
    )
    # The inner reference really is `other.claim`, and stays allowed on its own.
    assert not isinstance(check_shape("SELECT claim FROM other", columns=schema), Refusal)

    # The MIRROR: an OUTER table's column must not exempt a row reference in an INNER scope where
    # the name is a source. Measured on DuckDB, the inner bare `claim` binds to the inner table's
    # row struct, not to the correlated `other.claim` -- so the oracle works, and returns the row
    # on a correct guess and nothing on a wrong one.
    assert isinstance(
        check_shape("SELECT id FROM other WHERE EXISTS "
                    "(SELECT 1 FROM claim WHERE claim['ssn'] = 'x')", columns=schema),
        Refusal,
    )


def test_a_cte_shadowing_a_granted_table_does_not_borrow_its_columns():
    """A CTE reference is an `exp.Table`, so keying the schema on it lends a base table's columns
    to something that is not that table.

    `WITH other AS (SELECT id FROM other)` shadows the granted `other`, whose column list contains
    `claim`. That made the bare `claim` in the outer select look like a column, the row rule was
    never reached, and DuckDB returned the whole `claim` struct -- denied `ssn` included -- past a
    CLS check that sees no column for it.
    """
    schema = {"claim": {"claim_identifier", "ssn"}, "other": {"claim", "id"}}
    assert isinstance(
        check_shape("WITH other AS (SELECT id FROM other) SELECT claim FROM claim, other",
                    columns=schema),
        Refusal,
    )
    # Without the shadowing CTE the same name really is `other.claim`, and stays allowed.
    assert not isinstance(check_shape("SELECT claim FROM claim, other", columns=schema), Refusal)


def test_a_derived_table_cannot_exempt_anything_through_the_schema():
    """The schema is keyed by BASE table, so a derived table must not lend its alias to the lookup.

    What makes this hold is `object_key`, which answers `""` for an `exp.Subquery` -- so the map is
    missed whatever the alias is. The `isinstance(source, exp.Table)` guard beside it is belt to
    that braces and would not be missed by this test on its own; the mechanism worth knowing is the
    key, and a caller that built the map from ALIASES rather than table names is what would break
    it: the star walk then never opens the derived table and the star over `claim` inside it is
    never examined.
    """
    schema = {"claim": {"id", "ssn"}, "t": {"t", "id"}}
    assert isinstance(
        check_shape("SELECT max(t) AS x FROM (SELECT * FROM claim) t", columns=schema), Refusal
    )
    assert isinstance(
        check_shape("SELECT t FROM (SELECT * FROM claim) t", columns=schema), Refusal
    )


def test_a_column_of_ANOTHER_table_in_scope_also_resolves():
    """Bare names resolve across every source in the FROM, not only the one they look like."""
    schema = {"claim": {"claim_identifier"}, "other": {"claim"}}
    assert not isinstance(check_shape("SELECT claim FROM claim, other", columns=schema), Refusal)
    # And with no such column anywhere, it is the row again.
    assert isinstance(check_shape("SELECT claim FROM claim, other",
                                  columns={"claim": {"claim_identifier"}, "other": {"id"}}),
                      Refusal)
