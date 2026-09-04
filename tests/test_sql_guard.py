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


def test_QUOTING_is_half_a_cte_name_and_a_mismatch_does_not_vouch():
    """Postgres -- `decide`'s default transpile target -- folds an unquoted name and preserves a
    quoted one, so `"Claim"` and `Claim` are two different objects there. Matching on the name
    alone let the CTE vouch for a star that Postgres resolves to the base table `claim`, whose
    columns then reached no grant check: `decide` approved exactly that with `tables=['policy']`.

    Narrowing the match is fail-closed in every dialect, because only a WIDER match can vouch for
    a star that should have been refused."""
    for sql in ('WITH "Claim" AS (SELECT id FROM policy) SELECT * FROM Claim',
                'WITH Claim AS (SELECT id FROM policy) SELECT * FROM "Claim"'):
        assert _refused(sql).code == RefusalCode.SELECT_STAR, sql

    # spelled and quoted identically, so it really is the CTE
    _ok("WITH claim AS (SELECT id FROM policy) SELECT * FROM claim")
    _ok('WITH "claim" AS (SELECT id FROM policy) SELECT * FROM "claim"')

    # and a CTE reference carrying its own alias still resolves by the name it READS: `c x` is a
    # reference to `c`, not to `x`, and reading `alias_or_name` for both sides missed every one
    _ok("WITH c AS (SELECT id FROM claim) SELECT * FROM c x")


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
    assert _star_reaches_base(exp.Anonymous(this="opaque"), {}, {}, frozenset()) is True


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
