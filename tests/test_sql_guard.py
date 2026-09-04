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
    definition but not its reference."""
    for sql in ('WITH "Claim" AS (SELECT id FROM policy) SELECT * FROM Claim',
                'WITH Claim AS (SELECT id FROM policy) SELECT * FROM "Claim"'):
        assert _refused(sql).code == RefusalCode.SELECT_STAR, sql

    # And it must not over-refuse, which is the other direction and equally live. Everything below
    # is ONE object under the default `duckdb` these run on, and under every DOWN-folding engine --
    # NOT under Oracle, where the two mixed-quoting cases are two objects and are refused; the
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
    assert _star_reaches_base(exp.Anonymous(this="opaque"), {}, {}, frozenset(), "duckdb") is True


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
                "SELECT * FROM (SELECT UNNEST(claim) FROM claim) t"):
        assert _refused(sql).code == RefusalCode.SELECT_STAR, sql


def test_unnesting_a_COLUMN_is_untouched():
    """The control, and the reason this is keyed on naming a SOURCE rather than on `UNNEST`:
    unnesting a LIST column expands ROWS and returns one column, which is ordinary SQL."""
    _ok("SELECT UNNEST(tags) FROM claim")
    _ok("SELECT a.id FROM claim a JOIN policy p ON a.id = p.id")
    # a CTE source is bounded by its own projection, so naming it returns known columns
    _ok("WITH c AS (SELECT id FROM claim) SELECT c FROM c")

    # A QUALIFIED reference names a column even when the column shares its table's name, so the
    # check keys on the reference being BARE. Without that it refuses ordinary SQL: a `policy`
    # table with a `policy` column is not an exotic schema.
    _ok("SELECT p.policy FROM policy p")
    _ok("SELECT policy.policy FROM policy")
    _ok("SELECT claim.claim FROM claim")


def test_the_declared_sqlglot_floor_carries_the_symbols_the_guard_USES():
    """`exp.Columns` and `exp.SetOperation` are both absent from sqlglot 25.0.0 and present from
    25.34.1, and `check_shape` only wraps `sqlglot.parse` in a try -- so at the old declared floor
    of `>=25` an `AttributeError` would escape it and every star-bearing query would CRASH rather
    than refuse. uv.lock pins 30.12.0, so nothing resolved from the lock was affected; the
    declaration was."""
    import re
    import tomllib

    from packaging.version import Version

    root = pathlib.Path(__file__).resolve().parents[1]
    deps = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    spec = next(d for d in deps if d.startswith("sqlglot"))
    floor = Version(re.search(r">=\s*([0-9.]+)", spec).group(1))

    assert floor >= Version("25.34.1"), f"{spec} predates exp.Columns and exp.SetOperation"
    assert hasattr(exp, "Columns") and hasattr(exp, "SetOperation")
