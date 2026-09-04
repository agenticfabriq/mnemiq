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
    """Threading the dialect to SOME call sites is worse than to none. Measured while building
    this: `check_access` called `base_tables` without it and said the name was a CTE, while
    `decide`'s audit list called it WITH the target and said it was a base read -- one statement,
    two answers, Approved with `claim` in `tables` and no grant covering it."""
    from mnemiq.sql.authz_guard import check_access
    from mnemiq.sql.guard import check_shape

    for engine, expect_read in (("oracle", True), ("postgres", False)):
        ast = check_shape(_ORACLE_ONLY, dialect=engine, executes_as=engine)
        reads = {t.name for t in base_tables(ast, engine)}
        refused = check_access(ast, _VISIBLE, engine) is not None
        assert ("claim" in reads) is expect_read, engine
        assert refused is expect_read, f"{engine}: the two resolvers disagree"


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
