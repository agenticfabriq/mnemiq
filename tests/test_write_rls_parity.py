"""M30 + M7 — a write's READS are governed by the same RLS as a read's.

Two findings, one defect: `apply_row_and_mask` is the RLS implementation and the write decider
re-implemented it inline, so the two disagreed about *where* a filter binds (M30) and about
*whether a filter is valid at all* (M7).

The tests are written as PARITY tests on purpose. Asserting the write path's behaviour alone
would let the two implementations drift again in the other direction; asserting that both paths
answer the same question the same way is what makes the duplication impossible to reintroduce
without a red test.
"""

import duckdb
import pytest
import sqlglot

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import apply_row_and_mask
from mnemiq.sql.verdict import ApprovedWrite, Refusal, RefusalCode

_VISIBLE = {"claim": {"id", "amount", "region"}, "scratch": {"id", "amount"}}
_SCHEMA = {k: set(v) for k, v in _VISIBLE.items()}
_FILTER = "region = 'west'"


def _policy(filt: str = _FILTER) -> AccessPolicy:
    return AccessPolicy(row_filters={"claim": filt}, policy_schema=_SCHEMA)


def _grants() -> GrantSet:
    return GrantSet(frozenset(_VISIBLE), writable=frozenset({"claim", "scratch"}))


class _OkAdapter:
    def execute(self, sql):
        return []


def _write(sql: str, policy: AccessPolicy | None = None):
    return decide_write(sql, _VISIBLE, _grants(), adapter=_OkAdapter(), dialect="duckdb",
                        writes_enabled=True, policy=policy if policy is not None else _policy())


# -- placement: every table a write READS is filtered, not just the one it writes -----------------

# The target of the mutation (A, B) already bound before this fix; C-G are every way a write
# reads, and none of them did. C is the one that matters: the governed rows land in an ungoverned
# table, where a later plain SELECT returns them forever.
_READS = [
    ("C_insert_select", "INSERT INTO scratch (id, amount) SELECT id, amount FROM claim"),
    ("D_update_subquery", "UPDATE scratch SET amount = 0 WHERE id IN (SELECT id FROM claim)"),
    ("E_update_from", "UPDATE scratch SET amount = claim.amount FROM claim "
                      "WHERE scratch.id = claim.id"),
    ("F_delete_subquery", "DELETE FROM scratch WHERE id IN (SELECT id FROM claim)"),
    ("G_insert_cte", "INSERT INTO scratch (id, amount) "
                     "WITH c AS (SELECT id, amount FROM claim) SELECT id, amount FROM c"),
]
_TARGETS = [
    ("A_update_target", "UPDATE claim SET amount = 0 WHERE id = 1"),
    ("B_delete_target", "DELETE FROM claim WHERE id = 1"),
]


@pytest.mark.parametrize("label,sql", _READS + _TARGETS, ids=lambda v: v if isinstance(v, str) else "")
def test_every_governed_table_a_write_touches_is_filtered(label, sql):
    verdict = _write(sql)
    assert isinstance(verdict, ApprovedWrite), f"{label}: {verdict}"
    assert _FILTER.split("=")[0].strip() in verdict.plan_sql, (
        f"{label}: the write reads `claim` with no row filter -- {verdict.plan_sql}"
    )


def test_an_insert_target_is_not_filtered_because_an_insert_does_not_read_it():
    # The exemption is deliberate and must stay: filtering the target of an INSERT would refuse
    # rows on the way IN, which is not what a row filter means.
    verdict = _write("INSERT INTO claim (id, amount, region) VALUES (1, 2.0, 'east')")
    assert isinstance(verdict, ApprovedWrite)
    assert "region = 'west'" not in verdict.plan_sql


def test_the_target_is_excluded_by_identity_and_not_by_name():
    # Same table in two positions: the inner read must be wrapped, the outer target conjoined.
    # Excluding by NAME would skip both and reintroduce M31's flat-name-set bypass.
    verdict = _write("UPDATE claim SET amount = 0 "
                     "WHERE id IN (SELECT id FROM claim WHERE amount > 5)")
    assert isinstance(verdict, ApprovedWrite)
    assert verdict.plan_sql.count("region = 'west'") == 2, verdict.plan_sql


# -- validation parity: both paths refuse the same filters for the same reason --------------------

# The read path proves a filter speaks only about what it is entitled to. The write path parsed it
# raw, so a policy typo crashed the decider and a filter naming an arbitrary table was spliced in
# unvalidated -- "the database as the control", which is the failure `decide_write`'s own docstring
# says M3 was.
_INVALID = [
    ("unparseable", "region = = 'west'"),
    ("not_a_predicate", "DELETE FROM claim"),
    ("column_not_on_table", "no_such_column = 1"),
    ("foreign_table", "region = (SELECT region FROM secret_table)"),
    ("the_stand_in", "__mnemiq_filtered__.region = 'west'"),
]


@pytest.mark.parametrize("label,filt", _INVALID, ids=[c[0] for c in _INVALID])
def test_an_invalid_row_filter_is_refused_identically_on_both_paths(label, filt):
    read = apply_row_and_mask(sqlglot.parse_one("SELECT id FROM claim", read="duckdb"),
                              _policy(filt), _VISIBLE, dialect="duckdb")
    assert isinstance(read, Refusal) and read.code == RefusalCode.INVALID_ROW_FILTER

    write = _write("UPDATE claim SET amount = 0 WHERE id = 1", _policy(filt))
    assert isinstance(write, Refusal), f"{label}: the write path accepted it -- {write}"
    assert write.code == read.code, f"{label}: {write.code} != {read.code}"


@pytest.mark.parametrize("label,filt", _INVALID, ids=[c[0] for c in _INVALID])
def test_no_exception_escapes_the_write_decider_for_an_invalid_filter(label, filt):
    # A policy typo must be a refusal, not a traceback in the caller. `sqlglot.parse_one` on a
    # malformed filter raised straight out of `decide_write`.
    for sql in ("UPDATE claim SET amount = 0 WHERE id = 1",
                "INSERT INTO scratch (id, amount) SELECT id, amount FROM claim"):
        verdict = _write(sql, _policy(filt))  # must not raise
        assert isinstance(verdict, (ApprovedWrite, Refusal))


# -- executed, not merely emitted -----------------------------------------------------------------


def _con():
    con = duckdb.connect()
    con.execute("CREATE TABLE claim (id INTEGER, amount DOUBLE, region VARCHAR)")
    con.execute("INSERT INTO claim VALUES (1, 10, 'west'), (2, 20, 'east'), (3, 30, 'west')")
    con.execute("CREATE TABLE scratch (id INTEGER, amount DOUBLE)")
    return con


@pytest.mark.parametrize("label,sql", _READS, ids=[c[0] for c in _READS])
def test_the_governed_write_moves_only_in_filter_rows(label, sql):
    """Run the approved SQL. A test that greps `plan_sql` passes on SQL that does not execute."""
    con = _con()
    con.execute("INSERT INTO scratch VALUES (1, 99), (2, 99), (3, 99)")
    verdict = _write(sql)
    assert isinstance(verdict, ApprovedWrite), verdict
    con.execute(verdict.target_sql)
    governed = con.execute("SELECT id, amount FROM scratch ORDER BY id, amount").fetchall()

    # The passing control: the same statement with no policy at all. If the two agree, the test
    # is measuring nothing -- which is how a guard that never fires reads as a guard that holds.
    ctl = _con()
    ctl.execute("INSERT INTO scratch VALUES (1, 99), (2, 99), (3, 99)")
    ungoverned = _write(sql, AccessPolicy())
    assert isinstance(ungoverned, ApprovedWrite)
    ctl.execute(ungoverned.target_sql)
    unfiltered = ctl.execute("SELECT id, amount FROM scratch ORDER BY id, amount").fetchall()

    assert governed != unfiltered, f"{label}: the filter changed nothing -- the control agrees"
    assert (2, 20.0) not in governed, f"{label}: the east row escaped into scratch -- {governed}"


# -- one implementation ---------------------------------------------------------------------------


def test_the_write_decider_has_no_second_rls_implementation():
    """M7's actual remedy. Leaving the inline copy correct-looking beside `apply_row_and_mask` is
    how the two came to disagree; M31's fix DELETED the superseded helper for the same reason.

    Asserted as a property rather than as a symbol name: the write decider must build no filter
    of its own, and `rls.py` must remain the only module that constructs the wrap.
    """
    import inspect

    from mnemiq.sql import decide_write as write_module
    from mnemiq.sql import rls as rls_module

    # Comments stripped: the assertion is about what the module DOES. The first version of this
    # test matched the prose of the comment explaining the fix, which is a test of the changelog.
    write_src = "\n".join(
        line for line in inspect.getsource(write_module).splitlines()
        if not line.strip().startswith("#")
    )
    assert "mnemiq.sql.rls" in write_src, "the write path must route through the RLS module"
    for built_here in ("exp.Where(this=", "_validate_filter", "row_filters.get"):
        assert built_here not in write_src, (
            f"the write decider builds its own filter ({built_here!r}) -- that is the second "
            "implementation M7 filed"
        )
    # And the wrap itself is constructed in exactly one place.
    assert inspect.getsource(rls_module).count("def _derived_table") == 1
