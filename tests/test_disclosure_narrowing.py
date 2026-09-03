"""An answer that was narrowed by policy carries WHAT was narrowed.

The gap this closes: a governed query returns three rows of forty-seven and says nothing, so the
caller reads a filtered answer as a complete one. The signal is table-granular by design (a masked
column can be consumed by an aggregate and never reach the output, so per-column claims would
exceed what the loop knows) and carries no predicate and no policy identity -- the caller is
entitled to know it was narrowed, not to the rule that narrowed it.

The return TYPE carries it, rather than an out-parameter or a field a caller may forget to read.
A caller that ignores it gets a tuple where an expression was expected and fails immediately, which
is the only arrangement in which silence cannot be the default.
"""

import sqlglot

from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import Narrowing, apply_row_and_mask, apply_row_filters_to_write
from mnemiq.sql.verdict import Refusal

_VISIBLE = {"claim": {"id", "amount", "ssn"}}


def _ast(sql):
    return sqlglot.parse_one(sql, read="duckdb")


def test_a_row_filter_is_reported_as_a_row_narrowing():
    _, narrowed = apply_row_and_mask(
        _ast("SELECT id, amount FROM claim"),
        AccessPolicy(row_filters={"claim": "amount > 0"}), _VISIBLE, dialect="duckdb")
    assert narrowed == [Narrowing(object="claim", rows=True, columns=False)]


def test_a_masked_column_is_reported_as_a_column_narrowing():
    _, narrowed = apply_row_and_mask(
        _ast("SELECT id, ssn FROM claim"),
        AccessPolicy(masked={("claim", "ssn")}), _VISIBLE, dialect="duckdb")
    assert narrowed == [Narrowing(object="claim", rows=False, columns=True)]


def test_an_UNGOVERNED_query_reports_an_empty_list_and_not_an_absence():
    """`[]` is a claim -- "the policy narrowed nothing" -- and it must be distinguishable from a
    query that never reached a decider. That is M56's distinction, one object over."""
    out, narrowed = apply_row_and_mask(
        _ast("SELECT id FROM claim"), AccessPolicy(), _VISIBLE, dialect="duckdb")
    assert narrowed == []
    assert not isinstance(out, Refusal)


def test_a_REFUSAL_carries_no_narrowing():
    """Nothing was narrowed because nothing was answered. Reporting a narrowing beside a refusal
    would describe a rewrite that never happened."""
    out, narrowed = apply_row_and_mask(
        _ast("SELECT id FROM claim"),
        AccessPolicy(row_filters={"claim": "nonexistent > 0"}), _VISIBLE, dialect="duckdb")
    assert isinstance(out, Refusal)
    assert narrowed == []


def test_the_WRITE_TARGETS_own_filter_is_reported():
    """The case the spec calls easiest to miss, and the one an implementer of the read path alone
    would ship silent.

    An UPDATE/DELETE target cannot be wrapped in a derived table, so it is EXCLUDED from the loop
    that records narrowings and its filter is conjoined afterwards. Wire up the read path only and
    a governed DELETE narrowed from forty-seven rows to three reports nothing.
    """
    ast = _ast("DELETE FROM claim WHERE id = 1")
    target = next(iter(ast.find_all(sqlglot.exp.Table)))
    out, narrowed = apply_row_filters_to_write(
        ast, AccessPolicy(row_filters={"claim": "amount > 0"}), _VISIBLE, target, "duckdb")
    assert not isinstance(out, Refusal)
    assert Narrowing(object="claim", rows=True, columns=False) in narrowed, (
        f"the DELETE target's own row filter was applied and not reported: {narrowed}")
    assert "amount > 0" in out.sql(dialect="duckdb").lower()


def test_the_narrowing_names_no_predicate_and_no_policy():
    """What a caller may know is THAT it was narrowed, not the rule. A predicate leaks the shape of
    other principals' access, which is why the record carries a flag and an object and nothing else.
    """
    _, narrowed = apply_row_and_mask(
        _ast("SELECT id, ssn FROM claim"),
        AccessPolicy(row_filters={"claim": "amount > 4242"}, masked={("claim", "ssn")}),
        _VISIBLE, dialect="duckdb")
    assert len(narrowed) == 1, f"one table, one record: {narrowed}"
    assert narrowed[0].rows and narrowed[0].columns, (
        "a table narrowed BOTH ways must say both; reporting one hides the other")
    text = repr(narrowed[0])
    assert "4242" not in text and ">" not in text, f"the predicate leaked into the record: {text}"


def test_one_OBJECT_yields_one_record_however_often_it_is_referenced():
    """The loop walks table NODES. A self-join reaches `claim` twice and would report
    "claim, claim" to anyone listing or counting narrowed objects."""
    _, narrowed = apply_row_and_mask(
        _ast("SELECT a.id FROM claim a JOIN claim b ON a.id = b.id"),
        AccessPolicy(row_filters={"claim": "amount > 0"}), _VISIBLE, dialect="duckdb")
    assert narrowed == [Narrowing(object="claim", rows=True, columns=False)], narrowed


def test_a_write_whose_WHERE_reads_its_own_target_reports_it_once():
    """Two paths reach `claim`: the loop wraps the inner read, and the target is conjoined after
    it. Before merging, that emitted the same object twice from two different code paths."""
    ast = _ast("DELETE FROM claim WHERE id IN (SELECT id FROM claim)")
    target = next(iter(ast.find_all(sqlglot.exp.Table)))
    _, narrowed = apply_row_filters_to_write(
        ast, AccessPolicy(row_filters={"claim": "amount > 0"}), _VISIBLE, target, "duckdb")
    assert [n.object for n in narrowed] == ["claim"], narrowed


def test_NOT_EVALUATED_is_distinguishable_from_narrowed_nothing():
    """`Approved` is built outside any decider — `eval/verify_replay.py` does it — and there
    nobody evaluated governance at all. A default of `[]` would say "the policy narrowed nothing"
    on that record's behalf, which is M56's collapse in a new field."""
    from mnemiq.sql.verdict import Approved

    assert Approved(plan_sql="SELECT 1", target_sql="SELECT 1").narrowed is None
