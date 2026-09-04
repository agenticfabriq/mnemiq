"""A masked column name is attributed to its OWNING table before anything is claimed about it.

The masking loop matched masked column NAMES against every column in the query without resolving
ownership, so any table masking `ssn` was treated as masked whenever any `ssn` appeared. Harmless
while it only over-masked; not harmless once a caller-facing sentence was built on it, which told
people a column was withheld when nothing they read was.

SAFETY IS THE INVARIANT, not accuracy. The old behaviour masked MORE than necessary and never
less, so attribution may only narrow the set where the owner is KNOWN. Every unresolved case keeps
the old answer, and those cases are the ones this file guards hardest -- a wrong attribution here
would un-mask real data, which is a far worse defect than the one being fixed.
"""

import sqlglot

from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.rls import apply_row_and_mask

_V = {"claim": {"id", "amount", "ssn"}, "person": {"id", "ssn"}}


def _run(sql, policy, visible=None):
    out, narrowed = apply_row_and_mask(
        sqlglot.parse_one(sql, read="duckdb"), policy, visible or _V, dialect="duckdb")
    return out.sql(dialect="duckdb").lower(), narrowed


def test_another_tables_same_named_column_is_no_longer_claimed():
    """The reported defect. Only `claim.ssn` is masked and only `person.ssn` is read."""
    sql, narrowed = _run("SELECT p.ssn FROM person p JOIN claim c ON p.id = c.id",
                         AccessPolicy(masked={("claim", "ssn")}))
    assert narrowed == [], f"claim was claimed as masked though only person.ssn was read: {narrowed}"
    assert "null as ssn" not in sql


def test_the_OWNING_tables_column_is_still_masked():
    """The half that must not regress."""
    sql, narrowed = _run("SELECT c.ssn FROM person p JOIN claim c ON p.id = c.id",
                         AccessPolicy(masked={("claim", "ssn")}))
    assert [(n.object, n.columns) for n in narrowed] == [("claim", True)]
    assert "null as ssn" in sql


def test_an_AMBIGUOUS_unqualified_reference_still_masks_every_candidate():
    """Two tables mask `ssn` and the reference names neither. Resolution declines to guess, so the
    old over-approximation stands: both are masked. Getting this wrong would un-mask real data."""
    sql, narrowed = _run("SELECT ssn FROM person JOIN claim ON person.id = claim.id",
                         AccessPolicy(masked={("claim", "ssn"), ("person", "ssn")}))
    assert {n.object for n in narrowed} == {"claim", "person"}
    assert sql.count("null as ssn") == 2


def test_a_reference_through_a_CTE_still_masks():
    """`column_tables` resolves, but the qualifier names a CTE rather than a base table, so the
    owner is unknown -- which is an unresolved case and keeps the safe answer."""
    sql, narrowed = _run("WITH t AS (SELECT ssn FROM claim) SELECT ssn FROM t",
                         AccessPolicy(masked={("claim", "ssn")}), {"claim": {"id", "ssn"}})
    assert [(n.object, n.columns) for n in narrowed] == [("claim", True)]
    assert "null as ssn" in sql


def test_an_unqualified_reference_on_a_single_table_masks():
    sql, narrowed = _run("SELECT ssn FROM claim", AccessPolicy(masked={("claim", "ssn")}))
    assert [(n.object, n.columns) for n in narrowed] == [("claim", True)]
    assert "null as ssn" in sql


def test_a_query_touching_no_masked_column_claims_nothing():
    sql, narrowed = _run("SELECT id FROM person p JOIN claim c ON p.id = c.id",
                         AccessPolicy(masked={("claim", "ssn")}))
    assert narrowed == []
    assert "null as ssn" not in sql
