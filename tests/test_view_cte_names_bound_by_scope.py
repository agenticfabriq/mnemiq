"""Codex's review of #132 (M45): a CTE alias exempts a table from the unknown-object check only
where the reference actually binds to that CTE.

The check pooled every CTE alias in a view body and matched it by bare name, so a body declaring
`WITH secret AS (...)` and reading `main.secret` -- a real table the snapshot does not know -- was
approved, with any inventory, empty or not; and so was a `secret` read outside the scope the CTE
is declared in. A table is now exempt only when it is unqualified and a CTE of its name is in
scope where it is read: `scope._cte_in_scope`, the binding the deciders already use (DuckDB's, the
narrowest of the engines measured, so it never exempts a read an engine makes of the table).
"""

from __future__ import annotations

import pytest

from mnemiq.contract import ViewDefinition
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal, RefusalCode

INVENTORIES = {"empty": {}, "without secret": {"customer": {"id"}, "orders": {"id", "store_id"}}}


def _decide(body, schema):
    policy = AccessPolicy(row_filters={"orders": "store_id = 1"}, policy_schema=schema)
    views = {"v": ViewDefinition(object_id="v", dialect="duckdb", definition=body)}
    return decide("SELECT id FROM v", {"v": {"id"}}, dialect="duckdb", target="duckdb",
                  policy=policy, views=views)


@pytest.mark.parametrize("inventory", INVENTORIES)
@pytest.mark.parametrize("body", [
    "WITH secret AS (SELECT 1 AS id) SELECT id FROM main.secret",
    "SELECT id FROM (WITH secret AS (SELECT 1 AS id) SELECT id FROM secret) x "
    "UNION ALL SELECT id FROM secret",
])
def test_a_cte_name_does_not_hide_a_table_it_does_not_bind(inventory, body):
    verdict = _decide(body, INVENTORIES[inventory])
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW, verdict


def test_a_reference_the_cte_binds_is_still_exempt():
    """The control: an ordinary CTE view over a known table is approved, as before."""
    verdict = _decide("WITH c AS (SELECT id FROM customer) SELECT id FROM c",
                      INVENTORIES["without secret"])
    assert not isinstance(verdict, Refusal), verdict


def test_a_cte_named_like_a_filtered_table_still_trips_the_filtered_check():
    """Scope decides only what counts as unknown; the filtered check still sees every mention."""
    verdict = _decide("WITH orders AS (SELECT 1 AS id) SELECT id FROM orders",
                      INVENTORIES["without secret"])
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNGOVERNED_VIEW, verdict
