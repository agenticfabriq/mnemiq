"""M45: a source that knows no objects is not a caller that supplied none.

`check_views` refuses a view whose body reads an object the source does not know. It took
`known or set()` and then ran the rule `if known:`, so an EMPTY set -- a snapshot holding no
objects -- read as "not supplied" and skipped the rule: the fail-open direction, where every body
object should have been unknown. `None` now means not supplied and skips, as before; an empty set
means the source knows nothing, so every object a body reads is unknown and the view is refused.
"""

from __future__ import annotations

import sqlglot

from mnemiq.contract import ViewDefinition
from mnemiq.sql.decide import decide
from mnemiq.sql.policy import AccessPolicy
from mnemiq.sql.verdict import Refusal, RefusalCode
from mnemiq.sql.views import check_views

VIEWS = {"v": ViewDefinition(object_id="v", dialect="duckdb", definition="SELECT id FROM anything")}
FILTERED = {"customer"}


def _check(known):
    return check_views(sqlglot.parse_one("SELECT id FROM v", read="duckdb"), VIEWS, FILTERED,
                       known=known)


def test_an_empty_known_set_refuses_every_object_a_body_reads():
    verdict = _check(set())
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW


def test_not_supplying_known_still_skips_the_rule():
    """The control: a caller that passes no inventory of known objects is not checked by it."""
    assert _check(None) is None


def test_the_decider_refuses_with_a_policy_that_knows_no_objects():
    """The join: both deciders pass `set(policy.policy_schema)`, which is empty for a policy
    built from a snapshot holding no columns."""
    policy = AccessPolicy(row_filters={"customer": "store_id = 1"}, policy_schema={})
    verdict = decide("SELECT id FROM v", {"v": {"id"}}, dialect="duckdb", target="duckdb",
                     policy=policy, views=VIEWS)
    assert isinstance(verdict, Refusal) and verdict.code is RefusalCode.UNRESOLVABLE_VIEW
