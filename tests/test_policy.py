from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Snapshot
from mnemiq.sql.policy import AccessPolicy, build_access_policy


def _snap():
    return Snapshot(version="v1", source_id="acme", created_at="t", columns=[
        Column(id="c.id", object_id="claim", name="id", pii_level="none"),
        Column(id="c.ssn", object_id="claim", name="ssn", pii_level="pii"),
        Column(id="c.dx", object_id="claim", name="dx", pii_level="phi"),
        Column(id="p.x", object_id="party", name="x", pii_level="pii"),  # party not readable
    ])


def test_build_access_policy_from_pii_and_clearance():
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": "region='US'", "party": "1=1"},
                      pii_clearance=frozenset({"pii"}), pii_mask=frozenset({"phi"}))
    pol = build_access_policy(_snap(), grants)
    assert pol.denied == set()                       # pii cleared, phi masked -> none denied
    assert pol.masked == {("claim", "dx")}           # phi -> masked
    # `party` is dropped -- not for being ungranted, but for being UNREACHABLE. A caller
    # granted only a view has no grant on the tables behind it, and after inlining those are
    # what the query reads, so the rule is reachability (grants + the bases of granted views).
    # Here `claim` is not a view, so reachable == granted and the outcome is unchanged.
    assert pol.row_filters == {"claim": "region='US'"}
    assert not pol.empty


def test_no_clearance_denies_all_pii():
    grants = GrantSet(frozenset({"claim"}))  # no clearance, no mask
    pol = build_access_policy(_snap(), grants)
    assert pol.denied == {("claim", "ssn"), ("claim", "dx")} and pol.masked == set()


def test_empty_policy():
    assert AccessPolicy().empty
