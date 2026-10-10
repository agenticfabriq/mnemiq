"""**M136.** A dimension is offered only to an identity that reads every column it reads.

`select_dimensions` read the dimension's own `pii_level`, while since M135's option A a column
requires every level the personal dimensions over it carry. With `region` (pii) and
`region_health` (phi) both certified over `customer.region`, a pii-only identity was offered
`region` and denied its column, a phi-only one offered `region_health` and denied it too --
nothing leaked, the column was refused, but the packet invited a grouping the decider will not
run. Resolved with the parser M135 stamps the column with, so the two cannot disagree.
"""

from mnemiq.contract import Dimension
from mnemiq.enrichment.certified import apply_certified
from mnemiq.semantic.measures import select_dimensions
from tests.test_personal_dimension import _customers, _grants, _record


def _shown(snapshot, grants):
    return [d.id for d in select_dimensions(["customer"], snapshot.dimensions, grants,
                                            columns=snapshot.columns)]


def test_two_personal_dimensions_over_one_column_are_offered_only_to_the_doubly_cleared():
    snap = apply_certified(_customers(), [_record("region", "region", "pii"),
                                          _record("region_health", "`region`", "phi")])
    assert _shown(snap, _grants({"pii"})) == [], "region's column requires phi as well"
    assert _shown(snap, _grants({"phi"})) == [], "and region_health's requires pii"
    assert _shown(snap, _grants({"pii", "phi"})) == ["region", "region_health"]


def test_a_plain_dimension_over_a_personal_column_is_offered_only_to_the_cleared():
    snap = _customers("pii").model_copy(update={"dimensions": [
        Dimension(id="ssn_prefix", label="SSN prefix", source="customer",
                  expr="substr(ssn, 1, 3)")]})
    assert _shown(snap, _grants()) == []
    assert _shown(snap, _grants({"pii"})) == ["ssn_prefix"]


def test_a_dimension_over_open_columns_is_offered_to_everyone():
    snap = _customers().model_copy(update={"dimensions": [
        Dimension(id="region", label="Region", source="customer", expr="region")]})
    assert _shown(snap, _grants()) == ["region"]


def test_retrieve_offers_the_packet_only_the_dimensions_whose_columns_are_read(tmp_path):
    """The door: `retrieve`, which the runtime and the eval engine both call, hands the snapshot's
    columns to `select_dimensions` -- a dimension over a column carrying phi stays out of a
    pii-only identity's packet though the dimension itself is pii."""
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract import Column
    from mnemiq.llm.embeddings import FakeEmbedder
    from mnemiq.semantic.retrieval import retrieve
    from tests.test_retrieval import _con, _identity

    column = Column(id="claim.claim_identifier", object_id="claim", name="claim_identifier",
                    data_type="text", dimension_pii_levels=["phi", "pii"])
    dimension = Dimension(id="claim_ref", label="Claim reference", source="claim",
                          expr="claim_identifier", pii_level="pii")

    class _Cleared:
        def __init__(self, *levels):
            self._grants = GrantSet(frozenset({"claim"}), pii_clearance=frozenset(levels))

        def grants_for(self, identity):
            return self._grants

    def offered(*levels):
        packet = retrieve(_con(tmp_path), "claim_identifier", _identity(), _Cleared(*levels),
                          FakeEmbedder(), dimensions=[dimension], columns=[column])
        assert [c.object_id for c in packet.cards] == ["claim"], "setup: the table is in context"
        return [d.id for d in packet.dimensions]

    assert offered("pii") == []
    assert offered("pii", "phi") == ["claim_ref"]
