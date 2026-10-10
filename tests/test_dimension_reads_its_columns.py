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


# Codex's review of #95, both reproduced.

def test_case_distinct_columns_under_one_folded_name_each_require_their_clearance():
    """`customer."SSN"` (pii) and `customer.ssn` (plain) are two columns in Postgres. Keyed by the
    folded name, the selector kept whichever came last, so the order decided the clearance."""
    from mnemiq.contract import Column, Snapshot

    pii = Column(id='customer."SSN"', object_id="customer", name="SSN", data_type="text",
                 pii_level="pii")
    plain = Column(id="customer.ssn", object_id="customer", name="ssn", data_type="text")
    dimension = Dimension(id="ssn_dim", label="SSN", source="customer", expr='"SSN"')
    for columns in ([pii, plain], [plain, pii]):
        snap = Snapshot(version="v", source_id="s", created_at="t", columns=columns,
                        dimensions=[dimension])
        assert _shown(snap, _grants()) == [], [c.name for c in columns]
        assert _shown(snap, _grants({"pii"})) == ["ssn_dim"]


def test_a_dimension_with_no_expression_is_read_as_the_prompt_renders_it():
    """The prompt shows a dimension as `expr or id`; the selector parsed an empty string, so a
    dimension `ssn` with no expression, over a personal `ssn`, was offered to the uncleared."""
    for expr in (None, ""):
        snap = _customers("pii").model_copy(update={"dimensions": [
            Dimension(id="ssn", label="SSN", source="customer", expr=expr)]})
        assert _shown(snap, _grants()) == [], repr(expr)
        assert _shown(snap, _grants({"pii"})) == ["ssn"]


def test_a_personal_dimension_with_no_expression_classifies_the_column_its_id_names():
    """M135's stamping read `expr or ""` too, so a personal dimension without an expression
    classified no column -- the one its id names, and the prompt shows, stayed open."""
    from mnemiq.contract import CertifiedRecord

    record = CertifiedRecord.model_validate({
        "envelope": {"object_type": "dimension", "object_id": "ssn", "version": "v1",
                     "source_system": "verity", "provenance": {"status": "certified",
                                                               "certifier": "reviewer@acme"}},
        "payload": {"id": "ssn", "label": "SSN", "source": "customer", "pii_level": "pii"},
    })
    out = apply_certified(_customers(), [record])
    ssn = next(c for c in out.columns if c.name == "ssn")
    assert ssn.pii_levels() == frozenset({"pii"})
