import pyarrow as pa

from mnemiq.semantic.retrieval import ContextPacket
from mnemiq.sql.verdict import Approved
from mnemiq.verify.verifier import Verifier


def _packet(q="how many?"):
    return ContextPacket(question=q, cards=[], grant_fingerprint="", enrichment_version=None)


def _approved(sql="SELECT COUNT(*) FROM t"):
    return Approved(plan_sql=sql, target_sql=sql)


def test_sanity_short_circuits_before_grounding():
    # empty result AND an ungrounded 'SME' -> sanity wins (first in cascade)
    v = Verifier().verify(_packet("count in 'SME'"), _approved(), pa.table({"n": pa.array([], type=pa.int64())}))
    assert v.defer and v.layer == "sanity"


def test_grounding_fires_when_sanity_passes():
    v = Verifier().verify(_packet("total for 'SME'"), _approved("SELECT SUM(x) FROM t"), pa.table({"s": [10]}))
    assert v.defer and v.layer == "grounding"


def test_clean_result_passes():
    v = Verifier().verify(_packet(), _approved(), pa.table({"n": [42]}))
    assert not v.defer and v.layer == "pass"


def test_disabled_layers_pass_through():
    v = Verifier(sanity=False, grounding=False).verify(
        _packet("x for 'SME'"), _approved(), pa.table({"n": pa.array([], type=pa.int64())})
    )
    assert not v.defer and v.layer == "pass"
