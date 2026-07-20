from mnemiq.verify.verdict import VerifyVerdict


def test_passed_is_a_non_deferring_full_confidence_verdict():
    v = VerifyVerdict.passed()
    assert v.confidence == 1.0 and v.defer is False and v.layer == "pass"


def test_fields_round_trip():
    v = VerifyVerdict(confidence=0.2, defer=True, reason="nope", layer="sanity")
    assert (v.confidence, v.defer, v.reason, v.layer) == (0.2, True, "nope", "sanity")
