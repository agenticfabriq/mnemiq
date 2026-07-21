import dataclasses
import os

from mnemiq.config import Settings


def _settings(**env):
    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        return Settings.from_env()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_verify_override_tristate():
    assert _settings(MNEMIQ_VERIFY=None).verify_override is None
    assert _settings(MNEMIQ_VERIFY="0").verify_override == "0"
    assert _settings(MNEMIQ_VERIFY="1").verify_override == "1"
    # eval back-compat: the bool stays "== 1"
    assert _settings(MNEMIQ_VERIFY="1").verify is True
    assert _settings(MNEMIQ_VERIFY="0").verify is False


def test_resolve_verify_level():
    from mnemiq.runtime import _resolve_verify_level

    assert _resolve_verify_level("sanity", None) == "sanity"
    assert _resolve_verify_level("full", None) == "full"
    assert _resolve_verify_level("sanity", "0") == "off"     # force off
    assert _resolve_verify_level("full", "0") == "off"
    assert _resolve_verify_level("sanity", "1") == "full"    # force full
    assert _resolve_verify_level("off", "1") == "full"


def test_build_verifier_layers():
    from mnemiq.runtime import _build_verifier
    from mnemiq.verify.judge import FakeJudge

    judge = FakeJudge(0.9)
    assert _build_verifier("off", threshold=0.5, grounding=False, judge=judge) is None
    sanity_only = _build_verifier("sanity", threshold=0.5, grounding=False, judge=judge)
    assert sanity_only is not None and sanity_only.judge is None
    full = _build_verifier("full", threshold=0.5, grounding=False, judge=judge)
    assert full is not None and full.judge is judge
