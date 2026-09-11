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
    assert _build_verifier("off", threshold=0.5, grounding=False, judge=judge, fail_closed=True) is None
    sanity_only = _build_verifier("sanity", threshold=0.5, grounding=False, judge=judge, fail_closed=True)
    assert sanity_only is not None and sanity_only.judge is None
    full = _build_verifier("full", threshold=0.5, grounding=False, judge=judge, fail_closed=True)
    assert full is not None and full.judge is judge


def _base_settings(**over):
    s = Settings(llm_base_url="x", llm_api_key="k", llm_model="m", pg_dsn="d",
                 acme_data_dir="a")
    return s.model_copy(update=over)  # pydantic model (was dataclasses.replace)


def test_mode_verifiers_default_policy():
    from mnemiq.runtime import _build_mode_verifiers

    vs, judge_calls = _build_mode_verifiers(_base_settings(), client=object())
    assert vs["instant"].judge is None          # sanity only
    assert vs["thinking"].judge is None          # sanity only
    assert vs["deep"].judge is not None          # sanity + judge
    assert judge_calls == 1                       # judge built exactly once (shared)


def test_mode_verifiers_force_off():
    from mnemiq.runtime import _build_mode_verifiers

    vs, _ = _build_mode_verifiers(_base_settings(verify_override="0"), client=object())
    assert vs["instant"] is None and vs["thinking"] is None and vs["deep"] is None


def test_mode_verifiers_force_full():
    from mnemiq.runtime import _build_mode_verifiers

    vs, judge_calls = _build_mode_verifiers(_base_settings(verify_override="1"), client=object())
    assert vs["instant"].judge is not None and vs["thinking"].judge is not None
    assert judge_calls == 1


def test_mode_verifiers_judge_endpoint_override():
    # verify_model set -> judge client built via settings.model_copy (the ex-dataclasses.replace path)
    from mnemiq.runtime import _build_mode_verifiers

    vs, jc = _build_mode_verifiers(_base_settings(verify_model="judge-model"), client=object())
    assert vs["deep"].judge is not None and jc == 1


def test_the_deployment_switch_reaches_the_verifier_the_product_uses():
    """`_build_verifier` taking the argument proves nothing about the runtime passing it. Dropping
    `fail_closed=settings.verify_fail_closed` at the call site leaves every unit test above green
    and puts the deployment back on the constructor default, which is the shape of a setting that
    exists in the config, is documented in `.env.example`, and does nothing (issue #2).

    Both positions, because pinning only the default is satisfied by hardcoding it.
    """
    from mnemiq.runtime import _build_mode_verifiers

    for configured in (True, False):
        verifiers, _ = _build_mode_verifiers(
            _base_settings(verify_fail_closed=configured), client=object())
        built = [v for v in verifiers.values() if v is not None]
        assert built, "no verifier was built, so this asserts nothing"
        assert all(v.fail_closed is configured for v in built), configured
