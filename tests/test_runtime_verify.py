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
