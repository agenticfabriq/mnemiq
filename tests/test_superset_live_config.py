"""The live suite's Superset configuration (`tests/superset_live/superset_config.py`), read as Superset reads it, with
its one Superset import stood in: what it sets from the keys directory, and that a keys directory made before the
token route's settings existed still lets Superset start."""

import json
import logging
import runpy
import sys
import types
from pathlib import Path

CONFIG = Path(__file__).parent / "superset_live" / "superset_config.py"


def _keys(folder: Path, **issuer) -> Path:
    (folder / "issuer.json").write_text(json.dumps({"issuer": "https://idp.invalid/realms/live", "audience": "live",
                                                    "kid": "live-1", **issuer}))
    (folder / "public.pem").write_text("public key")
    (folder / "secret_key").write_text("secret")
    return folder


def _load(monkeypatch, keys: Path) -> dict:
    mcp_config = types.ModuleType("superset.mcp_service.mcp_config")
    mcp_config.MCP_TOOL_SEARCH_CONFIG = {"enabled": True}
    for name in ("superset", "superset.mcp_service"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "superset.mcp_service.mcp_config", mcp_config)
    monkeypatch.setenv("MNEMIQ_TEST_SUPERSET_KEYS_DIR", str(keys))
    monkeypatch.setenv("MNEMIQ_TEST_SUPERSET_METADATA_URI", "postgresql://live@127.0.0.1:1/meta")
    for name in ("MNEMIQ_TEST_SUPERSET_EXTENSION_DIR", "MNEMIQ_TEST_SUPERSET_DATA_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    return runpy.run_path(str(CONFIG))


def test_the_token_route_signs_as_the_test_identity_provider(monkeypatch, tmp_path):
    config = _load(monkeypatch, _keys(tmp_path, tenant="live"))
    assert (config["MNEMIQ_TOKEN_KID"], config["MNEMIQ_TOKEN_ISSUER"], config["MNEMIQ_TOKEN_AUDIENCE"],
            config["MNEMIQ_TOKEN_TENANT"]) == ("live-1", "https://idp.invalid/realms/live", "live", "live")
    assert config["MNEMIQ_TOKEN_KEY_FILE"] == str(tmp_path / "private.pem")
    assert "FEATURE_FLAGS" not in config  # no extension without MNEMIQ_TEST_SUPERSET_EXTENSION_DIR


def test_a_keys_dir_made_before_the_tenant_still_starts_superset(monkeypatch, tmp_path, caplog):
    """`issuer.json` written before it carried a tenant: Superset starts, the token route goes unconfigured (it
    answers 503 naming `MNEMIQ_TOKEN_TENANT`), and the log says to run `setup_superset.py keys` again."""
    with caplog.at_level(logging.WARNING):
        config = _load(monkeypatch, _keys(tmp_path))
    assert "MNEMIQ_TOKEN_TENANT" not in config
    assert "setup_superset.py keys" in caplog.text
