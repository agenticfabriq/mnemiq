"""Every Snowflake connection keeps certificate revocation checks on, as its profile allows.

snowflake-connector-python 4.8 made OCSP checks opt-in: a connection given no `ocsp_fail_open`
checks no certificate's revocation status, where 4.7 checked it fail-open by default. The scripts
each connected with a profile name alone, so the upgrade would have turned the check off for all
of them without a line changing (Codex's review of the bump). They connect through
`snowflake_connection` now, which opts in where the profile is silent, and leaves a profile that
chose fail-closed or fail-open as it chose (Codex's second review: the driver lays the call's
arguments over the profile's, so an unconditional opt-in made fail-closed fail-open).
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from mnemiq.eval.warehouse import snowflake_connection

ROOT = Path(__file__).resolve().parent.parent
PROFILES = {
    "silent": {"account": "a", "user": "u"},
    "fail-closed": {"account": "a", "user": "u", "ocsp_fail_open": False},
    "fail-open": {"account": "a", "user": "u", "ocsp_fail_open": True},
}


def _captured(monkeypatch, profile: dict) -> dict:
    """The keyword arguments the helper hands the driver for a profile, read through a stand-in
    driver and configuration so this runs where the optional extra is not installed."""
    seen: dict = {}
    connector = types.ModuleType("snowflake.connector")
    connector.connect = lambda **kwargs: seen.update(kwargs) or "connection"
    config = types.ModuleType("snowflake.connector.config_manager")
    config.CONFIG_MANAGER = {"connections": {"bench": profile}}
    package = types.ModuleType("snowflake")
    package.connector = connector
    for name, module in (("snowflake", package), ("snowflake.connector", connector),
                         ("snowflake.connector.config_manager", config)):
        monkeypatch.setitem(sys.modules, name, module)
    assert snowflake_connection("bench") == "connection"
    return seen


@pytest.mark.parametrize(("profile", "expected"), [
    ("silent", {"connection_name": "bench", "ocsp_fail_open": True}),
    ("fail-closed", {"connection_name": "bench"}),
    ("fail-open", {"connection_name": "bench"}),
])
def test_the_helper_opts_in_only_where_the_profile_is_silent(monkeypatch, profile, expected):
    assert _captured(monkeypatch, PROFILES[profile]) == expected


_PROBE = """
import json
import snowflake.connector
import snowflake.connector.connection as connection
from mnemiq.eval.warehouse import snowflake_connection

connection.SnowflakeConnection._SnowflakeConnection__open_connection = lambda self: None  # no network
print(json.dumps({name: [snowflake_connection(name)._ocsp_mode().name,
                         snowflake.connector.connect(connection_name=name)._ocsp_mode().name]
                  for name in ("silent", "closed", "open")}))
"""


def test_the_real_driver_resolves_each_profile_through_the_helper_to_checks_on_as_it_chose(tmp_path):
    # The installed driver end to end: its own configuration reading a connections.toml, its own
    # merge of the call over the profile, its own resolution; only the network step is stubbed.
    # A subprocess, so the driver reads this SNOWFLAKE_HOME and no other.
    pytest.importorskip("snowflake.connector")
    import json
    import os
    import subprocess

    lines = []
    for name, profile in (("silent", "silent"), ("closed", "fail-closed"), ("open", "fail-open")):
        lines.append(f"[{name}]")
        for key, value in {**PROFILES[profile], "password": "unused"}.items():
            lines.append(f"{key} = {json.dumps(value)}")
    (tmp_path / "connections.toml").write_text("\n".join(lines) + "\n")
    (tmp_path / "connections.toml").chmod(0o600)
    import mnemiq

    # The mnemiq under test, not whichever one the interpreter has installed.
    source = str(Path(mnemiq.__file__).resolve().parent.parent)
    env = {**os.environ, "SNOWFLAKE_HOME": str(tmp_path),
           "PYTHONPATH": os.pathsep.join(filter(None, [source, os.environ.get("PYTHONPATH")]))}
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True).stdout
    modes = json.loads(out.strip().splitlines()[-1])
    assert modes == {
        "silent": ["FAIL_OPEN", "DISABLE_OCSP_CHECKS"],  # the name alone is 4.8's default: off
        "closed": ["FAIL_CLOSED", "FAIL_CLOSED"],  # the profile's choice survives the helper
        "open": ["FAIL_OPEN", "FAIL_OPEN"],
    }


def test_nothing_connects_to_snowflake_except_through_the_helper():
    helper = ROOT / "src" / "mnemiq" / "eval" / "warehouse.py"
    direct = re.compile(r"snowflake\.connector\.connect\(|from snowflake\.connector import[^\n]*\bconnect\b"
                        r"|from snowflake import[^\n]*\bconnector\b")
    offenders = [str(path.relative_to(ROOT)) for path in [*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py")]
                 if path != helper and direct.search(path.read_text())]
    assert offenders == []
