"""Every Snowflake connection keeps certificate revocation checks on.

snowflake-connector-python 4.8 made OCSP checks opt-in: a connection given no `ocsp_fail_open`
checks no certificate's revocation status, where 4.7 checked it fail-open by default. The scripts
each connected with a profile name alone, so the upgrade would have turned the check off for all
of them without a line changing (Codex's review of the bump). They connect through
`snowflake_connection` now, which opts in.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from mnemiq.eval.warehouse import snowflake_connection

ROOT = Path(__file__).resolve().parent.parent


def _captured(monkeypatch) -> dict:
    """The keyword arguments the helper hands the driver, read through a stand-in driver so this
    runs where the optional extra is not installed."""
    seen: dict = {}
    connector = types.ModuleType("snowflake.connector")
    connector.connect = lambda **kwargs: seen.update(kwargs) or "connection"
    package = types.ModuleType("snowflake")
    package.connector = connector
    monkeypatch.setitem(sys.modules, "snowflake", package)
    monkeypatch.setitem(sys.modules, "snowflake.connector", connector)
    assert snowflake_connection("bench") == "connection"
    return seen


def test_the_helper_opts_into_fail_open_revocation_checks(monkeypatch):
    assert _captured(monkeypatch) == {"connection_name": "bench", "ocsp_fail_open": True}


def test_the_driver_resolves_the_helper_s_arguments_to_checks_on_and_a_name_alone_to_off(monkeypatch):
    # The driver's own resolution, so the claim about 4.8's default is the driver's, not ours.
    pytest.importorskip("snowflake.connector")
    from snowflake.connector._ocsp_mode import OCSPMode, resolve_ocsp_mode, snapshot_ocsp_explicit_params

    def mode(kwargs: dict) -> OCSPMode:
        return resolve_ocsp_mode(explicit=snapshot_ocsp_explicit_params(kwargs), disable_ocsp_checks=False,
                                 ocsp_fail_open=kwargs.get("ocsp_fail_open"))

    kwargs = _captured(monkeypatch)
    monkeypatch.undo()  # the real driver again
    assert mode(kwargs) is OCSPMode.FAIL_OPEN
    assert mode({"connection_name": "bench"}) is OCSPMode.DISABLE_OCSP_CHECKS


def test_nothing_connects_to_snowflake_except_through_the_helper():
    helper = ROOT / "src" / "mnemiq" / "eval" / "warehouse.py"
    direct = re.compile(r"snowflake\.connector\.connect\(|from snowflake\.connector import[^\n]*\bconnect\b"
                        r"|from snowflake import[^\n]*\bconnector\b")
    offenders = [str(path.relative_to(ROOT)) for path in [*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py")]
                 if path != helper and direct.search(path.read_text())]
    assert offenders == []
