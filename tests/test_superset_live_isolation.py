"""The live Superset suite never seeds the integration database, even when the shell running it names one.

The root conftest's `_seed_acme` loads ACME into MNEMIQ_PG_DSN, dropping each ACME table first, once per session whenever
integration tests are selected and MNEMIQ_ACME_DATA_DIR is set as well. The live suite never needs ACME, and a
session-scoped fixture runs before the suite's own environment cleaning and skip gate, so a shell holding those two
variables would have had that database seeded by a run whose live tests all skip.

Each test here runs pytest in a process of its own, with MNEMIQ_PG_DSN pointing at a TCP listener the test owns, which
counts the connections it accepts and closes each one unanswered. No database is ever reached.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = "tests/superset_live"


class _Listener:
    """A TCP port on 127.0.0.1 that counts each connection it accepts and closes it at once."""

    def __init__(self):
        self.accepted = 0
        self._server = socket.create_server(("127.0.0.1", 0))
        self._server.settimeout(0.1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self.port = self._server.getsockname()[1]

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            self.accepted += 1
            conn.close()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._server.close()


def _integration_run(tmp_path: Path, port: int, *paths: str) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
    """`pytest -m integration` over `paths`, with both seeding variables set and nothing else of mnemiq's in the
    environment, so none of the live suite's variables. Returns the run and each test's outcome, by node id.

    The data directory holds an empty ACME DDL and no CSVs: `seed` reads the DDL, then connects before it reads any
    CSV, so a seeder that runs reaches the listener."""
    data_dir = tmp_path / "acme"
    (data_dir / "DDL").mkdir(parents=True)
    (data_dir / "DDL" / "ACME_small.ddl").write_text("")
    (data_dir / "data").mkdir()
    env = {name: value for name, value in os.environ.items() if not name.upper().startswith(("MNEMIQ_", "PG"))}
    env["MNEMIQ_PG_DSN"] = f"postgresql://nobody@127.0.0.1:{port}/acme?sslmode=disable&connect_timeout=5"
    env["MNEMIQ_ACME_DATA_DIR"] = str(data_dir)
    report = tmp_path / "report.xml"
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "integration", *paths, "-p", "no:randomly",
         "-p", "no:cacheprovider", f"--junitxml={report}"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    outcomes = {}
    if report.exists():
        for case in ET.parse(report).getroot().iter("testcase"):
            kinds = [child.tag for child in case if child.tag in ("skipped", "error", "failure")]
            outcomes[f"{case.get('classname')}.{case.get('name')}"] = kinds[0] if kinds else "passed"
    return run, outcomes


def _said(run: subprocess.CompletedProcess) -> str:
    return f"exit {run.returncode}\n{run.stdout[-3000:]}\n{run.stderr[-2000:]}"


def _live(outcomes: dict[str, str]) -> dict[str, str]:
    return {test: outcome for test, outcome in outcomes.items() if test.startswith("tests.superset_live.")}


def test_the_live_suite_never_seeds_the_integration_database(tmp_path):
    """Integration tests selected, both seeding variables set, none of the live suite's: every live test skips, and
    nothing connects to MNEMIQ_PG_DSN."""
    with _Listener() as listener:
        run, outcomes = _integration_run(tmp_path, listener.port, LIVE)
    assert listener.accepted == 0, f"{listener.accepted} connection(s) to MNEMIQ_PG_DSN\n{_said(run)}"
    assert run.returncode == 0, _said(run)
    assert outcomes and outcomes == _live(outcomes), _said(run)
    assert set(outcomes.values()) == {"skipped"}, _said(run)


def test_an_integration_test_elsewhere_still_gets_the_seed(tmp_path):
    """The same run with the seeder's own test added. The root seeder still runs for a test outside the live suite and
    reaches the listener, which is what makes the zero above a measurement: the same variables and data directory do
    reach a connection whenever the root seeder runs. The live tests still all skip."""
    with _Listener() as listener:
        run, outcomes = _integration_run(tmp_path, listener.port, LIVE, "tests/test_seed_acme.py")
    assert listener.accepted >= 1, f"the root seeder never connected\n{_said(run)}"
    live = _live(outcomes)
    assert live and set(live.values()) == {"skipped"}, _said(run)
    assert len(outcomes) > len(live), _said(run)
