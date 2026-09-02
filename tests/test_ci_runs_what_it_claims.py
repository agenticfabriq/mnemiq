"""CI's own selection, asserted two ways.

`test_oracle_nodb.py` opens with `pytest.importorskip("oracledb")`, which fails OPEN: drop the
driver and seven tests -- the lease-leak guard and the expiring-`constrained` guard among them --
become silent SKIPPED lines under a green build. That is the same fail-open the file was created to
end, one level up, and it had already happened once: the driver lives in the `oracle` extra and
CI's sync line did not name it.

Two checks, because neither covers the other and the first drafts of both disarmed too easily:

  * the workflow TEXT, for the extra. It needs no driver, no env var and no CI, so it holds
    wherever the suite runs rather than only where the thing it guards already works.
  * pytest's OWN collection under CI's marker expression, for everything else. A textual guard
    against a module-level `pytestmark` reads as protection and is not: seven function-level
    `@pytest.mark.integration` decorators deselect exactly the same tests and leave the text
    clean. Asking the collector how many tests survive cannot be fooled by where a mark is
    written, or by reformatting the `importorskip` line it used to key on.
"""

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NODB = ROOT / "tests" / "test_oracle_nodb.py"
CI = ROOT / ".github" / "workflows" / "ci.yml"
MARKERS = "not integration and not live_llm"


def _the_job_that_runs_pytest() -> list[str]:
    """The `run:` lines of the one job that runs pytest.

    Split on the job keys rather than parsed: pyyaml is not a dependency of this project, and
    adding one so a guard can read a 30-line workflow is a worse trade than a block scan.
    """
    blocks, current = {}, None
    for line in CI.read_text().splitlines():
        if re.fullmatch(r"  ([\w-]+):", line):
            current = line.strip().rstrip(":")
            blocks[current] = []
        elif current and line.startswith("    "):
            blocks[current].append(line)
    hits = [name for name, body in blocks.items() if any("pytest" in ln for ln in body)]
    assert len(hits) == 1, f"expected exactly one pytest job in ci.yml, found {hits}"
    return blocks[hits[0]]


@pytest.mark.skipif(not NODB.exists(), reason="the module this guards is gone")
def test_ci_installs_the_extra_the_oracle_tests_import():
    # Scoped to the job that actually runs pytest: a docs or lint job syncing a narrower set is
    # correct, and `all()` over every sync line in the file would turn the suite red for it.
    syncs = [ln for ln in _the_job_that_runs_pytest() if "uv sync" in ln]
    assert syncs, "the pytest job has no `uv sync` step to check"
    assert all("--extra oracle" in ln for ln in syncs), (
        "tests/test_oracle_nodb.py skips itself without `oracledb`, and the job that runs pytest "
        f"does not install it: {syncs}. Seven tests would report SKIPPED, build still green."
    )


@pytest.mark.skipif(not NODB.exists(), reason="the module this guards is gone")
def test_cis_marker_expression_actually_selects_those_tests():
    pytest.importorskip("oracledb", reason="without the driver this measures the environment")
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", MARKERS, str(NODB)],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    selected = sum(1 for ln in out.stdout.splitlines() if "::test_" in ln)
    assert selected > 0, (
        f"CI runs `-m \"{MARKERS}\"` and it selects nothing from {NODB.name}: the database-free "
        f"guards are deselected by a mark and the build stays green.\n{out.stdout[-800:]}"
    )
