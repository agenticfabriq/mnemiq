"""CI's own selection, asserted statically.

`test_oracle_nodb.py` opens with `pytest.importorskip("oracledb")`, which fails OPEN: drop the
driver and seven tests -- the lease-leak guard and the expiring-`constrained` guard among them --
become silent SKIPPED lines under a green build. That is the same fail-open the file was created to
end, one level up, and it had already happened once: the driver lives in the `oracle` extra, and
CI's sync line did not name it.

Checking the workflow text needs no driver, no env var and no CI, so it holds wherever the suite
runs rather than only where the thing it guards is already working.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_ci_installs_the_extra_the_oracle_tests_import():
    nodb = ROOT / "tests" / "test_oracle_nodb.py"
    if not nodb.exists() or "importorskip(\"oracledb\"" not in nodb.read_text():
        return  # the gate this guards is gone; nothing left to protect

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    sync = [ln for ln in ci.splitlines() if "uv sync" in ln]
    assert sync, "no `uv sync` line in ci.yml to check"
    assert all("--extra oracle" in ln for ln in sync), (
        "tests/test_oracle_nodb.py skips itself without `oracledb`, and CI does not install it: "
        f"{sync}. Seven tests would report SKIPPED and the build would stay green."
    )


def test_the_database_free_oracle_tests_are_not_marked_integration():
    """CI runs `-m "not integration"`. The module they were split out of carries that mark at
    module level, and inheriting it would filter them out just as effectively as a missing driver.
    """
    nodb = ROOT / "tests" / "test_oracle_nodb.py"
    if not nodb.exists():
        return
    assert not re.search(r"^pytestmark\s*=", nodb.read_text(), re.M), (
        "test_oracle_nodb.py carries a module-level mark; CI's `-m \"not integration\"` may drop it"
    )
