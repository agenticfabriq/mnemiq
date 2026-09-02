"""CI's marker expression is asked what it selects, not told.

`tests/test_oracle_nodb.py` holds the database-free guards for the Oracle read-only controls --
the lease-leak fix and the expiring-`constrained` fix among them. They were unmeasured twice over:
the module they lived in skipped without a live DSN, and after the split an `importorskip` skipped
them wherever the driver was absent, which included CI.

The driver hole is closed by construction now: `oracledb` is a dev dependency and the module no
longer skips past it, so a missing driver turns the build red -- as four errors and three passes,
since the adapter imports the driver lazily -- rather than green with seven skips. What
construction cannot close is a MARK -- module-level or per-function, `integration` or a new one
minted tomorrow -- silently removing these tests from CI's SELECTION. Two holes remain open and
are recorded rather than papered over: a `@pytest.mark.skip` suppresses execution without changing
what is collected, so counting collected tests cannot see it; and CI naming explicit paths would
drop this file while the check, which collects it by path, stayed green. Both need a different
instrument than a collection count.

That is what this asks about,
and it asks the collector rather than reading the file, because every textual version of this check
was defeated by moving a mark or requoting a line.

Both counts come from the same collector, differing only in `-m`. An earlier draft compared
collected ITEMS against `def test_` lines: one added `parametrize` and one added `integration` mark
cancel out to an equal count, which is the fail-open it was written to prevent.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NODB = ROOT / "tests" / "test_oracle_nodb.py"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _cis_pytest_invocation() -> str:
    """The one `pytest` run line in the job that runs pytest.

    Uniqueness is asserted rather than assumed: returning the first of several would validate one
    invocation and stay green for another.
    """
    runs = [ln for ln in CI.read_text().splitlines()
            if re.search(r"\brun:.*\bpytest\b", ln)]
    assert len(runs) == 1, f"expected exactly one pytest invocation in ci.yml, found {runs}"
    return runs[0]


def _collect(*marker_args: str) -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *marker_args, str(NODB)],
        cwd=ROOT, capture_output=True, text=True, timeout=180)
    return sum(1 for ln in out.stdout.splitlines() if "::test_" in ln)


def test_cis_marker_expression_selects_every_database_free_oracle_test():
    line = _cis_pytest_invocation()
    m = re.search(r"""-m\s+(["'])(.+?)\1""", line)
    assert m, f"cannot read a marker expression out of CI's pytest line: {line.strip()}"
    markers = m.group(2)

    everything = _collect()
    # The floor. Without it a file that collects nothing at all -- gutted, or its tests moved under
    # a class -- makes both sides zero and the equality vacuously true.
    assert everything > 0, f"{NODB.name} collects no tests at all; nothing here is measuring CI"

    selected = _collect("-m", markers)
    assert selected == everything, (
        f"CI runs `-m \"{markers}\"`, which selects {selected} of {NODB.name}'s {everything} "
        f"tests. The rest carry a mark that deselects them, and the build stays green without "
        f"the Oracle read-only guards ever running."
    )
