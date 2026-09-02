"""CI's own pytest invocation is replayed and asked what it collects.

`tests/test_oracle_nodb.py` holds the database-free guards for the Oracle read-only controls --
the lease-leak fix and the expiring-`constrained` fix among them. They have been unmeasured twice:
the module they lived in skipped without a live DSN, and after the split an `importorskip` skipped
them wherever the driver was absent, which included CI.

The driver hole is closed by construction: `oracledb` is a dev dependency and the module no longer
skips past it, so a missing driver turns the build red -- as four errors and three passes, since
the adapter imports the driver lazily -- rather than green with seven skips.

What construction cannot close is CI quietly not selecting these tests. Earlier drafts each read
ONE mechanism out of the workflow -- first a mark, then the `-m` expression -- and each was
defeated by a different one, which is the lesson: `-k`, `--deselect`, `--ignore` and an explicit
path argument all deselect just as silently, and enumerating them is a losing game. So CI's pytest
line is replayed verbatim, whatever it contains, and the question is whether these tests survive
it.

One hole is left and is named rather than implied away: a `@pytest.mark.skip` suppresses execution
without changing what is collected, so no collection-based check can see it.
"""

import pathlib
import re
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NODB = ROOT / "tests" / "test_oracle_nodb.py"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _cis_pytest_args() -> list[str]:
    """Everything CI passes to pytest, minus the runner prefix.

    Uniqueness is asserted rather than assumed: replaying the first of several invocations would
    validate one and stay green for the other.
    """
    runs = [ln for ln in CI.read_text().splitlines() if re.search(r"\brun:.*\bpytest\b", ln)]
    assert len(runs) == 1, f"expected exactly one pytest invocation in ci.yml, found {runs}"
    cmd = shlex.split(runs[0].split("run:", 1)[1].strip())
    args = cmd[cmd.index("pytest") + 1:]
    # Verbosity is dropped and nothing else is. It cannot change WHAT is selected, and it changes
    # the shape of `--collect-only` output -- CI's `-v` prints a node tree rather than `path::test`
    # lines, which parsed as zero tests collected and failed this check against a correct workflow.
    return [a for a in args if a not in {"-v", "-vv", "-vvv", "--verbose", "-q", "--quiet"}]


def _collect(args: list[str]) -> set[str]:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    return {ln.split("::", 1)[1].split(" ")[0]
            for ln in out.stdout.splitlines()
            if "test_oracle_nodb.py::" in ln}


def test_cis_invocation_collects_every_database_free_oracle_test():
    alone = _collect([str(NODB)])
    # The floor: without it, a file collecting nothing makes both sides empty and the comparison
    # vacuously true.
    assert alone, f"{NODB.name} collects no tests at all; nothing here is measuring CI"

    under_ci = _collect(_cis_pytest_args())
    missing = alone - under_ci
    assert not missing, (
        f"CI runs `pytest {' '.join(_cis_pytest_args())}` and it does not collect "
        f"{len(missing)} of {NODB.name}'s {len(alone)} tests: {sorted(missing)}. They are "
        f"deselected -- by a mark, a path, `-k`, `--ignore` or `--deselect` -- and the build "
        f"stays green with the Oracle read-only guards never running."
    )
