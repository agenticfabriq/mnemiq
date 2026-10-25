"""The repo guard reads what a commit will hold (M219, M220, M221). Each test builds a throwaway repository
around a copy of the script and runs it as the hook and CI do; the shapes it must block are assembled at run
time so this file holds none of them. M220: staged mode scanned the working-tree copy of each staged path,
so a blocked line staged and then removed from the file was committed unscanned; M221: `--diff-filter=ACM`
skipped a rename git detected; M219: `--range` read `git log -p`, which prints no patch for a merge commit,
so a line a conflict resolution added was never scanned.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repo-guard.sh"
BLOCKED_NAME = "zebrablock"
KEY_SHAPE = "AKIA" + "Q" * 16  # a secret shape the script blocks, never spelled whole in this file


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _guard(repo: Path, *args: str) -> int:
    env = {**os.environ, "REPO_GUARD_NAME_PATTERNS": BLOCKED_NAME}
    return subprocess.run(["bash", "scripts/repo-guard.sh", *args], cwd=repo, env=env,
                          capture_output=True, text=True).returncode


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    (r / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, r / "scripts" / "repo-guard.sh")
    _git(r, "init", "-q", "-b", "main", ".")
    _git(r, "config", "user.email", "dev@example.test")
    _git(r, "config", "user.name", "dev")
    (r / "a.txt").write_text("clean\n")
    (r / "b.txt").write_text("".join(f"line {i}\n" for i in range(40)))
    _git(r, "add", ".")
    _git(r, "commit", "-qm", "base")
    return r


def test_staged_mode_reads_the_index_not_the_working_copy(repo: Path) -> None:
    # M220: the word is in the index and gone from the working copy; the commit would hold it
    (repo / "a.txt").write_text(f"clean\n{BLOCKED_NAME}\n")
    _git(repo, "add", "a.txt")
    (repo / "a.txt").write_text("clean\n")
    assert BLOCKED_NAME in _git(repo, "show", ":a.txt")
    assert _guard(repo, "--staged") == 1
    # a staged file deleted from the working copy is scanned too
    (repo / "a.txt").unlink()
    assert _guard(repo, "--staged") == 1
    # control: the word in the working copy alone is not in the commit
    _git(repo, "reset", "-q", "--hard")
    (repo / "a.txt").write_text(f"clean\n{BLOCKED_NAME}\n")
    assert _guard(repo, "--staged") == 0
    # control: the word in both is blocked, as before
    _git(repo, "add", "a.txt")
    assert _guard(repo, "--staged") == 1


def test_staged_mode_scans_a_renamed_file(repo: Path) -> None:
    # M221: a rename git detects, with a blocked line appended to the renamed file
    _git(repo, "mv", "b.txt", "c.txt")
    with (repo / "c.txt").open("a") as f:
        f.write(f"{BLOCKED_NAME}\n")
    _git(repo, "add", "c.txt")
    status = _git(repo, "diff", "--cached", "--name-status", "-M")
    assert status.startswith("R"), status
    assert _guard(repo, "--staged") == 1
    # control: the rename alone passes, and the same edit without the rename is blocked
    _git(repo, "reset", "-q", "--hard")
    _git(repo, "mv", "b.txt", "c.txt")
    assert _guard(repo, "--staged") == 0
    _git(repo, "reset", "-q", "--hard")
    with (repo / "b.txt").open("a") as f:
        f.write(f"{BLOCKED_NAME}\n")
    _git(repo, "add", "b.txt")
    assert _guard(repo, "--staged") == 1


def test_both_modes_read_a_path_git_would_quote_and_staged_mode_a_type_change(repo: Path) -> None:
    # the review gate on the fix, twice: git C-quotes a non-ASCII path by default and a path holding a
    # quote always, and the quoted spelling names no file and no blob, so the tree scan skipped it and
    # the index scan refused it; a symlink replaced by a regular file is staged as a type change
    for name in ("caf\u00e9.txt", 'q"x.txt', "0:s.txt", "x\nscripts/repo-guard.sh"):
        # `0:s.txt` reads as stage 0 of `s.txt` unless the stage is spelled; the path with a newline
        # has the excluded script's path as its second line (the review gate on the fix)
        (repo / name).parent.mkdir(exist_ok=True)
        # a home path too: the tree scan never checks names, only shapes
        (repo / name).write_text(f"{BLOCKED_NAME}\n" + "/" + "Users/someone/work\n")
        _git(repo, "add", name)
        if name != "0:s.txt":
            assert _git(repo, "diff", "--cached", "--name-only").startswith('"'), name  # git quotes it
        assert _guard(repo, "--staged") == 1, name
        assert _guard(repo, "--all") == 1, name
        _git(repo, "reset", "-q", "--hard")  # drops the staged new file from the index and the tree
        shutil.rmtree(repo / "x\nscripts", ignore_errors=True)
    (repo / "link.txt").symlink_to("a.txt")
    _git(repo, "add", "link.txt")
    _git(repo, "commit", "-qm", "link")
    (repo / "link.txt").unlink()
    (repo / "link.txt").write_text(f"{BLOCKED_NAME}\n")
    _git(repo, "add", "link.txt")
    assert _git(repo, "diff", "--cached", "--name-status").startswith("T")
    assert _guard(repo, "--staged") == 1


def test_staged_mode_skips_a_submodule_pointer_and_scans_the_file_beside_it(repo: Path, tmp_path: Path) -> None:
    # a gitlink is a commit in another repository, not a blob here (the review gate on the fix: read as
    # a blob it was refused as an instrument fault, exit 2); the submodule is named `!` and the file
    # beside it `*`, so a lookup of the file's mode by a bare pathspec would take the gitlink's (the
    # same gate); the submodule's URL is relative, so no absolute path reaches `.gitmodules`
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q", "-b", "main", ".")
    _git(other, "config", "user.email", "dev@example.test")
    _git(other, "config", "user.name", "dev")
    (other / "o.txt").write_text("clean\n")
    _git(other, "add", ".")
    _git(other, "commit", "-qm", "base")
    _git(repo, "remote", "add", "origin", str(tmp_path / "r"))
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", "-q", "../other", "!")
    assert "160000" in _git(repo, "ls-files", "-s", "--", ":(literal)!")
    assert str(tmp_path) not in (repo / ".gitmodules").read_text()
    assert _guard(repo, "--staged") == 0
    (repo / "*").write_text(f"clean\n{BLOCKED_NAME}\n")
    _git(repo, "add", "--", "*")
    assert _guard(repo, "--staged") == 1


def _merge_resolving_with(repo: Path, resolution: str) -> None:
    """A conflict on a.txt between main and a side branch, resolved on main with `resolution`."""
    _git(repo, "switch", "-q", "-c", "side")
    (repo / "a.txt").write_text("side\n")
    _git(repo, "commit", "-qam", "side")
    _git(repo, "switch", "-q", "main")
    (repo / "a.txt").write_text("main\n")
    _git(repo, "commit", "-qam", "main2")
    subprocess.run(["git", "-C", str(repo), "merge", "-q", "side"], capture_output=True)  # conflicts
    (repo / "a.txt").write_text(resolution)
    _git(repo, "add", "a.txt")
    _git(repo, "-c", "core.editor=true", "commit", "-qm", "merge side")


def test_range_mode_scans_a_merge_commits_conflict_resolution(repo: Path) -> None:
    # M219: the key shape exists only in the merge commit's resolution
    _merge_resolving_with(repo, f"resolved\n{KEY_SHAPE}\n")
    assert "AKIA" in _git(repo, "show", "HEAD:a.txt")
    assert _guard(repo, "--range", "HEAD~1..HEAD") == 1
    # control: a resolution with nothing blocked passes the same range
    _git(repo, "reset", "-q", "--hard", "HEAD~2")
    _git(repo, "branch", "-q", "-D", "side")
    _merge_resolving_with(repo, "resolved\n")
    assert _guard(repo, "--range", "HEAD~1..HEAD") == 0


def test_range_mode_still_scans_an_ordinary_commits_added_lines(repo: Path) -> None:
    # the positive control for the range scan itself: a plain commit adding a home path is blocked
    (repo / "a.txt").write_text("clean\n" + "/" + "Users/someone/work\n")
    _git(repo, "commit", "-qam", "path")
    assert _guard(repo, "--range", "HEAD~1..HEAD") == 1
