"""The repo guard checks who a commit says made it, not only what it holds.

A clone outside the maintainer's checkout has no repository-local git identity and commits as the machine's global one.
Content checks pass such a commit, and a public repository's history cannot be taken back. So the guard also reads each
commit's author and committer against a private identity list (`REPO_GUARD_IDENTITY_PATTERNS`, from the environment or
`.env`, like the name list): before a commit (`--staged`, the identity the commit will carry), and over the commits a push
adds (`--range`, as CI runs it). In `--range` it names the commit and the field, never the identity or the pattern, since
a CI log is public.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repo-guard.sh"
BLOCKED = "privatepersona"  # stands for an identity the maintainer keeps off the public repo
BASE_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
            "REPO_GUARD_NAME_PATTERNS": "off", "REPO_GUARD_IDENTITY_PATTERNS": BLOCKED}


def _env(**changes) -> dict:
    env = {**BASE_ENV, **changes}
    return {k: v for k, v in env.items() if v is not None}


def _git(repo: Path, *args: str, env=None) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env or BASE_ENV).stdout


def _guard(repo: Path, *args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "scripts/repo-guard.sh", *args], cwd=repo, env=env or BASE_ENV,
                          capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    (r / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, r / "scripts" / "repo-guard.sh")
    _git(r, "init", "-q", "-b", "main", ".")
    _git(r, "config", "user.email", "dev@example.test")
    _git(r, "config", "user.name", "dev")
    (r / "a.txt").write_text("clean\n")
    _git(r, "add", ".")
    _git(r, "commit", "-qm", "base")
    return r


def _stage(repo: Path) -> None:
    (repo / "a.txt").write_text("clean, changed\n")
    _git(repo, "add", "a.txt")


def test_a_commit_as_the_configured_identity_passes(repo: Path) -> None:
    _stage(repo)
    out = _guard(repo, "--staged")
    assert out.returncode == 0, out.stdout + out.stderr


def test_a_commit_whose_author_is_on_the_list_is_blocked(repo: Path) -> None:
    _git(repo, "config", "user.name", "Private Persona")
    _git(repo, "config", "user.email", f"{BLOCKED}@mail.example")
    _stage(repo)
    out = _guard(repo, "--staged")
    assert out.returncode == 1 and "BLOCKED [identity]" in out.stdout


def test_a_committer_alone_on_the_list_is_blocked(repo: Path) -> None:
    _stage(repo)
    env = _env(GIT_COMMITTER_NAME="Private Persona", GIT_COMMITTER_EMAIL=f"{BLOCKED}@mail.example")
    out = _guard(repo, "--staged", env=env)
    assert out.returncode == 1 and "committer" in out.stdout


def test_off_skips_the_identity_check_and_says_so(repo: Path) -> None:
    _git(repo, "config", "user.email", f"{BLOCKED}@mail.example")
    _stage(repo)
    out = _guard(repo, "--staged", env=_env(REPO_GUARD_IDENTITY_PATTERNS="off"))
    assert out.returncode == 0 and "identity check off" in out.stdout


def test_an_unset_list_blocks_a_commit(repo: Path) -> None:
    # The leak this closes came from a checkout with no identity configured: unset is the case to stop.
    _stage(repo)
    out = _guard(repo, "--staged", env=_env(REPO_GUARD_IDENTITY_PATTERNS=None))
    assert out.returncode == 1 and "REPO_GUARD_IDENTITY_PATTERNS is not set" in out.stdout


def test_the_list_is_read_from_dot_env_when_the_environment_has_none(repo: Path) -> None:
    (repo / ".env").write_text(f"REPO_GUARD_IDENTITY_PATTERNS={BLOCKED}\n")
    _git(repo, "config", "user.email", f"{BLOCKED}@mail.example")
    _stage(repo)
    out = _guard(repo, "--staged", env=_env(REPO_GUARD_IDENTITY_PATTERNS=None))
    assert out.returncode == 1 and "BLOCKED [identity]" in out.stdout


def test_range_mode_names_the_commit_and_field_never_the_identity(repo: Path) -> None:
    (repo / "a.txt").write_text("clean, by the wrong identity\n")
    _git(repo, "add", "a.txt")
    _git(repo, "-c", "user.name=Private Persona", "-c", f"user.email={BLOCKED}@mail.example", "commit", "-qm", "x")
    sha = _git(repo, "rev-parse", "--short=8", "HEAD").strip()
    out = _guard(repo, "--range", "HEAD~1..HEAD")
    assert out.returncode == 1
    assert sha in out.stdout and "author" in out.stdout
    assert BLOCKED not in out.stdout + out.stderr and "Private Persona" not in out.stdout + out.stderr


def test_range_mode_without_a_list_skips_the_identity_check_and_says_so(repo: Path) -> None:
    # CI on a fork's pull request receives no secret; the content checks still decide.
    (repo / "a.txt").write_text("clean again\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "y")
    out = _guard(repo, "--range", "HEAD~1..HEAD", env=_env(REPO_GUARD_IDENTITY_PATTERNS=None))
    assert out.returncode == 0 and "identity check skipped" in out.stdout


def test_range_mode_checks_a_merge_commits_own_identity(repo: Path) -> None:
    # A merge commit is a commit like any other: its author and committer are checked, not only the branch's commits.
    _git(repo, "switch", "-qc", "side")
    (repo / "b.txt").write_text("side\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "side")
    _git(repo, "switch", "-q", "main")
    (repo / "a.txt").write_text("main moved\n")
    _git(repo, "commit", "-qam", "main")
    _git(repo, "-c", "user.name=Private Persona", "-c", f"user.email={BLOCKED}@mail.example",
         "merge", "-q", "--no-ff", "side", "-m", "merge")
    merge = _git(repo, "rev-parse", "--short=8", "HEAD").strip()
    out = _guard(repo, "--range", "HEAD~1..HEAD")
    assert out.returncode == 1 and merge in out.stdout, out.stdout


def test_githubs_own_merge_identity_passes(repo: Path) -> None:
    (repo / "a.txt").write_text("merged on github\n")
    _git(repo, "add", "a.txt")
    env = _env(GIT_COMMITTER_NAME="GitHub", GIT_COMMITTER_EMAIL="noreply@github.com")
    _git(repo, "commit", "-qm", "Merge pull request", env=env)
    out = _guard(repo, "--range", "HEAD~1..HEAD")
    assert out.returncode == 0, out.stdout


# -- Codex's review of #290 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["--staged", "--range"])
def test_an_invalid_pattern_fails_closed_without_printing_it(repo: Path, mode: str) -> None:
    # grep exits 2 on a bad pattern; read as "no match", a malformed list would turn the check off and report clean.
    (repo / "a.txt").write_text("clean, changed\n")
    _git(repo, "add", "a.txt")
    if mode == "--range":
        _git(repo, "commit", "-qm", "c")
    args = ["--range", "HEAD~1..HEAD"] if mode == "--range" else ["--staged"]
    out = _guard(repo, *args, env=_env(REPO_GUARD_IDENTITY_PATTERNS="[unclosed"))
    assert out.returncode != 0 and "not a valid" in out.stdout + out.stderr
    assert "[unclosed" not in out.stdout + out.stderr


def test_range_mode_matches_an_identity_with_a_tab_in_it(repo: Path) -> None:
    # Git accepts a tab inside a name; split on tabs, the fields shifted and an anchored pattern matched neither.
    (repo / "a.txt").write_text("by a tabbed name\n")
    _git(repo, "add", "a.txt")
    env = _env(GIT_AUTHOR_NAME="Private\tPersona", GIT_AUTHOR_EMAIL="pp@mail.example")
    _git(repo, "commit", "-qm", "t", env=env)
    out = _guard(repo, "--range", "HEAD~1..HEAD",
                 env=_env(REPO_GUARD_IDENTITY_PATTERNS="^Private\tPersona <pp@mail[.]example>$"))
    assert out.returncode == 1 and "author" in out.stdout, out.stdout + out.stderr


def test_a_traced_run_shows_neither_identity_nor_pattern(repo: Path) -> None:
    (repo / "a.txt").write_text("clean, by the wrong identity\n")
    _git(repo, "add", "a.txt")
    _git(repo, "-c", "user.name=Private Persona", "-c", f"user.email={BLOCKED}@mail.example", "commit", "-qm", "x")
    out = subprocess.run(["bash", "-x", "scripts/repo-guard.sh", "--range", "HEAD~1..HEAD"], cwd=repo, env=BASE_ENV,
                         capture_output=True, text=True)
    assert out.returncode == 1
    assert BLOCKED not in out.stdout + out.stderr and "Private Persona" not in out.stdout + out.stderr


def test_a_failed_identity_read_is_an_error_not_a_clean_pass(repo: Path, tmp_path: Path) -> None:
    # Codex's second review of #290: the range validated, then the identity `git log` failed inside the process
    # substitution, and the loop read no commits and reported clean. Every commit in the range must be read.
    (repo / "a.txt").write_text("clean again\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "y")
    real = shutil.which("git")
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "git").write_text(f'#!/bin/bash\nfor a in "$@"; do [ "$a" = "-z" ] && exit 128; done\nexec "{real}" "$@"\n')
    (shim / "git").chmod(0o755)
    env = _env(PATH=f"{shim}:{os.environ['PATH']}")
    out = _guard(repo, "--range", "HEAD~1..HEAD", env=env)
    assert out.returncode == 2 and "0 of 1" in out.stderr, out.stdout + out.stderr


# -- the e2e lane's review of #290 (mutation-tested) -----------------------------------------------------------------


def test_range_mode_catches_a_committer_alone(repo: Path) -> None:
    # The incident's shape: one of the seven commits had the private identity as committer only.
    (repo / "a.txt").write_text("committed by the wrong identity\n")
    _git(repo, "add", "a.txt")
    env = _env(GIT_COMMITTER_NAME="Private Persona", GIT_COMMITTER_EMAIL=f"{BLOCKED}@mail.example")
    _git(repo, "commit", "-qm", "c", env=env)
    sha = _git(repo, "rev-parse", "--short=8", "HEAD").strip()
    out = _guard(repo, "--range", "HEAD~1..HEAD")
    assert out.returncode == 1 and f"commit {sha}'s committer" in out.stdout
    assert "author is on" not in out.stdout, "the author is clean here"


@pytest.mark.parametrize("mode", ["--staged", "--range"])
def test_the_match_ignores_case(repo: Path, mode: str) -> None:
    # A pattern written lowercase against a capitalised identity: the likeliest silent miss.
    upper = BLOCKED.upper()
    (repo / "a.txt").write_text("by a capitalised identity\n")
    _git(repo, "add", "a.txt")
    env = _env(GIT_AUTHOR_NAME="Private Persona", GIT_AUTHOR_EMAIL=f"{upper}@MAIL.EXAMPLE")
    if mode == "--range":
        _git(repo, "commit", "-qm", "u", env=env)
        out = _guard(repo, "--range", "HEAD~1..HEAD")
    else:
        out = _guard(repo, "--staged", env=env)
    assert out.returncode == 1 and "author" in out.stdout


def test_range_mode_says_how_many_commits_it_checked(repo: Path) -> None:
    (repo / "a.txt").write_text("one\n")
    _git(repo, "commit", "-qam", "one")
    (repo / "a.txt").write_text("two\n")
    _git(repo, "commit", "-qam", "two")
    out = _guard(repo, "--range", "HEAD~2..HEAD")
    assert out.returncode == 0 and "identity: 2 commit(s) checked" in out.stdout


def test_a_list_required_but_missing_fails_instead_of_skipping(repo: Path) -> None:
    # In CI on a same-repo push or pull request the secret is required: missing means deleted or renamed, not a fork.
    (repo / "a.txt").write_text("clean\n...")
    _git(repo, "commit", "-qam", "r")
    out = _guard(repo, "--range", "HEAD~1..HEAD",
                 env=_env(REPO_GUARD_IDENTITY_PATTERNS=None, REPO_GUARD_IDENTITY_REQUIRED="1"))
    assert out.returncode == 1 and "REPO_GUARD_IDENTITY_PATTERNS is not set" in out.stdout


# -- --identities: a fork's pull request, checked from its commit metadata alone ----------------------------------------
# A fork's pull request receives no secret in its own runs. A pull_request_target job (identity-guard.yml) runs this
# repository's code with the secret, reads the pull request's commit identities from GitHub's API, never the fork's
# code, and feeds them here as NUL-terminated triples: sha, author, committer.


def _identities(repo: Path, triples: list[tuple], **env) -> subprocess.CompletedProcess:
    # Each record: sha, author, committer, and the commit's whole message (every line is checked).
    data = "".join(f"{t[0]}\0{t[1]}\0{t[2]}\0{t[3] if len(t) > 3 else ''}\0\n" for t in triples).encode()
    return subprocess.run(["bash", "scripts/repo-guard.sh", "--identities"], cwd=repo, input=data,
                          env=_env(**env), capture_output=True)


CLEAN = ("a" * 40, "Hidden Dim <hd@mail.example>", "GitHub <noreply@github.com>")
LISTED = ("b" * 40, f"Private Persona <{BLOCKED}@mail.example>", "GitHub <noreply@github.com>")


def test_identities_mode_passes_clean_commits_and_counts_them(repo: Path) -> None:
    out = _identities(repo, [CLEAN, CLEAN], REPO_GUARD_IDENTITY_EXPECTED="2")
    assert out.returncode == 0 and b"identity: 2 commit(s) checked" in out.stdout


def test_identities_mode_blocks_a_listed_identity_without_printing_it(repo: Path) -> None:
    out = _identities(repo, [CLEAN, LISTED], REPO_GUARD_IDENTITY_EXPECTED="2")
    assert out.returncode == 1 and b"commit bbbbbbbb's author" in out.stdout
    assert BLOCKED.encode() not in out.stdout + out.stderr and b"Private Persona" not in out.stdout + out.stderr


def test_identities_mode_reads_a_tab_holding_name_whole(repo: Path) -> None:
    tabbed = ("c" * 40, "Private\tPersona <pp@mail.example>", "GitHub <noreply@github.com>")
    out = _identities(repo, [tabbed], REPO_GUARD_IDENTITY_EXPECTED="1",
                      REPO_GUARD_IDENTITY_PATTERNS="^Private\tPersona <pp@mail[.]example>$")
    assert out.returncode == 1 and b"author" in out.stdout


def test_identities_mode_refuses_fewer_identities_than_the_pull_request_has(repo: Path) -> None:
    # The API returns at most 250 commits; a longer pull request cannot be checked whole, so it fails.
    out = _identities(repo, [CLEAN], REPO_GUARD_IDENTITY_EXPECTED="3")
    assert out.returncode == 2 and b"1 of 3" in out.stderr


def test_identities_mode_requires_its_list(repo: Path) -> None:
    out = _identities(repo, [CLEAN], REPO_GUARD_IDENTITY_EXPECTED="1", REPO_GUARD_IDENTITY_PATTERNS=None,
                      REPO_GUARD_IDENTITY_REQUIRED="1")
    assert out.returncode == 1 and b"REPO_GUARD_IDENTITY_PATTERNS is not set" in out.stdout


def test_identities_mode_needs_no_name_list(repo: Path) -> None:
    # It runs in CI, where the name list never is.
    out = _identities(repo, [CLEAN], REPO_GUARD_IDENTITY_EXPECTED="1", REPO_GUARD_NAME_PATTERNS=None)
    assert out.returncode == 0, out.stdout + out.stderr



# -- Co-authored-by: GitHub credits a co-author named in a commit message's trailers ------------------------------------


def test_range_mode_catches_a_listed_co_author(repo: Path) -> None:
    (repo / "a.txt").write_text("with a co-author\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", f"change\n\nCo-authored-by: Private Persona <{BLOCKED}@mail.example>")
    sha = _git(repo, "rev-parse", "--short=8", "HEAD").strip()
    out = _guard(repo, "--range", "HEAD~1..HEAD")
    assert out.returncode == 1 and f"commit {sha}'s message" in out.stdout
    assert BLOCKED not in out.stdout + out.stderr


def test_range_mode_passes_a_clean_co_author(repo: Path) -> None:
    (repo / "a.txt").write_text("with a clean co-author\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "change\n\nCo-authored-by: Someone Else <else@mail.example>")
    assert _guard(repo, "--range", "HEAD~1..HEAD").returncode == 0


def test_identities_mode_catches_a_listed_co_author(repo: Path) -> None:
    co = ("d" * 40, "Hidden Dim <hd@mail.example>", "GitHub <noreply@github.com>",
          f"Someone Else <else@mail.example>\nPrivate Persona <{BLOCKED}@mail.example>")
    out = _identities(repo, [co], REPO_GUARD_IDENTITY_EXPECTED="1")
    assert out.returncode == 1 and b"commit dddddddd's message" in out.stdout
    assert BLOCKED.encode() not in out.stdout + out.stderr



# -- the e2e lane's review of 9d51a1dc ----------------------------------------------------------------------------------


def test_a_listed_identity_anywhere_in_a_message_is_blocked(repo: Path) -> None:
    # git's trailer parser reads only the last paragraph, and GitHub reads co-authors its own way: every line is checked.
    (repo / "a.txt").write_text("named mid-message\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", f"change\n\nCo-authored-by: Private Persona <{BLOCKED}@mail.example>\n\nmore text")
    out = _guard(repo, "--range", "HEAD~1..HEAD")
    assert out.returncode == 1 and "message" in out.stdout


def test_identities_mode_refuses_a_shifted_record(repo: Path) -> None:
    # A NUL inside a name splits it, every later field shifts, and a listed identity lands in the sha slot unread.
    data = (b"a" * 40 + b"\0Pub\0x <p@mail.example>\0GitHub <noreply@github.com>\0\0\n"
            + b"b" * 40 + b"\0Hidden Dim <hd@mail.example>\0Private Persona <" + BLOCKED.encode() + b"@mail.example>\0\0\n")
    out = subprocess.run(["bash", "scripts/repo-guard.sh", "--identities"], cwd=repo, input=data,
                         env=_env(REPO_GUARD_IDENTITY_EXPECTED="2"), capture_output=True)
    assert out.returncode == 2 and b"not a commit SHA" in out.stderr, out.stdout + out.stderr


def test_identities_mode_requires_an_expected_count(repo: Path) -> None:
    out = _identities(repo, [], REPO_GUARD_IDENTITY_EXPECTED=None)
    assert out.returncode == 2 and b"REPO_GUARD_IDENTITY_EXPECTED" in out.stdout + out.stderr


ENCODER = Path(__file__).resolve().parents[1] / "scripts" / "repo-guard-identities.py"


def _encode(lines: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["python3", "-I", str(ENCODER)], input="\n".join(lines).encode(), capture_output=True)


def test_the_encoder_writes_nul_terminated_records() -> None:
    line = json.dumps({"sha": "a" * 40, "a": "Hidden Dim <hd@x>", "c": "GitHub <noreply@github.com>", "m": "one\ntwo"})
    out = _encode([line])
    assert out.returncode == 0 and out.stdout == b"a" * 40 + b"\0Hidden Dim <hd@x>\0GitHub <noreply@github.com>\0one\ntwo\0"


@pytest.mark.parametrize("record", [
    {"sha": "a" * 40, "a": "Pub\u0000x <p@x>", "c": "GitHub <noreply@github.com>", "m": ""},
    {"sha": "a" * 40, "a": "Pub\nx <p@x>", "c": "GitHub <noreply@github.com>", "m": ""},
    {"sha": "a" * 40, "a": "Hidden Dim <hd@x>", "c": "GitHub <noreply@github.com>", "m": "nul\u0000here"},
    {"sha": "not-a-sha", "a": "Hidden Dim <hd@x>", "c": "GitHub <noreply@github.com>", "m": ""},
])
def test_the_encoder_refuses_a_value_that_could_shift_fields(record: dict) -> None:
    out = _encode([json.dumps(record)])
    assert out.returncode == 1 and out.stdout == b""
