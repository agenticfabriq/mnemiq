"""A results file has to say which mnemiq produced it.

Without this, a consumer importing the file can record the script path and nothing that
pins the behaviour -- and the grading rules moved twice in a single day.
"""

import json
import subprocess

from mnemiq.eval.bird_runner import _save_meta, source_rev


def test_the_meta_file_carries_the_revision(tmp_path):
    results = str(tmp_path / "run.jsonl")

    _save_meta(results, tokens=10, calls=2, excluded=[])

    meta = json.loads((tmp_path / "run.jsonl.meta.json").read_text())
    assert meta["source_rev"], "a run with no provenance is a run nobody can reproduce"
    assert meta["tokens"] == 10 and meta["llm_calls"] == 2


def _repo(tmp_path, monkeypatch):
    """A real checkout, so these assert behaviour rather than mirror the implementation.

    The test this replaces computed `dirty` with the same command the code used and asserted
    they agreed -- true by construction, and it would have passed for as long as the two
    states stayed conflated. It did.
    """
    import mnemiq.eval.bird_runner as runner

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@example.invalid"],
                 ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    (repo / "tracked.txt").write_text("one\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "first"], check=True,
                   capture_output=True)
    # Narrower than patching os.path: source_rev derives its checkout from the module file.
    monkeypatch.setattr(runner, "__file__", str(repo / "bird_runner.py"))
    return repo


def test_a_modified_tracked_file_makes_the_run_dirty(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    assert not source_rev().endswith("-dirty")
    (repo / "tracked.txt").write_text("two\n")
    assert source_rev().endswith("-dirty"), "a changed tracked file is unreproducible code"


def test_an_untracked_file_alone_does_not(tmp_path, monkeypatch):
    """Three orphan scratch files stamped every artifact in a checkout as unreproducible,
    including a Spider2 run whose code was exactly its commit. A flag that fires when nothing
    is wrong is a flag nobody reads when something is."""
    repo = _repo(tmp_path, monkeypatch)
    (repo / "scratch_authz.json").write_text("{}")
    assert not source_rev().endswith("-dirty")


def test_but_the_untracked_files_are_still_counted(tmp_path, monkeypatch):
    """Not discarded: an untracked file can be something the run READ."""
    from mnemiq.eval.bird_runner import untracked_count

    repo = _repo(tmp_path, monkeypatch)
    assert untracked_count() == 0
    (repo / "scratch_authz.json").write_text("{}")
    assert untracked_count() == 1


def test_the_meta_records_both_states_separately(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    (repo / "scratch.json").write_text("{}")
    results = str(tmp_path / "run.jsonl")  # outside the repo: a meta file must not count itself
    _save_meta(results, tokens=1, calls=1, excluded=[])
    meta = json.loads((tmp_path / "run.jsonl.meta.json").read_text())
    assert not meta["source_rev"].endswith("-dirty")
    assert meta["untracked_files"] == 1


def test_provenance_never_fails_a_run(monkeypatch, tmp_path):
    # An eval run costs real money; it must not die for want of a version string.
    def explode(*a, **k):
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", explode)
    results = str(tmp_path / "run.jsonl")

    _save_meta(results, tokens=0, calls=0, excluded=[])

    assert "source_rev" in json.loads((tmp_path / "run.jsonl.meta.json").read_text())
