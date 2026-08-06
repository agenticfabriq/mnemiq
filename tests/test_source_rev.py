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


def test_a_dirty_tree_says_so():
    # A bare rev that hides uncommitted changes is worse than no rev: it looks precise.
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )

    assert source_rev().endswith("-dirty") == dirty


def test_provenance_never_fails_a_run(monkeypatch, tmp_path):
    # An eval run costs real money; it must not die for want of a version string.
    def explode(*a, **k):
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", explode)
    results = str(tmp_path / "run.jsonl")

    _save_meta(results, tokens=0, calls=0, excluded=[])

    assert "source_rev" in json.loads((tmp_path / "run.jsonl.meta.json").read_text())
