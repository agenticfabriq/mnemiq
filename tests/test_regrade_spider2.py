"""scripts/regrade_spider2.py never reports a record it could not re-check as confirmed.

Its no-change summary says either that every graded case was re-checked or how many were not;
a got-facts the grader could not decide (register M113) is one more way a case goes unchecked,
and it has to land in that count -- missing it printed "the labels they have ARE the current
rule's" over a record nobody graded.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("regrade_spider2", ROOT / "scripts" / "regrade_spider2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A Spider2 checkout with one case: two rows of twelve flag columns against a three-column
    gold that no set of them states -- every column survives pruning, so the grader must try them."""
    spider2 = tmp_path / "spider2"
    (spider2 / "databases").mkdir(parents=True)
    con = sqlite3.connect(spider2 / "databases" / "alpha.sqlite")
    cols = ", ".join(f"x{i} INTEGER" for i in range(12))
    con.execute(f"CREATE TABLE t ({cols})")
    con.execute(f"INSERT INTO t VALUES ({', '.join(str(i % 2) for i in range(12))})")
    con.execute(f"INSERT INTO t VALUES ({', '.join(str(1 - i % 2) for i in range(12))})")
    con.commit()
    con.close()
    gold_dir = spider2 / "repo" / "spider2-lite" / "evaluation_suite" / "gold" / "exec_result"
    gold_dir.mkdir(parents=True)
    (gold_dir / "local001.csv").write_text("a,b,c\n0,1,0\n1,0,0\n")
    run = tmp_path / "run.jsonl"
    run.write_text(json.dumps({"case_id": "local001", "db_id": "alpha", "sql": "SELECT * FROM t",
                               "outcome": "wrong"}) + "\n")
    return spider2, run


def _regrade(monkeypatch, capsys, spider2: Path, run: Path) -> str:
    monkeypatch.setattr(sys, "argv", ["regrade_spider2.py", "--spider2-dir", str(spider2), str(run)])
    assert _script().main() == 0
    return capsys.readouterr().out


def test_an_undecided_record_is_counted_as_not_rechecked(monkeypatch, capsys, tmp_path):
    import mnemiq.eval.grade as grade

    spider2, run = _checkout(tmp_path)
    monkeypatch.setattr(grade, "MAX_CHOICES", 5)
    out = _regrade(monkeypatch, capsys, spider2, run)
    assert "1 undecided" in out
    assert "were NOT re-checked" in out and "1 the grader could not decide" in out
    assert "ARE the current rule's" not in out


def test_control_a_decided_record_is_confirmed(monkeypatch, capsys, tmp_path):
    """At the real limit the same record is decided (wrong, as labelled) and confirmed."""
    spider2, run = _checkout(tmp_path)
    out = _regrade(monkeypatch, capsys, spider2, run)
    assert "0 undecided" in out and "ARE the current rule's" in out
