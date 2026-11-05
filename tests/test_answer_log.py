"""Every answer says which version of which profile produced it, and a default deployment keeps that in a log.

With no `control_dsn` the metrics sink is a no-op, and with no `verity_traces_url` there is no trace, so without this file a
default deployment records nothing about what answered. It holds no question text, SQL or rows unless asked to.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemiq.agent.loop import AgentAnswer, ResultPreview
from mnemiq.config import Settings
from mnemiq.mcp.server import _db_read
from mnemiq.observability.answer_log import AnswerLog, build_answer_log, engine_commit
from mnemiq.observability.trace_sink import VerityTraceSink
from mnemiq.runtime import build_runtime
from mnemiq.server.serialize import answer_payload
from tests.test_runtime_profiles import _who, base, calls, retrieved  # noqa: F401  (fixtures)

PLANTED_QUESTION = "how many planted-question-text?"
PLANTED_SQL = "SELECT planted_sql_text FROM wide"
PLANTED_ROW = "planted-row-value"


class _Answered:
    """An agent that answered, so the log's answered path runs without a source."""

    def decline_if_too_long(self, question):
        return None

    def answer(self, packet, snapshot, grants, identity, emit=None):
        return AgentAnswer(
            answer="A", trace=SimpleNamespace(asked_sql=PLANTED_SQL, plan_sql=PLANTED_SQL, enrichment_version="v1"),
            preview=ResultPreview(columns=["c"], rows=[[PLANTED_ROW]], row_count=1, truncated=False), cached=True)


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _answers(path) -> list[dict]:
    return [line for line in _lines(path) if line["kind"] == "answer"]


def _logged(base, tmp_path, **fields):  # noqa: F811
    path = tmp_path / "answers.jsonl"
    return build_runtime(base.model_copy(update={"answer_log_path": str(path), **fields})), path


def test_a_declined_answer_is_stamped(base, calls, retrieved, tmp_path):  # noqa: F811
    rt, _ = _logged(base, tmp_path)
    ans = rt.ask("q", _who("analyst"), mode="instant", profile="gpt")
    assert ans.deferred is True
    assert ans.profile == "gpt"
    assert ans.profile_version == rt.profiles["gpt"].version
    assert ans.model == "gpt-model"
    assert ans.engine_commit == rt.engine_commit


def test_an_answered_question_is_stamped(base, calls, retrieved, tmp_path):  # noqa: F811
    rt, _ = _logged(base, tmp_path)
    rt.profiles["qwen"].agents["instant"] = _Answered()
    ans = rt.ask("q", _who("analyst"), mode="instant")
    assert (ans.profile, ans.model) == ("qwen", "qwen-model")
    assert ans.profile_version == rt.profiles["qwen"].version


def test_a_question_too_long_is_stamped_and_logged(base, calls, retrieved, tmp_path):  # noqa: F811
    rt, path = _logged(base, tmp_path, max_question_chars=3)
    ans = rt.ask("far too long", _who("analyst"), mode="instant", profile="gpt")
    assert ans.profile == "gpt"
    assert _answers(path)[-1]["profile"] == "gpt"


def test_the_wire_carries_the_stamp():
    ans = AgentAnswer(answer="A", mode="instant", profile="p", profile_version="v", model="m", engine_commit="c")
    for payload in (answer_payload(ans), _db_read(SimpleNamespace(ask=lambda *a, **k: ans), _who(), "q")):
        assert (payload["profile"], payload["profile_version"], payload["model"], payload["engine_commit"]) == (
            "p", "v", "m", "c")


def test_the_log_says_what_produced_each_answer(base, calls, retrieved, tmp_path):  # noqa: F811
    rt, path = _logged(base, tmp_path)
    rt.profiles["qwen"].agents["instant"] = _Answered()
    rt.ask(PLANTED_QUESTION, _who("analyst"), mode="instant")
    line = _answers(path)[-1]
    assert line["profile"] == "qwen"
    assert line["profile_version"] == rt.profiles["qwen"].version
    assert line["model"] == "qwen-model"
    assert line["engine_commit"] == rt.engine_commit
    assert line["mode"] == "instant"
    assert line["enrichment_version"] == "v1"
    assert line["grant_fingerprint"]
    assert line["cached"] is True
    assert line["deferred"] is False and line["failed"] is False
    assert line["total_ms"] >= 0
    assert line["question_sha256"] == hashlib.sha256(PLANTED_QUESTION.encode()).hexdigest()
    assert line["at"]


def test_one_line_per_answer(base, calls, retrieved, tmp_path):  # noqa: F811
    rt, path = _logged(base, tmp_path)
    for profile in ("gpt", "qwen", "gpt"):
        rt.ask("q", _who("analyst"), mode="instant", profile=profile)
    assert [line["profile"] for line in _answers(path)] == ["gpt", "qwen", "gpt"]


def test_the_log_holds_no_text_by_default(base, calls, retrieved, tmp_path):  # noqa: F811
    rt, path = _logged(base, tmp_path)
    rt.profiles["qwen"].agents["instant"] = _Answered()
    rt.ask(PLANTED_QUESTION, _who("analyst"), mode="instant")
    written = path.read_text()
    for planted in (PLANTED_QUESTION, PLANTED_SQL, PLANTED_ROW):
        assert planted not in written


def test_text_is_an_opt_in_and_rows_never_are(base, calls, retrieved, tmp_path):  # noqa: F811
    rt, path = _logged(base, tmp_path, answer_log_text=True)
    rt.profiles["qwen"].agents["instant"] = _Answered()
    rt.ask(PLANTED_QUESTION, _who("analyst"), mode="instant")
    line = _answers(path)[-1]
    assert line["question"] == PLANTED_QUESTION
    assert line["sql"] == PLANTED_SQL
    assert PLANTED_ROW not in path.read_text()


def test_boot_records_each_profile_so_a_version_can_be_read_back(base, tmp_path):  # noqa: F811
    rt, path = _logged(base, tmp_path)
    profiles = {line["name"]: line for line in _lines(path) if line["kind"] == "profile"}
    assert sorted(profiles) == ["gpt", "open", "qwen"]
    assert profiles["gpt"]["version"] == rt.profiles["gpt"].version
    assert profiles["gpt"]["view"]["settings"]["llm_model"] == "gpt-model"
    assert profiles["gpt"]["engine_commit"] == rt.engine_commit
    written = path.read_text()
    assert "gpt-key" not in written and "local-key" not in written


def test_the_default_path_is_beside_the_store():
    log = build_answer_log(Settings(acme_data_dir=None, store_path="/some/where/store.duckdb"))
    assert log.path == "/some/where/store.duckdb.answers.jsonl"


def test_off_turns_it_off():
    assert build_answer_log(Settings(acme_data_dir=None, store_path="/s.duckdb", answer_log_path="off")) is None


def test_a_store_in_memory_has_no_log_and_says_so_once(caplog):
    with caplog.at_level(logging.INFO, logger="mnemiq.observability.answer_log"):
        assert build_answer_log(Settings(acme_data_dir=None, store_path=":memory:")) is None
    assert caplog.text.count("answer log") == 1


def test_an_unwritable_log_never_stops_an_answer_and_warns_once(base, calls, retrieved, tmp_path, caplog):  # noqa: F811
    rt = build_runtime(base.model_copy(update={"answer_log_path": str(tmp_path / "no-such-dir" / "answers.jsonl")}))
    with caplog.at_level(logging.WARNING, logger="mnemiq.observability.answer_log"):
        first = rt.ask("q", _who("analyst"), mode="instant")
        second = rt.ask("q", _who("analyst"), mode="instant")
    assert first.deferred and second.deferred
    assert rt.answer_log is not None
    assert caplog.text.count("cannot be written") <= 1


def test_a_failing_log_is_not_an_exception(tmp_path, caplog):
    log = AnswerLog(str(tmp_path / "no-such-dir" / "a.jsonl"))
    with caplog.at_level(logging.WARNING, logger="mnemiq.observability.answer_log"):
        log.record_answer(AgentAnswer(answer="A"), "q", 1.0)
        log.record_answer(AgentAnswer(answer="A"), "q", 1.0)
    assert caplog.text.count("cannot be written") == 1


def test_the_file_is_private_to_its_owner(tmp_path):
    path = tmp_path / "a.jsonl"
    AnswerLog(str(path)).record_answer(AgentAnswer(answer="A"), "q", 1.0)
    assert path.stat().st_mode & 0o077 == 0


def test_the_trace_carries_the_stamp():
    sys.path.insert(0, "tests")
    from test_verity_trace_sink import _event, _Settings

    ans = AgentAnswer(answer="A", mode="instant", profile="p", profile_version="v", model="m", engine_commit="c")
    intent = VerityTraceSink(_Settings())._build_trace(_event(answer=ans))["resolved_intent"]
    assert (intent["profile"], intent["profile_version"], intent["model"], intent["engine_commit"]) == ("p", "v", "m", "c")


# --- the engine's commit ---------------------------------------------------------------------------------------------


def _git(cwd, *args) -> str:
    env = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    return subprocess.run(["git", *env, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _repo_with(tmp_path, *parts) -> tuple[Path, str]:
    root = tmp_path / "repo"
    package = root.joinpath(*parts)
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "x")
    return package, _git(root, "rev-parse", "HEAD")


def test_the_engines_own_checkout_stamps_its_commit(tmp_path):
    package, head = _repo_with(tmp_path, "src", "mnemiq")
    assert engine_commit(package=package) == head


def test_an_edited_source_tree_says_dirty(tmp_path):
    package, head = _repo_with(tmp_path, "src", "mnemiq")
    (package / "__init__.py").write_text("x = 1\n")
    assert engine_commit(package=package) == f"{head}-dirty"


def test_an_untracked_file_outside_the_source_is_not_dirty(tmp_path):
    package, head = _repo_with(tmp_path, "src", "mnemiq")
    (package.parent.parent / "notes.txt").write_text("x")
    assert engine_commit(package=package) == head


def test_a_package_inside_another_repo_stamps_nothing(tmp_path):
    """A wheel in a venv under some other project must not record that project's commit as the engine's."""
    package, _ = _repo_with(tmp_path, "env", "lib", "site-packages", "mnemiq")
    assert engine_commit(package=package) is None


def test_no_git_at_all_stamps_nothing(tmp_path):
    package = tmp_path / "src" / "mnemiq"
    package.mkdir(parents=True)
    assert engine_commit(package=package) is None


def test_a_declared_commit_wins(tmp_path):
    package, _ = _repo_with(tmp_path, "src", "mnemiq")
    assert engine_commit("declared-sha", package=package) == "declared-sha"


@pytest.mark.parametrize("field", ["engine_commit", "answer_log_path", "answer_log_text"])
def test_these_stay_per_deployment(field):
    from mnemiq.profiles import DEPLOYMENT_SETTINGS

    assert field in DEPLOYMENT_SETTINGS
