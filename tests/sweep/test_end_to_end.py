"""`mnemiq sweep` from the command line: a config in, a proposed profile and its evidence out, through the real command
on the product path; and `docs/sweeps.md`, checked against the code it describes."""

from __future__ import annotations

import multiprocessing
import re
import tomllib
from dataclasses import replace
from pathlib import Path

from mnemiq.cli import SWEEP_BUDGET_STOP, main
from mnemiq.sweep import propose as proposal
from mnemiq.sweep.config import Control, Search, SweepConfig, fingerprint, load_config
from mnemiq.sweep.stats import detectable_gain

from .conftest import (
    KEY_ENV, STUB_KEY, UNREACHABLE, answers_file, evidence, graded_sweep, json_lines, out, proposal_file, tiny_deployment,
    tiny_files, write_config,
)

REPO = Path(__file__).resolve().parents[2]


# --- helpers ------------------------------------------------------------------------------------------------------


def proposal_text(cfg_path: Path, model: str = "m") -> str:
    """The profiles file `propose` wrote for `model`, as text."""
    return proposal_file(load_config(cfg_path), model).read_text()


def evidence_of(cfg_path: Path, model: str = "m") -> dict:
    return evidence(load_config(cfg_path), model)


def confirmed_versions(cfg_path: Path, model: str = "m", mode: str = "instant") -> set:
    """The profile versions the runtime stamped on the rows of the arm the confirm round measured against the start."""
    (folder,) = (out(load_config(cfg_path)) / model / "confirm" / mode).glob("assembled-*")
    return {row["profile_version"] for run in sorted(folder.glob("run-*")) for row in json_lines(run / "rows.jsonl")}


def _doc(path: str) -> str:
    return (REPO / path).read_text()


def _section(text: str, heading: str) -> str:
    """The text under a `## ` heading, up to the next one."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, f"the doc has no '## {heading}' section"
    return match.group(1)


def doc_gain_table(path: str) -> dict[int, int | None]:
    """The doc's table of the smallest gain a search can show: arms compared with the start, to the fewest more right
    answers, with none lost, that pass. A cell that is not a number reads as None, as `detectable_gain` says it."""
    found: dict[int, int | None] = {}
    for line in _section(_doc(path), "What a sweep can show").splitlines():
        match = re.fullmatch(r"\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|.*", line)
        if match:
            found[int(match.group(1))] = int(match.group(2)) if match.group(2).isdigit() else None
    return found


def _named(text: str) -> set[str]:
    """The names a piece of the doc writes in code, a `[[table]]`'s brackets left off."""
    return {name.strip("[]") for name in re.findall(r"`([^`]+)`", text)}


def doc_resumable(path: str) -> tuple[set[str], set[str]]:
    """What the doc says a sweep can be resumed after changing: the `[search]` keys it says change no answer, and
    everything the resume bullet says a resumed config may change."""
    text = _doc(path)
    (search,) = re.findall(r"\n((?:`[a-z_]+`(?:, | and )?)+) change no answer", _section(text, "The config file"))
    (resumed,) = re.findall(r"A resumed config may change only (.*?)\.", _section(text, "Running, resuming and smoke"),
                            re.S)
    return _named(search), _named(resumed)


def doc_sections(path: str) -> set[str]:
    """The config sections the doc gives a heading of their own: `[name]`, `[[name]]` or a bare `name`, each in code."""
    headings = re.findall(r"^### `\[{0,2}([a-z_]+)\]{0,2}`", _section(_doc(path), "The config file"), re.M)
    return set(headings)


# --- the whole path ----------------------------------------------------------------------------------------------


def test_a_sweep_from_config_to_proposal(tmp_path, stub_model, monkeypatch, capsys):
    # The stub answers right only when the writer's request carries `response_format`, which the engine's generator
    # sends when guided_sql is true (`_guided_extra_body`); every other answer is wrong.
    model = stub_model(gold_when=lambda req: "response_format" in req)
    # The wait for a quiet machine before the timing round is tested on its own, and here would only add minutes on a
    # busy machine. Nothing else is changed: the timing round's own checks still run in each timed worker.
    monkeypatch.setattr(proposal, "LOAD_WAIT_S", 0.0)
    cfg_path = write_config(tmp_path, deployment=tiny_deployment(tmp_path, model, cases=40), model=model,
                            space={"guided_sql": [False, True], "retrieval_k": [24, 36]})
    assert main(["sweep", "plan", "--config", str(cfg_path)]) == 0
    assert main(["sweep", "run", "--config", str(cfg_path)]) == 0
    assert main(["sweep", "report", "--config", str(cfg_path)]) == 0
    assert main(["sweep", "propose", "--config", str(cfg_path)]) == 0
    assert "guided_sql = true" in proposal_text(cfg_path)
    # The version proposed is the one the runtime stamped on the rows of the arm the confirm round measured.
    version = evidence_of(cfg_path)["version"]
    assert version is not None and confirmed_versions(cfg_path) == {version}
    assert evidence_of(cfg_path)["chosen_by"] == "rule" and evidence_of(cfg_path)["confirmed"] is True
    printed = capsys.readouterr().out
    # By default every round that ran an arm of its own: the screen kept one value, so the grid had nothing to combine.
    assert "## m, instant: screen" in printed and "## m, instant: confirm" in printed
    assert "## m, instant: grid" not in printed
    # A person may choose another arm that ran, and it applies to every model.
    assert main(["sweep", "propose", "--config", str(cfg_path), "--choose", "start"]) == 0
    assert "guided_sql = false" in proposal_text(cfg_path) and evidence_of(cfg_path)["chosen_by"] == "person"


def test_the_doc_table_matches_detectable_gain():
    assert doc_gain_table("docs/sweeps.md") == {m: detectable_gain(500, m, 0.05) for m in (1, 5, 10, 20)}


def test_the_doc_names_every_config_section():
    assert doc_sections("docs/sweeps.md") >= set(SweepConfig.model_fields)


def test_the_doc_names_exactly_what_a_resumed_sweep_may_change(tmp_path, stub_model):
    # What a resumed sweep may change is what its fingerprint leaves out: each [search] key and a control are changed in
    # turn, and those that leave the fingerprint as it was are the ones the doc must name.
    cfg = load_config(write_config(tmp_path, model=stub_model))
    tried = {"strategy": "grid", "screen_alpha": 0.1, "budget_hours": 3.0, "budget_dollars": 5.0,
             "seconds_per_question": 9.0}
    assert set(tried) == set(Search.model_fields)
    unchanged = {name for name, value in tried.items()
                 if fingerprint(cfg.model_copy(update={"search": cfg.search.model_copy(update={name: value})}))
                 == fingerprint(cfg)}
    control = Control.model_validate({"name": "other", "answers": str(answers_file(tmp_path, gold_for=1))})
    assert fingerprint(cfg.model_copy(update={"control": [control]})) == fingerprint(cfg)
    search, resumed = doc_resumable("docs/sweeps.md")
    assert search == unchanged and resumed == unchanged | {"control"}


def test_the_docs_example_config_is_one_the_sweep_accepts():
    # Its values are checked against the engine's own fields; the files it names are not opened.
    (example,) = re.findall(r"```toml\n(.*?)```", _section(_doc("docs/sweeps.md"), "The config file"), re.S)
    cfg = SweepConfig.model_validate(tomllib.loads(example), context={"base": str(REPO)})
    assert cfg.space and cfg.models and cfg.modes


# --- what the command says, and how it exits ----------------------------------------------------------------------


def test_a_refusal_prints_its_reason_and_exits_non_zero(tmp_path, capsys):
    # A search confirms its pick on held-out questions, so a config that holds none out is refused before anything runs.
    cfg_path = write_config(tmp_path, dataset={"held_out": 0.0})
    for action in (["plan"], ["run"], ["report"], ["propose"]):
        assert main(["sweep", *action, "--config", str(cfg_path)]) == 1
        assert "held_out is 0" in capsys.readouterr().err


def test_a_round_the_strategy_does_not_run_is_refused_by_the_command(tmp_path, capsys):
    cfg_path = write_config(tmp_path, search={"strategy": "check"})
    assert main(["sweep", "run", "--config", str(cfg_path), "--until", "screen"]) == 1
    assert "with strategy 'check' it runs check" in capsys.readouterr().err


def test_a_dead_endpoint_ends_the_run_non_zero_with_its_message(tmp_path, stub_model, capsys):
    model = stub_model(gold=True, fail_after=5, status=500)
    cfg_path = write_config(tmp_path, model=model, arms_in_screen=1)
    assert main(["sweep", "run", "--config", str(cfg_path), "--until", "screen"]) == 1
    said = capsys.readouterr().err
    assert "An endpoint stopped answering" in said and model.url in said


def test_a_report_before_any_decision_reads_each_arm_at_one_run(tmp_path, capsys):
    cfg_path = write_config(tmp_path)
    assert main(["sweep", "report", "--config", str(cfg_path)]) == 0
    said = capsys.readouterr().out
    assert "one run" in said and "Nothing has run" in said


def test_a_report_says_why_when_the_engine_refused_every_start(tmp_path, stub_model, monkeypatch, capsys):
    cfg_path = write_config(tmp_path, model=stub_model(gold=True), arms_in_screen=1)
    monkeypatch.delenv(KEY_ENV)   # every profile names a key variable that is not set
    assert main(["sweep", "run", "--config", str(cfg_path)]) == 0
    capsys.readouterr()
    assert main(["sweep", "report", "--config", str(cfg_path), "--round", "screen"]) == 0
    said = capsys.readouterr().out
    assert "No start ran" in said and "one run" in said and KEY_ENV in said


def test_plan_prices_the_sweep_from_the_tokens_a_smoke_measured(tmp_path, stub_model, capsys):
    stub = stub_model(gold=True)
    priced = {"m": {"llm_model": "stub-model", "llm_base_url": stub.url, "api_key_env": KEY_ENV,
                    "price_in_per_mtok": 1.0, "price_out_per_mtok": 2.0}}
    cfg_path = write_config(tmp_path, model=stub, models=priced, arms_in_screen=1)
    assert main(["sweep", "plan", "--config", str(cfg_path)]) == 0
    assert "dollars: unknown until the first questions run" in capsys.readouterr().out
    assert main(["sweep", "run", "--config", str(cfg_path), "--smoke", "2"]) == 0
    assert "smoke" in capsys.readouterr().out
    assert main(["sweep", "plan", "--config", str(cfg_path)]) == 0
    assert re.search(r"dollars: \d+\.\d\d with one run per arm, \d+\.\d\d with two runs per arm",
                     capsys.readouterr().out)
    assert not (out(load_config(cfg_path)) / "sweep.json").exists()   # a smoke writes only in its own folder


def test_a_worker_that_dies_ends_the_run_non_zero_with_its_message(tmp_path, stub_model, capsys):
    def kill_the_worker(calls: int) -> None:
        if calls == 4:
            for worker in multiprocessing.active_children():
                worker.kill()

    cfg_path = write_config(tmp_path, model=stub_model(gold=True, on_call=kill_the_worker))
    assert main(["sweep", "run", "--config", str(cfg_path), "--until", "screen"]) == 1
    assert "stopped without finishing" in capsys.readouterr().err


def test_a_budget_stop_exits_3_and_says_a_larger_budget_resumes(tmp_path, stub_model, capsys):
    stub = stub_model(gold=True)
    cfg_path = write_config(tmp_path, model=stub, search={"budget_hours": 0.1, "seconds_per_question": 1000.0})
    assert main(["sweep", "run", "--config", str(cfg_path)]) == SWEEP_BUDGET_STOP == 3
    said = capsys.readouterr().out
    assert "Stopped before" in said and "a larger budget resumes" in said and stub.calls == 0


def test_a_control_whose_source_cannot_be_opened_is_refused_without_its_password(tmp_path, capfd):
    # Nothing listens on port 1, and the source's DSN carries a password, which the attach error quotes.
    d = tiny_files(tmp_path)
    env = d.root / "postgres.env"
    values = {"source_id": "library", "pg_dsn": "postgresql://sweep:SECRETPW@127.0.0.1:1/library",
              "store_path": str(d.store), "authz_path": str(d.authz), "embed_base_url": UNREACHABLE,
              "embed_api_key": STUB_KEY}
    env.write_text("".join(f"MNEMIQ_{name.upper()}='{value}'\n" for name, value in values.items()))
    cfg_path = write_config(tmp_path, deployment=replace(d, env=env),
                            control=[{"name": "current", "answers": str(answers_file(tmp_path, gold_for=2))}])
    assert main(["sweep", "report", "--config", str(cfg_path)]) == 1
    said = capfd.readouterr()
    assert "cannot be opened to grade the control 'current'" in said.err
    assert "SECRETPW" not in said.out + said.err and "Traceback" not in said.err


def test_an_arm_file_edited_by_hand_is_refused_by_report_and_withdraws_the_proposal(tmp_path, capsys):
    cfg = graded_sweep(tmp_path, arms={"writer_rules=none": dict(only=9, other=0)})
    assert main(["sweep", "propose", "--config", str(cfg.path)]) == 0
    (arm,) = (out(cfg) / "m" / "screen" / "instant").glob("arm-*")
    profile = arm / "run-1" / "profile.toml"
    profile.write_text(profile.read_text().replace('writer_rules = "none"', 'writer_rules = "full"'))
    capsys.readouterr()
    assert main(["sweep", "report", "--config", str(cfg.path), "--round", "screen"]) == 1
    assert "cannot be read back as that arm" in capsys.readouterr().err
    assert main(["sweep", "propose", "--config", str(cfg.path)]) == 1
    assert "cannot be read back as that arm" in capsys.readouterr().err and not proposal_file(cfg).exists()
    profile.write_text("default = \n[profiles.\n")   # one that cannot be read at all
    assert main(["sweep", "report", "--config", str(cfg.path), "--round", "screen"]) == 1
    assert "cannot be read back as that arm" in capsys.readouterr().err
