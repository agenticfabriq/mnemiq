"""`mnemiq sweep` from the command line: a config in, a proposed profile and its evidence out, through the real command
on the product path; and `docs/sweeps.md`, checked against the code it describes."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from mnemiq.cli import main
from mnemiq.sweep import propose as proposal
from mnemiq.sweep.config import SweepConfig, load_config
from mnemiq.sweep.stats import detectable_gain

from .conftest import KEY_ENV, evidence, json_lines, out, proposal_file, tiny_deployment, write_config

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
