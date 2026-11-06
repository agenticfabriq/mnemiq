"""docs/settings.md is the reference for every setting, so it is checked against the code that defines them: each
setting has exactly one row, in the table for where it may be set, showing the code's default (or, for a per-mode
setting, each mode's own value), and each technique a row names is a section of the doc. A setting added, removed,
reclassified or given a new default fails here until the doc says so."""

import os
import re
from pathlib import Path

import pytest

from mnemiq.agent.modes import MODES
from mnemiq.config import ModeSettings, Settings
from mnemiq.profiles import DEPLOYMENT_SETTINGS, PROFILE_SETTINGS

DOC = Path(__file__).resolve().parents[1] / "docs" / "settings.md"
ROW = re.compile(r"^\|\s*`([a-z_]+)`\s*\|(.*)\|\s*$")
NO_TECHNIQUE = "plain parameter"


def shown(value) -> str:
    """A value as the doc writes it: as a profile file (TOML) spells it, or `unset`."""
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, (int, float)):
        return f"`{value}`"
    return f'`"{value}"`'


def section(text: str, heading: str) -> str:
    """The text under a `## ` heading, up to the next one."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, f"docs/settings.md has no '## {heading}' section"
    return match.group(1)


def rows(text: str) -> dict[str, list[str]]:
    """Each table row whose first cell is a setting, as its name and its other cells; a name twice is refused."""
    found: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = ROW.match(line)
        if match:
            name = match.group(1)
            assert name not in found, f"{name} has two rows"
            found[name] = [cell.strip() for cell in match.group(2).split("|")]
    return found


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text()


@pytest.fixture(scope="module")
def defaults() -> Settings:
    """What a deployment gets with nothing set: the field defaults, and the ones `Settings` fills in itself (the
    chat and embedding models, the source id and the store path)."""
    with pytest.MonkeyPatch.context() as env:
        for name in [n for n in os.environ if n.startswith("MNEMIQ_")]:
            env.delenv(name)
        return Settings()


def techniques(doc: str) -> set[str]:
    return set(re.findall(r"^### (.+)$", section(doc, "Techniques"), re.M))


def test_every_profile_setting_has_one_row_with_its_default_and_technique(doc, defaults):
    table = rows(section(doc, "Profile settings"))
    assert set(table) == PROFILE_SETTINGS
    named = techniques(doc)
    for name, (default, technique, _what) in table.items():
        assert default == shown(getattr(defaults, name)), name
        assert technique == NO_TECHNIQUE or technique in named, (name, technique)


def test_every_per_mode_setting_has_one_row_with_each_modes_own_value(doc):
    table = rows(section(doc, "Per-mode settings"))
    assert set(table) == set(ModeSettings.model_fields)
    assert list(MODES) == ["instant", "thinking", "deep"]
    named = techniques(doc)
    for name, (instant, thinking, deep, technique, _what) in table.items():
        assert [instant, thinking, deep] == [shown(getattr(MODES[m], name)) for m in MODES], name
        assert technique == NO_TECHNIQUE or technique in named, (name, technique)


def test_every_deployment_setting_has_one_row_with_its_default(doc, defaults):
    table = rows(section(doc, "Deployment settings"))
    assert set(table) == DEPLOYMENT_SETTINGS
    for name, (default, _what) in table.items():
        assert default == shown(getattr(defaults, name)), name


def test_no_setting_is_in_two_tables(doc):
    profile = set(rows(section(doc, "Profile settings")))
    deployment = set(rows(section(doc, "Deployment settings")))
    assert not profile & deployment


def test_every_environment_variable_that_is_not_the_name_in_capitals_is_given(doc):
    for name, field in Settings.model_fields.items():
        alias = field.validation_alias
        if isinstance(alias, str) and alias != f"MNEMIQ_{name.upper()}":
            assert f"`{name}` is `{alias}`" in doc, name


def test_every_technique_has_a_setting_that_names_it(doc):
    used = {cells[1] for cells in rows(section(doc, "Profile settings")).values()}
    used |= {cells[3] for cells in rows(section(doc, "Per-mode settings")).values()}
    assert techniques(doc) <= used
