"""What the tests of starters and profile parents share: a starters folder of a test's own, the starter fixtures, and
the guard that keeps every test away from the folder the engine ships."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mnemiq

# The name carries a dot, as a real build name does, so every test reads a starter through a quoted TOML key.
STARTER_HEAD = 'default = "fixture-2.5-build"\n[profiles."fixture-2.5-build"]\nretrieval_k = 36\nwriter_temperature = 0.0\n'
STARTER_INSTANT = '[profiles."fixture-2.5-build".instant]\ncard_style = "bare"\n'
STARTER = STARTER_HEAD + STARTER_INSTANT

# The default `root` starts from the starter and sets its own temperature; `child` extends it with its own retrieval_k.
ROOT_AND_CHILD = (
    'default = "root"\n'
    '[profiles.root]\nstarter = "fixture-2.5-build"\nwriter_temperature = 0.2\n'
    '[profiles.child]\nroles = ["*"]\nextends = "root"\nretrieval_k = 12\n'
)


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def edit(path: Path, old: str, new: str) -> None:
    """Replace `old`, which must occur exactly once: an edit that misses its anchor would leave the test running on the
    file it meant to change."""
    text = path.read_text()
    assert text.count(old) == 1, f"{old!r} occurs {text.count(old)} times in {path.name}, not once"
    path.write_text(text.replace(old, new))


def starter_folder(tmp_path: Path, monkeypatch, starters: dict[str, str]) -> Path:
    """A starters folder of the test's own, holding `<name>.toml` for each starter, which every reader in the engine
    then lists in place of the shipped one. A subfolder, so a test's own `profiles.toml` is never listed as a starter."""
    folder = tmp_path / "starters"
    folder.mkdir(exist_ok=True)
    for name, text in starters.items():
        write(folder / f"{name}.toml", text)
    monkeypatch.setattr("mnemiq.profiles.starters_folder", lambda: folder)
    return folder


def _shipped_folder() -> Path:
    # Found from the package and not through `profiles.starters_folder`, which a test may have patched.
    return Path(mnemiq.__file__).resolve().parent / "starter_profiles"


def _recorded(folder: Path) -> dict[str, tuple[str, bytes]]:
    """Each file in the folder, by name, with its sha256 and its bytes to restore it from. Files only: importing the
    package writes its `__pycache__` there."""
    return {path.name: (hashlib.sha256(data := path.read_bytes()).hexdigest(), data)
            for path in folder.iterdir() if path.is_file()}


@contextmanager
def shipped_starters_untouched() -> Iterator[None]:
    """Fails whatever writes to the starters folder the engine ships, naming each change, and puts the folder back as
    it was: a file left there would ship in the next wheel, and every later test would list it as a starter."""
    folder = _shipped_folder()
    before = _recorded(folder)
    try:
        yield
    finally:
        after = _recorded(folder)
        changes = []
        for name in sorted(after.keys() - before.keys()):
            (folder / name).unlink()
            changes.append(f"{name} was added")
        for name in sorted(before.keys() - after.keys()):
            (folder / name).write_bytes(before[name][1])
            changes.append(f"{name} was removed")
        for name in sorted(before.keys() & after.keys()):
            if after[name][0] != before[name][0]:
                (folder / name).write_bytes(before[name][1])
                changes.append(f"{name} was changed")
        if changes:
            raise AssertionError(f"a test wrote to the shipped starters folder {folder.name}/, now restored: "
                                 + "; ".join(changes))
