"""`CITATION.cff` is parsed by GitHub to build the "Cite this repository" button.

An invalid file does not fail anything -- the widget simply does not render, on the public
repo page, silently. It shipped invalid once: CFF 1.2.0 requires `authors`, `title` and
`type` on every entry under `references`, and all three entries had no `authors`.

NOT a YAML parser, and not a schema validator. This repo declares neither, and adding a
dependency to validate one file is the wrong trade. It checks the one shape that broke,
by indentation, and says so rather than implying more.
"""

import pathlib
import re

CFF = pathlib.Path(__file__).resolve().parents[1] / "CITATION.cff"


def _reference_blocks() -> list[str]:
    """Each `  - type: ...` entry under `references:`, to the start of the next one."""
    body = CFF.read_text().split("\nreferences:\n", 1)
    assert len(body) == 2, "no `references:` block -- has the file been restructured?"
    return [b for b in re.split(r"\n(?=  - )", "\n" + body[1]) if b.strip()]


def test_the_file_exists_and_declares_its_cff_version():
    text = CFF.read_text()
    assert text.startswith("cff-version: 1.2.0"), (
        "the checks below are written against CFF 1.2.0's required keys")


def test_every_reference_carries_the_keys_cff_requires():
    """`authors`, `title`, `type`. The first is the one that was missing, and its absence
    is invisible until someone notices the button is gone."""
    blocks = _reference_blocks()
    assert blocks, "no references found; the parser or the file has changed shape"
    for block in blocks:
        kind = re.search(r"type:\s*(\S+)", block)
        # BOTH shapes: a key indented under the entry, and the one sharing the `- ` line.
        # Matching only the first reported a valid `  - authors:` reordering as missing
        # `authors` -- failing closed, and pointing at the wrong thing.
        keys = (set(re.findall(r"^\s{4}(\w[\w-]*):", block, re.M))
                | set(re.findall(r"^\s{2}- (\w[\w-]*):", block, re.M)))
        missing = [k for k in ("authors", "title", "type") if k not in keys]
        assert not missing, f"reference {kind.group(1) if kind else '?'} is missing {missing}"


def test_no_reference_author_list_is_empty():
    """An `authors:` key with nothing under it satisfies a key check and fails the schema.

    Indentation-RELATIVE, not a fixed depth. Hardcoding the nesting made a differently but
    validly indented file red with a message about an empty list it does not have, which is
    the same wrong-thing failure the key scan above had.
    """
    for block in _reference_blocks():
        lines = block.splitlines()
        for i, line in enumerate(lines):
            if not line.strip().lstrip("- ").startswith("authors:"):
                continue
            depth = len(line) - len(line.lstrip())
            rest = [ln for ln in lines[i + 1:] if ln.strip()]
            assert rest, "an `authors:` key with nothing after it at all"
            nxt = rest[0]
            assert (len(nxt) - len(nxt.lstrip())) > depth and nxt.strip().startswith("- "), (
                f"`authors:` has no list entries under it: next line was {nxt!r}")
