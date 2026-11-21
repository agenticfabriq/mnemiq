"""The extension README says what each of the forwarder's refusals answers: every sentence `forward.py` can answer a
panel with appears there, word for word. The list is read from the module's own constants, so a sentence added or
reworded there and not in the README fails here."""

from __future__ import annotations


from tests.superset_extension_path import EXTENSION_ROOT, use_extension_source

README = EXTENSION_ROOT / "README.md"


def _forward():
    use_extension_source()
    from agenticfabriq.mnemiq import forward

    return forward


def _sentences(module) -> dict[str, str]:
    """Each public upper-case text constant of `module`: in `forward.py` these are exactly what it answers a panel
    with (its patterns and limits are private, a number or a mapping)."""
    return {name: value for name, value in vars(module).items()
            if name.isupper() and not name.startswith("_") and isinstance(value, str)}


def _missing(sentences: dict[str, str], readme: str) -> dict[str, str]:
    """The sentences `readme` does not hold, word for word."""
    return {name: text for name, text in sentences.items() if text not in readme}


def test_every_forwarder_sentence_is_in_the_readme():
    found = _sentences(_forward())
    assert {"NOT_FORWARDED", "BODY_TOO_LARGE", "QUERY_REFUSED", "MNEMIQ_UNREACHABLE", "MNEMIQ_TIMED_OUT",
            "FORWARD_NOT_CONFIGURED", "SIGNING_NOT_CONFIGURED", "NOT_SIGNED_IN"} <= set(found), sorted(found)
    assert _missing(found, README.read_text()) == {}


def test_the_check_sees_a_sentence_the_readme_lacks():
    """The control: a sentence the README does not hold is caught by the same check."""
    planted = _sentences(type("M", (), {"PLANTED": "This sentence is in no README.", "_PRIVATE": "not answered"}))
    assert planted == {"PLANTED": "This sentence is in no README."}
    assert _missing(planted, README.read_text()) == planted
