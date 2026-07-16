import pytest

from mnemiq.agent.route import StaticRouter, UnknownMode


def test_override_wins():
    assert StaticRouter().route("any question", "deep") == "deep"


def test_none_falls_to_the_default():
    assert StaticRouter().route("any question", None) == "thinking"


def test_custom_default_respected():
    assert StaticRouter(default="instant").route("any question", None) == "instant"


def test_unknown_override_fails_closed_listing_the_valid_modes():
    with pytest.raises(UnknownMode) as exc:
        StaticRouter().route("any question", "fastest")
    msg = str(exc.value)
    assert "fastest" in msg and "thinking" in msg and "deep" in msg and "instant" in msg
