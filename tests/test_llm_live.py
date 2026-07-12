import pytest

from mnemiq.config import Settings
from mnemiq.llm.client import LLMClient


@pytest.mark.live_llm
def test_live_completion():
    s = Settings.from_env()
    if not (s.llm_base_url and s.llm_api_key):
        pytest.skip("no live LLM configured")
    out = LLMClient(s).complete("You reply with one word.", "Say ok", max_tokens=256)
    assert "ok" in out.lower()
