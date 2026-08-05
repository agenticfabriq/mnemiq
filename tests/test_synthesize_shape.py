"""Big results get summarised, small ones get stated -- and the caller decides which.

Asking the model to judge "more than a few rows" for itself did not work: given nine
rows it read all nine back. The row count is known at the call site, so the instruction
is selected there and arrives unconditional.
"""

from mnemiq.agent.synthesize import LIST_LIMIT, LLMSynthesizer


class _Recorder:
    def __init__(self) -> None:
        self.system = ""

    def complete(self, system: str, user: str, max_tokens: int = 512, **kwargs) -> str:
        self.system = system
        return "an answer"


def _system_for(row_count: int | None, forced: bool = False) -> str:
    client = _Recorder()
    LLMSynthesizer(client).answer("q", "SELECT 1", "n\n1\n(1 rows)", forced=forced,
                                  row_count=row_count)
    return client.system


def test_a_large_result_is_told_not_to_list_rows_back():
    system = _system_for(LIST_LIMIT + 1)
    assert "Do NOT list its rows back" in system
    assert "what stands out" in system


def test_a_small_result_is_not():
    system = _system_for(LIST_LIMIT)
    assert "Do NOT list its rows back" not in system


def test_an_unknown_row_count_leaves_the_rules_alone():
    # Any caller that does not know the size gets the base rules, never a half-applied one.
    assert "Do NOT list its rows back" not in _system_for(None)


def test_the_base_rules_always_survive():
    for count in (None, 1, LIST_LIMIT + 50):
        system = _system_for(count)
        assert "never invent a number" in system
        assert "was shortened to fit this prompt" in system


def test_the_budget_notice_still_applies_on_top():
    system = _system_for(LIST_LIMIT + 1, forced=True)
    assert "Do NOT list its rows back" in system
    assert "out of budget" in system


def _system_with_markdown(row_count: int | None = None) -> str:
    client = _Recorder()
    LLMSynthesizer(client, markdown=True).answer("q", "SELECT 1", "n\n1\n(1 rows)",
                                                 row_count=row_count)
    return client.system


def test_markdown_is_off_unless_asked_for():
    # Every other consumer of the answer -- MCP, the eval harness, beacon's SUT -- reads
    # plain text, so this cannot be on by default.
    assert "Use markdown when the answer has structure" not in _system_for(3)


def test_markdown_permission_is_added_when_configured():
    assert "Use markdown when the answer has structure" in _system_with_markdown()


def test_markdown_never_licenses_restating_the_table():
    system = _system_with_markdown(LIST_LIMIT + 1)
    assert "a small markdown table IS the answer" in system
    assert "Do NOT list its rows back" in system, "the summarise rule still applies"


def test_the_plain_opener_is_dropped_when_markdown_is_on():
    # "one or two plain sentences" outranked the markdown permission that followed it.
    assert "one or two plain sentences" in _system_for(3)
    assert "one or two plain sentences" not in _system_with_markdown(3)
