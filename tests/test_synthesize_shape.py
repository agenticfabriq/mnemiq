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
