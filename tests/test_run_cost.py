"""The reported cost of an eval run has to include every client the run used.

`build_engine` returned the generator's client under the comment "one client, so the token
count is the run's true cost". A configured judge runs on its OWN client, so its calls landed
on a counter nobody summed. It surfaced by measurement, not review: a judge arm reported 324
LLM calls against its control's 350 -- fewer, while adding a call per answered case.
"""

from mnemiq.eval.engine import RunCost


class _Client:
    def __init__(self, calls, prompt, completion):
        self.calls, self.prompt_tokens, self.completion_tokens = calls, prompt, completion

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


def test_cost_sums_every_client():
    generator, judge = _Client(200, 90_000, 10_000), _Client(100, 40_000, 2_000)

    cost = RunCost(generator, judge)

    assert cost.calls == 300, "the judge's calls are part of what the run cost"
    assert cost.total_tokens == 142_000
    assert cost.prompt_tokens == 130_000 and cost.completion_tokens == 12_000


def test_no_judge_configured_reports_the_generator_alone():
    # The common path: judge off. RunCost must not change the number it used to report.
    generator = _Client(200, 90_000, 10_000)

    cost = RunCost(generator, None)

    assert cost.calls == 200 and cost.total_tokens == 100_000


def test_a_judge_can_only_raise_the_reported_cost():
    """The regression that hid it: adding a judge made the REPORTED number go DOWN."""
    generator, judge = _Client(200, 90_000, 10_000), _Client(100, 40_000, 2_000)

    without = RunCost(generator, None)
    with_judge = RunCost(generator, judge)

    assert with_judge.calls > without.calls
    assert with_judge.total_tokens > without.total_tokens
