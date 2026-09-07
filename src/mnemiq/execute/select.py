from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

# Auto-accept only unanimity: a single cluster is the one case where selection is vacuous.
# Deliberate departure from DeepEye's 0.6 -- tuned for ~12 candidates; at our N=5 it would
# wave through 3/5 and 4/5 majorities, exactly the slices Plan 12 measured as usually wrong
# (28% and 33% correct). One constant makes the 0.6 experiment a one-line change.
AUTO_ACCEPT = 1.0

_INT = re.compile(r"-?\d+")


@dataclass
class ClusterView:
    """One executed result-cluster, as the judge sees it."""

    sql: str  # the representative candidate's plan_sql
    preview: str  # a few rendered rows + total row count
    size: int  # how many candidates landed in this cluster


class Selector(Protocol):
    def select(self, question: str, clusters: list[ClusterView]) -> int: ...


@dataclass(frozen=True)
class SelectorRead:
    """A pick, and whether it IS a pick or the fallback.

    The two are indistinguishable by value: a judge that studied the clusters and agreed with the
    majority returns the same index a dead endpoint returns. That is M11's second clause -- the
    loop derived `judge_engaged` from whether the clusters disagreed, so a failed selector shipped
    `judge_engaged=True, judge_override=False` on `/v1/ask` and in the audit record, byte for byte
    what a judgement produces. Carrying the fact beside the index is the only way a caller can tell
    them apart, and it is the same fix `JudgeRead` is for the verifier.

    NOT counters. `SemanticJudge` keeps cumulative ones for an eval sweep and its own docstring
    records why they cannot answer this: the selector is shared across every request, so a caller
    diffing a counter around its own call reads a concurrent request's failure as its own.
    """

    choice: int
    fell_back: bool
    # WHY, because the three causes want different responses and an operator must not have to
    # guess: `error` is an outage worth alerting on, `unparsed` means this model cannot emit the
    # format at all, and `out_of_range` means it answered in the right shape and named a cluster
    # that does not exist -- a prompt problem, not an availability one.
    reason: str = "ok"          # "ok" | "error" | "unparsed" | "out_of_range"


def majority_index(clusters: list[ClusterView]) -> int:
    """Largest cluster, first on ties -- the exact semantics of max(groups, key=len)."""
    best = 0
    for i, view in enumerate(clusters):
        if view.size > clusters[best].size:
            best = i
    return best


def auto_accepted(clusters: list[ClusterView]) -> bool:
    total = sum(v.size for v in clusters)
    return total > 0 and clusters[majority_index(clusters)].size / total >= AUTO_ACCEPT


class MajoritySelector:
    """Today's pick, behind the seam."""

    def select(self, question: str, clusters: list[ClusterView]) -> int:
        return majority_index(clusters)


_SYSTEM = """You are a careful data analyst judging candidate SQL answers to one question.
Each candidate already ran against the real database; you see its SQL, a preview of its
result, and how many independent attempts produced that same result.

Judge by which RESULT actually answers the question -- the right grain (one row per what?),
the right filters, the right aggregation. Agreement count is a hint, not a verdict: a
majority can share the same misreading.

Reply ONLY with a JSON object, no prose: {"choice": <candidate number>}"""


def _judge_prompt(question: str, clusters: list[ClusterView]) -> str:
    parts = [f"QUESTION: {question}"]
    for i, view in enumerate(clusters):
        parts += [
            "",
            f"CANDIDATE {i} (produced by {view.size} attempt{'s' if view.size != 1 else ''}):",
            f"SQL: {view.sql}",
            f"RESULT:\n{view.preview}",
        ]
    return "\n".join(parts)


class LLMSelector:
    """One judge call over the disagreeing clusters. Fail-closed: any failure -- a reply we
    cannot read, an out-of-range pick, a dead endpoint -- falls back to the majority. The
    judge only ever swaps one executed answer for another; it never writes SQL."""

    def __init__(self, client, max_tokens: int = 2000) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def read(self, question: str, clusters: list[ClusterView]) -> SelectorRead:
        """One pick, carrying whether the judge made it.

        The fallback is unchanged in every case -- this returns the same index `select` always
        returned. What is new is that the caller can tell a judgement from a fallback, which no
        consumer of the bare int ever could.
        """
        try:
            raw = self._client.complete(
                _SYSTEM, _judge_prompt(question, clusters), max_tokens=self._max_tokens
            )
        except Exception:
            return SelectorRead(majority_index(clusters), fell_back=True, reason="error")
        match = _INT.search(raw or "")
        # `int()` is guarded, not just the search: CPython refuses a conversion above 4300 digits
        # and a 2000-token reply has room for several times that, so a degenerate repetition
        # raises where the regex matched happily. Before this was a `read`, the parse sat inside
        # the try that catches the endpoint and fell back with everything else; pulling the client
        # call out of that try to tell an outage from an unreadable reply took the parse with it,
        # and turned a fallback into a killed request in the one function whose contract is that
        # it never kills one.
        try:
            choice = int(match.group(0)) if match is not None else None
        except ValueError:
            choice = None
        if choice is None:
            return SelectorRead(majority_index(clusters), fell_back=True, reason="unparsed")
        if not 0 <= choice < len(clusters):
            return SelectorRead(majority_index(clusters), fell_back=True, reason="out_of_range")
        return SelectorRead(choice, fell_back=False)

    def select(self, question: str, clusters: list[ClusterView]) -> int:
        """The int protocol every `Selector` speaks. Behaviour is unchanged, fallback included."""
        return self.read(question, clusters).choice


class FakeSelector:
    """Replays scripted picks; records what it was asked. Zero tokens."""

    def __init__(self, choices: list[int]) -> None:
        self._choices = list(choices)
        self.calls: list[tuple[str, list[ClusterView]]] = []

    def select(self, question: str, clusters: list[ClusterView]) -> int:
        self.calls.append((question, clusters))
        return self._choices.pop(0) if self._choices else majority_index(clusters)
