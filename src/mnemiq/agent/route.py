from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mnemiq.agent.modes import DEFAULT_MODE, MODES


class UnknownMode(ValueError):
    """An explicit mode override that names no registered mode. Fail closed, never fall back."""


class Router(Protocol):
    """The seam a future capability router plugs into (keyword fast-path once deep mode has
    capabilities worth selecting for). The shipped router is deterministic: a difficulty
    classifier measured no exploitable signal (plan 18 calibration), so it was benched,
    not deferred."""

    def route(self, question: str, override: str | None) -> str: ...


def offered_modes(router) -> list[str]:
    """The modes `router` answers a caller's override with, sorted: the set it offers, or every registered mode when
    it names none (a router that is not a `StaticRouter`, or none at all, names none)."""
    offered = getattr(router, "offered", None)
    return sorted(MODES if offered is None else offered)


@dataclass
class StaticRouter:
    default: str = DEFAULT_MODE
    # The modes this deployment offers; None offers every registered mode. A mode the deployment has no agent for is
    # refused here, never answered by the default mode's agent in its place.
    offered: frozenset[str] | None = None

    def route(self, question: str, override: str | None) -> str:
        if override is not None:
            valid = MODES if self.offered is None else self.offered
            if override not in valid:
                raise UnknownMode(
                    f"unknown mode {override!r}; valid modes: {sorted(valid)}"
                )
            return override
        return self.default
