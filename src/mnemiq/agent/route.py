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


@dataclass
class StaticRouter:
    default: str = DEFAULT_MODE

    def route(self, question: str, override: str | None) -> str:
        if override is not None:
            if override not in MODES:
                raise UnknownMode(
                    f"unknown mode {override!r}; valid modes: {sorted(MODES)}"
                )
            return override
        return self.default
