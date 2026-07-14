from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Deadline:
    wall_clock_s: float
    started_at: float

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    @property
    def expired(self) -> bool:
        return (time.perf_counter() - self.started_at) >= self.wall_clock_s


@dataclass
class Budget:
    """Hard caps. A budget that can be exceeded is not a budget."""

    wall_clock_s: float = 60.0
    max_attempts: int = 3

    def started(self) -> Deadline:
        return Deadline(wall_clock_s=self.wall_clock_s, started_at=time.perf_counter())
