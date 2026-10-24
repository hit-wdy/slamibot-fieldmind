"""Deciding which arriving frames to index.

Spec section 4 is explicit that this must not be "take every tenth frame": a
dropped input frame would silently stretch the index interval. The decision is
made on a monotonic clock instead, and the counts keep the two kinds of loss
apart -- frames skipped on purpose versus frames the pipeline could not keep up
with.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SamplerStats:
    seen: int = 0
    kept: int = 0
    skipped_by_plan: int = 0
    overdue_recoveries: int = 0
    mean_gap_s: float = 0.0
    longest_gap_s: float = 0.0
    shortest_gap_s: float = 0.0

    @property
    def observed_hz(self) -> float:
        """The rate frames were actually kept at, not the rate that was asked for."""
        return 0.0 if self.mean_gap_s <= 0 else 1.0 / self.mean_gap_s

    def as_dict(self) -> dict:
        return {
            "frames_seen": self.seen,
            "frames_kept": self.kept,
            "skipped_by_plan": self.skipped_by_plan,
            "overdue_recoveries": self.overdue_recoveries,
            "mean_gap_s": round(self.mean_gap_s, 4),
            "observed_hz": round(self.observed_hz, 3),
            "longest_gap_s": round(self.longest_gap_s, 4),
            "shortest_gap_s": round(self.shortest_gap_s, 4),
        }


class IntervalSampler:
    """Keep one frame per `interval_s`, measured on a monotonic clock.

    If the loop falls behind (a slow encode, a stalled device), the next kept
    frame is `overdue`, counted separately, rather than silently accepted as if
    it had arrived on time.
    """

    def __init__(self, interval_s: float, clock=time.monotonic, overdue_factor: float = 1.5) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s = float(interval_s)
        self.overdue_factor = float(overdue_factor)
        self._clock = clock
        self._next_at: float | None = None
        self._last_kept: float | None = None
        self._gap_sum = 0.0
        self.stats = SamplerStats()

    def observe(self) -> bool:
        """Call once per arriving frame; True means index this one."""
        now = self._clock()
        self.stats.seen += 1
        if self._next_at is None:
            self._next_at = now + self.interval_s
            return self._keep(now)
        if now < self._next_at:
            self.stats.skipped_by_plan += 1
            return False
        # Advance on a grid so the rate does not drift with loop jitter.
        behind = now - self._next_at
        steps = int(behind // self.interval_s) + 1
        self._next_at += steps * self.interval_s
        if behind > self.interval_s * self.overdue_factor:
            self.stats.overdue_recoveries += 1
        return self._keep(now)

    def _keep(self, now: float) -> bool:
        if self._last_kept is not None:
            gap = now - self._last_kept
            self._gap_sum += gap
            self.stats.longest_gap_s = max(self.stats.longest_gap_s, gap)
            if not self.stats.shortest_gap_s:
                self.stats.shortest_gap_s = gap
            else:
                self.stats.shortest_gap_s = min(self.stats.shortest_gap_s, gap)
            self.stats.mean_gap_s = self._gap_sum / max(self.stats.kept, 1)
        self._last_kept = now
        self.stats.kept += 1
        return True
