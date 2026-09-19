"""Minimal in-memory session summary.

Deliberately does not persist anything to disk by default: this is a
wellness tool, and the whole pitch is that nothing about the user's face,
posture, or session leaves the device (and by default, doesn't even get
written down). `export_json` is provided only for the judge-facing demo
log (`demo/sample_session_log.json`), and writes locally, never over the
network.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import List

from .fatigue_engine import FatigueEvent


@dataclass
class SessionSummary:
    started_at: float = field(default_factory=time.time)
    events: List[FatigueEvent] = field(default_factory=list)

    def record(self, event: FatigueEvent) -> None:
        self.events.append(event)

    def print_live(self, event: FatigueEvent) -> None:
        print(f"[FocusGuard] {event.kind.upper()} ({event.severity}): {event.detail}")

    def export_json(self, path: str) -> None:
        payload = {
            "started_at": self.started_at,
            "duration_seconds": time.time() - self.started_at,
            "event_count": len(self.events),
            "events": [asdict(e) for e in self.events],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
