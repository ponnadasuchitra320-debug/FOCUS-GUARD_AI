"""Turn a FatigueEvent into a short, specific coaching message.

`TemplateCoach` is the default and needs no model at all. `LocalLLMCoach`
shows the intended integration point for a small on-device model (e.g., a
Llama 3.2 3B-class model from Qualcomm AI Hub) — not wired up to an actual
model in this scaffold.
"""
from __future__ import annotations

from .fatigue_engine import FatigueEvent

_TEMPLATES = {
    "drowsiness": [
        "Your eyes have been closed for a while — a short break might help.",
        "Looking pretty drowsy. Maybe stand up and get some water?",
    ],
    "posture": [
        "You've been in a slouched position for a while — try straightening up.",
        "Your posture's been off for a bit. A 30-second stretch could help.",
    ],
}


class TemplateCoach:
    def __init__(self) -> None:
        self._i = 0

    def message_for(self, event: FatigueEvent) -> str:
        options = _TEMPLATES.get(event.kind, ["Time for a quick break."])
        msg = options[self._i % len(options)]
        self._i += 1
        return f"{msg} ({event.detail})"


class LocalLLMCoach:
    """Integration point for a local, on-device LLM.

    Not implemented against a real model in this scaffold — see
    docs/HONESTY_DISCLOSURE.md. Intended shape: pass the structured event
    (kind, severity, detail, recent session stats) to a small local model
    and get back one short, specific sentence, entirely on-device.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        raise NotImplementedError(
            "Wire this up to a local model runtime (e.g., llama.cpp / "
            "Qualcomm AI Hub-hosted small LLM) before using LocalLLMCoach."
        )

    def message_for(self, event: FatigueEvent) -> str:  # pragma: no cover
        raise NotImplementedError
