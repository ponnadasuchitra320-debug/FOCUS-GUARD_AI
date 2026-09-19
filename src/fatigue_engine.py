"""Classify rolling windows of features into fatigue/posture events.

Two implementations share one interface (`classify(window) -> Optional[Event]`)
so the rule-based engine used today can be swapped for a trained ONNX model
without touching any other module:

- `RuleBasedEngine` — thresholds over rolling windows. Implemented and used
  by default.
- `OnnxEngine` — loads a trained classifier via ONNX Runtime. Structured to
  request the QNN execution provider on Snapdragon, falling back to CPU
  elsewhere. NOT exercised against real hardware yet — see
  docs/HONESTY_DISCLOSURE.md.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .features import FeatureVector

EAR_CLOSED_THRESHOLD = 0.21
EAR_CLOSED_SECONDS = 3.0
HEAD_TILT_THRESHOLD_DEG = 15.0
SHOULDER_TILT_THRESHOLD_DEG = 10.0
POOR_POSTURE_SECONDS = 300.0  # 5 minutes sustained


@dataclass
class FatigueEvent:
    kind: str  # "drowsiness" | "posture"
    severity: str  # "mild" | "moderate"
    detail: str


class RuleBasedEngine:
    def __init__(self, window_seconds: float = 300.0) -> None:
        self._window_seconds = window_seconds
        self._buffer: Deque[FeatureVector] = deque()
        self._eyes_closed_since: Optional[float] = None
        self._bad_posture_since: Optional[float] = None
        self._drowsiness_fired = False
        self._posture_fired = False

    def _trim(self, now: float) -> None:
        while self._buffer and now - self._buffer[0].timestamp > self._window_seconds:
            self._buffer.popleft()

    def classify(self, fv: FeatureVector) -> Optional[FatigueEvent]:
        self._buffer.append(fv)
        self._trim(fv.timestamp)

        event: Optional[FatigueEvent] = None

        # Drowsiness: sustained low EAR (eyes mostly closed).
        if fv.ear < EAR_CLOSED_THRESHOLD:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = fv.timestamp
            elif (fv.timestamp - self._eyes_closed_since >= EAR_CLOSED_SECONDS
                  and not self._drowsiness_fired):
                event = FatigueEvent(
                    kind="drowsiness", severity="moderate",
                    detail=f"Eyes mostly closed for "
                           f"{fv.timestamp - self._eyes_closed_since:.0f}s",
                )
                self._drowsiness_fired = True
        else:
            self._eyes_closed_since = None
            self._drowsiness_fired = False

        # Posture: sustained head/shoulder tilt beyond threshold.
        bad_posture_now = (
            fv.head_tilt_deg > HEAD_TILT_THRESHOLD_DEG
            or fv.shoulder_tilt_deg > SHOULDER_TILT_THRESHOLD_DEG
        )
        if bad_posture_now:
            if self._bad_posture_since is None:
                self._bad_posture_since = fv.timestamp
            elif (fv.timestamp - self._bad_posture_since >= POOR_POSTURE_SECONDS
                  and not self._posture_fired and event is None):
                event = FatigueEvent(
                    kind="posture", severity="mild",
                    detail=f"Slouched/tilted posture for "
                           f"{(fv.timestamp - self._bad_posture_since) / 60:.0f} min",
                )
                self._posture_fired = True
        else:
            self._bad_posture_since = None
            self._posture_fired = False

        return event


class OnnxEngine:
    """Pluggable ONNX-backed classifier.

    Not used by default and not validated on real hardware. Structured so
    that swapping RuleBasedEngine for this class in main.py is the only
    change needed once a trained model + real Snapdragon target exist.
    """

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)

    def classify(self, fv: FeatureVector) -> Optional[FatigueEvent]:
        raise NotImplementedError(
            "OnnxEngine requires a trained model; see models/README.md."
        )
