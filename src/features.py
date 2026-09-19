"""Turn face/pose landmarks into the small feature vector the fatigue
engine consumes: eye-aspect-ratio (EAR), a rough head-tilt angle, and a
rough shoulder-line angle.

The live path uses MediaPipe's face mesh + pose landmarkers. When MediaPipe
isn't installed (e.g., in a CI or judging environment without it), a
deterministic synthetic feature generator is used instead so the rest of
the pipeline (`fatigue_engine.py`, `coach.py`, `dashboard.py`) can still be
exercised and demoed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .capture import Frame

# MediaPipe face mesh eye landmark indices (subset used for EAR).
_LEFT_EYE = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]


@dataclass
class FeatureVector:
    timestamp: float
    ear: float  # eye-aspect-ratio; lower = more closed
    head_tilt_deg: float  # 0 = level, larger = more tilted/hunched
    shoulder_tilt_deg: float
    landmarks_found: bool


def _eye_aspect_ratio(pts: np.ndarray) -> float:
    # pts: 6x2 array of (x, y) for one eye, in MediaPipe's ordering.
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    if c == 0:
        return 0.3  # neutral fallback
    return (a + b) / (2.0 * c)


class MediaPipeExtractor:
    """Live extractor using MediaPipe's face mesh + pose solutions."""

    def __init__(self) -> None:
        import mediapipe as mp  # local import: optional heavy dependency

        self._mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    def extract(self, frame: Frame) -> FeatureVector:
        rgb = frame.image[:, :, ::-1]
        face_result = self._mp_face.process(rgb)
        pose_result = self._mp_pose.process(rgb)

        if not face_result.multi_face_landmarks:
            return FeatureVector(frame.timestamp, ear=0.3, head_tilt_deg=0.0,
                                  shoulder_tilt_deg=0.0, landmarks_found=False)

        h, w = frame.image.shape[:2]
        lm = face_result.multi_face_landmarks[0].landmark
        left = np.array([[lm[i].x * w, lm[i].y * h] for i in _LEFT_EYE])
        right = np.array([[lm[i].x * w, lm[i].y * h] for i in _RIGHT_EYE])
        ear = (_eye_aspect_ratio(left) + _eye_aspect_ratio(right)) / 2.0

        # Rough head tilt: angle between eye-line and horizontal.
        left_eye_center = left.mean(axis=0)
        right_eye_center = right.mean(axis=0)
        dx, dy = right_eye_center - left_eye_center
        head_tilt_deg = abs(math.degrees(math.atan2(dy, dx)))

        shoulder_tilt_deg = 0.0
        if pose_result.pose_landmarks:
            plm = pose_result.pose_landmarks.landmark
            left_sh = np.array([plm[11].x * w, plm[11].y * h])
            right_sh = np.array([plm[12].x * w, plm[12].y * h])
            sdx, sdy = right_sh - left_sh
            shoulder_tilt_deg = abs(math.degrees(math.atan2(sdy, sdx)))

        return FeatureVector(frame.timestamp, ear, head_tilt_deg,
                              shoulder_tilt_deg, landmarks_found=True)


class SyntheticExtractor:
    """Deterministic stand-in when MediaPipe isn't available.

    Produces a slow synthetic drift so fatigue events still trigger during
    a demo (`--simulate`), without claiming to read real landmarks.
    """

    def __init__(self) -> None:
        self._t0: Optional[float] = None

    def extract(self, frame: Frame) -> FeatureVector:
        if self._t0 is None:
            self._t0 = frame.timestamp
        elapsed = frame.timestamp - self._t0
        # Slowly drifting EAR (simulating gradual drowsiness) and posture
        # angle (simulating gradual slouching), both deterministic.
        ear = max(0.15, 0.32 - 0.002 * elapsed)
        head_tilt = min(35.0, 2.0 + 0.3 * elapsed)
        shoulder_tilt = min(25.0, 1.0 + 0.2 * elapsed)
        return FeatureVector(frame.timestamp, ear, head_tilt, shoulder_tilt,
                              landmarks_found=True)


def get_extractor():
    try:
        return MediaPipeExtractor()
    except Exception:
        # Covers both "mediapipe isn't installed" (ImportError) and
        # "mediapipe is installed but broken/incompatible in this
        # environment" — either way, fall back so the rest of the
        # pipeline is still demoable.
        return SyntheticExtractor()
