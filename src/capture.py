"""Frame capture.

Two sources are supported:
- Live webcam via OpenCV (default).
- Simulated synthetic frames, so the pipeline can be demoed and judged
  without a webcam or Snapdragon hardware (see docs/HONESTY_DISCLOSURE.md).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # BGR, HxWx3
    timestamp: float


class WebcamSource:
    """Live capture from the default webcam."""

    def __init__(self, camera_index: int = 0) -> None:
        import cv2  # local import: only required for the live path

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Use --simulate to run without one."
            )

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            yield Frame(image=image, timestamp=time.time())

    def close(self) -> None:
        self._cap.release()


class SimulatedSource:
    """Synthetic frame generator for demos / judging without hardware.

    Produces plausible-looking noise frames with a slowly drifting
    brightness so downstream code has something to run against. This does
    NOT simulate real face/pose landmarks — it exists purely so the full
    pipeline is runnable end-to-end without a camera.
    """

    def __init__(self, num_frames: Optional[int] = None, fps: float = 15.0) -> None:
        self._num_frames = num_frames
        self._period = 1.0 / fps

    def frames(self) -> Iterator[Frame]:
        rng = np.random.default_rng(seed=42)
        count = 0
        while self._num_frames is None or count < self._num_frames:
            image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
            yield Frame(image=image, timestamp=time.time())
            count += 1
            time.sleep(self._period)

    def close(self) -> None:
        pass


def get_source(simulate: bool, camera_index: int = 0, num_frames: Optional[int] = None):
    if simulate:
        return SimulatedSource(num_frames=num_frames)
    return WebcamSource(camera_index=camera_index)
