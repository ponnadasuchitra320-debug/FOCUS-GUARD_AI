# Honesty Disclosure

This document states plainly what was and was not verified on real Snapdragon
hardware, per the spirit of the Build & Present Challenge.

## What we did NOT have

- No physical Snapdragon-powered HP PC was available to the team during
  development.
- No on-device NPU (Hexagon) execution was performed or timed by us.
- No Qualcomm AI Hub device-farm profiling run was executed for this specific
  model by us.

## What we DID build and verify

- The full capture → feature-extraction → fatigue-classification → coaching
  loop runs end-to-end on a standard x86 development laptop, on CPU, using
  MediaPipe/OpenCV for face and pose landmarks.
- The fatigue/posture rule engine (`src/fatigue_engine.py`) is implemented
  and unit-testable against the synthetic session log in
  `demo/sample_session_log.json`.
- The model export path to ONNX (`models/README.md`) follows Qualcomm AI
  Hub's public documentation for taking a landmark/classification model to
  the Hexagon NPU via the QNN execution provider, but that export and the
  resulting NPU run have not been executed by us against real hardware.

## What any performance numbers in the pitch deck represent

Any NPU latency/throughput figures cited in the pitch materials are drawn
from Qualcomm AI Hub's published profiling data for comparable
lightweight vision models on Snapdragon X Elite/Plus, not from our own
measurements. This is called out explicitly wherever such a number appears.

## Plan to close the gap

1. Export the trained landmark/classifier to ONNX.
2. Run Qualcomm AI Hub's cloud device-farm profiling against a real
   Snapdragon X Elite/X Plus target to get real latency/power numbers.
3. If/when physical hardware access is available, run the full pipeline
   locally with the QNN execution provider and replace all projected
   numbers in the deck with measured ones.
