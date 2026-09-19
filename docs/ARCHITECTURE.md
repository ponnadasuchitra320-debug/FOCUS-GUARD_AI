# Architecture

## Pipeline overview

```
Webcam frame
   │
   ▼
Landmark extraction (face mesh + pose)      ── src/capture.py, src/features.py
   │
   ▼
Feature vector: eye-aspect-ratio, blink
rate, head-tilt angle, shoulder-line angle,
stillness duration
   │
   ▼
Fatigue/posture classifier                  ── src/fatigue_engine.py
   │  (rule-based today; pluggable ONNX model)
   ▼
Event: {type: "posture" | "eye_strain" | "drowsiness", severity}
   │
   ▼
Local coaching layer                        ── src/coach.py
   │  (templated messages today; pluggable local LLM)
   ▼
On-screen nudge + in-memory session summary ── src/dashboard.py
```

Nothing in this pipeline writes raw video, landmarks, or derived features to
disk or network by default. The only persisted artifact is an anonymized,
in-memory session summary (counts and durations, no imagery).

## Current implementation (CPU, this repo)

- `src/capture.py` — OpenCV webcam capture loop.
- `src/features.py` — MediaPipe face mesh + pose landmarks converted into
  eye-aspect-ratio (EAR) and simple 2D posture angles.
- `src/fatigue_engine.py` — threshold/rule-based classifier over rolling
  windows of the feature stream. Exposes the same interface an ONNX model
  would, so it's a drop-in swap.
- `src/coach.py` — maps classifier events to short natural-language nudges.
  Template-based today; designed to call a local LLM (see below) when one is
  available.

## Target on-device path (Snapdragon, not yet executed)

1. **Landmark model → ONNX.** Export the face/pose landmark model to ONNX
   (MediaPipe's models are already TFLite/ONNX-exportable; alternatively use
   a Qualcomm AI Hub–hosted equivalent model directly).
2. **Fatigue classifier → ONNX.** Once enough synthetic/labeled session data
   exists, replace the rule-based `fatigue_engine.py` logic with a small
   trained classifier (a few dense layers over the feature vector), exported
   to ONNX.
3. **QNN execution provider.** Load both ONNX models through ONNX Runtime's
   QNN execution provider so inference runs on the Hexagon NPU instead of
   CPU/GPU — this is the part that needs real Snapdragon hardware or
   Qualcomm AI Hub's device farm to validate.
4. **Local LLM coaching layer.** Swap the templated messages in
   `src/coach.py` for a small local model (e.g., a Llama 3.2 3B–class model
   from Qualcomm AI Hub) so nudges are contextual rather than templated,
   while still running entirely on-device.

## Why this fits an NPU workload

The vision half of this pipeline needs to run continuously and cheaply for
hours at a time — exactly the "always-on, low power, sustained" profile the
Hexagon NPU is designed for, as opposed to bursty, high-power workloads
better suited to GPU.
