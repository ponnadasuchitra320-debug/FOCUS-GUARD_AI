# FocusGuard AI

**On-device digital wellness & fatigue coach for Snapdragon-powered HP PCs.**

Submitted for the Snapdragon® AI Lab Build & Present Challenge (Qualcomm, 2026).

## Problem

People working or studying for long stretches on a laptop build up eye strain,
poor posture, and mental fatigue long before they consciously notice it.
Existing wellness apps either require uploading webcam footage to the cloud
(a privacy non-starter) or are just dumb Pomodoro timers with no real signal
about the person's actual physical state.

## What it does

FocusGuard AI runs quietly in the background and uses the laptop's own webcam
to locally detect early signs of fatigue and poor posture — blink rate, eye
closure duration, head/neck angle, and prolonged stillness.The current prototype performs local CPU inference. The planned Snapdragon deployment is designed for on-device NPU inference so that raw webcam data can remain on the device.No frame, landmark, or derived feature ever leaves the
machine. When a pattern crosses a threshold, a local coaching engine turns detected patterns into a short, specific nudge ("You've been hunched forward for 12
minutes — want to stretch?") instead of a generic timer ping.

## Why Snapdragon

The vision pipeline (face/pose landmark extraction + a lightweight
classifier) is designed to run continuously, all day, in the background.
That's exactly the workload profile the Snapdragon Hexagon NPU is built for:
sustained, low-power, always-on inference that would drain a battery or spin
up a fan on a CPU/GPU. See `docs/ARCHITECTURE.md` for the target on-device
pipeline (ONNX export → Qualcomm AI Hub → QNN execution provider on Hexagon).

## Honesty disclosure

**Read `docs/HONESTY_DISCLOSURE.md` before evaluating this repo.** In short:
this project was built and tested entirely on a non-Snapdragon development
machine (no Snapdragon hardware was available to the team). Landmark
extraction and the fatigue classifier run today on CPU via MediaPipe/OpenCV.
The ONNX export path and QNN/NPU execution have been written against
Qualcomm AI Hub's documented APIs but have **not** been run on real Snapdragon
silicon. Anything claimed about NPU performance is a target based on public
Qualcomm AI Hub profiling data for comparable models, not a measurement we
made ourselves.

## Repo layout

```
focusguard-ai/
├── main.py                  # orchestrates the capture -> feature -> coach loop
├── requirements.txt
├── src/
│   ├── capture.py           # webcam frame capture
│   ├── features.py          # landmark -> EAR (eye aspect ratio) + posture angle
│   ├── fatigue_engine.py    # rule-based + pluggable ONNX classifier
│   ├── coach.py             # turns fatigue events into coaching messages
│   └── dashboard.py         # local, in-memory session summary (no persistence off-device)
├── models/
│   └── README.md            # ONNX export / QNN deployment notes
├── demo/
│   └── sample_session_log.json  # synthetic example output for judges
└── docs/
    ├── ARCHITECTURE.md
    └── HONESTY_DISCLOSURE.md
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py --simulate   # runs on synthetic frames, no webcam/hardware required
python main.py              # runs live against your webcam (CPU fallback path)
```

## Status

Early scaffold built for the Build & Present Challenge submission window.
Core loop, feature extraction, and rule-based fatigue detection are
implemented and runnable on CPU. NPU deployment (ONNX export + QNN provider)
is scoped in `models/README.md` but not yet executed on target hardware.
