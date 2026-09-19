# Models

No trained weights are checked into this repo yet.

## Planned models

1. **Face + pose landmark model** — used to compute eye-aspect-ratio and
   posture angle. Candidate sources: MediaPipe's landmark models (exportable
   to ONNX) or an equivalent model from the Qualcomm AI Hub model zoo tuned
   for Snapdragon.
2. **Fatigue/posture classifier** — small dense network over the rolling
   feature window, trained on session data collected from the rule-based
   engine's outputs plus manual labeling. Export target: ONNX, opset 17+.
3. **(Stretch) local coaching LLM** — a small instruction-tuned model
   (Llama 3.2 3B–class, via Qualcomm AI Hub) to turn structured fatigue
   events into natural-language nudges.

## Deployment path (not yet executed — see docs/HONESTY_DISCLOSURE.md)

```bash
# 1. Export to ONNX (illustrative; exact command depends on source model)
python export_to_onnx.py --model landmark_model.tflite --out models/landmark.onnx

# 2. Profile on Qualcomm AI Hub's device farm
qai-hub submit-profile-job --model models/landmark.onnx --device "Snapdragon X Elite CRD"

# 3. Run locally via ONNX Runtime's QNN execution provider (on real hardware)
python -c "
import onnxruntime as ort
sess = ort.InferenceSession('models/landmark.onnx', providers=['QNNExecutionProvider'])
"
```

Until this repo is run against real Snapdragon hardware or the Qualcomm AI
Hub device farm, treat every latency/power number in the pitch deck as a
target derived from Qualcomm's published profiling data for comparable
models — not a measurement of this specific model.
