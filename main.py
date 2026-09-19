"""FocusGuard AI — entry point.

Usage:
    python main.py --simulate            # no webcam/hardware required
    python main.py                       # live webcam, CPU fallback path
    python main.py --simulate --frames 200 --export demo/out.json
"""
from __future__ import annotations

import argparse

from src.capture import get_source
from src.coach import TemplateCoach
from src.dashboard import SessionSummary
from src.fatigue_engine import RuleBasedEngine
from src.features import get_extractor


def main() -> None:
    parser = argparse.ArgumentParser(description="FocusGuard AI")
    parser.add_argument("--simulate", action="store_true",
                         help="Run on synthetic frames; no webcam needed.")
    parser.add_argument("--frames", type=int, default=None,
                         help="Number of frames to process (simulate mode).")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--export", type=str, default=None,
                         help="Path to write a JSON session summary to.")
    args = parser.parse_args()

    source = get_source(simulate=args.simulate, camera_index=args.camera_index,
                         num_frames=args.frames)
    extractor = get_extractor()
    engine = RuleBasedEngine()
    coach = TemplateCoach()
    summary = SessionSummary()

    print("FocusGuard AI running. Ctrl+C to stop.")
    try:
        for frame in source.frames():
            fv = extractor.extract(frame)
            event = engine.classify(fv)
            if event is not None:
                summary.record(event)
                summary.print_live(event)
                print(f"  -> {coach.message_for(event)}")
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        if args.export:
            summary.export_json(args.export)
            print(f"Session summary written to {args.export}")
        print(f"Session ended. {len(summary.events)} event(s) detected.")


if __name__ == "__main__":
    main()
