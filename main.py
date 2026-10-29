"""Command-line entry point for the portal application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from portal_app.camera import CameraConfig, run_camera  # noqa: E402
from portal_app.capture import CaptureManager  # noqa: E402
from portal_app.hand_tracking import (  # noqa: E402
    HandTracker,
    HandTrackingConfig,
)


def positive_integer(value: str) -> int:
    """Parse a command-line value that must be greater than zero."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track and draw up to two hands on a live webcam feed."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=positive_integer, default=1280)
    parser.add_argument("--height", type=positive_integer, default=720)
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not horizontally mirror the webcam image.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Capture frames without opening a display window (useful for checks).",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_integer,
        help="Stop automatically after this many frames.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "hand_landmarker.task",
        help="Path to the MediaPipe Hand Landmarker .task model.",
    )
    parser.add_argument(
        "--captures-dir",
        type=Path,
        default=PROJECT_ROOT / "captures",
        help="Directory used by the S screenshot and R recording controls.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.headless and args.max_frames is None:
        print("error: --headless requires --max-frames", file=sys.stderr)
        return 2

    config = CameraConfig(
        index=args.camera_index,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
    )

    tracking_config = HandTrackingConfig(model_path=args.model_path, max_hands=2)
    capture_manager = CaptureManager(args.captures_dir)

    try:
        with HandTracker(tracking_config) as hand_tracker:
            frame_count = run_camera(
                config,
                frame_processor=hand_tracker,
                capture_manager=capture_manager,
                headless=args.headless,
                max_frames=args.max_frames,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Application error: {error}", file=sys.stderr)
        return 1

    print(f"Processed {frame_count} frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

