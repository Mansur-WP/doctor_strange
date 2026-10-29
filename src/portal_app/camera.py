"""Webcam capture and frame-rate utilities."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import cv2

from portal_app.capture import CaptureManager


class FrameProcessor(Protocol):
    """A component that annotates one camera frame in place."""

    def process(self, frame, timestamp_ms: int) -> None:
        """Process a BGR frame captured at a monotonic timestamp."""


@dataclass(frozen=True)
class CameraConfig:
    """Settings requested from the webcam driver."""

    index: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True


class FPSCounter:
    """Calculate a stable FPS value using exponential smoothing."""

    def __init__(self, smoothing: float = 0.9) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be between 0.0 (inclusive) and 1.0")

        self.smoothing = smoothing
        self._previous_time: float | None = None
        self._fps = 0.0

    @property
    def value(self) -> float:
        """Return the latest smoothed frames-per-second value."""

        return self._fps

    def update(self, now: float | None = None) -> float:
        """Record one frame and return the updated FPS estimate."""

        current_time = perf_counter() if now is None else now
        if self._previous_time is None:
            self._previous_time = current_time
            return self._fps

        elapsed = current_time - self._previous_time
        self._previous_time = current_time
        if elapsed <= 0.0:
            return self._fps

        instantaneous_fps = 1.0 / elapsed
        if self._fps == 0.0:
            self._fps = instantaneous_fps
        else:
            self._fps = (
                self.smoothing * self._fps
                + (1.0 - self.smoothing) * instantaneous_fps
            )
        return self._fps


def open_camera(config: CameraConfig) -> cv2.VideoCapture:
    """Open and configure a webcam, or raise a useful error."""

    capture = cv2.VideoCapture(config.index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera index {config.index}. "
            "Check camera permissions and whether another app is using it."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    return capture


def draw_fps(frame, fps: float) -> None:
    """Draw an easy-to-read FPS label onto a frame in place."""

    label = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        label,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )


def draw_recording_status(frame) -> None:
    """Draw a clear recording indicator without covering the FPS label."""

    cv2.circle(frame, (28, 78), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "REC",
        (45, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_camera(
    config: CameraConfig,
    *,
    frame_processor: FrameProcessor | None = None,
    capture_manager: CaptureManager | None = None,
    headless: bool = False,
    max_frames: int | None = None,
) -> int:
    """Run the capture loop and return the number of processed frames."""

    capture = open_camera(config)
    fps_counter = FPSCounter()
    frames_processed = 0
    window_name = "Doctor Strange Portal - Stages 13-14"
    start_time = perf_counter()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("The camera opened, but a frame could not be read.")

            if config.mirror:
                frame = cv2.flip(frame, 1)

            if frame_processor is not None:
                timestamp_ms = int((perf_counter() - start_time) * 1000)
                frame_processor.process(frame, timestamp_ms)

            fps = fps_counter.update()
            draw_fps(frame, fps)
            if capture_manager is not None and capture_manager.is_recording:
                draw_recording_status(frame)
            if capture_manager is not None:
                capture_manager.write_frame(frame)
            frames_processed += 1

            if not headless:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if capture_manager is not None and key == ord("s"):
                    path = capture_manager.save_screenshot(frame)
                    print(f"Saved screenshot: {path}")
                elif capture_manager is not None and key == ord("r"):
                    was_recording = capture_manager.is_recording
                    path = capture_manager.toggle_recording(frame)
                    action = "Saved recording" if was_recording else "Recording started"
                    print(f"{action}: {path}")

            if max_frames is not None and frames_processed >= max_frames:
                break
    finally:
        capture.release()
        if capture_manager is not None:
            capture_manager.close()
        cv2.destroyAllWindows()

    return frames_processed

