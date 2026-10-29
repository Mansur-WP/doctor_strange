"""Screenshot and video recording utilities for shareable spell captures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


class CaptureManager:
    """Save screenshots and own at most one active OpenCV video writer."""

    def __init__(
        self,
        output_dir: Path,
        *,
        recording_fps: float = 30.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if recording_fps <= 0.0:
            raise ValueError("recording_fps must be greater than zero")
        self.output_dir = output_dir
        self.recording_fps = recording_fps
        self._clock = clock or datetime.now
        self._writer = None
        self._recording_path: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def recording_path(self) -> Path | None:
        return self._recording_path

    def save_screenshot(self, frame: np.ndarray) -> Path:
        """Save one annotated frame as a timestamped PNG."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"portal_{self._timestamp()}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not save screenshot: {path}")
        return path

    def toggle_recording(self, frame: np.ndarray) -> Path:
        """Start a new MP4 recording or stop and return the current path."""

        if self._writer is not None:
            path = self._recording_path
            self._writer.release()
            self._writer = None
            self._recording_path = None
            if path is None:
                raise RuntimeError("Recording path was unexpectedly missing")
            return path

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("recording frames must be BGR images")
        frame_height, frame_width = frame.shape[:2]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"portal_{self._timestamp()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            self.recording_fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not start video recording: {path}")
        self._writer = writer
        self._recording_path = path
        return path

    def write_frame(self, frame: np.ndarray) -> None:
        """Append a frame when recording; otherwise do nothing."""

        if self._writer is not None:
            self._writer.write(frame)

    def close(self) -> None:
        """Release an active writer safely."""

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._recording_path = None

    def _timestamp(self) -> str:
        return self._clock().strftime("%Y%m%d_%H%M%S_%f")
