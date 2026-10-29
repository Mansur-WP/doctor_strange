"""Screenshot and recording tests without creating real media files."""

import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portal_app.capture import CaptureManager  # noqa: E402


class CaptureManagerTests(unittest.TestCase):
    def make_manager(self, directory: Path) -> CaptureManager:
        return CaptureManager(
            directory,
            clock=lambda: datetime(2026, 10, 29, 12, 34, 56, 123456),
        )

    @patch("pathlib.Path.mkdir")
    @patch("portal_app.capture.cv2.imwrite", return_value=True)
    def test_saves_timestamped_screenshot(
        self,
        imwrite: Mock,
        _mkdir: Mock,
    ) -> None:
        manager = self.make_manager(Path("virtual-captures"))
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        path = manager.save_screenshot(frame)

        self.assertEqual(path.name, "portal_20261029_123456_123456.png")
        imwrite.assert_called_once_with(str(path), frame)

    @patch("pathlib.Path.mkdir")
    @patch("portal_app.capture.cv2.imwrite", return_value=False)
    def test_reports_screenshot_write_failure(
        self,
        _imwrite: Mock,
        _mkdir: Mock,
    ) -> None:
        manager = self.make_manager(Path("virtual-captures"))
        with self.assertRaises(RuntimeError):
            manager.save_screenshot(np.zeros((10, 10, 3), dtype=np.uint8))

    @patch("pathlib.Path.mkdir")
    @patch("portal_app.capture.cv2.VideoWriter_fourcc", return_value=1234)
    @patch("portal_app.capture.cv2.VideoWriter")
    def test_starts_writes_and_stops_recording(
        self,
        video_writer: Mock,
        _fourcc: Mock,
        _mkdir: Mock,
    ) -> None:
        writer = Mock()
        writer.isOpened.return_value = True
        video_writer.return_value = writer
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        manager = self.make_manager(Path("virtual-captures"))
        path = manager.toggle_recording(frame)
        manager.write_frame(frame)
        stopped_path = manager.toggle_recording(frame)

        self.assertEqual(path.suffix, ".mp4")
        self.assertEqual(stopped_path, path)
        video_writer.assert_called_once_with(str(path), 1234, 30.0, (160, 120))
        writer.write.assert_called_once_with(frame)
        writer.release.assert_called_once_with()
        self.assertFalse(manager.is_recording)

    @patch("pathlib.Path.mkdir")
    @patch("portal_app.capture.cv2.VideoWriter_fourcc", return_value=1234)
    @patch("portal_app.capture.cv2.VideoWriter")
    def test_close_releases_active_writer(
        self,
        video_writer: Mock,
        _fourcc: Mock,
        _mkdir: Mock,
    ) -> None:
        writer = Mock()
        writer.isOpened.return_value = True
        video_writer.return_value = writer

        manager = self.make_manager(Path("virtual-captures"))
        manager.toggle_recording(np.zeros((20, 30, 3), dtype=np.uint8))
        manager.close()

        writer.release.assert_called_once_with()
        self.assertFalse(manager.is_recording)


if __name__ == "__main__":
    unittest.main()
