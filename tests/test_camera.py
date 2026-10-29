"""Unit checks for Stage 1 utilities that do not require a webcam."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portal_app.camera import CameraConfig, FPSCounter, run_camera  # noqa: E402


class FPSCounterTests(unittest.TestCase):
    def test_calculates_frames_per_second_from_elapsed_time(self) -> None:
        counter = FPSCounter(smoothing=0.0)

        self.assertEqual(counter.update(now=10.0), 0.0)
        self.assertAlmostEqual(counter.update(now=10.05), 20.0)

    def test_smooths_changes_between_frames(self) -> None:
        counter = FPSCounter(smoothing=0.5)
        counter.update(now=1.0)
        counter.update(now=1.1)  # 10 FPS

        self.assertAlmostEqual(counter.update(now=1.15), 15.0)

    def test_rejects_invalid_smoothing(self) -> None:
        with self.assertRaises(ValueError):
            FPSCounter(smoothing=1.0)


class CameraPipelineTests(unittest.TestCase):
    @patch("portal_app.camera.cv2.destroyAllWindows")
    @patch("portal_app.camera.open_camera")
    def test_passes_frames_to_processor_and_releases_camera(
        self,
        open_camera: Mock,
        destroy_windows: Mock,
    ) -> None:
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        capture = Mock()
        capture.read.side_effect = [(True, frame.copy()), (True, frame.copy())]
        open_camera.return_value = capture
        processor = Mock()

        processed = run_camera(
            CameraConfig(mirror=False),
            frame_processor=processor,
            headless=True,
            max_frames=2,
        )

        self.assertEqual(processed, 2)
        self.assertEqual(processor.process.call_count, 2)
        capture.release.assert_called_once_with()
        destroy_windows.assert_called_once_with()

    @patch("portal_app.camera.cv2.waitKey")
    @patch("portal_app.camera.cv2.imshow")
    @patch("portal_app.camera.cv2.destroyAllWindows")
    @patch("portal_app.camera.open_camera")
    def test_handles_capture_keys_and_closes_manager(
        self,
        open_camera: Mock,
        destroy_windows: Mock,
        _imshow: Mock,
        wait_key: Mock,
    ) -> None:
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        capture = Mock()
        capture.read.side_effect = [(True, frame.copy()) for _ in range(4)]
        open_camera.return_value = capture
        wait_key.side_effect = [ord("s"), ord("r"), ord("r"), ord("q")]
        manager = Mock()
        manager.is_recording = False
        manager.save_screenshot.return_value = Path("capture.png")
        manager.toggle_recording.return_value = Path("capture.mp4")

        processed = run_camera(
            CameraConfig(mirror=False),
            capture_manager=manager,
        )

        self.assertEqual(processed, 4)
        manager.save_screenshot.assert_called_once()
        self.assertEqual(manager.toggle_recording.call_count, 2)
        self.assertEqual(manager.write_frame.call_count, 4)
        manager.close.assert_called_once_with()
        capture.release.assert_called_once_with()
        destroy_windows.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

