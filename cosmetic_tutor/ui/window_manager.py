# window_manager.py
"""
Window manager for creating a custom UI with background image
and positioned webcam feed.
"""

import cv2
import numpy as np
from pathlib import Path
from cosmetic_tutor.types import ROI, BlushMetrics, Advice


class WindowManager:
    """Manages the main application window with background image and webcam overlay."""

    def __init__(self, webcam_width: int = 640, webcam_height: int = 480):
        self.webcam_width = webcam_width
        self.webcam_height = webcam_height

        # UI scale factor 
        self.ui_scale = 3

        # Load background image
        background_path = self._get_background_path()
        self.background = cv2.imread(str(background_path))

        if self.background is None:
            raise FileNotFoundError(
                f"Could not load background image from {background_path}"
            )

        # Background dimensions
        self.bg_height, self.bg_width = self.background.shape[:2]

        # Webcam position (top-left with padding)
        self.webcam_x = 20
        self.webcam_y = 20

        # Create window
        cv2.namedWindow("Cosmetic Tutor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cosmetic Tutor", self.bg_width, self.bg_height)

    def _get_background_path(self) -> Path:
        """Get the path to the background image."""
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent

        bg_jpg = project_root / "cosmetic_tutor" / "ui" / "background.jpg"
        bg_png = project_root / "cosmetic_tutor" / "ui" / "background.png"

        if bg_jpg.exists():
            return bg_jpg
        if bg_png.exists():
            return bg_png

        raise FileNotFoundError(
            "No background.jpg or background.png found in cosmetic_tutor/ui/"
        )

    def update_display(
        self,
        webcam_frame: np.ndarray,
        left_roi: ROI | None = None,
        right_roi: ROI | None = None,
        metrics: BlushMetrics | None = None,
        advice: Advice | None = None,
    ) -> np.ndarray:
        """Update the main display with webcam feed and overlays."""

        display_frame = self.background.copy()

        if webcam_frame is not None:
            resized_webcam = cv2.resize(
                webcam_frame, (self.webcam_width, self.webcam_height)
            )

            end_x = min(self.webcam_x + self.webcam_width, self.bg_width)
            end_y = min(self.webcam_y + self.webcam_height, self.bg_height)

            actual_width = end_x - self.webcam_x
            actual_height = end_y - self.webcam_y

            if actual_width > 0 and actual_height > 0:
                if (
                    actual_width < self.webcam_width
                    or actual_height < self.webcam_height
                ):
                    resized_webcam = resized_webcam[
                        :actual_height, :actual_width
                    ]

                display_frame[
                    self.webcam_y:end_y,
                    self.webcam_x:end_x
                ] = resized_webcam

        self._draw_ui_elements(display_frame, metrics, advice)
        return display_frame

    def _draw_ui_elements(
        self,
        frame: np.ndarray,
        metrics: BlushMetrics | None = None,
        advice: Advice | None = None,
    ) -> None:
        """Draw UI text and analysis results."""

        if metrics is None or advice is None:
            return

        text_x = self.webcam_x + self.webcam_width + 20
        text_y = self.webcam_y + 30

        if text_x + 300 > self.bg_width:
            text_x = self.webcam_x
            text_y = self.webcam_y + self.webcam_height + 30

        lines = [
            "=== BLUSH ANALYSIS ===",
            f"Left Delta: {metrics.left_delta_a:.1f}",
            f"Right Delta: {metrics.right_delta_a:.1f}",
            f"Uneven Score: {metrics.uneven_score:.1f}",
            f"Left Coverage: {metrics.left_coverage:.2f}",
            f"Right Coverage: {metrics.right_coverage:.2f}",
            f"Offset Direction: {metrics.offset_dir}",
            "",
            "=== ADVICE ===",
            advice.text,
            "",
            "Controls:",
            "[b] - Calibrate baseline",
            "[q] - Quit",
        ]

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            y_pos = text_y + i * 25
            if y_pos + 25 > self.bg_height:
                break

            (w, h), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            cv2.rectangle(
                frame,
                (text_x - 5, y_pos - 20),
                (text_x + w + 10, y_pos + 5),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                frame,
                (text_x - 5, y_pos - 20),
                (text_x + w + 10, y_pos + 5),
                (255, 255, 255),
                1,
            )

            color = (0, 255, 255) if line.startswith("===") else (255, 255, 255)
            cv2.putText(
                frame,
                line,
                (text_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    def show(self, frame: np.ndarray) -> None:
        """Display the frame."""
        frame = cv2.resize(frame, None, fx=self.ui_scale, fy=self.ui_scale)
        cv2.imshow("Cosmetic Tutor", frame)

    def destroy(self) -> None:
        """Destroy window."""
        cv2.destroyWindow("Cosmetic Tutor")
