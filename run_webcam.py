import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from pathlib import Path
from cosmetic_tutor.config import AppConfig
from cosmetic_tutor.pipeline.runner import AppRunner


class BackgroundWindowManager:
    def __init__(self, webcam_width: int = 960, webcam_height: int = 720):
        self.webcam_width = webcam_width
        self.webcam_height = webcam_height

        # Load background image
        background_path = self._get_background_path()
        self.background = cv2.imread(str(background_path))

        if self.background is None:
            raise FileNotFoundError(
                f"Could not load background image from {background_path}"
            )

        # Get background dimensions
        self.bg_height, self.bg_width = self.background.shape[:2]

        # Scale up the background if it's smaller than desired window size
        min_width = max(1200, webcam_width + 500)
        min_height = max(800, webcam_height + 100)

        if self.bg_width < min_width or self.bg_height < min_height:
            scale_x = min_width / self.bg_width
            scale_y = min_height / self.bg_height
            scale = max(scale_x, scale_y)

            new_width = int(self.bg_width * scale)
            new_height = int(self.bg_height * scale)
            self.background = cv2.resize(self.background, (new_width, new_height))
            self.bg_height, self.bg_width = self.background.shape[:2]

        # Calculate webcam position (top-left corner with some padding)
        # === white frame region based on background ratio ===
        self.frame_x = int(self.bg_width * 0.035)
        self.frame_y = int(self.bg_height * 0.055)
        self.frame_width = int(self.bg_width * 0.58)
        self.frame_height = int(self.bg_height * 0.55)

        self.webcam_x = self.frame_x
        self.webcam_y = self.frame_y

        # === right-side text column ===
        self.text_col_x = int(self.bg_width * 0.67)
        self.text_col_y = int(self.bg_height * 0.12)
        self.text_col_width = int(self.bg_width * 0.30)

        # Create the main window
        cv2.namedWindow(
            "Cosmetic Tutor",
            cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED
        )

        # === limit window size to screen-friendly resolution ===
        MAX_WIN_WIDTH = 1280
        MAX_WIN_HEIGHT = 800

        scale = min(
            MAX_WIN_WIDTH / self.bg_width,
            MAX_WIN_HEIGHT / self.bg_height,
            1.0
        )

        self.display_width = int(self.bg_width * scale)
        self.display_height = int(self.bg_height * scale)

        self.background = cv2.resize(
            self.background,
            (self.display_width, self.display_height)
        )

        self.bg_width = self.display_width
        self.bg_height = self.display_height

        cv2.resizeWindow("Cosmetic Tutor", self.bg_width, self.bg_height)

        # Set window properties to handle close button
        cv2.setWindowProperty(
            "Cosmetic Tutor",
            cv2.WND_PROP_ASPECT_RATIO,
            cv2.WINDOW_FREERATIO
        )
        
        self.button_area_x = int(self.bg_width * 0.06)
        self.button_area_y = int(self.bg_height * 0.81)
        self.button_area_width = int(self.bg_width * 0.65)
        self.button_area_height = int(self.bg_height * 0.11)

        self.buttons = []
        self.active_mode = None

        self._init_buttons()

        cv2.setMouseCallback(
            "Cosmetic Tutor",
            self._handle_mouse
        )

    def _get_background_path(self) -> Path:
        current_file = Path(__file__)
        project_root = current_file.parent

        bg_jpg = project_root / "cosmetic_tutor" / "ui" / "background.jpg"
        bg_png = project_root / "cosmetic_tutor" / "ui" / "background.png"

        if bg_jpg.exists():
            return bg_jpg
        elif bg_png.exists():
            return bg_png
        else:
            raise FileNotFoundError(
                "No background.jpg or background.png found in ui folder"
            )

    def update_display(self, webcam_frame: np.ndarray, metrics=None, advice=None) -> np.ndarray:
        display_frame = self.background.copy()

        if webcam_frame is not None:
            h, w = webcam_frame.shape[:2]

            # === keep aspect ratio ===
            scale = min(
                self.frame_width / w,
                self.frame_height / h
            )*1.15

            new_w = int(w * scale)
            new_h = int(h * scale)

            resized_webcam = cv2.resize(webcam_frame, (new_w, new_h))

            # Center webcam inside white frame (with slight left-down offset)
            OFFSET_X = int(self.frame_width * 0.05)
            OFFSET_Y = int(self.frame_height * 0.09)

            x = self.frame_x + (self.frame_width - new_w) // 2 + OFFSET_X
            y = self.frame_y + (self.frame_height - new_h) // 2 + OFFSET_Y

            end_x = min(x + new_w, self.bg_width)
            end_y = min(y + new_h, self.bg_height)

            display_frame[y:end_y, x:end_x] = resized_webcam[: end_y - y, : end_x - x]

        self._draw_analysis_info(display_frame, metrics, advice)
        self._draw_buttons(display_frame)

        return display_frame

    def _draw_analysis_info(self, frame: np.ndarray, metrics=None, advice=None) -> None:
        text_x = self.text_col_x
        text_y = self.text_col_y

        if metrics is None or advice is None:
            instructions = [
                "=== COSMETIC TUTOR ===",
                "",
                "Position your face in the webcam",
                "and wait for analysis to start.",
                "",
                "Controls:",
                "[b] - Calibrate baseline",
                "[q] - Quit application",
                "[X] - Close window"
            ]

            self._draw_text_with_background(frame, instructions, text_x, text_y)
            return

        lines = [
            "=== BLUSH ANALYSIS ===",
            "",
            f"Left Cheek Delta: {metrics.left_delta_a:.2f}",
            f"Right Cheek Delta: {metrics.right_delta_a:.2f}",
            f"Asymmetry Score: {metrics.uneven_score:.2f}",
            "",
            f"Left Coverage: {metrics.left_coverage:.1%}",
            f"Right Coverage: {metrics.right_coverage:.1%}",
            f"Balance Direction: {metrics.offset_dir}",
            "",
            "=== RECOMMENDATION ===",
            "",
            f"{advice.text}",
            "",
            "=== CONTROLS ===",
            "",
            "[b] - Calibrate baseline",
            "[q] - Quit application",
            "[X] - Close window"
        ]

        self._draw_text_with_background(frame, lines, text_x, text_y)

    def _draw_text_with_background(self, frame: np.ndarray, lines: list, start_x: int, start_y: int) -> None:
        line_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            y_pos = start_y + i * line_height
            if y_pos + line_height > self.bg_height:
                break

            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            padding = 10

            bg_x1 = start_x - padding
            bg_y1 = y_pos - line_height + 5
            bg_x2 = start_x + text_size[0] + padding
            bg_y2 = y_pos + 5

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                (0, 0, 0),
                -1
            )

            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(
                frame,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                (255, 255, 255),
                2
            )

            color = (0, 255, 255) if line.startswith("===") else (255, 255, 255)
            cv2.putText(
                frame,
                line,
                (start_x, y_pos),
                font,
                font_scale,
                color,
                font_thickness
            )

    def _init_buttons(self):
        btn_w = int(self.button_area_width * 0.22)
        btn_h = int(self.button_area_height * 0.55)
        gap = int(self.button_area_width * 0.16)

        base_x = self.button_area_x + self.button_area_width
        y = self.button_area_y + (self.button_area_height - btn_h) // 2

        labels = ["Talk", "Close Auxilary line", "Compare"]
        actions = ["talk", "close", "compare"]

        self.buttons.clear()

        for i, (label, action) in enumerate(zip(labels, actions)):
            x = base_x - (i + 1) * btn_w - i * gap
            self.buttons.append({
                "label": label,
                "action": action,
                "rect": (x, y, btn_w, btn_h),
                "hover": False,
                "active": False
            })

    def _draw_buttons(self, frame: np.ndarray):
        for btn in self.buttons:
            x, y, w, h = btn["rect"]

            if btn["active"]:
                color = (190, 160, 255)
            elif btn["hover"]:
                color = (230, 215, 255)
            else:
                color = (255, 255, 255)
            roi = frame[y:y+h, x:x+w].copy()
            overlay = roi.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (w, h),
                color,
                -1,
                cv2.LINE_AA
            )

            cv2.addWeighted(overlay, 0.45, roi, 0.55, 0, roi)
            frame[y:y+h, x:x+w] = roi

            # ---- text ----
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(
                btn["label"],
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )[0]

            tx = x + (w - text_size[0]) // 2
            ty = y + (h + text_size[1]) // 2

            cv2.putText(
                frame,
                btn["label"],
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (60, 60, 60),
                thickness,
                cv2.LINE_AA
            )


    def show(self, frame: np.ndarray) -> None:
        cv2.imshow("Cosmetic Tutor", frame)

    def is_window_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(
                "Cosmetic Tutor",
                cv2.WND_PROP_VISIBLE
            ) < 1
        except cv2.error:
            return True

    def destroy(self) -> None:
        cv2.destroyWindow("Cosmetic Tutor")

    def _handle_mouse(self, event, x, y, flags, param):
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            inside = bx <= x <= bx + bw and by <= y <= by + bh

            btn["hover"] = inside

            if event == cv2.EVENT_LBUTTONDOWN and inside:
                self._on_button_click(btn["action"])

    def _on_button_click(self, action: str):
        for btn in self.buttons:
            btn["active"] = (btn["action"] == action)

        self.active_mode = action
        print(f"[UI] Mode switched to: {action}")


def main():
    print(sys.executable)
    print(sys.version)

    cfg = AppConfig()
    app = AppRunner(cfg)

    window_manager = BackgroundWindowManager()

    print("Keys: [b] baseline calibrate, [q] quit, or click X to close window")

    last_metrics = None

    while True:
        ret = app.step()
        if ret is None:
            continue

        frame, metrics, rois, advice = ret

        if metrics is not None:
            last_metrics = metrics

        display_frame = window_manager.update_display(frame, metrics, advice)
        window_manager.show(display_frame)

        if window_manager.is_window_closed():
            break

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        if key == ord('b') and last_metrics is not None:
            app.calibrate_baseline(
                last_metrics.left_mean_a,
                last_metrics.right_mean_a
            )

    app.release()
    window_manager.destroy()


if __name__ == "__main__":
    main()


#.\.venv\Scripts\python.exe .\run_webcam.py
