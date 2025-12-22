# run_webcam.py
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
    def __init__(self, webcam_width: int = 640, webcam_height: int = 480):
        self.webcam_width = webcam_width
        self.webcam_height = webcam_height
        
        # Load background image
        background_path = self._get_background_path()
        self.background = cv2.imread(str(background_path))
        
        if self.background is None:
            raise FileNotFoundError(f"Could not load background image from {background_path}")
        
        # Get background dimensions
        self.bg_height, self.bg_width = self.background.shape[:2]
        
        # Calculate webcam position (top-left corner with some padding)
        self.webcam_x = 20
        self.webcam_y = 20
        
        # Create the main window
        cv2.namedWindow("Cosmetic Tutor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cosmetic Tutor", self.bg_width, self.bg_height)
    
    def _get_background_path(self) -> Path:
        current_file = Path(__file__)
        project_root = current_file.parent
        
        # Try both .jpg and .png extensions
        bg_jpg = project_root / "cosmetic_tutor" / "ui" / "background.jpg"
        bg_png = project_root / "cosmetic_tutor" / "ui" / "background.png"
        
        if bg_jpg.exists():
            return bg_jpg
        elif bg_png.exists():
            return bg_png
        else:
            raise FileNotFoundError("No background.jpg or background.png found in ui folder")
    
    def update_display(self, webcam_frame: np.ndarray) -> np.ndarray:
        # Start with background image
        display_frame = self.background.copy()
        
        # Resize webcam frame to fit in designated area
        if webcam_frame is not None:
            resized_webcam = cv2.resize(webcam_frame, (self.webcam_width, self.webcam_height))
            
            # Ensure webcam fits within background bounds
            end_x = min(self.webcam_x + self.webcam_width, self.bg_width)
            end_y = min(self.webcam_y + self.webcam_height, self.bg_height)
            
            actual_width = end_x - self.webcam_x
            actual_height = end_y - self.webcam_y
            
            if actual_width > 0 and actual_height > 0:
                if actual_width < self.webcam_width or actual_height < self.webcam_height:
                    resized_webcam = resized_webcam[:actual_height, :actual_width]
         
                # Overlay webcam frame onto background
                display_frame[self.webcam_y:end_y, self.webcam_x:end_x] = resized_webcam
        
        return display_frame
    
    def show(self, frame: np.ndarray) -> None:
        cv2.imshow("Cosmetic Tutor", frame)
    
    def destroy(self) -> None:
        cv2.destroyWindow("Cosmetic Tutor")

def main():
    print(sys.executable)
    print(sys.version)

    # Initialize global configuration
    cfg = AppConfig()

    # Create the main application runner (pipeline orchestrator)
    app = AppRunner(cfg)
    
    # Create the background window manager
    window_manager = BackgroundWindowManager()

    print("Keys: [b] baseline calibrate, [q] quit")

    # Store the most recent valid metrics for baseline calibration
    last_metrics = None

    while True:
        # Execute one step of the application pipeline
        ret = app.step()
        if ret is None:
            # No frame available; continue to next iteration
            continue

        frame, metrics, rois, advice = ret

        # Update last valid metrics if analysis succeeded
        if metrics is not None:
            last_metrics = metrics

        # Create the display frame with background and positioned webcam
        display_frame = window_manager.update_display(frame)
        
        # Display the output frame with background
        window_manager.show(display_frame)

        # Handle keyboard input (non-blocking)
        key = cv2.waitKey(1) & 0xFF

        # Quit application
        if key in (ord('q'), 27):  # 27 = ESC
            print("Exit requested")
            break

        # Baseline calibration using current mean a* values
        if key == ord('b') and last_metrics is not None:
            # Use mean Lab a* (not delta) as baseline reference
            app.calibrate_baseline(
                last_metrics.left_mean_a,
                last_metrics.right_mean_a
            )
            print("Baseline calibrated!")

    app.release()
    window_manager.destroy()

if __name__ == "__main__":
    main()


#.\.venv\Scripts\python.exe .\run_webcam.py
