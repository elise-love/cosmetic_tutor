# run_webcam.py
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from cosmetic_tutor.config import AppConfig
from cosmetic_tutor.pipeline.runner import AppRunner

def main():
    print(sys.executable)
    print(sys.version)

    #initialize global configuration
    cfg = AppConfig()

    #create the main application runner (pipeline orchestrator)
    app = AppRunner(cfg)

    print("Keys: [b] baseline calibrate, [q] quit")

    #store the most recent valid metrics for baseline calibration
    last_metrics = None

    while True:
        #execute one step of the application pipeline
        ret = app.step()
        if ret is None:
            #no frame available; continue to next iteration
            continue

        frame, metrics, rois, advice = ret

        #update last valid metrics if analysis succeeded
        if metrics is not None:
            last_metrics = metrics

        #display the output frame with overlays
        cv2.imshow("Blush MVP", frame)

        #handle keyboard input (non-blocking)
        key = cv2.waitKey(1) & 0xFF

        # Quit application
        if key in (ord('q'), 27):  # 27 = ESC
            print("Exit requested")
            break

        #baseline calibration using current mean a* values
        if key == ord('b') and last_metrics is not None:
            #use mean Lab a* (not delta) as baseline reference
            app.calibrate_baseline(
                last_metrics.left_mean_a,
                last_metrics.right_mean_a
            )
            print("Baseline calibrated!")

    app.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


#.\.venv\Scripts\python.exe .\run_webcam.py
