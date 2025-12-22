#!/usr/bin/env python3
"""
Diagnostic script to test blush detection components individually
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from cosmetic_tutor.config import AppConfig
from cosmetic_tutor.input.webcam import WebcamSource
from cosmetic_tutor.vision.face_mesh import FaceMeshDetector
from cosmetic_tutor.vision.roi_cheeks import CheekROIBuilders
from cosmetic_tutor.vision.scoring_blush import BlushScorer
from cosmetic_tutor.feedback import rule_based_advice


def test_camera():
    """Test if camera is working"""
    print("=== Testing Camera ===")
    try:
        cam = WebcamSource(0)
        frame = cam.read()

        if frame is None:
            print("❌ Camera not working - no frame returned")
            cam.release()
            return False
        else:
            print(f"✅ Camera working - got frame {frame.bgr.shape}")

        cam.release()
        return True

    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False


def test_face_detection():
    """Test if face detection is working"""
    print("\n=== Testing Face Detection ===")

    try:
        face_detector = FaceMeshDetector()
        cam = WebcamSource(0)

        print("Testing face detection for 10 seconds...")
        print("Please make sure your face is visible in the camera")

        face_detected = False

        for i in range(100):  # ~10 seconds @ 10fps
            frame = cam.read()
            if frame is None:
                continue

            landmarks = face_detector.detect(frame.bgr)
            if landmarks is not None:
                print(f"✅ Face detected! Got {len(landmarks.points)} landmarks")
                face_detected = True
                break

            if i % 10 == 0:
                print(f"Trying... {i // 10 + 1}/10")

        cam.release()

        if not face_detected:
            print("❌ No face detected in 10 seconds")
            print("Make sure:")
            print("  - Your face is clearly visible")
            print("  - There's good lighting")
            print("  - Camera is not blocked")
            return False

        return True

    except Exception as e:
        print(f"❌ Face detection error: {e}")
        return False


def test_roi_building():
    """Test if ROI building works with detected face"""
    print("\n=== Testing ROI Building ===")

    try:
        face_detector = FaceMeshDetector()
        roi_builder = CheekROIBuilders()
        cam = WebcamSource(0)

        print("Looking for face to build ROIs...")

        for _ in range(50):
            frame = cam.read()
            if frame is None:
                continue

            landmarks = face_detector.detect(frame.bgr)
            if landmarks is None:
                continue

            left_roi, right_roi = roi_builder.build(
                landmarks, frame.bgr.shape
            )

            print("✅ ROIs built successfully!")
            print(f"Left ROI points: {left_roi.polygon_px.shape[0]}")
            print(f"Right ROI points: {right_roi.polygon_px.shape[0]}")

            # Visualize ROIs
            debug_frame = frame.bgr.copy()
            cv2.polylines(
                debug_frame, [left_roi.polygon_px], True, (0, 255, 0), 2
            )
            cv2.polylines(
                debug_frame, [right_roi.polygon_px], True, (0, 255, 0), 2
            )

            cv2.putText(
                debug_frame,
                "Left",
                tuple(left_roi.polygon_px[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                debug_frame,
                "Right",
                tuple(right_roi.polygon_px[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("ROI Debug", debug_frame)
            cv2.waitKey(3000)
            cv2.destroyWindow("ROI Debug")

            cam.release()
            return True

        cam.release()
        print("❌ Could not build ROIs - no face detected")
        return False

    except Exception as e:
        print(f"❌ ROI building error: {e}")
        return False


def test_blush_scoring():
    """Test blush scoring system"""
    print("\n=== Testing Blush Scoring ===")

    try:
        cfg = AppConfig()
        face_detector = FaceMeshDetector()
        roi_builder = CheekROIBuilders()
        scorer = BlushScorer(cfg.thresholds)
        cam = WebcamSource(0)

        print("Testing blush scoring...")
        print("Move your face around to see different scores")

        for i in range(30):
            frame = cam.read()
            if frame is None:
                continue

            landmarks = face_detector.detect(frame.bgr)
            if landmarks is None:
                continue

            left_roi, right_roi = roi_builder.build(
                landmarks, frame.bgr.shape
            )

            metrics = scorer.score(
                frame.bgr, left_roi, right_roi, None, None
            )

            print(
                f"Frame {i}: "
                f"L_mean={metrics.left_mean_a:.1f}, "
                f"R_mean={metrics.right_mean_a:.1f}, "
                f"L_cov={metrics.left_coverage:.2f}, "
                f"R_cov={metrics.right_coverage:.2f}"
            )

            advice = rule_based_advice(metrics, cfg.thresholds)
            print(f"Advice: {advice.text}")

            if i == 0:
                print("✅ Blush scoring working!")

        cam.release()
        return True

    except Exception as e:
        print(f"❌ Blush scoring error: {e}")
        return False


def main():
    print("Blush Detection Diagnostic Tool")
    print("=" * 40)

    camera_ok = test_camera()
    if not camera_ok:
        print("\n❌ Camera test failed. Fix camera issues first.")
        return

    face_ok = test_face_detection()
    if not face_ok:
        print("\n❌ Face detection failed. Check lighting and positioning.")
        return

    roi_ok = test_roi_building()
    if not roi_ok:
        print("\n❌ ROI building failed.")
        return

    scoring_ok = test_blush_scoring()
    if not scoring_ok:
        print("\n❌ Blush scoring failed.")
        return

    print("\n🎉 All tests passed! Blush detection should be working.")
    print("\nIf the main app still doesn't work, the issue might be:")
    print("  - Lighting conditions")
    print("  - Camera permissions")
    print("  - Threshold settings too strict")
    print("  - Face positioning")


if __name__ == "__main__":
    main()
