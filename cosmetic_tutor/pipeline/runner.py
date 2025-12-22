# runner.py
"""
Application pipeline runner

This module orchestrates the end-to-end runtime flow of the makeup AI system,
including image acquisition, face landmark detection, ROI construction,
blush scoring, advice generation, and visual overlay rendering.
"""

from dataclasses import dataclass
from cosmetic_tutor.config import AppConfig
from cosmetic_tutor.input.webcam import WebcamSource
from cosmetic_tutor.vision.face_mesh import FaceMeshDetector
from cosmetic_tutor.vision.roi_cheeks import CheekROIBuilders
from cosmetic_tutor.vision.scoring_blush import BlushScorer
from cosmetic_tutor.feedback import rule_based_advice
from cosmetic_tutor.ui.overlay import render_overlay


@dataclass
class RuntimeState:
    """Save state across frames (used for baseline calibration)"""
    baseline_left_a: float | None = None
    baseline_right_a: float | None = None


class AppRunner:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg                                  # configuration
        self.cam = WebcamSource(cfg.camera_index)       # input layer
        self.face = FaceMeshDetector()                  # vision layer
        self.roi_builder = CheekROIBuilders()            # geometry layer
        self.scorer = BlushScorer(cfg.thresholds)        # analysis layer
        self.state = RuntimeState()

    def calibrate_baseline(self, left_a: float, right_a: float):
        """Set baseline reference values"""
        self.state.baseline_left_a = left_a
        self.state.baseline_right_a = right_a

    def step(self):
        """
        Execute one pipeline step.
        Returns:
            (output_frame, metrics, (left_roi, right_roi), advice)
            or None if no frame is available
        """

        # 1) Read frame from webcam
        frame = self.cam.read()
        if frame is None:
            return None

        # 2) Detect facial landmarks
        landmarks = self.face.detect(frame.bgr)
        if landmarks is None:
            # Face not detected, return raw frame only
            return frame.bgr, None, None, None

        # 3) Construct cheek ROIs
        left_roi, right_roi = self.roi_builder.build(
            landmarks, frame.bgr.shape
        )

        # 4) Compute blush metrics
        metrics = self.scorer.score(
            frame.bgr,
            left_roi,
            right_roi,
            self.state.baseline_left_a,
            self.state.baseline_right_a,
        )

        # 5) Generate rule-based advice
        advice = rule_based_advice(metrics, self.cfg.thresholds)

        # 6) Render visual overlays
        output = render_overlay(
            frame.bgr,
            left_roi,
            right_roi,
            metrics,
            advice,
        )

        return output, metrics, (left_roi, right_roi), advice

    def release(self):
        """Release camera resources"""
        self.cam.release()
