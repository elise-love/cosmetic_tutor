#runner.py
"""
Application pipeline runner,

This module orchestarates the end-to-end runtime flow of the makeup AI system.
including image aquisition, face landmark detection, ROI construction,
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
#save state across frames (used to save baseline)
class RuntimeState:
    baseline_left_a: float | None = None
    baseline_right_a: float | None = None

class AppRunner:
    def __init__(self, cfg:AppConfig):
        self.cfg = cfg #¥þ°ì³]©w
        self.cam = WebcamSource(cfg.camera_index) #input layer
        self.face = FaceMeshDetector() #vision layer
        self.roi_builder = CheekROIBuilders() #geometry layer(ROI build)
        self.scorer = BlushScorer(cfg.thresholds) #analysis layer
        self.state = RuntimeState()

    #set baseline reference
    def calibrate_baseline(self, left_a: float, right_a: float):
        self.state.baseline_left_a = left_a
        self.state.baseline_right_a = right_a

    def step(self):

        # 1)read frame from webcam
        frame = self.cam.read()
        if frame is None:
            #no frame availible
            return None

        # 2)dectect facial landmarks from the current frame
        lms = self.face.detect(frame.bgr)
        #face not detected
        if lms is None:
            return frame.bgr, None, None, None

        # 3)construct left and right cheeks ROIs using face landmarks
        left_roi, right_roi = self.roi_builder.build(lms, frame.bgr.shape)
        
        # 4)compute blush metrics using color features and baseline values
        metrics = self.scorer.score(
            frame.bgr,
            left_roi,
            right_roi,
            self.state.baseline_left_a,
            self.state.baseline_right_a
        )

        # 5)generate rule-based advice from computed metrics
        advice = rule_based_advice(metrics, self.cfg.thresholds)

        # 6)render visual overlays (ROIs, metrics, advice) on the image
        out = render_overlay(frame.bgr, left_roi, right_roi, metrics, advice)
        return out, metrics, (left_roi, right_roi), advice

    def release(self):
        self.cam.release()