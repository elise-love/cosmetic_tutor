#face_mesh.py
import numpy as np
import cv2
import mediapipe as mp
from cosmetic_tutor.types import FaceLandmarks

# Mediapipe Face Mesh detector
class FaceMeshDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh

        self.mesh = self.mp_face.FaceMesh(
            static_image_mode=False, #enables temporal tracking for video streams
            max_num_faces = 1, #limit to one face
            refine_landmarks=True, #enables refined landmarks around eyes and lips
            
            #confidence thresholds
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )
    
    def detect (self, bgr: np.ndarray) -> FaceLandmarks | None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x, p.y) for p in lm], dtype=np.float32) #normalize
        return FaceLandmarks(points=pts, confidence=1.0)


"""
MediaPipe is a real-time, 
cross-platform framework by Google for building perception pipelines such as face, hand, and pose detection.

What MediaPipe actually gives you
For Face Mesh specifically:
A pre-trained neural network
Detects 468+ facial landmarks
Runs in real time (webcam friendly)
Works on CPU (no GPU required)

This is why it’s very common in:
AR filters
virtual makeup
face alignment
emotion / expression analysis
"""

"""
What is Face Mesh in MediaPipe?

Face Mesh is a MediaPipe solution that:
detects dense facial landmarks
gives semantic consistency
cheeks
lips
eyes
nose
jawline

Each landmark is:
(x, y, z)  normalized

You are using:
(x, y)
because your task is 2D makeup analysis, not 3D reconstruction.
"""

"""
refine_landmarks=True

Means:
Slightly higher computation
Much higher spatial precision

For:
single face
real-time webcam
"""