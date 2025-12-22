# roi_cheeks.py
"""
Cheek ROI (Region of Interest) construction module.

This module converts facial landmarks from MediaPipe Face Mesh
into polygon-based cheek regions suitable for color analysis.
Each ROI includes both polygon boundaries and binary masks.
"""

import numpy as np
import cv2
from cosmetic_tutor.types import ROI, FaceLandmarks

class CheekROIBuilders:
    """
    Builds left and right cheek ROIs from facial landmarks.
    
    Uses MediaPipe Face Mesh landmark indices to define approximate
    cheek regions, then converts normalized coordinates to pixel space
    and generates binary masks for color feature extraction.
    """
    
    def __init__(self):
        # MediaPipe Face Mesh landmark indices for left and right cheeks
        # These are approximate regions - may need tuning for optimal results
        self.left_cheek_indices = [
            116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 177, 215, 138, 135, 169, 170, 140, 171, 175
        ]
        
        self.right_cheek_indices = [
            345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 436, 416, 376, 401, 435, 367, 364, 394, 395, 369, 396, 400
        ]
    
    def build(self, landmarks: FaceLandmarks, image_shape: tuple) -> tuple[ROI, ROI]:
        """
        Build left and right cheek ROIs from facial landmarks.
        
        Args:
            landmarks: FaceLandmarks object with normalized coordinates [0,1]
            image_shape: (height, width, channels) of the source image
        
        Returns:
            Tuple of (left_roi, right_roi) ROI objects
        """
        h, w = image_shape[:2]
        
        # Convert normalized landmarks to pixel coordinates
        points_px = landmarks.points * [w, h]
        
        # Extract cheek polygons using predefined landmark indices
        left_polygon = points_px[self.left_cheek_indices].astype(np.int32)
        right_polygon = points_px[self.right_cheek_indices].astype(np.int32)
        
        # Create binary masks for each cheek region
        left_mask = self._create_mask(left_polygon, (h, w))
        right_mask = self._create_mask(right_polygon, (h, w))
        
        # Create ROI objects
        left_roi = ROI(
            name="left_cheek",
            polygon_px=left_polygon,
            mask=left_mask
        )
        
        right_roi = ROI(
            name="right_cheek",
            polygon_px=right_polygon,
            mask=right_mask
        )
        
        return left_roi, right_roi

    def _create_mask(self, polygon: np.ndarray, image_shape: tuple) -> np.ndarray:
        """
        Create a binary mask from a polygon.
        
        Args:
            polygon: Array of polygon vertices in pixel coordinates (N, 2)
            image_shape: (height, width) of the target mask
    
        Returns:
            Binary mask array where 255 = inside polygon, 0 = outside
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return mask