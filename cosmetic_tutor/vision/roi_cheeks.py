#roi_cheeks.py
"""
Check ROI construction module.

This module constructs left and right regions of interest (ROI)
based on facial landmarks detected by FaceMesh.
The resulting ROIs are representes as both polygon coordinates and binary masks.
Enables subsequent color analysis and coverage computations.
"""

import numpy as np
import cv2
from cosmetic_tutor.types import ROI, FaceLandmarks

def _poly_from_indices(lms: FaceLandmarks, idxs: list[int], w: int, h: int) -> np.ndarray:
    pts_norm = lms.points[idxs]  # (M,2)
    #select landmark points by indexes

    pts_px = np.stack(
        [pts_norm[:, 0] * w, pts_norm[:, 1] * h],
        axis=1
    ).astype(np.int32)
    #convert normalized coordinates to pixel coordinates
    return pts_px

    """
    Convert selected facial landmark indices into a polygon in pixel coordinates.

    Args:
        lms (FaceLandmarks):
            Detected facial landmarks with normalized coordinates in [0, 1].

        idxs (list[int]):
            Indices of landmarks defining the target facial region (ROI).

        w (int):
            Width of the input image in pixels.

        h (int):
            Height of the input image in pixels.

    Returns:
        np.ndarray:
            An (M, 2) array of integer pixel coordinates representing
            the polygon boundary of the ROI.
    """

#從多邊形邊界產生二值 ROI 遮罩（mask）。
def _mask_from_poly(poly_px: np.ndarray, w: int, h: int) -> np.ndarray:
    #initialize an empyty binary mask
    mask = np.zeros((h, w), dtype=np.uint8)

    #fill polygon area with white (255)
    cv2.fillPoly(mask, [poly_px], 255)

    return mask

class CheekROIBuilders:
    """
    Build left/right cheek ROI masks from FaceMesh landmarks.
    Landmark indices are adjustable.
    """
    LEFT_CHEEK = [234, 93, 132, 58, 172, 136, 150, 176]
    RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 400]

    #create left/right cheek ROIs according to given landmark points and image shape
    def build(self, lms: FaceLandmarks, bgr_shape: tuple[int, int, int]) -> tuple[ROI, ROI]:
        #img height, width, channels
        h, w, _ = bgr_shape

        #convert landmark indices to pixel-space polygons
        left_poly = _poly_from_indices(lms, self.LEFT_CHEEK, w, h)
        right_poly = _poly_from_indices(lms, self.RIGHT_CHEEK, w, h)

        #generate binary masks for pixel-level analysis
        left_mask = _mask_from_poly(left_poly, w, h)
        right_mask = _mask_from_poly(right_poly, w, h)

        #wrap geometry and mask into standardized ROI objs
        left_roi = ROI(name="left_cheek", polygon_px=left_poly, mask=left_mask)
        right_roi = ROI(name="right_cheek", polygon_px=right_poly, mask=right_mask)
        return left_roi, right_roi

"""
ROI（Region of Interest）是影像中「你真正關心、要分析的那一塊區域」。
ROI refers to a selected subset of an image that is relevant for a specific analysis task, 
allowing computation to focus only on meaningful regions while ignoring irrelevant background information.
"""

"""
Pixel-wise (pixel-level) 
means performing operations independently 
on each individual pixel in an image.

An image is a grid of tiny squares (pixels).
Pixel-level → look at pixels one by one
Region-level → look at a whole area at once
Image-level → treat the entire image as one object
Pixel-wise means:
“For every pixel, do something.”
"""

"""
polygon_px=left_poly
👉 左臉頰的幾何邊界
是一個 (M, 2) 的 numpy array
每一點是 pixel 座標 (x, y)
代表臉頰的外框多邊形

用途：
畫在畫面上 debug
理解 ROI 的幾何形狀
方便視覺化
"""

"""
mask=left_mask
👉 左臉頰的像素級遮罩（binary mask）
shape = (H, W)
255 = 在左臉頰裡
0 = 不在

用途：
算平均顏色
算覆蓋率
快速 pixel-wise 運算
"""