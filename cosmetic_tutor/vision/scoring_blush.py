# makeup_ai/vision/scoring_blush.py
import numpy as np
import cv2
from cosmetic_tutor.types import ROI, BlushMetrics
from cosmetic_tutor.config import BlushThresholds
from cosmetic_tutor.vision.color_features import mean_lab_a, coverage_above_a
from cosmetic_tutor.utils.math2d import centroid_of_points

class BlushScorer:
    """
    Blush scoring module.

    This class converts pixel-level color features extracted from left/right cheek ROIs
    into a structured BlushMetrics result:
    - meanm redness (Lab a*)
    - baseline- aware redness change (delta a*)
    -left-right unevenness
    -coverage above redness threshold
    -course dirextion of blush offset (centroid-based heuristic)
    """

    def __init__(self, th: BlushThresholds):
        #store thresholds so that scoring behavior can be tuned without changeing logic
        self.th = th

    def score(
        self,
        bgr: np.ndarray,
        left_roi: ROI,
        right_roi: ROI,
        baseline_left_a: float | None,
        baseline_right_a: float | None
    ) -> BlushMetrics:
        """
        compute blush-related metrics for both cheeks and return a unified BlushMeturcs obj

        Args:
           bgr: input image in BGR color space
           left_roi/right_roi : ROI obfs containing polygon boundaries and binary masks
           baseline_left_a/right_a: Optional baseline mean a* values
                                    If None, the current value is used as baseline (delta becomes 0)

        Returns:
            BlushMetrics: Structured scoring output for downstream advice generation/UI.
        """

        #Mean a* (red-green axis) within each cheek ROI
        left_a = mean_lab_a(bgr, left_roi.mask)
        right_a = mean_lab_a(bgr, right_roi.mask)

        #Baseline-aware comparison:
        #if a baseline is provided, use it to compute delta; otherwise, delta is 0
        base_l = baseline_left_a if baseline_left_a is not None else left_a
        base_r = baseline_right_a if baseline_right_a is not None else right_a

        #delta reness (how much redder than baseline)
        dl = left_a - base_l
        dr = right_a - base_r

        #unevenness between cheeks
        uneven = abs(dl - dr)

        #coverage ratio : proportion of ROI pixels that exceed redness threshold
        left_cov = coverage_above_a(bgr, left_roi.mask, self.th.red_a_threshold)
        right_cov = coverage_above_a(bgr, right_roi.mask, self.th.red_a_threshold)

        #placement offset direction: compares high-red centroidvs ROI centroid
        (
            offset_dir,
            left_blush_centroid,
            right_blush_centroid,
            left_roi_centroid,
            right_roi_centroid,
            left_offset_vector,
            right_offset_vector,
        ) = self._offset_direction(bgr, left_roi, right_roi)

        #wrap up results to a metrics object
        return BlushMetrics(
            left_mean_a=left_a,
            right_mean_a=right_a,
            left_delta_a=dl,
            right_delta_a=dr,
            uneven_score=uneven,
            left_coverage=left_cov,
            right_coverage=right_cov,
            offset_dir=offset_dir,
            left_blush_centroid=left_blush_centroid,
            right_blush_centroid=right_blush_centroid,
            left_roi_centroid=left_roi_centroid,
            right_roi_centroid=right_roi_centroid,
            left_offset_vector=left_offset_vector,
            right_offset_vector=right_offset_vector,
            debug={} #placeholder for future debug info
        )

    def _offset_direction(self, bgr: np.ndarray, left_roi: ROI, right_roi: ROI) -> tuple[str, tuple[float, float] | None, tuple[float, float] | None, tuple[float, float], tuple[float, float], tuple[float, float] | None, tuple[float, float] | None]:
        """
        Heuristic offset detection using centroids.

        1.convert image to Lab and take the a* channel
        2.for each ROI, find pixels insied ROI whose a* exceeds a redness threshold.
        3.Compute the centroid of these "high-red" pixels
        4.Compare it to the ROI polygon centroid to get an offset vector
        5.Convert the offset into coarse directinal labels:
          OK / TOO_OUTSIDE / TOO_INSIDE / TOO_LOW / TOO_HIGH

        Note:
        - This is a lightweight heuristic (not a learned placement model).
        - Pixel thresholds and direction thresholds should be tuned using real data.
        """
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32) # a* in [0,255] under OpenCV Lab

        def roi_offset(roi: ROI) -> tuple[tuple[float, float] | None, tuple[float, float], tuple[float, float] | None]:
            #select pixels that are (1) insise ROI and (2) "red enough" to be considerded blush
            ys, xs = np.where((roi.mask == 255) & (a > self.th.red_a_threshold))
            
            # ROI centroid from polygon
            poly = roi.polygon_px.astype(np.float32)
            roi_cx = float(poly[:, 0].mean())
            roi_cy = float(poly[:, 1].mean())
            roi_centroid = (roi_cx, roi_cy)

            # too few pixels, treat as no blush / unstable
            if xs.size < 30:  
                return None, roi_centroid, None

            #convert pixel coords into an (N,2) point set for centroid computation
            pts = np.stack([xs, ys], axis=1)

            #centroid of high-red pixels
            blush_centroid = centroid_of_points(pts)
            if blush_centroid is None:
                return None, roi_centroid, None

            #return offset vector: positive x means to the right, positive y meanns downward
            offset_vector = (blush_centroid[0] - roi_cx, blush_centroid[1] - roi_cy)
            return blush_centroid, roi_centroid, offset_vector

        #compute offsets for left and right cheeks
        left_blush_centroid, left_roi_centroid, left_offset_vector = roi_offset(left_roi)
        right_blush_centroid, right_roi_centroid, right_offset_vector = roi_offset(right_roi)

        # decide direction: if both missing, OK
        if left_offset_vector is None and right_offset_vector is None:
            return "OK", left_blush_centroid, right_blush_centroid, left_roi_centroid, right_roi_centroid, left_offset_vector, right_offset_vector

        # average available offsets
        offsets = []
        if left_offset_vector is not None: 
            offsets.append(left_offset_vector)
        if right_offset_vector is not None: 
            offsets.append(right_offset_vector)

        ox = float(np.mean([o[0] for o in offsets]))
        oy = float(np.mean([o[1] for o in offsets]))

        #decision thresholds are in pixel units, tune with real data
        if abs(ox) < 8 and abs(oy) < 8:
            direction = "OK"

        #if horizontal offset domiates
        elif abs(ox) >= abs(oy):
            direction = "TOO_OUTSIDE" if ox > 0 else "TOO_INSIDE"

        #if vertical offset dominates
        else:
            direction = "TOO_LOW" if oy > 0 else "TOO_HIGH"
        
        return direction, left_blush_centroid, right_blush_centroid, left_roi_centroid, right_roi_centroid, left_offset_vector, right_offset_vector
