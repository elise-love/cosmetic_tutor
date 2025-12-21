# makeup_ai/ui/overlay.py
"""
Visualization utilities for rendering blush analysis results.

This module draws facial ROIs, numerical metrics, and textual advice
as overlays on the input image, providing real-time visual feedback
for users and developers.
"""

import cv2
import numpy as np
from cosmetic_tutor.types import ROI, BlushMetrics, Advice

#把 ROI 的多邊形邊界畫在影像上。
def draw_roi(bgr: np.ndarray, roi: ROI) -> None:
    cv2.polylines(bgr, [roi.polygon_px], isClosed=True, color=(0, 255, 0), thickness=2)

#在影像左上角逐行畫出文字資訊。
def draw_text(bgr: np.ndarray, lines: list[str], x: int = 10, y: int = 30) -> None:
    for i, s in enumerate(lines):
        cv2.putText(bgr, 
                    s, 
                    (x, y + i * 24), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255,255,255), 
                    2
                  )

#整合所有視覺元素，產生最終可顯示的影像。
def render_overlay(bgr: np.ndarray, left_roi: ROI, right_roi: ROI, m: BlushMetrics, advice: Advice) -> np.ndarray:
    out = bgr.copy()
    draw_roi(out, left_roi)
    draw_roi(out, right_roi)

    lines = [
        f"delta L={m.left_delta_a:.1f} R={m.right_delta_a:.1f} uneven={m.uneven_score:.1f}",
        f"cov   L={m.left_coverage:.2f} R={m.right_coverage:.2f} offset={m.offset_dir}",
        f"advice: {advice.text}"
    ]
    draw_text(out, lines)
    return out
