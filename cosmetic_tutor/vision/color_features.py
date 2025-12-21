#color_features.py
"""
This module extracts color-based features from a facial ROI using pixel-level analysis in Lab color space
Lab color space separates color information form brightness, making it more suitable for cosmetic color analysis.
"""
import cv2
import numpy as np

def mean_lab_a(bgr: np.ndarray, mask: np.ndarray) -> float: #compute the mean Lab a* value within a masked facial region
    #convert image from BGR to Lab color space
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    #extract the a* channel(red-green axis), range[0,255]
    a = lab[:, :, 1].astype(np.float32)

    #select pixel values inside the ROI using the binary mask
    vals = a[mask == 255]

    #handle empty ROI
    if vals.size == 0:
        return 0.0

    #return mean a* value as a scalar feature (comparing with baseline)
    return float(vals.mean())

#空間分佈（coverage）指標
def coverage_above_a(bgr: np.ndarray, mask: np.ndarray, a_threshold:float)->float:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    #extract a* channel
    a = lab[:, :, 1].astype(np.float32)

    #select a* values inside the ROI
    roi = a[mask ==255]

    if roi.size == 0:
        return 0.0

    #computr ratio of  pixels exceeding the red intesity threshold
    return float((roi >a_threshold).mean())


"""
Lab 是一種色彩空間（color space），由三個通道組成：
L：亮度（受光線影響大）
a：紅 ↔ 綠（腮紅關鍵）
b：黃 ↔ 藍
"""

"""
a* 是什麼？

數值高 → 越偏紅
數值低 → 越偏綠

在 OpenCV 裡：
a* ∈ [0, 255]
（128 左右是中性）
"""

"""
Mean（平均）

不是看單一 pixel，而是：
把 ROI 裡所有 pixel 的 a* 值取平均
這代表：
整體紅度趨勢
不被單一雜訊 pixel 影響



Mean Lab a* =
「這一塊臉整體看起來有多紅」
"""