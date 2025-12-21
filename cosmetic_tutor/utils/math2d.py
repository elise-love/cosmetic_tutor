#math2d.py
import numpy as np

def centroid_of_points(points_xy: np.ndarray) -> tuple[float, float] | None:
    if points_xy.size ==0:
        return None
    cx = float(points_xy[:,0].mean())
    cy = float(points_xy[:,1].mean())
    return cx, cy #幾何中心（centroid）

"""
centroid_of_points
計算一組 2D 點的「中心點（centroid）」，也就是所有點的平均位置。

輸入：
一組 2D 點 (x, y)

輸出：
(cx, cy)：中心點座標
None：如果沒有點
"""

"""
ROI 中心點

👉 左臉頰中心
👉 右臉頰中心

用來：
比較左右位置是否偏移
判斷腮紅是不是太外 / 太低
"""