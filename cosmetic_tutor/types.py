#types.py
from __future__ import annotations
from dataclasses import dataclass
from sys import float_info
from time import strftime
from typing import Optional, Tuple, Dict, Any
import numpy as np

Point2D = Tuple[float, float] #normalized (0~1) or pixel

@dataclass
class Frame:
    bgr: np.ndarray #(H,W,3) uint8
    timestamp_s:float
    fps: Optional[float] = None

@dataclass
class FaceLandmarks:
    #normaliized coordinates in [0,1]
    points: np.ndarray #shape: (N,2) float32
    confidence: float = 1.0

@dataclass
class ROI:
    name: str
    polygon_px: np.ndarray #shape (M,2) int32 in pixel coordinates
    mask: np.ndarray #shape (H,W) init 8 {0,225}

@dataclass
class BlushMetrics:
    #baseline-aware metrics

    #left and right refer to face sides
    left_mean_a: float
    right_mean_a: float

    #compare with baseline
    left_delta_a: float
    right_delta_a: float

    #§¡¤Ã«×
    uneven_score: float

    #coverage ratio
    left_coverage: float
    right_coverage: float

    #°¾²¾¤è¦V
    offset_dir: str
    debug: Dict[str, Any]

@dataclass
class Advice:
    text: str
    tags: Tuple[str, ...]= ()
