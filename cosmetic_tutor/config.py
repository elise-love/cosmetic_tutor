#config.py
from dataclasses import dataclass, field

@dataclass 
class BlushThresholds:
    #starting values
    strong_delta_a: float = 6.0 #delta threshold for "too strong"
    uneven_delta_a: float = 4.0 #abs(left_delta- right_delta)
    coverage_ratio: float = 0.88 #pixels above threshold / ROI area
    red_a_threshold: float = 145.0 #Lab a* threshold (0~255 in OpenCV lab)

@dataclass
class AppConfig:
    #index of the camera used by OpenCV
    #0 refers to default camera
    camera_index: int =0

    #target frames per second for video capture
    #control performance vs quality
    target_fps: int = 30

    thresholds: BlushThresholds = field(default_factory=BlushThresholds)
