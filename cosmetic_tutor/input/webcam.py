#webcam.py

import time
import cv2
from cosmetic_tutor.types import Frame

class WebcamSource:
    def __init__(self, camera_index: int = 0): #camera_index 0: default camera
        self.cap = cv2.VideoCapture(camera_index) #create OpenCV camera obj
    
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webam")

        self._last_t = None #remembers when the previous frame arrived
        self._fps = None #stores the current estimated FPS, based on how fast frames are coming in

    #try to read a frame from the webcam every time this is called
    def read(self) -> Frame | None:
        ok, bgr = self.cap.read()
        if not ok:
            return None
        t = time.time()

        #estimate FPS based on time since last frame
        if self._last_t is not None:
            dt = max(1e-6, t-self._last_t)
            self._fps = 1.0 / dt

        self._last_t = t #update last franme time
        return Frame(bgr=bgr, timestamp_s=t, fps=self._fps)

    def release(self):
        self.cap.release()

"""
bgr 是一張彩色影像，用 B-G-R（藍、綠、紅）順序存的像素資料。
OpenCV 預設色彩順序是 BGR
每個值是 0 ~ 255
對應：藍、綠、紅的亮度


fps（frames per second）表示「系統每秒實際處理幾張影像」。
fps = 30 👉 一秒鐘跑 30 張畫面
fps = 12 👉 系統變慢、卡卡的
反映「現在這台電腦 + 現在這個負載」的真實速度
target_fps	你「希望」跑多快
fps	你「實際」跑多快

為什麼要存 fps？
因為在即時 AI 系統中：FPS 會影響
延遲（latency）
使用者體驗
模型是否該降頻 / 跳幀
是效能監控的基本指標
"""