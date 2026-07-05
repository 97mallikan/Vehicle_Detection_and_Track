import cv2
import numpy as np

class CudaFgPipeline:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def process(self, fgmask_cpu, stream=None):
        _, out = cv2.threshold(fgmask_cpu, 200, 255, cv2.THRESH_BINARY)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, self.kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, self.kernel)
        out = cv2.medianBlur(out, 5)
        return out