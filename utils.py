# utils.py
import numpy as np

def bbox_center_xyxy(b):
    x1, y1, x2, y2 = map(float, b)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def bbox_area_xyxy(b):
    x1, y1, x2, y2 = map(float, b)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def clamp_bbox_xyxy(b, w, h):
    x1, y1, x2, y2 = map(float, b)
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

def adaptive_process_noise(box_h, h_ref=80.0, q_far=3e-4, q_near=8e-3):
    s = float(np.clip(box_h / h_ref, 0.3, 4.0))
    t = ((s - 0.3) / (4.0 - 0.3)) ** 2
    return float(q_far + (q_near - q_far) * t)