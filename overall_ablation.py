#!/usr/bin/env python3
"""
overall_ablation.py

Overall ablation evaluator for:

1. YOLOv8 Every Frame
2. Periodic YOLOv8
3. YOLOv8 + Kalman
4. YOLOv8 + KNN
5. YOLOv8 + KNN + Kalman

Metrics:
    - Mean IoU
    - Position Error (pixels)
    - Precision
    - Recall
    - TCR (%)
    - FPS

Ground-truth format:
    Standard YOLO labels:
        class_id x_center y_center width height

    where coordinates are normalized to [0, 1].

Important:
    This evaluator does NOT require persistent ground-truth object IDs.
    Therefore TCR is implemented as frame-level tracking coverage:
        matched GT instances / visible GT instances * 100

    This is not HOTA, IDF1, MOTA, or another identity-based MOT metric.

Dependencies:
    pip install ultralytics opencv-python numpy pandas scipy torch

Example:
    python overall_ablation.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# USER CONFIGURATION
# ============================================================

IMAGE_DIR = Path("/media/anurag/nas-anurag/Dataset/CarTrackingData/train/images")
LABEL_DIR = Path("/media/anurag/nas-anurag/Dataset/CarTrackingData/train/labels")
MODEL_PATH = Path("/home/anurag/yolo26/runs/detect/train12/weights/best.pt")

OUTPUT_CSV = Path("overall_ablation_results.csv")

# YOLO settings
YOLO_INTERVAL = 5
CONF_THRESHOLD = 0.25
YOLO_IMGSZ = 1280
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Evaluation
MATCH_IOU_THRESHOLD = 0.50

# KNN settings
KNN_HISTORY = 500
KNN_DIST2_THRESHOLD = 400.0
KNN_DETECT_SHADOWS = True
KNN_LEARNING_RATE = 0.0015
KNN_BINARY_THRESHOLD = 200
KNN_MIN_BLOB_AREA = 500

# Morphology: one erosion + one dilation, matching the paper description
MORPH_KERNEL_SIZE = 3

# Track association
TRACK_MAX_CENTER_DISTANCE = 150.0
TRACK_MAX_MISSED = 12

# Benchmark settings
WARMUP_GPU_FRAMES = 5

# If your dataset contains multiple independent sequences in one folder,
# evaluate each sequence separately or provide sequence-reset information.
# This script assumes the images in IMAGE_DIR form one sequential stream.


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h


class KalmanBoxTrack:
    """
    8-state constant-velocity Kalman filter:
        [cx, cy, vx, vy, w, h, vw, vh]

    Measurement:
        [cx, cy, w, h]
    """

    _next_id = 0

    def __init__(self, box: Box):
        self.id = KalmanBoxTrack._next_id
        KalmanBoxTrack._next_id += 1

        self.cls = int(box.cls)
        self.score = float(box.score)

        self.kf = cv2.KalmanFilter(8, 4)

        dt = 1.0

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0, 0, 0, 0],
            [0, 1, 0, dt, 0, 0, 0, 0],
            [0, 0, 1,  0, 0, 0, 0, 0],
            [0, 0, 0,  1, 0, 0, 0, 0],
            [0, 0, 0,  0, 1, 0, dt, 0],
            [0, 0, 0,  0, 0, 1, 0, dt],
            [0, 0, 0,  0, 0, 0, 1,  0],
            [0, 0, 0,  0, 0, 0, 0,  1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0
        self.kf.measurementMatrix[2, 4] = 1.0
        self.kf.measurementMatrix[3, 5] = 1.0

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        self.kf.statePost = np.array([
            [box.cx],
            [box.cy],
            [0.0],
            [0.0],
            [box.w],
            [box.h],
            [0.0],
            [0.0],
        ], dtype=np.float32)

        self.last_box = box
        self.missed = 0
        self.age = 1

    def predict(self) -> Box:
        state = self.kf.predict()

        cx = float(state[0, 0])
        cy = float(state[1, 0])
        w = max(2.0, float(state[4, 0]))
        h = max(2.0, float(state[5, 0]))

        self.last_box = center_size_to_box(
            self.cls, cx, cy, w, h, self.score
        )
        self.age += 1
        self.missed += 1
        return self.last_box

    def correct_box(self, box: Box) -> Box:
        measurement = np.array([
            [box.cx],
            [box.cy],
            [box.w],
            [box.h],
        ], dtype=np.float32)

        state = self.kf.correct(measurement)

        self.cls = int(box.cls)
        self.score = float(box.score)

        cx = float(state[0, 0])
        cy = float(state[1, 0])
        w = max(2.0, float(state[4, 0]))
        h = max(2.0, float(state[5, 0]))

        self.last_box = center_size_to_box(
            self.cls, cx, cy, w, h, self.score
        )
        self.missed = 0
        return self.last_box

    def correct_center_from_blob(self, blob_box: Box) -> Box:
        """
        KNN foreground blobs do not provide semantic class information.
        Preserve the track class and mainly use the blob for motion/location
        correction.

        Width/height are blended with the previous estimate rather than copied
        directly because foreground blobs may merge or fragment.
        """
        prev = self.last_box

        blended_w = 0.80 * prev.w + 0.20 * blob_box.w
        blended_h = 0.80 * prev.h + 0.20 * blob_box.h

        pseudo_measurement = Box(
            cls=self.cls,
            x1=blob_box.cx - blended_w / 2.0,
            y1=blob_box.cy - blended_h / 2.0,
            x2=blob_box.cx + blended_w / 2.0,
            y2=blob_box.cy + blended_h / 2.0,
            score=self.score,
        )

        return self.correct_box(pseudo_measurement)


class SimpleBoxTrack:
    """
    Non-Kalman track used by the YOLO + KNN configuration.
    """

    _next_id = 0

    def __init__(self, box: Box):
        self.id = SimpleBoxTrack._next_id
        SimpleBoxTrack._next_id += 1
        self.cls = int(box.cls)
        self.box = box
        self.missed = 0
        self.age = 1

    def update_yolo(self, box: Box):
        self.cls = int(box.cls)
        self.box = box
        self.missed = 0
        self.age += 1

    def update_blob(self, blob_box: Box):
        prev = self.box

        # Use the foreground centroid strongly but avoid replacing semantic
        # YOLO box dimensions completely with a possibly irregular blob.
        new_w = 0.80 * prev.w + 0.20 * blob_box.w
        new_h = 0.80 * prev.h + 0.20 * blob_box.h

        self.box = center_size_to_box(
            self.cls,
            blob_box.cx,
            blob_box.cy,
            new_w,
            new_h,
            prev.score,
        )

        self.missed = 0
        self.age += 1


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def natural_key(path: Path):
    import re
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def get_image_files(image_dir: Path) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(files, key=natural_key)


def center_size_to_box(
    cls: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    score: float = 1.0,
) -> Box:
    return Box(
        cls=int(cls),
        x1=float(cx - w / 2.0),
        y1=float(cy - h / 2.0),
        x2=float(cx + w / 2.0),
        y2=float(cy + h / 2.0),
        score=float(score),
    )


def clip_box(box: Box, width: int, height: int) -> Box:
    return Box(
        cls=box.cls,
        x1=max(0.0, min(float(width - 1), box.x1)),
        y1=max(0.0, min(float(height - 1), box.y1)),
        x2=max(0.0, min(float(width - 1), box.x2)),
        y2=max(0.0, min(float(height - 1), box.y2)),
        score=box.score,
    )


def box_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    union = a.area + b.area - intersection

    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def centroid_distance(a: Box, b: Box) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def read_yolo_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> List[Box]:

    if not label_path.exists():
        return []

    boxes = []

    with label_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(
                    f"Warning: invalid label at {label_path}:{line_number}: {line}"
                )
                continue

            cls_id = int(float(parts[0]))
            xc = float(parts[1]) * image_width
            yc = float(parts[2]) * image_height
            bw = float(parts[3]) * image_width
            bh = float(parts[4]) * image_height

            boxes.append(
                center_size_to_box(
                    cls=cls_id,
                    cx=xc,
                    cy=yc,
                    w=bw,
                    h=bh,
                    score=1.0,
                )
            )

    return boxes


def run_yolo(
    model: YOLO,
    frame: np.ndarray,
    device,
) -> List[Box]:

    result = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        imgsz=YOLO_IMGSZ,
        device=device,
        verbose=False,
    )[0]

    predictions: List[Box] = []

    if result.boxes is None:
        return predictions

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()

    for coords, cls_id, conf in zip(xyxy, classes, confidences):
        predictions.append(
            Box(
                cls=int(cls_id),
                x1=float(coords[0]),
                y1=float(coords[1]),
                x2=float(coords[2]),
                y2=float(coords[3]),
                score=float(conf),
            )
        )

    return predictions


# ============================================================
# KNN FOREGROUND PROCESSING
# ============================================================

class KNNForegroundPipeline:

    def __init__(self):
        self.knn = cv2.createBackgroundSubtractorKNN(
            history=KNN_HISTORY,
            dist2Threshold=KNN_DIST2_THRESHOLD,
            detectShadows=KNN_DETECT_SHADOWS,
        )

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        )

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Box]]:

        fgmask = self.knn.apply(
            frame,
            learningRate=KNN_LEARNING_RATE,
        )

        _, binary = cv2.threshold(
            fgmask,
            KNN_BINARY_THRESHOLD,
            255,
            cv2.THRESH_BINARY,
        )

        # One erosion followed by one dilation.
        binary = cv2.erode(binary, self.kernel, iterations=1)
        binary = cv2.dilate(binary, self.kernel, iterations=1)

        blobs = extract_foreground_blobs(binary)

        return binary, blobs


def extract_foreground_blobs(binary_mask: np.ndarray) -> List[Box]:

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    blobs: List[Box] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < KNN_MIN_BLOB_AREA:
            continue

        blobs.append(
            Box(
                cls=-1,
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h),
                score=1.0,
            )
        )

    return blobs


# ============================================================
# ASSOCIATION FUNCTIONS
# ============================================================

def solve_assignment(cost_matrix: np.ndarray):
    """
    Hungarian assignment if SciPy is available.
    Greedy fallback otherwise.
    """

    if cost_matrix.size == 0:
        return []

    if SCIPY_AVAILABLE:
        rows, cols = linear_sum_assignment(cost_matrix)
        return list(zip(rows.tolist(), cols.tolist()))

    pairs = []

    matrix = cost_matrix.copy()

    while matrix.size > 0:
        index = np.unravel_index(np.argmin(matrix), matrix.shape)
        r, c = int(index[0]), int(index[1])

        if not np.isfinite(matrix[r, c]):
            break

        pairs.append((r, c))

        matrix[r, :] = np.inf
        matrix[:, c] = np.inf

    return pairs


def associate_predictions_to_gt(
    predictions: List[Box],
    ground_truth: List[Box],
    iou_threshold: float,
):
    """
    Class-consistent prediction-to-ground-truth matching using IoU.

    Returns:
        matches: list of (pred_idx, gt_idx, iou)
        unmatched_pred
        unmatched_gt
    """

    if len(predictions) == 0:
        return [], [], list(range(len(ground_truth)))

    if len(ground_truth) == 0:
        return [], list(range(len(predictions))), []

    cost = np.full(
        (len(predictions), len(ground_truth)),
        fill_value=1e6,
        dtype=np.float32,
    )

    iou_matrix = np.zeros_like(cost)

    for p_idx, pred in enumerate(predictions):
        for g_idx, gt in enumerate(ground_truth):

            if pred.cls != gt.cls:
                continue

            iou = box_iou(pred, gt)
            iou_matrix[p_idx, g_idx] = iou

            # Hungarian solves a minimization problem.
            cost[p_idx, g_idx] = 1.0 - iou

    candidate_pairs = solve_assignment(cost)

    matches = []
    matched_pred = set()
    matched_gt = set()

    for p_idx, g_idx in candidate_pairs:

        if cost[p_idx, g_idx] >= 1e5:
            continue

        iou = float(iou_matrix[p_idx, g_idx])

        if iou < iou_threshold:
            continue

        matches.append((p_idx, g_idx, iou))
        matched_pred.add(p_idx)
        matched_gt.add(g_idx)

    unmatched_pred = [
        i for i in range(len(predictions))
        if i not in matched_pred
    ]

    unmatched_gt = [
        i for i in range(len(ground_truth))
        if i not in matched_gt
    ]

    return matches, unmatched_pred, unmatched_gt


def associate_yolo_to_tracks(
    track_boxes: List[Box],
    detections: List[Box],
):
    """
    Associate periodic YOLO detections to existing tracks.

    Cost combines class-consistent center distance and IoU.
    """

    if not track_boxes or not detections:
        return [], list(range(len(track_boxes))), list(range(len(detections)))

    cost = np.full(
        (len(track_boxes), len(detections)),
        1e6,
        dtype=np.float32,
    )

    for t_idx, track_box in enumerate(track_boxes):
        for d_idx, det in enumerate(detections):

            if track_box.cls != det.cls:
                continue

            dist = centroid_distance(track_box, det)
            iou = box_iou(track_box, det)

            if dist > TRACK_MAX_CENTER_DISTANCE and iou < 0.05:
                continue

            normalized_dist = min(
                dist / max(TRACK_MAX_CENTER_DISTANCE, 1.0),
                2.0,
            )

            cost[t_idx, d_idx] = (
                0.65 * normalized_dist +
                0.35 * (1.0 - iou)
            )

    pairs = solve_assignment(cost)

    matches = []
    used_tracks = set()
    used_dets = set()

    for t_idx, d_idx in pairs:
        if cost[t_idx, d_idx] >= 1e5:
            continue

        matches.append((t_idx, d_idx))
        used_tracks.add(t_idx)
        used_dets.add(d_idx)

    unmatched_tracks = [
        i for i in range(len(track_boxes))
        if i not in used_tracks
    ]

    unmatched_detections = [
        i for i in range(len(detections))
        if i not in used_dets
    ]

    return matches, unmatched_tracks, unmatched_detections


def associate_blobs_to_track_boxes(
    track_boxes: List[Box],
    blobs: List[Box],
):
    """
    Associate KNN foreground regions to tracks.

    KNN blobs have no semantic class IDs, so center distance and overlap
    are used.
    """

    if not track_boxes or not blobs:
        return [], list(range(len(track_boxes))), list(range(len(blobs)))

    cost = np.full(
        (len(track_boxes), len(blobs)),
        1e6,
        dtype=np.float32,
    )

    for t_idx, track_box in enumerate(track_boxes):
        for b_idx, blob in enumerate(blobs):

            dist = centroid_distance(track_box, blob)
            iou = box_iou(track_box, blob)

            if dist > TRACK_MAX_CENTER_DISTANCE and iou < 0.01:
                continue

            normalized_dist = min(
                dist / max(TRACK_MAX_CENTER_DISTANCE, 1.0),
                2.0,
            )

            cost[t_idx, b_idx] = (
                0.80 * normalized_dist +
                0.20 * (1.0 - iou)
            )

    pairs = solve_assignment(cost)

    matches = []
    used_tracks = set()
    used_blobs = set()

    for t_idx, b_idx in pairs:
        if cost[t_idx, b_idx] >= 1e5:
            continue

        matches.append((t_idx, b_idx))
        used_tracks.add(t_idx)
        used_blobs.add(b_idx)

    unmatched_tracks = [
        i for i in range(len(track_boxes))
        if i not in used_tracks
    ]

    unmatched_blobs = [
        i for i in range(len(blobs))
        if i not in used_blobs
    ]

    return matches, unmatched_tracks, unmatched_blobs


# ============================================================
# TRACKING CONFIGURATION IMPLEMENTATIONS
# ============================================================

def process_yolo_only(
    frame_idx: int,
    frame: np.ndarray,
    model: YOLO,
    interval: int,
    device,
) -> List[Box]:

    if frame_idx % interval == 0:
        return run_yolo(model, frame, device)

    # Important:
    # Do NOT reuse the previous YOLO boxes on skipped frames.
    return []


def process_kalman_configuration(
    frame_idx: int,
    frame: np.ndarray,
    model: YOLO,
    tracks: List[KalmanBoxTrack],
    interval: int,
    device,
) -> Tuple[List[Box], List[KalmanBoxTrack]]:

    predicted_boxes = []

    # Predict all existing tracks.
    if tracks:
        predicted_boxes = [track.predict() for track in tracks]

    if frame_idx % interval == 0:

        detections = run_yolo(model, frame, device)

        if not tracks:
            tracks = [KalmanBoxTrack(det) for det in detections]
            return [t.last_box for t in tracks], tracks

        matches, unmatched_tracks, unmatched_dets = associate_yolo_to_tracks(
            [t.last_box for t in tracks],
            detections,
        )

        for t_idx, d_idx in matches:
            tracks[t_idx].correct_box(detections[d_idx])

        for d_idx in unmatched_dets:
            tracks.append(KalmanBoxTrack(detections[d_idx]))

        # Unmatched existing tracks remain predictions.
        tracks = [
            t for t in tracks
            if t.missed <= TRACK_MAX_MISSED
        ]

    else:
        tracks = [
            t for t in tracks
            if t.missed <= TRACK_MAX_MISSED
        ]

    return [t.last_box for t in tracks], tracks


def process_knn_configuration(
    frame_idx: int,
    frame: np.ndarray,
    model: YOLO,
    tracks: List[SimpleBoxTrack],
    knn: KNNForegroundPipeline,
    interval: int,
    device,
) -> Tuple[List[Box], List[SimpleBoxTrack]]:

    _, blobs = knn.process(frame)

    # Use KNN blobs to update existing tracks each frame.
    if tracks:
        matches, unmatched_tracks, _ = associate_blobs_to_track_boxes(
            [t.box for t in tracks],
            blobs,
        )

        for t_idx, b_idx in matches:
            tracks[t_idx].update_blob(blobs[b_idx])

        for t_idx in unmatched_tracks:
            tracks[t_idx].missed += 1

    # Periodic semantic correction with YOLO.
    if frame_idx % interval == 0:
        detections = run_yolo(model, frame, device)

        if not tracks:
            tracks = [SimpleBoxTrack(det) for det in detections]

        else:
            matches, unmatched_tracks, unmatched_dets = associate_yolo_to_tracks(
                [t.box for t in tracks],
                detections,
            )

            for t_idx, d_idx in matches:
                tracks[t_idx].update_yolo(detections[d_idx])

            for d_idx in unmatched_dets:
                tracks.append(SimpleBoxTrack(detections[d_idx]))

    tracks = [
        t for t in tracks
        if t.missed <= TRACK_MAX_MISSED
    ]

    return [t.box for t in tracks], tracks


def process_knn_kalman_configuration(
    frame_idx: int,
    frame: np.ndarray,
    model: YOLO,
    tracks: List[KalmanBoxTrack],
    knn: KNNForegroundPipeline,
    interval: int,
    device,
) -> Tuple[List[Box], List[KalmanBoxTrack]]:

    # Kalman prediction.
    if tracks:
        for track in tracks:
            track.predict()

    # Continuous frame-level foreground observation.
    _, blobs = knn.process(frame)

    if tracks and blobs:
        matches, unmatched_tracks, _ = associate_blobs_to_track_boxes(
            [t.last_box for t in tracks],
            blobs,
        )

        for t_idx, b_idx in matches:
            tracks[t_idx].correct_center_from_blob(blobs[b_idx])

    # Periodic YOLO semantic correction.
    if frame_idx % interval == 0:

        detections = run_yolo(model, frame, device)

        if not tracks:
            tracks = [KalmanBoxTrack(det) for det in detections]

        else:
            matches, unmatched_tracks, unmatched_dets = associate_yolo_to_tracks(
                [t.last_box for t in tracks],
                detections,
            )

            for t_idx, d_idx in matches:
                tracks[t_idx].correct_box(detections[d_idx])

            for d_idx in unmatched_dets:
                tracks.append(KalmanBoxTrack(detections[d_idx]))

    tracks = [
        t for t in tracks
        if t.missed <= TRACK_MAX_MISSED
    ]

    return [t.last_box for t in tracks], tracks


# ============================================================
# METRIC ACCUMULATOR
# ============================================================

class MetricAccumulator:

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.matched_ious: List[float] = []
        self.position_errors: List[float] = []

        self.visible_gt_instances = 0
        self.covered_gt_instances = 0

        self.processing_times: List[float] = []

    def update(
        self,
        predictions: List[Box],
        ground_truth: List[Box],
        processing_time: float,
    ):

        matches, unmatched_pred, unmatched_gt = associate_predictions_to_gt(
            predictions,
            ground_truth,
            MATCH_IOU_THRESHOLD,
        )

        self.tp += len(matches)
        self.fp += len(unmatched_pred)
        self.fn += len(unmatched_gt)

        self.visible_gt_instances += len(ground_truth)
        self.covered_gt_instances += len(matches)

        for pred_idx, gt_idx, iou in matches:
            pred = predictions[pred_idx]
            gt = ground_truth[gt_idx]

            self.matched_ious.append(iou)
            self.position_errors.append(
                centroid_distance(pred, gt)
            )

        self.processing_times.append(processing_time)

    def results(self) -> Dict[str, float]:

        precision_denominator = self.tp + self.fp
        recall_denominator = self.tp + self.fn

        precision = (
            self.tp / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            self.tp / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        mean_iou = (
            float(np.mean(self.matched_ious))
            if self.matched_ious
            else 0.0
        )

        mean_position_error = (
            float(np.mean(self.position_errors))
            if self.position_errors
            else float("nan")
        )

        tcr = (
            100.0 * self.covered_gt_instances / self.visible_gt_instances
            if self.visible_gt_instances > 0
            else 0.0
        )

        total_time = sum(self.processing_times)

        fps = (
            len(self.processing_times) / total_time
            if total_time > 0.0
            else 0.0
        )

        return {
            "Mean IoU": mean_iou,
            "Position Error (px)": mean_position_error,
            "Precision": precision,
            "Recall": recall,
            "TCR (%)": tcr,
            "FPS": fps,
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
        }


# ============================================================
# BENCHMARK
# ============================================================

EXPERIMENTS = [
    {
        "name": "YOLOv8 Every Frame",
        "interval": 1,
        "use_knn": False,
        "use_kalman": False,
    },
    {
        "name": "Periodic YOLOv8",
        "interval": YOLO_INTERVAL,
        "use_knn": False,
        "use_kalman": False,
    },
    {
        "name": "YOLOv8 + Kalman",
        "interval": YOLO_INTERVAL,
        "use_knn": False,
        "use_kalman": True,
    },
    {
        "name": "YOLOv8 + KNN",
        "interval": YOLO_INTERVAL,
        "use_knn": True,
        "use_kalman": False,
    },
    {
        "name": "YOLOv8 + KNN + Kalman",
        "interval": YOLO_INTERVAL,
        "use_knn": True,
        "use_kalman": True,
    },
]


def synchronize_if_needed(device):
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()


def warmup_yolo(model: YOLO, image_files: List[Path], device):

    if not torch.cuda.is_available() or device == "cpu":
        return

    print("Warming up YOLO GPU inference...")

    count = min(WARMUP_GPU_FRAMES, len(image_files))

    for image_path in image_files[:count]:
        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        _ = run_yolo(model, frame, device)

    torch.cuda.synchronize()


def benchmark_experiment(
    experiment: dict,
    model: YOLO,
    image_files: List[Path],
) -> Dict[str, float]:

    name = experiment["name"]
    interval = int(experiment["interval"])
    use_knn = bool(experiment["use_knn"])
    use_kalman = bool(experiment["use_kalman"])

    print()
    print("=" * 72)
    print(f"Running: {name}")
    print(
        f"YOLO interval={interval}, "
        f"KNN={use_knn}, Kalman={use_kalman}"
    )
    print("=" * 72)

    # Reset IDs between experiments.
    KalmanBoxTrack._next_id = 0
    SimpleBoxTrack._next_id = 0

    knn = KNNForegroundPipeline() if use_knn else None

    kalman_tracks: List[KalmanBoxTrack] = []
    simple_tracks: List[SimpleBoxTrack] = []

    metrics = MetricAccumulator()

    for frame_idx, image_path in enumerate(image_files):

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Warning: unable to read {image_path}")
            continue

        height, width = frame.shape[:2]

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        gt_boxes = read_yolo_ground_truth(
            label_path,
            width,
            height,
        )

        synchronize_if_needed(YOLO_DEVICE)
        start = time.perf_counter()

        # ----------------------------------------------------
        # A1/A2: YOLO only
        # ----------------------------------------------------
        if not use_knn and not use_kalman:

            predictions = process_yolo_only(
                frame_idx,
                frame,
                model,
                interval,
                YOLO_DEVICE,
            )

        # ----------------------------------------------------
        # A3: YOLO + Kalman
        # ----------------------------------------------------
        elif not use_knn and use_kalman:

            predictions, kalman_tracks = process_kalman_configuration(
                frame_idx,
                frame,
                model,
                kalman_tracks,
                interval,
                YOLO_DEVICE,
            )

        # ----------------------------------------------------
        # A4: YOLO + KNN
        # ----------------------------------------------------
        elif use_knn and not use_kalman:

            predictions, simple_tracks = process_knn_configuration(
                frame_idx,
                frame,
                model,
                simple_tracks,
                knn,
                interval,
                YOLO_DEVICE,
            )

        # ----------------------------------------------------
        # A5: YOLO + KNN + Kalman
        # ----------------------------------------------------
        else:

            predictions, kalman_tracks = process_knn_kalman_configuration(
                frame_idx,
                frame,
                model,
                kalman_tracks,
                knn,
                interval,
                YOLO_DEVICE,
            )

        synchronize_if_needed(YOLO_DEVICE)
        elapsed = time.perf_counter() - start

        predictions = [
            clip_box(box, width, height)
            for box in predictions
            if box.w > 1 and box.h > 1
        ]

        metrics.update(
            predictions=predictions,
            ground_truth=gt_boxes,
            processing_time=elapsed,
        )

        if (frame_idx + 1) % 100 == 0:
            print(
                f"Processed {frame_idx + 1}/{len(image_files)} frames"
            )

    result = metrics.results()

    result.update({
        "Method": name,
        "YOLO Interval": interval,
        "KNN": "Yes" if use_knn else "No",
        "Kalman": "Yes" if use_kalman else "No",
    })

    print()
    print(f"Completed: {name}")
    print(f"Mean IoU           : {result['Mean IoU']:.4f}")
    print(f"Position Error (px): {result['Position Error (px)']:.3f}")
    print(f"Precision          : {result['Precision']:.4f}")
    print(f"Recall             : {result['Recall']:.4f}")
    print(f"TCR (%)            : {result['TCR (%)']:.2f}")
    print(f"FPS                : {result['FPS']:.2f}")

    return result


# ============================================================
# LATEX OUTPUT
# ============================================================

def latex_value(value, decimals=3):
    if value is None:
        return "--"

    try:
        if math.isnan(float(value)):
            return "--"
    except Exception:
        pass

    return f"{float(value):.{decimals}f}"


def print_latex_table(df: pd.DataFrame):

    print()
    print("=" * 72)
    print("LATEX TABLE")
    print("=" * 72)
    print()

    print(r"\begin{table*}[htbp]")
    print(r"\centering")
    print(r"\caption{Overall Ablation Results of the Proposed Vehicle Tracking Framework}")
    print(r"\label{tab:overall_ablation}")
    print(r"\renewcommand{\arraystretch}{1.2}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l c c c c c c c c c}")
    print(r"\hline")
    print(
        r"\textbf{Method} & "
        r"\textbf{YOLO Interval} & "
        r"\textbf{KNN} & "
        r"\textbf{Kalman} & "
        r"\textbf{Mean IoU $\uparrow$} & "
        r"\textbf{Position Error (px) $\downarrow$} & "
        r"\textbf{Precision $\uparrow$} & "
        r"\textbf{Recall $\uparrow$} & "
        r"\textbf{TCR (\%) $\uparrow$} & "
        r"\textbf{FPS $\uparrow$} \\"
    )
    print(r"\hline")

    for _, row in df.iterrows():

        method = str(row["Method"])

        if method == "YOLOv8 + KNN + Kalman":
            method_tex = r"\textbf{YOLOv8 + KNN + Kalman}"
            interval_tex = rf"\textbf{{{int(row['YOLO Interval'])}}}"
            knn_tex = r"\textbf{Yes}"
            kalman_tex = r"\textbf{Yes}"
            metric_wrapper = lambda x: rf"\textbf{{{x}}}"
        else:
            method_tex = method
            interval_tex = str(int(row["YOLO Interval"]))
            knn_tex = str(row["KNN"])
            kalman_tex = str(row["Kalman"])
            metric_wrapper = lambda x: x

        mean_iou = latex_value(row["Mean IoU"], 3)
        pos_error = latex_value(row["Position Error (px)"], 2)
        precision = latex_value(row["Precision"], 3)
        recall = latex_value(row["Recall"], 3)
        tcr = latex_value(row["TCR (%)"], 2)
        fps = latex_value(row["FPS"], 2)

        mean_iou = metric_wrapper(mean_iou)
        pos_error = metric_wrapper(pos_error)
        precision = metric_wrapper(precision)
        recall = metric_wrapper(recall)
        tcr = metric_wrapper(tcr)
        fps = metric_wrapper(fps)

        print(
            f"{method_tex} & "
            f"{interval_tex} & "
            f"{knn_tex} & "
            f"{kalman_tex} & "
            f"{mean_iou} & "
            f"{pos_error} & "
            f"{precision} & "
            f"{recall} & "
            f"{tcr} & "
            f"{fps} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")


# ============================================================
# VALIDATION
# ============================================================

def validate_configuration():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"IMAGE_DIR does not exist: {IMAGE_DIR.resolve()}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"LABEL_DIR does not exist: {LABEL_DIR.resolve()}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {MODEL_PATH.resolve()}"
        )

    image_files = get_image_files(IMAGE_DIR)

    if not image_files:
        raise RuntimeError(
            f"No images found in {IMAGE_DIR.resolve()}"
        )

    missing_labels = [
        image_path.name
        for image_path in image_files
        if not (LABEL_DIR / f"{image_path.stem}.txt").exists()
    ]

    if missing_labels:
        print(
            f"Warning: {len(missing_labels)} image(s) have no label file."
        )
        print(
            "They will be treated as frames containing zero ground-truth objects."
        )

    print(f"Images found : {len(image_files)}")
    print(f"Image folder : {IMAGE_DIR.resolve()}")
    print(f"Label folder : {LABEL_DIR.resolve()}")
    print(f"Model        : {MODEL_PATH.resolve()}")
    print(f"YOLO device  : {YOLO_DEVICE}")

    return image_files


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("Overall Vehicle Tracking Ablation Evaluation")
    print("=" * 72)

    image_files = validate_configuration()

    print()
    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    warmup_yolo(
        model,
        image_files,
        YOLO_DEVICE,
    )

    results = []

    for experiment in EXPERIMENTS:
        result = benchmark_experiment(
            experiment=experiment,
            model=model,
            image_files=image_files,
        )
        results.append(result)

    columns = [
        "Method",
        "YOLO Interval",
        "KNN",
        "Kalman",
        "Mean IoU",
        "Position Error (px)",
        "Precision",
        "Recall",
        "TCR (%)",
        "FPS",
        "TP",
        "FP",
        "FN",
    ]

    df = pd.DataFrame(results)[columns]

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)
    print(
        df[
            [
                "Method",
                "YOLO Interval",
                "KNN",
                "Kalman",
                "Mean IoU",
                "Position Error (px)",
                "Precision",
                "Recall",
                "TCR (%)",
                "FPS",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"CSV saved to: {OUTPUT_CSV.resolve()}")

    print_latex_table(df)


if __name__ == "__main__":
    main()
