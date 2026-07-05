import numpy as np
import cv2
from collections import deque

from utils import bbox_center_xyxy, adaptive_process_noise
from config import VEL_DAMP, R_YOLO, R_BLOB, Q_SCALE


class KalmanTrack:
    """
    Scale-aware Kalman tracker.

    State:
        [cx, cy, vx, vy, w, h, vw, vh]^T

    Measurement:
        [cx, cy, w, h]^T
    """

    def __init__(self, track_id, bbox_xyxy, cls_id, conf, frame_idx):
        self.id = int(track_id)
        self.cls_id = int(cls_id)
        self.conf = float(conf)

        x1, y1, x2, y2 = map(float, bbox_xyxy)
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        cx, cy = bbox_center_xyxy(bbox_xyxy)

        self.vx_f = 0.0
        self.vy_f = 0.0

        self.prev_meas = (cx, cy)

        self.hits = 1
        self.misses = 0
        self.yolo_misses = 0
        self.pred_only_streak = 0
        self.last_seen = int(frame_idx)

        self.kf = cv2.KalmanFilter(8, 4)
        dt = 1.0

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0,  0,  0,  0,  0],
            [0, 1, 0, dt,  0,  0,  0,  0],
            [0, 0, 1,  0,  0,  0,  0,  0],
            [0, 0, 0,  1,  0,  0,  0,  0],
            [0, 0, 0,  0,  1,  0, dt,  0],
            [0, 0, 0,  0,  0,  1,  0, dt],
            [0, 0, 0,  0,  0,  0,  1,  0],
            [0, 0, 0,  0,  0,  0,  0,  1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # cx
            [0, 1, 0, 0, 0, 0, 0, 0],  # cy
            [0, 0, 0, 0, 1, 0, 0, 0],  # w
            [0, 0, 0, 0, 0, 1, 0, 0],  # h
        ], dtype=np.float32)

        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        self.kf.measurementNoiseCov = np.diag([
            R_YOLO, R_YOLO,
            12.0, 12.0
        ]).astype(np.float32)

        q = Q_SCALE * adaptive_process_noise(h)
        Q = np.eye(8, dtype=np.float32) * q
        Q[2, 2] *= 0.15
        Q[3, 3] *= 0.15
        Q[6, 6] *= 0.25
        Q[7, 7] *= 0.25
        Q[4, 4] *= 0.20
        Q[5, 5] *= 0.20
        self.kf.processNoiseCov = Q

        self.kf.statePost = np.array([
            [cx], [cy], [0.0], [0.0],
            [w],  [h],  [0.0], [0.0]
        ], dtype=np.float32)

        self.last_pred = np.array([cx, cy], dtype=np.float32)

        # Only for color-change message
        self.center_history = deque(maxlen=40)
        self.vec_history = deque(maxlen=40)
        self.speed_history = deque(maxlen=40)
        self.center_history.append((float(cx), float(cy)))

        self.prev_dir_bin = None          # stable confirmed color/bin
        self.candidate_dir_bin = None     # possible new color/bin
        self.candidate_count = 0          # how many consecutive frames new color stayed
        self.message_hold_count = 0       # keep message visible for N frames

        # Speed trend message state
        self.speed_message = ""
        self.speed_hold_count = 0

        self.speed_history = deque(maxlen=20)

        self.motion_state = "unknown"
        self.motion_msg = ""
        self.motion_msg_hold = 0

        self.age = 0
        self.birth_frame = int(frame_idx)
        self.death_frame = None
        

    @property
    def w(self):
        return float(max(2.0, self.kf.statePost[4, 0]))

    @w.setter
    def w(self, value):
        self.kf.statePost[4, 0] = float(max(2.0, value))

    @property
    def h(self):
        return float(max(2.0, self.kf.statePost[5, 0]))

    @h.setter
    def h(self, value):
        self.kf.statePost[5, 0] = float(max(2.0, value))

    def _smooth_velocity(self):
        vx = float(self.kf.statePost[2, 0])
        vy = float(self.kf.statePost[3, 0])
        beta = 0.12
        self.vx_f = (1.0 - beta) * self.vx_f + beta * vx
        self.vy_f = (1.0 - beta) * self.vy_f + beta * vy

    def _clamp_size_state(self):
        self.kf.statePost[4, 0] = max(2.0, float(self.kf.statePost[4, 0]))
        self.kf.statePost[5, 0] = max(2.0, float(self.kf.statePost[5, 0]))
        self.kf.statePost[6, 0] = float(np.clip(self.kf.statePost[6, 0], -80.0, 80.0))
        self.kf.statePost[7, 0] = float(np.clip(self.kf.statePost[7, 0], -80.0, 80.0))

    def predict_only(self):
        pred = self.kf.predict()

        self.kf.statePost[2, 0] *= VEL_DAMP
        self.kf.statePost[3, 0] *= VEL_DAMP
        self.kf.statePost[6, 0] *= 0.92
        self.kf.statePost[7, 0] *= 0.92

        self._clamp_size_state()

        px = float(pred[0, 0])
        py = float(pred[1, 0])
        self.last_pred = np.array([px, py], dtype=np.float32)
        return px, py

    def correct_center(self, cx, cy, frame_idx, conf=0.0):
        old_R = self.kf.measurementNoiseCov.copy()
        self.kf.measurementNoiseCov = np.diag([
            R_BLOB, R_BLOB,
            1e6, 1e6
        ]).astype(np.float32)

        meas = np.array([
            [float(cx)],
            [float(cy)],
            [float(self.w)],
            [float(self.h)],
        ], dtype=np.float32)

        self.kf.correct(meas)
        self.kf.measurementNoiseCov = old_R

        self._clamp_size_state()

        self.prev_meas = (cx, cy)
        self.last_pred = np.array([
            float(self.kf.statePost[0, 0]),
            float(self.kf.statePost[1, 0])
        ], dtype=np.float32)
        self.last_seen = int(frame_idx)
        self.conf = float(conf)
        self.pred_only_streak = 0
        self._smooth_velocity()

    def update(self, bbox_xyxy, conf, frame_idx, update_size=True):
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        w_det = max(2.0, x2 - x1)
        h_det = max(2.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        old_R = self.kf.measurementNoiseCov.copy()

        if update_size and conf >= 0.20:
            size_noise = 6.0 if (h_det > 180 or w_det > 220) else 12.0
            self.kf.measurementNoiseCov = np.diag([
                R_YOLO, R_YOLO,
                size_noise, size_noise
            ]).astype(np.float32)

            meas = np.array([
                [float(cx)],
                [float(cy)],
                [float(w_det)],
                [float(h_det)],
            ], dtype=np.float32)
        else:
            self.kf.measurementNoiseCov = np.diag([
                R_YOLO, R_YOLO,
                1e6, 1e6
            ]).astype(np.float32)

            meas = np.array([
                [float(cx)],
                [float(cy)],
                [float(self.w)],
                [float(self.h)],
            ], dtype=np.float32)

        self.kf.correct(meas)
        self.kf.measurementNoiseCov = old_R

        self._clamp_size_state()

        self.prev_meas = (cx, cy)
        self.last_pred = np.array([
            float(self.kf.statePost[0, 0]),
            float(self.kf.statePost[1, 0])
        ], dtype=np.float32)
        self.conf = float(conf)
        self.last_seen = int(frame_idx)
        self.hits += 1
        self.misses = 0
        self.yolo_misses = 0
        self.pred_only_streak = 0
        self._smooth_velocity()

    def mark_pred_only(self):
        self.pred_only_streak += 1

    def mark_yolo_missed(self):
        self.yolo_misses += 1
        self.pred_only_streak += 1

    def draw_point(self):
        return float(self.last_pred[0]), float(self.last_pred[1])

    def bbox_from_center(self, cx, cy):
        w = self.w
        h = self.h
        x1 = cx - w / 2.0
        x2 = cx + w / 2.0
        y1 = cy - h / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def future_path(self, n_steps=25, dt=1.0):
        cx = float(self.kf.statePost[0, 0])
        cy = float(self.kf.statePost[1, 0])
        vx = float(self.vx_f)
        vy = float(self.vy_f)

        pts = []
        for k in range(1, int(n_steps) + 1):
            pts.append((int(cx + vx * dt * k), int(cy + vy * dt * k)))
        return pts

    def update_motion_history(self, cx, cy):
        cx = float(cx)
        cy = float(cy)

        if len(self.center_history) > 0:
            px, py = self.center_history[-1]
            vx = cx - px
            vy = cy - py
            sp = float(np.hypot(vx, vy))

            self.vec_history.append((vx, vy))
            self.speed_history.append(sp)

        self.center_history.append((cx, cy))

    def mean_motion_vector(self, n_recent=None):
        if len(self.vec_history) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        if n_recent is None or n_recent <= 0 or n_recent >= len(self.vec_history):
            arr = np.array(self.vec_history, dtype=np.float32)
        else:
            arr = np.array(list(self.vec_history)[-n_recent:], dtype=np.float32)

        return arr.mean(axis=0)

    def _direction_bin_from_vector(self, vx, vy):
        ang = (np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0

        if 0 <= ang < 60:
            return 0
        elif 60 <= ang < 120:
            return 1
        elif 120 <= ang < 180:
            return 2
        elif 180 <= ang < 240:
            return 3
        elif 240 <= ang < 300:
            return 4
        else:
            return 5

    def get_track_change_message(self, dir_window=3, min_speed=1.0, confirm_frames=3, hold_frames=10):
        # Hold already-triggered message
        if self.message_hold_count > 0:
            self.message_hold_count -= 1
            return "Vehicle track change"

        if len(self.vec_history) < dir_window:
            return ""

        v_now = np.mean(np.array(list(self.vec_history)[-dir_window:], dtype=np.float32), axis=0)
        vn = float(np.linalg.norm(v_now))
        if vn < min_speed:
            self.candidate_dir_bin = None
            self.candidate_count = 0
            return ""

        curr_bin = self._direction_bin_from_vector(v_now[0], v_now[1])

        # Initialize the stable bin once
        if self.prev_dir_bin is None:
            self.prev_dir_bin = curr_bin
            return ""

        # Same as current stable color -> no change
        if curr_bin == self.prev_dir_bin:
            self.candidate_dir_bin = None
            self.candidate_count = 0
            return ""

        # Different from stable color: verify it persists
        if self.candidate_dir_bin is None or curr_bin != self.candidate_dir_bin:
            self.candidate_dir_bin = curr_bin
            self.candidate_count = 1
            return ""

        self.candidate_count += 1

        # Confirm only after consecutive frames
        if self.candidate_count >= confirm_frames:
            self.prev_dir_bin = self.candidate_dir_bin
            self.candidate_dir_bin = None
            self.candidate_count = 0
            self.message_hold_count = max(0, hold_frames - 1)
            return "Vehicle track change"

        return ""
    
    def get_speed_status_message(self, history_frames=30, rel_thr=0.15, hold_frames=8, min_avg_speed=0.5):
        """
        Compare older half vs newer half of the last 30-frame speed history.

        Example with history_frames=30:
        - older part  = frames [-30:-15]
        - newer part  = frames [-15:]

        Returns:
            "Speeding up"
            "Slowing down"
            ""
        """

        if self.speed_hold_count > 0:
            self.speed_hold_count -= 1
            return self.speed_message

        if len(self.speed_history) < history_frames:
            return ""

        arr = np.array(list(self.speed_history)[-history_frames:], dtype=np.float32)
        half = history_frames // 2

        older_avg = float(np.mean(arr[:half]))
        newer_avg = float(np.mean(arr[half:]))

        # Ignore very tiny motion
        if older_avg < min_avg_speed and newer_avg < min_avg_speed:
            self.speed_message = ""
            return ""

        rel_change = (newer_avg - older_avg) / (older_avg + 1e-6)

        if rel_change > rel_thr:
            self.speed_message = "Speeding up"
            self.speed_hold_count = max(0, hold_frames - 1)
            return self.speed_message

        if rel_change < -rel_thr:
            self.speed_message = "Slowing down"
            self.speed_hold_count = max(0, hold_frames - 1)
            return self.speed_message

        self.speed_message = ""
        return ""
    
    def get_start_stop_message(self, speed_window=5, stop_thr=0.6, move_thr=1.2, hold_frames=10):
        """
        Show:
        - 'Slowing down' when vehicle transitions from moving to near-static
        - 'Speeding up' when vehicle transitions from near-static to moving
        """

        if self.motion_msg_hold > 0:
            self.motion_msg_hold -= 1
            return self.motion_msg

        if len(self.speed_history) < speed_window:
            return ""

        avg_speed = float(np.mean(np.array(list(self.speed_history)[-speed_window:], dtype=np.float32)))

        # Initialize state
        if self.motion_state == "unknown":
            if avg_speed <= stop_thr:
                self.motion_state = "static"
            elif avg_speed >= move_thr:
                self.motion_state = "moving"
            return ""

        # moving -> static
        if self.motion_state == "moving" and avg_speed <= stop_thr:
            self.motion_state = "static"
            self.motion_msg = "Slowing down"
            self.motion_msg_hold = max(0, hold_frames - 1)
            return self.motion_msg

        # static -> moving
        if self.motion_state == "static" and avg_speed >= move_thr:
            self.motion_state = "moving"
            self.motion_msg = "Speeding up"
            self.motion_msg_hold = max(0, hold_frames - 1)
            return self.motion_msg

        return ""
    

    # ===== BEGIN OPTIONAL KALMAN QUANT MODULE =====
    def get_quantitative_state(self, frame_idx, measurement=None, gt=None, save_csv=False):
        """
        Generate per-frame quantitative tracking data.

        Args:
            frame_idx (int)
            measurement (tuple or None): (mx, my)
            gt (tuple or None): (gtx, gty)
            save_csv (bool): if True, append to CSV file

        Returns:
            dict with all metrics
        """

        import math
        import numpy as np

        # --- Predicted position ---
        pred_x = float(self.last_pred[0])
        pred_y = float(self.last_pred[1])

        # --- Corrected position (current state) ---
        corr_x = float(self.kf.statePost[0, 0])
        corr_y = float(self.kf.statePost[1, 0])

        # --- Measurement ---
        if measurement is not None:
            meas_x, meas_y = float(measurement[0]), float(measurement[1])
        else:
            meas_x, meas_y = None, None

        # --- Velocity ---
        vx = float(self.kf.statePost[2, 0])
        vy = float(self.kf.statePost[3, 0])

        # --- Uncertainty (covariance) ---
        cov = self.kf.errorCovPost
        pos_var_x = float(cov[0, 0])
        pos_var_y = float(cov[1, 1])

        # --- Residual (measurement - prediction) ---
        if measurement is not None:
            res_x = meas_x - pred_x
            res_y = meas_y - pred_y
            res_mag = math.sqrt(res_x**2 + res_y**2)
        else:
            res_x, res_y, res_mag = None, None, None

        # --- Errors (if GT available) ---
        if gt is not None:
            gt_x, gt_y = float(gt[0]), float(gt[1])

            pred_err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            corr_err = math.sqrt((corr_x - gt_x)**2 + (corr_y - gt_y)**2)
        else:
            gt_x, gt_y = None, None
            pred_err, corr_err = None, None

        result = {
            "frame": frame_idx,
            "track_id": self.id,

            "pred_x": pred_x,
            "pred_y": pred_y,

            "meas_x": meas_x,
            "meas_y": meas_y,

            "corr_x": corr_x,
            "corr_y": corr_y,

            "gt_x": gt_x,
            "gt_y": gt_y,

            "pred_error": pred_err,
            "corr_error": corr_err,

            "residual_x": res_x,
            "residual_y": res_y,
            "residual_mag": res_mag,

            "vx": vx,
            "vy": vy,

            "var_x": pos_var_x,
            "var_y": pos_var_y,
        }

        # --- Optional CSV logging ---
        if save_csv:
            import csv
            import os

            file_exists = os.path.isfile("kalman_quant_results.csv")

            with open("kalman_quant_results.csv", "a", newline="") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(result.keys())

                writer.writerow(result.values())

        return result
    # ===== END OPTIONAL KALMAN QUANT MODULE =====