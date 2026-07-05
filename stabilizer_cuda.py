import cv2
import numpy as np
from collections import deque


class FrameStabilizerCUDA:
    def __init__(
        self,
        max_corners=400,
        quality=0.01,
        min_dist=20,
        smooth_window=12,
        max_translation=35.0,
        max_rotation_deg=3.0,
        max_scale_dev=0.08,
    ):
        self.prev_gray = None
        self.prev_pts = None

        self.max_corners = int(max_corners)
        self.quality = float(quality)
        self.min_dist = float(min_dist)

        self.smooth_window = int(smooth_window)
        self.max_translation = float(max_translation)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_scale_dev = float(max_scale_dev)

        # cumulative camera motion
        self.transforms = deque(maxlen=self.smooth_window)
        self.prev_T = np.eye(3, dtype=np.float32)

    def _build_feature_mask(self, gray):
        """
        Use mostly background regions:
        - avoid center/lower road where vehicles dominate
        - keep upper road / poles / static structures
        """
        h, w = gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Use top ~65% of frame
        y_cut = int(0.65 * h)
        mask[:y_cut, :] = 255

        # Suppress center-lower traffic-heavy zone
        cx1 = int(0.20 * w)
        cx2 = int(0.85 * w)
        cy1 = int(0.28 * h)
        cy2 = int(0.80 * h)
        mask[cy1:cy2, cx1:cx2] = 0

        return mask

    def _detect_features(self, gray):
        mask = self._build_feature_mask(gray)
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality,
            minDistance=self.min_dist,
            blockSize=7,
            mask=mask
        )
        return pts

    def _safe_affine(self, M):
        if M is None:
            return False

        a, b, tx = M[0]
        c, d, ty = M[1]

        scale = np.sqrt(a * a + c * c)
        rot_deg = np.degrees(np.arctan2(c, a))
        trans = np.hypot(tx, ty)

        if trans > self.max_translation:
            return False
        if abs(rot_deg) > self.max_rotation_deg:
            return False
        if abs(scale - 1.0) > self.max_scale_dev:
            return False

        return True

    def _smooth_transform(self, M):
        """
        Smooth affine transform over time to reduce jitter.
        """
        M3 = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        self.transforms.append(M3)

        avg = np.mean(np.stack(self.transforms, axis=0), axis=0)
        return avg[:2, :]

    def stabilize(self, frame_bgr, stream=None):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self._detect_features(gray)
            return frame_bgr, np.eye(2, 3, dtype=np.float32), False

        if self.prev_pts is None or len(self.prev_pts) < 30:
            self.prev_pts = self._detect_features(self.prev_gray)
            if self.prev_pts is None:
                self.prev_gray = gray
                return frame_bgr, np.eye(2, 3, dtype=np.float32), False

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3
        )

        if next_pts is None or status is None:
            self.prev_gray = gray
            self.prev_pts = self._detect_features(gray)
            return frame_bgr, np.eye(2, 3, dtype=np.float32), False

        status = status.reshape(-1)
        good_prev = self.prev_pts.reshape(-1, 2)[status == 1]
        good_next = next_pts.reshape(-1, 2)[status == 1]

        if len(good_prev) < 25:
            self.prev_gray = gray
            self.prev_pts = self._detect_features(gray)
            return frame_bgr, np.eye(2, 3, dtype=np.float32), False

        M, inliers = cv2.estimateAffinePartial2D(
            good_next,
            good_prev,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=2000,
            confidence=0.99
        )

        if M is None or not self._safe_affine(M):
            self.prev_gray = gray
            self.prev_pts = self._detect_features(gray)
            return frame_bgr, np.eye(2, 3, dtype=np.float32), False

        # optional: reject if too few inliers
        if inliers is not None:
            inlier_ratio = float(inliers.mean())
            if inlier_ratio < 0.55:
                self.prev_gray = gray
                self.prev_pts = self._detect_features(gray)
                return frame_bgr, np.eye(2, 3, dtype=np.float32), False

        M_smooth = self._smooth_transform(M)

        h, w = frame_bgr.shape[:2]
        stabilized = cv2.warpAffine(
            frame_bgr,
            M_smooth,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # IMPORTANT: re-detect features on current raw frame
        self.prev_gray = gray
        self.prev_pts = self._detect_features(gray)

        return stabilized, M_smooth.astype(np.float32), True