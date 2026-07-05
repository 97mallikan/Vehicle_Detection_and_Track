# blobs.py
import numpy as np
import cv2
from association import track_gate_px

def extract_blobs(fgmask, min_area=500):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(fgmask, connectivity=8)
    blobs = []
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = cents[i]
        blobs.append({"bbox": (x, y, x+w, y+h), "cx": float(cx), "cy": float(cy), "area": area})
    return blobs

def iou_xyxy_int(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

def assoc_tracks_to_blobs_strict(
    tracks_pred_xy, tracks, blobs, W, H,
    base=60, k=1.5, max_gate=220,
    iou_thr=0.01,
    pad_factor=0.35,
    area_ratio=(0.15, 6.0)
):
    if not tracks_pred_xy or not blobs:
        return []

    used_b = set()
    matches = []

    for (ti, (px, py)) in tracks_pred_xy:
        t = tracks[ti]
        max_dist_px = track_gate_px(t, base=base, k=k, max_gate=max_gate)

        tb = t.bbox_from_center(px, py)
        x1, y1, x2, y2 = tb
        tw = max(2.0, x2 - x1)
        th = max(2.0, y2 - y1)

        padx = pad_factor * tw
        pady = pad_factor * th
        ex1 = int(np.clip(x1 - padx, 0, W - 1))
        ey1 = int(np.clip(y1 - pady, 0, H - 1))
        ex2 = int(np.clip(x2 + padx, 0, W - 1))
        ey2 = int(np.clip(y2 + pady, 0, H - 1))
        exp_box = (ex1, ey1, ex2, ey2)

        track_area = float(tw * th)

        best_j = -1
        best_cost = 1e9

        for j, b in enumerate(blobs):
            if j in used_b:
                continue

            bx1, by1, bx2, by2 = b["bbox"]
            if bx2 < ex1 or bx1 > ex2 or by2 < ey1 or by1 > ey2:
                continue

            ar = float(b["area"]) / (track_area + 1e-6)
            if ar < area_ratio[0] or ar > area_ratio[1]:
                continue

            dist = float(np.hypot(px - b["cx"], py - b["cy"]))
            if dist > max_dist_px:
                continue

            iou = iou_xyxy_int(exp_box, b["bbox"])
            if iou < iou_thr:
                continue

            dist_norm = dist / (max_dist_px + 1e-6)
            cost = (1.0 - iou) + 0.35 * dist_norm
            if cost < best_cost:
                best_cost = cost
                best_j = j

        if best_j >= 0:
            used_b.add(best_j)
            matches.append((ti, best_j))

    return matches