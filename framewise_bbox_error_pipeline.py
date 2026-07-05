import argparse
import csv
import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _load_local_module(module_name, candidates):
    try:
        return __import__(module_name)
    except Exception:
        pass

    here = Path(__file__).resolve().parent
    for cand in candidates:
        p = here / cand
        if p.exists():
            spec = importlib.util.spec_from_file_location(module_name, str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(f"Could not load module {module_name}. Checked: {candidates}")


utils_mod = _load_local_module("utils", ["utils.py", "utils(8).py"])
config_mod = _load_local_module("config", ["config.py", "config(6).py"])
assoc_mod = _load_local_module("association", ["association.py", "association(3).py"])
blobs_mod = _load_local_module("blobs", ["blobs.py", "blobs(7).py"])
kalman_mod = _load_local_module("kalman_track", ["kalman_track.py", "kalman_track(9).py"])

try:
    cuda_backend_mod = _load_local_module("cuda_backend", ["cuda_backend.py", "cuda_backend(10).py"])
except Exception:
    cuda_backend_mod = None

bbox_area_xyxy = utils_mod.bbox_area_xyxy
bbox_center_xyxy = utils_mod.bbox_center_xyxy
clamp_bbox_xyxy = utils_mod.clamp_bbox_xyxy
iou_xyxy = utils_mod.iou_xyxy

assoc_by_center_distance_roadaware = assoc_mod.assoc_by_center_distance_roadaware
extract_blobs = blobs_mod.extract_blobs
assoc_tracks_to_blobs_strict = blobs_mod.assoc_tracks_to_blobs_strict
KalmanTrack = kalman_mod.KalmanTrack

YOLO_STRIDE = int(getattr(config_mod, "YOLO_STRIDE", 5))
MIN_HITS_TO_ACTIVATE = int(getattr(config_mod, "MIN_HITS_TO_ACTIVATE", 1))
TENTATIVE_MAX_MISSES = int(getattr(config_mod, "TENTATIVE_MAX_MISSES", 5))
ACTIVE_MAX_MISSES = int(getattr(config_mod, "ACTIVE_MAX_MISSES", 100))
ACTIVE_PRED_ONLY_MAX = int(getattr(config_mod, "ACTIVE_PRED_ONLY_MAX", 50))
HIGH_CONF = float(getattr(config_mod, "HIGH_CONF", 0.55))
SPAWN_CONF_THR = float(getattr(config_mod, "SPAWN_CONF_THR", 0.60))
MIN_BOX_AREA = float(getattr(config_mod, "MIN_BOX_AREA", 900))
SPAWN_MOTION_THR = float(getattr(config_mod, "SPAWN_MOTION_THR", 0.02))
BASE_GATE = float(getattr(config_mod, "BASE_GATE", 60))
K_GATE = float(getattr(config_mod, "K_GATE", 1.5))
MAX_GATE = float(getattr(config_mod, "MAX_GATE", 220))
YOLO_MODEL = str(getattr(config_mod, "YOLO_MODEL", "yolo26s.pt"))
DEVICE = str(getattr(config_mod, "DEVICE", "cuda"))
USE_CUSTOM_CUDA_BACKEND = bool(getattr(config_mod, "USE_CUSTOM_CUDA_BACKEND", False))
CUDA_BACKEND_LIB = str(getattr(config_mod, "CUDA_BACKEND_LIB", "./libmotion_backend.so"))
CUDA_BACKEND_MIN_AREA = int(getattr(config_mod, "CUDA_BACKEND_MIN_AREA", 800))


class CpuKnnMotion:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorKNN(
            history=1200,
            dist2Threshold=900.0,
            detectShadows=False,
        )
        self.morph_kernel = np.ones((3, 3), np.uint8)

    def process(self, frame, learning_rate=0.0015, min_area=800):
        knn_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        fgmask = self.fgbg.apply(knn_frame, learningRate=learning_rate)
        fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        fgmask = cv2.erode(fgmask, np.ones((3, 3), np.uint8), iterations=1)
        fgmask = cv2.dilate(fgmask, np.ones((5, 5), np.uint8), iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.morph_kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.morph_kernel)
        fgmask = cv2.medianBlur(fgmask, 5)
        blobs = extract_blobs(fgmask, min_area=min_area)
        return fgmask, blobs


def list_images(image_dir):
    paths = [p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort(key=lambda p: p.name)
    if not paths:
        raise RuntimeError(f"No images found in: {image_dir}")
    return paths


def xyxy_to_xywhn(bbox_xyxy, img_w, img_h):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = np.clip(x1, 0.0, max(0.0, img_w - 1.0))
    x2 = np.clip(x2, 0.0, max(0.0, img_w - 1.0))
    y1 = np.clip(y1, 0.0, max(0.0, img_h - 1.0))
    y2 = np.clip(y2, 0.0, max(0.0, img_h - 1.0))

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + 0.5 * w
    yc = y1 + 0.5 * h

    return np.array([
        xc / float(img_w),
        yc / float(img_h),
        w / float(img_w),
        h / float(img_h),
    ], dtype=np.float32)


def xywhn_to_xyxy(bbox_xywhn, img_w, img_h):
    xc, yc, w, h = [float(v) for v in bbox_xywhn]
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    x1 = xc - 0.5 * w
    y1 = yc - 0.5 * h
    x2 = xc + 0.5 * w
    y2 = yc + 0.5 * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def bbox_xywhn_str(b):
    xc, yc, w, h = [float(v) for v in b]
    return f"[{xc:.6f}, {yc:.6f}, {w:.6f}, {h:.6f}]"

'''
def load_yolo_gt_for_frame(label_path, frame_shape):
    h, w = frame_shape[:2]
    items = []
    if not os.path.exists(label_path):
        return items

    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for obj_idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        bbox_xywhn = np.array(list(map(float, parts[1:5])), dtype=np.float32)
        bbox_xyxy = xywhn_to_xyxy(bbox_xywhn, w, h)
        items.append({
            "gt_index": obj_idx,
            "class_id": cls_id,
            "bbox_xywhn": bbox_xywhn,
            "bbox_xyxy": bbox_xyxy,
        })
    return items
'''
def polygon_to_bbox_xywhn(coords):
    xs = coords[0::2]
    ys = coords[1::2]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    return np.array([xc, yc, w, h], dtype=np.float32)


def load_yolo_gt_for_frame(label_path, frame_shape):
    h, w = frame_shape[:2]
    items = []

    if not os.path.exists(label_path):
        return items

    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for obj_idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        nums = np.array(list(map(float, parts[1:])), dtype=np.float32)

        # Normal YOLO bbox: class xc yc w h
        if len(nums) == 4:
            bbox_xywhn = nums

        # YOLO segmentation polygon: class x1 y1 x2 y2 ...
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            bbox_xywhn = polygon_to_bbox_xywhn(nums)

        else:
            print(f"[WARN] Bad label format: {label_path}, line {obj_idx}: {line}")
            continue

        bbox_xyxy = xywhn_to_xyxy(bbox_xywhn, w, h)

        items.append({
            "gt_index": obj_idx,
            "class_id": cls_id,
            "bbox_xywhn": bbox_xywhn,
            "bbox_xyxy": bbox_xyxy,
        })

    return items

def motion_score(fgmask, bbox_xyxy):
    h, w = fgmask.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = fgmask[y1:y2, x1:x2]
    return float((roi > 0).mean()) if roi.size else 0.0


def too_close_to_existing(det_bbox, tracks, iou_thr=0.15, dist_thr_factor=0.9):
    dcx, dcy = bbox_center_xyxy(det_bbox)
    for t in tracks:
        active = (t.hits >= MIN_HITS_TO_ACTIVATE)
        recently_alive = getattr(t, "pred_only_streak", 999) <= 10
        if not active and not recently_alive:
            continue
        tcx, tcy = t.draw_point()
        tb = t.bbox_from_center(tcx, tcy)
        if iou_xyxy(det_bbox, tb) > iou_thr:
            return True
        bw = max(2.0, tb[2] - tb[0])
        bh = max(2.0, tb[3] - tb[1])
        diag = np.hypot(bw, bh)
        scale = max(bw, bh)
        if scale > 120:
            if np.hypot(dcx - tcx, dcy - tcy) < 1.25 * scale:
                return True
        else:
            if np.hypot(dcx - tcx, dcy - tcy) < dist_thr_factor * diag:
                return True
    return False


def find_recoverable_track(det_bbox, tracks, max_center_dist_factor=1.6, min_iou=0.03):
    dcx, dcy = bbox_center_xyxy(det_bbox)
    best_idx = -1
    best_score = -1e9
    for i, t in enumerate(tracks):
        if getattr(t, "pred_only_streak", 0) <= 0:
            continue
        tcx, tcy = t.draw_point()
        tb = t.bbox_from_center(tcx, tcy)
        tw = max(2.0, tb[2] - tb[0])
        th = max(2.0, tb[3] - tb[1])
        diag = np.hypot(tw, th)
        dist = np.hypot(dcx - tcx, dcy - tcy)
        iou = iou_xyxy(det_bbox, tb)
        if dist > max_center_dist_factor * diag and iou < min_iou:
            continue
        score = iou - 0.15 * (dist / (diag + 1e-6))
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def match_predictions_to_gt(preds, gts, img_w, img_h):
    candidates = []
    for pi, p in enumerate(preds):
        pb = xywhn_to_xyxy(p["bbox_xywhn"], img_w, img_h)
        for gi, g in enumerate(gts):
            gb = xywhn_to_xyxy(g["bbox_xywhn"], img_w, img_h)
            iou = iou_xyxy(pb, gb)
            candidates.append((iou, pi, gi))
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_p, used_g = set(), set()
    matches = []
    for iou, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, float(iou)))

    unmatched_p = [i for i in range(len(preds)) if i not in used_p]
    unmatched_g = [i for i in range(len(gts)) if i not in used_g]
    return matches, unmatched_p, unmatched_g


def make_motion_backend(width, height):
    if USE_CUSTOM_CUDA_BACKEND and cuda_backend_mod is not None and os.path.exists(CUDA_BACKEND_LIB):
        try:
            backend = cuda_backend_mod.CudaMotionBackend(CUDA_BACKEND_LIB, width, height)
            print(f"[INFO] Using custom CUDA KNN backend: {CUDA_BACKEND_LIB}")
            return backend, True
        except Exception as e:
            print(f"[WARN] CUDA backend init failed, using CPU KNN fallback: {e}")
    print("[INFO] Using CPU KNN fallback.")
    return CpuKnnMotion(), False


def parse_args():
    ap = argparse.ArgumentParser(
        description="YOLO every 5th frame + continuous Kalman + KNN refinement + normalized framewise GT error table"
    )
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--gt_label_dir", required=True)
    ap.add_argument("--weights", default=YOLO_MODEL)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size. Default: 1280")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=5, help="YOLO prediction interval. Default: 5")
    ap.add_argument("--num_frames", type=int, default=496)
    ap.add_argument("--start_frame", type=int, default=1)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--overlay", action="store_true", help="Save overlay images")
    return ap.parse_args()


def draw_overlay(frame, gt_items, final_preds, frame_no):
    vis = frame.copy()
    h, w = frame.shape[:2]

    for g in gt_items:
        x1, y1, x2, y2 = xywhn_to_xyxy(g["bbox_xywhn"], w, h).astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "GT", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    for p in final_preds:
        x1, y1, x2, y2 = xywhn_to_xyxy(p["bbox_xywhn"], w, h).astype(int)
        col = (255, 0, 0) if p["PredictionMethod"] == "YOLO" else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            vis,
            f"{p['PredictionMethod']} T{p['track_id']}",
            (x1, min(frame.shape[0] - 8, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            2,
        )

    cv2.putText(vis, f"Frame {frame_no}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    overlay_dir = os.path.join(args.output_dir, "overlays")
    if args.overlay:
        os.makedirs(overlay_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "framewise_error_table.csv")

    images = list_images(args.image_dir)
    start_idx = max(0, args.start_frame - 1)
    selected_images = images[start_idx:start_idx + args.num_frames]
    if not selected_images:
        raise RuntimeError("No frames selected. Check --start_frame and --num_frames.")

    first_frame = cv2.imread(str(selected_images[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {selected_images[0]}")

    motion_backend, using_cuda_backend = make_motion_backend(first_frame.shape[1], first_frame.shape[0])

    model = YOLO(args.weights)
    model.to(args.device)

    tracks = []
    next_id = 0
    prev_fg_fraction = 0.0

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "Frame No.",
            "PredictionMethode(YOLO/Kalman)",
            "PredictedBBOX_Normalized_xywh",
            "GTBBOX_Normalized_xywh",
            "IOU",
            "Error",
        ])

        for local_idx, img_path in enumerate(selected_images):
            frame_no = args.start_frame + local_idx
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"[WARN] Skipping unreadable image: {img_path}")
                continue

            H, W = frame.shape[:2]
            label_path = os.path.join(args.gt_label_dir, img_path.stem + ".txt")
            gt_items = load_yolo_gt_for_frame(label_path, frame.shape)

            run_yolo = (local_idx % args.stride == 0)

            tracks_pred_xy = []
            for ti, t in enumerate(tracks):
                t.age += 1
                px, py = t.predict_only()
                tracks_pred_xy.append((ti, (px, py)))
                setattr(t, "frame_source", "Kalman")

            adaptive_lr = 0.0015
            if prev_fg_fraction > 0.18:
                adaptive_lr = 0.0
            elif prev_fg_fraction > 0.10:
                adaptive_lr = min(adaptive_lr, 0.0005)

            if using_cuda_backend:
                fgmask, backend_blobs = motion_backend.process(
                    frame,
                    learning_rate=adaptive_lr,
                    min_area=CUDA_BACKEND_MIN_AREA,
                )
                blobs = [{
                    "bbox": np.array([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], dtype=np.float32),
                    "cx": float(b["cx"]),
                    "cy": float(b["cy"]),
                    "area": float(b["area"]),
                } for b in backend_blobs]
            else:
                fgmask, blobs = motion_backend.process(frame, learning_rate=adaptive_lr, min_area=CUDA_BACKEND_MIN_AREA)

            prev_fg_fraction = float((fgmask > 0).mean())

            dets = []
            did_yolo_update = False

            if run_yolo:
                half = args.device.startswith("cuda") and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
                res = model.predict(
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    half=half,
                    verbose=False,
                )[0]

                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    clss = res.boxes.cls.cpu().numpy()
                    for b, c, k in zip(xyxy, confs, clss):
                        b = clamp_bbox_xyxy(b, W, H)
                        area = bbox_area_xyxy(b)
                        ms = motion_score(fgmask, b)
                        dets.append((b, int(k), float(c), float(ms), float(area)))

                if dets and tracks_pred_xy:
                    matches, um_tracks, um_dets = assoc_by_center_distance_roadaware(
                        tracks_pred_xy,
                        dets,
                        tracks,
                        None,
                        base=BASE_GATE,
                        k=K_GATE,
                        max_gate=MAX_GATE,
                        conf_min=args.conf,
                        min_area=MIN_BOX_AREA * 0.7,
                    )

                    for pti, di in matches:
                        track_index = tracks_pred_xy[pti][0]
                        t = tracks[track_index]
                        b, cls_id, conf, _, _ = dets[di]
                        t.update(b, conf, frame_no, update_size=True)
                        t.cls_id = cls_id
                        setattr(t, "frame_source", "YOLO")

                    for pti in um_tracks:
                        tracks[tracks_pred_xy[pti][0]].mark_yolo_missed()

                    for di in um_dets:
                        b, cls_id, conf, ms, area = dets[di]
                        if conf < SPAWN_CONF_THR or area < MIN_BOX_AREA:
                            continue
                        if ms < SPAWN_MOTION_THR and conf < HIGH_CONF:
                            continue

                        rescue_idx = find_recoverable_track(b, tracks)
                        if rescue_idx >= 0:
                            t = tracks[rescue_idx]
                            t.update(b, conf, frame_no, update_size=True)
                            t.cls_id = cls_id
                            t.yolo_misses = 0
                            t.pred_only_streak = 0
                            setattr(t, "frame_source", "YOLO")
                            continue

                        if too_close_to_existing(b, tracks, iou_thr=0.10, dist_thr_factor=1.2):
                            continue

                        nt = KalmanTrack(next_id, b, cls_id, conf, frame_no)
                        setattr(nt, "frame_source", "YOLO")
                        tracks.append(nt)
                        next_id += 1

                    did_yolo_update = True

                elif dets and not tracks_pred_xy:
                    for b, cls_id, conf, ms, area in dets:
                        if conf < SPAWN_CONF_THR or area < MIN_BOX_AREA or ms < SPAWN_MOTION_THR:
                            continue
                        nt = KalmanTrack(next_id, b, cls_id, conf, frame_no)
                        setattr(nt, "frame_source", "YOLO")
                        tracks.append(nt)
                        next_id += 1
                    did_yolo_update = True

            if not did_yolo_update and tracks_pred_xy:
                blob_matches = assoc_tracks_to_blobs_strict(
                    tracks_pred_xy,
                    tracks,
                    blobs,
                    W,
                    H,
                    base=BASE_GATE,
                    k=K_GATE,
                    max_gate=MAX_GATE,
                    iou_thr=0.01,
                    pad_factor=0.60,
                    area_ratio=(0.10, 12.0),
                )
                matched = set()
                for ti, bj in blob_matches:
                    if tracks[ti].hits < MIN_HITS_TO_ACTIVATE:
                        continue
                    b = blobs[bj]
                    tracks[ti].correct_center(b["cx"], b["cy"], frame_no, conf=0.0)
                    setattr(tracks[ti], "frame_source", "Kalman")
                    matched.add(ti)
                for ti, _ in tracks_pred_xy:
                    if ti not in matched:
                        tracks[ti].mark_pred_only()
                        setattr(tracks[ti], "frame_source", "Kalman")

            kept = []
            for t in tracks:
                active = (t.hits >= MIN_HITS_TO_ACTIVATE)
                remove_track = False
                if not active:
                    if t.pred_only_streak <= TENTATIVE_MAX_MISSES:
                        kept.append(t)
                    else:
                        remove_track = True
                else:
                    if t.yolo_misses > ACTIVE_MAX_MISSES:
                        remove_track = True
                    elif t.pred_only_streak > ACTIVE_PRED_ONLY_MAX:
                        remove_track = True
                    else:
                        kept.append(t)
                if remove_track and getattr(t, "death_frame", None) is None:
                    t.death_frame = frame_no
            tracks = kept

            final_preds = []
            for t in tracks:
                cx, cy = t.draw_point()
                tb_xyxy = t.bbox_from_center(cx, cy).copy()
                tb_xyxy = clamp_bbox_xyxy(tb_xyxy, W, H)
                tb_xywhn = xyxy_to_xywhn(tb_xyxy, W, H)
                method = getattr(t, "frame_source", "Kalman")
                final_preds.append({
                    "track_id": int(t.id),
                    "PredictionMethod": method,
                    "bbox_xywhn": tb_xywhn,
                })

            matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(final_preds, gt_items, W, H)

            for pi, gi, iou in matches:
                pred_bbox = final_preds[pi]["bbox_xywhn"]
                gt_bbox = gt_items[gi]["bbox_xywhn"]
                error = 1.0 - float(iou)
                writer.writerow([
                    frame_no,
                    final_preds[pi]["PredictionMethod"],
                    bbox_xywhn_str(pred_bbox),
                    bbox_xywhn_str(gt_bbox),
                    f"{iou:.6f}",
                    f"{error:.6f}",
                ])

            for pi in unmatched_preds:
                pred_bbox = final_preds[pi]["bbox_xywhn"]
                writer.writerow([
                    frame_no,
                    final_preds[pi]["PredictionMethod"],
                    bbox_xywhn_str(pred_bbox),
                    "[]",
                    f"{0.0:.6f}",
                    f"{1.0:.6f}",
                ])

            for gi in unmatched_gts:
                gt_bbox = gt_items[gi]["bbox_xywhn"]
                writer.writerow([
                    frame_no,
                    "None",
                    "[]",
                    bbox_xywhn_str(gt_bbox),
                    f"{0.0:.6f}",
                    f"{1.0:.6f}",
                ])

            if args.overlay:
                overlay = draw_overlay(frame, gt_items, final_preds, frame_no)
                cv2.imwrite(os.path.join(overlay_dir, f"frame_{frame_no:04d}.jpg"), overlay)

    print(f"Saved CSV: {csv_path}")
    if args.overlay:
        print(f"Saved overlays: {overlay_dir}")


if __name__ == "__main__":
    main()
