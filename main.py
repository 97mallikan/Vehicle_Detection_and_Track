import argparse
import re
import math
import time
import cv2
import numpy as np
from ultralytics import YOLO

from config import *
from utils import clamp_bbox_xyxy, bbox_area_xyxy
from polygon import PolygonZone
from kalman_track import KalmanTrack
from road_direction import RoadDirectionEstimator
from association import assoc_by_center_distance_roadaware
from blobs import extract_blobs, assoc_tracks_to_blobs_strict
from fg_cuda import CudaFgPipeline
from cuda_backend import CudaMotionBackend
from stabilizer_cuda import FrameStabilizerCUDA


FLOW_MIN_MAG = 0.35
FLOW_ANGLE_THR_DEG = 35.0
FLOW_MAG_RATIO_LOW = 0.35
FLOW_POINT_DRAW_SCALE = 3.0


def open_video_source(src: str):
    src = src.strip()
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    elif src.startswith("/dev/video"):
        m = re.search(r"/dev/video(\d+)", src)
        if not m:
            raise ValueError(f"Invalid device path: {src}")
        cap = cv2.VideoCapture(int(m.group(1)))
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap


def has_cuda():
    return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0


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
    return float((roi > 0).mean())


def too_close_to_existing(det_bbox, tracks, iou_thr=0.15, dist_thr_factor=0.9):
    from utils import bbox_center_xyxy, iou_xyxy

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

        w = max(2.0, tb[2] - tb[0])
        h = max(2.0, tb[3] - tb[1])
        diag = np.hypot(w, h)
        scale = max(w, h)

        if scale > 120:
            if np.hypot(dcx - tcx, dcy - tcy) < 1.25 * scale:
                return True
        else:
            if np.hypot(dcx - tcx, dcy - tcy) < dist_thr_factor * diag:
                return True

    return False


def draw_kernel_box(img, kernel_box, color=(0, 255, 255), thickness=1):
    x1, y1, x2, y2 = [int(v) for v in kernel_box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def velocity_to_color(vx, vy):
    angle = math.atan2(vy, vx)
    angle_deg = (math.degrees(angle) + 360.0) % 360.0

    if 0 <= angle_deg < 60:
        return (255, 0, 0)
    elif 60 <= angle_deg < 120:
        return (255, 255, 0)
    elif 120 <= angle_deg < 180:
        return (0, 255, 0)
    elif 180 <= angle_deg < 240:
        return (0, 255, 255)
    elif 240 <= angle_deg < 300:
        return (255, 0, 255)
    else:
        return (0, 165, 255)


def draw_history_from_list(vis, pts, thickness=2, min_step=1.0):
    if len(pts) < 2:
        return

    pts_int = [(int(round(x)), int(round(y))) for (x, y) in pts]

    for i in range(1, len(pts_int)):
        x1, y1 = pts_int[i - 1]
        x2, y2 = pts_int[i]

        vx = x2 - x1
        vy = y2 - y1
        mag = math.sqrt(vx * vx + vy * vy)
        if mag < min_step:
            continue

        col = velocity_to_color(vx, vy)
        cv2.line(vis, (x1, y1), (x2, y2), col, thickness, lineType=cv2.LINE_AA)


def draw_local_flow_points(vis, points, scale=3.0, force_red=False):
    for p in points:
        x0 = int(round(p["x0"]))
        y0 = int(round(p["y0"]))
        x1 = int(round(p["x0"] + p["vx"] * scale))
        y1 = int(round(p["y0"] + p["vy"] * scale))

        if force_red or p["anomalous"]:
            color = (0, 0, 255)
            thickness = 3
            radius = 3
        else:
            color = velocity_to_color(p["vx"], p["vy"])
            thickness = 2
            radius = 2

        cv2.arrowedLine(vis, (x0, y0), (x1, y1), color, thickness, line_type=cv2.LINE_AA, tipLength=0.3)
        cv2.circle(vis, (x0, y0), radius, color, -1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, default=INPUT_VIDEO)
    p.add_argument("--weights", type=str, default=YOLO_MODEL)
    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=YOLO_STRIDE)
    p.add_argument("--poly", type=int, default=4)
    return p.parse_args()


def find_recoverable_track(det_bbox, tracks, max_center_dist_factor=1.6, min_iou=0.03):
    from utils import bbox_center_xyxy, iou_xyxy

    dcx, dcy = bbox_center_xyxy(det_bbox)
    best_idx = -1
    best_score = -1e9

    for i, t in enumerate(tracks):
        if t.pred_only_streak <= 0:
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


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    return v / n


def estimate_flow_clusters(tracks, min_hits=3, recent_window=8, min_speed=0.8):
    samples = []
    speeds = []

    for t in tracks:
        if t.hits < min_hits:
            continue
        v = t.mean_motion_vector(n_recent=recent_window)
        s = float(np.linalg.norm(v))
        if s < min_speed:
            continue
        u = _normalize(v)
        if u is None:
            continue
        samples.append(u)
        speeds.append(s)

    if len(samples) == 0:
        return []

    samples = np.array(samples, dtype=np.float32)
    speeds = np.array(speeds, dtype=np.float32)

    if len(samples) == 1:
        return [{"u": samples[0], "mean_speed": float(speeds[0])}]

    c1 = samples[0]
    dots = samples @ c1
    c2 = samples[np.argmin(dots)]

    for _ in range(8):
        g1 = []
        g2 = []
        s1 = []
        s2 = []

        for u, sp in zip(samples, speeds):
            if float(np.dot(u, c1)) >= float(np.dot(u, c2)):
                g1.append(u)
                s1.append(sp)
            else:
                g2.append(u)
                s2.append(sp)

        if len(g1) == 0 or len(g2) == 0:
            break

        c1_new = _normalize(np.mean(np.array(g1, dtype=np.float32), axis=0))
        c2_new = _normalize(np.mean(np.array(g2, dtype=np.float32), axis=0))
        if c1_new is None or c2_new is None:
            break

        c1 = c1_new
        c2 = c2_new

    clusters = []
    for center in [c1, c2]:
        members = []
        member_speeds = []
        for u, sp in zip(samples, speeds):
            if center is c1:
                cond = float(np.dot(u, c1)) >= float(np.dot(u, c2))
            else:
                cond = float(np.dot(u, c1)) < float(np.dot(u, c2))
            if cond:
                members.append(u)
                member_speeds.append(sp)

        if len(members) == 0:
            continue

        cu = _normalize(np.mean(np.array(members, dtype=np.float32), axis=0))
        if cu is None:
            continue

        clusters.append({
            "u": cu,
            "mean_speed": float(np.mean(np.array(member_speeds, dtype=np.float32)))
        })

    if len(clusters) == 2:
        sim = float(np.dot(clusters[0]["u"], clusters[1]["u"]))
        if sim > 0.95:
            merged_u = _normalize((clusters[0]["u"] + clusters[1]["u"]) * 0.5)
            merged_speed = 0.5 * (clusters[0]["mean_speed"] + clusters[1]["mean_speed"])
            return [{"u": merged_u, "mean_speed": float(merged_speed)}]

    return clusters


def assign_track_to_flow_cluster(track, clusters, recent_window=8):
    if len(clusters) == 0:
        return None, None, 0.0

    v = track.mean_motion_vector(n_recent=recent_window)
    u = _normalize(v)
    if u is None:
        return None, None, 0.0

    best_idx = -1
    best_cos = -2.0
    for i, c in enumerate(clusters):
        cs = float(np.dot(u, c["u"]))
        if cs > best_cos:
            best_cos = cs
            best_idx = i

    if best_idx < 0:
        return None, None, 0.0

    return clusters[best_idx]["u"], float(clusters[best_idx]["mean_speed"]), float(best_cos)


def main():
    args = parse_args()

    zone = PolygonZone()
    print(f"Polygon points: {int(args.poly)}")

    use_cuda = USE_OPENCV_CUDA and has_cuda()
    stream = cv2.cuda_Stream() if use_cuda else None

    stabilizer = FrameStabilizerCUDA()
    road_dir = RoadDirectionEstimator(ema=0.92, min_speed=0.8)

    model = YOLO(args.weights)
    model.to(args.device)

    base_learn_rate = 0.0015
    learn_rate = base_learn_rate
    prev_fg_fraction = 0.0
    cuda_fg = CudaFgPipeline()
    morph_kernel = np.ones((3, 3), np.uint8)

    fgbg = None
    backend = None
    backend_ready = False

    tracks = []
    next_id = 0
    track_flow_history = {}
    all_tracks = []

    cv2.namedWindow("Video")
    cv2.setMouseCallback(
        "Video",
        lambda e, x, y, f, p: zone.mouse_cb(e, x, y, f, int(args.poly)),
        int(args.poly),
    )

    frame_idx = 0
    cap = open_video_source(args.video)

    ret0, first_frame = cap.read()
    if not ret0:
        raise RuntimeError(f"Could not read first frame from source: {args.video}")

    first_frame, _, _ = stabilizer.stabilize(first_frame, stream=stream)
    H0, W0 = first_frame.shape[:2]

    if USE_CUSTOM_CUDA_BACKEND:
        try:
            backend = CudaMotionBackend(CUDA_BACKEND_LIB, W0, H0)
            backend_ready = True
            print(f"[INFO] Custom CUDA backend enabled: {CUDA_BACKEND_LIB}")
        except Exception as e:
            print(f"[WARN] Custom CUDA backend init failed, falling back to Python path: {e}")
            backend_ready = False

    if not backend_ready:
        fgbg = cv2.createBackgroundSubtractorKNN(
            history=1200,
            dist2Threshold=900.0,
            detectShadows=False,
        )

    pending_frame = first_frame

    while True:
        start = time.perf_counter()

        if pending_frame is not None:
            frame = pending_frame
            pending_frame = None
            ret = True
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame, _, stab_ok = stabilizer.stabilize(frame, stream=stream)
        if not stab_ok:
            learn_rate = 0.0

        frame_idx += 1
        H, W = frame.shape[:2]

        adaptive_lr = base_learn_rate
        if prev_fg_fraction > 0.18:
            adaptive_lr = 0.0
        elif prev_fg_fraction > 0.10:
            adaptive_lr = min(base_learn_rate, 0.0005)

        learn_rate = adaptive_lr
        backend_blobs = None

        if backend_ready:
            fgmask, backend_blobs = backend.process(
                frame,
                learning_rate=learn_rate,
                min_area=CUDA_BACKEND_MIN_AREA,
            )
        else:
            knn_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            fgmask = fgbg.apply(knn_frame, learningRate=learn_rate)
            if use_cuda:
                fgmask = cuda_fg.process(fgmask, stream=stream)
            else:
                fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
                fgmask = cv2.erode(fgmask, np.ones((3, 3), np.uint8), iterations=1)
                fgmask = cv2.dilate(fgmask, np.ones((5, 5), np.uint8), iterations=1)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, morph_kernel)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, morph_kernel)
                fgmask = cv2.medianBlur(fgmask, 5)

        prev_fg_fraction = float((fgmask > 0).mean())

        large_object_present = False

        for t in tracks:
            cx, cy = t.draw_point()

            # ===== KALMAN QUANT LOGGING (ADD HERE) =====
            t.get_quantitative_state(
                frame_idx,
                measurement=(cx, cy),   # current observed/estimated center
                gt=None,                # replace if you have ground truth
                save_csv=True
            )
            # ===========================================
            tb = t.bbox_from_center(cx, cy)
            th = max(0.0, tb[3] - tb[1])
            tw = max(0.0, tb[2] - tb[0])
            if th > 180 or tw > 220:
                large_object_present = True
                break

        run_yolo = (
            frame_idx <= WARMUP_FRAMES
            or (frame_idx % args.stride == 0)
            or large_object_present
        )

        tracks_pred_xy = []
        for ti, t in enumerate(tracks):
            t.age += 1
            px, py = t.predict_only()
            tracks_pred_xy.append((ti, (px, py)))

        road_u = road_dir.update(tracks, min_hits=MIN_HITS_TO_ACTIVATE)

        dets = []
        raw_yolo = 0
        did_yolo_update = False

        if run_yolo:
            half = args.device.startswith("cuda") and has_cuda()
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
                raw_yolo = len(xyxy)

                for b, c, k in zip(xyxy, confs, clss):
                    b = clamp_bbox_xyxy(b, W, H)
                    area = bbox_area_xyxy(b)
                    ms = motion_score(fgmask, b)
                    dets.append((b, int(k), float(c), float(ms), float(area)))

            if dets and tracks_pred_xy:
                matches, um_tracks, um_dets = assoc_by_center_distance_roadaware(
                    tracks_pred_xy, dets, tracks, None,
                    base=BASE_GATE, k=K_GATE, max_gate=MAX_GATE,
                    conf_min=args.conf, min_area=MIN_BOX_AREA * 0.7,
                )

                for pti, di in matches:
                    track_index = tracks_pred_xy[pti][0]
                    t = tracks[track_index]
                    b, cls_id, conf, _, _ = dets[di]
                    t.last_update_type = "yolo"
                    t.update(b, conf, frame_idx, update_size=True)
                    t.cls_id = cls_id

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
                        t.update(b, conf, frame_idx, update_size=True)
                        t.cls_id = cls_id
                        t.yolo_misses = 0
                        t.pred_only_streak = 0
                        continue

                    if too_close_to_existing(b, tracks, iou_thr=0.10, dist_thr_factor=1.2):
                        continue

                    new_track = KalmanTrack(next_id, b, cls_id, conf, frame_idx)
                    tracks.append(new_track)
                    all_tracks.append(new_track)
                    next_id += 1

                did_yolo_update = True

            elif dets and not tracks_pred_xy:
                for (b, cls_id, conf, ms, area) in dets:
                    if conf < SPAWN_CONF_THR or area < MIN_BOX_AREA or ms < SPAWN_MOTION_THR:
                        continue
                    tracks.append(KalmanTrack(next_id, b, cls_id, conf, frame_idx))
                    next_id += 1
                did_yolo_update = True

        if not did_yolo_update and tracks_pred_xy:
            if backend_ready and backend_blobs is not None:
                blobs = [{
                    "bbox": np.array([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], dtype=np.float32),
                    "cx": float(b["cx"]),
                    "cy": float(b["cy"]),
                    "area": float(b["area"]),
                } for b in backend_blobs]
            else:
                blobs = extract_blobs(fgmask, min_area=800)

            blob_matches = assoc_tracks_to_blobs_strict(
                tracks_pred_xy, tracks, blobs, W, H,
                base=BASE_GATE, k=K_GATE, max_gate=MAX_GATE,
                iou_thr=0.01,
                pad_factor=0.60,
                area_ratio=(0.10, 12.0),
            )

            matched = set()
            for ti, bj in blob_matches:
                if tracks[ti].hits < MIN_HITS_TO_ACTIVATE:
                    continue
                b = blobs[bj]
                tracks[ti].correct_center(b["cx"], b["cy"], frame_idx, conf=0.0)
                matched.add(ti)

            for ti, _ in tracks_pred_xy:
                if ti not in matched:
                    tracks[ti].mark_pred_only()

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

            if remove_track and t.death_frame is None:
                t.death_frame = frame_idx

        tracks = kept

        vis = frame.copy()
        cv2.putText(
            vis,
            f"YOLO:{raw_yolo} tracks:{len(tracks)} lr:{learn_rate:.4f} stride:{args.stride}",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        poly_pts_int = zone.poly_np(int(args.poly))
        if poly_pts_int is not None:
            overlay = vis.copy()
            cv2.fillPoly(overlay, [poly_pts_int], (0, 255, 255))
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            cv2.polylines(vis, [poly_pts_int], True, (0, 255, 255), 2)

        track_canvas = np.full_like(frame, 0)

        for tid, hist in track_flow_history.items():
            draw_history_from_list(vis, hist, thickness=2, min_step=1.0)
            draw_history_from_list(track_canvas, hist, thickness=2, min_step=1.0)

        active_tracks = []
        centers = []

        for t in tracks:
            cx, cy = t.draw_point()

            if t.id not in track_flow_history:
                track_flow_history[t.id] = []
            track_flow_history[t.id].append((cx, cy))

            # Update motion history for status message generation
            t.update_motion_history(cx, cy)

            active_tracks.append((t, cx, cy))
            centers.append((int(round(cx)), int(round(cy))))

        RUN_FLOW_EVERY = 2
        if backend_ready and centers and (frame_idx % RUN_FLOW_EVERY == 0):
            flow_results = backend.analyze_objects(
                centers,
                min_mag=FLOW_MIN_MAG,
                angle_thr_deg=FLOW_ANGLE_THR_DEG,
                mag_ratio_low=FLOW_MAG_RATIO_LOW,
            )
        else:
            flow_results = []

        for idx, (t, cx, cy) in enumerate(active_tracks):
            fut = t.future_path(25)
            entering = False
            inside_now = False

            if poly_pts_int is not None:
                inside_now = zone.inside(cx, cy, poly_pts_int)
                future_inside = zone.future_any_inside(fut, poly_pts_int)
                entering = future_inside and (not inside_now)
                if (entering or inside_now) and (t.id not in zone.violated):
                    zone.violated.add(t.id)

            active = (t.hits >= MIN_HITS_TO_ACTIVATE)
            pred_only = (t.pred_only_streak > 0)

            tb = t.bbox_from_center(cx, cy).astype(int)
            x1, y1, x2, y2 = tb

            if poly_pts_int is not None and (entering or inside_now):
                col = (0, 0, 255)
            else:
                col = (255, 0, 0) if not active else ((0, 200, 255) if pred_only else (0, 255, 0))

            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.circle(vis, (int(cx), int(cy)), 4, col, -1)

            cv2.putText(
                vis,
                f"ID {t.id} hits {t.hits} ymiss {t.yolo_misses} streak {t.pred_only_streak}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2,
            )
            '''
            # Stable red status message only
            msg = t.get_track_change_message(
                dir_window=3,
                min_speed=1.0,
                confirm_frames=3,
                hold_frames=30,
            )

            if msg:
                cv2.putText(
                    vis,
                    msg,
                    (x1, min(H - 10, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            '''
            speed_msg = t.get_start_stop_message(
                speed_window=5,
                stop_thr=0.9,
                move_thr=1.2,
                hold_frames=30,
            )

            if speed_msg:
                cv2.putText(
                    vis,
                    speed_msg,
                    (x1, min(H - 10, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if idx < len(flow_results):
                fr = flow_results[idx]
                draw_kernel_box(vis, fr["kernel_box"], color=(0, 255, 255), thickness=1)
                draw_local_flow_points(
                    vis,
                    fr["points"],
                    scale=FLOW_POINT_DRAW_SCALE,
                    force_red=False
                )

        cv2.imshow("Video", vis)
        # cv2.imshow("Tracked Objects on White Canvas", track_canvas)

        end = time.perf_counter()
        print(f"Execution time: {end - start:.6f} seconds")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            zone.reset()
            print("Reset polygon + violated tracks.")
        elif key == ord('='):
            learn_rate = min(0.05, learn_rate * 1.5)
        elif key == ord('-'):
            learn_rate = max(0.0, learn_rate / 1.5)


    for t in tracks:
        if t.death_frame is None:
            t.death_frame = frame_idx
        

    snapshot_rows = []

    max_frame = frame_idx

    for snap_frame in range(30, max_frame + 1, 30):
        alive_tracks = []

        for t in all_tracks:
            if t.birth_frame <= snap_frame <= t.death_frame:
                alive_tracks.append(t)

        total_tracks = len(alive_tracks)
        true_tracks = 0
        false_tracks = 0
        undecided_tracks = 0

        for t in alive_tracks:
            if t.death_frame >= snap_frame + 15:
                true_tracks += 1
            elif t.death_frame <= snap_frame + 5:
                false_tracks += 1
            else:
                undecided_tracks += 1

        snapshot_rows.append([
            snap_frame,
            total_tracks,
            true_tracks,
            false_tracks,
            undecided_tracks
        ])

    print("\nTrack Persistence Table (every 30th frame)")
    print("Frame\tTotal\tTrue\tFalse\tUndecided")
    for row in snapshot_rows:
        print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}")
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()