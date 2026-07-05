# association.py
import numpy as np
from utils import bbox_center_xyxy, iou_xyxy

def track_gate_px(t, base=60, k=1.5, max_gate=220):
    vx = float(t.kf.statePost[2, 0])
    vy = float(t.kf.statePost[3, 0])
    speed = np.hypot(vx, vy)
    return float(np.clip(base + k * speed, base, max_gate))

def elliptical_gate(pred_xy, det_xy, road_u, gate_long, gate_lat):
    if road_u is None:
        return True, 0.0
    u = road_u / (np.linalg.norm(road_u) + 1e-6)
    u_perp = np.array([-u[1], u[0]], dtype=np.float32)
    d = np.array(det_xy, dtype=np.float32) - np.array(pred_xy, dtype=np.float32)
    d_par = float(d.dot(u))
    d_lat = float(d.dot(u_perp))
    score = (d_par / (gate_long + 1e-6)) ** 2 + (d_lat / (gate_lat + 1e-6)) ** 2
    return (score <= 1.0), score

def direction_cost_from_track(track, road_u, min_speed=0.5):
    if road_u is None:
        return 0.0
    vx = float(track.kf.statePost[2, 0])
    vy = float(track.kf.statePost[3, 0])
    sp = float(np.hypot(vx, vy))
    if sp < min_speed:
        return 0.0
    u = road_u / (np.linalg.norm(road_u) + 1e-6)
    align = float((vx*u[0] + vy*u[1]) / (sp + 1e-6))
    align = max(0.0, align)
    return 1.0 - align

def assoc_by_center_distance_roadaware(
    tracks_pred_xy,
    dets,
    tracks,
    road_u,
    base=60,
    k=1.5,
    max_gate=220,
    lat_ratio=0.35,
    w_dist=0.70,
    w_dir=0.20,
    w_iou=0.10,
    use_class_gate=True,
    class_strict_hits=3,
    class_soft_penalty=0.15,
    iou_thr=0.02,
    conf_min=0.0,
    min_area=0.0,
    max_area_ratio=(0.2, 5.0),
    motion_min=None
):
    if len(tracks_pred_xy) == 0 or len(dets) == 0:
        return [], list(range(len(tracks_pred_xy))), list(range(len(dets)))

    det_centers = [bbox_center_xyxy(d[0]) for d in dets]
    candidates = []

    for pti, (track_index, (px, py)) in enumerate(tracks_pred_xy):
        t = tracks[track_index]
        gate_long = track_gate_px(t, base=base, k=k, max_gate=max_gate)
        gate_lat = max(10.0, lat_ratio * gate_long)

        tb = t.bbox_from_center(px, py)
        tw = max(2.0, float(tb[2]-tb[0]))
        th = max(2.0, float(tb[3]-tb[1]))
        track_area = tw * th

        strict_class = use_class_gate and (t.hits >= class_strict_hits)

        for di, ((b, cls_id, conf, ms, area), (dx, dy)) in enumerate(zip(dets, det_centers)):
            if conf < conf_min or area < min_area:
                continue
            if motion_min is not None and ms < motion_min:
                continue

            dist = float(np.hypot(px - dx, py - dy))
            if dist > gate_long:
                continue

            ok, _ = elliptical_gate((px, py), (dx, dy), road_u, gate_long, gate_lat)
            if not ok:
                continue

            class_pen = 0.0
            if use_class_gate:
                if strict_class and int(cls_id) != int(t.cls_id):
                    continue
                if (not strict_class) and int(cls_id) != int(t.cls_id):
                    class_pen = class_soft_penalty

            if t.hits >= 2:
                iou = iou_xyxy(tb, b)
                if iou < iou_thr:
                    continue
                ar = float(area) / (track_area + 1e-6)
                if ar < max_area_ratio[0] or ar > max_area_ratio[1]:
                    continue
            else:
                iou = 0.0

            dist_norm = dist / (gate_long + 1e-6)
            dir_cost = direction_cost_from_track(t, road_u)
            iou_cost = 1.0 - float(np.clip(iou, 0.0, 1.0))
            cost = (w_dist * dist_norm) + (w_dir * dir_cost) + (w_iou * iou_cost) + class_pen
            candidates.append((cost, pti, di))

    if not candidates:
        return [], list(range(len(tracks_pred_xy))), list(range(len(dets)))

    candidates.sort(key=lambda x: x[0])
    used_t, used_d = set(), set()
    matches = []
    for cost, pti, di in candidates:
        if pti in used_t or di in used_d:
            continue
        matches.append((pti, di))
        used_t.add(pti)
        used_d.add(di)

    um_tracks = [i for i in range(len(tracks_pred_xy)) if i not in used_t]
    um_dets   = [i for i in range(len(dets)) if i not in used_d]
    return matches, um_tracks, um_dets