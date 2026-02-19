import time
import math
from collections import defaultdict

def iou_xyxy(a, b):
    # a,b: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def center_xyxy(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, (x2-x1)) * max(0.0, (y2-y1))

def _filter_dets(dets, conf_keep=0.55, min_area_px=800):
    out = []
    for d in dets:
        label, conf, x1, y1, x2, y2 = d
        if conf < conf_keep:
            continue
        area = max(0.0, (x2-x1)) * max(0.0, (y2-y1))
        if area < min_area_px:
            continue
        out.append(d)
    return out

def match_detections_robust(prev, curr, iou_thresh=0.3, center_dist_thresh = 40.0):
    """
    Greedy matching by IoU within same label.
    Returns list of (prev_det, curr_det) matches, plus unmatched_curr, unmatched_prev
    det format: (label, conf, x1,y1,x2,y2)
    """
    prev_used = set()
    curr_used = set()
    matches = []

    # sort current by confidence desc so best boxes match first
    curr_idx = sorted(range(len(curr)), key=lambda i: curr[i][1], reverse=True)

    for ci in curr_idx:
        clabel, cconf, *cbox = curr[ci]
        cc = center_xyxy(cbox)


        best_pi = None
        best_key = None
        best_iou = 0.0
        best_dist = float("inf")

        for pi in range(len(prev)):
            if pi in prev_used:
                continue
            plabel, pconf, *pbox = prev[pi]
            if plabel != clabel:
                continue
            v_iou = iou_xyxy(pbox, cbox)
            v_dist = dist(center_xyxy(pbox), cc)

            ok_iou = (v_iou >= iou_thresh)
            ok_dist = (v_dist <= center_dist_thresh)

            if not (ok_iou or ok_dist):
                continue

            if v_iou > best_iou or (abs(v_iou - best_iou) < 1e-6 and v_dist < best_dist):
                best_iou = v_iou
                best_dist = v_dist
                best_pi = pi
                best_key = ("iou", best_iou) if ok_iou else ("dist", best_dist)

        if best_pi is not None:
            prev_used.add(best_pi)
            curr_used.add(ci)
            matches.append((prev[best_pi], curr[ci], {"best": best_key, "iou": best_iou, "dist": best_dist}))

    unmatched_curr = [curr[i] for i in range(len(curr)) if i not in curr_used]
    unmatched_prev = [prev[i] for i in range(len(prev)) if i not in prev_used]
    return matches, unmatched_curr, unmatched_prev

def _cell_signature(det, frame_w, frame_h, grid=8):
    label, conf, x1, y1, x2, y2 = det
    cx, cy = center_xyxy((x1, y1, x2, y2))

    cx = max(0.0, min(frame_w-1.0, cx))
    cy = max(0.0, min(frame_h - 1.0, cy))
    cell_x = int((cx / frame_w) * grid)
    cell_y = int((cy / frame_h) * grid)
    cell_x = max(0, min(grid - 1, cell_x))
    cell_y = max(0, min(grid - 1, cell_y))
    return(label, cell_x, cell_y)

def is_frame_important(prev_dets, curr_dets, frame_shape, state, 
                       conf_keep=0.55, min_area_px=800,
                       iou_match_thresh=0.25, center_match_px=40.0,
                       new_persist=2, move_persist=2,
                       min_move_px=12.0, move_rel=0.20,
                       cooldown_s=3.0,
                       sig_grid=8):
    """
    Important if:
      - New object/item detected (unmatched current detection)
      - Any matched object moved more than move_px_thresh pixels (center displacement)
    """
    now = time.time()
    if (now - state.get("last_fire", 0.0)) < cooldown_s:
        return False, {"reason": "cooldown"}

    h, w = frame_shape[:2]

    if "new_counts" not in state:
        state["new_counts"] = defaultdict(int)
    if "move_counts" not in state:
        state["move_counts"] = defaultdict(int)

    # 1) Filter
    prev = _filter_dets(prev_dets or [], conf_keep=conf_keep, min_area_px=min_area_px)
    curr = _filter_dets(curr_dets or [], conf_keep=conf_keep, min_area_px=min_area_px)

    # 2) Match robustly
    matches, unmatched_curr, _ = match_detections_robust(
        prev, curr, iou_thresh=iou_match_thresh, center_dist_thresh=center_match_px
    )

    # --- Debounced NEW OBJECT ---
    # Increment counts for signatures seen this tick
    seen_sigs = set()
    for d in unmatched_curr:
        sig = _cell_signature(d, w, h, grid=sig_grid)
        seen_sigs.add(sig)
        state["new_counts"][sig] += 1

    # Decay counts for signatures not seen this tick (prevents stale buildup)
    # (light decay: drop by 1; remove at 0)
    for sig in list(state["new_counts"].keys()):
        if sig not in seen_sigs:
            state["new_counts"][sig] = max(0, state["new_counts"][sig] - 1)
            if state["new_counts"][sig] == 0:
                del state["new_counts"][sig]

    # Fire if any signature persisted enough
    for sig, cnt in state["new_counts"].items():
        if cnt >= new_persist:
            state["last_fire"] = now
            # reset so it doesn't immediately refire
            state["new_counts"][sig] = 0
            return True, {"reason": "new_object", "sig": sig, "persist": cnt, "count_new": len(unmatched_curr)}

    # --- Debounced MOVEMENT ---
    moved_labels = set()
    for pdet, cdet, _meta in matches:
        plabel, pconf, *pbox = pdet
        clabel, cconf, *cbox = cdet

        pc = center_xyxy(pbox)
        cc = center_xyxy(cbox)
        dpx = dist(pc, cc)

        area = box_area_xyxy(cbox)
        # threshold scales with object size; avoids jitter triggering movement
        thresh = max(min_move_px, move_rel * math.sqrt(max(area, 1.0)))

        if dpx >= thresh:
            moved_labels.add(clabel)

    # increment move counts for moved labels; decay others
    for lbl in moved_labels:
        state["move_counts"][lbl] += 1

    for lbl in list(state["move_counts"].keys()):
        if lbl not in moved_labels:
            state["move_counts"][lbl] = max(0, state["move_counts"][lbl] - 1)
            if state["move_counts"][lbl] == 0:
                del state["move_counts"][lbl]

    for lbl, cnt in state["move_counts"].items():
        if cnt >= move_persist:
            state["last_fire"] = now
            state["move_counts"][lbl] = 0
            return True, {"reason": "moved", "label": lbl, "persist": cnt}

    return False, {"reason": "no_change"}
