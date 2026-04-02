import cv2
import torch
import threading
import time
from collections import defaultdict
from queue import Queue, Empty
from ultralytics import YOLO
from is_important import is_frame_important
from face_recog import FaceRecognizer

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_PATH       = "yolo11n.pt"   # auto-downloaded on first run (~5MB)
REPORT_EVERY_S  = 8.0
DETECT_EVERY_S  = 0.08
CONF_THRESH     = 0.35
FACES_DIR       = "faces"        # faces/{Name}/*.jpg — set to None to skip
FACE_SIM_THRESH = 0.68
device          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Queues ────────────────────────────────────────────────────────────────────
raw_q    = Queue(maxsize=1)
result_q = Queue()

stop_event = threading.Event()

# ── Models ────────────────────────────────────────────────────────────────────
print("Loading YOLO...")
yolo = YOLO(YOLO_PATH).to(device)

print("Loading face recognizer...")
face_recognizer = (
    FaceRecognizer(faces_dir=FACES_DIR, device=device, sim_threshold=FACE_SIM_THRESH)
    if FACES_DIR else None
)

# Names we can actually recognise — used to colour boxes differently
known_names = set(face_recognizer.known_names) if face_recognizer else set()

# ── Shared display state (written by detection, read by main thread) ──────────
_display_lock  = threading.Lock()
_display_state = {"frame": None, "dets": []}

def _set_display(frame, dets):
    with _display_lock:
        _display_state["frame"] = frame
        _display_state["dets"]  = list(dets)

def _get_display():
    with _display_lock:
        return _display_state["frame"], list(_display_state["dets"])

# ── Helper ────────────────────────────────────────────────────────────────────
def run_yolo(frame):
    results = yolo.predict(frame, conf=CONF_THRESH, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    dets = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf  = float(b.conf[0].item())
        cls   = int(b.cls[0].item())
        label = r.names.get(cls, str(cls))
        dets.append((label, conf, x1, y1, x2, y2))
    return dets

def format_labels(dets):
    counts = defaultdict(int)
    for label, *_ in dets:
        counts[label] += 1
    parts = []
    for label, n in sorted(counts.items()):
        parts.append(label if n == 1 else f"{label} (x{n})")
    return ", ".join(parts) if parts else "nothing"

def box_color(label):
    if label in known_names:
        return (0, 255, 0)      # green  — recognised person
    if label == "unknown":
        return (0, 60, 255)     # red    — face seen but not recognised
    if label == "person":
        return (0, 200, 255)    # yellow — person, no face detected
    return (255, 180, 0)        # blue   — object

def draw_dets(frame, dets):
    out = frame.copy()
    for label, conf, x1, y1, x2, y2 in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = box_color(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out

# ── Thread 1: Capture ─────────────────────────────────────────────────────────
def capture_thread():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        # Always update display with the latest raw frame so preview is smooth
        frame_copy = frame.copy()
        with _display_lock:
            if _display_state["frame"] is None:
                _display_state["frame"] = frame_copy
        if not raw_q.full():
            raw_q.put(frame_copy)
    cap.release()

# ── Thread 2: YOLO + Face + Importance ───────────────────────────────────────
def detection_thread():
    prev_dets      = []
    last_sent_time = 0.0
    report_count   = 0
    importance_state = {
        "last_fire":   0.0,
        "new_counts":  defaultdict(int),
        "move_counts": defaultdict(int),
    }
    next_det = 0.0

    while not stop_event.is_set():
        try:
            frame = raw_q.get(timeout=0.5)
        except Empty:
            continue

        now = time.time()
        if now < next_det:
            continue
        next_det = now + DETECT_EVERY_S

        curr_dets = run_yolo(frame)
        if face_recognizer is not None:
            curr_dets = face_recognizer.recognize_in_frame(frame, curr_dets)

        # Push latest annotated frame to display
        _set_display(frame, curr_dets)

        important, info = is_frame_important(
            prev_dets, curr_dets,
            frame_shape      = frame.shape,
            state            = importance_state,
            conf_keep        = 0.55,
            min_area_px      = 800,
            iou_match_thresh = 0.25,
            center_match_px  = 40.0,
            new_persist      = 2,
            move_persist     = 2,
            cooldown_s       = 3.0,
        )

        time_up = (now - last_sent_time) >= REPORT_EVERY_S

        if important or time_up:
            report_count += 1
            result_q.put({
                "id":     report_count,
                "reason": "important" if important else "timer",
                "info":   info,
                "dets":   curr_dets,
                "ts":     now,
            })
            last_sent_time = now

        prev_dets = curr_dets

# ── Thread 3: Print Results ───────────────────────────────────────────────────
def printer_thread():
    while not stop_event.is_set():
        try:
            r = result_q.get(timeout=0.5)
        except Empty:
            continue
        ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        labels = format_labels(r["dets"])
        print(f"[#{r['id']:04d} {ts}] ({r['reason']}) {labels}")

# ── Main — display loop runs here (OpenCV needs main thread on Windows) ───────
if __name__ == "__main__":
    threads = [
        threading.Thread(target=capture_thread,   daemon=True, name="Capture"),
        threading.Thread(target=detection_thread, daemon=True, name="Detection"),
        threading.Thread(target=printer_thread,   daemon=True, name="Printer"),
    ]
    for t in threads:
        t.start()
    print("Pipeline running. Press Q in the preview window or Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            frame, dets = _get_display()
            if frame is not None:
                annotated = draw_dets(frame, dets)
                cv2.imshow("EDEN — Input Layer", annotated)

            key = cv2.waitKey(16) & 0xFF   # ~60 fps
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    print("\nStopping...")
    stop_event.set()
    cv2.destroyAllWindows()
    for t in threads:
        t.join(timeout=3)
    print("Done.")
