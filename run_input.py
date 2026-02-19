import cv2
import torch
import threading
import time
from collections import defaultdict, deque
from queue import Queue, Empty
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
from ultralytics import YOLO
from PIL import Image
from is_important import is_frame_important
import tempfile
import os
import moondream as md

# ── Config ──────────────────────────────────────────────────────────────────
YOLO_PATH        = "C:/Users/dayse/VSCODE files/Projects/Yolov11_PL/Models/YOLOv11obj_model.pt"  
SEND_EVERY_S     = 8.0       
DETECT_EVERY_S   = 0.08      
CONF_THRESH      = 0.35
device           = "cuda" if torch.cuda.is_available() else "cpu"

# ── Queues ───────────────────────────────────────────────────────────────────
raw_q       = Queue(maxsize=1)   
vlm_q       = Queue(maxsize=1)   
result_q    = Queue()            

stop_event  = threading.Event()

# ── Models ───────────────────────────────────────────────────────────────────
print("Loading YOLO...")
yolo = YOLO(YOLO_PATH).to(device)

# ── Helper ───────────────────────────────────────────────────────────────────
def cv2_to_pil(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

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
        
        if not raw_q.full():
            raw_q.put(frame)
    cap.release()

# ── Thread 2: YOLO + Importance ───────────────────────────────────────────────
def detection_thread():
    prev_dets       = []
    last_sent_time  = 0.0
    vlm_count       = 0
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

        important, info = is_frame_important(
            prev_dets, curr_dets,
            frame_shape       = frame.shape,
            state             = importance_state,
            conf_keep         = 0.55,
            min_area_px       = 800,
            iou_match_thresh  = 0.25,
            center_match_px   = 40.0,
            new_persist       = 2,
            move_persist      = 2,
            cooldown_s        = 3.0,
        )

        time_up = (now - last_sent_time) >= SEND_EVERY_S

        if (important or time_up) and not vlm_q.full():
            vlm_count += 1
            vlm_q.put({
                "frame":    frame.copy(),
                "reason":   "important" if important else "timer",
                "info":     info,
                "vlm_id":   vlm_count,
                "ts":       now,
            })
            last_sent_time = now

        prev_dets = curr_dets

# ── Thread 3: VLM Inference ───────────────────────────────────────────────────
def vlm_thread():
    print("Loading Moondream...")
    moon_model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)
    print("Moondream ready.")

    while not stop_event.is_set():
        try:
            job = vlm_q.get(timeout=0.5)
        except Empty:
            continue

        image = cv2_to_pil(job["frame"])
        enc_image = moon_model.encode_image(image)
        text = moon_model.answer_question(enc_image, "Briefly describe what you see.", tokenizer)

        result_q.put({
            "text":   text,
            "reason": job["reason"],
            "info":   job["info"],
            "vlm_id": job["vlm_id"],
        })

# ── Thread 4: Print Results ───────────────────────────────────────────────────
def printer_thread():
    while not stop_event.is_set():
        try:
            r = result_q.get(timeout=0.5)
        except Empty:
            continue
        print(f"\n[VLM #{r['vlm_id']}] ({r['reason']}) {r['info']}")
        print(f"  → {r['text']}\n")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threads = [
        threading.Thread(target=capture_thread,   daemon=True, name="Capture"),
        threading.Thread(target=detection_thread, daemon=True, name="Detection"),
        threading.Thread(target=vlm_thread,        daemon=True, name="VLM"),
        threading.Thread(target=printer_thread,    daemon=True, name="Printer"),
    ]
    for t in threads:
        t.start()
    print("Pipeline running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
    for t in threads:
        t.join(timeout=3)
    print("Done.")
