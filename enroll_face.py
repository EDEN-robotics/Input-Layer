"""
enroll_face.py — capture reference images for face recognition.

Usage:
    python enroll_face.py --name Krish           # webcam capture
    python enroll_face.py --name Krish --source path/to/photos/  # import folder

Webcam controls:
    SPACE  — capture current frame
    Q      — quit (or auto-quits after --count frames are saved)
"""

import argparse
import os
import shutil
import sys
import threading
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

FACES_DIR = "faces"
DEFAULT_COUNT = 20


def save_from_folder(name: str, source_dir: str):
    out_dir = os.path.join(FACES_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(exts)]
    if not files:
        print(f"No image files found in {source_dir}")
        sys.exit(1)

    mtcnn = MTCNN(keep_all=False, min_face_size=40)
    saved = 0
    for fname in files:
        src = os.path.join(source_dir, fname)
        img = Image.open(src).convert("RGB")
        face = mtcnn(img)
        if face is None:
            print(f"  skip (no face): {fname}")
            continue
        dst = os.path.join(out_dir, f"{name}_{saved:04d}.jpg")
        shutil.copy2(src, dst)
        saved += 1
        print(f"  saved: {dst}")

    print(f"\nDone — {saved}/{len(files)} images saved to {out_dir}/")


def capture_from_webcam(name: str, count: int):
    out_dir = os.path.join(FACES_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading face detector...")
    mtcnn = MTCNN(keep_all=False, min_face_size=40)
    print("Done. Opening webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    print(f"\nWebcam open — saving to: {out_dir}/")
    print(f"Target: {count} images")
    print("Controls: SPACE = capture  |  Q = quit\n")

    # Detection state shared between threads
    det_lock = threading.Lock()
    det_boxes = []        # list of (x1,y1,x2,y2)
    det_face_found = False
    detecting = threading.Event()

    latest_frame = [None]
    frame_lock = threading.Lock()

    def detect_loop():
        nonlocal det_boxes, det_face_found
        while not stop.is_set():
            detecting.wait()
            detecting.clear()
            with frame_lock:
                f = latest_frame[0]
            if f is None:
                continue
            pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            boxes, probs = mtcnn.detect(pil)
            found_boxes = []
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob is not None and prob >= 0.85:
                        found_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            with det_lock:
                det_boxes = found_boxes
                det_face_found = len(found_boxes) > 0

    stop = threading.Event()
    det_thread = threading.Thread(target=detect_loop, daemon=True)
    det_thread.start()

    saved = 0
    frame_idx = 0

    while saved < count:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_idx += 1

        # Feed a new frame to the detector every 3 frames (keeps it busy without flooding)
        if frame_idx % 3 == 0 and not detecting.is_set():
            with frame_lock:
                latest_frame[0] = frame.copy()
            detecting.set()

        # Draw last known detection results (never blocks display)
        display = frame.copy()
        with det_lock:
            boxes_now = list(det_boxes)
            face_ok = det_face_found

        for (x1, y1, x2, y2) in boxes_now:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "face", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        bar = "#" * saved + "-" * (count - saved)
        cv2.putText(display, f"[{bar}] {saved}/{count}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        hint = "FACE DETECTED — press SPACE" if face_ok else "No face — move closer or adjust lighting"
        hint_color = (0, 255, 0) if face_ok else (0, 100, 255)
        cv2.putText(display, hint, (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hint_color, 2)

        cv2.imshow(f"Enroll: {name}", display)

        key = cv2.waitKey(16) & 0xFF   # ~60 fps display
        if key == ord("q"):
            print("Quit.")
            break
        if key == ord(" "):
            with det_lock:
                ok = det_face_found
            if not ok:
                print("  No face detected — skipping, reposition and try again.")
                continue
            dst = os.path.join(out_dir, f"{name}_{saved:04d}.jpg")
            cv2.imwrite(dst, frame)
            saved += 1
            print(f"  [{saved}/{count}] saved: {dst}")

    stop.set()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone — {saved} image(s) saved to {out_dir}/")
    if saved < 5:
        print("Tip: fewer than 5 images may reduce accuracy. Run again to add more.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name",   required=True, help="Person's name (e.g. Krish)")
    ap.add_argument("--source", default=None,  help="Import from folder instead of webcam")
    ap.add_argument("--count",  type=int, default=DEFAULT_COUNT,
                    help=f"Number of webcam captures (default {DEFAULT_COUNT})")
    args = ap.parse_args()

    if args.source:
        save_from_folder(args.name, args.source)
    else:
        capture_from_webcam(args.name, args.count)


if __name__ == "__main__":
    main()
