import asyncio
import websockets
import json
import cv2
import base64
import time
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch

# ---------------- CONFIG ----------------
URI = "ws://127.0.0.1:8765"
SEND_INTERVAL = 0.4   
MOTION_THRESHOLD = 30  
MEMORY_SIZE = 4        
CONF_THRESHOLD = 0.4  
# ----------------------------------------

connected_clients = set()

# Load YOLO model (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[server] Loading YOLO model on {device.upper()}...")
model = YOLO("Eden/Input-Layer/YOLOv11obj_model.pt").to(device)
print("[server] Model ready.")

# Keep a small buffer of previous grayscale frames for motion detection
frame_history = deque(maxlen=MEMORY_SIZE)


# ---------- Motion / Importance Detection ----------
def is_important_frame(frame, detections):
    """
    Decide whether this frame is important enough to send.
    Criteria: motion magnitude + high-confidence detections.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_history.append(gray)

    # Compute motion intensity (vs. previous frame)
    motion_score = 0
    if len(frame_history) > 1:
        diff = cv2.absdiff(frame_history[-1], frame_history[-2])
        motion_score = np.mean(diff)

    # Check YOLO detection confidence
    has_strong_detection = any(float(b.conf) > CONF_THRESHOLD for b in detections)

    importance = (motion_score > MOTION_THRESHOLD) or has_strong_detection
    return importance, motion_score


# ---------- WebSocket Handler ----------
async def handler(websocket, path=None):
    connected_clients.add(websocket)
    print(f"[server] Client connected ({len(connected_clients)} total)")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print("[server] Client disconnected.")

# ---------- Frame Capture + Sender ----------
async def frame_sender():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[server] Error: Cannot open camera.")
        return

    last_send = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.05)
            continue

        # Run YOLO inference
        try:
            results = model(frame, verbose=False)
            detections = results[0].boxes if hasattr(results[0], "boxes") else []
        except Exception as e:
            print(f"[server] YOLO inference error: {e}")
            await asyncio.sleep(0.1)
            continue

        important, motion_score = is_important_frame(frame, detections)

        # Send only if frame is "important" and interval passed
        if important and (time.time() - last_send) > SEND_INTERVAL:
            last_send = time.time()

            # Prepare JSON metadata
            json_data = {
                "timestamp": time.time(),
                "motion_score": motion_score,
                "detections": [],
            }

            for box in detections:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                json_data["detections"].append({
                    "class": model.names.get(cls, str(cls)),
                    "confidence": conf,
                    "bbox": xyxy,
                })

            # Encode frame to Base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "metadata": json_data,
                "frame": frame_b64
            }

            data = json.dumps(payload)

            async def broadcaster(queue: asyncio.Queue):
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if connected_clients:
                        msg = json.dumps(item)
                        print(f"[server] Broadcasting frame {item.get('frame_id')} with {len(item.get('detections', []))} detections")  # 👈 add this
                        await asyncio.gather(*(client.send(msg) for client in connected_clients), return_exceptions=True)
                    queue.task_done()

            # Broadcast to all connected clients
            print(f"[server] Important frame detected — sending to {len(connected_clients)} clients")
            for ws in list(connected_clients):
                try:
                    await ws.send(data)
                except Exception as e:
                    print(f"[server] Send error: {e}")
                    connected_clients.discard(ws)

        await asyncio.sleep(0.01)


# ---------- Main Entrypoint ----------
async def main():
    # Start the WebSocket server
    server = await websockets.serve(handler, "127.0.0.1", 8765)
    print("[server] Running on ws://127.0.0.1:8765")

    # Run frame sender concurrently
    sender_task = asyncio.create_task(frame_sender())

    # Keep the server alive
    await server.wait_closed()
    await sender_task

if __name__ == "__main__":
    asyncio.run(main())
