import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import ollama
import time

MODEL = "llama3.2-vision:11b"   # your local VLM
URI = "ws://127.0.0.1:8765"     # WebSocket server address

async def analyze_with_vlm(frame):
    """Send a frame to LLaMA Vision for description or reasoning."""
    # Encode the frame as base64 (VLM expects this)
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")

    prompt = (
        "You are a visual analysis model watching a live feed, describe what is happening as you see frame"
        "by frame. try your best to string the scenario together as this is a video feed split up into very select"
        "frames so it does not tell the whole picture"
        "List the main objects, their relationships, and any notable actions."
    )

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [frame_b64]
                }
            ],
        )
        text = response["message"]["content"]
        return text.strip()
    except Exception as e:
        print(f"[client] VLM error: {e}")
        return None


async def listen():
    async with websockets.connect(URI, max_size=None) as websocket:
        print("[client] Connected to server")

        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                print(f"[client] JSON decode error: {e}")
                continue

            meta = data.get("metadata", {})
            frame_b64 = data.get("frame")

            if not frame_b64:
                print("[client] No frame data found")
                continue

            try:
                img_bytes = base64.b64decode(frame_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[client] Frame decode error: {e}")
                continue

            if frame is None:
                print("[client] Decoded frame is None — skipping")
                continue

            # --- Print detections from YOLO ---
            motion = meta.get("motion_score", 0)
            detections = meta.get("detections", [])
            print(f"\n[client] Important Frame | Motion={motion:.2f} | Objects={len(detections)}")

            for det in detections:
                cls = det.get("class", "unknown")
                conf = det.get("confidence", 0)
                bbox = det.get("bbox", [])
                print(f"   → {cls} ({conf:.2f})")

                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls} {conf:.2f}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- Run VLM analysis (LLaMA Vision) ---
            print("[client] Sending frame to LLaMA Vision for analysis...")
            vlm_start = time.time()
            vlm_text = await analyze_with_vlm(frame)
            vlm_end = time.time()

            if vlm_text:
                print(f"[VLM] ({vlm_end - vlm_start:.1f}s): {vlm_text}")
            else:
                print("[VLM] ❌ No response or error")

            # --- Show frame with overlay ---
            cv2.imshow("Client Feed (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(listen())
