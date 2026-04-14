import os
import cv2
import torch
import numpy as np
from PIL import Image
from collections import deque, defaultdict
from facenet_pytorch import MTCNN, InceptionResnetV1


# ── Simple per-face tracker for temporal voting ───────────────────────────────
class _Track:
    """Remembers the last `window` recognition results for one face position."""
    MATCH_PX = 80   # max center-distance to consider same track

    def __init__(self, center, window=10):
        self.center  = center           # (x, y)
        self.history = deque(maxlen=window)

    def update(self, center, name):
        self.center = center
        self.history.append(name)

    def voted_name(self, min_ratio=0.60):
        """Return name only if it won >= min_ratio of recent frames."""
        named = [n for n in self.history if n is not None]
        if not named:
            return None
        counts = defaultdict(int)
        for n in named:
            counts[n] += 1
        best, best_n = max(counts.items(), key=lambda x: x[1])
        if best_n / len(self.history) >= min_ratio:
            return best
        return None

    def dist(self, center):
        return ((self.center[0] - center[0]) ** 2 +
                (self.center[1] - center[1]) ** 2) ** 0.5


class FaceRecognizer:
    """
    Identifies people in YOLO 'person' detections by matching faces
    against a library of reference images in faces/{Name}/*.jpg.

    Guardrails applied:
      - High similarity threshold (default 0.68)
      - Minimum margin above threshold (default 0.08) — match must clearly win
      - Temporal voting across last 10 frames — name only sticks if seen in
        >= 60% of recent frames, preventing single-frame false positives

    Usage:
        rec = FaceRecognizer(faces_dir="faces", device="cuda")
        updated_dets = rec.recognize_in_frame(frame_bgr, yolo_dets)
    """

    def __init__(self, faces_dir="faces", device="cpu",
                 sim_threshold=0.68, margin=0.08,
                 vote_window=10, vote_ratio=0.60):
        self.device        = device
        self.sim_threshold = sim_threshold
        self.margin        = margin        # must beat threshold by at least this
        self.vote_window   = vote_window
        self.vote_ratio    = vote_ratio
        self.faces_dir     = faces_dir

        self.mtcnn = MTCNN(
            image_size=160, margin=20, keep_all=True,
            min_face_size=40, device=device, post_process=True,
        )
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

        self.known_names: list[str] = []
        self.known_embeddings: torch.Tensor | None = None
        self._tracks: list[_Track] = []

        if os.path.isdir(faces_dir):
            self._load_known_faces()
        else:
            print(f"[FaceRecog] '{faces_dir}' not found — recognition disabled.")

    # ------------------------------------------------------------------
    def _load_known_faces(self):
        names, embeddings = [], []

        for name in sorted(os.listdir(self.faces_dir)):
            person_dir = os.path.join(self.faces_dir, name)
            if not os.path.isdir(person_dir):
                continue
            count = 0
            for fname in os.listdir(person_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fpath = os.path.join(person_dir, fname)
                try:
                    img  = Image.open(fpath).convert("RGB")
                    face = self.mtcnn(img)
                    if face is None:
                        continue
                    if face.ndim == 3:
                        face = face.unsqueeze(0)
                    with torch.no_grad():
                        emb = self.resnet(face[:1].to(self.device)).cpu()
                    embeddings.append(emb)
                    names.append(name)
                    count += 1
                except Exception as exc:
                    print(f"[FaceRecog] skip {fname}: {exc}")
            if count:
                print(f"[FaceRecog] {name}: {count} embedding(s)")

        if embeddings:
            self.known_names      = names
            self.known_embeddings = torch.cat(embeddings, dim=0)
            self._build_centroids()
            print(f"[FaceRecog] Ready — {len(names)} embedding(s) for: {sorted(set(names))}")
            if len(set(names)) < 2:
                print("[FaceRecog] WARNING: only one person enrolled. "
                      "False positives are likely — run enroll_face.py for others too.")
        else:
            print("[FaceRecog] No usable face images found.")

    # ------------------------------------------------------------------
    def _build_centroids(self):
        """
        Average all embeddings per person into one centroid vector, then
        compute a per-person acceptance radius from how tightly the enrolled
        images cluster around that centroid.

        Rejection rule: incoming similarity < (mean - radius_k * std)
        → classified as 'unknown' even if it's the best match.
        """
        per_person: dict[str, list[torch.Tensor]] = defaultdict(list)
        for name, emb in zip(self.known_names, self.known_embeddings):
            per_person[name].append(emb)

        self._centroid_names: list[str] = []
        centroid_list: list[torch.Tensor] = []
        self._reject_below: dict[str, float] = {}   # name -> min acceptable similarity

        for name, embs in sorted(per_person.items()):
            stack    = torch.stack(embs)
            centroid = stack.mean(0)
            centroid = centroid / centroid.norm().clamp(min=1e-8)
            centroid_list.append(centroid)
            self._centroid_names.append(name)

            # Cosine similarity of each enrolled image to its centroid
            stack_n  = stack / stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
            sims     = (stack_n @ centroid).tolist()
            mean_s   = sum(sims) / len(sims)
            var_s    = sum((s - mean_s) ** 2 for s in sims) / max(len(sims) - 1, 1)
            std_s    = var_s ** 0.5

            # Reject anything below mean - 2*std (covers ~97% of genuine matches)
            reject_below = mean_s - 2.0 * std_s
            self._reject_below[name] = reject_below

            print(f"[FaceRecog] '{name}': {len(embs)} image(s), "
                  f"cluster sim {mean_s:.3f}±{std_s:.3f}, "
                  f"reject below {reject_below:.3f}")

        self._centroids = torch.stack(centroid_list)   # (num_people, 512)

    # ------------------------------------------------------------------
    def _get_or_create_track(self, center) -> _Track:
        best, best_d = None, float("inf")
        for t in self._tracks:
            d = t.dist(center)
            if d < best_d:
                best_d, best = d, t
        if best is not None and best_d <= _Track.MATCH_PX:
            return best
        t = _Track(center, window=self.vote_window)
        self._tracks.append(t)
        return t

    def _prune_tracks(self, active_centers):
        """Drop tracks whose last known position is far from any current face."""
        if not active_centers:
            self._tracks.clear()
            return
        self._tracks = [
            t for t in self._tracks
            if any(t.dist(c) <= _Track.MATCH_PX for c in active_centers)
        ]

    # ------------------------------------------------------------------
    def recognize_in_frame(self, frame_bgr: np.ndarray, yolo_dets: list) -> list:
        """
        Returns yolo_dets with 'person' labels replaced by the recognised
        name, or kept as 'person' if unknown / not detected / vote not yet
        confident enough.
        """
        if not self.known_names:
            return yolo_dets

        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # ── 1. Detect faces ───────────────────────────────────────────
        boxes, probs = self.mtcnn.detect(pil_img)
        if boxes is None:
            self._prune_tracks([])
            return yolo_dets

        # ── 2. Build face crops + embeddings ─────────────────────────
        h, w   = frame_bgr.shape[:2]
        face_tensors, face_boxes, face_centers = [], [], []

        for box, prob in zip(boxes, probs):
            if prob is None or prob < 0.90:
                continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            crop = pil_img.crop((x1, y1, x2, y2)).resize((160, 160))
            t = torch.from_numpy(np.array(crop)).permute(2, 0, 1).float() / 255.0
            t = (t - 0.5) / 0.5
            face_tensors.append(t)
            face_boxes.append((x1, y1, x2, y2))
            face_centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        self._prune_tracks(face_centers)

        if not face_tensors:
            return yolo_dets

        batch = torch.stack(face_tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.resnet(batch).cpu()

        # ── 3. Centroid similarity + guardrails ───────────────────────
        # Compare each face against per-person centroids (not individual images).
        # This means the margin gap is always between two *different people*,
        # making it a real discriminative comparison.
        emb_n    = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # _centroids are already L2-normalised
        sims     = emb_n @ self._centroids.T   # (M faces, num_people)

        face_voted: list[str | None] = []
        num_people = len(self._centroid_names)

        for mi, center in enumerate(face_centers):
            row = sims[mi]                          # (num_people,)
            sorted_sims, sorted_idx = row.sort(descending=True)
            best_sim  = sorted_sims[0].item()
            best_name = self._centroid_names[sorted_idx[0].item()]

            # Guardrail A — flat threshold
            if best_sim < self.sim_threshold:
                raw_name = None

            # Guardrail B — must fall within this person's enrolled cluster radius
            elif best_sim < self._reject_below.get(best_name, 0.0):
                raw_name = None   # outside cluster → unknown

            # Guardrail C — margin vs second-best *person* (only meaningful with 2+)
            elif num_people >= 2:
                second_sim = sorted_sims[1].item()
                gap = best_sim - second_sim
                raw_name = best_name if gap >= self.margin else None

            else:
                raw_name = best_name

            # Guardrail C — temporal vote
            track = self._get_or_create_track(center)
            track.update(center, raw_name)
            face_voted.append(track.voted_name(min_ratio=self.vote_ratio))

        # ── 4. Associate faces with YOLO person boxes ─────────────────
        updated = []
        for det in yolo_dets:
            label, conf, px1, py1, px2, py2 = det
            if label != "person":
                updated.append(det)
                continue

            best_name: str | None = None
            best_area = 0.0
            for fi, (fx1, fy1, fx2, fy2) in enumerate(face_boxes):
                fc_x = (fx1 + fx2) / 2.0
                fc_y = (fy1 + fy2) / 2.0
                if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                    area = (fx2 - fx1) * (fy2 - fy1)
                    if area > best_area:
                        best_area = area
                        best_name = face_voted[fi]

            # best_name=None means face detected but rejected → label "unknown"
            # no face found in person box → keep "person"
            if best_name is not None:
                new_label = best_name
            elif any(px1 <= (fx1+fx2)/2 <= px2 and py1 <= (fy1+fy2)/2 <= py2
                     for fx1, fy1, fx2, fy2 in face_boxes):
                new_label = "unknown"   # face seen but didn't pass guardrails
            else:
                new_label = "person"    # no face detected in this box at all
            updated.append((new_label, conf, px1, py1, px2, py2))

        return updated
