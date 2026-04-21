"""
Parking Lot Demo
Runs vehicle detection on a recorded video, tracks entry/exit zone crossings,
and displays a live annotated window with a running counter.

Usage:
    python src/demo.py

Press 'q' to quit, SPACE to pause/resume.
"""

import cv2
import json
import math
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ── Configuration ─────────────────────────────────────────────────────────────

VIDEO_PATH   = Path("data/raw/2in2outTest.MOV")
ZONES_JSON   = Path("data/zones.json")
MODEL_PATH   = "yolov8n.pt"

CONFIDENCE        = 0.4    # YOLO detection confidence threshold
MAX_TRACK_DIST    = 80     # Max pixels a centroid can move between frames (same vehicle)
MAX_MISSING       = 8      # Frames a track can disappear before being dropped
LOT_CAPACITY      = 20     # Starting number of available spots
FRAME_SKIP        = 4      # Run YOLO every Nth frame; display every frame at full speed
ZONE_COOLDOWN     = 30     # Frames to block re-firing after an entry or exit event

VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

# ── Zone Loading ───────────────────────────────────────────────────────────────

def load_zones(path: Path) -> tuple[list, list]:
    """Load entry and exit zone polygons from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Zones file not found: {path}\n"
            f"Run zone_setup.py first to define your entry/exit zones."
        )
    with open(path) as f:
        data = json.load(f)

    entry = data.get("entry", [])
    exit_ = data.get("exit", [])
    return entry, exit_


# ── Vehicle Detection ──────────────────────────────────────────────────────────

def detect_vehicles(model: YOLO, frame: np.ndarray) -> list[dict]:
    """Run YOLO on a frame and return vehicle bounding boxes."""
    results = model.predict(frame, conf=CONFIDENCE, verbose=False)[0]

    detections = []
    for box in results.boxes:
        if int(box.cls) not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": float(box.conf),
            # Bottom-center anchors to the ground, much better for side-angle cameras
            "center": ((x1 + x2) // 2, y2),
        })

    return detections


# ── Centroid Tracker ───────────────────────────────────────────────────────────

class CentroidTracker:
    """
    Tracks vehicles across frames by matching centroids.
    Fires entry/exit events when a vehicle transitions into a zone.
    """

    def __init__(self, entry_zone: list, exit_zone: list):
        self.entry_zone = entry_zone
        self.exit_zone  = exit_zone

        self.next_id = 0
        self.tracks  = {}   # track_id -> {"center": (x,y), "in_entry": bool, "in_exit": bool, "missing": int}

        self.entry_count = 0
        self.exit_count  = 0

        # Cooldown: frame index of last event per zone, blocks double-counting
        self.frame_index        = 0
        self.entry_last_fired   = -ZONE_COOLDOWN
        self.exit_last_fired    = -ZONE_COOLDOWN

    def update(self, detections: list[dict]) -> list[str]:
        """
        Match detections to existing tracks, update zone states,
        and return a list of events fired this frame ("entry" or "exit").
        """
        events = []
        unmatched_ids = set(self.tracks.keys())
        self.frame_index += 1

        for det in detections:
            center = det["center"]
            track_id = self._match(center, unmatched_ids)

            if track_id is None:
                # New vehicle - create track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "center":   center,
                    "in_entry": self._in_zone(center, self.entry_zone),
                    "in_exit":  self._in_zone(center, self.exit_zone),
                    "missing":  0,
                }
            else:
                unmatched_ids.discard(track_id)
                track = self.tracks[track_id]

                # Check zone transitions
                now_in_entry = self._in_zone(center, self.entry_zone)
                now_in_exit  = self._in_zone(center, self.exit_zone)

                if now_in_entry and not track["in_entry"]:
                    if self.frame_index - self.entry_last_fired > ZONE_COOLDOWN:
                        self.entry_count += 1
                        self.entry_last_fired = self.frame_index
                        events.append("entry")
                        print(f"[EVENT] Entry detected  → entries={self.entry_count}")
                    else:
                        print(f"[SKIP]  Entry blocked by cooldown (frame {self.frame_index})")

                if now_in_exit and not track["in_exit"]:
                    if self.frame_index - self.exit_last_fired > ZONE_COOLDOWN:
                        self.exit_count += 1
                        self.exit_last_fired = self.frame_index
                        events.append("exit")
                        print(f"[EVENT] Exit detected   → exits={self.exit_count}")
                    else:
                        print(f"[SKIP]  Exit blocked by cooldown (frame {self.frame_index})")

                track["center"]   = center
                track["in_entry"] = now_in_entry
                track["in_exit"]  = now_in_exit
                track["missing"]  = 0

        # Age out unmatched tracks
        for track_id in list(unmatched_ids):
            self.tracks[track_id]["missing"] += 1
            if self.tracks[track_id]["missing"] > MAX_MISSING:
                del self.tracks[track_id]

        return events

    def _match(self, center: tuple, candidates: set) -> int | None:
        """Return the closest track ID within MAX_TRACK_DIST, or None."""
        best_id   = None
        best_dist = MAX_TRACK_DIST

        for track_id in candidates:
            dist = math.dist(center, self.tracks[track_id]["center"])
            if dist < best_dist:
                best_dist = dist
                best_id   = track_id

        return best_id

    @staticmethod
    def _in_zone(point: tuple, polygon: list) -> bool:
        """Check if a point is inside a polygon."""
        if not polygon:
            return False
        contour = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(contour, point, False) >= 0


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_zone(frame: np.ndarray, polygon: list, color: tuple, label: str) -> np.ndarray:
    """Draw a filled semi-transparent zone polygon with label."""
    if not polygon:
        return frame

    pts     = np.array(polygon, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    cx = sum(p[0] for p in polygon) // len(polygon)
    cy = sum(p[1] for p in polygon) // len(polygon)
    cv2.putText(frame, label, (cx - 25, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return frame


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and confidence scores."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(frame, f"{det['confidence']:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
    return frame


def draw_tracks(frame: np.ndarray, tracks: dict) -> np.ndarray:
    """Draw centroid dots and track IDs."""
    for track_id, track in tracks.items():
        cx, cy = track["center"]
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"T{track_id}", (cx + 7, cy - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return frame


def draw_hud(frame: np.ndarray, entries: int, exits: int, available: int, paused: bool) -> np.ndarray:
    """Draw the info overlay in the top-left corner."""
    cv2.rectangle(frame, (10, 10), (280, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Entries:   {entries}",   (20, 38),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),   2)
    cv2.putText(frame, f"Exits:     {exits}",     (20, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),   2)
    cv2.putText(frame, f"Available: {available}", (20, 92),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if paused:
        cv2.putText(frame, "PAUSED", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return frame


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Parking Lot Demo")
    print("=" * 50)

    # Load zones
    print(f"Loading zones from {ZONES_JSON}...")
    entry_zone, exit_zone = load_zones(ZONES_JSON)
    print(f"  Entry zone: {len(entry_zone)} points")
    print(f"  Exit zone:  {len(exit_zone)} points")

    # Load model
    print(f"Loading YOLO model ({MODEL_PATH})...")
    model = YOLO(MODEL_PATH)
    print("  Model ready.")

    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps:.1f}  |  Total frames: {total}")
    print()
    print("Controls: q = quit  |  SPACE = pause/resume")
    print("-" * 50)

    tracker      = CentroidTracker(entry_zone, exit_zone)
    available    = LOT_CAPACITY
    paused       = False
    frame_index  = 0
    detections   = []   # Reuse last detections on skipped frames

    cv2.namedWindow("Parking Lot Demo", cv2.WINDOW_NORMAL)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

        if paused:
            continue

        ret, frame = cap.read()
        if not ret:
            print("\nVideo finished.")
            break

        # Resize to 640x360 if not already
        frame = cv2.resize(frame, (640, 360))
        frame_index += 1

        # Only run YOLO every FRAME_SKIP frames for speed
        if frame_index % FRAME_SKIP == 0:
            detections = detect_vehicles(model, frame)

        # Track and get events
        events = tracker.update(detections)

        # Update available count
        for event in events:
            if event == "entry":
                available = max(0, available - 1)
            elif event == "exit":
                available = min(LOT_CAPACITY, available + 1)

        # Draw
        frame = draw_zone(frame, entry_zone, (0, 255, 0), "ENTRY")
        frame = draw_zone(frame, exit_zone,  (0, 0, 255), "EXIT")
        frame = draw_detections(frame, detections)
        frame = draw_tracks(frame, tracker.tracks)
        frame = draw_hud(frame, tracker.entry_count, tracker.exit_count, available, paused)

        cv2.imshow("Parking Lot Demo", frame)

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 50)
    print(f"Final results:")
    print(f"  Entries:   {tracker.entry_count}")
    print(f"  Exits:     {tracker.exit_count}")
    print(f"  Available: {available}/{LOT_CAPACITY}")


if __name__ == "__main__":
    main()