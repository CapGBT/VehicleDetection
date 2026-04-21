"""
Zone Setup Tool
Draw entry and exit zones on a reference frame and save to data/zones.json

Usage:
    python src/zone_setup.py

Controls:
    Left click  - Add a point to the current zone
    n           - Finish current zone, move to next
    u           - Undo last point
    r           - Reset current zone
    s           - Save and exit
    ESC         - Exit without saving
"""

import cv2
import json
import numpy as np
from pathlib import Path


# Configuration
REFERENCE_FRAME = Path("data/reference_frame.jpg")
OUTPUT_JSON = Path("data/zones.json")

# Zone definitions in order - (name, display_label, color_BGR)
ZONES = [
    ("entry", "ENTRY", (0, 255, 0)),   # Green
    ("exit",  "EXIT",  (0, 0, 255)),   # Red
]


# State
current_zone_index = 0
current_points = []
completed_zones = {}
display_img = None

# Load reference frame
img = cv2.imread(str(REFERENCE_FRAME))
if img is None:
    raise FileNotFoundError(
        f"Could not load reference frame: {REFERENCE_FRAME}\n"
        f"Run this first:\n"
        f"  ffmpeg -ss 00:00:02 -i data/raw/2in2outTest.MOV -vframes 1 -vf scale=640:360 data/reference_frame.jpg"
    )
display_img = img.copy()


def draw_completed_zones():
    """Draw all finished zones in their respective colors."""
    for zone_name, points in completed_zones.items():
        # Find color for this zone
        color = next(color for name, _, color in ZONES if name == zone_name)
        label = next(lbl for name, lbl, _ in ZONES if name == zone_name)

        pts = np.array(points, dtype=np.int32)
        cv2.polylines(display_img, [pts], isClosed=True, color=color, thickness=2)

        # Fill with transparent overlay
        overlay = display_img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, display_img, 0.75, 0, display_img)

        # Label at centroid
        cx = sum(p[0] for p in points) // len(points)
        cy = sum(p[1] for p in points) // len(points)
        cv2.putText(display_img, label, (cx - 25, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_current_points():
    """Draw in-progress points and lines for the zone being defined."""
    if not current_points:
        return

    color = (0, 255, 255)  # Yellow for in-progress

    for i, (x, y) in enumerate(current_points):
        cv2.circle(display_img, (x, y), 5, color, -1)
        cv2.putText(display_img, str(i + 1), (x + 8, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw lines between points
    for i in range(len(current_points) - 1):
        cv2.line(display_img, current_points[i], current_points[i + 1], color, 2)


def redraw():
    """Refresh display image from scratch."""
    global display_img
    display_img = img.copy()
    draw_completed_zones()
    draw_current_points()


def mouse_callback(event, x, y, flags, param):
    """Add a point on left click."""
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"  Point {len(current_points)}: ({x}, {y})")
        redraw()


def save_zones():
    """Save completed zones to JSON."""
    output = {}
    for zone_name, points in completed_zones.items():
        output[zone_name] = [[int(x), int(y)] for x, y in points]

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved zones to {OUTPUT_JSON}")
    for name, pts in output.items():
        print(f"  {name}: {len(pts)} points")


# Main loop
print("Zone Setup Tool")
print("=" * 50)
print(f"Reference frame: {REFERENCE_FRAME}")
print(f"Output: {OUTPUT_JSON}")
print()

cv2.namedWindow("Zone Setup")
cv2.setMouseCallback("Zone Setup", mouse_callback)
redraw()

while True:
    zone_name, zone_label, zone_color = ZONES[min(current_zone_index, len(ZONES) - 1)]

    # Draw status bar
    status_img = display_img.copy()
    cv2.rectangle(status_img, (0, 0), (640, 50), (0, 0, 0), -1)
    cv2.putText(status_img,
                f"Drawing: {zone_label} zone  |  Points: {len(current_points)}  |  n=Done  u=Undo  r=Reset  s=Save  ESC=Quit",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Zone Setup", status_img)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("Exiting without saving.")
        break

    elif key == ord("u"):  # Undo
        if current_points:
            current_points.pop()
            redraw()

    elif key == ord("r"):  # Reset current zone
        current_points = []
        redraw()

    elif key == ord("n"):  # Finish current zone
        if len(current_points) >= 3:
            completed_zones[zone_name] = current_points.copy()
            print(f"Zone '{zone_label}' saved with {len(current_points)} points.")
            current_points = []
            current_zone_index += 1
            redraw()

            if current_zone_index >= len(ZONES):
                print("\nAll zones defined! Press 's' to save.")
        else:
            print(f"Need at least 3 points (have {len(current_points)}).")

    elif key == ord("s"):  # Save
        # Save any in-progress zone first
        if len(current_points) >= 3:
            completed_zones[zone_name] = current_points.copy()
            current_points = []
            redraw()

        if completed_zones:
            save_zones()
            break
        else:
            print("No zones to save yet.")

cv2.destroyAllWindows()