import argparse
import json
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Label entry/exit count zones on a frame")
    parser.add_argument(
        "--image",
        default="results/monitor/monitor_frame_0801.png",
        help="Reference image to label",
    )
    parser.add_argument(
        "--config",
        default="data/driveway_config.sample.json",
        help="Lot config JSON to update",
    )
    return parser.parse_args()


args = parse_args()
IMAGE_PATH = Path(args.image)
CONFIG_PATH = Path(args.config)

ZONE_ORDER = [
    ("entry_trigger", "entry"),
    ("exit_trigger", "exit"),
]

current_zone_index = 0
current_points = []
display_img = None


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "metadata": {
            "name": "sample-driveway-camera",
            "camera_angle": "roadside-driveway",
        },
        "spots": [],
        "detection_zones": [],
        "count_zones": [],
        "count_lines": [],
    }


config = load_config()
img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
display_img = img.copy()


def get_zone(zone_id):
    for zone in config.get("count_zones", []):
        if zone.get("id") == zone_id:
            return zone
    return None


def set_zone(zone_id, label, polygon):
    existing = get_zone(zone_id)
    if existing:
        existing["polygon"] = polygon
        existing["label"] = label
        existing["anchor"] = "bottom_center"
        return

    config.setdefault("count_zones", []).append(
        {
            "id": zone_id,
            "label": label,
            "anchor": "bottom_center",
            "polygon": polygon,
        }
    )


def draw_existing_zones():
    for zone in config.get("count_zones", []):
        points = zone.get("polygon", [])
        if len(points) < 2:
            continue
        pts = [(int(x), int(y)) for x, y in points]
        color = (255, 0, 0) if zone.get("label") == "exit" else (0, 165, 255)
        for index in range(len(pts)):
            cv2.line(display_img, pts[index], pts[(index + 1) % len(pts)], color, 3)
        anchor = pts[0]
        cv2.putText(
            display_img,
            f"{zone['id']}:{zone['label']}",
            (anchor[0] + 4, anchor[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


def draw_current_points():
    color = (0, 255, 255)
    for index, (x, y) in enumerate(current_points):
        cv2.circle(display_img, (x, y), 5, color, -1)
        cv2.putText(
            display_img,
            str(index + 1),
            (x + 8, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    if len(current_points) > 1:
        for index in range(len(current_points) - 1):
            cv2.line(display_img, current_points[index], current_points[index + 1], color, 2)


def redraw():
    global display_img
    display_img = img.copy()
    draw_existing_zones()
    draw_current_points()


def mouse_callback(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"Point {len(current_points)}: ({x}, {y})")
        redraw()


def save_config():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Saved count zones to {CONFIG_PATH}")


print("Count Zone Labeler")
print(f"Image: {IMAGE_PATH}")
print(f"Config: {CONFIG_PATH}")
print("Click points around the current zone.")
print("Keys:")
print("  n = finish current zone and move to next")
print("  u = undo last point")
print("  r = reset current zone points")
print("  s = save config")
print("  ESC = exit without saving")
print("-" * 50)

cv2.namedWindow("Count Zone Labeler")
cv2.setMouseCallback("Count Zone Labeler", mouse_callback)
redraw()

while True:
    zone_id, zone_label = ZONE_ORDER[min(current_zone_index, len(ZONE_ORDER) - 1)]
    status_img = display_img.copy()
    cv2.putText(
        status_img,
        f"Zone: {zone_id}:{zone_label} | Points: {len(current_points)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        status_img,
        "n=Next  u=Undo  r=Reset  s=Save  ESC=Exit",
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Count Zone Labeler", status_img)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        print("Exiting without saving")
        break

    if key == ord("u") and current_points:
        current_points.pop()
        redraw()

    if key == ord("r"):
        current_points = []
        redraw()

    if key == ord("n"):
        if len(current_points) >= 3:
            set_zone(zone_id, zone_label, [[int(x), int(y)] for x, y in current_points])
            print(f"Saved zone {zone_id} with {len(current_points)} points")
            current_points = []
            current_zone_index = min(current_zone_index + 1, len(ZONE_ORDER) - 1)
            redraw()
        else:
            print("Need at least 3 points for a polygon")

    if key == ord("s"):
        if current_points:
            zone_id, zone_label = ZONE_ORDER[min(current_zone_index, len(ZONE_ORDER) - 1)]
            if len(current_points) >= 3:
                set_zone(zone_id, zone_label, [[int(x), int(y)] for x, y in current_points])
                current_points = []
                redraw()
        save_config()
        break

cv2.destroyAllWindows()
