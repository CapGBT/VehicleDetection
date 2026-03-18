import cv2
import json
from pathlib import Path

# Configuration
IMAGE_PATH = Path("data/frames/frame_0000.png")
OUTPUT_JSON = Path("data/parking_spot_coordinates.json")
NUM_SPOTS = 18

# Variables for overall state
spots = {}
current_spot_id = 1
current_points = []

# Load image
img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
display_img = img.copy()


def mouse_callback(event, x, y, flags, param):
    """
    Handle mouse clicks to add parking spot corners.
    """
    global current_points, display_img

    if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 4:
        current_points.append((x, y))
        print(f"Spot {current_spot_id} - Point {len(current_points)}/4: ({x}, {y})")

        display_img = img.copy()
        draw_all_spots()
        draw_current_points()


def draw_current_points():
    """
    Draw the points for the spot currently being defined.
    """
    global display_img

    for index, (x, y) in enumerate(current_points):
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            display_img,
            str(index + 1),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    if len(current_points) > 1:
        for index in range(len(current_points) - 1):
            cv2.line(display_img, current_points[index], current_points[index + 1], (0, 255, 255), 2)

        if len(current_points) == 4:
            cv2.line(display_img, current_points[3], current_points[0], (0, 255, 255), 2)


def draw_all_spots():
    """
    Draw all completed spots in green.
    """
    global display_img

    for spot_id, points in spots.items():
        pts = [(int(x), int(y)) for x, y in points]
        for index in range(4):
            cv2.line(display_img, pts[index], pts[(index + 1) % 4], (0, 255, 0), 2)

        center_x = sum(point[0] for point in pts) // 4
        center_y = sum(point[1] for point in pts) // 4
        cv2.putText(
            display_img,
            f"#{spot_id}",
            (center_x - 10, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


def save_spots():
    """
    Save labeled spots using the richer lot-config structure.
    """
    output_data = {
        "metadata": {
            "name": "default-lot",
            "camera_angle": "top-angle",
            "notes": "Add count lines manually if you want entrance/exit tracking.",
        },
        "spots": [],
        "count_lines": [],
    }
    for spot_id, points in spots.items():
        output_data["spots"].append({"id": spot_id, "coordinates": points})

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2)

    print(f"\nSaved {len(spots)} spots to {OUTPUT_JSON}")


print(f"Labeling {NUM_SPOTS} parking spots")
print("Click 4 corners for each spot (clockwise from top-left)")
print("Press 'n' after 4 points to go to next spot")
print("Press 's' to save and exit")
print("Press ESC to exit without saving")
print("-" * 50)

cv2.namedWindow("Parking Lot Labeler")
cv2.setMouseCallback("Parking Lot Labeler", mouse_callback)

while True:
    status_img = display_img.copy()
    status = f"Spot: {current_spot_id}/{NUM_SPOTS} | Points: {len(current_points)}/4 | Completed: {len(spots)}"
    cv2.putText(status_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(status_img, "n=Next | s=Save | ESC=Exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("Parking Lot Labeler", status_img)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        print("\nExiting without saving")
        break

    if key == ord("n"):
        if len(current_points) == 4:
            spots[current_spot_id] = current_points.copy()
            print(f"Saved spot {current_spot_id}")

            current_spot_id += 1
            current_points = []
            display_img = img.copy()
            draw_all_spots()

            if current_spot_id > NUM_SPOTS:
                print(f"\nAll {NUM_SPOTS} spots labeled. Press 's' to save.")
        else:
            print(f"Need 4 points (have {len(current_points)})")

    if key == ord("s"):
        if spots:
            save_spots()
            break
        print("No spots to save")

cv2.destroyAllWindows()
