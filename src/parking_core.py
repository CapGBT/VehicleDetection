"""
Shared parking-lot detection utilities.

This module keeps the original occupancy-detection approach intact while making
it easier to reuse across one-off frame processing and continuous monitoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


@dataclass
class ParkingSpot:
    id: int
    coordinates: list[list[int]]


@dataclass
class VehicleDetection:
    bbox: list[int]
    confidence: float
    class_id: int


@dataclass
class OccupancyResult:
    id: int
    coordinates: list[list[int]]
    occupied: bool
    max_iou: float


@dataclass
class CountLine:
    id: str
    start: list[int]
    end: list[int]
    direction: str = "vertical"
    positive_label: str = "entry"
    negative_label: str = "exit"
    deadband: float = 20.0


@dataclass
class DetectionZone:
    id: str
    polygon: list[list[int]]
    anchor: str = "bottom_center"


@dataclass
class CountZone:
    id: str
    polygon: list[list[int]]
    label: str
    anchor: str = "bottom_center"


def load_lot_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "spots" in data:
        return data

    # Backwards compatibility with the original spot-only JSON structure.
    if isinstance(data, dict):
        return {
            "metadata": {
                "name": "default-lot",
                "camera_angle": "top-angle",
            },
            "spots": [
                {
                    "id": spot_info["id"],
                    "coordinates": spot_info["coordinates"],
                }
                for _, spot_info in sorted(data.items())
            ],
            "count_lines": [],
        }

    raise ValueError(f"Unsupported lot configuration format: {config_path}")


def load_parking_spots(config_path: Path) -> list[ParkingSpot]:
    config = load_lot_config(config_path)
    return [
        ParkingSpot(id=int(spot["id"]), coordinates=spot["coordinates"])
        for spot in config.get("spots", [])
    ]


def load_count_lines(config_path: Path) -> list[CountLine]:
    config = load_lot_config(config_path)
    return [
        CountLine(
            id=line.get("id", f"line_{index + 1}"),
            start=line["start"],
            end=line["end"],
            direction=line.get("direction", "vertical"),
            positive_label=line.get("positive_label", "entry"),
            negative_label=line.get("negative_label", "exit"),
            deadband=float(line.get("deadband", 20.0)),
        )
        for index, line in enumerate(config.get("count_lines", []))
    ]


def load_detection_zones(config_path: Path) -> list[DetectionZone]:
    config = load_lot_config(config_path)
    return [
        DetectionZone(
            id=zone.get("id", f"zone_{index + 1}"),
            polygon=zone["polygon"],
            anchor=zone.get("anchor", "bottom_center"),
        )
        for index, zone in enumerate(config.get("detection_zones", []))
    ]


def load_count_zones(config_path: Path) -> list[CountZone]:
    config = load_lot_config(config_path)
    return [
        CountZone(
            id=zone.get("id", f"count_zone_{index + 1}"),
            polygon=zone["polygon"],
            label=zone["label"],
            anchor=zone.get("anchor", "bottom_center"),
        )
        for index, zone in enumerate(config.get("count_zones", []))
    ]


def detect_vehicles(
    model: YOLO,
    image: np.ndarray,
    conf_threshold: float,
    detection_zones: list[DetectionZone] | None = None,
) -> list[VehicleDetection]:
    results = model.predict(image, conf=conf_threshold, verbose=False)[0]

    detections: list[VehicleDetection] = []
    for box in results.boxes:
        cls_id = int(box.cls)
        if cls_id not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detection = VehicleDetection(
            bbox=[int(x1), int(y1), int(x2), int(y2)],
            confidence=float(box.conf),
            class_id=cls_id,
        )
        if detection_zones and not detection_in_zones(detection, detection_zones):
            continue
        detections.append(detection)

    return detections


def detection_center(detection: VehicleDetection) -> tuple[int, int]:
    x1, y1, x2, y2 = detection.bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def detection_anchor_point(detection: VehicleDetection, anchor: str = "center") -> tuple[int, int]:
    x1, y1, x2, y2 = detection.bbox
    if anchor == "bottom_center":
        return ((x1 + x2) // 2, y2)
    if anchor == "bottom_left":
        return (x1, y2)
    if anchor == "bottom_right":
        return (x2, y2)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def point_in_polygon(point: tuple[int, int], polygon: list[list[int]]) -> bool:
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def detection_in_zones(detection: VehicleDetection, zones: list[DetectionZone]) -> bool:
    return any(
        point_in_polygon(detection_anchor_point(detection, zone.anchor), zone.polygon)
        for zone in zones
    )


def polygon_to_bbox(polygon: Iterable[Iterable[int]]) -> list[int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def compute_iou(bbox1: list[int], bbox2: list[int]) -> float:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = bbox1_area + bbox2_area - intersection
    return intersection / union if union > 0 else 0.0


def determine_occupancy(
    spots: list[ParkingSpot],
    detections: list[VehicleDetection],
    iou_threshold: float,
) -> list[OccupancyResult]:
    results: list[OccupancyResult] = []

    for spot in spots:
        spot_bbox = polygon_to_bbox(spot.coordinates)
        occupied = False
        max_iou = 0.0

        for detection in detections:
            iou = compute_iou(spot_bbox, detection.bbox)
            max_iou = max(max_iou, iou)
            if iou >= iou_threshold:
                occupied = True

        results.append(
            OccupancyResult(
                id=spot.id,
                coordinates=spot.coordinates,
                occupied=occupied,
                max_iou=max_iou,
            )
        )

    return results


def draw_results(
    image: np.ndarray,
    spots_occupancy: list[OccupancyResult],
    detections: list[VehicleDetection],
    counts: dict | None = None,
    count_lines: list[CountLine] | None = None,
    detection_zones: list[DetectionZone] | None = None,
    count_zones: list[CountZone] | None = None,
    tracked_centers: dict[int, tuple[int, int]] | None = None,
    debug_metrics: dict | None = None,
) -> np.ndarray:
    output = image.copy()

    for spot in spots_occupancy:
        color = (0, 0, 255) if spot.occupied else (0, 255, 0)
        points = np.array(spot.coordinates, dtype=np.int32)
        cv2.polylines(output, [points], isClosed=True, color=color, thickness=2)

        center_x = sum(point[0] for point in spot.coordinates) // len(spot.coordinates)
        center_y = sum(point[1] for point in spot.coordinates) // len(spot.coordinates)
        cv2.putText(
            output,
            f"#{spot.id}",
            (center_x - 10, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            output,
            f"{detection.confidence:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    for line in count_lines or []:
        color = (0, 255, 255)
        cv2.line(output, tuple(line.start), tuple(line.end), color, 2)
        cv2.putText(
            output,
            line.id,
            (line.start[0] + 5, line.start[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    for zone in detection_zones or []:
        points = np.array(zone.polygon, dtype=np.int32)
        cv2.polylines(output, [points], isClosed=True, color=(0, 165, 255), thickness=2)
        anchor = tuple(points[0])
        cv2.putText(
            output,
            zone.id,
            (anchor[0] + 4, anchor[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

    for zone in count_zones or []:
        points = np.array(zone.polygon, dtype=np.int32)
        overlay = output.copy()
        cv2.fillPoly(overlay, [points], color=(255, 120, 0))
        output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)
        cv2.polylines(output, [points], isClosed=True, color=(255, 0, 0), thickness=5)
        anchor = tuple(points[0])
        cv2.putText(
            output,
            f"{zone.id}:{zone.label}",
            (anchor[0] + 4, anchor[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            3,
        )

    for track_id, center in (tracked_centers or {}).items():
        cv2.circle(output, center, 4, (255, 255, 0), -1)
        cv2.putText(
            output,
            f"T{track_id}",
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

    total_spots = len(spots_occupancy)
    occupied_count = sum(1 for spot in spots_occupancy if spot.occupied)
    available_count = total_spots - occupied_count

    cv2.rectangle(output, (10, 10), (430, 90), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"Available: {available_count}/{total_spots}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    if counts:
        counter_text = " | ".join(f"{label}: {value}" for label, value in counts.items())
        cv2.putText(
            output,
            counter_text,
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    if debug_metrics:
        debug_text = " | ".join(f"{key}: {value}" for key, value in debug_metrics.items())
        cv2.putText(
            output,
            debug_text,
            (20, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    return output


def occupancy_summary(spots_occupancy: list[OccupancyResult]) -> dict:
    total_spots = len(spots_occupancy)
    occupied = sum(1 for spot in spots_occupancy if spot.occupied)
    return {
        "total_spots": total_spots,
        "occupied_spots": occupied,
        "available_spots": total_spots - occupied,
        "spots": [
            {
                "id": spot.id,
                "occupied": spot.occupied,
                "max_iou": round(spot.max_iou, 4),
            }
            for spot in spots_occupancy
        ],
    }
