"""
Continuous parking-lot monitoring pipeline.

Supports:
- Parking occupancy from labeled spot polygons
- Entrance/exit counting using configured count lines
- Optional server updates when counts or availability change
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from ultralytics import YOLO

from parking_core import (
    CountLine,
    CountZone,
    detect_vehicles,
    detection_anchor_point,
    determine_occupancy,
    detection_center,
    draw_results,
    load_count_lines,
    load_count_zones,
    load_detection_zones,
    load_lot_config,
    load_parking_spots,
    point_in_polygon,
    occupancy_summary,
)
from server_sync import ServerSyncClient


DEFAULT_LOT_CONFIG = Path("data/parking_spot_coordinates.json")
DEFAULT_OUTPUT_DIR = Path("results/monitor")
DEFAULT_MODEL_PATH = "yolov8s.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous parking lot monitor")
    parser.add_argument("--source", default="data/frames", help="Image, directory, or video source")
    parser.add_argument("--lot-config", default=str(DEFAULT_LOT_CONFIG), help="Lot JSON config path")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.2, help="Spot occupancy IoU threshold")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Annotated frame output directory")
    parser.add_argument(
        "--show-detection-zones",
        action="store_true",
        help="Draw the broad detection ROI overlays on the output frames",
    )
    parser.add_argument("--summary-json", default="", help="Optional path for a final summary JSON")
    parser.add_argument("--server-url", default="", help="Optional endpoint for count/availability updates")
    parser.add_argument("--server-api-key", default="", help="Optional API key for server updates")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame from a video source",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=10.0,
        help="Target processing FPS for video sources; ignored when --frame-step is greater than 1",
    )
    parser.add_argument(
        "--server-min-interval",
        type=float,
        default=1.0,
        help="Minimum seconds between server updates",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 processes all")
    return parser.parse_args()


def iter_frame_paths(source: Path) -> list[Path]:
    if source.is_dir():
        pngs = sorted(source.glob("*.png"))
        jpgs = sorted(source.glob("*.jpg"))
        return pngs + jpgs
    if source.is_file() and source.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return [source]
    return []


def point_position(point: tuple[int, int], line: CountLine) -> float:
    x1, y1 = line.start
    x2, y2 = line.end
    px, py = point
    numerator = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    denominator = math.hypot(x2 - x1, y2 - y1) or 1.0
    return numerator / denominator


@dataclass
class TrackState:
    track_id: int
    center: tuple[int, int]
    detection_bbox: list[int] | None = None
    missing_frames: int = 0
    line_states: dict[str, str] = field(default_factory=dict)
    zone_states: dict[str, bool] = field(default_factory=dict)


@dataclass
class RecentZoneHit:
    zone_id: str
    point: tuple[int, int]
    frame_index: int


class CentroidLineCounter:
    def __init__(
        self,
        count_lines: list[CountLine],
        count_zones: list[CountZone] | None = None,
        max_distance: float = 70.0,
        max_missing: int = 10,
        zone_recount_cooldown_frames: int = 20,
        zone_recount_distance: float = 90.0,
    ):
        self.count_lines = count_lines
        self.count_zones = count_zones or []
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.zone_recount_cooldown_frames = zone_recount_cooldown_frames
        self.zone_recount_distance = zone_recount_distance
        self.next_track_id = 1
        self.tracks: dict[int, TrackState] = {}
        self.frame_counter = 0
        self.recent_zone_hits: list[RecentZoneHit] = []
        labels = {line.positive_label for line in count_lines} | {line.negative_label for line in count_lines}
        labels |= {zone.label for zone in self.count_zones}
        self.counts = {label: 0 for label in sorted(labels)}

    def update(self, detections: list) -> tuple[dict[str, int], dict[int, tuple[int, int]]]:
        self.frame_counter += 1
        detections_with_centers = [(detection, detection_center(detection)) for detection in detections]
        unmatched_track_ids = set(self.tracks)
        matched_track_ids: set[int] = set()

        for detection, center in detections_with_centers:
            track_id = self._find_track(center, unmatched_track_ids)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = TrackState(track_id=track_id, center=center, detection_bbox=detection.bbox)
            else:
                unmatched_track_ids.discard(track_id)
                self.tracks[track_id].center = center
                self.tracks[track_id].detection_bbox = detection.bbox
                self.tracks[track_id].missing_frames = 0

            matched_track_ids.add(track_id)
            self._update_line_crossings(self.tracks[track_id], center)
            self._update_zone_crossings(self.tracks[track_id], detection)

        for track_id in list(self.tracks):
            if track_id in matched_track_ids:
                continue
            self.tracks[track_id].missing_frames += 1
            if self.tracks[track_id].missing_frames > self.max_missing:
                del self.tracks[track_id]

        return dict(self.counts), {track_id: track.center for track_id, track in self.tracks.items()}

    def _find_track(self, center: tuple[int, int], candidates: set[int]) -> int | None:
        best_id = None
        best_distance = self.max_distance
        for track_id in candidates:
            track = self.tracks[track_id]
            distance = math.dist(center, track.center)
            if distance < best_distance:
                best_distance = distance
                best_id = track_id
        return best_id

    def _update_line_crossings(self, track: TrackState, center: tuple[int, int]) -> None:
        for line in self.count_lines:
            current_side = point_position(center, line)
            current_region = self._region_for_position(current_side, line.deadband)
            previous_region = track.line_states.get(line.id)

            if previous_region is not None and current_region != "middle" and current_region != previous_region:
                if previous_region == "negative" and current_region == "positive":
                    self.counts[line.positive_label] = self.counts.get(line.positive_label, 0) + 1
                elif previous_region == "positive" and current_region == "negative":
                    self.counts[line.negative_label] = self.counts.get(line.negative_label, 0) + 1

            if current_region != "middle":
                track.line_states[line.id] = current_region

    def _update_zone_crossings(self, track: TrackState, detection) -> None:
        for zone in self.count_zones:
            anchor_point = detection_anchor_point(detection, zone.anchor)
            is_inside = point_in_polygon(anchor_point, zone.polygon)
            was_inside = track.zone_states.get(zone.id, False)
            if is_inside and not was_inside and not self._is_recent_zone_hit(zone.id, anchor_point):
                self.counts[zone.label] = self.counts.get(zone.label, 0) + 1
                self.recent_zone_hits.append(
                    RecentZoneHit(
                        zone_id=zone.id,
                        point=anchor_point,
                        frame_index=self.frame_counter,
                    )
                )
            track.zone_states[zone.id] = is_inside
        self._prune_recent_zone_hits()

    @staticmethod
    def _region_for_position(position: float, deadband: float) -> str:
        if position > deadband:
            return "positive"
        if position < -deadband:
            return "negative"
        return "middle"

    def _is_recent_zone_hit(self, zone_id: str, point: tuple[int, int]) -> bool:
        for hit in self.recent_zone_hits:
            if hit.zone_id != zone_id:
                continue
            if self.frame_counter - hit.frame_index > self.zone_recount_cooldown_frames:
                continue
            if math.dist(point, hit.point) <= self.zone_recount_distance:
                return True
        return False

    def _prune_recent_zone_hits(self) -> None:
        self.recent_zone_hits = [
            hit
            for hit in self.recent_zone_hits
            if self.frame_counter - hit.frame_index <= self.zone_recount_cooldown_frames
        ]


def process_frame_sequence(
    frame_paths: list[Path],
    model: YOLO,
    spots,
    count_lines: list[CountLine],
    detection_zones,
    count_zones,
    args: argparse.Namespace,
) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counter = CentroidLineCounter(count_lines, count_zones=count_zones)
    sync_client = ServerSyncClient(
        endpoint=args.server_url or None,
        api_key=args.server_api_key or None,
        min_interval_seconds=args.server_min_interval,
    )

    last_summary: dict | None = None
    processed = 0

    for frame_path in frame_paths:
        if args.max_frames and processed >= args.max_frames:
            break

        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"[WARN] Skipping unreadable frame: {frame_path}")
            continue

        detections = detect_vehicles(model, image, args.confidence, detection_zones=detection_zones)
        occupancy = determine_occupancy(spots, detections, args.iou)
        counts, tracked_centers = counter.update(detections)

        output_image = draw_results(
            image,
            occupancy,
            detections,
            counts=counts,
            count_lines=count_lines,
            detection_zones=detection_zones if args.show_detection_zones else [],
            count_zones=count_zones,
            tracked_centers=tracked_centers,
            debug_metrics={
                "vehicles_detected": len(detections),
                "tracks": len(tracked_centers),
            },
        )
        cv2.imwrite(str(output_dir / f"monitor_{frame_path.name}"), output_image)

        last_summary = occupancy_summary(occupancy)
        last_summary["counts"] = counts
        last_summary["frame"] = frame_path.name
        sync_client.send_update(last_summary)

        processed += 1
        print(
            f"[FRAME {processed}] {frame_path.name} | "
            f"vehicles={len(detections)} | available={last_summary['available_spots']} | counts={counts}"
        )

    return last_summary or {"total_spots": 0, "occupied_spots": 0, "available_spots": 0, "counts": {}}


def process_video_source(
    source: Path,
    model: YOLO,
    spots,
    count_lines: list[CountLine],
    detection_zones,
    count_zones,
    args: argparse.Namespace,
) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source), cv2.CAP_MSMF)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_step = max(1, args.frame_step)
    if frame_step == 1 and args.target_fps > 0 and source_fps > 0:
        frame_step = max(1, int(round(source_fps / args.target_fps)))
    print(
        f"[INFO] Source FPS: {source_fps:.2f} | "
        f"Target FPS: {args.target_fps:.2f} | "
        f"Processing every {frame_step} frame(s)"
    )

    counter = CentroidLineCounter(count_lines, count_zones=count_zones)
    sync_client = ServerSyncClient(
        endpoint=args.server_url or None,
        api_key=args.server_api_key or None,
        min_interval_seconds=args.server_min_interval,
    )

    processed = 0
    frame_index = 0
    last_summary: dict | None = None

    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_index % frame_step != 0:
            frame_index += 1
            continue
        if args.max_frames and processed >= args.max_frames:
            break

        detections = detect_vehicles(model, frame, args.confidence, detection_zones=detection_zones)
        occupancy = determine_occupancy(spots, detections, args.iou)
        counts, tracked_centers = counter.update(detections)

        rendered = draw_results(
            frame,
            occupancy,
            detections,
            counts=counts,
            count_lines=count_lines,
            detection_zones=detection_zones if args.show_detection_zones else [],
            count_zones=count_zones,
            tracked_centers=tracked_centers,
            debug_metrics={
                "vehicles_detected": len(detections),
                "tracks": len(tracked_centers),
            },
        )
        output_path = output_dir / f"monitor_frame_{processed:04d}.png"
        cv2.imwrite(str(output_path), rendered)

        last_summary = occupancy_summary(occupancy)
        last_summary["counts"] = counts
        last_summary["frame"] = output_path.name
        sync_client.send_update(last_summary)

        processed += 1
        print(
            f"[FRAME {processed}] source_index={frame_index} | vehicles={len(detections)} | "
            f"available={last_summary['available_spots']} | counts={counts}"
        )
        frame_index += 1

    capture.release()
    return last_summary or {"total_spots": 0, "occupied_spots": 0, "available_spots": 0, "counts": {}}


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    lot_config_path = Path(args.lot_config)

    print(f"[INFO] Loading lot config: {lot_config_path}")
    lot_config = load_lot_config(lot_config_path)
    print(f"[INFO] Camera angle: {lot_config.get('metadata', {}).get('camera_angle', 'unknown')}")

    print(f"[INFO] Loading parking spots from: {lot_config_path}")
    spots = load_parking_spots(lot_config_path)
    count_lines = load_count_lines(lot_config_path)
    count_zones = load_count_zones(lot_config_path)
    detection_zones = load_detection_zones(lot_config_path)
    print(
        f"[INFO] Spots loaded: {len(spots)} | "
        f"Count lines loaded: {len(count_lines)} | "
        f"Count zones loaded: {len(count_zones)} | "
        f"Detection zones loaded: {len(detection_zones)}"
    )

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    frame_paths = iter_frame_paths(source)
    if frame_paths:
        summary = process_frame_sequence(frame_paths, model, spots, count_lines, detection_zones, count_zones, args)
    elif source.is_file():
        summary = process_video_source(source, model, spots, count_lines, detection_zones, count_zones, args)
    else:
        raise FileNotFoundError(f"Source was not found or not supported: {source}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[INFO] Summary saved to: {summary_path}")

    print("[DONE] Monitoring run complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
