"""
Batch parking occupancy detection for saved frames.

This keeps the original workflow intact while delegating the shared logic to the
continuous monitoring pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from continuous_monitor import process_frame_sequence
from parking_core import (
    load_count_lines,
    load_count_zones,
    load_detection_zones,
    load_lot_config,
    load_parking_spots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect parking occupancy from saved frames")
    parser.add_argument("--frames-dir", default="data/frames", help="Directory of extracted frames")
    parser.add_argument(
        "--lot-config",
        default="data/parking_spot_coordinates.json",
        help="Parking spot JSON path",
    )
    parser.add_argument("--output-dir", default="results/detections", help="Directory for annotated frames")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.2, help="Parking spot IoU threshold")
    parser.add_argument("--summary-json", default="results/detections/summary.json", help="Summary JSON path")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 processes all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lot_config_path = Path(args.lot_config)
    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))

    if not frame_paths:
        raise FileNotFoundError(f"No frame images found in {frames_dir}")

    lot_config = load_lot_config(lot_config_path)
    print(f"Loaded lot config for camera angle: {lot_config.get('metadata', {}).get('camera_angle', 'unknown')}")

    spots = load_parking_spots(lot_config_path)
    count_lines = load_count_lines(lot_config_path)
    count_zones = load_count_zones(lot_config_path)
    detection_zones = load_detection_zones(lot_config_path)
    model = YOLO(args.model)
    args.server_url = ""
    args.server_api_key = ""
    args.server_min_interval = 0.0

    summary = process_frame_sequence(frame_paths, model, spots, count_lines, detection_zones, count_zones, args)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Processed {len(frame_paths)} frame(s)")
    print(f"Final availability: {summary['available_spots']}/{summary['total_spots']}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
