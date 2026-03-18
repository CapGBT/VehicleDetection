# CapVehicleDetection

This project now supports a cleaner parking-lot detection pipeline while keeping the original core methods:

- YOLO-based vehicle detection
- Polygon-based parking spot occupancy checks
- Driveway ROI filtering to ignore unrelated road traffic
- Frame extraction from a source video
- Manual parking-spot labeling
- Optional entrance/exit counting with diagonal count lines and deadband logic
- Optional entrance/exit counting with polygon trigger zones
- Optional server updates for availability/count changes

## Project scripts

- `python src/extract_frames.py`
  - Extracts frames from `data/raw/parkingSet.mp4` into `data/frames`
- `python src/label_parking_spots.py`
  - Labels parking spots and saves a lot config JSON
- `python src/detect_parking.py`
  - Processes saved frames and writes annotated results to `results/detections`
- `python src/continuous_monitor.py --source data/frames`
  - Runs the continuous monitoring pipeline on a frame directory, image, or video
  - Best fit for entry/exit testing on your recorded sample videos

## Lot config format

The project supports the original spot-only JSON and a richer config format:

```json
{
  "metadata": {
    "name": "default-lot",
    "camera_angle": "top-angle"
  },
  "spots": [
    {
      "id": 1,
      "coordinates": [[0, 0], [10, 0], [10, 10], [0, 10]]
    }
  ],
  "count_lines": [
    {
      "id": "gate_main",
      "start": [470, 120],
      "end": [470, 470],
      "direction": "vertical",
      "positive_label": "entry",
      "negative_label": "exit"
    }
  ],
  "count_zones": [
    {
      "id": "entry_trigger",
      "label": "entry",
      "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]
    }
  ]
}
```

See `data/lot_config.example.json` for a sample.
See `data/driveway_config.sample.json` for an entry/exit-only config shaped for the provided driveway camera angle.
Use `count_zones` when the entry and exit paths are easier to describe as small trigger polygons than as a single crossing line.

## Install

```bash
pip install -r requirements.txt
```

For Raspberry Pi deployment, prefer `yolov8n.pt`, a narrow detection ROI, and frame skipping or a reduced input cadence if needed.

## Example commands

```bash
python src/detect_parking.py --max-frames 10
python src/continuous_monitor.py --source data/raw/parkingSet.mp4 --summary-json results/monitor/summary.json
python src/continuous_monitor.py --source data/frames --server-url http://localhost:8000/api/parking/update
python src/continuous_monitor.py --source data/raw/2in2outTest.MOV --lot-config data/driveway_config.sample.json --model yolov8n.pt --summary-json results/monitor/driveway_summary.json
```
