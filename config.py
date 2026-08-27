"""
Centralized configuration for the CV pipeline.
Tune these values first when optimizing for Raspberry Pi 4B performance.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
DB_PATH = os.path.join(BASE_DIR, "data", "retail_ai.db")
HEATMAP_SAVE_PATH = os.path.join(BASE_DIR, "data", "heatmap.png")

# ---------------------------------------------------------------------------
# Detector (YOLOv8n)
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
PERSON_CLASS_ID = 0            # COCO class 0 == "person"
INFERENCE_SIZE = 320            # 256/320/416 — lower = faster on RPi4 CPU
DEVICE = "cpu"                  # RPi4 has no CUDA GPU

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_SOURCE = "data/sample.mp4"                # 0 = default USB camera, or RTSP/file path
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SKIP = 2                   # run inference every Nth frame to save CPU
DISPLAY_WINDOW = True            # False for headless deployment (systemd service)

# ---------------------------------------------------------------------------
# Tracker (SORT-style, Kalman filter + IOU association)
# ---------------------------------------------------------------------------
MAX_TRACK_AGE = 30               # frames a lost track survives before deletion
MIN_HITS = 3                     # frames required before a track is "confirmed"
IOU_MATCH_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Virtual line for entry/exit counting (pixel coords of resized frame)
# NOTE: If entries/exits report as reversed, swap LINE_START/END or invert
# the sign check in cv/counter.py — this depends on camera mounting.
# ---------------------------------------------------------------------------
LINE_START = (0, FRAME_HEIGHT // 2)
LINE_END = (FRAME_WIDTH, FRAME_HEIGHT // 2)
LINE_MARGIN = 8.0                # dead-zone (px) around the line to prevent jitter double-counts

# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
HEATMAP_BLUR_KERNEL = 25
HEATMAP_POINT_RADIUS = 15
HEATMAP_DECAY = 0.0               # 0 = all-time cumulative, e.g. 0.001 for "recent activity" bias
HEATMAP_AUTOSAVE_INTERVAL_FRAMES = 300

# ---------------------------------------------------------------------------
# Database logging throttling
# ---------------------------------------------------------------------------
OCCUPANCY_LOG_INTERVAL_SECONDS = 10
POSITION_LOG_EVERY_N_FRAMES = 5

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
