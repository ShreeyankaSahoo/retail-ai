"""
Centralized configuration for the CV pipeline.
Tune these values first when optimizing for Raspberry Pi 4B performance.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
DB_PATH = os.path.join(BASE_DIR, "data", "retail_ai.db")
HEATMAP_SAVE_PATH = os.path.join(BASE_DIR, "data", "heatmap.png")

# ---------------------------------------------------------------------------
# Detector (YOLOv8n)
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.3       # validated during Phase 2 tracking tests
IOU_THRESHOLD = 0.45
PERSON_CLASS_ID = 0            # COCO class 0 == "person"
INFERENCE_SIZE = 320            # 256/320/416 — lower = faster on RPi4 CPU
DEVICE = "cpu"                  # RPi4 has no CUDA GPU

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# The deployment camera's resolution and aspect ratio are NOT known ahead of
# time (could be a portrait phone clip, a landscape webcam, a square RTSP
# feed, etc.), so there is no fixed FRAME_WIDTH/FRAME_HEIGHT here. Instead,
# main.py reads the real source dimensions at runtime and derives a
# processing size that preserves the source's native aspect ratio - it only
# ever scales down uniformly, capped by PROCESSING_MAX_DIMENSION, and never
# stretches/distorts. See main.py's compute_processing_size().
CAMERA_SOURCE = "data/sample.mp4"                # 0 = default USB camera, or RTSP/file path
PROCESSING_MAX_DIMENSION = 640   # cap on the longer edge after aspect-preserving resize (RPi4 perf knob)
FRAME_SKIP = 2                   # run inference every Nth frame to save CPU
DISPLAY_WINDOW = True            # False for headless deployment (systemd service)

# ---------------------------------------------------------------------------
# Tracker (SORT-style, Kalman filter + IOU association)
# ---------------------------------------------------------------------------
MAX_TRACK_AGE = 90               # frames a lost track survives before deletion (validated during Phase 2 tracking tests)
MIN_HITS = 3                     # frames required before a track is "confirmed"
IOU_MATCH_THRESHOLD = 0.3
MAX_MISSED_FOR_DISPLAY = 4      # consecutive missed frames a confirmed track may still be reported for
REACQUIRE_WINDOW = 10            # frames within which a lost track is eligible for lenient re-matching
REACQUIRE_IOU_RATIO = 0.5        # lenient re-match IOU threshold = IOU_MATCH_THRESHOLD * this ratio

# ---------------------------------------------------------------------------
# Virtual line for entry/exit counting, expressed as FRACTIONS (0.0-1.0) of
# the actual runtime processing frame width/height - not absolute pixels -
# so the line geometry is meaningful regardless of the source's resolution
# or aspect ratio. main.py converts these to concrete pixel coordinates once
# it knows the real processing frame size for this run.
# NOTE: If entries/exits report as reversed, swap LINE_START_FRACTION/
# LINE_END_FRACTION or invert the sign check in counter.py — this depends on
# camera mounting.
# ---------------------------------------------------------------------------
LINE_START_FRACTION = (0.0, 0.5)   # default: horizontal line across the frame's mid-height
LINE_END_FRACTION = (1.0, 0.5)
LINE_MARGIN_FRACTION = 0.015       # dead-zone as a fraction of min(processing_w, processing_h)

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
