"""
Phase 2 end-to-end pipeline.

Video/camera source -> PersonDetector -> Tracker -> LineCounter -> HeatmapGenerator.

This wires together the four existing, independently-tested modules
(detector.py, tracker.py, counter.py, heatmap.py) using config.py for every
tunable. It intentionally does NOT do SQLite persistence or serve an API yet
- those are later steps. Where the pipeline would hand off to the database,
that call is a clearly-marked no-op for now (see _log_to_database_stub).

Usage:
    python main.py --source path/to/video.mp4
    python main.py --source path/to/video.mp4 --output annotated.mp4 --no-display
    python main.py                          # uses config.CAMERA_SOURCE
"""
import argparse
import logging
import os
import time

import cv2

import config
from detector import PersonDetector
from tracker import Tracker
from counter import LineCounter
from heatmap import HeatmapGenerator

logger = logging.getLogger(__name__)


def _log_to_database_stub(stats: dict) -> None:
    """
    Placeholder for the SQLite persistence layer (not implemented yet -
    that's a later step, not Step 3). Intentionally a no-op so main.py's
    structure already has the right call site without inventing a schema
    now.
    """
    pass


def build_pipeline():
    """Constructs the four pipeline stages from config.py values."""
    detector = PersonDetector(
        model_path=config.MODEL_PATH,
        conf_threshold=config.CONFIDENCE_THRESHOLD,
        iou_threshold=config.IOU_THRESHOLD,
        imgsz=config.INFERENCE_SIZE,
        device=config.DEVICE,
        person_class_id=config.PERSON_CLASS_ID,
    )
    tracker = Tracker(
        max_age=config.MAX_TRACK_AGE,
        min_hits=config.MIN_HITS,
        iou_threshold=config.IOU_MATCH_THRESHOLD,
        max_missed_for_display=config.MAX_MISSED_FOR_DISPLAY,
        reacquire_window=config.REACQUIRE_WINDOW,
        reacquire_iou_ratio=config.REACQUIRE_IOU_RATIO,
    )
    line_counter = LineCounter(
        line_start=config.LINE_START,
        line_end=config.LINE_END,
        margin=config.LINE_MARGIN,
    )
    heatmap_gen = HeatmapGenerator(
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        blur_kernel_size=config.HEATMAP_BLUR_KERNEL,
        point_radius=config.HEATMAP_POINT_RADIUS,
        decay_factor=config.HEATMAP_DECAY,
    )
    return detector, tracker, line_counter, heatmap_gen


def draw_overlay(frame, tracks, line_counter_stats):
    """Draws track boxes/IDs, the counting line, and running stats onto frame (in place)."""
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"ID {t['id']}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    ls = tuple(int(v) for v in config.LINE_START)
    le = tuple(int(v) for v in config.LINE_END)
    cv2.line(frame, ls, le, (0, 0, 255), 2)

    cv2.putText(
        frame,
        f"Entries: {line_counter_stats['entries']}  Exits: {line_counter_stats['exits']}  "
        f"Occupancy: {line_counter_stats['occupancy']}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    return frame


def run(source, output_path=None, display=None, max_frames=None):
    """
    Runs the full pipeline against `source` (a video file path, RTSP URL, or
    integer webcam index). Returns final LineCounter stats as a dict.
    """
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))

    if display is None:
        display = config.DISPLAY_WINDOW

    os.makedirs(os.path.dirname(config.HEATMAP_SAVE_PATH), exist_ok=True)

    # cv2.VideoCapture wants an int for a webcam index, a str for a file/RTSP path.
    cap_source = int(source) if isinstance(source, str) and source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        f"Source: {source} | native FPS: {fps:.2f} | total frames: {total_frames} "
        f"| resizing every frame to {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}"
    )

    detector, tracker, line_counter, heatmap_gen = build_pipeline()

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

    display_available = display
    frame_idx = 0
    processed_idx = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Resize FIRST so detector, tracker, counter, and heatmap all
            # operate in the exact same coordinate space (config.FRAME_WIDTH
            # x config.FRAME_HEIGHT) regardless of the source's native size.
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

            if (frame_idx - 1) % config.FRAME_SKIP != 0:
                if writer is not None:
                    writer.write(frame)
                continue

            processed_idx += 1
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            line_counter.update(tracks)
            centroids = [
                ((t["bbox"][0] + t["bbox"][2]) / 2.0, (t["bbox"][1] + t["bbox"][3]) / 2.0)
                for t in tracks
            ]
            heatmap_gen.update(centroids)

            if processed_idx % config.HEATMAP_AUTOSAVE_INTERVAL_FRAMES == 0:
                heatmap_gen.save(config.HEATMAP_SAVE_PATH)

            # Where DB logging will eventually hook in (throttled by
            # config.OCCUPANCY_LOG_INTERVAL_SECONDS) - stubbed for now.
            _log_to_database_stub(line_counter.get_stats())

            stats = line_counter.get_stats()
            draw_overlay(frame, tracks, stats)

            if writer is not None:
                writer.write(frame)

            if display_available:
                try:
                    cv2.imshow("Retail AI - Phase 2", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Quit key pressed, stopping.")
                        break
                except cv2.error as e:
                    logger.warning(f"Display unavailable ({e}); continuing headless.")
                    display_available = False

            if processed_idx % 100 == 0:
                logger.info(
                    f"[frame {frame_idx}/{total_frames}] processed={processed_idx} "
                    f"active_tracks={len(tracks)} stats={stats}"
                )
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down cleanly.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display_available:
            cv2.destroyAllWindows()
        heatmap_gen.save(config.HEATMAP_SAVE_PATH)

    elapsed = time.time() - start_time
    final_stats = line_counter.get_stats()
    logger.info(
        f"DONE. frames_read={frame_idx} frames_processed={processed_idx} "
        f"elapsed={elapsed:.1f}s stats={final_stats}"
    )
    return final_stats


def main():
    parser = argparse.ArgumentParser(description="Retail AI Phase 2 pipeline")
    parser.add_argument("--source", default=config.CAMERA_SOURCE, help="Video file, RTSP URL, or webcam index")
    parser.add_argument("--output", default=None, help="Optional path to write an annotated output video")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (for testing)")
    args = parser.parse_args()

    run(
        source=args.source,
        output_path=args.output,
        display=(False if args.no_display else None),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
