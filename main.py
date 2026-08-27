"""
Phase 2 end-to-end pipeline.

Video/camera source -> PersonDetector -> Tracker -> LineCounter -> HeatmapGenerator.

This wires together the four existing, independently-tested modules
(detector.py, tracker.py, counter.py, heatmap.py) using config.py for every
tunable. It intentionally does NOT do SQLite persistence or serve an API yet
- those are later steps. Where the pipeline would hand off to the database,
that call is a clearly-marked no-op for now (see _log_to_database_stub).

Resolution/aspect-ratio independence: the deployment camera's resolution and
aspect ratio are not known ahead of time, so this pipeline never assumes a
fixed frame size. It reads the real source dimensions from the first frame,
derives a processing size that preserves the source's native aspect ratio
(uniform downscale only, capped by config.PROCESSING_MAX_DIMENSION - see
compute_processing_size()), and derives the counting-line geometry and
heatmap dimensions from that same runtime-computed size (config.py stores
the line as fractions of frame width/height, not absolute pixels). Detector,
tracker, counter, and heatmap all then operate in that one consistent
coordinate space for the whole run.

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
from queue_predictor import QueuePredictor

logger = logging.getLogger(__name__)


def _log_to_database_stub(stats: dict) -> None:
    """
    Placeholder for the SQLite persistence layer (not implemented yet -
    that's a later step, not Step 3). Intentionally a no-op so main.py's
    structure already has the right call site without inventing a schema
    now.
    """
    pass


def compute_processing_size(native_w: int, native_h: int, max_dimension: int) -> tuple:
    """
    Derives an aspect-ratio-preserving processing size from the source's real
    dimensions. Scales both dimensions by the SAME factor (so it can only
    ever uniformly shrink, never stretch/distort), capped so the longer edge
    never exceeds max_dimension. Never upscales a source smaller than the cap.

    Each dimension is then rounded DOWN to the nearest even number. Odd
    frame dimensions cause some video codecs (e.g. mp4v, used for the
    annotated output video) to silently coerce the width/height by 1px on
    write, so the actual saved video ends up a different size than what was
    requested. Rounding here keeps every stage of the pipeline (detector,
    tracker, counter, heatmap, and the output video writer) using the exact
    same dimensions throughout - no downstream size mismatch.

    Returns (processing_w, processing_h), each an even number >= 2.
    """
    scale = min(1.0, max_dimension / max(native_w, native_h))
    processing_w = max(1, round(native_w * scale))
    processing_h = max(1, round(native_h * scale))
    processing_w -= processing_w % 2
    processing_h -= processing_h % 2
    processing_w = max(2, processing_w)
    processing_h = max(2, processing_h)
    return processing_w, processing_h


def build_pipeline(line_start, line_end, line_margin, heatmap_width, heatmap_height, queue_roi):
    """Constructs the pipeline stages. Detector/tracker come straight from
    config.py; the counter's line, the heatmap's dimensions, and the queue
    ROI are passed in because they depend on this run's runtime-computed
    processing size."""
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
        line_start=line_start,
        line_end=line_end,
        margin=line_margin,
    )
    heatmap_gen = HeatmapGenerator(
        width=heatmap_width,
        height=heatmap_height,
        blur_kernel_size=config.HEATMAP_BLUR_KERNEL,
        point_radius=config.HEATMAP_POINT_RADIUS,
        decay_factor=config.HEATMAP_DECAY,
    )
    queue_predictor = None
    if config.QUEUE_ENABLED:
        queue_predictor = QueuePredictor(
            roi=queue_roi,
            arrival_window_seconds=config.QUEUE_ARRIVAL_WINDOW_SECONDS,
            service_window_seconds=config.QUEUE_SERVICE_WINDOW_SECONDS,
            min_service_rate=config.QUEUE_MIN_SERVICE_RATE,
        )
    return detector, tracker, line_counter, heatmap_gen, queue_predictor


def draw_overlay(frame, tracks, line_counter_stats, line_start, line_end, queue_roi=None, queue_stats=None):
    """Draws track boxes/IDs, the counting line, the queue ROI (if enabled),
    and running stats onto frame (in place)."""
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"ID {t['id']}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    ls = tuple(int(v) for v in line_start)
    le = tuple(int(v) for v in line_end)
    cv2.line(frame, ls, le, (0, 0, 255), 2)

    cv2.putText(
        frame,
        f"Entries: {line_counter_stats['entries']}  Exits: {line_counter_stats['exits']}  "
        f"Occupancy: {line_counter_stats['occupancy']}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )

    if queue_roi is not None:
        rx1, ry1, rx2, ry2 = [int(v) for v in queue_roi]
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)

    if queue_stats is not None:
        wait = queue_stats["estimated_wait_seconds"]
        wait_str = f"{wait:.0f}s" if wait is not None else "n/a"
        cv2.putText(
            frame,
            f"Queue: {queue_stats['queue_length']}  Est. wait: {wait_str}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2,
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

    # Read the actual first frame rather than trusting CAP_PROP_FRAME_WIDTH/
    # HEIGHT (unreliable on some backends/RTSP streams) - frame.shape is
    # ground truth for the real native size. This frame is then processed
    # normally below, not discarded.
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read any frames from source: {source}")
    native_h, native_w = first_frame.shape[:2]

    processing_w, processing_h = compute_processing_size(
        native_w, native_h, config.PROCESSING_MAX_DIMENSION
    )

    # Convert the config's fractional line geometry into concrete pixel
    # coordinates for THIS run's processing size - never a hardcoded size.
    line_start = (
        config.LINE_START_FRACTION[0] * processing_w,
        config.LINE_START_FRACTION[1] * processing_h,
    )
    line_end = (
        config.LINE_END_FRACTION[0] * processing_w,
        config.LINE_END_FRACTION[1] * processing_h,
    )
    line_margin = config.LINE_MARGIN_FRACTION * min(processing_w, processing_h)

    # Convert the config's fractional queue ROI into concrete pixel
    # coordinates for THIS run's processing size, same convention as the
    # counting line above.
    qx1, qy1, qx2, qy2 = config.QUEUE_ROI_FRACTION
    queue_roi = (qx1 * processing_w, qy1 * processing_h, qx2 * processing_w, qy2 * processing_h)

    logger.info(
        f"Source: {source} | native FPS: {fps:.2f} | total frames: {total_frames} | "
        f"native size: {native_w}x{native_h} -> processing size: {processing_w}x{processing_h} "
        f"(aspect ratio preserved, max_dimension={config.PROCESSING_MAX_DIMENSION})"
    )
    logger.info(
        f"Line geometry for this run: start={line_start} end={line_end} margin={line_margin:.1f}px"
    )

    detector, tracker, line_counter, heatmap_gen, queue_predictor = build_pipeline(
        line_start=line_start,
        line_end=line_end,
        line_margin=line_margin,
        heatmap_width=processing_w,
        heatmap_height=processing_h,
        queue_roi=queue_roi,
    )

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (processing_w, processing_h))

    display_available = display
    frame_idx = 0
    processed_idx = 0
    queue_stats = None  # defensive default in case the loop is interrupted before any frame is processed
    start_time = time.time()
    pending_frame = first_frame  # process the first frame we already read, then read() as normal

    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
                ret = True
            else:
                ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Resize FIRST so detector, tracker, counter, and heatmap all
            # operate in the exact same coordinate space (this run's
            # aspect-ratio-preserving processing size). Because
            # processing_w/processing_h were derived from this exact
            # source's own aspect ratio, this resize only ever uniformly
            # scales - it never stretches or distorts.
            frame = cv2.resize(frame, (processing_w, processing_h))

            if (frame_idx - 1) % config.FRAME_SKIP != 0:
                if writer is not None:
                    writer.write(frame)
                continue

            processed_idx += 1
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            line_counter.update(tracks)

            # Video-relative timestamp (frame_idx / native fps), not
            # wall-clock time - keeps arrival/service rates meaningful
            # whether a recorded file is processed slower or faster than
            # real time, and matches wall-clock behavior for a live camera.
            queue_stats = None
            if queue_predictor is not None:
                queue_stats = queue_predictor.update(tracks, frame_idx / fps)

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
            draw_overlay(frame, tracks, stats, line_start, line_end, queue_roi=queue_roi, queue_stats=queue_stats)

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
                    f"active_tracks={len(tracks)} stats={stats} queue_stats={queue_stats}"
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
        f"elapsed={elapsed:.1f}s stats={final_stats} final_queue_stats={queue_stats}"
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
