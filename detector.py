"""
YOLOv8n-based person detector.
Restricted to the 'person' COCO class only, optimized for CPU inference on RPi4.
"""
import logging
from typing import List, Tuple

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def _box_area(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: List[float], b: List[float]) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment(smaller: List[float], larger: List[float]) -> float:
    """Fraction of the smaller box's area that is covered by the larger box."""
    xA, yA = max(smaller[0], larger[0]), max(smaller[1], larger[1])
    xB, yB = min(smaller[2], larger[2]), min(smaller[3], larger[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    smaller_area = _box_area(smaller)
    return inter / smaller_area if smaller_area > 0 else 0.0


def suppress_nested_duplicate_boxes(
    detections: List[Tuple[List[float], float]],
    min_iou: float = 0.35,
    containment_min: float = 0.65,
    aspect_larger_max: float = 0.60,
) -> List[Tuple[List[float], float]]:
    """
    Lightweight post-processing pass, run AFTER YOLO's own NMS, to remove
    redundant nested duplicate detections of the SAME physical person (e.g. a
    partial upper-body/head box that survives NMS alongside a full-body box
    for the same person because their IoU sits just under IOU_THRESHOLD).

    Only a handful of boxes are ever compared per frame (typically <10), so
    this pairwise check is negligible extra CPU - safe for a Raspberry Pi 4B.

    Rule (validated against real footage - see Phase 2 detector-tuning
    notes): for every pair of overlapping boxes with IoU >= min_iou, let
    `smaller`/`larger` be the lower-area / higher-area box. A true same-
    person duplicate consistently has the smaller box almost entirely
    covered by the larger one (containment >= containment_min) AND the
    larger box shaped like a single near-full-body crop - narrow relative to
    its height (aspect ratio = width/height <= aspect_larger_max). Two
    genuinely different people standing close together do NOT show this:
    the box spanning both of them ends up wide relative to its height
    (aspect ratio > aspect_larger_max) because it covers two shoulders/heads
    side-by-side rather than one head-to-feet column, so that case is left
    untouched. When the rule matches, the smaller (redundant) box is
    dropped and the larger (full-body) box is kept.
    """
    n = len(detections)
    if n < 2:
        return detections

    suppressed = set()
    for i in range(n):
        if i in suppressed:
            continue
        for j in range(i + 1, n):
            if j in suppressed:
                continue
            box_i, box_j = detections[i][0], detections[j][0]
            iou = _iou(box_i, box_j)
            if iou < min_iou:
                continue
            area_i, area_j = _box_area(box_i), _box_area(box_j)
            if area_i <= area_j:
                smaller_idx, smaller, larger = i, box_i, box_j
            else:
                smaller_idx, smaller, larger = j, box_j, box_i
            containment = _containment(smaller, larger)
            larger_w, larger_h = larger[2] - larger[0], larger[3] - larger[1]
            aspect_larger = larger_w / larger_h if larger_h > 0 else float("inf")
            if containment >= containment_min and aspect_larger <= aspect_larger_max:
                suppressed.add(smaller_idx)

    if not suppressed:
        return detections
    return [d for idx, d in enumerate(detections) if idx not in suppressed]


class PersonDetector:
    """Wraps an Ultralytics YOLOv8n model for single-class (person) detection."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        imgsz: int = 320,
        device: str = "cpu",
        person_class_id: int = 0,
    ):
        logger.info(f"Loading YOLOv8n model from '{model_path}' ...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.person_class_id = person_class_id
        logger.info("YOLOv8n model loaded successfully.")

    def detect(self, frame) -> List[Tuple[List[float], float]]:
        """
        Run inference on a single BGR frame.

        Returns:
            List of (bbox, confidence) where bbox = [x1, y1, x2, y2]
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )

        detections: List[Tuple[List[float], float]] = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                detections.append((box.tolist(), float(conf)))

        detections = suppress_nested_duplicate_boxes(detections)

        logger.debug(f"Detected {len(detections)} person(s) in current frame")
        return detections
