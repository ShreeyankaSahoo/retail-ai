"""
YOLOv8n-based person detector.
Restricted to the 'person' COCO class only, optimized for CPU inference on RPi4.
"""
import logging
from typing import List, Tuple

from ultralytics import YOLO

logger = logging.getLogger(__name__)


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

        logger.debug(f"Detected {len(detections)} person(s) in current frame")
        return detections
