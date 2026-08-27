"""
Lightweight multi-object tracker (SORT-style): Kalman filter motion model +
IOU-based greedy association. Uses only OpenCV + NumPy (no filterpy/scipy),
keeping the dependency footprint small for Raspberry Pi 4B.

Each track is assigned a persistent integer ID that survives brief occlusions
(up to MAX_TRACK_AGE frames), which is what makes accurate entry/exit counting
and heatmap trajectories possible.
"""
import logging
from typing import List, Tuple, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def bbox_to_measurement(bbox: List[float]) -> np.ndarray:
    """[x1,y1,x2,y2] -> [cx, cy, scale(area), aspect_ratio]"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    s = max(w * h, 1e-6)
    r = w / h if h > 0 else 1e-6
    return np.array([cx, cy, s, r], dtype=np.float32)


def state_to_bbox(state: np.ndarray) -> List[float]:
    """[cx, cy, scale, aspect_ratio, ...] -> [x1, y1, x2, y2]"""
    cx, cy, s, r = state[0], state[1], max(state[2], 1e-6), max(state[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w if w > 0 else 0.0
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class KalmanBoxTracker:
    """A single tracked person, modeled with a constant-velocity Kalman filter."""

    _next_id = 0

    def __init__(self, bbox: List[float]):
        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        # State: [cx, cy, s, r, vcx, vcy, vs] — 7D state, 4D measurement
        self.kf = cv2.KalmanFilter(7, 4, 0, cv2.CV_32F)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1.0
        self.kf.processNoiseCov[4:, 4:] *= 0.01  # velocity components change slowly
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 10.0

        z = bbox_to_measurement(bbox)
        self.kf.statePost = np.array(
            [z[0], z[1], z[2], z[3], 0, 0, 0], dtype=np.float32
        ).reshape(7, 1)

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self) -> List[float]:
        predicted = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return state_to_bbox(predicted.flatten())

    def update(self, bbox: List[float]) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        z = bbox_to_measurement(bbox).reshape(4, 1)
        self.kf.correct(z)

    def get_state(self) -> List[float]:
        return state_to_bbox(self.kf.statePost.flatten())


class Tracker:
    """SORT-style multi-object tracker managing a pool of KalmanBoxTracker instances."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, detections: List[Tuple[List[float], float]]) -> List[Dict]:
        """
        Advances all tracks by one frame and associates them with new detections.

        Returns:
            List of confirmed tracks: [{'id': int, 'bbox': [x1,y1,x2,y2]}, ...]
        """
        det_boxes = [d[0] for d in detections]
        predicted_boxes = [t.predict() for t in self.trackers]

        matches, unmatched_dets, _ = self._associate(predicted_boxes, det_boxes)

        for trk_idx, det_idx in matches:
            self.trackers[trk_idx].update(det_boxes[det_idx])

        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(det_boxes[det_idx]))

        # Drop tracks that have been lost for too long
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        results = []
        for t in self.trackers:
            if t.time_since_update == 0 and (t.hit_streak >= self.min_hits or t.age <= self.min_hits):
                results.append({"id": t.id, "bbox": t.get_state()})
        return results

    def _associate(self, predicted_boxes, det_boxes):
        if not predicted_boxes:
            return [], list(range(len(det_boxes))), []
        if not det_boxes:
            return [], [], list(range(len(predicted_boxes)))

        iou_matrix = np.zeros((len(predicted_boxes), len(det_boxes)), dtype=np.float32)
        for t_idx, tb in enumerate(predicted_boxes):
            for d_idx, db in enumerate(det_boxes):
                iou_matrix[t_idx, d_idx] = compute_iou(tb, db)

        pairs = sorted(
            (
                (iou_matrix[t, d], t, d)
                for t in range(iou_matrix.shape[0])
                for d in range(iou_matrix.shape[1])
            ),
            key=lambda x: x[0],
            reverse=True,
        )

        used_trks, used_dets, matches = set(), set(), []
        for iou_val, t_idx, d_idx in pairs:
            if iou_val < self.iou_threshold:
                break
            if t_idx in used_trks or d_idx in used_dets:
                continue
            matches.append((t_idx, d_idx))
            used_trks.add(t_idx)
            used_dets.add(d_idx)

        unmatched_trks = [i for i in range(len(predicted_boxes)) if i not in used_trks]
        unmatched_dets = [i for i in range(len(det_boxes)) if i not in used_dets]
        return matches, unmatched_dets, unmatched_trks

    def get_active_count(self) -> int:
        """Number of tracks currently matched to a detection this frame."""
        return len([t for t in self.trackers if t.time_since_update == 0])
