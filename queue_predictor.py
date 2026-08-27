"""
Lightweight queue length + wait-time estimator (Component 4, Tier 1).

Uses ONLY the tracking data the pipeline already produces (track IDs +
bounding boxes each frame) - no new model, no historical dataset, no
SQLite/API. A small rectangular Queue ROI (region of interest) distinguishes
"people currently in the queue" from general room occupancy (which
LineCounter already tracks separately).

Method: a simple queueing-theory heuristic derived from Little's Law
(L = lambda * W, i.e. average number in system = arrival rate * average wait
time). Rearranged for wait time: W = L / rate. We use the RECENT SERVICE
(departure-from-ROI) RATE as the throughput term, since that reflects the
queue's actual observed processing capacity better than the arrival rate
does when the queue is not yet in steady state - this is a standard
practical approximation of Little's Law, not a strict derivation, and is
documented as such rather than presented as an exact result.

    current_queue_length (L)   = number of currently-tracked people whose
                                  centroid falls inside the Queue ROI, right now.
    recent_arrival_rate  (lambda) = (# tracks that entered the ROI in the last
                                  QUEUE_ARRIVAL_WINDOW_SECONDS) / that window
    recent_service_rate  (mu)     = (# tracks that left the ROI in the last
                                  QUEUE_SERVICE_WINDOW_SECONDS) / that window
    estimated_wait_time  (W)   = L / mu   (only reported once mu is above a
                                  minimum-confidence floor - otherwise there
                                  is not enough recent departure evidence to
                                  trust an estimate, and None is returned)

All state is a few small deques of timestamps plus one dict of currently-
tracked ROI membership - negligible memory and CPU, well within Raspberry
Pi 4B budget, no ML/heavy dependency involved.

Known limitation, stated plainly: a track "leaving the ROI" is used as a
proxy for "finished being served." It cannot distinguish a person who was
actually served from one who simply walked out of frame/ROI without being
served (e.g. they left the queue, or tracking lost them). This is an
accepted simplification for Tier 1 and is not hidden.
"""
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueuePredictor:
    def __init__(
        self,
        roi: Tuple[float, float, float, float],
        arrival_window_seconds: float = 60.0,
        service_window_seconds: float = 60.0,
        min_service_rate: float = 1.0 / 300.0,
    ):
        """
        roi: (x1, y1, x2, y2) in the SAME pixel coordinate space as track
             bboxes for this run (i.e. already converted from
             config.QUEUE_ROI_FRACTION using the run's processing frame
             size, the same way main.py converts the counting line).
        arrival_window_seconds / service_window_seconds: how far back to
             look when computing recent rates. Independent windows because
             arrivals and departures can be observed at different natural
             cadences.
        min_service_rate: below this (people/second), there have been too
             few recent departures to trust a wait-time estimate, so
             estimated_wait_seconds is reported as None rather than a wild
             number from dividing by a near-zero rate.
        """
        self.roi = roi
        self.arrival_window_seconds = arrival_window_seconds
        self.service_window_seconds = service_window_seconds
        self.min_service_rate = min_service_rate

        self._in_roi_prev: Dict[int, Tuple[float, float]] = {}
        self._arrival_times: deque = deque()
        self._departure_times: deque = deque()

    def _in_roi(self, centroid: Tuple[float, float]) -> bool:
        x1, y1, x2, y2 = self.roi
        cx, cy = centroid
        return x1 <= cx <= x2 and y1 <= cy <= y2

    @staticmethod
    def _prune(times: deque, now: float, window_seconds: float) -> None:
        cutoff = now - window_seconds
        while times and times[0] < cutoff:
            times.popleft()

    def update(self, tracks: List[Dict], timestamp_seconds: float) -> Dict:
        """
        tracks: the SAME list main.py already gets from tracker.update() -
                [{'id': int, 'bbox': [x1,y1,x2,y2]}, ...]. No detector/
                tracker changes needed.
        timestamp_seconds: a monotonically increasing time value for this
                frame. main.py passes frame_idx / native_fps, i.e. video-
                relative time - this keeps arrival/service rates meaningful
                whether a recorded video is processed slower or faster than
                real time, and matches wall-clock behavior for a live camera.

        Returns a stats dict (see module docstring for the formulas).
        """
        in_roi_now: Dict[int, Tuple[float, float]] = {}
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if self._in_roi(centroid):
                in_roi_now[t["id"]] = centroid

        prev_ids = set(self._in_roi_prev.keys())
        now_ids = set(in_roi_now.keys())

        for tid in now_ids - prev_ids:
            self._arrival_times.append(timestamp_seconds)
            logger.debug(f"Queue: track {tid} entered ROI at t={timestamp_seconds:.1f}s")
        for tid in prev_ids - now_ids:
            self._departure_times.append(timestamp_seconds)
            logger.debug(f"Queue: track {tid} left ROI at t={timestamp_seconds:.1f}s")

        self._prune(self._arrival_times, timestamp_seconds, self.arrival_window_seconds)
        self._prune(self._departure_times, timestamp_seconds, self.service_window_seconds)

        self._in_roi_prev = in_roi_now

        queue_length = len(in_roi_now)
        arrival_rate = len(self._arrival_times) / self.arrival_window_seconds
        service_rate = len(self._departure_times) / self.service_window_seconds

        estimated_wait_seconds: Optional[float] = None
        if service_rate >= self.min_service_rate:
            estimated_wait_seconds = queue_length / service_rate

        return {
            "queue_length": queue_length,
            "arrival_rate_per_min": arrival_rate * 60.0,
            "service_rate_per_min": service_rate * 60.0,
            "estimated_wait_seconds": estimated_wait_seconds,
            "arrival_samples": len(self._arrival_times),
            "departure_samples": len(self._departure_times),
        }
