"""
Virtual line-crossing entry/exit counter.

Design notes on avoiding double counting:
- Each track_id's side of the line is remembered. A crossing event only fires
  on an actual side flip (-1 -> 1 or 1 -> -1), so a single continuous track
  can only register one event per physical crossing.
- A margin (dead-zone) around the line ignores centroids that are too close
  to the line, which prevents detector/tracker jitter from flipping the
  recorded side back and forth and firing spurious duplicate events.
"""
import logging
import threading
import time
from typing import List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LineCounter:
    def __init__(self, line_start: Tuple[int, int], line_end: Tuple[int, int], margin: float = 8.0):
        self.line_start = np.array(line_start, dtype=np.float32)
        self.line_end = np.array(line_end, dtype=np.float32)
        self.margin = margin

        self._track_sides: Dict[int, int] = {}
        self._last_seen: Dict[int, float] = {}

        self.total_entries = 0
        self.total_exits = 0
        self._lock = threading.Lock()

    def _side(self, point: Tuple[float, float]) -> int:
        """Returns +1, -1, or 0 (inside dead-zone) relative to the virtual line."""
        p = np.array(point, dtype=np.float32)
        d = self.line_end - self.line_start
        v = p - self.line_start
        cross = d[0] * v[1] - d[1] * v[0]
        length = float(np.linalg.norm(d)) + 1e-6
        distance = cross / length
        if distance > self.margin:
            return 1
        if distance < -self.margin:
            return -1
        return 0

    def update(self, tracks: List[Dict]) -> List[Dict]:
        """
        tracks: [{'id': int, 'bbox': [x1,y1,x2,y2]}, ...]
        Returns crossing events: [{'track_id', 'event_type', 'x', 'y'}, ...]
        """
        events = []
        now = time.time()
        with self._lock:
            for t in tracks:
                tid = t["id"]
                x1, y1, x2, y2 = t["bbox"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                self._last_seen[tid] = now

                side = self._side((cx, cy))
                if side == 0:
                    continue  # ambiguous zone — wait for a clearer reading

                prev_side = self._track_sides.get(tid)
                if prev_side is None:
                    self._track_sides[tid] = side
                    continue

                if side != prev_side:
                    if prev_side == -1 and side == 1:
                        self.total_entries += 1
                        events.append({"track_id": tid, "event_type": "entry", "x": cx, "y": cy})
                        logger.info(f"Track {tid}: ENTRY (total entries={self.total_entries})")
                    elif prev_side == 1 and side == -1:
                        self.total_exits += 1
                        events.append({"track_id": tid, "event_type": "exit", "x": cx, "y": cy})
                        logger.info(f"Track {tid}: EXIT (total exits={self.total_exits})")
                    self._track_sides[tid] = side
        return events

    def cleanup(self, max_idle_seconds: float = 300.0) -> None:
        """Periodically purge state for tracks not seen in a while (memory hygiene)."""
        now = time.time()
        with self._lock:
            stale = [tid for tid, ts in self._last_seen.items() if now - ts > max_idle_seconds]
            for tid in stale:
                self._track_sides.pop(tid, None)
                self._last_seen.pop(tid, None)
        if stale:
            logger.debug(f"Cleaned up {len(stale)} stale track states")

    def get_stats(self) -> Dict:
        with self._lock:
            entries, exits = self.total_entries, self.total_exits
        return {
            "entries": entries,
            "exits": exits,
            "occupancy": max(0, entries - exits),
        }
