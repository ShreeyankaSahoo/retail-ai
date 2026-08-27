"""
Accumulates customer centroid positions into a persistent density map and
renders it as a color-mapped OpenCV heatmap on demand (used by GET /heatmap).
"""
import logging
import threading
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    def __init__(
        self,
        width: int,
        height: int,
        blur_kernel_size: int = 25,
        point_radius: int = 15,
        decay_factor: float = 0.0,
    ):
        self.width = width
        self.height = height
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.point_radius = point_radius
        self.decay_factor = decay_factor  # 0 = cumulative all-time map
        self.accumulator = np.zeros((height, width), dtype=np.float32)
        self._lock = threading.Lock()

    def update(self, points: List[Tuple[float, float]]) -> None:
        """Adds a splat of weight at each centroid location for the current frame."""
        with self._lock:
            if self.decay_factor > 0:
                self.accumulator *= (1.0 - self.decay_factor)
            for x, y in points:
                xi, yi = int(x), int(y)
                if 0 <= xi < self.width and 0 <= yi < self.height:
                    cv2.circle(self.accumulator, (xi, yi), self.point_radius, 1.0, -1)

    def _render(self, background: Optional[np.ndarray] = None) -> np.ndarray:
        with self._lock:
            data = self.accumulator.copy()

        if data.max() > 0:
            blurred = cv2.GaussianBlur(data, (self.blur_kernel_size, self.blur_kernel_size), 0)
            normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            normalized = np.zeros((self.height, self.width), dtype=np.uint8)

        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        if background is not None:
            bg = cv2.resize(background, (self.width, self.height))
            colored = cv2.addWeighted(bg, 0.5, colored, 0.5, 0)

        return colored

    def get_png_bytes(self, background: Optional[np.ndarray] = None) -> bytes:
        image = self._render(background)
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode heatmap as PNG")
        return buffer.tobytes()

    def save(self, path: str, background: Optional[np.ndarray] = None) -> None:
        image = self._render(background)
        cv2.imwrite(path, image)
        logger.info(f"Heatmap snapshot saved to {path}")

    def reset(self) -> None:
        with self._lock:
            self.accumulator[:] = 0
        logger.info("Heatmap accumulator reset")

