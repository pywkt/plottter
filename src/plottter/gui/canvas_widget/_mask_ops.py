"""_MaskOpsMixin — mask paint operation helpers for CanvasWidget."""
from __future__ import annotations

import math

import numpy as np

from .enums import _MASK_PX_PER_MM


class _MaskOpsMixin:
    """Mixin providing mask painting and shape-fill operations for CanvasWidget.

    Must not inherit from QObject.
    """

    def _ensure_mask(self) -> None:
        """Lazily create the mask array if it doesn't exist yet."""
        if self._mask_array is None:
            canvas = self._controller.current_project.canvas
            h = int(canvas.height_mm * _MASK_PX_PER_MM)
            w = int(canvas.width_mm * _MASK_PX_PER_MM)
            self._mask_array = np.zeros((h, w), dtype=np.float32)

    def _paint_at(self, x_mm: float, y_mm: float) -> None:
        """Stamp the brush at a mm position onto the mask array."""
        self._ensure_mask()
        assert self._mask_array is not None
        h, w = self._mask_array.shape

        cx = x_mm * _MASK_PX_PER_MM
        cy = y_mm * _MASK_PX_PER_MM
        radius_px = max(0.5, self._mask_brush_size_mm * _MASK_PX_PER_MM / 2.0)

        # Bounding box for the brush stamp
        x1 = max(0, int(cx - radius_px - 1))
        y1 = max(0, int(cy - radius_px - 1))
        x2 = min(w, int(cx + radius_px + 2))
        y2 = min(h, int(cy + radius_px + 2))
        if x1 >= x2 or y1 >= y2:
            return

        xs = np.arange(x1, x2, dtype=np.float32) - cx
        ys = np.arange(y1, y2, dtype=np.float32) - cy
        X, Y = np.meshgrid(xs, ys)
        dist = np.sqrt(X * X + Y * Y)

        hardness = self._mask_brush_hardness
        if hardness >= 0.999:
            stamp = (dist <= radius_px).astype(np.float32)
        else:
            # Gaussian falloff; sigma shrinks as hardness increases
            sigma = radius_px * (1.0 - hardness * 0.9) * 0.5
            if sigma < 0.01:
                stamp = (dist <= radius_px).astype(np.float32)
            else:
                stamp = np.exp(-(dist * dist) / (2.0 * sigma * sigma))
                # Scale so the centre has value 1.0 and clip
                stamp = np.clip(stamp / max(stamp.max(), 1e-6), 0.0, 1.0)

        patch = self._mask_array[y1:y2, x1:x2]
        if self._mask_erase:
            self._mask_array[y1:y2, x1:x2] = np.maximum(patch - stamp, 0.0)
        else:
            self._mask_array[y1:y2, x1:x2] = np.minimum(patch + stamp, 1.0)

        self.update()

    def _interpolate_stroke(
        self, last_pos: tuple[float, float], pos: tuple[float, float]
    ) -> None:
        """Paint brush stamps along the line from last_pos to pos."""
        dx = pos[0] - last_pos[0]
        dy = pos[1] - last_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        step = max(self._mask_brush_size_mm / 4.0, 0.1)
        if dist <= step:
            self._paint_at(*pos)
            return
        n_steps = max(1, int(dist / step))
        for i in range(1, n_steps + 1):
            t = i / n_steps
            self._paint_at(last_pos[0] + t * dx, last_pos[1] + t * dy)

    def _snapshot_mask(self) -> np.ndarray | None:
        """Return a copy of the current mask array, or None if no mask exists."""
        if self._mask_array is None:
            return None
        return self._mask_array.copy()

    def _handle_polygon_press(self, pos_mm: tuple[float, float]) -> None:
        """Add a vertex to the in-progress polygon."""
        if not self._polygon_vertices:
            # First vertex: save snapshot for undo
            self._pre_op_mask = self._snapshot_mask()
        self._polygon_vertices.append(pos_mm)
        self.update()

    def _apply_rectangle_mask(self) -> None:
        """Fill a hard-edged rectangle into the mask array (or erase)."""
        if self._shape_start_mm is None or self._shape_end_mm is None:
            return

        x1_mm, y1_mm = self._shape_start_mm
        x2_mm, y2_mm = self._shape_end_mm

        # Compute unclamped pixel bounds; reject degenerate shapes before
        # allocating the mask (avoids creating a zeros-array on a bare click).
        raw_col1 = int(round(min(x1_mm, x2_mm) * _MASK_PX_PER_MM))
        raw_col2 = int(round(max(x1_mm, x2_mm) * _MASK_PX_PER_MM))
        raw_row1 = int(round(min(y1_mm, y2_mm) * _MASK_PX_PER_MM))
        raw_row2 = int(round(max(y1_mm, y2_mm) * _MASK_PX_PER_MM))
        if raw_col1 >= raw_col2 or raw_row1 >= raw_row2:
            return

        self._ensure_mask()
        assert self._mask_array is not None
        h, w = self._mask_array.shape

        col1 = max(0, raw_col1)
        row1 = max(0, raw_row1)
        col2 = min(w, raw_col2)
        row2 = min(h, raw_row2)

        if col1 >= col2 or row1 >= row2:
            return

        if self._mask_erase:
            self._mask_array[row1:row2, col1:col2] = 0.0
        else:
            self._mask_array[row1:row2, col1:col2] = 1.0

    def _apply_ellipse_mask(self) -> None:
        """Fill a hard-edged ellipse into the mask array (or erase)."""
        if self._shape_start_mm is None or self._shape_end_mm is None:
            return
        self._ensure_mask()
        assert self._mask_array is not None
        h, w = self._mask_array.shape

        import cv2  # available project-wide dependency

        x1_mm, y1_mm = self._shape_start_mm
        x2_mm, y2_mm = self._shape_end_mm

        x1 = min(x1_mm, x2_mm) * _MASK_PX_PER_MM
        y1 = min(y1_mm, y2_mm) * _MASK_PX_PER_MM
        x2 = max(x1_mm, x2_mm) * _MASK_PX_PER_MM
        y2 = max(y1_mm, y2_mm) * _MASK_PX_PER_MM

        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        ax = max(1, int(round((x2 - x1) / 2)))
        ay = max(1, int(round((y2 - y1) / 2)))

        ellipse_buf = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(ellipse_buf, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        if self._mask_erase:
            self._mask_array[ellipse_buf > 0] = 0.0
        else:
            self._mask_array[ellipse_buf > 0] = 1.0

    def _apply_polygon_mask(self) -> None:
        """Fill a hard-edged polygon into the mask array (or erase)."""
        if len(self._polygon_vertices) < 3:
            return
        self._ensure_mask()
        assert self._mask_array is not None
        h, w = self._mask_array.shape

        import cv2

        pts = np.array(
            [
                (int(round(x * _MASK_PX_PER_MM)), int(round(y * _MASK_PX_PER_MM)))
                for x, y in self._polygon_vertices
            ],
            dtype=np.int32,
        )

        poly_buf = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_buf, [pts], 255)

        if self._mask_erase:
            self._mask_array[poly_buf > 0] = 0.0
        else:
            self._mask_array[poly_buf > 0] = 1.0

    def _apply_pen_mask(self) -> None:
        """Fill a hard-edged lasso (freeform closed shape) into the mask array (or erase)."""
        if len(self._pen_points) < 3:
            return
        self._ensure_mask()
        assert self._mask_array is not None
        h, w = self._mask_array.shape

        import cv2

        pts = np.array(
            [
                (int(round(x * _MASK_PX_PER_MM)), int(round(y * _MASK_PX_PER_MM)))
                for x, y in self._pen_points
            ],
            dtype=np.int32,
        )

        pen_buf = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(pen_buf, [pts], 255)

        if self._mask_erase:
            self._mask_array[pen_buf > 0] = 0.0
        else:
            self._mask_array[pen_buf > 0] = 1.0
