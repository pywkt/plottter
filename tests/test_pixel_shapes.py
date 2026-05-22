"""Tests for _pixel_shapes.cell_polygon (task 121.1).

Unit tests:
  - Vertex counts match spec: None/square, 4/diamond, 8/octagonal,
    24/circle, 28/rounded_square.

Integration test:
  - Running PixelArtGenerator with cell_shape="circle" on a small solid-black
    image, all output point coordinates must lie within the inscribed circle of
    each cell (distance from cell centre ≤ radius + 0.1 mm tolerance).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators._pixel_shapes import cell_polygon


# ---------------------------------------------------------------------------
# Unit tests — vertex counts
# ---------------------------------------------------------------------------


class TestCellPolygonVertexCounts:
    """Verify each shape returns the correct number of vertices."""

    def _verts(self, shape: str) -> list | None:
        return cell_polygon(shape, cell_x_mm=0.0, cell_y_mm=0.0, size_mm=10.0)

    def test_square_returns_none(self):
        assert self._verts("square") is None

    def test_unknown_shape_returns_none(self):
        assert cell_polygon("hex", 0.0, 0.0, 10.0) is None

    def test_diamond_has_4_vertices(self):
        verts = self._verts("diamond")
        assert verts is not None
        assert len(verts) == 4

    def test_octagonal_has_8_vertices(self):
        verts = self._verts("octagonal")
        assert verts is not None
        assert len(verts) == 8

    def test_circle_has_24_vertices(self):
        verts = self._verts("circle")
        assert verts is not None
        assert len(verts) == 24

    def test_rounded_square_has_28_vertices(self):
        verts = self._verts("rounded_square")
        assert verts is not None
        assert len(verts) == 28


# ---------------------------------------------------------------------------
# Unit tests — geometric correctness
# ---------------------------------------------------------------------------


class TestCellPolygonGeometry:
    """Spot-check the geometry of each shape for a 10 mm cell at (0, 0)."""

    SIZE = 10.0
    CX = SIZE / 2.0  # cell centre x
    CY = SIZE / 2.0  # cell centre y
    R = SIZE / 2.0   # inscribed circle radius

    def _verts(self, shape: str) -> list[tuple[float, float]]:
        v = cell_polygon("square" if shape == "sq" else shape, 0.0, 0.0, self.SIZE)
        assert v is not None
        return v

    # Diamond ----------------------------------------------------------------

    def test_diamond_vertices_are_at_edge_midpoints(self):
        verts = self._verts("diamond")
        xs = [x for x, _ in verts]
        ys = [y for _, y in verts]
        assert min(xs) == pytest.approx(0.0)
        assert max(xs) == pytest.approx(self.SIZE)
        assert min(ys) == pytest.approx(0.0)
        assert max(ys) == pytest.approx(self.SIZE)

    def test_diamond_all_vertices_on_circle(self):
        verts = self._verts("diamond")
        for x, y in verts:
            d = math.hypot(x - self.CX, y - self.CY)
            assert d == pytest.approx(self.R, abs=1e-9)

    # Octagonal --------------------------------------------------------------

    def test_octagonal_all_vertices_inside_or_on_bounding_square(self):
        verts = self._verts("octagonal")
        for x, y in verts:
            assert -1e-9 <= x <= self.SIZE + 1e-9
            assert -1e-9 <= y <= self.SIZE + 1e-9

    def test_octagonal_vertices_on_bounding_square_edges(self):
        """Each vertex must touch at least one bounding-square edge."""
        verts = self._verts("octagonal")
        EPS = 1e-9
        for x, y in verts:
            on_edge = (
                abs(x - 0.0) < EPS
                or abs(x - self.SIZE) < EPS
                or abs(y - 0.0) < EPS
                or abs(y - self.SIZE) < EPS
            )
            assert on_edge, f"Vertex ({x}, {y}) not on a bounding edge"

    # Circle -----------------------------------------------------------------

    def test_circle_all_vertices_on_radius(self):
        verts = self._verts("circle")
        for x, y in verts:
            d = math.hypot(x - self.CX, y - self.CY)
            assert d == pytest.approx(self.R, abs=1e-9)

    def test_circle_vertices_evenly_spaced(self):
        verts = self._verts("circle")
        # All adjacent arc distances should be equal.
        n = len(verts)
        dists = [
            math.hypot(verts[(i + 1) % n][0] - verts[i][0],
                       verts[(i + 1) % n][1] - verts[i][1])
            for i in range(n)
        ]
        assert max(dists) - min(dists) == pytest.approx(0.0, abs=1e-9)

    # Rounded square ---------------------------------------------------------

    def test_rounded_square_all_vertices_inside_bounding_square(self):
        verts = self._verts("rounded_square")
        for x, y in verts:
            assert -1e-9 <= x <= self.SIZE + 1e-9
            assert -1e-9 <= y <= self.SIZE + 1e-9

    def test_rounded_square_corner_arc_radii_consistent(self):
        """Every vertex must lie at distance == corner_radius from its arc centre."""
        r = self.SIZE * 0.2  # default corner_ratio (spec §7.5)
        verts = self._verts("rounded_square")
        # Arc centres for a 10 mm cell with r=2.0mm corner radius
        x0, y0 = 0.0, 0.0
        x1, y1 = self.SIZE, self.SIZE
        arc_centres = [
            (x1 - r, y0 + r),  # top-right
            (x1 - r, y1 - r),  # bottom-right
            (x0 + r, y1 - r),  # bottom-left
            (x0 + r, y0 + r),  # top-left
        ]
        pts_per_corner = 7
        for corner_idx, (acx, acy) in enumerate(arc_centres):
            start = corner_idx * pts_per_corner
            for v in verts[start:start + pts_per_corner]:
                d = math.hypot(v[0] - acx, v[1] - acy)
                assert d == pytest.approx(r, abs=1e-9), (
                    f"Corner {corner_idx} vertex {v} at dist {d} != r={r}"
                )


# ---------------------------------------------------------------------------
# Integration test — circle clipping via PixelArtGenerator
# ---------------------------------------------------------------------------


def _make_solid_black_image(size: int = 4) -> np.ndarray:
    """Return a small solid-black RGB image."""
    return np.zeros((size, size, 3), dtype=np.uint8)


def _make_canvas():
    from plottter.models.canvas import Canvas
    return Canvas.from_preset("A4", margin=10.0)


class TestCircleCellShapeIntegration:
    """Confirm that all output points lie within the inscribed circle."""

    TOLERANCE_MM = 0.1  # allowed overshoot beyond the true circle radius

    def _run_circle_generator(self) -> list:
        from plottter.generators.pixel_art import PixelArtGenerator

        gen = PixelArtGenerator()
        canvas = _make_canvas()
        params = {
            "_source_image": _make_solid_black_image(4),
            "grid_width": 4,
            "palette": "grayscale_2",  # just black/white — only black cells drawn
            "cell_shape": "circle",
            "cell_fill_style": "solid_hatch",
            "fill_density": 0.8,
            "cell_border": False,
            "cell_gap_mm": 0.0,
            "dithering": "none",
            "quantization": "nearest",
            "color_space": "rgb",
        }
        return gen.generate_layers(params, canvas)

    def test_circle_shape_emits_layers(self):
        specs = self._run_circle_generator()
        assert len(specs) > 0, "Expected at least one layer"

    def test_all_points_within_inscribed_circle(self):
        """Each output point must be within cell_radius + tolerance of its cell centre."""
        canvas = _make_canvas()
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()

        grid_width = 4
        draw_w = draw_x2 - draw_x1
        cell_size_mm = draw_w / grid_width
        cell_radius = cell_size_mm / 2.0

        specs = self._run_circle_generator()
        assert specs, "No layers emitted — cannot verify circle clipping"

        violations = []
        for spec in specs:
            for path in spec.paths:
                for x, y in path:
                    # Find which cell this point belongs to (by grid position)
                    col = int((x - draw_x1) / cell_size_mm)
                    row = int((y - draw_y1) / cell_size_mm)
                    # Clamp to valid grid range
                    col = max(0, min(grid_width - 1, col))
                    row = max(0, min(grid_width - 1, row))
                    cx = draw_x1 + (col + 0.5) * cell_size_mm
                    cy = draw_y1 + (row + 0.5) * cell_size_mm
                    dist = math.hypot(x - cx, y - cy)
                    if dist > cell_radius + self.TOLERANCE_MM:
                        violations.append((x, y, dist, cell_radius))

        assert not violations, (
            f"{len(violations)} point(s) outside inscribed circle: "
            f"first={violations[0]}"
        )
