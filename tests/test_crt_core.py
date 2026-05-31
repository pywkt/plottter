"""Tests for src/plottter/generators/_crt_core.py.

Covers every assertion in spec §10 "Core helpers" subsection.
"""

import math
import numpy as np
import pytest

from plottter.generators._crt_core import (
    barrel_warp,
    render_subpixel_rects,
    scanline_mask,
    subpixel_layout,
    vignette_mask,
)


# ---------------------------------------------------------------------------
# subpixel_layout — shadow_mask
# ---------------------------------------------------------------------------

class TestSubpixelLayoutShadowMask:
    """shadow_mask positions per spec §5."""

    def test_n3_pen0(self):
        x, y = subpixel_layout("shadow_mask", 0, 3)
        assert (x, y) == pytest.approx((0.50, 0.25))

    def test_n3_pen1(self):
        x, y = subpixel_layout("shadow_mask", 1, 3)
        assert (x, y) == pytest.approx((0.25, 0.75))

    def test_n3_pen2(self):
        x, y = subpixel_layout("shadow_mask", 2, 3)
        assert (x, y) == pytest.approx((0.75, 0.75))

    def test_n4_pen0(self):
        x, y = subpixel_layout("shadow_mask", 0, 4)
        assert (x, y) == pytest.approx((0.25, 0.25))

    def test_n4_pen1(self):
        x, y = subpixel_layout("shadow_mask", 1, 4)
        assert (x, y) == pytest.approx((0.75, 0.25))

    def test_n4_pen2(self):
        x, y = subpixel_layout("shadow_mask", 2, 4)
        assert (x, y) == pytest.approx((0.25, 0.75))

    def test_n4_pen3(self):
        x, y = subpixel_layout("shadow_mask", 3, 4)
        assert (x, y) == pytest.approx((0.75, 0.75))

    def test_n5_distributes_on_unit_circle_radius_035(self):
        """All 5 pens must lie on a circle of radius 0.35 centred at (0.5, 0.5)."""
        n = 5
        for i in range(n):
            x, y = subpixel_layout("shadow_mask", i, n)
            r = math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
            assert r == pytest.approx(0.35, abs=1e-9), (
                f"pen {i}: radius {r} != 0.35"
            )

    def test_n6_distributes_on_unit_circle_radius_035(self):
        n = 6
        for i in range(n):
            x, y = subpixel_layout("shadow_mask", i, n)
            r = math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
            assert r == pytest.approx(0.35, abs=1e-9)

    def test_n5_first_pen_at_top(self):
        """First pen (index 0) should be at the top (y smallest, x ≈ 0.5)."""
        x, y = subpixel_layout("shadow_mask", 0, 5)
        assert x == pytest.approx(0.5, abs=1e-9)
        assert y == pytest.approx(0.5 - 0.35, abs=1e-9)

    def test_n8_all_on_circle(self):
        n = 8
        for i in range(n):
            x, y = subpixel_layout("shadow_mask", i, n)
            r = math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
            assert r == pytest.approx(0.35, abs=1e-9)

    def test_n5_angles_evenly_spaced(self):
        """Angular step between consecutive pens should be 2π/n."""
        n = 5
        angles = []
        for i in range(n):
            x, y = subpixel_layout("shadow_mask", i, n)
            angles.append(math.atan2(y - 0.5, x - 0.5))
        # Wrap differences into (-π, π] and compare to 2π/n
        expected_step = 2 * math.pi / n
        for i in range(n - 1):
            diff = (angles[i + 1] - angles[i]) % (2 * math.pi)
            assert diff == pytest.approx(expected_step, abs=1e-9)


# ---------------------------------------------------------------------------
# subpixel_layout — aperture_grille
# ---------------------------------------------------------------------------

class TestSubpixelLayoutApertureGrille:
    """aperture_grille positions per spec §5."""

    def test_n4_all_at_y05(self):
        for i in range(4):
            x, y = subpixel_layout("aperture_grille", i, 4)
            assert y == pytest.approx(0.5)

    def test_n4_x_positions(self):
        """x = (i + 0.5) / 4 for i in 0..3 → 0.125, 0.375, 0.625, 0.875."""
        expected_x = [0.125, 0.375, 0.625, 0.875]
        for i, ex in enumerate(expected_x):
            x, y = subpixel_layout("aperture_grille", i, 4)
            assert x == pytest.approx(ex), f"pen {i}: x={x} != {ex}"

    def test_n3_x_positions(self):
        expected_x = [1 / 6, 3 / 6, 5 / 6]
        for i, ex in enumerate(expected_x):
            x, _ = subpixel_layout("aperture_grille", i, 3)
            assert x == pytest.approx(ex)

    def test_n6_all_at_y05(self):
        for i in range(6):
            _, y = subpixel_layout("aperture_grille", i, 6)
            assert y == pytest.approx(0.5)

    def test_n6_x_positions(self):
        for i in range(6):
            x, _ = subpixel_layout("aperture_grille", i, 6)
            assert x == pytest.approx((i + 0.5) / 6)


# ---------------------------------------------------------------------------
# subpixel_layout — slot_mask
# ---------------------------------------------------------------------------

class TestSubpixelLayoutSlotMask:
    """slot_mask alternates y by row parity per spec §5."""

    def test_even_row_y035(self):
        _, y = subpixel_layout("slot_mask", 0, 3, row=0)
        assert y == pytest.approx(0.35)

    def test_odd_row_y065(self):
        _, y = subpixel_layout("slot_mask", 0, 3, row=1)
        assert y == pytest.approx(0.65)

    def test_row2_is_even(self):
        _, y = subpixel_layout("slot_mask", 0, 3, row=2)
        assert y == pytest.approx(0.35)

    def test_row3_is_odd(self):
        _, y = subpixel_layout("slot_mask", 0, 3, row=3)
        assert y == pytest.approx(0.65)

    def test_x_same_as_aperture_grille(self):
        """x positions must match aperture_grille for the same (i, n_pens)."""
        for n in (3, 4, 6):
            for i in range(n):
                x_slot, _ = subpixel_layout("slot_mask", i, n, row=0)
                x_ag, _ = subpixel_layout("aperture_grille", i, n)
                assert x_slot == pytest.approx(x_ag), (
                    f"n={n}, i={i}: slot x={x_slot} != ag x={x_ag}"
                )

    def test_default_row_is_even(self):
        """Default row=0 should give even-row y."""
        _, y = subpixel_layout("slot_mask", 0, 3)
        assert y == pytest.approx(0.35)

    def test_all_pens_same_row_parity(self):
        """All pens in the same row share the same y parity."""
        for row in (0, 1, 4, 7):
            ys = {subpixel_layout("slot_mask", i, 4, row=row)[1] for i in range(4)}
            assert len(ys) == 1, f"row={row}: expected one y value, got {ys}"


# ---------------------------------------------------------------------------
# subpixel_layout — error handling
# ---------------------------------------------------------------------------

class TestSubpixelLayoutErrors:
    def test_negative_pen_index_raises(self):
        with pytest.raises(ValueError):
            subpixel_layout("shadow_mask", -1, 3)

    def test_pen_index_equals_n_pens_raises(self):
        with pytest.raises(ValueError):
            subpixel_layout("shadow_mask", 3, 3)

    def test_pen_index_exceeds_n_pens_raises(self):
        with pytest.raises(ValueError):
            subpixel_layout("aperture_grille", 5, 4)

    def test_unknown_mask_type_raises(self):
        with pytest.raises(ValueError):
            subpixel_layout("unknown_mask", 0, 3)


# ---------------------------------------------------------------------------
# scanline_mask
# ---------------------------------------------------------------------------

class TestScanlineMask:
    def test_intensity_zero_returns_all_ones(self):
        m = scanline_mask(8, 8, intensity=0.0, period=2)
        assert m.dtype == np.float32
        np.testing.assert_array_equal(m, np.ones((8, 8), dtype=np.float32))

    def test_intensity_one_period2_alternating(self):
        """intensity=1, period=2 → even rows 0.0, odd rows 1.0."""
        m = scanline_mask(6, 4, intensity=1.0, period=2)
        for r in range(6):
            expected = 0.0 if r % 2 == 0 else 1.0
            np.testing.assert_array_equal(
                m[r, :],
                np.full(4, expected, dtype=np.float32),
                err_msg=f"row {r}: expected {expected}",
            )

    def test_period3_zeros_every_third_row_only(self):
        """period=3 → rows 0, 3, 6 get (1-intensity), others stay 1.0."""
        rows, cols = 9, 5
        intensity = 0.6
        m = scanline_mask(rows, cols, intensity=intensity, period=3)
        for r in range(rows):
            if r % 3 == 0:
                np.testing.assert_allclose(
                    m[r, :],
                    np.full(cols, 1.0 - intensity, dtype=np.float32),
                    atol=1e-6,
                )
            else:
                np.testing.assert_array_equal(m[r, :], np.ones(cols, dtype=np.float32))

    def test_period4_only_quarter_rows_darkened(self):
        rows, cols = 8, 3
        m = scanline_mask(rows, cols, intensity=0.5, period=4)
        darkened_rows = [r for r in range(rows) if r % 4 == 0]
        normal_rows = [r for r in range(rows) if r % 4 != 0]
        for r in darkened_rows:
            np.testing.assert_allclose(m[r, :], np.full(cols, 0.5, dtype=np.float32), atol=1e-6)
        for r in normal_rows:
            np.testing.assert_array_equal(m[r, :], np.ones(cols, dtype=np.float32))

    def test_output_shape(self):
        m = scanline_mask(10, 7, intensity=0.5, period=2)
        assert m.shape == (10, 7)

    def test_output_dtype_float32(self):
        m = scanline_mask(4, 4, intensity=0.5, period=2)
        assert m.dtype == np.float32

    def test_intensity_half_period2(self):
        """intensity=0.5, period=2 → even rows 0.5, odd rows 1.0."""
        m = scanline_mask(4, 2, intensity=0.5, period=2)
        np.testing.assert_allclose(m[0, :], [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(m[1, :], [1.0, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# vignette_mask
# ---------------------------------------------------------------------------

class TestVignetteMask:
    def test_strength_zero_returns_all_ones(self):
        m = vignette_mask(10, 10, strength=0.0)
        np.testing.assert_allclose(m, np.ones((10, 10), dtype=np.float32), atol=1e-6)

    def test_output_dtype_float32(self):
        m = vignette_mask(8, 8, strength=0.5)
        assert m.dtype == np.float32

    def test_output_shape(self):
        m = vignette_mask(11, 7, strength=0.3)
        assert m.shape == (11, 7)

    def test_centre_pixel_is_one(self):
        """The exact centre pixel should be 1.0 for odd dimensions.

        With pixel-centred normalisation, odd rows/cols place an exact pixel
        at the midpoint ((rows-1)//2, (cols-1)//2) where norm == 0 → d2 == 0
        → v == 1.0.
        """
        rows, cols = 11, 11  # odd → exact centre pixel at (5, 5)
        m = vignette_mask(rows, cols, strength=0.8)
        cr, cc = rows // 2, cols // 2
        assert m[cr, cc] == pytest.approx(1.0, abs=1e-5)

    def test_corners_equal_one_minus_strength(self):
        """Corner pixels should be (1 - strength) because d² = 2 there.

        With pixel-centred normalisation, corners are at norm = ±1 exactly,
        so d2 = 2 and v = 1 - strength * min(1, 2/2) = 1 - strength.
        """
        rows, cols = 11, 11  # odd → corners exactly at norm = ±1
        strength = 0.6
        m = vignette_mask(rows, cols, strength=strength)
        expected = 1.0 - strength
        for r, c in [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]:
            assert m[r, c] == pytest.approx(expected, abs=1e-5), (
                f"corner ({r},{c}): got {m[r,c]}, expected {expected}"
            )

    def test_radially_symmetric(self):
        """The mask should be symmetric about both the horizontal and vertical axes.

        Pixel-centred normalisation ensures m[r,c] == m[r, cols-1-c] and
        m[r,c] == m[rows-1-r, c] for all (r,c).
        """
        rows, cols = 21, 21  # odd → exact centre + perfect flip symmetry
        m = vignette_mask(rows, cols, strength=0.5)
        # Flip horizontally
        np.testing.assert_allclose(m, m[:, ::-1], atol=1e-5)
        # Flip vertically
        np.testing.assert_allclose(m, m[::-1, :], atol=1e-5)

    def test_radially_symmetric_even(self):
        """Flip symmetry holds for even dimensions too."""
        rows, cols = 20, 20
        m = vignette_mask(rows, cols, strength=0.5)
        np.testing.assert_allclose(m, m[:, ::-1], atol=1e-5)
        np.testing.assert_allclose(m, m[::-1, :], atol=1e-5)

    def test_radially_symmetric_nonsquare(self):
        """Symmetry must hold for non-square images too."""
        rows, cols = 15, 21  # odd → exact symmetry
        m = vignette_mask(rows, cols, strength=0.4)
        np.testing.assert_allclose(m, m[:, ::-1], atol=1e-5)
        np.testing.assert_allclose(m, m[::-1, :], atol=1e-5)

    def test_values_between_zero_and_one(self):
        m = vignette_mask(20, 30, strength=1.0)
        assert float(m.min()) >= 0.0 - 1e-6
        assert float(m.max()) <= 1.0 + 1e-6

    def test_monotonically_decreasing_from_centre(self):
        """Values along a horizontal centre row should decrease toward edges."""
        rows, cols = 21, 21
        m = vignette_mask(rows, cols, strength=0.8)
        centre_row = m[rows // 2, :]
        mid = cols // 2
        # Left half should be non-increasing toward left edge
        for c in range(mid):
            assert centre_row[c] <= centre_row[c + 1] + 1e-6, (
                f"c={c}: {centre_row[c]} > {centre_row[c+1]}"
            )


# ---------------------------------------------------------------------------
# barrel_warp
# ---------------------------------------------------------------------------

class TestBarrelWarp:
    def _centre(self):
        return (50.0, 50.0)

    def _max_r(self):
        return 70.71  # half-diagonal of a 100×100 canvas

    def test_strength_zero_returns_input_unchanged(self):
        coords = np.array([[10.0, 20.0], [50.0, 50.0], [80.0, 90.0]])
        centre = self._centre()
        out = barrel_warp(coords, centre, strength=0.0, max_radius_mm=self._max_r())
        np.testing.assert_allclose(out, coords, atol=1e-9)

    def test_centre_point_unchanged_regardless_of_strength(self):
        centre = self._centre()
        coords = np.array([list(centre)])
        for s in (0.0, 0.05, 0.10, 0.15, 0.20, 1.0):
            out = barrel_warp(coords, centre, strength=s, max_radius_mm=self._max_r())
            np.testing.assert_allclose(
                out[0], list(centre), atol=1e-9,
                err_msg=f"Centre moved at strength={s}",
            )

    def test_edge_point_moves_outward(self):
        """A point away from the centre should have a larger radius after warp."""
        centre = self._centre()
        point = np.array([[80.0, 50.0]])  # on the right of centre
        out = barrel_warp(point, centre, strength=0.1, max_radius_mm=self._max_r())
        r_before = math.sqrt((point[0, 0] - centre[0]) ** 2 + (point[0, 1] - centre[1]) ** 2)
        r_after = math.sqrt((out[0, 0] - centre[0]) ** 2 + (out[0, 1] - centre[1]) ** 2)
        assert r_after > r_before, (
            f"Expected outward push: r_before={r_before}, r_after={r_after}"
        )

    def test_oversized_strength_clamped_to_015(self):
        """strength > 0.15 must be silently clamped; no exception raised."""
        centre = self._centre()
        point = np.array([[80.0, 50.0]])
        # Should not raise
        out_big = barrel_warp(point, centre, strength=1.0, max_radius_mm=self._max_r())
        out_capped = barrel_warp(point, centre, strength=0.15, max_radius_mm=self._max_r())
        np.testing.assert_allclose(out_big, out_capped, atol=1e-9)

    def test_strength_025_clamped_same_as_015(self):
        centre = self._centre()
        coords = np.array([[20.0, 30.0], [70.0, 80.0]])
        out_025 = barrel_warp(coords, centre, strength=0.25, max_radius_mm=self._max_r())
        out_015 = barrel_warp(coords, centre, strength=0.15, max_radius_mm=self._max_r())
        np.testing.assert_allclose(out_025, out_015, atol=1e-9)

    def test_output_shape_preserved(self):
        coords = np.random.rand(10, 2) * 100
        out = barrel_warp(coords, (50.0, 50.0), strength=0.1, max_radius_mm=70.0)
        assert out.shape == coords.shape

    def test_direction_preserved(self):
        """Warped points should remain in the same angular direction from centre."""
        centre = (50.0, 50.0)
        # Four points at the cardinal directions
        points = np.array([
            [80.0, 50.0],  # right
            [20.0, 50.0],  # left
            [50.0, 80.0],  # down
            [50.0, 20.0],  # up
        ])
        out = barrel_warp(points, centre, strength=0.1, max_radius_mm=70.0)
        for i in range(len(points)):
            dx_in = points[i, 0] - centre[0]
            dy_in = points[i, 1] - centre[1]
            dx_out = out[i, 0] - centre[0]
            dy_out = out[i, 1] - centre[1]
            # Same sign means same direction
            assert math.copysign(1, dx_in) == math.copysign(1, dx_out) or abs(dx_out) < 1e-9
            assert math.copysign(1, dy_in) == math.copysign(1, dy_out) or abs(dy_out) < 1e-9

    def test_strength_015_warp_is_nonzero(self):
        """Ensure clamped maximum (0.15) actually moves points."""
        centre = (50.0, 50.0)
        point = np.array([[90.0, 50.0]])
        out = barrel_warp(point, centre, strength=0.15, max_radius_mm=70.0)
        r_before = abs(point[0, 0] - centre[0])
        r_after = abs(out[0, 0] - centre[0])
        assert r_after > r_before


# ---------------------------------------------------------------------------
# render_subpixel_rects
# ---------------------------------------------------------------------------

class TestRenderSubpixelRects:
    def test_two_polylines_per_centre(self):
        coords = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        out = render_subpixel_rects(coords, width_mm=0.5, height_mm=2.0)
        assert len(out) == 6  # 2 polylines per centre × 3 centres

    def test_outline_is_5_point_closed_rect(self):
        coords = np.array([[10.0, 20.0]])
        out = render_subpixel_rects(coords, width_mm=0.4, height_mm=2.0)
        outline = out[0]
        assert len(outline) == 5
        assert outline[0] == outline[-1]  # closed
        # Width: max_x - min_x = 0.4
        xs = [p[0] for p in outline]
        assert max(xs) - min(xs) == pytest.approx(0.4)
        # Height: max_y - min_y = 2.0
        ys = [p[1] for p in outline]
        assert max(ys) - min(ys) == pytest.approx(2.0)

    def test_outline_centred_on_input_coord(self):
        coords = np.array([[10.0, 20.0]])
        out = render_subpixel_rects(coords, width_mm=0.4, height_mm=2.0)
        outline = out[0]
        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        # Centre = mean of min/max in both axes
        assert (min(xs) + max(xs)) / 2.0 == pytest.approx(10.0)
        assert (min(ys) + max(ys)) / 2.0 == pytest.approx(20.0)

    def test_fill_is_2_point_vertical_stroke(self):
        coords = np.array([[10.0, 20.0]])
        out = render_subpixel_rects(coords, width_mm=0.4, height_mm=2.0)
        fill = out[1]
        assert len(fill) == 2
        # Same x (vertical line)
        assert fill[0][0] == pytest.approx(fill[1][0])
        # Spans the full bar height
        assert abs(fill[1][1] - fill[0][1]) == pytest.approx(2.0)

    def test_empty_input_returns_empty_list(self):
        coords = np.empty((0, 2), dtype=np.float64)
        out = render_subpixel_rects(coords, width_mm=0.5, height_mm=2.0)
        assert out == []
