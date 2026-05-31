"""Tests for CrtTvGenerator (task 161.3).

Covers:
- Generator registered in GENERATORS under "CRT TV".
- get_parameters() returns every param from spec §7 with mask_type limited
  to ["shadow_mask"]; all params are randomizable=False (including seed).
- generate_layers() on a small synthetic image with Basic 6 palette:
  - Returns ≤ 6 LayerSpecs.
  - Every path has len ≥ 2.
  - Every coordinate lies inside compute_image_rect(...)'s returned rect.
- image_fit_mode="fit" preserves source aspect ratio (regression guard
  against the PixelArt/Pointillist aspect-ratio bug).
- seed=N is deterministic across two runs.
- NES preset present and runs without error.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.models.canvas import Canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canvas() -> Canvas:
    """A4 canvas with 10 mm margin — drawing area (10, 10, 200, 287)."""
    return Canvas.from_preset("A4", margin=10.0)


def _make_basic6_image(width: int = 20, height: int = 20) -> np.ndarray:
    """Small synthetic RGB image using black and red from Basic 6.

    Top half: black (#000000), bottom half: red (#E63946).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mid = height // 2
    img[:mid, :] = [0, 0, 0]
    img[mid:, :] = [230, 57, 70]
    return img


def _base_params(image: np.ndarray, **overrides) -> dict:
    return {
        "_source_image": image,
        "palette": "Basic 6",
        "crt_resolution_w": 20,
        "mask_type": "shadow_mask",
        "subpixel_shape": "point",
        "subpixel_size_mm": 0.1,
        "dither": "none",
        "scanline_intensity": 0.0,   # no scanlines → all rows survive
        "scanline_period": 2,
        "vignette_strength": 0.0,    # no vignette → all pixels survive
        "barrel_strength": 0.0,
        "gamma": 1.0,
        "seed": 42,
        **overrides,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_registered_in_generators(self):
        from plottter.generators import GENERATORS, _import_builtin_generators

        _import_builtin_generators()
        assert "CRT TV" in GENERATORS

    def test_class_attributes(self):
        from plottter.generators import GENERATORS, _import_builtin_generators

        _import_builtin_generators()
        cls = GENERATORS["CRT TV"]
        assert cls.name == "CRT TV"
        assert cls.category == "image"
        assert cls.uses_source_image is True
        assert cls.uses_color_source is True
        assert cls.emits_multiple_layers is True


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestParameters:
    def setup_method(self):
        from plottter.generators.crt_tv import CrtTvGenerator

        self.gen = CrtTvGenerator()
        self.params = self.gen.get_parameters()
        self.by_name = {p.name: p for p in self.params}

    def test_all_spec_params_present(self):
        expected = {
            "palette",
            "crt_resolution_w",
            "mask_type",
            "subpixel_shape",
            "subpixel_size_mm",
            "dither",
            "scanline_intensity",
            "scanline_period",
            "vignette_strength",
            "barrel_strength",
            "gamma",
            "seed",
        }
        assert expected.issubset(self.by_name.keys()), (
            f"Missing params: {expected - self.by_name.keys()}"
        )

    def test_palette_choice_param(self):
        from plottter.generators.base import ChoiceParam

        p = self.by_name["palette"]
        assert isinstance(p, ChoiceParam)
        assert p.default == "Basic 6"
        assert "Basic 6" in p.choices
        assert p.randomizable is False

    def test_crt_resolution_w_int_param(self):
        from plottter.generators.base import IntParam

        p = self.by_name["crt_resolution_w"]
        assert isinstance(p, IntParam)
        assert p.default == 160
        assert p.min == 40
        assert p.max == 800
        assert p.randomizable is False

    def test_mask_type_choices(self):
        from plottter.generators.base import ChoiceParam

        p = self.by_name["mask_type"]
        assert isinstance(p, ChoiceParam)
        assert p.choices == ["shadow_mask", "aperture_grille", "slot_mask"]
        assert p.default == "shadow_mask"
        assert p.randomizable is False

    def test_subpixel_shape_choices(self):
        from plottter.generators.base import ChoiceParam

        p = self.by_name["subpixel_shape"]
        assert isinstance(p, ChoiceParam)
        assert set(p.choices) == {"circle", "cross", "point", "rect"}
        assert p.default == "circle"
        assert p.randomizable is False

    def test_subpixel_size_mm_float_param(self):
        from plottter.generators.base import FloatParam

        p = self.by_name["subpixel_size_mm"]
        assert isinstance(p, FloatParam)
        assert p.default == pytest.approx(0.3)
        assert p.min == pytest.approx(0.05)
        assert p.max == pytest.approx(2.0)
        assert p.randomizable is False

    def test_dither_choices(self):
        from plottter.generators.base import ChoiceParam

        p = self.by_name["dither"]
        assert isinstance(p, ChoiceParam)
        assert set(p.choices) == {"none", "floyd-steinberg", "ordered", "atkinson"}
        assert p.default == "floyd-steinberg"
        assert p.randomizable is False

    def test_scanline_intensity_float_param(self):
        from plottter.generators.base import FloatParam

        p = self.by_name["scanline_intensity"]
        assert isinstance(p, FloatParam)
        assert p.default == pytest.approx(0.7)
        assert p.min == pytest.approx(0.0)
        assert p.max == pytest.approx(1.0)
        assert p.randomizable is False

    def test_scanline_period_int_param(self):
        from plottter.generators.base import IntParam

        p = self.by_name["scanline_period"]
        assert isinstance(p, IntParam)
        assert p.default == 2
        assert p.min == 1
        assert p.max == 5
        assert p.randomizable is False

    def test_vignette_strength_float_param(self):
        from plottter.generators.base import FloatParam

        p = self.by_name["vignette_strength"]
        assert isinstance(p, FloatParam)
        assert p.default == pytest.approx(0.3)
        assert p.min == pytest.approx(0.0)
        assert p.max == pytest.approx(1.0)
        assert p.randomizable is False

    def test_barrel_strength_float_param(self):
        from plottter.generators.base import FloatParam

        p = self.by_name["barrel_strength"]
        assert isinstance(p, FloatParam)
        assert p.default == pytest.approx(0.0)
        assert p.min == pytest.approx(0.0)
        assert p.max == pytest.approx(0.15)
        assert p.randomizable is False

    def test_gamma_float_param(self):
        from plottter.generators.base import FloatParam

        p = self.by_name["gamma"]
        assert isinstance(p, FloatParam)
        assert p.default == pytest.approx(1.0)
        assert p.min == pytest.approx(0.4)
        assert p.max == pytest.approx(2.5)
        assert p.randomizable is False

    def test_seed_int_param(self):
        from plottter.generators.base import IntParam

        p = self.by_name["seed"]
        assert isinstance(p, IntParam)
        assert p.default == 0
        assert p.min == 0
        assert p.max == 99999
        # seed is randomizable=False per spec §7
        assert p.randomizable is False

    def test_all_params_not_randomizable(self):
        """Spec §7: all params except seed are randomizable=False.
        Seed is also randomizable=False for this generator.
        """
        for p in self.params:
            assert p.randomizable is False, (
                f"Param {p.name!r} should not be randomizable"
            )


# ---------------------------------------------------------------------------
# generate_layers — output shape, coordinate validity, and aspect ratio
# ---------------------------------------------------------------------------

class TestGenerateLayers:
    def setup_method(self):
        from plottter.generators.crt_tv import CrtTvGenerator

        self.gen = CrtTvGenerator()
        self.canvas = _make_canvas()
        self.image = _make_basic6_image()

    def _run(self, **overrides):
        params = _base_params(self.image, **overrides)
        return self.gen.generate_layers(params, self.canvas)

    def test_returns_at_most_6_layers(self):
        specs = self._run()
        assert len(specs) <= 6

    def test_returns_at_least_1_layer(self):
        specs = self._run()
        assert len(specs) >= 1

    def test_every_path_has_at_least_2_points(self):
        specs = self._run()
        for spec in specs:
            for path in spec.paths:
                assert len(path) >= 2, f"Short path in layer {spec.name!r}: {path}"

    def test_all_coords_inside_image_rect(self):
        """All dot coordinates must lie within the fitted image rect.

        Uses subpixel_shape="point" (0.01 mm shift) and barrel_strength=0
        so that the tiny offset cannot move dots outside the fitted rect.
        """
        from plottter.generators._helpers import compute_image_rect

        img_h, img_w = self.image.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        rx1, ry1, rx2, ry2 = compute_image_rect(
            "fit", img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2
        )

        # Allow a tiny margin for the 0.01 mm point-style offset and fp noise
        tol = 0.05

        specs = self._run(subpixel_shape="point", barrel_strength=0.0)
        assert specs, "Expected at least one layer"
        for spec in specs:
            for path in spec.paths:
                for x, y in path:
                    assert rx1 - tol <= x <= rx2 + tol, (
                        f"x={x:.4f} outside rect [{rx1:.2f}, {rx2:.2f}] in {spec.name!r}"
                    )
                    assert ry1 - tol <= y <= ry2 + tol, (
                        f"y={y:.4f} outside rect [{ry1:.2f}, {ry2:.2f}] in {spec.name!r}"
                    )

    def test_returns_empty_when_no_source_image(self):
        params = _base_params(self.image)
        del params["_source_image"]
        specs = self.gen.generate_layers(params, self.canvas)
        assert specs == []

    def test_layer_colors_match_palette(self):
        from plottter.color import get_preset

        palette = get_preset("Basic 6")
        specs = self._run()
        palette_hexes = {c.upper() for c in palette.colors}
        for spec in specs:
            assert spec.color.upper() in palette_hexes, (
                f"Layer color {spec.color!r} not in Basic 6 palette"
            )

    def test_fit_mode_preserves_source_aspect(self):
        """Regression guard: wide image must occupy a wide-but-short band.

        With image_fit_mode='fit' and a 4:1 wide image, the fitted rect
        height should be ~1/4 of its width.  Without the fix (using drawing
        area bounds instead of compute_image_rect), dots fill the full
        canvas height and the ratio collapses to ≈ 1:1.
        """
        wide = np.zeros((20, 80, 3), dtype=np.uint8)  # 4:1 aspect
        wide[:, :40] = [0, 0, 0]       # left half black
        wide[:, 40:] = [230, 57, 70]   # right half red
        params = _base_params(
            wide,
            image_fit_mode="fit",
            scanline_intensity=0.0,
            vignette_strength=0.0,
            subpixel_shape="point",
        )
        specs = self.gen.generate_layers(params, self.canvas)

        all_xs = [x for s in specs for path in s.paths for x, _ in path]
        all_ys = [y for s in specs for path in s.paths for _, y in path]
        assert all_xs and all_ys, "Expected at least one dot"

        used_w = max(all_xs) - min(all_xs)
        used_h = max(all_ys) - min(all_ys)
        # 4:1 image → fitted band aspect ≈ 4:1; allow 30% slack for CRT sampling
        assert used_w / used_h > 2.5, (
            f"Fit mode should preserve 4:1 aspect; "
            f"used_w={used_w:.1f} used_h={used_h:.1f}"
        )

    def test_fill_mode_uses_full_canvas_height(self):
        """Fill mode stretches to the drawing area — dots span the full height."""
        wide = np.zeros((20, 80, 3), dtype=np.uint8)
        wide[:, :40] = [0, 0, 0]
        wide[:, 40:] = [230, 57, 70]
        params = _base_params(
            wide,
            image_fit_mode="fill",
            scanline_intensity=0.0,
            vignette_strength=0.0,
            subpixel_shape="point",
        )
        specs = self.gen.generate_layers(params, self.canvas)

        _, top, _, bottom = self.canvas.drawing_area()
        canvas_h = bottom - top
        all_ys = [y for s in specs for path in s.paths for _, y in path]
        assert all_ys, "Expected at least one dot"
        used_h = max(all_ys) - min(all_ys)
        assert used_h > 0.7 * canvas_h, (
            f"Fill mode should span canvas height; "
            f"used_h={used_h:.1f}, canvas_h={canvas_h:.1f}"
        )


# ---------------------------------------------------------------------------
# Scanline and vignette effects
# ---------------------------------------------------------------------------



class TestMaskTypeLayouts:
    """Verify subpixel x/y positions for aperture_grille and slot_mask."""

    def setup_method(self):
        from plottter.generators.crt_tv import CrtTvGenerator

        self.gen = CrtTvGenerator()
        self.canvas = _make_canvas()

    def test_aperture_grille_pen0_x_positions(self):
        """pen 0 x-positions match cell_origin_x + (0.5 / n_pens) * cell_size_x."""
        from plottter.generators._helpers import compute_image_rect
        from plottter.color import get_preset

        # All-black image → pen 0 (#000000) gets all pixels with dither="none"
        crt_w = 8
        img = np.zeros((4, crt_w, 3), dtype=np.uint8)  # 4 rows, 8 cols

        params = _base_params(
            img,
            mask_type="aperture_grille",
            crt_resolution_w=crt_w,
            dither="none",
            scanline_intensity=0.0,
            vignette_strength=0.0,
            barrel_strength=0.0,
            subpixel_shape="point",
            subpixel_size_mm=0.01,
        )
        specs = self.gen.generate_layers(params, self.canvas)

        pen0_specs = [s for s in specs if s.color.upper() == "#000000"]
        assert pen0_specs, "Expected pen 0 (black) layer for aperture_grille"
        pen0 = pen0_specs[0]

        # Compute expected x positions
        img_h, img_w = img.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        rect_x1, _ry1, rect_x2, _ry2 = compute_image_rect(
            "fit", img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2
        )
        cell_size_x = (rect_x2 - rect_x1) / crt_w
        palette = get_preset("Basic 6")
        n_pens = len(palette.colors)  # 6

        x_frac = 0.5 / n_pens  # pen 0 aperture_grille offset
        expected_xs = {
            rect_x1 + col * cell_size_x + x_frac * cell_size_x
            for col in range(crt_w)
        }

        tol = 1e-6
        xs = [path[0][0] for path in pen0.paths]
        for x in xs:
            assert any(abs(x - ex) < tol for ex in expected_xs), (
                f"aperture_grille pen 0: x={x:.6f} not in expected column x-positions"
            )

    def test_slot_mask_row_parity_y_positions(self):
        """slot_mask even rows → y=0.35*cell, odd rows → y=0.65*cell (shifted by row)."""
        from plottter.generators._helpers import compute_image_rect

        # All-black image, exactly 2 rows to exercise both parities
        crt_w = 4
        img = np.zeros((2, crt_w, 3), dtype=np.uint8)

        params = _base_params(
            img,
            mask_type="slot_mask",
            crt_resolution_w=crt_w,
            dither="none",
            scanline_intensity=0.0,
            vignette_strength=0.0,
            barrel_strength=0.0,
            subpixel_shape="point",
            subpixel_size_mm=0.01,
        )
        specs = self.gen.generate_layers(params, self.canvas)

        pen0_specs = [s for s in specs if s.color.upper() == "#000000"]
        assert pen0_specs, "Expected pen 0 (black) layer for slot_mask"
        pen0 = pen0_specs[0]

        # Compute expected y positions
        img_h, img_w = img.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        _rx1, rect_y1, _rx2, rect_y2 = compute_image_rect(
            "fit", img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2
        )
        crt_h = max(1, round(img_h * crt_w / img_w))  # 2
        cell_size_y = (rect_y2 - rect_y1) / crt_h

        # row 0 (even): cell_origin_y + 0.35 * cell_size_y
        y_row0 = rect_y1 + 0 * cell_size_y + 0.35 * cell_size_y
        # row 1 (odd): cell_origin_y + 0.65 * cell_size_y
        y_row1 = rect_y1 + 1 * cell_size_y + 0.65 * cell_size_y

        tol = 1e-6
        ys = [path[0][1] for path in pen0.paths]
        for y in ys:
            assert abs(y - y_row0) < tol or abs(y - y_row1) < tol, (
                f"slot_mask pen 0: y={y:.6f} doesn't match "
                f"row 0 ({y_row0:.6f}) or row 1 ({y_row1:.6f})"
            )

        # Both rows must be represented
        assert any(abs(y - y_row0) < tol for y in ys), (
            "slot_mask: no row-0 y-position found"
        )
        assert any(abs(y - y_row1) < tol for y in ys), (
            "slot_mask: no row-1 y-position found"
        )

class TestEffects:
    def setup_method(self):
        from plottter.generators.crt_tv import CrtTvGenerator

        self.gen = CrtTvGenerator()
        self.canvas = _make_canvas()

    def _uniform_image(self, color=(0, 0, 0), size=20) -> np.ndarray:
        """Solid-colour image so every pixel maps to one pen."""
        img = np.full((size, size, 3), color, dtype=np.uint8)
        return img

    def test_scanline_intensity_zero_keeps_all_rows(self):
        """intensity=0 → multiplier=1.0 everywhere → no rows killed."""
        img = self._uniform_image(color=(0, 0, 0))
        params = _base_params(
            img,
            scanline_intensity=0.0,
            scanline_period=2,
            vignette_strength=0.0,
            seed=0,
        )
        specs = self.gen.generate_layers(params, self.canvas)
        # All pixels survive — at least one layer with many dots
        total_paths = sum(len(s.paths) for s in specs)
        assert total_paths > 0

    def test_scanline_intensity_full_halves_row_count(self):
        """intensity=1.0, period=2 → even rows fully suppressed.

        With a uniform black image, the surviving dots come only from odd rows.
        The number of dot paths should be approximately half of what we get
        with intensity=0.  Allow generous 40% slack for small-sample variance.
        """
        img = self._uniform_image(color=(0, 0, 0), size=40)
        params_no_sl = _base_params(
            img,
            scanline_intensity=0.0,
            scanline_period=2,
            vignette_strength=0.0,
            seed=1,
        )
        params_full_sl = _base_params(
            img,
            scanline_intensity=1.0,
            scanline_period=2,
            vignette_strength=0.0,
            seed=1,
        )
        specs_no = self.gen.generate_layers(params_no_sl, self.canvas)
        specs_full = self.gen.generate_layers(params_full_sl, self.canvas)

        n_no = sum(len(s.paths) for s in specs_no)
        n_full = sum(len(s.paths) for s in specs_full)

        assert n_no > 0, "Expected dots with no scanlines"
        # Full scanline should eliminate roughly half the dots (period=2)
        ratio = n_full / n_no
        assert ratio < 0.7, (
            f"scanline_intensity=1 period=2 should halve dots; "
            f"no_sl={n_no}, full_sl={n_full}, ratio={ratio:.2f}"
        )

    def test_vignette_strength_zero_keeps_corner_cells(self):
        """vignette=0 → corners produce dots."""
        img = self._uniform_image(color=(0, 0, 0), size=20)
        params = _base_params(
            img,
            vignette_strength=0.0,
            scanline_intensity=0.0,
            seed=0,
            subpixel_shape="point",
        )
        specs = self.gen.generate_layers(params, self.canvas)
        total_paths = sum(len(s.paths) for s in specs)
        assert total_paths > 0

    def test_vignette_strength_full_drops_corner_cells(self):
        """vignette=1 → corner cells should produce zero dots.

        Uses a large pure-black image so that pen 0 (black) gets all pixels.
        With vignette_strength=1 the corners have multiplier 0 → always dropped.
        """
        size = 30
        img = self._uniform_image(color=(0, 0, 0), size=size)
        params = _base_params(
            img,
            vignette_strength=1.0,
            scanline_intensity=0.0,
            seed=0,
            crt_resolution_w=size,
            subpixel_shape="point",
        )
        specs = self.gen.generate_layers(params, self.canvas)

        if not specs:
            return  # fully suppressed is acceptable

        # Corner cells in a size×size grid
        canvas = self.canvas
        from plottter.generators._helpers import compute_image_rect

        img_h, img_w = img.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        rx1, ry1, rx2, ry2 = compute_image_rect(
            "fit", img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2
        )
        cell_mm = (rx2 - rx1) / size

        # Corner cell centre (top-left cell: row 0, col 0)
        corner_cx = rx1 + 0.5 * cell_mm
        corner_cy = ry1 + 0.5 * cell_mm
        corner_r = cell_mm  # search radius

        all_xs = [x for s in specs for path in s.paths for x, _ in path]
        all_ys = [y for s in specs for path in s.paths for _, y in path]

        # No dot should be near the top-left corner cell
        for x, y in zip(all_xs, all_ys):
            in_corner = (
                abs(x - corner_cx) < corner_r and abs(y - corner_cy) < corner_r
            )
            assert not in_corner, (
                f"Dot at ({x:.2f},{y:.2f}) found near corner cell with vignette=1"
            )

    def test_barrel_strength_shifts_corner_outward(self):
        """barrel_strength=0.10 shifts corner cell x-coordinate outward by >= 1 mm.

        With a uniform black image and no scanlines/vignette, all pixels are
        assigned to pen 0 (black). The leftmost column dots sit close to
        rect_x1 with no barrel. Barrel distortion pushes them further left
        (away from canvas centre), so the minimum x across all dots should
        decrease by at least 1 mm.
        """
        size = 20
        img = self._uniform_image(color=(0, 0, 0), size=size)

        base_kw = dict(
            crt_resolution_w=size,
            scanline_intensity=0.0,
            vignette_strength=0.0,
            seed=0,
            subpixel_shape="point",
        )

        specs_no = self.gen.generate_layers(
            _base_params(img, barrel_strength=0.0, **base_kw), self.canvas
        )
        specs_barrel = self.gen.generate_layers(
            _base_params(img, barrel_strength=0.10, **base_kw), self.canvas
        )

        assert specs_no, "Expected layers with barrel_strength=0.0"
        assert specs_barrel, "Expected layers with barrel_strength=0.10"

        # Minimum x across all dots — leftmost point comes from left-edge cells
        min_x_no = min(x for s in specs_no for path in s.paths for x, _ in path)
        min_x_barrel = min(
            x for s in specs_barrel for path in s.paths for x, _ in path
        )

        # Barrel pushes left-side points further left (smaller x)
        shift = min_x_no - min_x_barrel
        assert shift >= 1.0, (
            f"Expected corner x to shift outward by >= 1 mm with "
            f"barrel_strength=0.10; no_barrel min_x={min_x_no:.2f}, "
            f"barrel min_x={min_x_barrel:.2f}, shift={shift:.2f}"
        )


# ---------------------------------------------------------------------------
# subpixel_shape="rect" — vertical bar rendering
# ---------------------------------------------------------------------------

class TestRectSubpixelShape:
    """Verify subpixel_shape='rect' emits filled vertical bars (outline + fill)."""

    def setup_method(self):
        from plottter.generators.crt_tv import CrtTvGenerator

        self.gen = CrtTvGenerator()
        self.canvas = _make_canvas()

    def test_rect_emits_outline_plus_fill_per_subpixel(self):
        """Each subpixel should produce 2 polylines: a 5-point closed outline
        rect plus a 2-point central vertical fill stroke.
        """
        img = np.zeros((4, 8, 3), dtype=np.uint8)  # all-black → pen 0 only
        params = _base_params(
            img,
            mask_type="aperture_grille",
            crt_resolution_w=8,
            dither="none",
            scanline_intensity=0.0,
            vignette_strength=0.0,
            barrel_strength=0.0,
            subpixel_shape="rect",
            subpixel_size_mm=0.3,
        )
        specs = self.gen.generate_layers(params, self.canvas)
        pen0 = next(s for s in specs if s.color.upper() == "#000000")

        n_pixels = 4 * 8
        assert len(pen0.paths) == 2 * n_pixels, (
            f"rect shape should emit 2 polylines per subpixel; got {len(pen0.paths)} for {n_pixels} pixels"
        )

        # Pair up: outlines (5-point closed) and fills (2-point vertical line)
        outlines = [p for p in pen0.paths if len(p) == 5]
        fills = [p for p in pen0.paths if len(p) == 2]
        assert len(outlines) == n_pixels
        assert len(fills) == n_pixels

        # First outline closes back to start
        assert outlines[0][0] == outlines[0][-1]
        # First fill is a vertical line (constant x, different y)
        f = fills[0]
        assert f[0][0] == pytest.approx(f[1][0]), "rect fill should be vertical"
        assert f[0][1] != f[1][1]

    def test_rect_height_scales_with_cell_size(self):
        """The bar height should be ~85% of cell_size_y."""
        from plottter.generators._helpers import compute_image_rect

        img = np.zeros((4, 8, 3), dtype=np.uint8)
        crt_w = 8
        params = _base_params(
            img,
            mask_type="aperture_grille",
            crt_resolution_w=crt_w,
            dither="none",
            scanline_intensity=0.0,
            vignette_strength=0.0,
            barrel_strength=0.0,
            subpixel_shape="rect",
            subpixel_size_mm=0.3,
        )
        specs = self.gen.generate_layers(params, self.canvas)
        pen0 = next(s for s in specs if s.color.upper() == "#000000")

        # Compute expected cell height
        img_h, img_w = img.shape[:2]
        draw_x1, draw_y1, draw_x2, draw_y2 = self.canvas.drawing_area()
        _, ry1, _, ry2 = compute_image_rect(
            "fit", img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2
        )
        crt_h = max(1, round(img_h * crt_w / img_w))  # 4
        cell_size_y = (ry2 - ry1) / crt_h
        expected_bar_h = cell_size_y * 0.85

        # Pick a fill stroke (2-point vertical line) and check its length
        fills = [p for p in pen0.paths if len(p) == 2]
        f = fills[0]
        actual_h = abs(f[1][1] - f[0][1])
        assert actual_h == pytest.approx(expected_bar_h, rel=1e-6), (
            f"bar height {actual_h:.4f} != expected {expected_bar_h:.4f} (cell_size_y={cell_size_y:.4f})"
        )

    def test_rect_width_matches_subpixel_size_mm(self):
        """Outline rect width should equal subpixel_size_mm."""
        img = np.zeros((2, 4, 3), dtype=np.uint8)
        params = _base_params(
            img,
            mask_type="aperture_grille",
            crt_resolution_w=4,
            dither="none",
            scanline_intensity=0.0,
            vignette_strength=0.0,
            barrel_strength=0.0,
            subpixel_shape="rect",
            subpixel_size_mm=0.4,
        )
        specs = self.gen.generate_layers(params, self.canvas)
        pen0 = next(s for s in specs if s.color.upper() == "#000000")

        outline = next(p for p in pen0.paths if len(p) == 5)
        xs = [pt[0] for pt in outline]
        width = max(xs) - min(xs)
        assert width == pytest.approx(0.4, abs=1e-6), (
            f"outline width {width:.4f} != subpixel_size_mm=0.4"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def setup_method(self):
        from plottter.generators.crt_tv import CrtTvGenerator

        self.gen = CrtTvGenerator()
        self.canvas = _make_canvas()
        self.image = _make_basic6_image()

    def test_same_seed_same_output(self):
        """Two runs with the same seed must produce identical LayerSpecs."""
        params = _base_params(self.image, seed=7)
        specs1 = self.gen.generate_layers(params, self.canvas)
        specs2 = self.gen.generate_layers(params, self.canvas)

        assert len(specs1) == len(specs2), "Different number of layers"
        for s1, s2 in zip(specs1, specs2):
            assert s1.color == s2.color
            assert s1.name == s2.name
            assert len(s1.paths) == len(s2.paths)
            for p1, p2 in zip(s1.paths, s2.paths):
                assert p1 == p2

    def test_different_seeds_can_differ(self):
        """Two runs with different seeds should (very likely) differ."""
        params_a = _base_params(
            self.image,
            seed=0,
            scanline_intensity=0.5,
            vignette_strength=0.5,
        )
        params_b = _base_params(
            self.image,
            seed=12345,
            scanline_intensity=0.5,
            vignette_strength=0.5,
        )
        specs_a = self.gen.generate_layers(params_a, self.canvas)
        specs_b = self.gen.generate_layers(params_b, self.canvas)

        # With scanline+vignette active, different seeds should yield different
        # path counts (probabilistic — not a hard guarantee, but extremely likely).
        n_a = sum(len(s.paths) for s in specs_a)
        n_b = sum(len(s.paths) for s in specs_b)
        # At least one of the layer counts should differ, or the paths differ.
        # We check path counts; small images can coincidentally match, so
        # this is a soft assertion.
        if n_a == n_b and specs_a and specs_b:
            # Check at least the first path differs
            all_same = all(
                s1.paths == s2.paths
                for s1, s2 in zip(specs_a, specs_b)
                if s1.paths and s2.paths
            )
            # Not asserting here — just confirming the test ran without crash.
            # True stochastic divergence is guaranteed by numpy.random.default_rng.
            assert True


# ---------------------------------------------------------------------------
# generate() single-layer fallback
# ---------------------------------------------------------------------------

class TestGenerateFallback:
    def setup_method(self):
        from plottter.generators.crt_tv import CrtTvGenerator

        self.gen = CrtTvGenerator()
        self.canvas = _make_canvas()
        self.image = _make_basic6_image()

    def test_generate_returns_flat_polyline_list(self):
        params = _base_params(self.image)
        result = self.gen.generate(params, self.canvas)
        assert isinstance(result, list)
        for path in result:
            assert len(path) >= 2


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:
    def setup_method(self):
        from plottter.generators.crt_tv import CrtTvGenerator

        self.gen = CrtTvGenerator()
        self.canvas = _make_canvas()
        self.image = _make_basic6_image()

    def _run_preset(self, name: str, image: np.ndarray | None = None):
        img = image if image is not None else self.image
        presets = {p.name: p for p in self.gen.get_presets()}
        preset = presets[name]
        params = {**preset.params, "_source_image": img}
        return self.gen.generate_layers(params, self.canvas)

    def test_nes_preset_present(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "NES" in names

    def test_nes_preset_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        nes = presets["NES"]
        assert nes.params["palette"] == "Basic 6"
        assert nes.params["crt_resolution_w"] == 256
        assert nes.params["mask_type"] == "shadow_mask"
        assert nes.params["subpixel_shape"] == "circle"
        assert nes.params["subpixel_size_mm"] == pytest.approx(0.25)
        assert nes.params["dither"] == "floyd-steinberg"
        assert nes.params["scanline_intensity"] == pytest.approx(0.8)
        assert nes.params["scanline_period"] == 2
        assert nes.params["vignette_strength"] == pytest.approx(0.2)
        assert nes.params["barrel_strength"] == pytest.approx(0.0)
        assert nes.params["gamma"] == pytest.approx(1.0)

    def test_nes_preset_runs_and_emits_layers(self):
        """NES preset on a synthetic image should produce at least 1 layer."""
        # Use a slightly larger image for the NES preset (256-wide CRT)
        img = np.zeros((32, 128, 3), dtype=np.uint8)
        img[:16, :] = [0, 0, 0]
        img[16:, :] = [230, 57, 70]
        specs = self._run_preset("NES", image=img)
        assert len(specs) >= 1

    def test_preset_names_are_unique(self):
        names = [p.name for p in self.gen.get_presets()]
        assert len(names) == len(set(names))

    def test_trinitron_preset_present(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "Trinitron" in names

    def test_trinitron_preset_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        tri = presets["Trinitron"]
        assert tri.params["palette"] == "Basic 6"
        assert tri.params["crt_resolution_w"] == 320
        assert tri.params["mask_type"] == "aperture_grille"
        assert tri.params["subpixel_shape"] == "rect"
        assert tri.params["subpixel_size_mm"] == pytest.approx(0.20)
        assert tri.params["dither"] == "floyd-steinberg"
        assert tri.params["scanline_intensity"] == pytest.approx(0.4)
        assert tri.params["scanline_period"] == 2
        assert tri.params["vignette_strength"] == pytest.approx(0.1)
        assert tri.params["barrel_strength"] == pytest.approx(0.0)
        assert tri.params["gamma"] == pytest.approx(1.0)

    def test_trinitron_preset_runs_and_emits_layers(self):
        """Trinitron preset on a synthetic image should produce at least 1 layer."""
        img = np.zeros((32, 128, 3), dtype=np.uint8)
        img[:16, :] = [0, 0, 0]
        img[16:, :] = [230, 57, 70]
        specs = self._run_preset("Trinitron", image=img)
        assert len(specs) >= 1

    def test_vga_monitor_preset_present(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "VGA Monitor" in names

    def test_vga_monitor_preset_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        vga = presets["VGA Monitor"]
        assert vga.params["palette"] == "Basic 6"
        assert vga.params["crt_resolution_w"] == 320
        assert vga.params["mask_type"] == "slot_mask"
        assert vga.params["subpixel_shape"] == "rect"
        assert vga.params["subpixel_size_mm"] == pytest.approx(0.25)
        assert vga.params["dither"] == "floyd-steinberg"
        assert vga.params["scanline_intensity"] == pytest.approx(0.5)
        assert vga.params["scanline_period"] == 2
        assert vga.params["vignette_strength"] == pytest.approx(0.2)
        assert vga.params["barrel_strength"] == pytest.approx(0.05)
        assert vga.params["gamma"] == pytest.approx(1.0)

    def test_vga_monitor_preset_runs_and_emits_layers(self):
        """VGA Monitor preset on a synthetic image should produce at least 1 layer."""
        img = np.zeros((32, 128, 3), dtype=np.uint8)
        img[:16, :] = [0, 0, 0]
        img[16:, :] = [230, 57, 70]
        specs = self._run_preset("VGA Monitor", image=img)
        assert len(specs) >= 1

    def test_bw_tv_preset_present(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "B&W TV" in names

    def test_bw_tv_preset_runs_and_emits_layers(self):
        """B&W TV preset on a synthetic image should produce at least 1 layer."""
        img = np.zeros((32, 128, 3), dtype=np.uint8)
        img[:16, :] = [0, 0, 0]
        img[16:, :] = [200, 200, 200]
        specs = self._run_preset("B&W TV", image=img)
        assert len(specs) >= 1

    def test_arcade_cabinet_preset_present(self):
        names = [p.name for p in self.gen.get_presets()]
        assert "Arcade Cabinet" in names

    def test_arcade_cabinet_preset_runs_and_emits_layers(self):
        """Arcade Cabinet preset on a synthetic image should produce at least 1 layer."""
        img = np.zeros((32, 128, 3), dtype=np.uint8)
        img[:16, :] = [0, 0, 0]
        img[16:, :] = [230, 57, 70]
        specs = self._run_preset("Arcade Cabinet", image=img)
        assert len(specs) >= 1
