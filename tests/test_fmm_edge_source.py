"""Edge-seeded FMM source for the Contour generator (Edge Hug effect).

The FMM 'Contours' render mode can seed the marching front from a whole image
edge (not just a point), so the isolines enter parallel to that edge and bunch /
'hug' dark regions as they advance. These tests cover the seed geometry, the
travel-time field, and end-to-end generation + the 'Edge Hug' preset.

skfmm (the [fmm] extra) may be absent; the generator then uses the scipy-EDT
fallback. The assertions hold for both paths.
"""

from __future__ import annotations

import numpy as np

from plottter.generators.contour._fmm import (
    _FMM_EDGE_SOURCES,
    _compute_fmm_field,
    _fmm_seed_indices,
)
from plottter.models.canvas import Canvas


def _gradient(h: int = 60, w: int = 60) -> np.ndarray:
    """Left-to-right brightness gradient (dark left, light right), grayscale."""
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))


def _dark_center(h: int = 60, w: int = 60) -> np.ndarray:
    img = np.full((h, w), 255, dtype=np.uint8)
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0
    return img


# ---------------------------------------------------------------------------
# Seed geometry
# ---------------------------------------------------------------------------

def test_seed_indices_cover_each_edge():
    h, w = 4, 5
    rows, cols = _fmm_seed_indices("Top Edge", h, w)
    assert np.all(rows == 0) and sorted(cols) == list(range(w))

    rows, cols = _fmm_seed_indices("Bottom Edge", h, w)
    assert np.all(rows == h - 1) and sorted(cols) == list(range(w))

    rows, cols = _fmm_seed_indices("Left Edge", h, w)
    assert np.all(cols == 0) and sorted(rows) == list(range(h))

    rows, cols = _fmm_seed_indices("Right Edge", h, w)
    assert np.all(cols == w - 1) and sorted(rows) == list(range(h))


def test_seed_indices_point_sources_are_single_pixels():
    rows, cols = _fmm_seed_indices("Center", 10, 8)
    assert (rows.tolist(), cols.tolist()) == ([5], [4])

    rows, cols = _fmm_seed_indices("Custom", 10, 10, source_x_pct=0.0, source_y_pct=100.0)
    assert (rows.tolist(), cols.tolist()) == ([9], [0])


# ---------------------------------------------------------------------------
# Travel-time field
# ---------------------------------------------------------------------------

def test_top_edge_field_grows_away_from_seeded_edge():
    gray = _gradient(50, 50)
    T, sy, sx = _compute_fmm_field(gray, "Top Edge", gamma=1.0, speed_floor=0.01)
    # Travel time is lowest along the seeded top edge and grows inward; assert
    # the trend (holds for both true skfmm and the scipy-EDT fallback) rather
    # than an exact minimum, whose pixel placement differs between backends.
    assert T[:3].mean() < T[-3:].mean()
    assert T[0, :].mean() <= T[10, :].mean()
    # Representative source point sits on the top edge.
    assert sy == 0


def test_edge_source_differs_from_center_source():
    gray = _dark_center(50, 50)
    T_edge, _, _ = _compute_fmm_field(gray, "Top Edge", gamma=1.0, speed_floor=0.01)
    T_center, _, _ = _compute_fmm_field(gray, "Center", gamma=1.0, speed_floor=0.01)
    # Different seed geometry => materially different travel-time fields.
    assert not np.allclose(T_edge, T_center)


def test_all_four_edges_supported():
    gray = _gradient(40, 40)
    for src in _FMM_EDGE_SOURCES:
        T, _, _ = _compute_fmm_field(gray, src, gamma=1.5, speed_floor=0.01)
        assert np.isfinite(T).all()
        assert T.shape == gray.shape


# ---------------------------------------------------------------------------
# End-to-end generation + preset
# ---------------------------------------------------------------------------

def _fmm_params(source: str) -> dict:
    return {
        "mode": "FMM Topographic",
        "fmm_render_mode": "Contours",
        "fmm_source_point": source,
        "fmm_num_contours": 20,
        "fmm_gamma": 2.0,
        "fmm_speed_floor": 0.01,
        "fmm_contour_spacing": "Linear",
        "fmm_min_contour_length_mm": 1.0,
        "smooth_iterations": 0,
        "invert": False,
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_radius": 0.0,
        "simplify_mm": 0.0,
        "min_contour_px": 3,
    }


def test_generate_top_edge_contours_nonempty():
    from plottter.generators.contour import ContourGenerator

    gen = ContourGenerator()
    canvas = Canvas.from_preset("A4", margin=10.0)
    params = {"_source_image": _gradient(80, 80), **_fmm_params("Top Edge")}
    result = gen.generate(params, canvas)
    assert isinstance(result, list)
    assert len(result) > 0
    for poly in result:
        assert len(poly) >= 2


def test_edge_hug_preset_present_and_functional():
    from plottter.generators.contour import ContourGenerator

    gen = ContourGenerator()
    preset = next(
        (p for p in gen.get_presets() if p.name.startswith("Edge Hug")), None
    )
    assert preset is not None, "Edge Hug preset should be registered"
    assert preset.params["fmm_source_point"] == "Top Edge"
    assert preset.params["mode"] == "FMM Topographic"
    assert preset.params["fmm_render_mode"] == "Contours"

    canvas = Canvas.from_preset("A4", margin=10.0)
    params = {"_source_image": _dark_center(80, 80), **preset.params}
    result = gen.generate(params, canvas)
    assert isinstance(result, list)
    assert len(result) > 0
