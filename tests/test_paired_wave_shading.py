"""Tests for the Paired Wave Shading generator.

Covers the algorithmic contract — every scan line emits a top/bottom pair,
deviation tracks local darkness, white regions break the pair into
sub-segments — plus the registration plumbing so the generator shows up in
Color Separation's per-channel dropdown.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.generators import GENERATORS
from plottter.generators.paired_wave_shading import (
    PairedWaveShadingGenerator,
    _box_smooth,
    _emit_pairs_for_row,
    _sample_brightness_row,
)
from plottter.models import Canvas


# ---------------------------------------------------------------------------
# Registration / shape of the generator
# ---------------------------------------------------------------------------


def test_generator_is_registered_under_canonical_name():
    """Color Separation iterates GENERATORS to populate its per-channel
    dropdown, so registration is a hard requirement."""
    assert "Paired Wave Shading" in GENERATORS
    cls = GENERATORS["Paired Wave Shading"]
    assert cls is PairedWaveShadingGenerator


def test_generator_is_image_category():
    """Must be 'image' so the Color-Separation / Image-to-Lines mode pick it up."""
    assert PairedWaveShadingGenerator().category == "image"


def test_generator_exposes_core_params():
    gen = PairedWaveShadingGenerator()
    names = {p.name for p in gen.get_parameters()}
    for required in (
        "line_spacing_mm", "max_deviation_mm", "min_deviation_mm",
        "sample_interval_mm", "tone_gamma", "smoothing_mm",
        "skip_white_above", "invert", "brightness", "contrast", "blur_radius",
    ):
        assert required in names, f"missing parameter: {required}"


def test_presets_contain_color_split_starting_point():
    """The first shipped preset is the colour-separation starting point —
    intentionally named neutrally so it covers both CMYK and RGB splits."""
    presets = PairedWaveShadingGenerator().get_presets()
    names = [p.name for p in presets]
    assert any("Color Split" in n for n in names), names


# ---------------------------------------------------------------------------
# Helper-level unit tests
# ---------------------------------------------------------------------------


def test_box_smooth_passthrough_for_window_le_one():
    arr = np.array([1.0, 5.0, 9.0], dtype=np.float32)
    assert np.array_equal(_box_smooth(arr, 0), arr)
    assert np.array_equal(_box_smooth(arr, 1), arr)


def test_box_smooth_three_point_average():
    arr = np.array([0.0, 0.0, 6.0, 0.0, 0.0], dtype=np.float32)
    out = _box_smooth(arr, 3)
    # Centre sample averaged with its neighbours → 6/3 == 2; neighbours pick up
    # one-third of the centre's energy.
    assert out[2] == pytest.approx(2.0)
    assert out[1] == pytest.approx(2.0)
    assert out[3] == pytest.approx(2.0)


def test_emit_pairs_for_row_basic_pairing():
    """Solid keep-mask + non-zero deviation → exactly two polylines that mirror
    each other vertically around the baseline."""
    baseline = 50.0
    xs = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    dev = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    keep = np.array([True, True, True])

    pairs = _emit_pairs_for_row(baseline, xs, dev, keep)

    assert len(pairs) == 2
    top, bot = pairs
    for t, b in zip(top, bot):
        assert t[0] == b[0]                  # same x
        assert (b[1] - t[1]) == pytest.approx(1.0)  # gap == deviation
        assert (t[1] + b[1]) / 2 == pytest.approx(baseline)


def test_emit_pairs_drops_white_gap_into_sub_segments():
    """A False sample in the middle of a row splits BOTH polylines, producing
    two top-bottom pairs (4 polylines total)."""
    baseline = 10.0
    xs = np.linspace(0.0, 4.0, 5, dtype=np.float32)
    dev = np.full(5, 1.0, dtype=np.float32)
    keep = np.array([True, True, False, True, True])

    pairs = _emit_pairs_for_row(baseline, xs, dev, keep)

    assert len(pairs) == 4
    # First pair (top + bottom) covers samples [0, 1]; second covers [3, 4]
    assert len(pairs[0]) == 2 and len(pairs[1]) == 2
    assert len(pairs[2]) == 2 and len(pairs[3]) == 2


def test_emit_pairs_all_white_returns_nothing():
    assert _emit_pairs_for_row(
        0.0,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.array([False, False, False]),
    ) == []


def test_sample_brightness_row_bilinear():
    """Sampling on a known constant-brightness image returns that brightness."""
    gray = np.full((20, 20), 128, dtype=np.uint8)
    xs = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    out = _sample_brightness_row(gray, 10.0, xs, (0.0, 0.0, 20.0, 20.0))
    assert (out == 128).all()


def test_sample_brightness_row_out_of_bounds_returns_white():
    """Anything outside the image rectangle defaults to 255 (no ink)."""
    gray = np.zeros((10, 10), dtype=np.uint8)
    xs = np.array([-5.0, 5.0, 15.0], dtype=np.float32)
    out = _sample_brightness_row(gray, 5.0, xs, (0.0, 0.0, 10.0, 10.0))
    assert out[0] == 255
    assert out[2] == 255
    assert out[1] == 0


# ---------------------------------------------------------------------------
# End-to-end algorithmic behaviour
# ---------------------------------------------------------------------------


def _make_canvas() -> Canvas:
    return Canvas(width_mm=100, height_mm=100, margin_mm=0)


def _gradient_image(black_left: bool = True) -> np.ndarray:
    """100×100 horizontal gradient, dark on one side, white on the other."""
    row = np.linspace(0, 255, 100, dtype=np.uint8)
    if not black_left:
        row = row[::-1]
    return np.tile(row, (100, 1))


def _run(params: dict | None = None) -> list:
    gen = PairedWaveShadingGenerator()
    p = {
        "line_spacing_mm": 2.0,
        "max_deviation_mm": 1.0,
        "min_deviation_mm": 0.0,
        "sample_interval_mm": 0.5,
        "tone_gamma": 1.0,
        "smoothing_mm": 0.0,
        "skip_white_above": 255,  # don't drop any samples — pure-pair contract
        "invert": False,
        "image_fit_mode": "fit",
        "_source_image": _gradient_image(),
    }
    if params:
        p.update(params)
    return gen.generate(p, _make_canvas())


def test_generate_returns_polylines_in_top_bottom_pairs():
    """With ``skip_white_above=255`` every scan line emits one (top, bottom)
    pair — total polylines must be even and ≥ 2."""
    paths = _run()
    assert len(paths) >= 2
    assert len(paths) % 2 == 0


def test_generate_deviation_widens_in_dark_regions():
    """First polyline pair: gap on the BLACK side must exceed the gap on the
    WHITE side."""
    paths = _run({"max_deviation_mm": 2.0})
    top, bot = paths[0], paths[1]
    left_gap = abs(top[0][1] - bot[0][1])
    right_gap = abs(top[-1][1] - bot[-1][1])
    assert left_gap > right_gap
    # Reasonable bounds: black should be near max, white near min
    assert left_gap > 1.5
    assert right_gap < 0.1


def test_generate_pair_centred_on_baseline():
    """For every paired (top_i, bot_i) sample, the midpoint must equal the
    scan-line baseline — the 'balanced ±d/2' invariant that makes neither
    line on its own carry shading information."""
    paths = _run({"max_deviation_mm": 1.0})
    top, bot = paths[0], paths[1]
    midpoints = [(t[1] + b[1]) / 2 for t, b in zip(top, bot)]
    # All midpoints equal (within float epsilon)
    assert max(midpoints) - min(midpoints) < 1e-6


def test_min_deviation_keeps_pair_open_in_white():
    """With ``min_deviation_mm`` > 0 the pair never collapses, even on pure
    white pixels."""
    paths = _run({
        "max_deviation_mm": 1.0,
        "min_deviation_mm": 0.3,
        "skip_white_above": 255,
    })
    top, bot = paths[0], paths[1]
    right_gap = abs(top[-1][1] - bot[-1][1])
    assert right_gap == pytest.approx(0.3, abs=0.05)


def test_skip_white_above_breaks_pair_into_subsegments():
    """A white half of the image with ``skip_white_above`` set low enough
    must split each scan line into a single (top, bottom) pair covering only
    the dark half."""
    paths = _run({"skip_white_above": 50, "max_deviation_mm": 0.5})
    # Still pairs
    assert len(paths) % 2 == 0
    # Every polyline's x range must lie in the dark (left) half
    for path in paths:
        max_x = max(pt[0] for pt in path)
        assert max_x < 60  # canvas is 100mm wide, white starts ~ x=50


def test_empty_source_returns_empty():
    gen = PairedWaveShadingGenerator()
    assert gen.generate({}, _make_canvas()) == []


def test_zero_max_deviation_keeps_pairs_overlapping():
    """A pathological config with both max and min deviation at 0 must not
    crash — the two lines just sit on top of each other."""
    paths = _run({"max_deviation_mm": 0.0, "min_deviation_mm": 0.0})
    top, bot = paths[0], paths[1]
    for t, b in zip(top, bot):
        assert t[1] == pytest.approx(b[1])


def test_invert_swaps_dark_and_light():
    """A CMYK-style ink-coverage source (0 = no ink, 255 = full ink) becomes
    correct when invert=True.  The dark gap should land on the side with
    HIGH pixel values, not low."""
    img = _gradient_image()  # black left, white right
    gen = PairedWaveShadingGenerator()
    p = {
        "line_spacing_mm": 5.0,
        "max_deviation_mm": 1.0,
        "sample_interval_mm": 1.0,
        "smoothing_mm": 0.0,
        "skip_white_above": 255,
        "invert": True,
        "image_fit_mode": "fit",
        "_source_image": img,
    }
    paths = gen.generate(p, _make_canvas())
    top, bot = paths[0], paths[1]
    # With invert, brightness flips: LEFT (was 0) → 255 → light → narrow gap.
    left_gap = abs(top[0][1] - bot[0][1])
    right_gap = abs(top[-1][1] - bot[-1][1])
    assert right_gap > left_gap
