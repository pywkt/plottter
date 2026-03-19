"""Tests for tam._select_strokes_for_image() — 41.4."""

from __future__ import annotations

import numpy as np
import pytest

from plottter.generators.tam import _build_tone_levels, _select_strokes_for_image


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CANVAS_W = 100.0
CANVAS_H = 100.0
NUM_LEVELS = 6


def make_levels(rng_seed: int = 42) -> list[list[tuple[float, float, float]]]:
    rng = np.random.default_rng(rng_seed)
    return _build_tone_levels(
        canvas_w=CANVAS_W,
        canvas_h=CANVAS_H,
        num_levels=NUM_LEVELS,
        stroke_density=0.1,
        orientation_field=0.0,
        rng=rng,
    )


def solid_image(brightness: float, size: int = 64) -> np.ndarray:
    """Return a constant-brightness float64 image in [0, 1]."""
    return np.full((size, size), brightness, dtype=np.float64)


# ---------------------------------------------------------------------------
# (a) Pure white image → no strokes
# ---------------------------------------------------------------------------


def test_white_image_returns_no_strokes():
    levels = make_levels()
    img = solid_image(1.0)
    result = _select_strokes_for_image(levels, img, CANVAS_W, CANVAS_H)
    assert result == [], f"Expected no strokes for white image, got {len(result)}"


# ---------------------------------------------------------------------------
# (b) Pure black image → all strokes from darkest level
# ---------------------------------------------------------------------------


def test_black_image_returns_all_darkest_strokes():
    levels = make_levels()
    img = solid_image(0.0)
    result = _select_strokes_for_image(levels, img, CANVAS_W, CANVAS_H)
    darkest = levels[-1]
    assert len(result) == len(darkest), (
        f"Expected {len(darkest)} strokes for black image, got {len(result)}"
    )
    # Verify contents match (order may differ)
    assert set(result) == set(darkest)


# ---------------------------------------------------------------------------
# (c) 50% gray returns roughly half the strokes
# ---------------------------------------------------------------------------


def test_mid_gray_returns_roughly_half_strokes():
    levels = make_levels()
    img = solid_image(0.5)
    result = _select_strokes_for_image(levels, img, CANVAS_W, CANVAS_H)
    darkest = levels[-1]
    ratio = len(result) / len(darkest)
    # At 50% gray, linear mapping yields tone_index = 0.5 * (N-1) = 2.5
    # Strokes with birth_level <= 2 should be included (roughly first half)
    assert 0.2 <= ratio <= 0.8, (
        f"Expected roughly half the strokes for 50% gray, got ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# (d) Quadratic curve produces more strokes than linear for mid-gray
# ---------------------------------------------------------------------------


def test_quadratic_more_strokes_than_linear_for_mid_gray():
    levels = make_levels()
    img = solid_image(0.5)
    linear_result = _select_strokes_for_image(
        levels, img, CANVAS_W, CANVAS_H, density_curve="linear"
    )
    quadratic_result = _select_strokes_for_image(
        levels, img, CANVAS_W, CANVAS_H, density_curve="quadratic"
    )
    # Quadratic uses sqrt(darkness) which is concave (> linear for mid-gray),
    # so it produces more strokes than linear at the same mid-gray brightness.
    assert len(quadratic_result) >= len(linear_result), (
        f"Quadratic should produce >= strokes than linear at 50% gray "
        f"(quadratic={len(quadratic_result)}, linear={len(linear_result)})"
    )


# ---------------------------------------------------------------------------
# Extra: logarithmic curve returns valid results
# ---------------------------------------------------------------------------


def test_logarithmic_curve_valid():
    levels = make_levels()
    for brightness in [0.0, 0.25, 0.5, 0.75, 1.0]:
        img = solid_image(brightness)
        result = _select_strokes_for_image(
            levels, img, CANVAS_W, CANVAS_H, density_curve="logarithmic"
        )
        darkest = levels[-1]
        assert 0 <= len(result) <= len(darkest), (
            f"brightness={brightness}: stroke count {len(result)} out of range "
            f"[0, {len(darkest)}]"
        )


# ---------------------------------------------------------------------------
# Extra: monotonicity — darker images yield >= strokes than lighter ones
# ---------------------------------------------------------------------------


def test_monotonicity_linear():
    levels = make_levels()
    prev = 0
    for brightness in [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]:
        img = solid_image(brightness)
        count = len(
            _select_strokes_for_image(levels, img, CANVAS_W, CANVAS_H, density_curve="linear")
        )
        assert count >= prev, (
            f"Stroke count should be non-decreasing as brightness decreases; "
            f"brightness={brightness} gave {count}, previous was {prev}"
        )
        prev = count


# ---------------------------------------------------------------------------
# Extra: uint8 RGB image input is accepted
# ---------------------------------------------------------------------------


def test_uint8_rgb_image():
    levels = make_levels()
    # Black RGB image
    img_black = np.zeros((64, 64, 3), dtype=np.uint8)
    result = _select_strokes_for_image(levels, img_black, CANVAS_W, CANVAS_H)
    assert len(result) == len(levels[-1])

    # White RGB image
    img_white = np.full((64, 64, 3), 255, dtype=np.uint8)
    result = _select_strokes_for_image(levels, img_white, CANVAS_W, CANVAS_H)
    assert result == []


# ---------------------------------------------------------------------------
# Edge case: single tone level
# ---------------------------------------------------------------------------


def test_single_level():
    rng = np.random.default_rng(0)
    levels = _build_tone_levels(
        canvas_w=CANVAS_W,
        canvas_h=CANVAS_H,
        num_levels=1,
        stroke_density=0.05,
        orientation_field=0.0,
        rng=rng,
    )
    assert len(levels) == 1
    # Black image → all strokes
    result = _select_strokes_for_image(levels, solid_image(0.0), CANVAS_W, CANVAS_H)
    assert len(result) == len(levels[0])
    # White image → no strokes
    result = _select_strokes_for_image(levels, solid_image(1.0), CANVAS_W, CANVAS_H)
    assert result == []
