"""Tests for tam._build_tone_levels() — 41.2."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators.tam import _build_tone_levels, _sample_orientation


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CANVAS_W = 190.0  # A4 drawing area width (mm)
CANVAS_H = 277.0  # A4 drawing area height (mm)
NUM_LEVELS = 6


def default_rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# (a) Level 0 has the fewest strokes, level N-1 the most
# ---------------------------------------------------------------------------


def test_level_0_fewest_level_last_most():
    levels = _build_tone_levels(
        CANVAS_W, CANVAS_H, NUM_LEVELS, stroke_density=0.05,
        orientation_field=0.0, rng=default_rng()
    )
    assert len(levels) == NUM_LEVELS
    assert len(levels[0]) < len(levels[-1])


# ---------------------------------------------------------------------------
# (b) Nesting property: every stroke in level K is also in level K+1
# ---------------------------------------------------------------------------


def test_nesting_property():
    levels = _build_tone_levels(
        CANVAS_W, CANVAS_H, NUM_LEVELS, stroke_density=0.05,
        orientation_field=0.0, rng=default_rng()
    )
    for k in range(NUM_LEVELS - 1):
        set_k = set(levels[k])
        set_k1 = set(levels[k + 1])
        assert set_k.issubset(set_k1), (
            f"Level {k} is not a subset of level {k + 1}: "
            f"{len(set_k)} vs {len(set_k1)} strokes"
        )


# ---------------------------------------------------------------------------
# (c) Stroke count increases monotonically across levels
# ---------------------------------------------------------------------------


def test_monotonic_stroke_count():
    levels = _build_tone_levels(
        CANVAS_W, CANVAS_H, NUM_LEVELS, stroke_density=0.05,
        orientation_field=0.0, rng=default_rng()
    )
    counts = [len(lvl) for lvl in levels]
    for i in range(len(counts) - 1):
        assert counts[i] < counts[i + 1], (
            f"Counts not strictly increasing: {counts}"
        )


# ---------------------------------------------------------------------------
# (d) Fixed-angle mode produces uniform angles
# ---------------------------------------------------------------------------


def test_fixed_angle_uniform():
    fixed_angle = math.pi / 4  # 45°
    levels = _build_tone_levels(
        CANVAS_W, CANVAS_H, NUM_LEVELS, stroke_density=0.05,
        orientation_field=fixed_angle, rng=default_rng()
    )
    for level in levels:
        for x, y, angle in level:
            assert angle == pytest.approx(fixed_angle), (
                f"Expected fixed angle {fixed_angle}, got {angle}"
            )


# ---------------------------------------------------------------------------
# (e) Returned positions are within canvas bounds
# ---------------------------------------------------------------------------


def test_positions_within_bounds():
    levels = _build_tone_levels(
        CANVAS_W, CANVAS_H, NUM_LEVELS, stroke_density=0.05,
        orientation_field=0.0, rng=default_rng()
    )
    for k, level in enumerate(levels):
        for x, y, angle in level:
            assert 0.0 <= x <= CANVAS_W, f"Level {k}: x={x} out of bounds"
            assert 0.0 <= y <= CANVAS_H, f"Level {k}: y={y} out of bounds"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_level():
    levels = _build_tone_levels(
        50.0, 50.0, 1, stroke_density=0.1,
        orientation_field=0.0, rng=default_rng()
    )
    assert len(levels) == 1
    assert len(levels[0]) >= 1


def test_two_levels_nesting():
    levels = _build_tone_levels(
        100.0, 100.0, 2, stroke_density=0.1,
        orientation_field=0.0, rng=default_rng()
    )
    assert len(levels) == 2
    assert len(levels[0]) < len(levels[1])
    set0 = set(levels[0])
    set1 = set(levels[1])
    assert set0.issubset(set1)


def test_2d_orientation_field():
    """Test that a 2-D orientation field is sampled correctly."""
    # Create a gradient angle field: angles vary linearly from 0 to π across width
    H, W = 50, 50
    field = np.zeros((H, W), dtype=np.float64)
    for col in range(W):
        field[:, col] = math.pi * col / (W - 1)

    levels = _build_tone_levels(
        100.0, 100.0, 3, stroke_density=0.05,
        orientation_field=field, rng=default_rng()
    )
    # All angles must be in [0, π]
    for level in levels:
        for x, y, angle in level:
            assert -1e-9 <= angle <= math.pi + 1e-9, (
                f"Angle {angle} outside [0, π]"
            )


def test_low_density_still_produces_strokes():
    """Very low density should still produce at least num_levels strokes."""
    num_levels = 4
    levels = _build_tone_levels(
        10.0, 10.0, num_levels, stroke_density=0.001,
        orientation_field=0.0, rng=default_rng()
    )
    assert len(levels) == num_levels
    for lvl in levels:
        assert len(lvl) >= 1


def test_reproducibility():
    """Same seed → identical output."""
    kwargs = dict(
        canvas_w=100.0, canvas_h=100.0, num_levels=3,
        stroke_density=0.05, orientation_field=1.0,
    )
    levels_a = _build_tone_levels(**kwargs, rng=np.random.default_rng(7))
    levels_b = _build_tone_levels(**kwargs, rng=np.random.default_rng(7))
    assert levels_a == levels_b


# ---------------------------------------------------------------------------
# _sample_orientation unit tests
# ---------------------------------------------------------------------------


def test_sample_orientation_scalar():
    angle = _sample_orientation(5.0, 5.0, 100.0, 100.0, math.pi / 3)
    assert angle == pytest.approx(math.pi / 3)


def test_sample_orientation_array_corners():
    """Corners of a constant-0 array should return 0."""
    field = np.zeros((10, 10), dtype=np.float64)
    for x, y in [(0.0, 0.0), (100.0, 100.0), (0.0, 100.0), (100.0, 0.0)]:
        result = _sample_orientation(x, y, 100.0, 100.0, field)
        assert result == pytest.approx(0.0)


def test_sample_orientation_array_interpolated():
    """A field with a known gradient should interpolate correctly."""
    # field[row, col] = col  → angle at (x=50, y=0) ≈ middle column
    field = np.tile(np.arange(10, dtype=np.float64), (10, 1))  # shape (10,10)
    # At x=50mm (half of 100mm) → col index 4.5 → angle ≈ 4.5
    result = _sample_orientation(50.0, 50.0, 100.0, 100.0, field)
    assert result == pytest.approx(4.5, abs=0.1)
