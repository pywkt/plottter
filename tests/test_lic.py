"""Tests for the LIC generator seed utilities — 42.1.

Covers:
  (a) Grid produces expected number of seeds for given canvas size and spacing
  (b) Jitter keeps points within one cell of their grid position
  (c) Brightness filter with threshold=0 removes all seeds from a white image
  (d) Brightness filter with threshold=1 (normalised) removes no seeds
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plottter.generators.lic import _brightness_filter, _seed_grid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CANVAS_W = 100.0
CANVAS_H = 80.0
SEED_SPACING = 10.0  # mm  → 10×8 = 80 cells expected

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# (a) Expected seed count
# ---------------------------------------------------------------------------


def test_seed_count_matches_grid():
    """Number of seeds equals number of grid cells (ceil(W/s) × ceil(H/s))."""
    rng = np.random.default_rng(0)
    seeds = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, rng)

    nx = math.ceil(CANVAS_W / SEED_SPACING)
    ny = math.ceil(CANVAS_H / SEED_SPACING)
    assert len(seeds) == nx * ny


def test_seed_count_non_divisible_spacing():
    """Verify ceil behaviour when canvas is not an integer multiple of spacing."""
    rng = np.random.default_rng(1)
    spacing = 7.0
    seeds = _seed_grid(CANVAS_W, CANVAS_H, spacing, rng)

    nx = math.ceil(CANVAS_W / spacing)
    ny = math.ceil(CANVAS_H / spacing)
    assert len(seeds) == nx * ny


def test_all_seeds_within_canvas():
    """All returned seeds must lie inside the canvas bounds."""
    rng = np.random.default_rng(2)
    seeds = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, rng)

    for x, y in seeds:
        assert 0.0 <= x <= CANVAS_W, f"x={x} out of bounds"
        assert 0.0 <= y <= CANVAS_H, f"y={y} out of bounds"


# ---------------------------------------------------------------------------
# (b) Jitter keeps seeds within one cell of grid centre
# ---------------------------------------------------------------------------


def test_jitter_within_one_cell():
    """Each seed must be within one full cell side of its nominal centre."""
    rng = np.random.default_rng(3)
    seeds = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, rng)

    nx = math.ceil(CANVAS_W / SEED_SPACING)
    ny = math.ceil(CANVAS_H / SEED_SPACING)
    cell_w = CANVAS_W / nx
    cell_h = CANVAS_H / ny

    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            cx = (ix + 0.5) * cell_w
            cy = (iy + 0.5) * cell_h
            x, y = seeds[idx]
            # Max jitter radius = spacing * 0.3; clamp may push it slightly
            # but never beyond one full cell side from the centre.
            assert abs(x - cx) <= cell_w, f"seed {idx} x deviation too large"
            assert abs(y - cy) <= cell_h, f"seed {idx} y deviation too large"
            idx += 1


def test_jitter_not_all_at_centre():
    """With enough seeds, jitter should produce variation (not all at centres)."""
    rng = np.random.default_rng(4)
    seeds = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, rng)

    nx = math.ceil(CANVAS_W / SEED_SPACING)
    ny = math.ceil(CANVAS_H / SEED_SPACING)
    cell_w = CANVAS_W / nx
    cell_h = CANVAS_H / ny

    deviations = []
    for idx, (iy, ix) in enumerate(
        (iy, ix) for iy in range(ny) for ix in range(nx)
    ):
        cx = (ix + 0.5) * cell_w
        cy = (iy + 0.5) * cell_h
        x, y = seeds[idx]
        deviations.append(math.hypot(x - cx, y - cy))

    assert max(deviations) > 0.0, "All seeds at centre — jitter has no effect"


# ---------------------------------------------------------------------------
# (c) Brightness filter threshold=0 removes all seeds from white image
# ---------------------------------------------------------------------------


def test_brightness_filter_threshold_zero_removes_all():
    """threshold=0 → all seeds removed because brightness ≥ 0 for every pixel."""
    rng = np.random.default_rng(5)
    seeds = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, rng)

    # Pure white image (normalised to [0, 1])
    white_img = np.ones((64, 64), dtype=np.float32)

    kept = _brightness_filter(seeds, CANVAS_W, CANVAS_H, white_img, brightness_threshold=0.0)
    assert kept == [], f"Expected no seeds, got {len(kept)}"


def test_brightness_filter_threshold_zero_black_image_removes_all():
    """threshold=0 also removes seeds on a black image (brightness 0 ≥ 0)."""
    rng = np.random.default_rng(6)
    seeds = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, rng)

    black_img = np.zeros((64, 64), dtype=np.float32)

    kept = _brightness_filter(seeds, CANVAS_W, CANVAS_H, black_img, brightness_threshold=0.0)
    assert kept == []


# ---------------------------------------------------------------------------
# (d) Brightness filter threshold=255 (normalised >1) removes no seeds
# ---------------------------------------------------------------------------


def test_brightness_filter_high_threshold_keeps_all():
    """threshold > max image brightness → all seeds kept."""
    rng = np.random.default_rng(7)
    seeds = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, rng)

    # White image — max brightness = 1.0 (normalised)
    white_img = np.ones((64, 64), dtype=np.float32)

    # threshold=255/255 = 1.0 would remove white pixels (brightness ≥ 1.0),
    # so use slightly above 1 to guarantee no removal.
    kept = _brightness_filter(seeds, CANVAS_W, CANVAS_H, white_img, brightness_threshold=2.0)
    assert len(kept) == len(seeds), (
        f"Expected all {len(seeds)} seeds kept, got {len(kept)}"
    )


def test_brightness_filter_threshold_one_on_black_keeps_all():
    """threshold=1 on a pure-black image keeps every seed."""
    rng = np.random.default_rng(8)
    seeds = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, rng)

    black_img = np.zeros((64, 64), dtype=np.float32)

    kept = _brightness_filter(seeds, CANVAS_W, CANVAS_H, black_img, brightness_threshold=1.0)
    assert len(kept) == len(seeds)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_seed_list_passthrough():
    """_brightness_filter returns empty list when given empty input."""
    white_img = np.ones((32, 32), dtype=np.float32)
    kept = _brightness_filter([], CANVAS_W, CANVAS_H, white_img, brightness_threshold=0.5)
    assert kept == []


def test_partial_filter():
    """A half-white / half-black image with threshold=0.5 filters approximately half."""
    rng = np.random.default_rng(9)
    seeds = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, rng)

    # Left half black (0.0), right half white (1.0)
    img = np.zeros((64, 64), dtype=np.float32)
    img[:, 32:] = 1.0  # right half white

    kept = _brightness_filter(seeds, CANVAS_W, CANVAS_H, img, brightness_threshold=0.5)

    # Roughly half should survive (those on the black left side)
    assert 0 < len(kept) < len(seeds), (
        f"Expected partial filtering, got {len(kept)}/{len(seeds)}"
    )


def test_single_pixel_image():
    """Filter works correctly with a 1×1 image."""
    rng = np.random.default_rng(10)
    seeds = _seed_grid(20.0, 20.0, 5.0, rng)

    black_1x1 = np.zeros((1, 1), dtype=np.float32)
    kept = _brightness_filter(seeds, 20.0, 20.0, black_1x1, brightness_threshold=1.0)
    assert len(kept) == len(seeds)


def test_reproducibility():
    """Same seed produces identical output."""
    seeds_a = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, np.random.default_rng(99))
    seeds_b = _seed_grid(CANVAS_W, CANVAS_H, SEED_SPACING, np.random.default_rng(99))
    assert seeds_a == seeds_b
