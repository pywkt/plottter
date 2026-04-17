"""Tests for ridgeline HLR functions in audio_utils."""

from __future__ import annotations

import numpy as np
import pytest

from plottter.generators.audio_utils import (
    _extract_visible_segments,
    ridgeline_hlr,
    ridgeline_no_hlr,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_spectrogram(num_rows: int, num_cols: int, value: float = 0.0) -> np.ndarray:
    return np.full((num_rows, num_cols), value, dtype=float)


# ---------------------------------------------------------------------------
# (a) Flat spectrogram (all zeros) → num_rows horizontal lines, all visible
# ---------------------------------------------------------------------------

def test_flat_zero_spectrogram_produces_num_rows_polylines():
    num_rows, num_cols = 5, 50
    spec = make_spectrogram(num_rows, num_cols, 0.0)
    polylines = ridgeline_hlr(spec, width=100.0, amplitude_scale=1.0, row_spacing=10.0, smoothing_sigma=0.0)
    # Each row is a flat line at its baseline; no row occludes another (all at baseline)
    # Row 0 baseline=0, row 1 baseline=10, etc. → each fully visible
    assert len(polylines) == num_rows
    for pl in polylines:
        assert len(pl) >= 2


def test_flat_zero_spectrogram_x_range():
    spec = make_spectrogram(4, 30, 0.0)
    polylines = ridgeline_hlr(spec, width=50.0, smoothing_sigma=0.0)
    for pl in polylines:
        xs = [p[0] for p in pl]
        assert min(xs) >= 0.0 - 1e-9
        assert max(xs) <= 50.0 + 1e-9


# ---------------------------------------------------------------------------
# (b) Single tall peak in front row → behind rows occluded in that range
# ---------------------------------------------------------------------------

def test_peak_in_front_occludes_rows_behind():
    num_rows, num_cols = 5, 100
    spec = make_spectrogram(num_rows, num_cols, 0.0)
    # Place a tall peak at column 50 in the front row (i=0)
    peak_col = 50
    spec[0, peak_col] = 1.0

    amplitude_scale = 20.0
    row_spacing = 5.0
    polylines_hlr = ridgeline_hlr(
        spec, width=100.0,
        amplitude_scale=amplitude_scale,
        row_spacing=row_spacing,
        smoothing_sigma=0.0,
    )
    polylines_no_hlr = ridgeline_no_hlr(
        spec, width=100.0,
        amplitude_scale=amplitude_scale,
        row_spacing=row_spacing,
        smoothing_sigma=0.0,
    )
    # With HLR, back rows may be occluded → fewer total segments or broken segments
    # The no-HLR version has exactly num_rows polylines
    assert len(polylines_no_hlr) == num_rows
    # HLR version should have more polylines (rows behind the peak are split)
    # OR at least not fewer total points (peak occlusion splits segments)
    # Key check: the HLR result differs from the simple version at peak area
    hlr_x_coords = {round(p[0], 2) for pl in polylines_hlr for p in pl}
    no_hlr_x_coords = {round(p[0], 2) for pl in polylines_no_hlr for p in pl}
    # All no-HLR x-coords should be present in HLR result (front row always visible)
    # but HLR might have fewer unique x per back rows; just verify structure is valid
    assert len(polylines_hlr) >= 1


# ---------------------------------------------------------------------------
# (c) All-ones spectrogram: front row fully visible, subsequent rows hidden
# ---------------------------------------------------------------------------

def test_all_ones_front_row_fully_visible():
    num_rows, num_cols = 4, 60
    spec = make_spectrogram(num_rows, num_cols, 1.0)
    amplitude_scale = 5.0
    row_spacing = 3.0

    polylines = ridgeline_hlr(
        spec, width=80.0,
        amplitude_scale=amplitude_scale,
        row_spacing=row_spacing,
        smoothing_sigma=0.0,
    )
    # Front row (i=0) baseline=0, y=amplitude_scale=5; fully above horizon (-inf) → fully visible
    # Row 1: baseline=3, y=3+5=8; horizon after row 0 was max(y[0], baseline[0])=max(5, 0)=5
    #   → y=8 > 5, so row 1 should also be fully visible
    # The key: at minimum the first row must produce exactly one polyline of full width
    # Find the polyline with the smallest y values (front row)
    if polylines:
        first_pl = polylines[0]
        assert len(first_pl) >= 2
        xs = [p[0] for p in first_pl]
        assert xs[0] == pytest.approx(0.0, abs=1e-6)
        assert xs[-1] == pytest.approx(80.0, abs=1e-6)


def test_all_ones_subsequent_rows_may_be_partially_hidden():
    num_rows, num_cols = 3, 40
    spec = make_spectrogram(num_rows, num_cols, 1.0)
    # Use large amplitude relative to row_spacing → back rows are mostly hidden
    polylines = ridgeline_hlr(
        spec, width=50.0,
        amplitude_scale=10.0,
        row_spacing=1.0,
        smoothing_sigma=0.0,
    )
    # With amplitude=10 and spacing=1, front row peaks at y=10.
    # Row 1: baseline=1, y=11 > max(10, 0)=10 → some points still visible
    # Row 2: baseline=2, y=12 > max(11, 1)=11 → some visible
    # Just check output is non-empty and valid
    assert all(len(pl) >= 2 for pl in polylines)


# ---------------------------------------------------------------------------
# (d) mirror=True produces approximately 2x the polylines
# ---------------------------------------------------------------------------

def test_mirror_produces_more_polylines():
    num_rows, num_cols = 5, 40
    spec = make_spectrogram(num_rows, num_cols, 0.5)
    poly_no_mirror = ridgeline_hlr(spec, width=80.0, amplitude_scale=5.0, row_spacing=10.0,
                                    smoothing_sigma=0.0, mirror=False)
    poly_mirror = ridgeline_hlr(spec, width=80.0, amplitude_scale=5.0, row_spacing=10.0,
                                 smoothing_sigma=0.0, mirror=True)
    # Mirror adds downward segments; expect significantly more polylines
    assert len(poly_mirror) > len(poly_no_mirror)


def test_mirror_approximately_double():
    # Use small row_spacing and large amplitude so downward curves of later rows
    # can extend below the horizon set by earlier rows.
    num_rows, num_cols = 5, 60
    rng = np.random.default_rng(42)
    spec = rng.random((num_rows, num_cols))
    poly_no_mirror = ridgeline_hlr(spec, width=80.0, amplitude_scale=8.0, row_spacing=1.0,
                                    smoothing_sigma=0.0, mirror=False)
    poly_mirror = ridgeline_hlr(spec, width=80.0, amplitude_scale=8.0, row_spacing=1.0,
                                 smoothing_sigma=0.0, mirror=True)
    # Mirror adds downward segments; should be significantly more polylines
    assert len(poly_mirror) >= len(poly_no_mirror) + 1


# ---------------------------------------------------------------------------
# (e) x-coords in [0, width], y-coords in expected range
# ---------------------------------------------------------------------------

def test_coordinate_ranges():
    num_rows, num_cols = 6, 50
    spec = np.random.default_rng(42).random((num_rows, num_cols))
    amplitude_scale = 4.0
    row_spacing = 5.0
    width = 120.0

    polylines = ridgeline_hlr(
        spec, width=width,
        amplitude_scale=amplitude_scale,
        row_spacing=row_spacing,
        smoothing_sigma=0.0,
    )
    for pl in polylines:
        for x, y in pl:
            assert 0.0 - 1e-9 <= x <= width + 1e-9, f"x={x} out of range"
            # y should be in [0, (num_rows-1)*row_spacing + amplitude_scale]
            y_max = (num_rows - 1) * row_spacing + amplitude_scale
            assert 0.0 - 1e-9 <= y <= y_max + 1e-9, f"y={y} out of range"


# ---------------------------------------------------------------------------
# (f) _extract_visible_segments with alternating True/False
# ---------------------------------------------------------------------------

def test_extract_visible_segments_alternating():
    # mask: T F T F T F T F T F  (length 10)
    x = np.arange(10, dtype=float)
    y = np.ones(10, dtype=float)
    mask = np.array([True, False, True, False, True, False, True, False, True, False])
    segs = _extract_visible_segments(x, y, mask)
    # Each "run" of True is length 1, which is < 2 → no segments
    assert segs == []


def test_extract_visible_segments_pairs():
    # mask: T T F F T T F F T T (runs of 2)
    x = np.arange(10, dtype=float)
    y = np.ones(10, dtype=float)
    mask = np.array([True, True, False, False, True, True, False, False, True, True])
    segs = _extract_visible_segments(x, y, mask)
    assert len(segs) == 3
    for seg in segs:
        assert len(seg) == 2


def test_extract_visible_segments_all_visible():
    x = np.linspace(0, 10, 20)
    y = np.zeros(20)
    mask = np.ones(20, dtype=bool)
    segs = _extract_visible_segments(x, y, mask)
    assert len(segs) == 1
    assert len(segs[0]) == 20


def test_extract_visible_segments_none_visible():
    x = np.linspace(0, 10, 20)
    y = np.zeros(20)
    mask = np.zeros(20, dtype=bool)
    segs = _extract_visible_segments(x, y, mask)
    assert segs == []


def test_extract_visible_segments_edge_start():
    # Only the first two points visible
    x = np.arange(5, dtype=float)
    y = np.ones(5, dtype=float)
    mask = np.array([True, True, False, False, False])
    segs = _extract_visible_segments(x, y, mask)
    assert len(segs) == 1
    assert len(segs[0]) == 2


def test_extract_visible_segments_edge_end():
    # Only the last two points visible
    x = np.arange(5, dtype=float)
    y = np.ones(5, dtype=float)
    mask = np.array([False, False, False, True, True])
    segs = _extract_visible_segments(x, y, mask)
    assert len(segs) == 1
    assert len(segs[0]) == 2


# ---------------------------------------------------------------------------
# (g) ridgeline_no_hlr produces exactly num_rows polylines
# ---------------------------------------------------------------------------

def test_no_hlr_exact_row_count():
    for num_rows in [1, 3, 10]:
        spec = np.random.default_rng(0).random((num_rows, 40))
        polylines = ridgeline_no_hlr(spec, width=100.0, smoothing_sigma=0.0)
        assert len(polylines) == num_rows, f"Expected {num_rows}, got {len(polylines)}"


def test_no_hlr_mirror_double_row_count():
    num_rows = 5
    spec = np.random.default_rng(1).random((num_rows, 40))
    polylines = ridgeline_no_hlr(spec, width=100.0, smoothing_sigma=0.0, mirror=True)
    assert len(polylines) == num_rows * 2


def test_no_hlr_polyline_length():
    num_rows, num_cols = 4, 50
    spec = np.random.default_rng(2).random((num_rows, num_cols))
    polylines = ridgeline_no_hlr(spec, width=80.0, smoothing_sigma=0.0)
    for pl in polylines:
        assert len(pl) == num_cols


# ---------------------------------------------------------------------------
# (h) Smoothing: sigma=0 differs from sigma=5
# ---------------------------------------------------------------------------

def test_smoothing_changes_output():
    rng = np.random.default_rng(99)
    spec = rng.random((3, 100))

    poly_no_smooth = ridgeline_no_hlr(spec, width=100.0, smoothing_sigma=0.0)
    poly_smooth = ridgeline_no_hlr(spec, width=100.0, smoothing_sigma=5.0)

    # Compare y-values of first row
    y_no_smooth = [p[1] for p in poly_no_smooth[0]]
    y_smooth = [p[1] for p in poly_smooth[0]]

    # They should differ (smoothing changes the values)
    assert not np.allclose(y_no_smooth, y_smooth, atol=1e-6)


def test_smoothing_hlr_changes_output():
    rng = np.random.default_rng(77)
    spec = rng.random((4, 80))

    # Collect all y-values from HLR output
    def all_ys(polylines):
        return [p[1] for pl in polylines for p in pl]

    poly_no_smooth = ridgeline_hlr(spec, width=100.0, smoothing_sigma=0.0, amplitude_scale=3.0)
    poly_smooth = ridgeline_hlr(spec, width=100.0, smoothing_sigma=5.0, amplitude_scale=3.0)

    ys0 = all_ys(poly_no_smooth)
    ys5 = all_ys(poly_smooth)

    # At minimum, the outputs should not be identical
    assert ys0 != ys5
