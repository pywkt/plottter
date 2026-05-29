"""Tests for the mask-clip routine in :mod:`plottter.gui.settings_panel._mask`.

The function used to sample only at polyline vertices, which silently
deleted any 2-point hatch line whose endpoints both fell outside the mask
even when the segment crossed the masked region.  These tests pin the
fixed behaviour: clipping happens at the actual mask boundary, not at the
next vertex.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.gui.settings_panel._mask import (
    _MASK_PX_PER_MM,
    _clip_paths_to_mask,
)


# Mask grid covering 100 × 100 mm of canvas at 5 px/mm = 500 × 500 px.
_CANVAS_MM = 100
_PX = _CANVAS_MM * _MASK_PX_PER_MM
_TOL_MM = 1.0 / _MASK_PX_PER_MM + 1e-6  # one-mask-pixel snap tolerance


def _empty_mask() -> np.ndarray:
    return np.zeros((_PX, _PX), dtype=np.float32)


def _rect_mask(x0_mm: float, y0_mm: float, x1_mm: float, y1_mm: float) -> np.ndarray:
    """Mask painted everywhere inside the given mm rectangle."""
    mask = _empty_mask()
    px0 = int(x0_mm * _MASK_PX_PER_MM)
    py0 = int(y0_mm * _MASK_PX_PER_MM)
    px1 = int(x1_mm * _MASK_PX_PER_MM)
    py1 = int(y1_mm * _MASK_PX_PER_MM)
    mask[py0:py1, px0:px1] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Sparse-polyline regression — the bug the user actually hit
# ---------------------------------------------------------------------------


def test_hatch_line_crossing_mask_is_clipped_not_dropped():
    """A 2-point hatch line with both endpoints OUTSIDE the mask but the
    segment crossing through it must be clipped to the in-mask sub-segment,
    not deleted entirely (the pre-fix per-vertex algorithm dropped it)."""
    mask = _rect_mask(40.0, 40.0, 60.0, 60.0)
    hatch = [(10.0, 50.0), (90.0, 50.0)]

    result = _clip_paths_to_mask([hatch], mask)

    assert len(result) == 1, "expected a single clipped sub-segment"
    seg = result[0]
    xs = [p[0] for p in seg]
    ys = [p[1] for p in seg]
    assert min(xs) == pytest.approx(40.0, abs=_TOL_MM)
    assert max(xs) == pytest.approx(60.0, abs=_TOL_MM)
    # y stayed constant on the input → stays constant on the output
    assert max(ys) - min(ys) < _TOL_MM


def test_hatch_line_fully_outside_mask_yields_nothing():
    mask = _rect_mask(40.0, 40.0, 60.0, 60.0)
    hatch = [(10.0, 10.0), (90.0, 10.0)]   # y=10 is well below the rectangle
    assert _clip_paths_to_mask([hatch], mask) == []


def test_hatch_line_fully_inside_mask_kept_intact():
    mask = _rect_mask(0.0, 0.0, 100.0, 100.0)  # whole canvas painted
    hatch = [(10.0, 50.0), (90.0, 50.0)]

    result = _clip_paths_to_mask([hatch], mask)

    assert len(result) == 1
    # First + last sample correspond to the original endpoints
    assert result[0][0] == pytest.approx((10.0, 50.0), abs=_TOL_MM)
    assert result[0][-1] == pytest.approx((90.0, 50.0), abs=_TOL_MM)


def test_segment_enters_and_exits_mask_twice_produces_two_subsegments():
    """A horizontal line crossing TWO disjoint rectangles must produce two
    clipped sub-segments — proves the splitter handles re-entry."""
    mask = np.maximum(
        _rect_mask(20.0, 50.0, 30.0, 60.0),
        _rect_mask(70.0, 50.0, 80.0, 60.0),
    )
    line = [(0.0, 55.0), (100.0, 55.0)]

    result = _clip_paths_to_mask([line], mask)

    assert len(result) == 2
    # Sub-segments are sorted by x via the walk order
    first, second = result
    assert first[0][0] == pytest.approx(20.0, abs=_TOL_MM)
    assert first[-1][0] == pytest.approx(30.0, abs=_TOL_MM)
    assert second[0][0] == pytest.approx(70.0, abs=_TOL_MM)
    assert second[-1][0] == pytest.approx(80.0, abs=_TOL_MM)


# ---------------------------------------------------------------------------
# "Cut a hole" inverted-mask workflow — the user's actual use case
# ---------------------------------------------------------------------------


def test_inverted_mask_carves_hole_in_hatch_line():
    """The intended workflow: paint a small rectangle around a label, click
    Invert Mask, click Apply.  A long hatch line that runs through the
    label area must come back as two sub-segments meeting at the rectangle
    boundary, with the rectangle interior cleanly removed."""
    rect = _rect_mask(45.0, 45.0, 55.0, 55.0)
    inverted = 1.0 - rect          # "everywhere except the rectangle"
    hatch = [(0.0, 50.0), (100.0, 50.0)]

    result = _clip_paths_to_mask([hatch], inverted)

    assert len(result) == 2, "expected hole in the hatch line"
    left, right = result
    # Left sub-segment runs 0 → 45; right runs 55 → 100.  Hole = [45, 55].
    assert left[0][0] == pytest.approx(0.0, abs=_TOL_MM)
    assert left[-1][0] == pytest.approx(45.0, abs=_TOL_MM)
    assert right[0][0] == pytest.approx(55.0, abs=_TOL_MM)
    assert right[-1][0] == pytest.approx(100.0, abs=_TOL_MM)


# ---------------------------------------------------------------------------
# Dense-polyline parity — the supersample step must not corrupt curves
# ---------------------------------------------------------------------------


def test_dense_polyline_inside_mask_preserves_topology():
    """A many-vertex polyline that lives entirely inside the mask comes
    back as one connected segment (no spurious splits from supersampling)."""
    mask = _rect_mask(0.0, 0.0, 100.0, 100.0)
    poly = [(10.0 + i * 0.5, 50.0) for i in range(160)]  # 160 points spanning ~80mm

    result = _clip_paths_to_mask([poly], mask)

    assert len(result) == 1
    # Endpoints preserved (or coincident within tolerance)
    assert result[0][0] == pytest.approx(poly[0], abs=_TOL_MM)
    assert result[0][-1] == pytest.approx(poly[-1], abs=_TOL_MM)


def test_empty_paths_yield_empty_result():
    assert _clip_paths_to_mask([], _empty_mask()) == []


def test_single_point_polyline_is_dropped():
    """A 1-point polyline can't form a segment — never returned."""
    mask = _rect_mask(0.0, 0.0, 100.0, 100.0)
    assert _clip_paths_to_mask([[(50.0, 50.0)]], mask) == []
