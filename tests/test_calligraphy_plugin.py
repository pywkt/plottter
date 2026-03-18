"""Tests for the Calligraphy plugin (Phase A — skeleton and offset curve engine).

Covers:
  (a) _compute_normals on a horizontal line returns (0, -1) normals
  (b) _compute_normals on a vertical line returns (1, 0) normals
  (c) _offset_polyline with constant width produces paths at the correct
      distance from the centerline
  (d) _offset_polyline with num_lines=2 returns exactly 2 polylines
  (e) _offset_polyline with num_lines=6 returns exactly 6 polylines
  (f) Circle demo shape produces non-empty output
  (g) generator is registered and appears in the GENERATORS dict
"""
from __future__ import annotations

import math
import sys
import os

import pytest

# Ensure the plugins directory is on the path so calligraphy.py can be
# imported directly, and that plottter is importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLUGINS_DIR = os.path.join(_ROOT, "plugins")

# Import plottter.generators first to populate the registry, then load
# the calligraphy plugin via the plugin loader.
import plottter.generators  # noqa: E402 — triggers _import_builtin_generators
from plottter.generators import GENERATORS, load_plugins
from plottter.models import Canvas

# Load plugins from the project-level plugins/ directory.
load_plugins(extra_dirs=[_PLUGINS_DIR])

# Now import the helper functions directly for unit testing.
# We use importlib so that tests work regardless of whether the plugins/
# directory is on sys.path.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "plottter_plugin_calligraphy",
    os.path.join(_PLUGINS_DIR, "calligraphy.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

_compute_normals = _mod._compute_normals
_offset_polyline = _mod._offset_polyline
_clean_offset_path = _mod._clean_offset_path
_calligraphic_widths = _mod._calligraphic_widths
_speed_widths = _mod._speed_widths
_make_circle_path = _mod._make_circle_path
_text_to_centerlines = _mod._text_to_centerlines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _a4_canvas() -> Canvas:
    """Return a standard A4 canvas with 10 mm margins."""
    return Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0)


def _horizontal_path(length: float = 20.0, n: int = 3) -> list[tuple[float, float]]:
    """A path travelling in the +X direction."""
    return [(length * i / (n - 1), 0.0) for i in range(n)]


def _vertical_path(length: float = 20.0, n: int = 3) -> list[tuple[float, float]]:
    """A path travelling in the +Y direction."""
    return [(0.0, length * i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# (a) _compute_normals on a horizontal line returns (0, -1) normals
# ---------------------------------------------------------------------------

def test_compute_normals_horizontal_line():
    """A path going right (+X) should have normals pointing up (0, -1)."""
    path = _horizontal_path()
    normals = _compute_normals(path)
    assert len(normals) == len(path)
    for nx, ny in normals:
        assert abs(nx - 0.0) < 1e-9, f"Expected nx=0, got {nx}"
        assert abs(ny - (-1.0)) < 1e-9, f"Expected ny=-1, got {ny}"


# ---------------------------------------------------------------------------
# (b) _compute_normals on a vertical line returns (1, 0) normals
# ---------------------------------------------------------------------------

def test_compute_normals_vertical_line():
    """A path going down (+Y) should have normals pointing right (1, 0)."""
    path = _vertical_path()
    normals = _compute_normals(path)
    assert len(normals) == len(path)
    for nx, ny in normals:
        assert abs(nx - 1.0) < 1e-9, f"Expected nx=1, got {nx}"
        assert abs(ny - 0.0) < 1e-9, f"Expected ny=0, got {ny}"


# ---------------------------------------------------------------------------
# (c) _offset_polyline with constant width → correct distance from centerline
# ---------------------------------------------------------------------------

def test_offset_polyline_distance_from_centerline():
    """Offset lines should be exactly ±width/2 from the centerline.

    Uses a simple horizontal path at y=0 with width=4 mm.
    Expected offsets: y = -2 mm and y = +2 mm (for num_lines=2).
    """
    path = _horizontal_path(length=20.0, n=5)
    width = 4.0
    widths = [width] * len(path)

    result = _offset_polyline(path, widths, num_lines=2)
    assert len(result) == 2

    # Collect all unique y-values (x-axis path → offsets differ only in y)
    y_values = set()
    for polyline in result:
        for _x, y in polyline:
            y_values.add(round(y, 6))

    # Should have exactly two y-values: -(width/2) and +(width/2)
    expected_ys = {-width / 2, width / 2}
    assert y_values == expected_ys, f"Expected y offsets {expected_ys}, got {y_values}"


# ---------------------------------------------------------------------------
# (d) _offset_polyline with num_lines=2 returns exactly 2 polylines
# ---------------------------------------------------------------------------

def test_offset_polyline_num_lines_2():
    """_offset_polyline(path, widths, 2) must return exactly 2 polylines."""
    path = _horizontal_path()
    widths = [4.0] * len(path)
    result = _offset_polyline(path, widths, num_lines=2)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# (e) _offset_polyline with num_lines=6 returns exactly 6 polylines
# ---------------------------------------------------------------------------

def test_offset_polyline_num_lines_6():
    """_offset_polyline(path, widths, 6) must return exactly 6 polylines."""
    path = _horizontal_path(length=30.0, n=10)
    widths = [4.0] * len(path)
    result = _offset_polyline(path, widths, num_lines=6)
    assert len(result) == 6


# ---------------------------------------------------------------------------
# Additional _offset_polyline correctness tests
# ---------------------------------------------------------------------------

def test_offset_polyline_evenly_spaced():
    """Intermediate offsets for num_lines=5 should be evenly spaced."""
    path = _horizontal_path(length=20.0, n=4)
    width = 10.0
    widths = [width] * len(path)
    result = _offset_polyline(path, widths, num_lines=5)
    assert len(result) == 5

    # Collect the y-value of each offset line (all points in a line share y)
    y_vals = sorted(polyline[0][1] for polyline in result)

    # Expected y-values: -5, -2.5, 0, 2.5, 5
    expected = [width * (i / 4 - 0.5) for i in range(5)]
    for got, exp in zip(y_vals, expected):
        assert abs(got - exp) < 1e-9, f"Expected y={exp}, got {got}"


def test_offset_polyline_returns_empty_for_single_point():
    """A path with fewer than 2 points produces no output."""
    result = _offset_polyline([(5.0, 5.0)], [4.0], num_lines=4)
    assert result == []


def test_offset_polyline_short_widths_list():
    """If widths is shorter than path, the last value is repeated."""
    path = [(float(i), 0.0) for i in range(10)]
    widths = [4.0]  # shorter than path
    result = _offset_polyline(path, widths, num_lines=2)
    assert len(result) == 2
    # Both lines should be at ±2 mm
    for polyline in result:
        y_vals = set(round(pt[1], 6) for pt in polyline)
        assert len(y_vals) == 1  # all y-values in each line should be the same


# ---------------------------------------------------------------------------
# _clean_offset_path tests
# ---------------------------------------------------------------------------

def test_clean_offset_path_no_jumps():
    """A smooth path should be returned unchanged (as a single sub-path)."""
    path = [(float(i), 0.0) for i in range(10)]
    result = _clean_offset_path(path)
    assert len(result) == 1
    assert result[0] == path


def test_clean_offset_path_large_jump():
    """A path with a large jump should be split into two sub-paths."""
    # Create a path with a huge jump in the middle
    path = [(0.0, 0.0), (1.0, 0.0), (1000.0, 0.0), (1001.0, 0.0)]
    result = _clean_offset_path(path)
    assert len(result) == 2
    assert result[0] == [(0.0, 0.0), (1.0, 0.0)]
    assert result[1] == [(1000.0, 0.0), (1001.0, 0.0)]


def test_clean_offset_path_empty():
    """An empty path returns an empty list."""
    assert _clean_offset_path([]) == []


def test_clean_offset_path_single_point():
    """A single-point path returns an empty list (no valid segments)."""
    assert _clean_offset_path([(0.0, 0.0)]) == []


# ---------------------------------------------------------------------------
# _compute_normals edge cases
# ---------------------------------------------------------------------------

def test_compute_normals_empty():
    """Empty path returns empty list."""
    assert _compute_normals([]) == []


def test_compute_normals_single_point():
    """Single point returns one default normal."""
    result = _compute_normals([(5.0, 3.0)])
    assert len(result) == 1


def test_compute_normals_degenerate_duplicate_points():
    """Duplicate consecutive points should not cause division by zero."""
    path = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
    normals = _compute_normals(path)
    assert len(normals) == 3
    # All normals should be valid unit vectors
    for nx, ny in normals:
        assert abs(math.hypot(nx, ny) - 1.0) < 1e-9 or (nx == 0.0 and ny == -1.0)


# ---------------------------------------------------------------------------
# (f) Circle demo shape produces non-empty output
# ---------------------------------------------------------------------------

def test_generate_circle_produces_output():
    """The Circle path source must produce non-empty polylines."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {"path_source": "Circle", "num_parallel_lines": 6, "stroke_width_mm": 4.0}
    result = gen.generate(params, canvas)
    assert isinstance(result, list)
    assert len(result) > 0
    # Each polyline should have at least 2 points
    for polyline in result:
        assert len(polyline) >= 2


def test_generate_spiral_produces_output():
    """The Spiral path source must produce non-empty polylines."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {"path_source": "Spiral", "num_parallel_lines": 4, "stroke_width_mm": 3.0}
    result = gen.generate(params, canvas)
    assert len(result) > 0


def test_generate_wave_produces_output():
    """The Wave path source must produce non-empty polylines."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {"path_source": "Wave", "num_parallel_lines": 3, "stroke_width_mm": 5.0}
    result = gen.generate(params, canvas)
    assert len(result) > 0


def test_generate_figure8_produces_output():
    """The Figure 8 path source must produce non-empty polylines."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {"path_source": "Figure 8", "num_parallel_lines": 5, "stroke_width_mm": 4.0}
    result = gen.generate(params, canvas)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# (g) Generator is registered and appears in the GENERATORS dict
# ---------------------------------------------------------------------------

def test_generator_is_registered():
    """The CalligraphyGenerator must be present in the GENERATORS registry."""
    assert "Calligraphy" in GENERATORS, (
        f"'Calligraphy' not found in GENERATORS. Available: {list(GENERATORS.keys())}"
    )


def test_generator_category():
    """The CalligraphyGenerator must be in the 'math' category."""
    cls = GENERATORS["Calligraphy"]
    assert cls.category == "math"


def test_generator_has_parameters():
    """get_parameters() must return a non-empty list."""
    gen = GENERATORS["Calligraphy"]()
    params = gen.get_parameters()
    assert len(params) > 0
    # Verify the three required parameters are present
    param_names = {p.name for p in params}
    assert "path_source" in param_names
    assert "num_parallel_lines" in param_names
    assert "stroke_width_mm" in param_names


def test_generator_has_presets():
    """get_presets() must return at least one preset."""
    gen = GENERATORS["Calligraphy"]()
    presets = gen.get_presets()
    assert len(presets) > 0
    for preset in presets:
        assert preset.name
        assert isinstance(preset.params, dict)


def test_preset_params_are_valid():
    """Every preset must only reference params defined in get_parameters()."""
    gen = GENERATORS["Calligraphy"]()
    valid_names = {p.name for p in gen.get_parameters()}
    for preset in gen.get_presets():
        for key in preset.params:
            assert key in valid_names, (
                f"Preset '{preset.name}' references unknown param '{key}'"
            )


def test_generate_with_progress_callback():
    """progress_callback must be called at least once during generate()."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {"path_source": "Circle", "num_parallel_lines": 2, "stroke_width_mm": 4.0}
    calls: list[float] = []
    gen.generate(params, canvas, progress_callback=lambda v: calls.append(v))
    assert len(calls) > 0
    assert calls[-1] == 1.0


def test_generate_cancelled():
    """When cancelled_callback returns True, generate() returns empty list."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {"path_source": "Wave", "num_parallel_lines": 6, "stroke_width_mm": 4.0}
    result = gen.generate(
        params, canvas,
        progress_callback=None,
        cancelled_callback=lambda: True,
    )
    assert result == []


# ===========================================================================
# Phase B tests — Calligraphic width model and pen angle (task 17.2)
# ===========================================================================

# ---------------------------------------------------------------------------
# (a) _calligraphic_widths on a horizontal line with pen_angle=0 → min widths
#
# A horizontal line (direction = 0°) is PARALLEL to a pen nib at 0°.
# |sin(0 - 0)| = 0 → should return min_width at every point (after smoothing).
# ---------------------------------------------------------------------------

def test_calligraphic_widths_horizontal_parallel_to_nib():
    """Horizontal path with pen_angle=0 → widths should equal min_width_mm."""
    path = _horizontal_path(length=40.0, n=20)
    min_w, max_w = 0.5, 5.0
    widths = _calligraphic_widths(path, pen_angle_deg=0.0, min_width_mm=min_w, max_width_mm=max_w)
    assert len(widths) == len(path)
    for w in widths:
        assert abs(w - min_w) < 0.05, (
            f"Expected min width ~{min_w}, got {w} (path parallel to pen nib)"
        )


# ---------------------------------------------------------------------------
# (b) _calligraphic_widths on a vertical line with pen_angle=0 → max widths
#
# A vertical path has direction = 90°.  |sin(90° - 0°)| = 1 → max_width.
# ---------------------------------------------------------------------------

def test_calligraphic_widths_vertical_perpendicular_to_nib():
    """Vertical path with pen_angle=0 → widths should equal max_width_mm."""
    path = _vertical_path(length=40.0, n=20)
    min_w, max_w = 0.5, 5.0
    widths = _calligraphic_widths(path, pen_angle_deg=0.0, min_width_mm=min_w, max_width_mm=max_w)
    assert len(widths) == len(path)
    for w in widths:
        assert abs(w - max_w) < 0.05, (
            f"Expected max width ~{max_w}, got {w} (path perpendicular to pen nib)"
        )


# ---------------------------------------------------------------------------
# (c) _calligraphic_widths on a circle with pen_angle=45 → oscillates min/max
# ---------------------------------------------------------------------------

def test_calligraphic_widths_circle_oscillates():
    """Circle path with pen_angle=45 should produce widths spanning min..max."""
    # A circle traverses all directions → |sin(θ-45°)| sweeps 0..1
    cx, cy = 100.0, 148.5  # centre of A4 canvas
    n_pts = 128
    path = _make_circle_path(cx, cy, 50.0, n_pts=n_pts)
    min_w, max_w = 0.5, 5.0
    widths = _calligraphic_widths(path, pen_angle_deg=45.0, min_width_mm=min_w, max_width_mm=max_w)
    assert len(widths) == len(path)
    # All widths must be in [min_w, max_w]
    for w in widths:
        assert min_w - 0.1 <= w <= max_w + 0.1, f"Width {w} out of [{min_w}, {max_w}]"
    # Widths must vary — range should be at least 80 % of total span
    w_range = max(widths) - min(widths)
    expected_range = max_w - min_w
    assert w_range >= 0.8 * expected_range, (
        f"Width range {w_range:.2f} too small (expected ≥ {0.8*expected_range:.2f})"
    )


# ---------------------------------------------------------------------------
# (d) Width smoothing — no abrupt jumps between consecutive widths
# ---------------------------------------------------------------------------

def test_calligraphic_widths_smoothing_no_abrupt_jumps():
    """Consecutive width values must not differ by more than half the total range."""
    path = _make_circle_path(100.0, 100.0, 40.0, n_pts=64)
    min_w, max_w = 0.5, 5.0
    widths = _calligraphic_widths(path, pen_angle_deg=30.0, min_width_mm=min_w, max_width_mm=max_w)
    total_range = max_w - min_w
    max_allowed_delta = 0.5 * total_range  # generous threshold
    for i in range(1, len(widths)):
        delta = abs(widths[i] - widths[i - 1])
        assert delta <= max_allowed_delta, (
            f"Abrupt width jump at index {i}: {widths[i-1]:.3f} → {widths[i]:.3f} "
            f"(delta={delta:.3f}, allowed≤{max_allowed_delta:.3f})"
        )


# ---------------------------------------------------------------------------
# (e) width_mode="Constant" produces uniform-width output matching Phase A
# ---------------------------------------------------------------------------

def test_generate_constant_mode_matches_phase_a():
    """width_mode='Constant' must produce the same widths as Phase A behaviour."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    stroke_w = 4.0
    num_lines = 6
    params_constant = {
        "path_source": "Circle",
        "width_mode": "Constant",
        "stroke_width_mm": stroke_w,
        "num_parallel_lines": num_lines,
    }
    result = gen.generate(params_constant, canvas)
    assert len(result) > 0
    # Each line should exist — count should equal num_lines (no splits for circle)
    assert len(result) == num_lines


def test_generate_calligraphic_mode_produces_variable_widths():
    """Calligraphic mode must genuinely vary stroke widths, not produce constant offsets.

    On a Circle path with pen_angle=0 the width oscillates between min_width
    (when tangent is parallel to pen) and max_width (when tangent is
    perpendicular).  The outermost offset lines therefore extend further from
    the centerline than a Constant mode render using the same min_width —
    the calligraphic bounding box must be measurably wider.
    """
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()

    # Baseline: Constant mode at the minimum width (no width variation at all).
    params_const = {
        "path_source": "Circle",
        "width_mode": "Constant",
        "stroke_width_mm": 0.5,  # == min_width_mm below
        "num_parallel_lines": 4,
    }
    result_const = gen.generate(params_const, canvas)
    assert len(result_const) > 0

    # Calligraphic: pen_angle=0 on a circle causes widths to swing from
    # min_width=0.5 mm up to max_width=6.0 mm.
    params_callig = {
        "path_source": "Circle",
        "width_mode": "Calligraphic",
        "pen_angle_deg": 0.0,
        "min_width_mm": 0.5,
        "max_width_mm": 6.0,
        "speed_influence": 0.0,
        "num_parallel_lines": 4,
    }
    result_callig = gen.generate(params_callig, canvas)
    assert len(result_callig) > 0

    def _x_extent(paths: list) -> float:
        all_x = [pt[0] for poly in paths for pt in poly]
        return max(all_x) - min(all_x)

    extent_const = _x_extent(result_const)
    extent_callig = _x_extent(result_callig)

    # Calligraphic output spans significantly more in X because the maximum
    # offset (max_width/2 = 3 mm) is much larger than the constant offset
    # (stroke_width/2 = 0.25 mm) at the points where the circle's tangent is
    # perpendicular to the pen.
    assert extent_callig > extent_const, (
        f"Calligraphic x-extent ({extent_callig:.2f} mm) should exceed "
        f"constant x-extent ({extent_const:.2f} mm)"
    )


def test_generate_speed_influence_produces_output():
    """speed_influence > 0 must produce non-empty output without errors."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {
        "path_source": "Spiral",
        "width_mode": "Calligraphic",
        "pen_angle_deg": 45.0,
        "min_width_mm": 0.3,
        "max_width_mm": 4.0,
        "speed_influence": 0.5,
        "num_parallel_lines": 4,
    }
    result = gen.generate(params, canvas)
    assert len(result) > 0


def test_calligraphic_widths_empty_path():
    """Empty path returns empty list."""
    assert _calligraphic_widths([], 45.0, 0.5, 5.0) == []


def test_calligraphic_widths_single_point():
    """Single-point path returns a list of one width value."""
    result = _calligraphic_widths([(0.0, 0.0)], 45.0, 0.5, 5.0)
    assert len(result) == 1
    assert 0.5 <= result[0] <= 5.0


def test_calligraphic_widths_clipped_to_range():
    """All returned widths must be within [min_width_mm, max_width_mm]."""
    path = _make_circle_path(0.0, 0.0, 30.0, n_pts=50)
    min_w, max_w = 1.0, 8.0
    widths = _calligraphic_widths(path, pen_angle_deg=22.5, min_width_mm=min_w, max_width_mm=max_w)
    for w in widths:
        assert min_w - 1e-6 <= w <= max_w + 1e-6, f"Width {w} out of [{min_w}, {max_w}]"


def test_new_params_in_get_parameters():
    """All Phase B parameters must appear in get_parameters()."""
    gen = GENERATORS["Calligraphy"]()
    param_names = {p.name for p in gen.get_parameters()}
    required = {"width_mode", "pen_angle_deg", "min_width_mm", "max_width_mm", "speed_influence"}
    for name in required:
        assert name in param_names, f"Missing parameter: {name}"


def test_preset_params_valid_phase_b():
    """All presets (including new Phase B presets) must reference valid params."""
    gen = GENERATORS["Calligraphy"]()
    valid_names = {p.name for p in gen.get_parameters()}
    for preset in gen.get_presets():
        for key in preset.params:
            assert key in valid_names, (
                f"Preset '{preset.name}' references unknown param '{key}'"
            )


# ---------------------------------------------------------------------------
# _speed_widths unit tests
# ---------------------------------------------------------------------------

def test_speed_widths_empty_path():
    """Empty path returns empty list."""
    assert _speed_widths([], 0.5, 5.0) == []


def test_speed_widths_single_point():
    """Single-point path returns [min_width_mm]."""
    result = _speed_widths([(0.0, 0.0)], 0.5, 5.0)
    assert result == [0.5]


def test_speed_widths_all_zero_length_guard():
    """All-zero-length path triggers the max_speed < 1e-10 guard → all min_width."""
    # Three coincident points produce zero-length segments.
    path = [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)]
    result = _speed_widths(path, 0.5, 5.0)
    assert len(result) == 3
    assert all(w == pytest.approx(0.5) for w in result)


def test_speed_widths_uniform_segments_returns_min_width():
    """Equal-length segments → all points at max speed → all widths equal min_width."""
    # Uniform grid: every segment has length 1.
    path = [(float(i), 0.0) for i in range(5)]
    result = _speed_widths(path, 0.5, 5.0)
    assert len(result) == 5
    # Each speed equals the global max speed, so (1 - s/max_speed) = 0.
    assert all(w == pytest.approx(0.5) for w in result), (
        f"Expected all min_width (0.5), got {result}"
    )


def test_speed_widths_inverse_speed_relationship():
    """Slow segments produce wider lines; fast segments produce narrower lines.

    Path: two segments — a very short one (≈ 0 length, slow stroke) followed
    by a very long one (fast stroke).  The first point (slow) should get a
    width near max_width; the last point (fast) should get min_width.
    """
    short = 0.001   # near-zero = very slow
    long_ = 1000.0  # large = very fast
    path = [(0.0, 0.0), (short, 0.0), (short + long_, 0.0)]

    min_w, max_w = 0.5, 5.0
    result = _speed_widths(path, min_w, max_w)

    assert len(result) == 3

    # First point: speed = short segment length (tiny) → near max_width.
    assert result[0] > (min_w + max_w) / 2.0, (
        f"First point (slow) should be wider than midpoint; got {result[0]:.4f}"
    )

    # Last point: speed = long segment length (max) → min_width.
    assert result[2] == pytest.approx(min_w, abs=1e-6), (
        f"Last point (fastest) should be min_width ({min_w}); got {result[2]:.4f}"
    )

    # Widths must stay within [min_w, max_w].
    for w in result:
        assert min_w - 1e-9 <= w <= max_w + 1e-9, f"Width {w} out of range"


# ===========================================================================
# Phase C tests — Text rendering with Hershey fonts (task 17.3)
# ===========================================================================

# ---------------------------------------------------------------------------
# (a) _text_to_centerlines("A", "Simplex", 20, 0, 1.5) returns non-empty paths
# ---------------------------------------------------------------------------

def test_text_to_centerlines_single_char_returns_paths():
    """Single uppercase letter must produce at least one centerline path."""
    paths, w, h = _text_to_centerlines("A", "Simplex", 20.0, 0.0, 1.5)
    assert len(paths) > 0, "Expected at least one centerline for 'A'"
    assert w > 0.0
    assert h > 0.0
    for path in paths:
        assert len(path) >= 2


# ---------------------------------------------------------------------------
# (b) Multi-character text returns more strokes than single character
# ---------------------------------------------------------------------------

def test_text_to_centerlines_multi_char_more_strokes():
    """'AB' should produce more strokes than 'A' alone."""
    paths_a, _, _ = _text_to_centerlines("A", "Simplex", 20.0, 0.0, 1.5)
    paths_ab, _, _ = _text_to_centerlines("AB", "Simplex", 20.0, 0.0, 1.5)
    assert len(paths_ab) > len(paths_a), (
        f"'AB' should have more strokes than 'A': {len(paths_ab)} vs {len(paths_a)}"
    )


# ---------------------------------------------------------------------------
# (c) Multi-line text has total height > single line
# ---------------------------------------------------------------------------

def test_text_to_centerlines_multiline_taller():
    """'A\\nB' should have greater total height than 'A'."""
    _, _, h_single = _text_to_centerlines("A", "Simplex", 20.0, 0.0, 1.5)
    _, _, h_multi = _text_to_centerlines("A\nB", "Simplex", 20.0, 0.0, 1.5)
    assert h_multi > h_single, (
        f"Multi-line height ({h_multi:.2f}) should exceed single-line ({h_single:.2f})"
    )


# ---------------------------------------------------------------------------
# (d) All four Hershey fonts produce non-empty output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("font_name", ["Simplex", "Duplex", "Script", "Gothic"])
def test_text_to_centerlines_all_fonts(font_name):
    """Every Hershey font variant must produce non-empty output for 'Hi'."""
    paths, w, h = _text_to_centerlines("Hi", font_name, 20.0, 0.5, 1.5)
    assert len(paths) > 0, f"Font '{font_name}' produced no paths for 'Hi'"
    assert w > 0.0
    assert h > 0.0


# ---------------------------------------------------------------------------
# (e) Full generate() with path_source="Text" and width_mode="Calligraphic"
# ---------------------------------------------------------------------------

def test_generate_text_calligraphic_produces_output():
    """generate() in Text + Calligraphic mode must return non-empty polylines."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {
        "path_source": "Text",
        "text": "Hello",
        "hershey_font": "Script",
        "font_size_mm": 30.0,
        "letter_spacing_mm": 1.0,
        "line_spacing": 1.5,
        "text_align": "Center",
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
        "width_mode": "Calligraphic",
        "pen_angle_deg": 45.0,
        "min_width_mm": 0.3,
        "max_width_mm": 5.0,
        "speed_influence": 0.0,
        "num_parallel_lines": 4,
    }
    result = gen.generate(params, canvas)
    assert isinstance(result, list)
    assert len(result) > 0
    for polyline in result:
        assert len(polyline) >= 2


# ---------------------------------------------------------------------------
# (f) Text output paths are within canvas drawing area (with tolerance)
# ---------------------------------------------------------------------------

def test_generate_text_paths_within_canvas():
    """Text output paths should lie within the canvas drawing area."""
    canvas = _a4_canvas()
    x1, y1, x2, y2 = canvas.drawing_area()
    gen = GENERATORS["Calligraphy"]()
    params = {
        "path_source": "Text",
        "text": "Hi",
        "hershey_font": "Simplex",
        "font_size_mm": 20.0,
        "letter_spacing_mm": 0.5,
        "line_spacing": 1.5,
        "text_align": "Center",
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
        "width_mode": "Constant",
        "stroke_width_mm": 1.0,
        "num_parallel_lines": 2,
    }
    result = gen.generate(params, canvas)
    assert len(result) > 0
    # Allow a generous tolerance for offset curves slightly outside drawing bounds
    tol = 10.0
    for polyline in result:
        for x, y in polyline:
            assert x1 - tol <= x <= x2 + tol, f"X={x:.2f} outside [{x1:.2f}, {x2:.2f}]"
            assert y1 - tol <= y <= y2 + tol, f"Y={y:.2f} outside [{y1:.2f}, {y2:.2f}]"


# ---------------------------------------------------------------------------
# (g) Empty text returns empty paths without error
# ---------------------------------------------------------------------------

def test_generate_empty_text_returns_empty():
    """generate() with empty text must return [] without raising."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {
        "path_source": "Text",
        "text": "",
        "hershey_font": "Simplex",
        "font_size_mm": 20.0,
        "letter_spacing_mm": 0.5,
        "line_spacing": 1.5,
        "text_align": "Center",
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
        "width_mode": "Constant",
        "stroke_width_mm": 1.0,
        "num_parallel_lines": 2,
    }
    result = gen.generate(params, canvas)
    assert result == []


# ---------------------------------------------------------------------------
# Phase C: text parameters exist in get_parameters()
# ---------------------------------------------------------------------------

def test_text_params_in_get_parameters():
    """All Phase C text parameters must appear in get_parameters()."""
    gen = GENERATORS["Calligraphy"]()
    param_names = {p.name for p in gen.get_parameters()}
    required = {
        "text", "hershey_font", "font_size_mm",
        "letter_spacing_mm", "line_spacing",
        "text_align", "x_offset_mm", "y_offset_mm",
    }
    for name in required:
        assert name in param_names, f"Missing Phase C parameter: {name}"


# ---------------------------------------------------------------------------
# Phase C: text presets reference only valid params
# ---------------------------------------------------------------------------

def test_text_presets_valid_params():
    """Text presets must only reference params defined in get_parameters()."""
    gen = GENERATORS["Calligraphy"]()
    valid_names = {p.name for p in gen.get_parameters()}
    text_presets = [p for p in gen.get_presets() if p.params.get("path_source") == "Text"]
    assert len(text_presets) >= 1, "Expected at least one Text preset"
    for preset in text_presets:
        for key in preset.params:
            assert key in valid_names, (
                f"Preset '{preset.name}' references unknown param '{key}'"
            )


# ---------------------------------------------------------------------------
# _text_to_centerlines: empty text returns empty list
# ---------------------------------------------------------------------------

def test_text_to_centerlines_empty_text():
    """Empty text string returns ([], 0.0, 0.0)."""
    paths, w, h = _text_to_centerlines("", "Simplex", 20.0, 0.0, 1.5)
    assert paths == []
    assert w == 0.0
    assert h == 0.0


# ===========================================================================
# Phase D tests — Presets, documentation, and polish (task 17.4)
# ===========================================================================

# ---------------------------------------------------------------------------
# (a) All 6 spec presets produce non-empty output on a default A4 canvas.
#     The spec lists: "Broad Nib Italic", "Monoline Script",
#     "Gothic Blackletter", "Brush Pen Spiral" (already existed),
#     "Thin Copperplate", "Decorative Wave".
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset_name", [
    "Broad Nib Italic",
    "Monoline Script",
    "Gothic Blackletter",
    "Brush Pen Spiral",
    "Thin Copperplate",
    "Decorative Wave",
])
def test_phase_d_presets_produce_output(preset_name):
    """Each Phase D preset must produce non-empty polylines on an A4 canvas."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    # Find the preset by name
    presets = {p.name: p for p in gen.get_presets()}
    assert preset_name in presets, (
        f"Preset '{preset_name}' not found. Available: {list(presets.keys())}"
    )
    preset = presets[preset_name]
    result = gen.generate(preset.params, canvas)
    assert isinstance(result, list)
    assert len(result) > 0, f"Preset '{preset_name}' produced no output"
    for polyline in result:
        assert len(polyline) >= 2, (
            f"Preset '{preset_name}' produced a polyline with < 2 points"
        )


# ---------------------------------------------------------------------------
# (b) Empty text returns empty paths without error  (re-tested with context)
# ---------------------------------------------------------------------------

def test_phase_d_empty_text_no_error():
    """Empty text input to generate() returns [] and does not raise."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {
        "path_source": "Text",
        "text": "",
        "hershey_font": "Gothic",
        "font_size_mm": 40.0,
        "letter_spacing_mm": 1.0,
        "line_spacing": 1.5,
        "text_align": "Center",
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
        "width_mode": "Calligraphic",
        "pen_angle_deg": 30.0,
        "min_width_mm": 0.5,
        "max_width_mm": 6.0,
        "speed_influence": 0.0,
        "num_parallel_lines": 10,
    }
    result = gen.generate(params, canvas)
    assert result == []


# ---------------------------------------------------------------------------
# (c) Single-character text produces valid output
# ---------------------------------------------------------------------------

def test_phase_d_single_char_text_valid_output():
    """generate() with a single character must produce valid (non-empty) output."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {
        "path_source": "Text",
        "text": "A",
        "hershey_font": "Script",
        "font_size_mm": 35.0,
        "letter_spacing_mm": 1.0,
        "line_spacing": 1.5,
        "text_align": "Center",
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
        "width_mode": "Calligraphic",
        "pen_angle_deg": 40.0,
        "min_width_mm": 0.3,
        "max_width_mm": 5.0,
        "speed_influence": 0.0,
        "num_parallel_lines": 8,
    }
    result = gen.generate(params, canvas)
    assert len(result) > 0
    for polyline in result:
        assert len(polyline) >= 2


# ---------------------------------------------------------------------------
# (d) progress_callback is called during generation for each path source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_source", ["Circle", "Spiral", "Wave", "Figure 8", "Text"])
def test_phase_d_progress_callback_called(path_source):
    """progress_callback must be called at least once for every path source."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params: dict = {
        "path_source": path_source,
        "width_mode": "Constant",
        "stroke_width_mm": 3.0,
        "num_parallel_lines": 2,
    }
    if path_source == "Text":
        params.update({"text": "Hi", "hershey_font": "Simplex",
                       "font_size_mm": 20.0, "letter_spacing_mm": 0.5,
                       "line_spacing": 1.5, "text_align": "Center",
                       "x_offset_mm": 0.0, "y_offset_mm": 0.0})
    calls: list[float] = []
    gen.generate(params, canvas, progress_callback=lambda v: calls.append(v))
    assert len(calls) > 0, f"progress_callback not called for path_source='{path_source}'"
    # Final call must report 1.0 (complete)
    assert calls[-1] == pytest.approx(1.0), (
        f"Last progress value for '{path_source}' should be 1.0, got {calls[-1]}"
    )


# ---------------------------------------------------------------------------
# (e) cancelled_callback stops generation early for non-text paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_source", ["Circle", "Spiral", "Wave", "Figure 8"])
def test_phase_d_cancelled_callback_non_text(path_source):
    """cancelled_callback returning True must stop generation for demo shapes."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {
        "path_source": path_source,
        "width_mode": "Constant",
        "stroke_width_mm": 4.0,
        "num_parallel_lines": 6,
    }
    result = gen.generate(
        params, canvas,
        progress_callback=None,
        cancelled_callback=lambda: True,
    )
    assert result == [], (
        f"Expected [] when cancelled for path_source='{path_source}', got {result}"
    )


# ---------------------------------------------------------------------------
# (f) All demo shapes produce output fitting within canvas drawing area
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_source", ["Circle", "Spiral", "Wave", "Figure 8"])
def test_phase_d_demo_shapes_within_canvas(path_source):
    """Demo shape output must fit within the canvas drawing area (with tolerance)."""
    canvas = _a4_canvas()
    x1, y1, x2, y2 = canvas.drawing_area()
    gen = GENERATORS["Calligraphy"]()
    params = {
        "path_source": path_source,
        "width_mode": "Calligraphic",
        "pen_angle_deg": 45.0,
        "min_width_mm": 0.5,
        "max_width_mm": 5.0,
        "speed_influence": 0.0,
        "num_parallel_lines": 4,
    }
    result = gen.generate(params, canvas)
    assert len(result) > 0, f"'{path_source}' produced no output"
    # Allow a generous tolerance for offset curves near the boundary
    tol = 15.0
    for polyline in result:
        for x, y in polyline:
            assert x1 - tol <= x <= x2 + tol, (
                f"'{path_source}': X={x:.2f} outside [{x1:.2f}, {x2:.2f}]"
            )
            assert y1 - tol <= y <= y2 + tol, (
                f"'{path_source}': Y={y:.2f} outside [{y1:.2f}, {y2:.2f}]"
            )


# ---------------------------------------------------------------------------
# (g) Plugin appears in GENERATORS with name "Calligraphy" and category "math"
#     (re-tested explicitly for Phase D)
# ---------------------------------------------------------------------------

def test_phase_d_generator_registration():
    """Phase D: generator must still be registered as 'Calligraphy' in 'math'."""
    assert "Calligraphy" in GENERATORS
    cls = GENERATORS["Calligraphy"]
    assert cls.name == "Calligraphy"
    assert cls.category == "math"


# ---------------------------------------------------------------------------
# num_parallel_lines clamping
# ---------------------------------------------------------------------------

def test_phase_d_num_lines_clamped_to_minimum():
    """num_parallel_lines=0 should be clamped to 1 (not raise or produce 0 lines)."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {
        "path_source": "Circle",
        "width_mode": "Constant",
        "stroke_width_mm": 4.0,
        "num_parallel_lines": 0,  # below minimum
    }
    result = gen.generate(params, canvas)
    # Should produce 1 line (center only), not error
    assert isinstance(result, list)
    # With a circle + num_lines=1 we expect at least 1 polyline
    assert len(result) >= 1


def test_phase_d_num_lines_clamped_to_maximum():
    """num_parallel_lines=100 should be clamped to 20 (not produce 100 lines)."""
    canvas = _a4_canvas()
    gen = GENERATORS["Calligraphy"]()
    params = {
        "path_source": "Circle",
        "width_mode": "Constant",
        "stroke_width_mm": 4.0,
        "num_parallel_lines": 100,  # above maximum
    }
    result = gen.generate(params, canvas)
    # Should produce exactly 20 lines, not 100
    assert isinstance(result, list)
    assert len(result) == 20


# ---------------------------------------------------------------------------
# Phase D: verify all 6 spec presets are present by name in get_presets()
# ---------------------------------------------------------------------------

def test_phase_d_all_spec_presets_exist():
    """get_presets() must contain all 6 presets specified in task 17.4."""
    gen = GENERATORS["Calligraphy"]()
    preset_names = {p.name for p in gen.get_presets()}
    required = {
        "Broad Nib Italic",
        "Monoline Script",
        "Gothic Blackletter",
        "Brush Pen Spiral",
        "Thin Copperplate",
        "Decorative Wave",
    }
    for name in required:
        assert name in preset_names, (
            f"Required preset '{name}' not found in get_presets(). "
            f"Available: {preset_names}"
        )


# ---------------------------------------------------------------------------
# Phase D: all Phase D presets only reference valid parameter names
# ---------------------------------------------------------------------------

def test_phase_d_preset_params_are_valid():
    """All Phase D presets must only reference params defined in get_parameters()."""
    gen = GENERATORS["Calligraphy"]()
    valid_names = {p.name for p in gen.get_parameters()}
    phase_d_names = {
        "Broad Nib Italic", "Monoline Script", "Gothic Blackletter",
        "Thin Copperplate", "Decorative Wave",
    }
    presets = {p.name: p for p in gen.get_presets()}
    for name in phase_d_names:
        assert name in presets, f"Preset '{name}' not found"
        for key in presets[name].params:
            assert key in valid_names, (
                f"Preset '{name}' references unknown param '{key}'"
            )
