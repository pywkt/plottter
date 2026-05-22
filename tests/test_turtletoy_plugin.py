"""Tests for the TurtleToy plugin (phase 126.1).

Covers:
- Star sketch (spec §8.1) produces exactly 5 segments grouped into 1 polyline
- All output coords lie within the canvas drawing area bounds
- quickjs-missing path raises a RuntimeError with clear install instructions
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from plottter.models.canvas import Canvas

# ---------------------------------------------------------------------------
# Load the plugin module directly (it lives in plugins/, not src/).
# We use a fixed module name so sys.modules deduplicates across test runs.
# ---------------------------------------------------------------------------

_PLUGIN_PATH = Path(__file__).parent.parent / "plugins" / "turtletoy.py"
_MODULE_NAME = "plottter_plugin_turtletoy"


def _load_plugin() -> Any:
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _PLUGIN_PATH)
    assert spec is not None and spec.loader is not None, (
        f"Could not create module spec for {_PLUGIN_PATH}"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_tt = _load_plugin()
TurtleToyGenerator = _tt.TurtleToyGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canvas() -> Canvas:
    """Standard A4 canvas used across all tests."""
    return Canvas.from_preset("A4", margin=10.0)


@pytest.fixture
def generator() -> TurtleToyGenerator:
    return TurtleToyGenerator()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _star_polylines(canvas: Canvas) -> list[list[tuple[float, float]]]:
    """Run the star sketch and return the resulting polylines."""
    gen = TurtleToyGenerator()
    return gen.generate({"code": _tt._STAR_SKETCH}, canvas)


# ---------------------------------------------------------------------------
# Tests: star sketch geometry
# ---------------------------------------------------------------------------


class TestStarSketch:
    """The canonical 5-pointed star (spec §8.1) must produce well-formed output."""

    def test_produces_exactly_5_segments(self, canvas: Canvas) -> None:
        """Five forward() calls → 5 line segments in total."""
        polylines = _star_polylines(canvas)
        total_segments = sum(len(pl) - 1 for pl in polylines)
        assert total_segments == 5, (
            f"Expected 5 segments, got {total_segments} "
            f"(polyline point counts: {[len(pl) for pl in polylines]})"
        )

    def test_produces_exactly_1_polyline(self, canvas: Canvas) -> None:
        """All 5 star segments are consecutive → must merge into 1 polyline."""
        polylines = _star_polylines(canvas)
        assert len(polylines) == 1, (
            f"Expected 1 polyline after grouping, got {len(polylines)}"
        )

    def test_polyline_has_6_points(self, canvas: Canvas) -> None:
        """5 segments = 6 points (start + 4 intermediate + end ≈ start)."""
        polylines = _star_polylines(canvas)
        assert len(polylines[0]) == 6, (
            f"Expected 6 points in polyline, got {len(polylines[0])}"
        )

    def test_all_coords_within_canvas_bounds(self, canvas: Canvas) -> None:
        """Every (x, y) mm coordinate must lie within the printable area."""
        polylines = _star_polylines(canvas)
        left, top, right, bottom = canvas.drawing_area()

        for polyline in polylines:
            for x_mm, y_mm in polyline:
                assert left <= x_mm <= right, (
                    f"x={x_mm:.3f}mm outside printable range [{left}, {right}]"
                )
                assert top <= y_mm <= bottom, (
                    f"y={y_mm:.3f}mm outside printable range [{top}, {bottom}]"
                )

    def test_star_is_closed(self, canvas: Canvas) -> None:
        """The star's last point should approximately equal its first point."""
        polylines = _star_polylines(canvas)
        pts = polylines[0]
        x0, y0 = pts[0]
        xn, yn = pts[-1]
        assert abs(x0 - xn) < 0.1, f"Star not closed in x: {x0:.4f} vs {xn:.4f}"
        assert abs(y0 - yn) < 0.1, f"Star not closed in y: {y0:.4f} vs {yn:.4f}"


# ---------------------------------------------------------------------------
# Tests: quickjs-missing failure mode (spec §7.2)
# ---------------------------------------------------------------------------


class TestQuickjsMissing:
    """When quickjs is not installed, generate() must raise a clear RuntimeError."""

    def test_generate_raises_runtime_error(self, canvas: Canvas) -> None:
        """generate() raises RuntimeError when _QUICKJS_AVAILABLE is False."""
        gen = TurtleToyGenerator()
        with patch.object(_tt, "_QUICKJS_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc_info:
                gen.generate({"code": _tt._STAR_SKETCH}, canvas)

        msg = str(exc_info.value)
        assert "quickjs" in msg.lower(), (
            f"Error message should mention 'quickjs', got: {msg!r}"
        )

    def test_error_includes_install_command(self, canvas: Canvas) -> None:
        """The RuntimeError message includes 'pip install quickjs'."""
        gen = TurtleToyGenerator()
        with patch.object(_tt, "_QUICKJS_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc_info:
                gen.generate({"code": _tt._STAR_SKETCH}, canvas)

        assert "pip install" in str(exc_info.value), (
            "Error message should include 'pip install' instructions"
        )

    def test_presets_mention_missing_package(self) -> None:
        """get_presets() returns an entry describing the missing dependency."""
        gen = TurtleToyGenerator()
        with patch.object(_tt, "_QUICKJS_AVAILABLE", False):
            presets = gen.get_presets()

        assert len(presets) == 1
        description_or_name = presets[0].description + presets[0].name
        assert "quickjs" in description_or_name.lower(), (
            f"Preset should mention 'quickjs', got name={presets[0].name!r}, "
            f"description={presets[0].description!r}"
        )


# ---------------------------------------------------------------------------
# Tests: method aliases (spec §4.1 – §4.3)
# ---------------------------------------------------------------------------


def _run_sketch(code: str, canvas: Canvas) -> list[list[tuple[float, float]]]:
    """Helper: run an arbitrary JS sketch and return the resulting polylines."""
    gen = TurtleToyGenerator()
    return gen.generate({"code": code}, canvas)


def _seg_count(polylines: list[list[tuple[float, float]]]) -> int:
    """Total number of segments across all polylines."""
    return sum(len(pl) - 1 for pl in polylines)


class TestAliases:
    """Every Turtle method alias must run without error and produce the expected geometry."""

    # --- §4.1 Movement aliases ---

    def test_fd_alias(self, canvas: Canvas) -> None:
        """fd(d) is an alias for forward(d) — produces 1 segment."""
        polylines = _run_sketch("const t = new Turtle(); t.fd(50);", canvas)
        assert _seg_count(polylines) == 1

    def test_backward_produces_segment(self, canvas: Canvas) -> None:
        """backward(d) moves in the opposite direction and produces a segment when pen is down."""
        polylines = _run_sketch("const t = new Turtle(); t.backward(50);", canvas)
        assert _seg_count(polylines) == 1

    def test_bk_alias(self, canvas: Canvas) -> None:
        """bk(d) is an alias for backward(d) — produces 1 segment."""
        polylines = _run_sketch("const t = new Turtle(); t.bk(50);", canvas)
        assert _seg_count(polylines) == 1

    def test_back_alias(self, canvas: Canvas) -> None:
        """back(d) is an alias for backward(d) — produces 1 segment."""
        polylines = _run_sketch("const t = new Turtle(); t.back(50);", canvas)
        assert _seg_count(polylines) == 1

    # --- §4.2 Rotation aliases ---

    def test_rt_alias(self, canvas: Canvas) -> None:
        """rt(deg) is an alias for right(deg) — direction changes, forward still draws."""
        polylines = _run_sketch("const t = new Turtle(); t.rt(90); t.forward(50);", canvas)
        assert _seg_count(polylines) == 1

    def test_lt_alias(self, canvas: Canvas) -> None:
        """lt(deg) is an alias for left(deg) — direction changes, forward still draws."""
        polylines = _run_sketch("const t = new Turtle(); t.lt(90); t.forward(50);", canvas)
        assert _seg_count(polylines) == 1

    # --- §4.3 Pen control aliases ---

    def test_pu_alias(self, canvas: Canvas) -> None:
        """pu() lifts the pen — forward does not draw."""
        polylines = _run_sketch("const t = new Turtle(); t.pu(); t.forward(50);", canvas)
        assert _seg_count(polylines) == 0

    def test_up_alias(self, canvas: Canvas) -> None:
        """up() is an alias for penup() — forward does not draw."""
        polylines = _run_sketch("const t = new Turtle(); t.up(); t.forward(50);", canvas)
        assert _seg_count(polylines) == 0

    def test_pd_alias(self, canvas: Canvas) -> None:
        """pd() re-lowers the pen — forward draws after penup/pd."""
        polylines = _run_sketch(
            "const t = new Turtle(); t.penup(); t.pd(); t.forward(50);", canvas
        )
        assert _seg_count(polylines) == 1

    def test_down_alias(self, canvas: Canvas) -> None:
        """down() is an alias for pendown() — forward draws after penup/down."""
        polylines = _run_sketch(
            "const t = new Turtle(); t.penup(); t.down(); t.forward(50);", canvas
        )
        assert _seg_count(polylines) == 1

    # --- §4.1 Absolute movement aliases ---

    def test_setpos_alias(self, canvas: Canvas) -> None:
        """setpos(x, y) is an alias for goto(x, y) — draws when pen is down."""
        polylines = _run_sketch("const t = new Turtle(); t.setpos(10, 10);", canvas)
        assert _seg_count(polylines) == 1

    def test_setposition_alias(self, canvas: Canvas) -> None:
        """setposition(x, y) is an alias for goto(x, y) — draws when pen is down."""
        polylines = _run_sketch("const t = new Turtle(); t.setposition(10, 10);", canvas)
        assert _seg_count(polylines) == 1

    def test_goto_with_array_arg(self, canvas: Canvas) -> None:
        """goto([x, y]) (array form, spec §4.1) draws a segment when pen is down."""
        polylines = _run_sketch("const t = new Turtle(); t.goto([10, 10]);", canvas)
        assert _seg_count(polylines) == 1

    # --- §4.1 Jump aliases ---

    def test_jmp_alias(self, canvas: Canvas) -> None:
        """jmp(x, y) is an alias for jump(x, y) — teleports without drawing."""
        polylines = _run_sketch(
            "const t = new Turtle(); t.jmp(10, 10); t.forward(20);", canvas
        )
        assert _seg_count(polylines) == 1

    def test_jump_with_array_arg(self, canvas: Canvas) -> None:
        """jump([x, y]) (array form, spec §4.1) teleports without drawing."""
        polylines = _run_sketch(
            "const t = new Turtle(); t.jump([10, 10]); t.forward(20);", canvas
        )
        assert _seg_count(polylines) == 1

    # --- Combined: backward + jump sketch ---

    def test_backward_and_jump_sketch(self, canvas: Canvas) -> None:
        """A sketch using backward() and jump() produces the expected segment count."""
        sketch = (
            "const t = new Turtle();\n"
            "t.backward(30);\n"
            "t.jump(0, 20);\n"
            "t.bk(30);\n"
            "t.jmp([0, 40]);\n"
            "t.back(30);\n"
        )
        polylines = _run_sketch(sketch, canvas)
        assert _seg_count(polylines) == 3


class TestHeadingControl:
    """Spec §4.1 & §4.2: setheading/seth, setx, sety, home — segment-coord assertions."""

    def test_setx_draws_horizontal_segment(self, canvas: Canvas) -> None:
        """setx(20) moves from (0,0) to (20,0) and emits one segment when pen is down."""
        import math

        polylines = _run_sketch("const t = new Turtle(); t.setx(20);", canvas)
        assert _seg_count(polylines) == 1
        # Flatten all points; the segment should be (0,0)->(20,0) in world space.
        # After scaling to canvas the x-coordinates are proportional.
        pts = [pt for pl in polylines for pt in pl]
        assert len(pts) == 2
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        # y stays 0; x moves by 20 units in world space (scaled to canvas mm)
        assert y0 == pytest.approx(y1)
        assert x1 > x0  # moved in positive-x direction

    def test_sety_draws_vertical_segment(self, canvas: Canvas) -> None:
        """sety(30) (after setx(20)) moves from (20,0) to (20,30) drawing a vertical line."""
        polylines = _run_sketch(
            "const t = new Turtle(); t.setx(20); t.sety(30);", canvas
        )
        assert _seg_count(polylines) == 2

    def test_setheading_sets_direction(self, canvas: Canvas) -> None:
        """setheading(90) points north; forward(10) should produce a vertical segment."""
        import math

        polylines = _run_sketch(
            "const t = new Turtle(); t.penup(); t.setheading(90); t.pendown(); t.forward(10);",
            canvas,
        )
        assert _seg_count(polylines) == 1
        pts = [pt for pl in polylines for pt in pl]
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        # heading 90 = north: x stays same; canvas Y-down means canvas y1 < y0
        assert x0 == pytest.approx(x1, abs=1e-9)
        assert y1 < y0

    def test_seth_alias(self, canvas: Canvas) -> None:
        """seth(deg) is an alias for setheading(deg)."""
        polylines = _run_sketch(
            "const t = new Turtle(); t.seth(0); t.forward(10);", canvas
        )
        assert _seg_count(polylines) == 1

    def test_home_draws_segment_and_resets(self, canvas: Canvas) -> None:
        """home() draws back to (0,0) when pen is down, then heading is reset to 0."""
        # After home() heading=0 (east); forward(10) should be horizontal.
        import math

        polylines = _run_sketch(
            "const t = new Turtle(); t.goto(50, 50); t.home(); t.forward(10);",
            canvas,
        )
        # goto(50,50): 1 seg; home(): 1 seg; forward(10): 1 seg
        assert _seg_count(polylines) == 3

    def test_home_no_draw_when_pen_up(self, canvas: Canvas) -> None:
        """home() does not draw if pen is up, but still moves to (0,0)."""
        polylines = _run_sketch(
            "const t = new Turtle(); t.goto(50, 0); t.penup(); t.home(); t.pendown(); t.forward(10);",
            canvas,
        )
        # goto: 1 seg; home (pen up): 0 segs; forward: 1 seg
        assert _seg_count(polylines) == 2

    def test_full_heading_control_sketch_segment_coords(self, canvas: Canvas) -> None:
        """Run the canonical heading-control sketch and assert explicit segment coordinates.

        Sketch: t.setx(20); t.sety(30); t.setheading(45); t.forward(10); t.home();
        Expected world-space segments (before canvas scaling):
          seg0: (0, 0) -> (20, 0)       [setx]
          seg1: (20, 0) -> (20, 30)     [sety]
          seg2: (20, 30) -> (20+10*cos45, 30+10*sin45)  [forward at 45°]
          seg3: (20+10*cos45, 30+10*sin45) -> (0, 0)    [home]
        """
        import math

        sketch = (
            "const t = new Turtle();"
            " t.setx(20);"
            " t.sety(30);"
            " t.setheading(45);"
            " t.forward(10);"
            " t.home();"
        )
        polylines = _run_sketch(sketch, canvas)
        assert _seg_count(polylines) == 4

        # Collect all segments in order from the flattened polylines.
        segments: list[tuple[float, float, float, float]] = []
        for pl in polylines:
            for i in range(len(pl) - 1):
                x0, y0 = pl[i]
                x1, y1 = pl[i + 1]
                segments.append((x0, y0, x1, y1))

        # Replicate the letterbox mapping from _segments_to_polylines.
        left, top, right, bottom = canvas.drawing_area()
        width = right - left
        height = bottom - top
        cx = left + width / 2.0
        cy = top + height / 2.0
        scale = min(width / 200.0, height / 200.0)

        def w2c(wx: float, wy: float) -> tuple[float, float]:
            """World → canvas mm (Y-up world → Y-down canvas)."""
            return cx + wx * scale, cy - wy * scale

        # Expected world-space endpoints
        fw = 10.0
        d45 = math.sqrt(2) / 2
        wx = [0.0, 20.0, 20.0, 20.0 + fw * d45, 0.0]
        wy = [0.0, 0.0, 30.0, 30.0 + fw * d45, 0.0]

        for i, seg in enumerate(segments):
            x0_exp, y0_exp = w2c(wx[i], wy[i])
            x1_exp, y1_exp = w2c(wx[i + 1], wy[i + 1])
            assert seg[0] == pytest.approx(x0_exp, abs=1e-9), f"seg{i} x0 mismatch"
            assert seg[1] == pytest.approx(y0_exp, abs=1e-9), f"seg{i} y0 mismatch"
            assert seg[2] == pytest.approx(x1_exp, abs=1e-9), f"seg{i} x1 mismatch"
            assert seg[3] == pytest.approx(y1_exp, abs=1e-9), f"seg{i} y1 mismatch"


# ---------------------------------------------------------------------------
# Tests: state query methods (spec §4.5 + §8.4)
# ---------------------------------------------------------------------------

_QUERY_SKETCH_84 = """\
const t = new Turtle();
t.forward(50);
t.left(90);
t.forward(30);

// Set into a global the test can read
result = [t.x(), t.y(), t.heading(), t.distance(0, 0)];
"""


class TestStateQueries:
    """State-query methods return correct values from JS (spec §4.5 and §8.4)."""

    # ------------------------------------------------------------------
    # §8.4 sketch — runs the canonical query sketch and reads result[] back
    # from the JS context via runtime.ctx.eval("result").
    # ------------------------------------------------------------------

    def test_query_sketch_returns_four_values(self) -> None:
        """The §8.4 sketch populates a JS global `result` with 4 elements."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval(_QUERY_SKETCH_84)
        length = runtime.ctx.eval("result.length")
        assert length == 4, f"Expected result.length == 4, got {length}"

    def test_query_sketch_xcor(self) -> None:
        """result[0] = x() ≈ 50 after forward(50) then left(90) then forward(30)."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval(_QUERY_SKETCH_84)
        x = runtime.ctx.eval("result[0]")
        assert abs(x - 50.0) < 1e-6, f"x() expected 50, got {x}"

    def test_query_sketch_ycor(self) -> None:
        """result[1] = y() ≈ 30 after left(90) + forward(30) (Y-up)."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval(_QUERY_SKETCH_84)
        y = runtime.ctx.eval("result[1]")
        assert abs(y - 30.0) < 1e-6, f"y() expected 30, got {y}"

    def test_query_sketch_heading(self) -> None:
        """result[2] = heading() ≈ 90 after left(90) from heading=0 (east)."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval(_QUERY_SKETCH_84)
        h = runtime.ctx.eval("result[2]")
        assert abs(h - 90.0) < 1e-6, f"heading() expected 90, got {h}"

    def test_query_sketch_distance(self) -> None:
        """result[3] = distance(0,0) ≈ sqrt(50²+30²) ≈ 58.309..."""
        import math

        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval(_QUERY_SKETCH_84)
        d = runtime.ctx.eval("result[3]")
        expected = math.hypot(50, 30)
        assert abs(d - expected) < 1e-6, (
            f"distance(0,0) expected {expected:.6f}, got {d}"
        )

    # ------------------------------------------------------------------
    # Per-method unit tests
    # ------------------------------------------------------------------

    def test_position_returns_array(self, canvas: Canvas) -> None:
        """position() returns a 2-element JS array with current x and y."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.forward(10);")
        px = runtime.ctx.eval("t.position()[0]")
        py = runtime.ctx.eval("t.position()[1]")
        assert abs(px - 10.0) < 1e-6, f"position()[0] expected 10, got {px}"
        assert abs(py - 0.0) < 1e-6, f"position()[1] expected 0, got {py}"

    def test_pos_alias(self) -> None:
        """pos() is an alias for position() — returns same array."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.forward(20);")
        px = runtime.ctx.eval("t.pos()[0]")
        assert abs(px - 20.0) < 1e-6, f"pos()[0] expected 20, got {px}"

    def test_xcor_and_x(self) -> None:
        """xcor() and x() both return the current x coordinate."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.forward(15);")
        assert abs(runtime.ctx.eval("t.xcor()") - 15.0) < 1e-6
        assert abs(runtime.ctx.eval("t.x()") - 15.0) < 1e-6

    def test_ycor_and_y(self) -> None:
        """ycor() and y() both return the current y coordinate."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.setheading(90); t.forward(25);")
        assert abs(runtime.ctx.eval("t.ycor()") - 25.0) < 1e-6
        assert abs(runtime.ctx.eval("t.y()") - 25.0) < 1e-6

    def test_heading_and_h(self) -> None:
        """heading() and h() return degrees in [0, 360)."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.left(45);")
        assert abs(runtime.ctx.eval("t.heading()") - 45.0) < 1e-6
        assert abs(runtime.ctx.eval("t.h()") - 45.0) < 1e-6

    def test_heading_wraps_to_zero_to_360(self) -> None:
        """Heading after right(45) from 0 is 315 (not -45)."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.right(45);")
        h = runtime.ctx.eval("t.heading()")
        assert abs(h - 315.0) < 1e-6, f"Expected 315, got {h}"

    def test_isdown_pen_down(self) -> None:
        """isdown() returns true when pen is down (default)."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle();")
        assert runtime.ctx.eval("t.isdown()") is True

    def test_isdown_pen_up(self) -> None:
        """isdown() returns false after penup()."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.penup();")
        assert runtime.ctx.eval("t.isdown()") is False

    def test_distance_scalar_args(self) -> None:
        """distance(x, y) returns Euclidean distance from current position."""
        import math

        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.forward(3); t.setheading(90); t.forward(4);")
        d = runtime.ctx.eval("t.distance(0, 0)")
        assert abs(d - 5.0) < 1e-6, f"Expected 5.0, got {d}"

    def test_distance_array_arg(self) -> None:
        """distance([x, y]) (array form) also works."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.forward(3); t.setheading(90); t.forward(4);")
        d = runtime.ctx.eval("t.distance([0, 0])")
        assert abs(d - 5.0) < 1e-6, f"Expected 5.0, got {d}"

    def test_towards_east(self) -> None:
        """towards(100, 0) from origin = 0 degrees (east)."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle();")
        ang = runtime.ctx.eval("t.towards(100, 0)")
        assert abs(ang - 0.0) < 1e-6, f"Expected 0, got {ang}"

    def test_towards_north(self) -> None:
        """towards(0, 100) from origin = 90 degrees (north)."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle();")
        ang = runtime.ctx.eval("t.towards(0, 100)")
        assert abs(ang - 90.0) < 1e-6, f"Expected 90, got {ang}"

    def test_towards_array_arg(self) -> None:
        """towards([x, y]) (array form) also works."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle();")
        ang = runtime.ctx.eval("t.towards([0, 100])")
        assert abs(ang - 90.0) < 1e-6, f"Expected 90, got {ang}"


# ---------------------------------------------------------------------------
# Tests: circle() — spec §4.4 and §8.2
# ---------------------------------------------------------------------------

_SPIRAL_SKETCH_82 = """\
const turtle = new Turtle();
function walk(i) {
    turtle.circle(i * 0.5, 30);
    return i < 200;
}
"""


class TestCircle:
    """circle(radius, extent, steps) — spec §4.4, test sketches §8.2."""

    def test_full_circle_segment_count(self, canvas: Canvas) -> None:
        """t.circle(50) → extent=360, steps=max(8,ceil(360/5))=72 segments."""
        polylines = _run_sketch("const t = new Turtle(); t.circle(50);", canvas)
        total = _seg_count(polylines)
        assert total == 72, (
            f"circle(50) expected 72 segments (steps=max(8,72)), got {total}"
        )

    def test_full_circle_is_closed(self, canvas: Canvas) -> None:
        """A full circle should approximately return to its starting point."""
        polylines = _run_sketch("const t = new Turtle(); t.circle(50);", canvas)
        assert len(polylines) >= 1
        pts = [pt for pl in polylines for pt in pl]
        x0, y0 = pts[0]
        xn, yn = pts[-1]
        assert abs(x0 - xn) < 0.5, f"Full circle not closed in x: {x0:.4f} vs {xn:.4f}"
        assert abs(y0 - yn) < 0.5, f"Full circle not closed in y: {y0:.4f} vs {yn:.4f}"

    def test_partial_arc_segment_count(self, canvas: Canvas) -> None:
        """circle(50, 90) → steps=max(8,ceil(90/5))=18 segments."""
        polylines = _run_sketch("const t = new Turtle(); t.circle(50, 90);", canvas)
        total = _seg_count(polylines)
        assert total == 18, (
            f"circle(50, 90) expected 18 segments, got {total}"
        )

    def test_explicit_steps(self, canvas: Canvas) -> None:
        """circle(50, 360, 36) → exactly 36 segments."""
        polylines = _run_sketch("const t = new Turtle(); t.circle(50, 360, 36);", canvas)
        total = _seg_count(polylines)
        assert total == 36, f"circle(50, 360, 36) expected 36 segments, got {total}"

    def test_penup_suppresses_segments(self, canvas: Canvas) -> None:
        """circle() with pen up produces no segments."""
        polylines = _run_sketch(
            "const t = new Turtle(); t.penup(); t.circle(50);", canvas
        )
        assert _seg_count(polylines) == 0

    def test_small_steps_default_minimum(self, canvas: Canvas) -> None:
        """Small extent: steps default = max(8, ceil(10/5)) = max(8,2) = 8."""
        polylines = _run_sketch("const t = new Turtle(); t.circle(50, 10);", canvas)
        total = _seg_count(polylines)
        assert total == 8, (
            f"circle(50, 10) expected 8 segments (default min), got {total}"
        )

    def test_negative_radius_cw(self, canvas: Canvas) -> None:
        """Negative radius traces CW arc — segment count same as positive radius."""
        poly_pos = _run_sketch("const t = new Turtle(); t.circle(50, 90);", canvas)
        poly_neg = _run_sketch("const t = new Turtle(); t.circle(-50, 90);", canvas)
        assert _seg_count(poly_neg) == _seg_count(poly_pos), (
            "Negative radius should produce same segment count as positive"
        )

    def test_spiral_sketch_segment_count(self, canvas: Canvas) -> None:
        """Spec §8.2 spiral: segment count must exceed 1000."""
        polylines = _run_sketch(_SPIRAL_SKETCH_82, canvas)
        total = _seg_count(polylines)
        assert total > 1000, (
            f"Spiral sketch (§8.2) expected >1000 segments, got {total}"
        )

    def test_spiral_sketch_has_polylines(self, canvas: Canvas) -> None:
        """Spiral sketch produces at least one polyline."""
        polylines = _run_sketch(_SPIRAL_SKETCH_82, canvas)
        assert len(polylines) >= 1

    def test_circle_runtime_direct(self) -> None:
        """TurtleRuntime: circle(50) via JS produces 72 segments in runtime.segments."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.circle(50);")
        assert len(runtime.segments) == 72, (
            f"Expected 72 segments in runtime.segments, got {len(runtime.segments)}"
        )

    def test_circle_radius_zero_no_segments(self, canvas: Canvas) -> None:
        """circle(0) should produce no segments (zero-radius arc)."""
        polylines = _run_sketch("const t = new Turtle(); t.circle(0);", canvas)
        assert _seg_count(polylines) == 0


# ---------------------------------------------------------------------------
# Tests: clone() — spec §4.7 and §8.3
# ---------------------------------------------------------------------------

_TREE_SKETCH_83 = """\
const turtle = new Turtle();
turtle.penup();
turtle.goto(0, -80);
turtle.setheading(90);
turtle.pendown();

function branch(t, length, depth) {
    if (depth === 0) return;
    t.forward(length);
    const left = t.clone();
    left.left(30);
    branch(left, length * 0.7, depth - 1);
    t.right(30);
    branch(t, length * 0.7, depth - 1);
}

branch(turtle, 30, 6);
"""


class TestClone:
    """clone() creates an independent turtle copying all state (spec §4.7, §8.3)."""

    def test_tree_sketch_segment_count(self, canvas: Canvas) -> None:
        """Spec §8.3 tree sketch (no walk function): must capture ≥ 60 segments."""
        polylines = _run_sketch(_TREE_SKETCH_83, canvas)
        total = _seg_count(polylines)
        assert total >= 60, (
            f"Tree sketch (§8.3) expected ≥60 segments, got {total}"
        )

    def test_clone_is_independent(self) -> None:
        """Moving the clone does not affect the original turtle's position."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval(
            "const t = new Turtle(); t.forward(10);"
            "const c = t.clone(); c.forward(50);"
        )
        # original turtle x should still be 10
        x_orig = runtime.ctx.eval("t.x()")
        x_clone = runtime.ctx.eval("c.x()")
        assert abs(x_orig - 10.0) < 1e-6, f"Original x expected 10, got {x_orig}"
        assert abs(x_clone - 60.0) < 1e-6, f"Clone x expected 60, got {x_clone}"

    def test_clone_inherits_position(self) -> None:
        """Clone starts at the same position as the source."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.goto(25, 40);")
        runtime.ctx.eval("const c = t.clone();")
        cx = runtime.ctx.eval("c.x()")
        cy = runtime.ctx.eval("c.y()")
        assert abs(cx - 25.0) < 1e-6, f"Clone x expected 25, got {cx}"
        assert abs(cy - 40.0) < 1e-6, f"Clone y expected 40, got {cy}"

    def test_clone_inherits_heading(self) -> None:
        """Clone has the same heading as the source at the time of cloning."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.setheading(45);")
        runtime.ctx.eval("const c = t.clone();")
        h = runtime.ctx.eval("c.heading()")
        assert abs(h - 45.0) < 1e-6, f"Clone heading expected 45, got {h}"

    def test_clone_inherits_pen_state(self) -> None:
        """Clone inherits pen-up/pen-down state from source."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.penup();")
        runtime.ctx.eval("const c = t.clone();")
        assert runtime.ctx.eval("c.isdown()") is False

    def test_clone_segments_go_to_global_list(self) -> None:
        """Segments drawn by the clone are captured in the shared segment list."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval(
            "const t = new Turtle(); t.penup();"
            "const c = t.clone(); c.pendown(); c.forward(30);"
        )
        assert len(runtime.segments) == 1, (
            f"Expected 1 segment from clone, got {len(runtime.segments)}"
        )


# ---------------------------------------------------------------------------
# Tests: angular units — spec §4.6
# ---------------------------------------------------------------------------


class TestAngularUnits:
    """degrees(), radians(), fullCircle() adjust the angular unit (spec §4.6)."""

    def test_full_circle_default_is_360(self) -> None:
        """fullCircle() returns 360 by default (degrees mode)."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle();")
        fc = runtime.ctx.eval("t.fullCircle()")
        assert abs(fc - 360.0) < 1e-9, f"Default fullCircle expected 360, got {fc}"

    def test_degrees_sets_full_circle(self) -> None:
        """degrees(100) sets fullCircle() to 100."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.degrees(100);")
        fc = runtime.ctx.eval("t.fullCircle()")
        assert abs(fc - 100.0) < 1e-9, f"fullCircle expected 100 after degrees(100), got {fc}"

    def test_radians_sets_full_circle_to_2pi(self) -> None:
        """radians() sets fullCircle() to 2π."""
        import math

        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.radians();")
        fc = runtime.ctx.eval("t.fullCircle()")
        assert abs(fc - 2 * math.pi) < 1e-9, (
            f"fullCircle expected 2π after radians(), got {fc}"
        )

    def test_degrees_default_arg_is_360(self) -> None:
        """degrees() with no argument defaults to 360."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.radians(); t.degrees();")
        fc = runtime.ctx.eval("t.fullCircle()")
        assert abs(fc - 360.0) < 1e-9, (
            f"fullCircle expected 360 after degrees(), got {fc}"
        )

    def test_right_uses_user_units(self) -> None:
        """degrees(100): right(25) = quarter-turn CW; heading() = 315/360*100 = 75 user units."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.degrees(100); t.right(25);")
        h = runtime.ctx.eval("t.heading()")
        # right(25) in user units [100/circle] = 90 deg CW from east (0)
        # heading in user units: 270 deg → 270/360*100 = 75
        assert abs(h - 75.0) < 1e-6, (
            f"After degrees(100) + right(25), heading expected 75, got {h}"
        )

    def test_left_uses_user_units(self) -> None:
        """degrees(100): left(25) = quarter-turn CCW; heading() = 25 user units."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.degrees(100); t.left(25);")
        h = runtime.ctx.eval("t.heading()")
        # left(25) from 0 = +90 deg CCW = heading 90 → 90/360*100 = 25
        assert abs(h - 25.0) < 1e-6, (
            f"After degrees(100) + left(25), heading expected 25, got {h}"
        )

    def test_radians_mode_half_turn(self) -> None:
        """radians() mode: left(π) = half turn; heading ≈ π (180 degrees)."""
        import math

        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.radians(); t.left(Math.PI);")
        h = runtime.ctx.eval("t.heading()")
        assert abs(h - math.pi) < 1e-9, (
            f"After radians() + left(π), heading expected π, got {h}"
        )

    def test_circle_uses_user_units_for_extent(self, canvas: Canvas) -> None:
        """degrees(100): circle(50) uses fullCircle()=100 as default extent → 72 segs."""
        # fullCircle=100, steps = max(8, ceil(|100|*360/100/5)) = max(8, ceil(72)) = 72
        polylines = _run_sketch(
            "const t = new Turtle(); t.degrees(100); t.circle(50);", canvas
        )
        total = _seg_count(polylines)
        assert total == 72, (
            f"circle(50) with degrees(100) expected 72 segments, got {total}"
        )

    def test_clone_inherits_angular_units(self) -> None:
        """A clone inherits the source turtle's angular unit setting."""
        runtime = _tt.TurtleRuntime(seed=0)
        runtime.ctx.eval("const t = new Turtle(); t.degrees(100); const c = t.clone();")
        fc = runtime.ctx.eval("c.fullCircle()")
        assert abs(fc - 100.0) < 1e-9, (
            f"Clone fullCircle expected 100 (inherited from source), got {fc}"
        )


# ---------------------------------------------------------------------------
# Tests: walk loop — max_steps cap and error handling (spec §3, task 129.1)
# ---------------------------------------------------------------------------

_INFINITE_WALK_SKETCH = """\
const t = new Turtle();
function walk(i) {
    t.forward(0.1);
    return true;
}
"""

_ERROR_AT_STEP_5_SKETCH = """\
const t = new Turtle();
function walk(i) {
    t.forward(1);
    if (i >= 5) { throw new Error("injected error at step " + i); }
    return true;
}
"""


class TestWalkLoop:
    """Walk loop: max_steps cap, timeout, and error-injection (spec §3)."""

    def test_infinite_walk_terminates_by_max_steps(self, canvas: Canvas) -> None:
        """An infinite walk (always returns true) must stop at max_steps."""
        gen = TurtleToyGenerator()
        polylines = gen.generate(
            {"code": _INFINITE_WALK_SKETCH, "max_steps": 50},
            canvas,
        )
        # Each walk(i) call does forward(0.1) → 1 segment per step.
        # With max_steps=50 the loop must stop and return exactly 50 segments.
        total = sum(len(pl) - 1 for pl in polylines)
        assert total == 50, (
            f"Infinite walk with max_steps=50 expected 50 segments, got {total}"
        )

    def test_infinite_walk_returns_partial_output(self, canvas: Canvas) -> None:
        """Partial output from an infinite walk is non-empty (not discarded)."""
        gen = TurtleToyGenerator()
        polylines = gen.generate(
            {"code": _INFINITE_WALK_SKETCH, "max_steps": 10},
            canvas,
        )
        assert len(polylines) >= 1, "Partial output should contain at least one polyline"
        total = sum(len(pl) - 1 for pl in polylines)
        assert total > 0, "Partial output must not be empty"

    def test_error_in_walk_preserves_partial_output(self, canvas: Canvas) -> None:
        """walk() raising an error stops the loop but keeps already-captured segments."""
        gen = TurtleToyGenerator()
        polylines = gen.generate(
            {"code": _ERROR_AT_STEP_5_SKETCH, "max_steps": 100_000},
            canvas,
        )
        # Steps 0–5 each do forward(1) before throwing at step 5.
        # Segments for steps 0–4 (5 segments) are captured before the error.
        # Step 5 draws forward(1) first THEN throws, so we get 6 segments total.
        total = sum(len(pl) - 1 for pl in polylines)
        assert total >= 5, (
            f"Error at step 5 should preserve ≥5 segments, got {total}"
        )

    def test_max_steps_param_is_respected(self, canvas: Canvas) -> None:
        """generate() honours the max_steps param passed in params dict."""
        gen = TurtleToyGenerator()
        for cap in (1, 5, 20):
            polylines = gen.generate(
                {"code": _INFINITE_WALK_SKETCH, "max_steps": cap},
                canvas,
            )
            total = sum(len(pl) - 1 for pl in polylines)
            assert total == cap, (
                f"max_steps={cap} expected {cap} segments, got {total}"
            )

    def test_no_walk_function_returns_top_level_output(self, canvas: Canvas) -> None:
        """Sketch with no walk function: top-level segments are returned immediately."""
        # The tree sketch (§8.3) has no walk function — segments come from top-level code.
        gen = TurtleToyGenerator()
        polylines = gen.generate(
            {"code": _TREE_SKETCH_83, "max_steps": 1},
            canvas,
        )
        total = sum(len(pl) - 1 for pl in polylines)
        # Tree sketch produces ≥60 segments (verified in TestClone) even with max_steps=1
        # because the walk loop never runs when typeof walk !== 'function'.
        assert total >= 60, (
            f"No-walk sketch with max_steps=1 should still return ≥60 segments, got {total}"
        )


# ---------------------------------------------------------------------------
# Sketch constants for seeding + console tests
# ---------------------------------------------------------------------------

# Sketch that draws 10 forward steps using Math.random for distance.
_RANDOM_SKETCH = """\
const t = new Turtle();
for (let i = 0; i < 10; i++) {
    t.forward(Math.random() * 100);
}
"""

# Sketch that calls console.log — should not crash.
_CONSOLE_LOG_SKETCH = """\
console.log("hello from turtletoy");
console.log(42, true, null);
const t = new Turtle();
t.forward(50);
"""


class TestSeedDeterminism:
    """Math.random seeding (spec §5): same seed → identical output; different seed → different."""

    def test_same_seed_produces_identical_segments(self, canvas: Canvas) -> None:
        """Running the same sketch twice with the same seed yields bit-identical polylines."""
        gen = TurtleToyGenerator()
        params = {"code": _RANDOM_SKETCH, "seed": 12345}
        result_a = gen.generate(params, canvas)
        result_b = gen.generate(params, canvas)

        assert len(result_a) == len(result_b), (
            "Same seed must produce the same number of polylines"
        )
        for pl_a, pl_b in zip(result_a, result_b):
            assert pl_a == pl_b, (
                "Same seed must produce identical point coordinates"
            )

    def test_different_seeds_produce_different_segments(self, canvas: Canvas) -> None:
        """Different seeds must produce different output for a Math.random-using sketch."""
        gen = TurtleToyGenerator()
        result_a = gen.generate({"code": _RANDOM_SKETCH, "seed": 1}, canvas)
        result_b = gen.generate({"code": _RANDOM_SKETCH, "seed": 2}, canvas)

        # Flatten to a list of all coordinates for easy comparison.
        coords_a = [pt for pl in result_a for pt in pl]
        coords_b = [pt for pl in result_b for pt in pl]

        assert coords_a != coords_b, (
            "Different seeds should yield different coordinates for a random sketch"
        )

    def test_seed_zero_is_deterministic(self, canvas: Canvas) -> None:
        """Seed=0 (default) is still deterministic across two runs."""
        gen = TurtleToyGenerator()
        params = {"code": _RANDOM_SKETCH, "seed": 0}
        result_a = gen.generate(params, canvas)
        result_b = gen.generate(params, canvas)

        assert result_a == result_b, "Seed=0 must also be deterministic"

    def test_seed_param_in_get_parameters(self) -> None:
        """The generator exposes a 'seed' IntParam with default 0."""
        gen = TurtleToyGenerator()
        param_names = [p.name for p in gen.get_parameters()]
        assert "seed" in param_names, "get_parameters() must include a 'seed' parameter"
        seed_param = next(p for p in gen.get_parameters() if p.name == "seed")
        assert seed_param.default == 0, "seed default should be 0"


class TestConsoleLog:
    """console.log stub (spec §5): sketches that call console.log must not crash."""

    def test_console_log_does_not_raise(self, canvas: Canvas) -> None:
        """A sketch calling console.log() runs without error."""
        gen = TurtleToyGenerator()
        # Should not raise any exception.
        polylines = gen.generate({"code": _CONSOLE_LOG_SKETCH}, canvas)
        # The sketch still draws a forward segment.
        total = sum(len(pl) - 1 for pl in polylines)
        assert total >= 1, "Sketch with console.log should still produce segments"

    def test_console_log_with_multiple_args(self, canvas: Canvas) -> None:
        """console.log with multiple args of different types does not raise."""
        sketch = """\
console.log(1, "two", true, null, undefined, {a: 1});
const t = new Turtle();
t.forward(10);
"""
        gen = TurtleToyGenerator()
        polylines = gen.generate({"code": sketch}, canvas)
        assert len(polylines) >= 1
