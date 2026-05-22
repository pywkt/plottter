"""turtletoy.py — TurtleToy-compatible JavaScript sketch runner for Plottter.

Embeds the ``quickjs`` JS engine, exposes a Python-backed ``Turtle`` class to
the JS context, and runs user-pasted JavaScript verbatim — a sketch copied from
`turtletoy.net <https://turtletoy.net>`_ produces identical output in plottter.

Optional dependency: ``quickjs`` from PyPI.  If not installed, the plugin
registers a stub generator that raises a clear error on use.

Plugin phases: 126–131 (see IMPLEMENTATION_PLAN.md).
Spec: ``docs/specs/turtletoy-plugin.md``.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional quickjs import — graceful failure if not installed
# ---------------------------------------------------------------------------

try:
    import quickjs  # type: ignore[import]

    _QUICKJS_AVAILABLE = True
except ImportError:
    _QUICKJS_AVAILABLE = False
    logger.warning(
        "TurtleToy plugin: 'quickjs' package not found. "
        "Install it with: pip install quickjs"
    )

from plottter.generators import register_generator
from plottter.generators.base import (
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
    StringParam,
)
from plottter.models.canvas import Canvas

Polyline = list[tuple[float, float]]

# ---------------------------------------------------------------------------
# Default sketch — the canonical 5-pointed star from spec §8.1
# ---------------------------------------------------------------------------

_STAR_SKETCH = """\
Canvas.setpenopacity(1);
const turtle = new Turtle();
turtle.penup();
turtle.goto(-50, -20);
turtle.pendown();

function walk(i) {
    turtle.forward(100);
    turtle.right(144);
    return i < 4;
}
"""

# ---------------------------------------------------------------------------
# JS shim — eval'd into the quickjs context after callables are registered.
# Defines the ``Turtle`` class, ``Canvas`` singleton, and ``console`` no-op.
# The seeded Math.random is installed separately (see TurtleRuntime._eval_shim).
# ---------------------------------------------------------------------------

_JS_SHIM = """\
// console no-op stub — user sketches may call console.log for debugging.
const console = { log: function() {} };

// Turtle class — mirrors the turtletoy.net API surface.
class Turtle {
    constructor() {
        this._id = _createTurtle();
    }

    // --- Movement ---
    forward(d)                { _turtleForward(this._id, d); }
    fd(d)                     { _turtleForward(this._id, d); }
    backward(d)               { _turtleForward(this._id, -d); }
    bk(d)                     { _turtleForward(this._id, -d); }
    back(d)                   { _turtleForward(this._id, -d); }

    // --- Rotation ---
    right(deg)                { _turtleRight(this._id, deg); }
    rt(deg)                   { _turtleRight(this._id, deg); }
    left(deg)                 { _turtleLeft(this._id, deg); }
    lt(deg)                   { _turtleLeft(this._id, deg); }

    // --- Pen control ---
    penup()                   { _turtlePenup(this._id); }
    pu()                      { _turtlePenup(this._id); }
    up()                      { _turtlePenup(this._id); }
    pendown()                 { _turtlePendown(this._id); }
    pd()                      { _turtlePendown(this._id); }
    down()                    { _turtlePendown(this._id); }

    // --- Absolute movement (draw if pen down) ---
    goto(x, y) {
        if (Array.isArray(x)) { _turtleGoto(this._id, x[0], x[1]); }
        else                  { _turtleGoto(this._id, x, y); }
    }
    setpos(x, y) {
        if (Array.isArray(x)) { _turtleGoto(this._id, x[0], x[1]); }
        else                  { _turtleGoto(this._id, x, y); }
    }
    setposition(x, y) {
        if (Array.isArray(x)) { _turtleGoto(this._id, x[0], x[1]); }
        else                  { _turtleGoto(this._id, x, y); }
    }

    // --- Jump (never draws) ---
    jump(x, y) {
        if (Array.isArray(x)) { _turtleJump(this._id, x[0], x[1]); }
        else                  { _turtleJump(this._id, x, y); }
    }
    jmp(x, y) {
        if (Array.isArray(x)) { _turtleJump(this._id, x[0], x[1]); }
        else                  { _turtleJump(this._id, x, y); }
    }

    // --- Heading control (§4.2) ---
    setheading(deg)           { _turtleSetheading(this._id, deg); }
    seth(deg)                 { _turtleSetheading(this._id, deg); }

    // --- Axis-constrained movement (§4.1, draws if pen down) ---
    setx(x)                   { _turtleSetx(this._id, x); }
    sety(y)                   { _turtleSety(this._id, y); }

    // --- Curves (§4.4) ---
    circle(radius, extent, steps) {
        const fc = _turtleFullCircle(this._id);
        if (extent === undefined) extent = fc;
        // Compute degree-equivalent for the steps default (5° per segment)
        const absExtDeg = Math.abs(extent) * 360 / fc;
        if (steps === undefined) steps = Math.max(8, Math.ceil(absExtDeg / 5));
        _turtleCircle(this._id, radius, extent, steps);
    }

    // --- Home (§4.2, draws if pen down, resets pos + heading) ---
    home()                    { _turtleHome(this._id); }

    // --- State queries (§4.5) ---
    // position() builds the array in JS — quickjs cannot marshal Python lists to JS arrays.
    position()                { return [_turtleXcor(this._id), _turtleYcor(this._id)]; }
    pos()                     { return [_turtleXcor(this._id), _turtleYcor(this._id)]; }
    xcor()                    { return _turtleXcor(this._id); }
    x()                       { return _turtleXcor(this._id); }
    ycor()                    { return _turtleYcor(this._id); }
    y()                       { return _turtleYcor(this._id); }
    heading()                 { return _turtleHeading(this._id); }
    h()                       { return _turtleHeading(this._id); }
    isdown()                  { return _turtleIsdown(this._id); }
    distance(x, y) {
        if (Array.isArray(x)) { return _turtleDistance(this._id, x[0], x[1]); }
        else                  { return _turtleDistance(this._id, x, y); }
    }
    towards(x, y) {
        if (Array.isArray(x)) { return _turtleTowards(this._id, x[0], x[1]); }
        else                  { return _turtleTowards(this._id, x, y); }
    }

    // --- Angular units (§4.6) ---
    degrees(n)                { if (n === undefined) n = 360; _turtleDegrees(this._id, n); }
    radians()                 { _turtleRadians(this._id); }
    fullCircle()              { return _turtleFullCircle(this._id); }

    // --- Clone (§4.7) ---
    // Uses Object.create to avoid calling _createTurtle in the constructor.
    clone() {
        const t = Object.create(Turtle.prototype);
        t._id = _cloneTurtle(this._id);
        return t;
    }
}

// Canvas singleton — v1 records opacity but does not act on it.
const Canvas = {
    setpenopacity(value) { _canvasSetPenOpacity(value); }
};
"""

# ---------------------------------------------------------------------------
# Python-side turtle state
# ---------------------------------------------------------------------------


class TurtleState:
    """Per-turtle state: position, heading (radians), pen_down.

    Heading convention: 0 = East (+x), Y-up (math convention).
    ``right()`` rotates clockwise (decreases heading_rad).
    """

    def __init__(self) -> None:
        self.x: float = 0.0
        self.y: float = 0.0
        self.heading_rad: float = 0.0  # 0 = East, Y-up
        self.pen_down: bool = True
        self.units_per_rotation: float = 360.0  # angular unit: 360 = degrees, 2π = radians

    def forward(self, distance: float) -> tuple[float, float, float, float] | None:
        """Move forward; return segment tuple if pen is down, else None."""
        nx = self.x + math.cos(self.heading_rad) * distance
        ny = self.y + math.sin(self.heading_rad) * distance
        seg = (self.x, self.y, nx, ny) if self.pen_down else None
        self.x, self.y = nx, ny
        return seg

    def right(self, angle: float) -> None:
        self.heading_rad -= angle * (2.0 * math.pi / self.units_per_rotation)

    def left(self, angle: float) -> None:
        self.heading_rad += angle * (2.0 * math.pi / self.units_per_rotation)

    def penup(self) -> None:
        self.pen_down = False

    def pendown(self) -> None:
        self.pen_down = True

    def goto(self, x: float, y: float) -> tuple[float, float, float, float] | None:
        """Move to (x, y); return segment if pen down, else None."""
        seg = (self.x, self.y, x, y) if self.pen_down else None
        self.x, self.y = x, y
        return seg

    def jump(self, x: float, y: float) -> None:
        """Teleport to (x, y) without drawing."""
        self.x, self.y = x, y

    def setheading(self, angle: float) -> None:
        """Set absolute heading in user units (0=east, quarter_turn=north). Never draws."""
        self.heading_rad = angle * (2.0 * math.pi / self.units_per_rotation)

    def setx(self, x: float) -> tuple[float, float, float, float] | None:
        """Change x coordinate, keeping y. Draws if pen down."""
        seg = (self.x, self.y, x, self.y) if self.pen_down else None
        self.x = x
        return seg

    def sety(self, y: float) -> tuple[float, float, float, float] | None:
        """Change y coordinate, keeping x. Draws if pen down."""
        seg = (self.x, self.y, self.x, y) if self.pen_down else None
        self.y = y
        return seg

    def home(self) -> tuple[float, float, float, float] | None:
        """Move to (0, 0) and reset heading to 0. Draws if pen down."""
        seg = (self.x, self.y, 0.0, 0.0) if self.pen_down else None
        self.x, self.y = 0.0, 0.0
        self.heading_rad = 0.0
        return seg

    def circle(
        self, radius: float, extent: float | None = None, steps: int = 72
    ) -> list[tuple[float, float, float, float]]:
        """Trace a circular arc.

        Positive radius = CCW (turn left each step).
        Negative radius = CW (turn right each step).
        ``extent`` is in current user angular units (default = full circle).
        Returns list of segments emitted (pen-down state applies normally).
        """
        if extent is None:
            extent = self.units_per_rotation
        if steps <= 0 or radius == 0:
            return []
        step_angle_user = abs(extent) / steps
        # Convert step angle to radians for chord-length geometry
        step_angle_rad = step_angle_user * (2.0 * math.pi / self.units_per_rotation)
        half_step_rad = step_angle_rad / 2.0
        chord = 2.0 * abs(radius) * math.sin(half_step_rad)
        segs: list[tuple[float, float, float, float]] = []
        for _ in range(steps):
            if radius >= 0:
                self.left(step_angle_user)
            else:
                self.right(step_angle_user)
            seg = self.forward(chord)
            if seg is not None:
                segs.append(seg)
        return segs

    # ------------------------------------------------------------------
    # State queries (spec §4.5)
    # ------------------------------------------------------------------

    def position(self) -> list[float]:
        """Return current position as [x, y]."""
        return [self.x, self.y]

    def xcor(self) -> float:
        """Return current x coordinate."""
        return self.x

    def ycor(self) -> float:
        """Return current y coordinate."""
        return self.y

    def heading_user(self) -> float:
        """Heading in current user units, normalised to [0, units_per_rotation)."""
        degrees = math.degrees(self.heading_rad) % 360
        return degrees * self.units_per_rotation / 360.0

    def isdown(self) -> bool:
        """Return True if the pen is currently down."""
        return self.pen_down

    def distance(self, tx: float, ty: float) -> float:
        """Euclidean distance from current position to (tx, ty)."""
        return math.hypot(tx - self.x, ty - self.y)

    def towards(self, tx: float, ty: float) -> float:
        """Angle in user units toward (tx, ty), normalised to [0, units_per_rotation)."""
        deg = math.degrees(math.atan2(ty - self.y, tx - self.x)) % 360
        return deg * self.units_per_rotation / 360.0



# ---------------------------------------------------------------------------
# TurtleRuntime — owns the quickjs context, turtle registry, and segment list
# ---------------------------------------------------------------------------


class TurtleRuntime:
    """Owns the quickjs.Context and all Python-side turtle state.

    One TurtleRuntime is created per generator invocation.
    """

    def __init__(self, seed: int = 0) -> None:
        self.ctx = quickjs.Context()  # type: ignore[name-defined]
        self.turtles: dict[int, TurtleState] = {}
        self.next_id: int = 0
        self.segments: list[tuple[float, float, float, float]] = []
        self.pen_opacity: float = 1.0
        self.seed = seed
        self._register_callables()
        self._eval_shim()

    def _register_callables(self) -> None:
        ctx = self.ctx
        ctx.add_callable("_createTurtle", self._py_create_turtle)
        ctx.add_callable("_cloneTurtle", self._py_clone_turtle)
        ctx.add_callable("_turtleForward", self._py_turtle_forward)
        ctx.add_callable("_turtleRight", self._py_turtle_right)
        ctx.add_callable("_turtleLeft", self._py_turtle_left)
        ctx.add_callable("_turtlePenup", self._py_turtle_penup)
        ctx.add_callable("_turtlePendown", self._py_turtle_pendown)
        ctx.add_callable("_turtleGoto", self._py_turtle_goto)
        ctx.add_callable("_turtleJump", self._py_turtle_jump)
        ctx.add_callable("_turtleSetheading", self._py_turtle_setheading)
        ctx.add_callable("_turtleSetx", self._py_turtle_setx)
        ctx.add_callable("_turtleSety", self._py_turtle_sety)
        ctx.add_callable("_turtleHome", self._py_turtle_home)
        ctx.add_callable("_turtleCircle", self._py_turtle_circle)
        ctx.add_callable("_canvasSetPenOpacity", self._py_canvas_set_pen_opacity)
        ctx.add_callable("_turtleXcor", self._py_turtle_xcor)
        ctx.add_callable("_turtleYcor", self._py_turtle_ycor)
        ctx.add_callable("_turtleHeading", self._py_turtle_heading)
        ctx.add_callable("_turtleIsdown", self._py_turtle_isdown)
        ctx.add_callable("_turtleDistance", self._py_turtle_distance)
        ctx.add_callable("_turtleTowards", self._py_turtle_towards)
        ctx.add_callable("_turtleDegrees", self._py_turtle_degrees)
        ctx.add_callable("_turtleRadians", self._py_turtle_radians)
        ctx.add_callable("_turtleFullCircle", self._py_turtle_full_circle)

    def _eval_shim(self) -> None:
        # Inject seeded Math.random first (seed value is templated via f-string)
        self.ctx.eval(
            f"""
(function() {{
    let s = {self.seed} >>> 0;
    Math.random = function() {{
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        return s / 0x80000000;
    }};
}})();
"""
        )
        # Eval the Turtle class + Canvas + console stubs
        self.ctx.eval(_JS_SHIM)

    # ------------------------------------------------------------------
    # Python callables registered into the JS context
    # ------------------------------------------------------------------

    def _py_create_turtle(self) -> int:
        tid = self.next_id
        self.next_id += 1
        self.turtles[tid] = TurtleState()
        return tid

    def _py_turtle_forward(self, tid: int, d: float) -> None:
        seg = self.turtles[int(tid)].forward(float(d))
        if seg is not None:
            self.segments.append(seg)

    def _py_turtle_right(self, tid: int, deg: float) -> None:
        self.turtles[int(tid)].right(float(deg))

    def _py_turtle_left(self, tid: int, deg: float) -> None:
        self.turtles[int(tid)].left(float(deg))

    def _py_turtle_penup(self, tid: int) -> None:
        self.turtles[int(tid)].penup()

    def _py_turtle_pendown(self, tid: int) -> None:
        self.turtles[int(tid)].pendown()

    def _py_turtle_goto(self, tid: int, x: float, y: float) -> None:
        seg = self.turtles[int(tid)].goto(float(x), float(y))
        if seg is not None:
            self.segments.append(seg)

    def _py_turtle_jump(self, tid: int, x: float, y: float) -> None:
        self.turtles[int(tid)].jump(float(x), float(y))

    def _py_turtle_setheading(self, tid: int, deg: float) -> None:
        self.turtles[int(tid)].setheading(float(deg))

    def _py_turtle_setx(self, tid: int, x: float) -> None:
        seg = self.turtles[int(tid)].setx(float(x))
        if seg is not None:
            self.segments.append(seg)

    def _py_turtle_sety(self, tid: int, y: float) -> None:
        seg = self.turtles[int(tid)].sety(float(y))
        if seg is not None:
            self.segments.append(seg)

    def _py_turtle_home(self, tid: int) -> None:
        seg = self.turtles[int(tid)].home()
        if seg is not None:
            self.segments.append(seg)

    def _py_turtle_circle(self, tid: int, radius: float, extent: float, steps: int) -> None:
        segs = self.turtles[int(tid)].circle(float(radius), float(extent), int(steps))
        self.segments.extend(segs)

    def _py_canvas_set_pen_opacity(self, value: float) -> None:
        self.pen_opacity = float(value)

    # ------------------------------------------------------------------
    # State query callables (spec §4.5)
    # ------------------------------------------------------------------

    def _py_turtle_xcor(self, tid: int) -> float:
        return self.turtles[int(tid)].x

    def _py_turtle_ycor(self, tid: int) -> float:
        return self.turtles[int(tid)].y

    def _py_turtle_heading(self, tid: int) -> float:
        return self.turtles[int(tid)].heading_user()

    def _py_turtle_isdown(self, tid: int) -> bool:
        return self.turtles[int(tid)].isdown()

    def _py_turtle_distance(self, tid: int, x: float, y: float) -> float:
        return self.turtles[int(tid)].distance(float(x), float(y))

    def _py_turtle_towards(self, tid: int, x: float, y: float) -> float:
        return self.turtles[int(tid)].towards(float(x), float(y))

    # ------------------------------------------------------------------
    # Clone (spec §4.7)
    # ------------------------------------------------------------------

    def _py_clone_turtle(self, tid: int) -> int:
        src = self.turtles[int(tid)]
        new_tid = self.next_id
        self.next_id += 1
        new_state = TurtleState()
        new_state.x = src.x
        new_state.y = src.y
        new_state.heading_rad = src.heading_rad
        new_state.pen_down = src.pen_down
        new_state.units_per_rotation = src.units_per_rotation
        self.turtles[new_tid] = new_state
        return new_tid

    # ------------------------------------------------------------------
    # Angular units (spec §4.6)
    # ------------------------------------------------------------------

    def _py_turtle_degrees(self, tid: int, n: float = 360.0) -> None:
        self.turtles[int(tid)].units_per_rotation = float(n)

    def _py_turtle_radians(self, tid: int) -> None:
        self.turtles[int(tid)].units_per_rotation = 2.0 * math.pi

    def _py_turtle_full_circle(self, tid: int) -> float:
        return self.turtles[int(tid)].units_per_rotation



# ---------------------------------------------------------------------------
# Segment-to-polyline grouping + coordinate mapping (spec §7.3–§7.4)
# ---------------------------------------------------------------------------


def _segments_to_polylines(
    segments: list[tuple[float, float, float, float]],
    canvas: Canvas,
    fit_mode: str = "letterbox",
    mm_per_unit: float = 1.0,
) -> list[Polyline]:
    """Convert world-space segments to canvas-mm polylines.

    Step 1: Group consecutive co-terminal segments into polylines.
    Step 2: Map world coordinates to canvas mm.

    Args:
        segments:    Raw ``(x1, y1, x2, y2)`` tuples in TurtleToy world space.
        canvas:      Target canvas (provides printable area bounds).
        fit_mode:    One of ``"letterbox"``, ``"stretch"``, ``"fixed_scale"``.
        mm_per_unit: Scale factor used only when ``fit_mode="fixed_scale"``.

    Returns:
        A list of polylines, each a list of ``(x_mm, y_mm)`` tuples.
    """
    if not segments:
        return []

    # --- Step 1: group consecutive co-terminal segments ---
    raw_polylines: list[list[tuple[float, float]]] = []
    x1, y1, x2, y2 = segments[0]
    current: list[tuple[float, float]] = [(x1, y1), (x2, y2)]

    for seg in segments[1:]:
        sx1, sy1, sx2, sy2 = seg
        px, py = current[-1]
        if abs(sx1 - px) < 1e-6 and abs(sy1 - py) < 1e-6:
            # Consecutive — extend the current polyline
            current.append((sx2, sy2))
        else:
            # Gap — start a new polyline
            raw_polylines.append(current)
            current = [(sx1, sy1), (sx2, sy2)]

    raw_polylines.append(current)

    # --- Step 2: compute coordinate mapping (spec §7.3) ---
    left, top, right, bottom = canvas.drawing_area()
    width = right - left
    height = bottom - top
    cx = left + width / 2.0
    cy = top + height / 2.0

    if fit_mode == "stretch":
        sx = width / 200.0
        sy = height / 200.0
    elif fit_mode == "fixed_scale":
        sx = sy = mm_per_unit
    else:  # letterbox (default)
        scale = min(width / 200.0, height / 200.0)
        sx = sy = scale

    def world_to_mm(wx: float, wy: float) -> tuple[float, float]:
        # Y-flip: TurtleToy Y-up → plottter/SVG Y-down
        return (cx + wx * sx, cy - wy * sy)

    # --- Step 3: apply mapping ---
    return [[world_to_mm(wx, wy) for wx, wy in raw] for raw in raw_polylines]


# ---------------------------------------------------------------------------
# The generator class
# ---------------------------------------------------------------------------


@register_generator
class TurtleToyGenerator(Generator):
    """Run TurtleToy-compatible JavaScript sketches and emit plotter paths.

    Embeds the ``quickjs`` engine; user code is executed verbatim in an
    isolated JS context with a Python-backed ``Turtle`` API.

    Requires: ``pip install quickjs``
    """

    name = "TurtleToy"
    category = "code"
    uses_source_image = False
    emits_multiple_layers = False

    def get_parameters(self) -> list[Parameter]:
        return [
            StringParam(
                name="code",
                label="Code",
                default=_STAR_SKETCH,
                description=(
                    "JavaScript code to execute. "
                    "Paste any sketch from turtletoy.net — it runs verbatim."
                ),
                multiline=True,
            ),
            ChoiceParam(
                name="fit_mode",
                label="Fit Mode",
                choices=["letterbox", "stretch", "fixed_scale"],
                default="letterbox",
                description=(
                    "How to map the TurtleToy world (-100..100) to canvas mm. "
                    "letterbox: preserve aspect ratio (default). "
                    "stretch: fill canvas independently in x and y (distorts). "
                    "fixed_scale: use mm_per_unit as a literal scale factor."
                ),
            ),
            FloatParam(
                name="mm_per_unit",
                label="mm per unit",
                min=0.1,
                max=10.0,
                step=0.1,
                default=1.0,
                description=(
                    "Scale factor: how many mm correspond to one TurtleToy world unit. "
                    "Only used when fit_mode is fixed_scale."
                ),
                visible_when={"fit_mode": ["fixed_scale"]},
            ),
            IntParam(
                name="seed",
                label="Seed",
                min=0,
                max=2**31 - 1,
                step=1,
                default=0,
                description=(
                    "Random seed for Math.random. "
                    "Same seed + same sketch always produces identical output."
                ),
                randomizable=False,
            ),
            IntParam(
                name="max_steps",
                label="Max Steps",
                min=1,
                max=1_000_000,
                step=1000,
                default=100_000,
                description="Maximum number of walk() iterations before the loop is stopped.",
                randomizable=False,
            ),
            FloatParam(
                name="timeout_seconds",
                label="Timeout (s)",
                min=1.0,
                max=120.0,
                step=1.0,
                default=30.0,
                description="Wall-clock time limit for the walk loop in seconds.",
                randomizable=False,
            ),
        ]

    def get_presets(self) -> list[Preset]:
        if not _QUICKJS_AVAILABLE:
            return [
                Preset(
                    name="(quickjs not installed)",
                    params={},
                    description=(
                        "TurtleToy requires the 'quickjs' package. "
                        "Install it with: pip install quickjs"
                    ),
                )
            ]
        return [
            Preset(
                name="Five-Point Star",
                params={"code": _STAR_SKETCH},
                description="The canonical 5-pointed star from the TurtleToy docs.",
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        if not _QUICKJS_AVAILABLE:
            raise RuntimeError(
                "TurtleToy requires the 'quickjs' package. "
                "Install it with: pip install quickjs"
            )

        code: str = params.get("code", _STAR_SKETCH)
        max_steps: int = int(params.get("max_steps", 100_000))
        timeout_seconds: float = float(params.get("timeout_seconds", 30.0))
        fit_mode: str = str(params.get("fit_mode", "letterbox"))
        mm_per_unit: float = float(params.get("mm_per_unit", 1.0))

        seed: int = int(params.get("seed", 0))
        runtime = TurtleRuntime(seed=seed)

        # Eval user code — abort on error
        runtime.ctx.eval(code)

        # Check if walk function is defined
        has_walk: bool = bool(runtime.ctx.eval("typeof walk === 'function'"))
        if has_walk:
            deadline = time.monotonic() + timeout_seconds
            for i in range(max_steps):
                if time.monotonic() > deadline:
                    logger.warning("TurtleToy: walk loop hit timeout (%ss)", timeout_seconds)
                    break
                if cancelled_callback is not None and cancelled_callback():
                    break
                try:
                    keep_going = runtime.ctx.eval(f"walk({i})")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("TurtleToy: walk(%d) raised: %s (output is partial)", i, exc)
                    break
                if not keep_going:
                    break

        return _segments_to_polylines(
            runtime.segments, canvas, fit_mode=fit_mode, mm_per_unit=mm_per_unit
        )
