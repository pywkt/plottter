"""
Performance test for parse_adjustable_vars (spec §6).

Requirement: a 500-line sketch with 30 adjustable-variable declarations must
be parsed in < 50 ms mean wall-clock time over 100 repetitions.
"""

from __future__ import annotations

import textwrap
import time

import pytest

from plottter.generators._adjustable_vars import parse_adjustable_vars


# ---------------------------------------------------------------------------
# Synthetic sketch builder
# ---------------------------------------------------------------------------

_ADJUSTABLE_DECLS = [
    # numeric int
    "const iterations = 50; // min=1, max=200, step=1, Iteration count",
    "const gridSize = 10; // min=2, max=100, Cells per side",
    "const lineWidth = 1; // min=1, max=20",
    "const segments = 8; // min=3, max=64, step=1",
    "const layers = 3; // min=1, max=10",
    # numeric float
    "const scale = 1.0; // min=0.1, max=5.0, step=0.1, Scale factor",
    "const angle = 0.5; // min=0.0, max=6.283, step=0.01",
    "const decay = 0.95; // min=0.5, max=1.0, step=0.01",
    "const frequency = 2.5; // min=0.1, max=10.0, step=0.1",
    "const amplitude = 1.5; // min=0.0, max=3.0, step=0.05",
    # numeric with negative min
    "const offsetX = 0; // min=-100, max=100",
    "const offsetY = 0; // min=-100, max=100",
    "const rotation = 0.0; // min=-3.14159, max=3.14159, step=0.01",
    # numeric with scientific notation
    "const epsilon = 1e-3; // min=1e-6, max=1e-1",
    "const bigNum = 1e4; // min=1e2, max=1e6",
    # choice
    "const colorMode = 'rgb'; // (rgb, hsv, lab) Color space",
    "const blendMode = 'normal'; // (normal, multiply, screen, overlay)",
    "const fillStyle = 'solid'; // (solid, hatched, dotted)",
    "const lineCap = 'round'; // (butt, round, square)",
    "const direction = 'horizontal'; // (horizontal, vertical, diagonal)",
    # string
    "const label = 'hello'; // type=string, Title label",
    "const fontName = 'Arial'; // type=string",
    "const outputPath = ''; // type=string, Optional output path",
    # path
    "const maskPath = ''; // type=path, Mask image",
    "const texturePath = ''; // type=path",
    # let keyword
    "let speed = 5; // min=1, max=20",
    "let density = 0.5; // min=0.0, max=1.0, step=0.01",
    "let noiseScale = 3; // min=1, max=50",
    "let seed = 42; // min=0, max=9999",
    "let margin = 10; // min=0, max=50, step=5, Page margin",
]

assert len(_ADJUSTABLE_DECLS) == 30, "Fixture must have exactly 30 adjustable declarations"

_FILLER_LINES = [
    "// A standard TurtleToy-style sketch",
    "const WIDTH = 200;",
    "const HEIGHT = 200;",
    "function setup() {",
    "  createCanvas(WIDTH, HEIGHT);",
    "}",
    "function draw() {",
    "  background(255);",
    "  noFill();",
    "  stroke(0);",
    "  strokeWeight(1);",
    "}",
    "function line(x1, y1, x2, y2) {",
    "  // draw a line from (x1, y1) to (x2, y2)",
    "  drawLine(x1, y1, x2, y2);",
    "}",
    "function circle(cx, cy, r) {",
    "  for (let i = 0; i < 64; i++) {",
    "    const a = (i / 64) * Math.PI * 2;",
    "    const b = ((i + 1) / 64) * Math.PI * 2;",
    "    line(cx + Math.cos(a) * r, cy + Math.sin(a) * r,",
    "         cx + Math.cos(b) * r, cy + Math.sin(b) * r);",
    "  }",
    "}",
    "// End of helper functions",
    "",
    "// Main render",
    "function render() {",
    "  for (let row = 0; row < gridSize; row++) {",
    "    for (let col = 0; col < gridSize; col++) {",
    "      const x = (col / gridSize) * WIDTH;",
    "      const y = (row / gridSize) * HEIGHT;",
    "      circle(x, y, scale * 10);",
    "    }",
    "  }",
    "}",
    "render();",
    "",
]


def _build_sketch(target_lines: int = 500) -> str:
    """Build a synthetic sketch of *target_lines* lines with 30 adjustable vars."""
    # Interleave adjustable declarations with filler so the parser must scan the
    # full file.  Place one adjustable decl roughly every 16 filler lines.
    lines: list[str] = []

    filler_cycle = _FILLER_LINES * (target_lines // len(_FILLER_LINES) + 2)
    filler_idx = 0
    decl_idx = 0

    spacing = (target_lines - len(_ADJUSTABLE_DECLS)) // len(_ADJUSTABLE_DECLS)

    while len(lines) < target_lines:
        # Insert a batch of filler lines
        for _ in range(spacing):
            if len(lines) >= target_lines:
                break
            lines.append(filler_cycle[filler_idx % len(filler_cycle)])
            filler_idx += 1

        # Insert the next adjustable declaration (if any remain)
        if decl_idx < len(_ADJUSTABLE_DECLS) and len(lines) < target_lines:
            lines.append(_ADJUSTABLE_DECLS[decl_idx])
            decl_idx += 1

    # Pad with filler to reach exactly target_lines
    while len(lines) < target_lines:
        lines.append(filler_cycle[filler_idx % len(filler_cycle)])
        filler_idx += 1

    return "\n".join(lines[:target_lines])


# Build the sketch once at module load so import overhead doesn't affect timing.
_SKETCH_500 = _build_sketch(500)
_SKETCH_LINE_COUNT = len(_SKETCH_500.splitlines())


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def test_sketch_has_500_lines():
    """Confirm the synthetic sketch is exactly 500 lines."""
    assert _SKETCH_LINE_COUNT == 500


def test_sketch_has_30_adjustable_vars():
    """Confirm all 30 adjustable declarations are present and parseable."""
    vars_ = parse_adjustable_vars(_SKETCH_500)
    assert len(vars_) == 30, (
        f"Expected 30 adjustable vars, got {len(vars_)}: "
        + str([v.name for v in vars_])
    )


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------


def test_parse_adjustable_vars_perf():
    """Mean parse time over 100 calls must be < 50 ms (spec §6)."""
    REPETITIONS = 100
    BUDGET_MS = 50.0

    # Warm-up pass (avoid cold-start JIT / import overhead distorting results)
    parse_adjustable_vars(_SKETCH_500)

    start = time.perf_counter()
    for _ in range(REPETITIONS):
        parse_adjustable_vars(_SKETCH_500)
    elapsed = time.perf_counter() - start

    mean_ms = (elapsed / REPETITIONS) * 1000.0
    print(
        f"\nparse_adjustable_vars: {REPETITIONS} calls on {_SKETCH_LINE_COUNT}-line sketch "
        f"→ mean {mean_ms:.3f} ms  (budget {BUDGET_MS} ms)"
    )

    assert mean_ms < BUDGET_MS, (
        f"Parser too slow: mean {mean_ms:.2f} ms > budget {BUDGET_MS} ms per call "
        f"on {_SKETCH_LINE_COUNT}-line sketch"
    )
