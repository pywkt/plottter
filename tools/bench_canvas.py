#!/usr/bin/env python3
"""Offscreen canvas rendering benchmark (spec §5.2).

Builds a real :class:`ProjectController` + :class:`CanvasWidget`, fills the
project with a synthetic random-walk scene, sizes the widget 1400×900, and
times ``widget.render(QImage)`` over a handful of frames — reporting min/mean
milliseconds per frame.

Run under the offscreen platform plugin::

    QT_QPA_PLATFORM=offscreen python tools/bench_canvas.py --paths 10000 --pts 12
    QT_QPA_PLATFORM=offscreen python tools/bench_canvas.py --paths 38000 --pts 12 --json

``--no-cache`` sets the §9 bypass flag (``PLOTTTER_NO_CANVAS_CACHE``) so the
uncached paint path can be measured apples-to-apples against the cached one as
later phases land. The recorded baseline numbers live in
``docs/performance.md`` under "Canvas rendering".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Default to the offscreen platform plugin so the bench runs headless without
# the caller having to remember to export it (a real X/Wayland display is also
# fine — we never override an explicitly chosen plugin).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

# Widget size per spec §5.2.
WIDGET_W = 1400
WIDGET_H = 900

# A2 canvas — the project's real working surface for 10k–38k path plots.
A2_W_MM = 420.0
A2_H_MM = 594.0

# Seed fixed per spec for reproducible scenes.
RNG_SEED = 42


def build_scene(n_paths: int, n_pts: int):
    """Return a Project filled with *n_paths* random-walk polylines.

    Each polyline has *n_pts* points: a seeded start somewhere on the A2
    canvas followed by a Gaussian random walk (clamped to the sheet) — a
    cheap stand-in for the dense line work of a real plot.
    """
    from plottter.models import Canvas, Layer, Project

    rng = np.random.default_rng(RNG_SEED)

    # Random start per path, then a cumulative Gaussian walk. Vectorised so the
    # scene build itself stays a negligible fraction of the bench runtime.
    starts = rng.uniform(
        low=(0.0, 0.0), high=(A2_W_MM, A2_H_MM), size=(n_paths, 2)
    )
    steps = rng.normal(loc=0.0, scale=4.0, size=(n_paths, n_pts - 1, 2))
    walk = np.empty((n_paths, n_pts, 2), dtype=np.float64)
    walk[:, 0, :] = starts
    walk[:, 1:, :] = starts[:, None, :] + np.cumsum(steps, axis=1)
    np.clip(walk[..., 0], 0.0, A2_W_MM, out=walk[..., 0])
    np.clip(walk[..., 1], 0.0, A2_H_MM, out=walk[..., 1])

    canvas = Canvas(width_mm=A2_W_MM, height_mm=A2_H_MM, margin_mm=15.0)
    project = Project(name="BenchScene", canvas=canvas, registration_marks=False)
    layer = Layer(name="walk", color="#000000")
    layer.paths = [[(float(x), float(y)) for x, y in path] for path in walk]
    project.add_layer(layer)
    return project


def run_bench(n_paths: int, n_pts: int, frames: int) -> dict:
    """Render the synthetic scene *frames* times; return timing stats (ms)."""
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication

    from plottter.gui.canvas_widget import CanvasWidget
    from plottter.gui.project_controller import ProjectController

    app = QApplication.instance() or QApplication(sys.argv[:1])

    project = build_scene(n_paths, n_pts)
    controller = ProjectController(project)
    widget = CanvasWidget(controller)
    widget.resize(WIDGET_W, WIDGET_H)
    # Pin a fit view directly (the widget never gets a showEvent offscreen).
    widget._fitted = True
    widget.fit_to_window()

    img = QImage(WIDGET_W, WIDGET_H, QImage.Format.Format_ARGB32)

    times_ms: list[float] = []
    for _ in range(frames):
        img.fill(0)
        start = time.perf_counter()
        widget.render(img)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    # Keep `app` referenced until rendering is done (avoid premature teardown).
    del app
    return {
        "paths": n_paths,
        "pts": n_pts,
        "frames": frames,
        "cache_enabled": os.environ.get("PLOTTTER_NO_CANVAS_CACHE") != "1",
        "min_ms": min(times_ms),
        "mean_ms": sum(times_ms) / len(times_ms),
        "times_ms": times_ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", type=int, default=10000, help="number of polylines")
    parser.add_argument("--pts", type=int, default=12, help="points per polyline")
    parser.add_argument("--frames", type=int, default=5, help="frames to render")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="set PLOTTTER_NO_CANVAS_CACHE=1 to measure the uncached path",
    )
    args = parser.parse_args(argv)

    if args.no_cache:
        os.environ["PLOTTTER_NO_CANVAS_CACHE"] = "1"

    stats = run_bench(args.paths, args.pts, args.frames)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        cache = "on" if stats["cache_enabled"] else "off (--no-cache)"
        print(
            f"{stats['paths']}×{stats['pts']} paths, {stats['frames']} frames, "
            f"cache {cache}:"
        )
        print(f"  min  {stats['min_ms']:.1f} ms")
        print(f"  mean {stats['mean_ms']:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
