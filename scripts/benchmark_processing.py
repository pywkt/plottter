#!/usr/bin/env python3
"""Benchmark: full optimize pipeline on three synthetic workloads.

Three synthetic inputs represent typical real-world plottter workloads:

  stipple  —  500 paths  — short 2-3 point paths scattered across a 200×200 mm
               canvas (stipple/dot art, Voronoi stippling output).
  map      — 2000 paths  — road-network-style grid with endpoint jitter and
               overlapping segments, as produced by MapGenerator.
  flow     — 5000 paths  — dense short paths following curved flow directions,
               as produced by flow-field / image-to-lines generators.

Pipeline stages measured (mirrors _OptimizeWorker order):
  1. weld      — remove duplicate overlapping segments (JIT-accelerated)
  2. simplify  — Ramer-Douglas-Peucker path simplification
  3. merge     — snap nearby endpoints together (JIT-accelerated)
  4. reorder   — nearest-neighbour path ordering
  5. 2-opt     — 2-opt improvement passes
  6. or-opt    — Or-opt relocate passes

Usage::

    # Benchmark in the current JIT mode (auto-detected):
    python scripts/benchmark_processing.py

    # Compare pure-Python vs JIT in one run (spawns two subprocesses):
    python scripts/benchmark_processing.py --compare

    # Emit machine-readable JSON (used by --compare internally):
    python scripts/benchmark_processing.py --json

    # Force pure-Python mode regardless of numba installation:
    NUMBA_DISABLE_JIT=1 python scripts/benchmark_processing.py

Exit code is 0 on success.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plottter.processing._jit import JIT_ENABLED
from plottter.processing import (
    weld_overlapping_paths,
    simplify_paths,
    merge_nearby_paths,
    reorder_paths,
    optimize_2opt,
    optimize_or_opt,
)

# ---------------------------------------------------------------------------
# JIT mode detection
# ---------------------------------------------------------------------------

# NUMBA_DISABLE_JIT=1 disables JIT compilation even when numba is installed.
_NUMBA_DISABLED = bool(os.getenv("NUMBA_DISABLE_JIT", ""))
JIT_ACTIVE: bool = JIT_ENABLED and not _NUMBA_DISABLED
MODE_LABEL: str = "JIT (numba)" if JIT_ACTIVE else "pure Python"


# ---------------------------------------------------------------------------
# Synthetic input generators
# ---------------------------------------------------------------------------

def _jitter(rng: random.Random, scale: float = 0.04) -> float:
    return rng.uniform(-scale, scale)


def make_stipple_scatter(
    n_paths: int = 500,
    canvas_mm: float = 200.0,
    seed: int = 42,
) -> list[list[tuple[float, float]]]:
    """Return *n_paths* short (2–3-point) paths scattered randomly on the canvas.

    Represents output from stippling or Voronoi-stipple generators: each
    'dot' is a tiny 2-point stroke.  A quarter of paths get a middle point
    to simulate 3-point micro-strokes.
    """
    rng = random.Random(seed)
    paths: list[list[tuple[float, float]]] = []
    for i in range(n_paths):
        cx = rng.uniform(2.0, canvas_mm - 2.0)
        cy = rng.uniform(2.0, canvas_mm - 2.0)
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(0.3, 1.5)
        dx = math.cos(angle) * length * 0.5
        dy = math.sin(angle) * length * 0.5
        p0 = (cx - dx, cy - dy)
        p1 = (cx + dx, cy + dy)
        if i % 4 == 0:  # 25% get a midpoint
            pm = (cx + _jitter(rng, 0.1), cy + _jitter(rng, 0.1))
            paths.append([p0, pm, p1])
        else:
            paths.append([p0, p1])
    return paths


def make_map_network(
    n_paths: int = 2000,
    canvas_mm: float = 200.0,
    seed: int = 42,
) -> list[list[tuple[float, float]]]:
    """Return a road-network-style path collection.

    Mimics MapGenerator output: a mix of longer road polylines (3–6 points)
    arranged in a roughly grid-like pattern with sub-0.1 mm endpoint jitter
    (OSM projection noise), some overlapping segments on parallel streets,
    and a handful of diagonal connectors.
    """
    rng = random.Random(seed)
    paths: list[list[tuple[float, float]]] = []
    jitter = 0.04  # mm — typical OSM projection noise at plotter scale

    grid_cols = 20
    grid_rows = 20
    step = canvas_mm / grid_cols

    # Horizontal road segments (each road = one polyline spanning one block)
    for r in range(grid_rows + 1):
        y_base = r * step
        for c in range(grid_cols):
            x0 = c * step
            x1 = (c + 1) * step
            # Road with 2–4 intermediate points
            n_pts = rng.randint(2, 4)
            pts: list[tuple[float, float]] = []
            for k in range(n_pts):
                t = k / (n_pts - 1)
                x = x0 + t * (x1 - x0) + _jitter(rng, jitter)
                y = y_base + _jitter(rng, jitter)
                pts.append((x, y))
            paths.append(pts)
            if len(paths) >= n_paths:
                break
        if len(paths) >= n_paths:
            break

    # Vertical road segments
    for c in range(grid_cols + 1):
        x_base = c * step
        for r in range(grid_rows):
            y0 = r * step
            y1 = (r + 1) * step
            n_pts = rng.randint(2, 4)
            pts = []
            for k in range(n_pts):
                t = k / (n_pts - 1)
                x = x_base + _jitter(rng, jitter)
                y = y0 + t * (y1 - y0) + _jitter(rng, jitter)
                pts.append((x, y))
            paths.append(pts)
            if len(paths) >= n_paths:
                break
        if len(paths) >= n_paths:
            break

    # Diagonal connectors (simulate minor roads / paths)
    while len(paths) < n_paths:
        cx = rng.uniform(5.0, canvas_mm - 5.0)
        cy = rng.uniform(5.0, canvas_mm - 5.0)
        angle = rng.choice([math.pi / 4, -math.pi / 4, 3 * math.pi / 4])
        length = rng.uniform(step * 0.8, step * 1.5)
        dx = math.cos(angle) * length * 0.5
        dy = math.sin(angle) * length * 0.5
        paths.append([(cx - dx, cy - dy), (cx + dx, cy + dy)])

    # Add a few intentional duplicate segments for weld benchmarking
    n_dupes = min(50, len(paths) // 10)
    for i in range(n_dupes):
        original = paths[rng.randint(0, len(paths) - 1)]
        # Slightly perturbed copy (within weld tolerance)
        duped = [
            (pt[0] + _jitter(rng, 0.03), pt[1] + _jitter(rng, 0.03))
            for pt in original
        ]
        paths.append(duped)

    return paths[:n_paths]


def make_flow_scatter(
    n_paths: int = 5000,
    canvas_mm: float = 200.0,
    seed: int = 42,
) -> list[list[tuple[float, float]]]:
    """Return a dense collection of short curved flow-field paths.

    Mimics flow-field / squiggle / LIC generator output: thousands of short
    2–4-point strokes following smooth curvilinear directions across the canvas.
    """
    rng = random.Random(seed)
    paths: list[list[tuple[float, float]]] = []

    for _ in range(n_paths):
        # Seed point anywhere on canvas
        x = rng.uniform(1.0, canvas_mm - 1.0)
        y = rng.uniform(1.0, canvas_mm - 1.0)

        # Flow direction from a simple sine-based field
        angle = (
            math.sin(x * 0.05) * math.cos(y * 0.05) * math.pi
            + rng.uniform(-0.3, 0.3)
        )
        n_pts = rng.randint(2, 4)
        seg_len = rng.uniform(0.5, 2.5) / (n_pts - 1)

        pts: list[tuple[float, float]] = [(x, y)]
        for k in range(1, n_pts):
            # Gently curve the direction
            angle += rng.uniform(-0.15, 0.15)
            nx = pts[-1][0] + math.cos(angle) * seg_len
            ny = pts[-1][1] + math.sin(angle) * seg_len
            # Clamp to canvas
            nx = max(0.5, min(canvas_mm - 0.5, nx))
            ny = max(0.5, min(canvas_mm - 0.5, ny))
            pts.append((nx, ny))

        paths.append(pts)

    return paths


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _tick() -> float:
    return time.perf_counter()


def run_pipeline(
    paths: list[list[tuple[float, float]]],
    label: str,
    progress_file=None,
) -> dict[str, Any]:
    """Run the full optimize pipeline and return per-stage wall-clock times (s).

    *progress_file*: file to print progress to (default stdout).  Pass
    ``sys.stderr`` in ``--json`` mode so stdout stays clean for JSON output.
    """
    import sys as _sys
    _out = progress_file if progress_file is not None else _sys.stdout
    timings: dict[str, float] = {}
    path_counts: dict[str, int] = {"input": len(paths)}

    print(f"  Running {label} ({len(paths)} paths)…", file=_out, flush=True)

    # 1. Weld
    t0 = _tick()
    paths = weld_overlapping_paths(paths, tolerance_mm=0.1)
    timings["weld"] = _tick() - t0
    path_counts["after_weld"] = len(paths)

    # 2. Simplify
    t0 = _tick()
    paths = simplify_paths(paths, tolerance_mm=0.1)
    timings["simplify"] = _tick() - t0
    path_counts["after_simplify"] = len(paths)

    # 3. Merge
    t0 = _tick()
    paths = merge_nearby_paths(paths, threshold_mm=0.5)
    timings["merge"] = _tick() - t0
    path_counts["after_merge"] = len(paths)

    # 4. Reorder (NN)
    t0 = _tick()
    paths = reorder_paths(paths, num_starts=5)
    timings["reorder"] = _tick() - t0

    # 5. 2-opt
    t0 = _tick()
    paths = optimize_2opt(paths)
    timings["2opt"] = _tick() - t0

    # 6. Or-opt
    t0 = _tick()
    paths = optimize_or_opt(paths)
    timings["or_opt"] = _tick() - t0

    timings["total"] = sum(timings.values())
    path_counts["output"] = len(paths)

    for stage, t in timings.items():
        if stage == "total":
            continue
        print(f"    {stage:10s}  {t:7.3f} s", file=_out, flush=True)
    print(f"    {'total':10s}  {timings['total']:7.3f} s", file=_out, flush=True)
    print(f"    paths: {path_counts['input']} → {path_counts['output']}", file=_out, flush=True)
    print(file=_out, flush=True)

    return {"label": label, "mode": MODE_LABEL, "timings": timings, "counts": path_counts}


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------

WORKLOADS = [
    ("stipple", make_stipple_scatter, {"n_paths": 500}),
    ("map",     make_map_network,     {"n_paths": 2000}),
    ("flow",    make_flow_scatter,    {"n_paths": 5000}),
]


def run_all(progress_file=None) -> list[dict[str, Any]]:
    results = []
    for name, factory, kwargs in WORKLOADS:
        paths = factory(**kwargs)
        result = run_pipeline(paths, name, progress_file=progress_file)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Compare mode — spawns two subprocesses
# ---------------------------------------------------------------------------

def _spawn_json(extra_env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Run this script with --json in a subprocess, return parsed results."""
    env = {**os.environ, **(extra_env or {})}
    proc = subprocess.run(
        [sys.executable, __file__, "--json"],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("subprocess stderr:", proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Subprocess exited with code {proc.returncode}")
    return json.loads(proc.stdout)


def print_compare_table(
    pure: list[dict[str, Any]],
    jit: list[dict[str, Any]],
) -> None:
    stages = ["weld", "simplify", "merge", "reorder", "2opt", "or_opt", "total"]
    workload_names = [r["label"] for r in pure]

    print()
    print("=" * 72)
    print("BENCHMARK COMPARISON: pure Python vs JIT (numba)")
    print("=" * 72)

    for i, name in enumerate(workload_names):
        p = pure[i]
        j = jit[i]
        n = p["counts"]["input"]
        print()
        print(f"  Workload: {name}  ({n} paths)")
        print(f"  {'Stage':10s}  {'pure (s)':>10s}  {'jit (s)':>10s}  {'speedup':>10s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for stage in stages:
            pt = p["timings"].get(stage, 0.0)
            jt = j["timings"].get(stage, 0.0)
            speedup = pt / jt if jt > 1e-6 else float("inf")
            marker = " ◀ JIT" if speedup >= 2.0 and stage != "total" else ""
            print(
                f"  {stage:10s}  {pt:10.3f}  {jt:10.3f}  {speedup:9.2f}x{marker}"
            )

    print()
    print("=" * 72)
    print("  JIT cold-start note: first run in a fresh Python process adds ~2–5 s")
    print("  for numba compilation. The 'cache=True' decorator persists compiled")
    print("  code to disk so subsequent runs skip recompilation entirely.")
    print("=" * 72)
    print()

    # Recommendation based on map 2000-path case
    map_pure_idx = next(i for i, r in enumerate(pure) if r["label"] == "map")
    map_jit_idx  = next(i for i, r in enumerate(jit)  if r["label"] == "map")
    map_speedup = (
        pure[map_pure_idx]["timings"]["total"] /
        jit[map_jit_idx]["timings"]["total"]
        if jit[map_jit_idx]["timings"]["total"] > 1e-6
        else 0.0
    )
    if map_speedup >= 5.0:
        print(
            f"  RECOMMENDATION: {map_speedup:.1f}× speedup on the map workload — "
            "installing [fast] is recommended\n"
            "  for users plotting maps or dense generative work.\n"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark plottter processing pipeline on synthetic workloads."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--compare",
        action="store_true",
        help="Compare pure-Python vs JIT by running two subprocesses.",
    )
    group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (used internally by --compare).",
    )
    args = parser.parse_args()

    if args.compare:
        print("Running pure-Python mode (NUMBA_DISABLE_JIT=1)…", flush=True)
        pure = _spawn_json({"NUMBA_DISABLE_JIT": "1"})
        if not JIT_ENABLED:
            print(
                "\nWARNING: numba is not installed — JIT mode unavailable.\n"
                "Install with:  pip install -e \".[fast]\"\n",
                file=sys.stderr,
            )
            print("Pure-Python results only:\n")
            for r in pure:
                print(f"  {r['label']:10s}  {r['timings']['total']:.3f} s")
            return 0
        print("Running JIT mode…", flush=True)
        jit = _spawn_json()
        print_compare_table(pure, jit)
        return 0

    # Single-mode run (normal or --json)
    if not args.json:
        print()
        print("=" * 60)
        print(f"  plottter processing benchmark  [{MODE_LABEL}]")
        print("=" * 60)
        print()
        results = run_all()
    else:
        # In JSON mode: route progress to stderr, keep stdout clean for JSON
        results = run_all(progress_file=sys.stderr)
        print(json.dumps(results))
        return 0

    print("=" * 60)
    if not JIT_ENABLED:
        print("  numba not installed — run with [fast] extra for JIT mode:")
        print('  pip install -e ".[fast]"')
    elif _NUMBA_DISABLED:
        print("  JIT disabled via NUMBA_DISABLE_JIT=1")
    else:
        print("  JIT active — first run may include ~2–5 s compilation overhead")
        print('  Use --compare to see pure-Python vs JIT side-by-side')
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
