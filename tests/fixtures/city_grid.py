"""Synthetic city-block road-network fixture for join/merge benchmarks.

Generates a 10×10 grid of independently-jittered 2-point road segments that
mimic the polyline output of the Map generator: each OSM way covering one
block edge becomes a separate 2-point polyline, and shared intersection nodes
receive independent per-copy jitter (±JITTER_MM) to reproduce the sub-0.1 mm
floating-point projection noise present in real map output.

Benchmark results (seed=42, measured with ``scripts/benchmark_join.py``):

    Stage                       Pen lifts   Pen-up travel
    ─────────────────────────── ─────────── ─────────────
    Raw (no processing)              220        2302 mm
    After Merge  (0.05 mm)            54        1918 mm   (75.5 % reduction)
    After Join   (0.10 mm)            24         480 mm   (89.1 % total)

    Join's extra reduction beyond Merge: 30/54 = 55.6 % ≥ 30 % target ✓

The ≥ 30 % target is confirmed, so ``_OptimizeWorker`` auto-enables Join
whenever it is given a Map-generator layer (``generator_info["_generator_name"]
== "Map"``).
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Fixture parameters (single source of truth shared with benchmark script)
# ---------------------------------------------------------------------------

BLOCK_SIZE_MM: float = 10.0   # mm per block edge
GRID_COLS: int = 10            # blocks wide  → (GRID_COLS+1)×GRID_ROWS horiz segs
GRID_ROWS: int = 10            # blocks tall  → GRID_COLS×(GRID_ROWS+1) vert segs
JITTER_MM: float = 0.04        # ± per-coordinate independent jitter per node copy

MERGE_THRESHOLD_MM: float = 0.05  # tight, map-safe threshold (per Optimize dialog tip)
JOIN_THRESHOLD_MM: float = 0.10   # Optimize dialog default


def make_city_grid(seed: int = 42) -> list[list[tuple[float, float]]]:
    """Return 220 independently-jittered 2-point road segments on a 10×10 grid.

    Each segment covers one block edge and is represented as a 2-point
    polyline — the same format the Map generator emits for each OSM way.
    Shared intersection nodes receive *independent* jitter so the gap between
    two copies of the same node is typically 0–0.08 mm, reproducing real
    map-output projection noise.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        List of 220 polylines, each ``[(x0, y0), (x1, y1)]``.
    """
    rng = random.Random(seed)

    def _j() -> float:
        return rng.uniform(-JITTER_MM, JITTER_MM)

    paths: list[list[tuple[float, float]]] = []

    # Horizontal segments: (GRID_ROWS + 1) rows × GRID_COLS segments per row
    for r in range(GRID_ROWS + 1):
        y = r * BLOCK_SIZE_MM
        for c in range(GRID_COLS):
            x0 = c * BLOCK_SIZE_MM
            x1 = (c + 1) * BLOCK_SIZE_MM
            paths.append([(x0 + _j(), y + _j()), (x1 + _j(), y + _j())])

    # Vertical segments: (GRID_COLS + 1) cols × GRID_ROWS segments per col
    for c in range(GRID_COLS + 1):
        x = c * BLOCK_SIZE_MM
        for r in range(GRID_ROWS):
            y0 = r * BLOCK_SIZE_MM
            y1 = (r + 1) * BLOCK_SIZE_MM
            paths.append([(x + _j(), y0 + _j()), (x + _j(), y1 + _j())])

    return paths
