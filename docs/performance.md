# Performance Guide

This page covers the path-processing pipeline performance — how to measure it,
what to expect on typical workloads, and when the optional `[fast]` (numba JIT)
extra is worth installing.

If your laptop is just plain slow, see [Remote Optimization](remote-optimization.md)
for offloading the Optimize pipeline to a faster machine on your network
(works over Tailscale SSH with no open ports on the remote).

---

## The processing pipeline

The **Optimize** action in Plottter runs a multi-stage pipeline on every layer's
paths.  The stages mirror `_OptimizeWorker` in `gui/main_window/workers.py`:

| Stage | Purpose | JIT-accelerated? |
|-------|---------|-----------------|
| weld | Remove duplicate / overlapping segments | Yes (`_segments_match_jit`) |
| simplify | Ramer–Douglas–Peucker path simplification | No |
| merge | Snap nearby endpoints together | Yes (`_find_and_snap`) |
| reorder | Nearest-neighbour path ordering | No |
| 2-opt | 2-opt route-improvement passes | No |
| or-opt | Or-opt relocate passes | No |

The two JIT-accelerated functions are the innermost loops of `weld` and `merge`.
All other stages are pure Python backed by SciPy KD-trees.

---

## Benchmark results

Timings below were recorded with `scripts/benchmark_processing.py` on three
synthetic workloads designed to represent real-world generator output.

### Workloads

| Name | Paths | Description |
|------|-------|-------------|
| **stipple** | 500 | Short 2–3-point strokes scattered across a 200 × 200 mm canvas (Voronoi stippling, dot art) |
| **map** | 2 000 | Road-network grid with sub-0.1 mm endpoint jitter and duplicate segments (MapGenerator output) |
| **flow** | 5 000 | Dense short curved paths from a sine-based flow field (flow-image / squiggle generators) |

### Timings — pure Python (no numba)

| Stage | stipple 500 | map 2 000 | flow 5 000 |
|-------|------------|----------|-----------|
| weld | 0.002 s | 0.008 s | 0.050 s |
| simplify | < 0.001 s | < 0.001 s | 0.005 s |
| merge | 0.010 s | 0.100 s | 0.133 s |
| reorder | 0.076 s | 0.209 s | 1.354 s |
| 2-opt | 0.159 s | 0.142 s | 5.201 s |
| or-opt | 0.540 s | 2.425 s | 45.956 s |
| **total** | **0.786 s** | **2.884 s** | **52.699 s** |

### Timings — JIT (numba installed)

| Stage | stipple 500 | map 2 000 | flow 5 000 |
|-------|------------|----------|-----------|
| weld | 0.002 s | 0.011 s | 0.031 s |
| simplify | < 0.001 s | < 0.001 s | 0.005 s |
| merge | 0.116 s | 0.110 s | 0.144 s |
| reorder | 0.118 s | 0.221 s | 1.484 s |
| 2-opt | 0.169 s | 0.155 s | 5.639 s |
| or-opt | 0.590 s | 2.682 s | 46.438 s |
| **total** | **0.995 s** | **3.180 s** | **53.741 s** |

### Speedup summary (pure Python ÷ JIT)

| Workload | Total speedup |
|----------|--------------|
| stipple 500 | 0.79× |
| map 2 000 | 0.91× |
| flow 5 000 | 0.98× |

**Interpretation:** On these workloads the JIT path provides no meaningful
speedup.  The dominant cost on every workload is `or-opt`, which is pure Python
and accounts for ~70–90 % of total runtime.  The JIT-compiled helpers
(`weld`'s `_segments_match_jit` and `merge`'s `_find_and_snap`) each contribute
less than 3 % of total time even on the largest input, so the compilation
overhead is visible but the total difference is noise.

The slight regression visible in the `stipple` `merge` column (0.010 s → 0.116 s)
is expected: constructing the NumPy arrays required by the JIT'd function
dominates when the path count is small.  At 500 paths the absolute time is still
well under 0.2 s, so it does not affect the user experience.

---

## JIT cold-start note

When numba is installed the JIT'd functions are compiled **once** per Python
process on first call.  With `cache=True` (the default in this project) the
compiled machine code is written to `__pycache__` and reused on subsequent
runs, so cold-start overhead is paid only once.

- **First run in a fresh Python process**: adds approximately **2–5 seconds**
  while numba compiles `_segments_match_jit` and `_find_and_snap`.
- **Subsequent runs** (same process, or any process with a warm cache): no
  compilation overhead at all.

---

## Should I install `[fast]`?

Based on the measurements above, the `[fast]` extra does **not** provide a
significant speedup on the current workloads.  The bottleneck is `or-opt` (pure
Python), not the two JIT'd inner functions.

The map-2000-path speedup is **0.91×** — well below the 5× threshold at which
installing `[fast]` would be recommended for regular use.

Install `[fast]` only if you are profiling or experimenting with JIT-accelerated
code paths:

```bash
pip install -e ".[fast]"
```

Verify JIT is active at runtime:

```python
from plottter.processing._jit import JIT_ENABLED
print(JIT_ENABLED)  # True when numba is installed and NUMBA_DISABLE_JIT is unset
```

To force pure-Python mode even with numba installed, set the environment variable
`NUMBA_DISABLE_JIT=1` before launching Plottter or the benchmark script.

---

## Running the benchmark yourself

```bash
# Single-mode run (auto-detects JIT availability):
python scripts/benchmark_processing.py

# Side-by-side comparison (spawns two subprocesses):
python scripts/benchmark_processing.py --compare

# Force pure-Python mode:
NUMBA_DISABLE_JIT=1 python scripts/benchmark_processing.py

# Machine-readable JSON output:
python scripts/benchmark_processing.py --json
```

Results are expected to reproduce within ±20 % on the same hardware.  Variation
beyond that typically indicates background CPU load or thermal throttling.

---

## Reproducing the numbers above

The benchmark was run with:

- Python 3.12, numba 0.59+
- `pip install -e ".[fast,dev]"`
- No other significant processes running
- Workload seed fixed at 42 for reproducibility

---

## Canvas rendering

This section tracks the cost of a single canvas repaint — the work behind pan,
zoom, drag, and animation frames. The driver is `tools/bench_canvas.py`: it
builds a real `ProjectController` + `CanvasWidget` (1400×900), fills the project
with a seeded random-walk scene over an A2 sheet (`--paths N --pts M`, RNG seed
42), and times `widget.render(QImage)` over `--frames` frames (default 5),
reporting min/mean ms per frame.

```bash
# Defaults to QT_QPA_PLATFORM=offscreen; --json for machine-readable output.
python tools/bench_canvas.py --paths 10000 --pts 12
python tools/bench_canvas.py --paths 38000 --pts 12 --json
python tools/bench_canvas.py --paths 38000 --pts 12 --no-cache   # uncached path
```

### Baseline — pre-cache (Phase 164)

These are the **legacy per-segment `drawLine` numbers** captured before the
layer path cache and scene pixmap cache land (spec `canvas-performance.md`
§3–§7). They are the targets later phases must beat — a representative frame at
the project's real working scale costs hundreds of ms to over a second:

| Scene          | min (ms) | mean (ms) |
| -------------- | -------: | --------: |
| 10000 × 12     |      332 |       343 |
| 38000 × 12     |     1323 |      1344 |

> Absolute ms are hardware-dependent (these were measured offscreen on the dev
> loop machine); reproduce within ±20 % on comparable hardware. The shape — a
> roughly linear ~3.9× jump from 10k to 38k paths, all of it in the per-point
> Python loops of `_draw_layer` — is the part that matters. Target after the
> rewrite: gesture frames under ~5 ms (cached blit), crisp re-render under
> ~60 ms at 38k paths.

### After path cache (Phase 165)

These are measured with the `LayerPathCache` + travel-path cache landed (spec
`canvas-performance.md` §6, §9). Each layer's mm-space `QPainterPath` is built
once and reused across frames; the renderer now hands one `drawPath` per layer
to Qt's C++ stroker instead of rebuilding the geometry every repaint. `cache on`
is the default paint path; `cache off` is the `PLOTTTER_NO_CANVAS_CACHE=1`
bypass (`--no-cache`), which rebuilds the path fresh per frame for an
apples-to-apples comparison. Min/mean over 5 frames; the first frame is the
cache-build warmup, so `min` is the steady-state floor:

| Scene      | cache off min/mean (ms) | cache on min/mean (ms) | vs §164 baseline (min) |
| ---------- | ----------------------: | ---------------------: | ---------------------: |
| 10000 × 12 |             103 / 108   |          59 / 69       |                  5.6×  |
| 38000 × 12 |             347 / 361   |         228 / 303      |                  5.8×  |

> The ≥5× win at 38k vs the §164 per-segment baseline (1323 → 228 ms) is met.
> Most of it comes from the `drawPath` migration itself — even `cache off` is
> already ~3.8× faster than the legacy per-`drawLine` loop — and the layer-path
> cache adds the remaining **~1.5–1.75×** by skipping the per-frame path
> rebuild (38k: 347 → 228 ms; 10k: 103 → 59 ms). The cached/uncached ratio
> floors around ~0.6 because the surviving cost is the C++ stroke, which caching
> doesn't touch; the sub-5 ms gesture frames in the §164 target depend on the
> later scene-pixmap blit cache, not the path cache alone. Numbers are
> hardware-dependent (offscreen, dev-loop machine); reproduce the *shape* within
> ±20 %. The `tests/test_canvas_path_cache.py::TestRenderCachePerfSmoke` test
> guards the cache-on-beats-cache-off margin at a CI-friendly 5000 × 12.

### After scene pixmap cache (Phase 166)

The scene pixmap cache (`canvas-performance.md` §7) bakes the static content —
grid, registration marks, every visible layer's path, travel lines — into one
slop-padded `QPixmap` and blits it with a single `drawPixmap` per frame. The
frame splits into two cost regimes:

- **Gesture frame (blit)** — a pan within the 0.5-viewport slop, a soft scaled
  zoom mid-gesture, or any repaint where the static content is unchanged: just
  the `drawPixmap`. This is the number the §164 target ("gesture frames under
  ~5 ms") is about.
- **Crisp frame (rebuild)** — the pixmap is re-baked: on a `scene_revision` bump
  (§7.4), a pan past the slop, a DPR change, or the 120 ms post-zoom idle timer.
  This re-strokes every layer path once into the pixmap. Measured here with the
  **path cache already warm** (the §6 `QPainterPath`s reused), so it isolates the
  stroke-into-pixmap cost — not the one-time cold path build, which the very
  first paint of a freshly loaded project folds in on top.

Driver: `bench_canvas.py` renders the same view repeatedly, so `min` over frames
after the first is the steady-state blit; the crisp column re-bakes each frame by
bumping `scene_revision`. `no-cache` is the `--no-cache` (`PLOTTTER_NO_CANVAS_CACHE=1`)
full per-frame draw, re-measured on this machine for an apples-to-apples baseline.
Min/mean ms over 8 frames:

| Scene      | no-cache min/mean | crisp rebuild min/mean | gesture blit min/mean | blit vs no-cache |
| ---------- | ----------------: | ---------------------: | --------------------: | ---------------: |
| 10000 × 12 |      92.6 / 105.1 |          31.2 / 34.4   |          2.24 / 2.86  |           ~41×   |
| 38000 × 12 |     387.3 / 389.9 |          99.7 / 104.3  |          2.16 / 2.59  |          ~179×   |

> **Gesture frames hit the target.** A blit is ~2.2 ms at *both* scales — flat in
> path count, since it copies one pixmap regardless of how many paths it baked —
> comfortably under the §164 ~5 ms goal (387 → 2.2 ms at 38k, ~179×). The crisp
> rebuild is path-count bound (it re-strokes everything once): ~100 ms at 38k on
> this machine, ~3.9× faster than the uncached full draw because the geometry is
> reused and only the C++ stroke runs. That clears the §164 ~60 ms crisp target
> at 10k (31 ms) but not at 38k (100 ms) on this (slower) dev-loop machine — the
> crisp cost is the surviving stroke, which the pixmap cache cannot remove, only
> defer off the gesture path; the 120 ms idle timer keeps it off-screen during
> interaction so the user sees blits, not rebuilds. Numbers are hardware-dependent
> (offscreen, dev-loop machine); reproduce the *shape* — flat blit, path-bound
> crisp — within ±20 %.

### Antialiasing during gestures — not toggled (Phase 168)

The source document's §6 ("render-hints toggling") proposed turning antialiasing
**off** during pan/zoom/drag gestures and restoring it on idle, on the theory
that AA is a per-frame rasterization cost worth shedding while the view is
moving (the findings doc measured `drawPath` at 58 ms vs 43 ms AA-off at 38k).
The findings doc gated that work on the post-cache HUD numbers
(`canvas_improvements_findings.md` Phase 4 §1: *"only if gesture frames still
exceed budget … likely unnecessary; blits don't re-rasterize"*), and the
benchmark settles it: **AA toggling is unnecessary and is not implemented.**

Measured with `tools/bench_canvas.py` (38000 × 12, cache on) plus a simulated
pan gesture — bake the scene pixmap once, then nudge `_pan_offset` 10 px/tick for
20 ticks (all within the 0.5-viewport slop, so every frame is a pure
`drawPixmap` blit, **0 cache rebuilds**), timing each:

| Scene      | pan-gesture frame min/mean/max (ms) | budget (60 FPS) | margin |
| ---------- | ----------------------------------: | --------------: | -----: |
| 38000 × 12 |                  2.3 / 2.6 / 3.3    |           16.7  |  ~6×   |

Why toggling buys nothing here: a gesture frame is no longer a `drawPath` over
456k points — Phase 166 turned it into a single blit of an already-rasterized
pixmap. AA applies when geometry is **stroked into** that pixmap (the crisp
rebuild, which happens on idle / invalidation, *not* during the gesture), so the
expensive AA stroke is already off the gesture path by construction. Disabling
the `Antialiasing` render hint on the blit would not change the copy cost — Qt
does not re-antialias a `drawPixmap` source — it would only make the static
paper/margin/overlay chrome alias for no measurable gain. The 120 ms post-zoom
idle timer (`_zoom_idle_timer`, §7.3) that would have driven the "restore AA on
idle" half of the feature is already in place for the crisp rebuild; no second
timer or render-hint bookkeeping is added.

At 2.6 ms mean (6× under the 16.7 ms budget) there is no headroom to reclaim, so
`paintEvent` keeps `Antialiasing` on unconditionally and the source document's §6
is closed as **adopted-then-found-unnecessary**, exactly as the findings doc
predicted.

### Final canvas perf table — baseline → path cache → scene cache

The whole arc, one representative frame at each scale, as each cache landed.
Baseline is the legacy per-segment `drawLine` loop (Phase 164); "path cache" is
the steady-state cached `drawPath` frame (Phase 165, cache on, `min`); "scene
cache (gesture blit)" is the pan/zoom gesture frame served from the baked pixmap
(Phase 166, `min`). All offscreen on the dev-loop machine; reproduce the *shape*
(orders-of-magnitude collapse, blit flat in path count) within ±20 %:

| Scene      | baseline (ms) | path cache (ms) | scene cache — gesture blit (ms) | total speedup |
| ---------- | ------------: | --------------: | ------------------------------: | ------------: |
| 10000 × 12 |           332 |              59 |                            2.2  |        ~150×  |
| 38000 × 12 |          1323 |             228 |                            2.2  |        ~600×  |

> The two caches attack different costs and compound. The **path cache** removes
> the per-frame geometry rebuild (1323 → 228 ms at 38k, ~5.8×), leaving the C++
> stroke as the floor. The **scene pixmap cache** then removes the stroke from
> *gesture* frames entirely by blitting a pre-rasterized pixmap (228 → 2.2 ms,
> another ~100×), and — being a fixed-size pixmap copy — the blit is **flat in
> path count**: 2.2 ms at both 10k and 38k. End to end a 38k-path working scene
> goes from ~1.3 s/frame (under 1 FPS) to ~2.2 ms/frame (well past 60 FPS), a
> ~600× gesture-frame speedup. The surviving cost is the crisp rebuild
> (path-count bound, ~100 ms at 38k), which the 120 ms idle timer keeps off the
> interactive path so the user only ever sees blits while gesturing.
