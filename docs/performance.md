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
