# Vectorize / Trace Bitmap Guide

The **Vectorize / Trace Bitmap** generator converts raster images into smooth
vector polylines using the [potrace](http://potrace.sourceforge.net/) algorithm
(via the optional `potracer` Python binding).  It produces clean filled-shape
outlines with smooth cubic Bézier curves flattened to plotter-friendly
polylines.

---

## Installation (opt-in)

The generator is shipped as a plugin (`plugins/vectorize_trace.py`) and
**registers automatically** at startup.  However, tracing requires an extra
package that is **not** installed by default:

```bash
pip install potracer
```

If `potracer` is absent the generator appears in the UI and accepts parameters
normally, but clicking **Generate** raises a clear error message directing you
to run the command above.  No other generator is affected.

### Licence boundary

| Component | Licence |
|-----------|---------|
| Plottter application | MIT |
| `plugins/vectorize_trace.py` plugin | MIT |
| `potracer` Python package | GPLv2 (wraps the potrace C library, also GPLv2) |

Because `potracer` is an **optional, user-installed** dependency — loaded at
generation time and never bundled with Plottter — the GPL licence does not
propagate to the rest of the codebase.  This is the same pattern used by other
optional dependencies such as `pydub` (audio import) and `numba` (JIT
acceleration).

If you need a fully MIT-clean workflow, use the built-in **Contour Lines**
generator (Image → Lines mode) which implements sub-pixel isoline extraction
without any external licence dependency.

---

## Workflow

```
Load image → Mode: Image to Lines → Algorithm: Vectorize / Trace Bitmap
         → Adjust parameters → Generate
```

1. Switch to **Image to Lines** mode.
2. Load a source image (JPG, PNG, WebP).
3. Select **Vectorize / Trace Bitmap** from the Algorithm dropdown.
4. Adjust parameters (see below).
5. Click **Generate**.

For multi-layer tonal output set **Tone Levels > 1** before generating — each
threshold level lands on its own layer automatically.

---

## Parameters

### Image Placement

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Image Fit** | `fill` | `fill` — image covers the full drawing area; `fit` — scale to fit preserving aspect ratio; `custom` — explicit width/height in mm. |
| **Width (mm)** | 200 | Output width when Image Fit is `custom`. |
| **Height (mm)** | 200 | Output height when Image Fit is `custom`. |
| **Offset X (mm)** | 0 | Horizontal offset from the centred position (`fit`/`custom` modes). |
| **Offset Y (mm)** | 0 | Vertical offset from the centred position (`fit`/`custom` modes). |

### Tracing

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Threshold** | 128 | 1–254 | Pixels *darker* than this value are treated as foreground and traced. Lower values trace only very dark regions; higher values trace more of the image. |
| **Curve Tolerance (mm)** | 0.2 | 0.01–5.0 | Bézier flattening tolerance. Smaller = smoother/more detailed polylines; larger = fewer points. |
| **Despeckle** | 2 | 0–500 | Suppress speckles and small regions (area in pixels). Increase to remove noise. |
| **Corner Sharpness** | 1.0 | 0.0–1.334 | Corner detection threshold. Lower → sharper corners; higher → rounder corners. |
| **Optimize Tolerance** | 0.2 | 0.0–1.0 | potrace's internal Bézier optimisation tolerance. Higher = fewer, coarser curves. |

### Multi-level Tonal

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Tone Levels** | 1 | 1–8 | Number of threshold levels to trace. `1` = single silhouette; `>1` = one layer per tone band, thresholds spaced evenly from `32` up to the configured **Threshold**. |

---

## Multi-level Tonal Tracing

Setting **Tone Levels > 1** enables layered tonal tracing: the generator
computes *N* evenly-spaced threshold values between a dark anchor (~32) and the
configured **Threshold**, then runs potrace once per level.  Each level is
returned as a separate `LayerSpec` and placed on its own canvas layer.

This is useful for pen-plotting portraits or photographs where you want
different pens (or hatch densities) to represent different tonal zones.

### How thresholds are distributed

Given `num_levels = N` and a configured `threshold = T`:

```
step = T // N
thresholds = [step, 2*step, …, N*step]   # sorted, duplicates removed
```

For example, `num_levels=4, threshold=200` → `[50, 100, 150, 200]`.

### Layer naming and colour

Each generated layer is named `Level N (thr=T)` and assigned a colour from a
built-in palette so the layers are visually distinct in the layer panel.  When
only one level is used the layer is named simply `Trace`.

### Example preset: Tonal Layers (4)

The built-in **Tonal Layers (4)** preset uses:

```
threshold    = 200
num_levels   = 4
turdsize     = 4   (suppress small speckles)
alphamax     = 1.0
opttolerance = 0.2
curve_tolerance_mm = 0.3
image_fit_mode = fill
```

This preset works well for portrait photos and high-contrast illustrations.

---

## Presets

| Preset | Description |
|--------|-------------|
| **Logo / Silhouette** | Clean single-threshold trace of a high-contrast image. Best for logos, icons, and bold linework. |
| **Tonal Layers (4)** | Four threshold levels for shaded tonal output — assign a different pen to each layer. |
| **Fine Detail** | Tight curve tolerance (`0.05 mm`) and no despeckle for highly detailed artwork. |

---

## Comparison with Contour Lines

| | Vectorize / Trace Bitmap | Contour Lines |
|--|--------------------------|---------------|
| Algorithm | potrace (Bézier curves → polylines) | Sub-pixel isoline extraction (OpenCV) |
| Extra dependency | `pip install potracer` (GPLv2) | None (MIT) |
| Output style | Filled-shape outlines | Isovalue contour bands |
| Multi-layer tonal | Yes (Tone Levels param) | Yes (via Levels param) |
| Best for | Logos, silhouettes, clean linework | Halftone, topographic shading |

---

## Troubleshooting

**"The potracer package is required…"**
Run `pip install potracer` in the same Python environment as Plottter.

**Jagged or staircase outlines**
Decrease **Curve Tolerance** (try `0.05–0.1 mm`) or lower **Corner Sharpness**.

**Too much noise / small speckles**
Increase **Despeckle** to 10–50.

**Multi-level layers all look identical**
Lower the **Threshold** value so the tone bands are spread across the actual
tonal range of the image, or increase the image contrast during preprocessing.
