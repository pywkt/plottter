# Tips & Workflows

Practical advice for getting the best results from Plottter with real plotters and
real paper.

---

## Choosing the Right Algorithm for Your Image

| Image type | Recommended algorithm | Why |
|------------|-----------------------|-----|
| Photo / portrait | Stipple + TSP | Tonal gradients, single continuous line |
| Photo / portrait | Hatching (parallel) | Classic engraving feel |
| Illustration / line art | Edge Detection | Preserves original outlines |
| Landscape / topographic | Contour Lines | Maps tonal structure naturally |
| Abstract / artistic | Flow Image (flow) | Organic, impressionistic curves |
| Typography / text | Edge Detection | Clean, high-contrast input; threshold first |
| Woodcut-style | Hatching (contour mode) | Lines follow image edges |

---

## Combining Math Art + Image Modes on Separate Layers

One of Plottter's most powerful features is layering different generation modes:

**Example: Portrait with decorative background**

1. Load a portrait photo → run Stipple generation → add to **Layer 1 (black pen)**
2. Switch to Math Art mode → generate a Flow Field → add to **Layer 2 (blue pen)**
3. Set Layer 2 opacity or visibility to review how layers interact
4. Export each layer separately for a two-pen plot

**Example: Topographic art photo**

1. Import a landscape photo
2. Run Contour Lines (8 levels) → add to **Layer 1 (black pen)**
3. Run Hatching (parallel, angle 0°, low density) on same photo → add to **Layer 2 (grey pen)**
4. Result: topographic contours with a subtle texture fill

---

## Dialing In Hatching Density

The `min_spacing_mm` and `max_spacing_mm` parameters control the tonal range:

- **Too dense everywhere:** increase `max_spacing_mm` (lighter areas are too dark)
- **Too sparse everywhere:** decrease `min_spacing_mm` (dark areas not dark enough)
- **Flat tonal range:** try `density_curve = quadratic` or `logarithmic`

A good starting point for A4 at 0.3 mm pen width:
- `min_spacing_mm = 0.5` (dark areas)
- `max_spacing_mm = 4.0` (light areas)
- `density_curve = quadratic`

For cross-hatching, the second layer is always at 90° to the first and at half density —
so the visual weight of the cross direction adds to the base hatching.

---

## Optimizing for Plot Speed

Large images with many paths can take hours to plot if not optimized:

1. **Tools › Optimize Current Layer** — always run this before plotting
2. Check **View › Toggle Travel Lines** (`T`) — see the remaining pen-up moves
3. If travel is still high, try **Tools › Merge Nearby Paths** with a larger threshold (1–2 mm)
4. For stipple paths, enable **connect_tsp** to produce a single continuous polyline (zero pen lifts)
5. For edge detection results, try increasing `close_gaps_mm` to connect more contour segments

Typical optimization improvement: 30–60% reduction in pen-up travel distance.

---

## Working with Large / Complex SVGs

If you have a very dense project (>50 000 paths):

- **Enable viewport culling** — Plottter automatically skips paths outside the visible area
- **Use layer visibility toggles** to hide layers you are not currently working with
- **Simplify first** — run Tools › Simplify Paths with tolerance 0.2–0.5 mm to reduce point counts
- **Split into multiple projects** — plot each layer from a separate saved project file

---

## Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New project |
| `Ctrl+O` | Open project |
| `Ctrl+S` | Save project |
| `Ctrl+Shift+S` | Save As |
| `Ctrl+Q` | Quit |
| `Ctrl+E` | Export current layer |
| `Ctrl+Shift+E` | Export all layers |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+=` | Zoom in |
| `Ctrl+-` | Zoom out |
| `Ctrl+0` | Fit to window |
| `Ctrl+G` | Generate |
| `Ctrl+R` | Randomize |
| `G` | Toggle grid overlay |
| `T` | Toggle pen-up travel lines |
| `R` | Toggle registration marks |

---

## Multi-Layer Registration for Physical Plotters

When plotting with multiple pens, registration between passes is critical:

1. Export each layer as a separate SVG with registration marks enabled
2. Load the first SVG into your plotter software and plot
3. **Do not remove the paper** — if using AxiDraw, use `raise_pen()` and leave the paper clamped
4. Load the second SVG (same canvas dimensions) and plot
5. The corner crosshairs from the SVG appear identically on both passes, confirming alignment

For plotters where you must remove and re-insert paper:
1. Use a physical jig (taped paper guide) to ensure consistent paper position
2. Verify alignment with a dry-run (pen up only) before lowering the pen

---

## Choosing Paper and Pens

**Paper:**
- Cartridge / Bristol board (200–300 gsm) — ideal for most pens, minimal feathering
- Smooth watercolor paper — excellent for alcohol markers and brush pens
- Tracing paper / vellum — for overlay and backlit effects

**Pen types:**
- **Pigment-based pens** (Staedtler Pigment Liner, Sakura Micron) — archival, no bleed
- **Gel pens** — smooth, but can skip; reduce speed to 15–20%
- **Fountain pens** — beautiful line quality; use flow field or smooth curve generators
- **Ballpoint** — lowest quality, but useful for very fast plotting
- **Brush pens** — avoid for fine detail; great for calligraphic flow fields

---

## Saving Project Files vs. Exporting

| Action | When to use |
|--------|------------|
| **Save (.plottter)** | Between sessions — preserves all layers, settings, and generator info |
| **Export SVG** | Before plotting — send to plotter software |
| **Export All Layers** | Multi-pen plotting — one SVG per pen |

The `.plottter` format saves generator parameters alongside paths, so you can
re-generate with different settings without re-entering all parameters.

---

## Using the Plugin System

Plottter supports custom generator plugins written in Python:

1. Create the user plugin directory: **Tools › Manage Plugins…** creates it automatically
2. The directory is: `~/.config/plottter/plugins/`
3. Drop any `.py` file into this directory that implements the `Generator` interface
4. **Tools › Manage Plugins…** → plugins are loaded and registered immediately

Example minimal plugin (`~/.config/plottter/plugins/my_circles.py`):

```python
from plottter.generators import register_generator
from plottter.generators.base import FloatParam, Generator, IntParam, Preset
from plottter.models import Canvas, Polyline
import math

@register_generator
class ConcentricCirclesGenerator(Generator):
    name = "Concentric Circles"
    category = "math"

    def get_parameters(self):
        return [
            IntParam("count", "Circle Count", min=1, max=100, step=1, default=10),
            FloatParam("spacing_mm", "Spacing (mm)", min=0.5, max=50.0, step=0.5, default=5.0),
        ]

    def get_presets(self):
        return []

    def generate(self, params, canvas, progress_callback=None):
        cx, cy = canvas.width_mm / 2, canvas.height_mm / 2
        count = params.get("count", 10)
        spacing = params.get("spacing_mm", 5.0)
        paths = []
        for i in range(1, count + 1):
            r = i * spacing
            n = max(64, int(2 * math.pi * r / 0.5))
            pts = [(cx + r * math.cos(2 * math.pi * k / n),
                    cy + r * math.sin(2 * math.pi * k / n))
                   for k in range(n + 1)]
            paths.append(pts)
        return paths
```

After saving the file, open Plottter and use **Tools › Manage Plugins…** to reload.
Your new generator appears in the Math Art generator dropdown immediately.

---

## Headless / Batch Generation (CLI Mode)

Plottter includes a CLI mode for generating art without the GUI:

```bash
plottter --generator "Parametric Curves" --preset lissajous --output out.svg
```

Available flags:

| Flag | Description |
|------|-------------|
| `--generator` | Generator name (e.g. `"Parametric Curves"`, `"Polar Curves"`, `"L-System / Fractal"`); run `--list-generators` to see all names |
| `--preset` | Preset name (case-insensitive) |
| `--output` | Output SVG file path |
| `--paper` | Paper preset: `A4` (default), `A3`, `A2`, `Letter`, `Legal`, or `Custom` (use `--width` / `--height` with `Custom`) |
| `--param key=value` | Override individual parameters |

CLI mode is useful for:
- Generating large batches of SVG variations via shell scripts
- Integrating Plottter into automated art pipelines
- Testing generator parameters without launching the GUI
