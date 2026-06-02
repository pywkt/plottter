# Preview & Simulation

Plottter includes a full stroke-order animation system that simulates exactly how a
physical plotter would draw your artwork — path by path, layer by layer.

---

## Canvas Preview

The canvas always shows a live preview of all visible layers. As you generate, add, or
modify paths, the canvas updates immediately.

### Zoom and pan

| Action | How |
|--------|-----|
| Zoom in / out | Scroll wheel (0.1× to 20×) |
| Zoom in | `Ctrl+=` |
| Zoom out | `Ctrl+-` |
| Fit to window | `Ctrl+0` |
| Pan | Middle-click drag, or `Ctrl` + left-click drag |

Zoom is centered on the cursor position, so the point under your cursor stays fixed as
you zoom in or out.

### Preview pen width

Each pen's preview stroke width is configurable. Open **View › Preview Pen Width**
(or use the inline canvas control) to pick a width in mm. The **Marker** preset
emulates a thicker felt-tip marker (~1.2 mm) — useful when previewing plots that
will use thick pens, since the default thin preview can make spacing look more
generous than it'll actually plot.

### Ink Preview mode

Toggle **View › Ink Preview** to switch from the default "pen preview" rendering
(each layer drawn on top opaquely, in stacking order) to **subtractive ink
mixing** — where strokes overlap, the displayed colour darkens as if the inks
had physically mixed on paper. Yellow + cyan visibly approaches green, magenta +
yellow approaches red, and CMYK stacks correctly approach black where all four
overlap.

Use this mode when previewing:

- CMYK separations to see how the channel layers will combine on paper
- Pointillist or Paired Wave Shading output where the optical-mixing effect
  matters more than per-pen line clarity
- Any multi-pen plot where you want to spot accidental over-inking before
  committing to paper

Pen Preview remains the better choice when you want to see individual layer
strokes clearly (e.g. for path-routing debugging or sparse line work).

---

## Stroke-Order Animation

The animation panel lets you watch a simulation of the plot before committing to paper.

### Opening the animation controls

The animation toolbar appears at the bottom of the canvas. Controls:

| Control | Description |
|---------|-------------|
| **Play / Pause** | Start or pause the animation |
| **Step Forward** | Advance by one complete path |
| **Step Backward** | Go back one complete path |
| **Rewind** | Return to the first path of the first layer |
| **Speed slider** | Playback speed multiplier (0.1× to 10×) |

### How animation works

The animation uses a **distance-budget** model that simulates realistic plotter speed:

1. Each timer tick advances the pen by `plotter_speed × tick_duration × speed_multiplier` mm
2. The pen travels along the current polyline's segments, consuming the distance budget
3. When a polyline is complete, the animation lifts the pen and moves to the next path start
4. Paths are played in layer order (bottom layer first), then path order within each layer

This means animation speed is proportional to actual pen travel distance — a short path
draws quickly, a long path takes proportionally longer — matching real plotter behavior.

### Visual state during animation

| State | Appearance |
|-------|-----------|
| Completed strokes | Full opacity, in layer color |
| Current stroke | Being drawn in real time |
| Future strokes | Hidden |
| Pen position | Small colored crosshair at the current draw point |

---

## Pen-Up Travel Visualization

**View › Toggle Travel Lines** (`T`) overlays dashed grey lines showing every pen-up move
between the end of one path and the start of the next.

This visualization is critical for understanding **pen lift count** and **travel efficiency**.
Dense dashed-line networks indicate high travel overhead — run **Tools › Optimize** to
reorder paths and reduce travel.

### Travel line appearance

- **Dashed grey line** — pen-up travel move
- Rendered for all visible layers simultaneously
- Does not appear in any exported file

---

## Travel Metrics

The status bar (and the optimization result dialog after **Optimize**) shows:

| Metric | Description |
|--------|-------------|
| **Pen-down distance** | Total distance drawn (mm) |
| **Pen-up distance** | Total pen travel distance (mm) |
| **Travel efficiency %** | `pen_down / (pen_down + pen_up) × 100` |
| **Pen lift count** | Number of times the pen lifts off the paper |

A well-optimized plot has efficiency > 70–80%. Raw (unoptimized) image generator output
often starts at 30–50%.

---

## Registration Mark Preview

**View › Toggle Registration Marks** (`R`) shows corner crosshairs on the canvas.
These marks represent the registration marks that will appear in exported SVG files.

Registration marks are 3 mm arm crosshairs at `stroke-width: 0.1mm`. They appear at the
four corners (and/or center, depending on the project setting) of the canvas, just inside
the margin boundary.

Use them to:
- Align a second pen pass precisely to the first
- Verify that registration marks fit within the plottable area

---

## Paper Texture Background

**View › Paper Texture** toggles a subtle off-white (`#FAFAFA`) background on the canvas
to simulate the look of paper. This is purely cosmetic and has no effect on exported files.

The default canvas background is grey (the "table" behind the paper sheet is white by default).

---

## Pen Jitter Simulation

**View › Pen Jitter Simulation** (or via the jitter intensity dialog accessible from the
View menu) adds a configurable amount of random organic noise to the rendered preview lines.

This simulates the natural wobble of a physical pen:

- **Intensity 0** — perfectly smooth lines (default)
- **Intensity 1–5** — subtle organic variation
- **Intensity 10+** — exaggerated wobble for artistic effect

Jitter applies to the **preview only** — it does not affect the underlying polyline data
or any exported file. The actual paths remain clean for the plotter.

---

## Viewport Culling

For projects with large numbers of paths (>10 000), Plottter automatically skips
rendering paths that are entirely outside the visible viewport. This keeps the canvas
responsive at all zoom levels.

If you notice paths appearing or disappearing at zoom boundaries, this is normal culling
behavior — the paths are still present in the data.

---

## Tips for Preview Use

- Use **Fit to Window** (`Ctrl+0`) before starting the animation to see the full canvas
- Zoom in on specific areas during animation to verify detail quality
- Toggle **Travel Lines** before running Optimize to see the before-state, then again
  after to compare improvement
- Set animation **Speed** to 0.1× for dense stipple or hatching plots to see individual
  stroke order clearly
- Use **Step Forward** (`→`) to manually step through paths when looking for a specific
  problem area (e.g. a path that starts far from where the previous one ended)
