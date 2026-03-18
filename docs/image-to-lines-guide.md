# Image-to-Lines Guide

Image-to-Lines mode converts raster images (photos, illustrations, scans) into plotter-ready
polyline paths. The workflow has two stages: **preprocessing** (adjusting the image) and
**line generation** (choosing an algorithm to trace paths).

---

## Workflow Overview

```
Load image → Adjust preprocessing → Choose algorithm → Generate → Paths on layer
```

1. Select **Image to Lines** in the mode panel
2. Click **Load Image** to pick a JPG, PNG, or WebP file
3. Adjust the preprocessing sliders — the canvas shows a live preview
4. Select an algorithm from the **Algorithm** dropdown
5. Tune algorithm parameters
6. Click **Generate**

---

## Loading an Image

Click the **Load Image** button (or use the file picker) and select a supported file:

| Format | Extension |
|--------|-----------|
| JPEG | `.jpg`, `.jpeg` |
| PNG | `.png` |
| WebP | `.webp` |
| GIF | `.gif` (first frame) |

After loading, a scaled thumbnail appears in the Settings Panel and a semi-transparent
overlay of the preprocessed grayscale image appears on the canvas behind the paths.

---

## Preprocessing Controls

Preprocessing happens before the line generation algorithm sees the image. All adjustments
update the canvas overlay in real time.

| Control | Range | Effect |
|---------|-------|--------|
| **Brightness** | −100 to +100 | Shift all pixel values uniformly |
| **Contrast** | −100 to +100 | Stretch or compress the tonal range |
| **Gamma** | 0.1 to 5.0 | Nonlinear brightness correction (>1 brightens midtones) |
| **Blur Radius** | 0 to 20 | Gaussian blur — smooths noise and edge artifacts |
| **Threshold** | 0 to 255 | Binary threshold (0 = off); converts to black and white |
| **Invert** | checkbox | Swap blacks and whites |
| **Remove Background** | checkbox + tolerance | Treat near-white pixels as transparent |
| **Crop to Canvas** | checkbox | Resize and crop the image to match the canvas aspect ratio |

### Tips for preprocessing

- Start with a **contrast boost** (+20 to +40) to make edges crisper before edge detection
- Use **blur** (radius 1–3) before stippling to reduce texture noise in photos
- **Invert** is useful when your image has light subjects on dark backgrounds — plotters draw
  dark lines, so you usually want dark areas in the image to map to dense lines
- **Threshold** produces the cleanest input for edge detection when you want bold outlines only

---

## Line Generation Algorithms

### Edge Detection

**Algorithm:** Edge Detection (Canny)

Applies Canny edge detection to find sharp transitions in the image, then traces the
resulting contours as polylines.

| Parameter | Description |
|-----------|-------------|
| `low_threshold` | Lower hysteresis threshold for Canny (0–255, default 50) |
| `high_threshold` | Upper hysteresis threshold for Canny (0–255, default 150) |
| `min_contour_length` | Skip contours shorter than this many pixels (default 10) |
| `simplify_tolerance_mm` | RDP simplification tolerance — reduce points while preserving shape |
| `close_gaps_mm` | Connect contour endpoints within this distance |

**Best for:** Illustrations, line drawings, high-contrast photos, text.

**Tips:**
- Increase `high_threshold` to capture only the strongest edges
- Use preprocessing **blur** + high threshold for a bold, woodcut-style result
- Use **Simplify Paths** (Tools menu) after generation to reduce point count further

---

### Parallel / Cross Hatching

**Algorithm:** Hatching

Draws parallel lines whose spacing varies inversely with image brightness — denser lines
in darker areas, sparser lines in lighter areas.

| Parameter | Description |
|-----------|-------------|
| `mode` | `parallel` (one angle), `cross` (two angles), or `contour` (follow edges) |
| `angle_deg` | Angle of hatch lines (default 45°) |
| `angle2_deg` | Second angle for cross-hatching (default 135°) |
| `min_spacing_mm` | Spacing in darkest areas |
| `max_spacing_mm` | Spacing in lightest areas |
| `density_curve` | How spacing scales: `linear`, `quadratic`, or `logarithmic` |
| `line_length_mm` | Maximum length of each individual hatch segment |

**Best for:** Portraits, gradients, any image with smooth tonal transitions.

**Tips:**
- **Contour mode** follows the image's own edges — produces a woodcut / engraving look
- Cross-hatching at 45° / 135° (default) looks most natural; try 0° / 90° for a technical feel
- Use `quadratic` density curve for better tonal separation in shadows

---

### Flow Field / Squiggle

**Algorithm:** Flow Image

Two sub-modes, both driven by the image's gradient field.

**Flow mode** — streamlines follow the image gradient (Sobel/Scharr). Lines curve in the
direction of edges and pack more densely in dark areas.

**Squiggle mode** — scans the image horizontally with a sine-wave line, modulating amplitude
and frequency by per-pixel brightness.

| Parameter | Description |
|-----------|-------------|
| `mode` | `flow` or `squiggle` |
| `num_lines` | Number of flow lines or scan lines |
| `step_size_mm` | Integration step for flow lines |
| `max_steps` | Maximum steps per flow line |
| `curvature_strength` | How strongly lines deflect toward edges (flow mode) |
| `amplitude_mm` | Wave amplitude (squiggle mode) |
| `frequency` | Wave frequency (squiggle mode) |
| `seed` | Random seed for particle starting positions |

**Best for:** Photos, faces, organic subjects — produces an impressionistic feel.

---

### Stipple / TSP Dots

**Algorithm:** Stipple

Uses weighted Voronoi stippling (iterative Lloyd relaxation) to place dots proportional
to image brightness. Darker areas receive more dots.

Optionally connects all dots into a single continuous path using a nearest-neighbor
Travelling Salesman heuristic.

| Parameter | Description |
|-----------|-------------|
| `num_points` | Total dots (100–50 000, default 5 000) |
| `iterations` | Lloyd relaxation iterations (1–100, default 30) |
| `connect_tsp` | If checked, output is one connected polyline (TSP path) |
| `min_dot_spacing_mm` | Minimum distance between dots |

**Best for:** Portraits, silhouettes — produces an engraving-style stipple portrait.

**Tips:**
- 5 000–10 000 points and 20–30 iterations gives good quality with acceptable generation time
- Enable `connect_tsp` for a continuous-line portrait with no pen lifts
- Pre-blur the image (radius 2–4) before stippling for smoother dot distributions

---

### Contour Lines (Topographic)

**Algorithm:** Contour Lines

Traces isolines at evenly-spaced brightness thresholds, creating concentric rings around
image features — like a topographic map of the image's tonal landscape.

| Parameter | Description |
|-----------|-------------|
| `num_levels` | Number of contour levels (2–64, default 8) |
| `spacing` | `linear`, `logarithmic` (dense in shadows), or `quadratic` |
| `simplify_mm` | RDP simplification tolerance |
| `min_contour_px` | Skip contours shorter than this many pixels |
| `invert` | Invert input image before tracing |
| `brightness`, `contrast`, `blur_radius` | Integrated preprocessing |

**Best for:** Faces, landscapes, any image with clear tonal regions.

**Presets:**
- **Fine (16 levels)** — detailed topographic map
- **Coarse (6 levels)** — bold, graphic contour style
- **Logarithmic** — emphasizes shadow detail
- **Inverted** — emphasizes highlight detail

**Tips:**
- Increase blur before tracing to smooth contours and reduce noise rings
- Use `contrast +20` to separate tonal bands more clearly
- Layer multiple contour generations with different `num_levels` on separate layers

---

## After Generation

After generating paths, the standard post-processing tools are available:

- **Tools › Optimize Current Layer** — reorder paths to minimize pen travel
- **Tools › Simplify Paths** — reduce point count with RDP simplification
- **Tools › Merge Nearby Paths** — join endpoints within a threshold distance
- **Tools › Clip to Canvas** — remove paths outside the drawing margin

See [Export & Plotting](export-and-plotting.md) to export as SVG, HPGL, or G-code.
