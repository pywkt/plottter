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

### XDoG (Extended Difference of Gaussians)

**Algorithm:** XDoG

An advanced edge detection method that produces stylized, illustration-like line art. XDoG extends the standard Difference of Gaussians with a sharpening step that creates crisp, ink-like strokes with controllable edge thickness.

| Parameter | Description |
|-----------|-------------|
| `sigma` | Edge scale — controls how wide edges are detected (0.3–3.0, default 1.0) |
| `k` | DoG ratio — ratio between the two Gaussian kernels (1.1–5.0, default 1.6) |
| `phi` | Sharpness of the edge transition — higher values produce crisper edges (1–200, default 100) |
| `epsilon` | Black-level offset — negative values reveal more edges (-0.5–0.5, default 0.01) |
| `min_contour_length` | Minimum points to keep a contour (removes noise) |
| `simplify_tolerance_mm` | RDP simplification tolerance |
| `close_gaps_mm` | Maximum gap distance to bridge between endpoints |
| `centerline` | Thin edge bands to single-pixel centerlines for cleaner line art |
| `smooth_curves` | Fit cubic Bezier curves for smooth, plotter-friendly output |

**Presets:**
- **Pencil Sketch** — thin centerlines with merged fragments, mimics pencil drawing
- **Woodcut** — bold, high-contrast edges with strong sharpening
- **Soft Charcoal** — soft, diffuse edges with low sharpness

**Best for:** Stylized portraits, illustrations, ink-drawing effects. More artistic than standard Canny edge detection.

**Tips:**
- Start with the **Pencil Sketch** preset and adjust `sigma` for edge thickness
- Enable `centerline` for single-stroke line art (best for pen plotters)
- Enable `smooth_curves` for Bezier output that plots more smoothly than raw contours
- Lower `epsilon` to reveal more subtle edges; raise it to keep only the strongest

---

### FDoG (Coherent Line Drawing)

**Algorithm:** FDoG

Flow-guided Difference of Gaussians — produces coherent, flowing lines that follow the image's edge structure. Based on the Edge Tangent Flow (ETF) algorithm, which aligns edges along dominant orientations, creating a hand-drawn illustration style.

| Parameter | Description |
|-----------|-------------|
| `sigma_c` | Line width scale for DoG Gaussians (0.5–3.0, default 1.0) |
| `sigma_m` | ETF smoothing scale — controls flow smoothing width (0.5–6.0, default 3.0) |
| `rho` | DoG ratio — higher values capture wider edge bands (1.1–5.0, default 3.0) |
| `etf_iterations` | Edge Tangent Flow refinement passes — more passes = smoother flow (1–10, default 3) |
| `fdog_iterations` | FDoG filter + threshold passes (1–5, default 1) |
| `min_contour_length` | Minimum contour length to keep |
| `simplify_tolerance_mm` | RDP simplification |
| `close_gaps_mm` | Gap bridging threshold |
| `smooth_iterations` | Chaikin smoothing passes for organic strokes |
| `centerline` | Single-pixel centerline tracing |
| `smooth_curves` | Bezier curve fitting |

**Presets:**
- **Coherent Lines** — balanced flow-aligned edges
- **Fine Lines** — thin, detailed strokes with tighter Gaussians
- **Bold Strokes** — wide, prominent edges with heavy ETF smoothing

**Best for:** Portraits, figures, and images with strong directional features. Produces more natural, flowing lines than XDoG or Canny.

**Tips:**
- Increase `etf_iterations` (3–5) for smoother, more coherent line directions
- Use `smooth_iterations = 1` for organic-feeling strokes
- FDoG is slower than XDoG due to the ETF computation — start with small images to tune parameters

---

### Hedcut (Portrait Style)

**Algorithm:** Hedcut

Combines three techniques to create Wall Street Journal-style stipple portraits: **edge outlines** for structure, **midtone stipple dots** for tone, and **shadow hatching** for depth.

**Tonal Zones:**

| Parameter | Description |
|-----------|-------------|
| `highlight_threshold` | Brightness above which pixels are highlights — no dots or hatching (128–255, default 200) |
| `shadow_threshold` | Brightness below which pixels are shadows — hatching region (0–128, default 80) |

**Edge Outlines:**

| Parameter | Description |
|-----------|-------------|
| `edge_method` | Edge detection algorithm: XDoG, FDoG, or Canny |
| `edge_sigma` | Edge scale parameter (0.3–3.0) |
| `edge_min_len` | Minimum contour length |
| `edge_simplify_mm` | RDP tolerance for edge simplification |

**Midtone Stipple:**

| Parameter | Description |
|-----------|-------------|
| `stipple_points` | Number of stipple dots (500–30,000, default 5,000) |
| `stipple_iterations` | Lloyd relaxation iterations — more = more even distribution (5–50, default 20) |
| `min_dot_size_mm` | Dot radius in bright areas (0.1–2.0, default 0.2) |
| `max_dot_size_mm` | Dot radius in dark areas (0.2–4.0, default 0.8) |
| `dot_size_gamma` | Tone curve for dot sizing (0.5–3.0, default 1.0) |
| `dot_style` | Outline (single ring) or Filled (concentric rings for solid dots) |
| `pen_width_mm` | Spacing for filled-dot concentric rings |

**Shadow Hatching:**

| Parameter | Description |
|-----------|-------------|
| `hatch_angle` | Direction of shadow hatching lines (0–180, default 45) |
| `hatch_spacing_mm` | Line spacing (0.2–3.0, default 0.5) |
| `cross_hatch_shadows` | Add perpendicular hatching in the deepest shadows |

**Presets:**
- **Classic WSJ** — standard hedcut with XDoG edges, 5,000 dots, 45deg hatching
- **Dense Detail** — 15,000 dots with FDoG edges and cross-hatching
- **Minimal** — 2,000 dots, sparse and fast to plot
- **Bold Hedcut** — filled dots with wide size range and cross-hatching
- **Fine Stipple** — 15,000 outline dots with narrow size range
- **WSJ Portrait** — filled dots tuned for portraits with FDoG edges

**Best for:** Portraits, faces, editorial illustrations. Produces the classic newspaper stipple-portrait look.

**Tips:**
- Use **Filled** dot style for bolder, more visible dots at larger sizes
- Increase `stipple_points` to 10,000–15,000 for detailed portraits
- Adjust `highlight_threshold` and `shadow_threshold` to control which areas get dots vs. hatching
- Pre-process with `contrast +10` and `blur 1.0` for cleaner tonal zones

---

### Scanline Halftone

**Algorithm:** Scanline Halftone

Draws parallel scan lines across the image with line thickness varying based on local brightness. Dark areas get multiple closely-spaced parallel strokes (appearing thick), bright areas get a single thin line or nothing. Creates a classic line halftone / engraving-style effect.

| Parameter | Description |
|-----------|-------------|
| `line_spacing_mm` | Vertical distance between scan lines (0.5–10.0, default 2.0) |
| `angle_deg` | Rotation angle of scan lines — 0 is horizontal, 90 is vertical |
| `max_thickness` | Maximum parallel offset lines per side in darkest areas (0–10, default 4) |
| `pen_width_mm` | Spacing between parallel offset lines (0.1–1.0, default 0.3) |
| `sample_interval_mm` | How often brightness is sampled along each line (0.5–5.0, default 1.0) |
| `tone_gamma` | Tone curve — higher values emphasize dark areas (0.5–3.0, default 1.5) |
| `skip_white` | Remove scan line segments in very bright areas (default on) |
| `white_threshold` | Brightness above which lines are removed when skip_white is on |
| `edge_sensitivity` | Reduce line thickness near detected edges — 0 disables, 1 full effect |

**Presets:**
- **Newspaper** — horizontal lines, moderate spacing, classic print look
- **Engraving** — angled lines (15deg), tight spacing, high detail with edge preservation
- **Bold Poster** — wide spacing, high contrast, dramatic thick/thin bands
- **Fine Detail** — dense, subtle reproduction for detailed photos
- **Cross Scan** — angled at 30deg; duplicate the layer at -30deg for a cross-hatch effect
- **Vertical Blinds** — vertical lines for a distinctive look

**Best for:** Portraits, photos, any image where you want a recognizable halftone/engraving style.

**Tips:**
- `max_thickness` controls the tonal range — higher = darker darks
- Set `pen_width_mm` to match your actual pen width for accurate thickness
- Enable `edge_sensitivity` (0.3–0.5) to sharpen detail around eyes, text, and object boundaries
- For a cross-hatch halftone, create two layers at opposing angles (e.g. 30deg and -30deg)

---

### Circular Scribble

**Algorithm:** Circular Scribble

Generates tone-aware circular scribble marks across the image. Darker areas receive denser, tighter scribbles; lighter areas get fewer, larger ones. Based on the Pacific Graphics 2015 paper on Circular Scribble Art.

| Parameter | Description |
|-----------|-------------|
| `seed` | Random seed for reproducibility |
| `min_spacing_px` | Exclusion radius in dark areas — controls maximum density (1–50, default 2) |
| `max_spacing_px` | Exclusion radius in bright areas — controls minimum density (5–100, default 20) |
| `scribble_radius_mm` | Base radius of circular scribbles (0.5–20, default 3.0) |
| `scribble_angle_variance` | Random angle variation per scribble in degrees (0–360, default 45) |
| `brightness_influence` | How much image brightness affects scribble radius (0–1, default 0.5) |

**Best for:** Artistic, hand-drawn-looking image reproductions. Produces a distinctive scribble texture.

**Tips:**
- Decrease `min_spacing_px` for denser coverage in dark areas
- Increase `scribble_radius_mm` for more visible, expressive scribble marks
- Use preprocessing blur (2–4) to smooth out noise before scribbling

---

## After Generation

After generating paths, the standard post-processing tools are available:

- **Tools › Optimize Current Layer** — reorder paths to minimize pen travel
- **Tools › Simplify Paths** — reduce point count with RDP simplification
- **Tools › Merge Nearby Paths** — join endpoints within a threshold distance
- **Tools › Clip to Canvas** — remove paths outside the drawing margin
- **Tools › Weld Overlapping Paths** — remove duplicate overlapping segments
- **Tools › Apply Brush to Layer** — apply stippled, sketchy, or calligraphic brush effects

See [Export & Plotting](export-and-plotting.md) to export as SVG, HPGL, G-code, or Mural format.
