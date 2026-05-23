# Image-to-Lines Guide

Image-to-Lines mode converts raster images (photos, illustrations, scans) into plotter-ready
polyline paths. The workflow has two stages: **preprocessing** (adjusting the image) and
**line generation** (choosing an algorithm to trace paths). There are **19 image algorithms**
available, ranging from edge detection and hatching to stippling, halftone, flow-based
streamline art, and multi-layer pixel-art rendering.

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

### Line Integral Convolution (LIC)

**Algorithm:** Line Integral Convolution

Traces streamlines from a jittered seed grid along a dense vector field derived from the source
image. Each streamline follows the local vector direction, producing long sweeping brush-stroke-like
marks that follow the image structure. Darker areas are seeded more densely; brighter areas are
thinned out when density modulation is enabled.

Three **vector field modes** control how the flow direction is derived from the image:

| Mode | Behaviour |
|------|-----------|
| `gradient` | Streamlines follow the Sobel brightness gradient — they cross edges perpendicularly |
| `etf` | Streamlines follow the Edge Tangent Flow (ETF) — coherent alignment along image edges for a painterly look |
| `perpendicular_gradient` | Sobel gradient rotated 90° — streamlines run parallel to edges (contour-like result) |

| Parameter | Description |
|-----------|-------------|
| `vector_field` | Flow direction source: `gradient`, `etf`, or `perpendicular_gradient` (default `etf`) |
| `kernel_length_mm` | Length of each streamline (2–50 mm, default 15) — longer = bolder strokes |
| `seed_spacing_mm` | Distance between candidate seed points (0.5–10 mm, default 2) — smaller = denser coverage |
| `separation_distance_mm` | Minimum distance between accepted streamlines (0.2–5 mm, default 0.8) |
| `step_size_mm` | Euler integration step (0.1–2 mm, default 0.5) — smaller = smoother curves, more compute |
| `density_modulation` | Remove seeds in bright image areas — thins streamlines in highlights |
| `brightness_threshold` | Brightness above which seeds are removed (0–255, default 220; only when density modulation is on) |
| `etf_kernel_radius` | Spatial scale for ETF smoothing in pixels (ETF mode only, default 5) |
| `etf_iterations` | ETF smoothing passes — more = smoother, more coherent flow (ETF mode only, default 3) |

**Presets:**
- **Default** — ETF mode, balanced streamlines with density modulation
- **Dense ETF Flow** — tighter seeds, longer strokes, high ETF smoothing for a dense painterly result
- **Contour Lines** — `perpendicular_gradient` mode; streamlines run along iso-brightness contours

**Best for:** Portraits, organic subjects — produces an impressionistic, painterly feel where lines
follow the image's structure rather than scanning across it.

**Tips:**
- ETF mode gives the most coherent, artistic results — start there and tune `etf_iterations`
- Increase `etf_iterations` to 5+ for smoother, more unified flow direction
- Use `perpendicular_gradient` for a topographic-contour look without a fixed level count
- Lower `seed_spacing_mm` and raise `separation_distance_mm` to control how many lines survive

---

### Tonal Art Maps (TAM)

**Algorithm:** Tonal Art Maps (TAM)

Constructs multiple nested tone levels using grid-jittered short strokes, then selects which
strokes to draw at each point based on local image brightness. Darker areas receive strokes from
more tone levels (dense hatching); lighter areas receive fewer. The **nesting property** guarantees
consistent tonal gradation: every stroke present at a lighter level is also present at all darker
levels, so the artwork looks correct at any brightness.

Three **orientation modes** control stroke direction:

| Mode | Behaviour |
|------|-----------|
| `fixed` | All strokes use the constant angle set by Stroke Angle |
| `gradient` | Strokes follow the Sobel brightness gradient — run across brightness transitions |
| `etf` | Strokes follow the Edge Tangent Flow — coherent alignment with image edges |

| Parameter | Description |
|-----------|-------------|
| `num_tone_levels` | Number of discrete tone levels (3–8, default 6) — more = smoother gradation |
| `stroke_length_mm` | Length of each stroke in mm (1–20, default 5) |
| `stroke_angle` | Primary stroke direction in degrees — used in `fixed` orientation mode (default 45°) |
| `cross_hatch` | Add perpendicular strokes in darker regions for a cross-hatch texture |
| `cross_hatch_threshold` | Tone-level fraction where cross-hatching begins (0 = everywhere, 1 = darkest shadows only) |
| `orientation_mode` | `fixed`, `gradient`, or `etf` |
| `stroke_density` | Strokes per mm² for the darkest tone level (0.5–5.0, default 1.5) |
| `density_curve` | Brightness-to-tone mapping: `linear`, `quadratic` (emphasises shadows), or `logarithmic` |
| `curvature` | Blend factor: 0 = straight 2-point strokes, 1 = fully curved streamlines following the field |

**Presets:**
- **Default** — fixed 45° strokes, 6 levels, linear density curve
- **Cross-Hatch Portrait** — quadratic curve, cross-hatching enabled, higher density
- **ETF Flow Strokes** — ETF orientation with slight curvature for an organic feel
- **Fine Engraving** — 8 levels, short tight strokes, logarithmic curve, cross-hatching in shadows

**Best for:** Portraits and illustrations — produces an engraving or pen-and-ink hatching style
with accurate tonal reproduction across the full brightness range.

**Tips:**
- Start with `num_tone_levels = 6` and adjust `stroke_density` to control overall darkness
- Use `quadratic` density curve for stronger tonal contrast in shadows
- Enable `cross_hatch` with `cross_hatch_threshold = 0.5–0.7` to add depth only in shadows
- ETF orientation + curvature 0.2–0.3 produces organic, hand-drawn-looking results
- Pre-blur (radius 1–2) to smooth tone boundaries before generating

---

### Dot Grid Halftone

**Algorithm:** Dot Grid Halftone

Places dots on a regular grid and sizes each dot according to local image brightness. Dark areas
receive large dots; bright areas receive small (or no) dots. Unlike the **Math Art › Dot Grid**
generator — which uses Perlin noise to modulate dot sizes for abstract, pattern-based output —
this generator directly maps pixel brightness to dot radius for faithful image reproduction.

Three **grid layouts** are available:

| Grid Type | Layout |
|-----------|--------|
| `Square` | Standard regular grid, optionally rotated |
| `Hexagonal` | Close-packed offset rows for more uniform coverage |
| `Diagonal` | Square grid rotated 45° |

Six **dot shapes** are supported: **Circle** (outline ring), **Filled Circle** (concentric rings),
**Spiral Fill** (single continuous Archimedean spiral), **Square**, **Diamond**, and **Cross**.

| Parameter | Description |
|-----------|-------------|
| `grid_type` | Grid layout: `Square`, `Hexagonal`, or `Diagonal` |
| `grid_spacing_mm` | Distance between dot centers (0.5–20 mm, default 3) |
| `grid_angle_deg` | Grid rotation angle in degrees (0–90) |
| `dot_shape` | Rendered shape: Circle, Filled Circle, Spiral Fill, Square, Diamond, Cross |
| `max_dot_radius_mm` | Dot radius in darkest areas (0.2–10 mm, default 1.4) |
| `min_dot_radius_mm` | Minimum dot radius in lightest areas — set to 0 to skip highlights entirely |
| `size_curve` | Brightness-to-radius mapping: `Area-Proportional`, `Linear`, or `Logarithmic` |
| `size_gamma` | Gamma — <1 emphasizes highlights, >1 emphasizes shadows (default 1.5) |
| `fill_line_spacing_mm` | Ring/spiral spacing for Filled Circle and Spiral Fill shapes |
| `circle_segments` | Circle polygon resolution for circle-based shapes (6–64, default 16) |

**Presets:**
- **Classic Halftone** — square grid at 45°, circle outlines
- **Newspaper** — diagonal grid, filled circles, tight spacing
- **Pop Art Dots** — hexagonal grid, large filled circles
- **Fine Detail** — square grid, small circle outlines, high resolution
- **Cross Halftone** — hexagonal grid, cross-shaped dots
- **Diamond Grid** — square grid, diamond-shaped dots

**Best for:** Photos and portraits where you want a recognizable halftone or screen-printing
aesthetic.

**Difference from Dot Grid (Math Art):** The Math Art › Dot Grid generator uses Perlin noise to
modulate dot sizes, producing abstract repeating patterns independent of any image. This generator
(`Dot Grid Halftone`) reads pixel brightness directly, mapping each dot's radius to local image
tonality for faithful reproduction of photographs.

**Tips:**
- Set `grid_spacing_mm` to 2–4× your actual pen width for clear dot separation
- Use `Area-Proportional` size curve for the most visually accurate halftone (area ∝ darkness)
- `Filled Circle` or `Spiral Fill` shapes produce bold, inked-looking dots at larger radii
- For CMYK-style color separation, generate four layers (one per channel) at slightly different
  `grid_angle_deg` values (e.g. 15°, 45°, 75°, 0°) to minimize moiré patterns

---

## Sketch

**Generator:** Sketch

An iterative "find darkest area → trace path → erase" algorithm that progressively builds up a pen sketch from an image. At each step it finds the darkest remaining area, traces a multi-segment squiggle through it, then lightens the drawn area so subsequent strokes naturally seek out unexplored dark regions. The result looks like a hand-drawn sketch with denser strokes in dark areas and sparse strokes in highlights.

### How It Works

1. **Seed selection** — samples seed points from a probability distribution weighted by remaining darkness, edge strength, and a coverage penalty (areas already drawn get deprioritized)
2. **Squiggle tracing** — from each seed, greedily traces a multi-segment path by testing candidate directions and picking the one that passes through the darkest area
3. **Erasing** — brightens the drawn area in a working copy of the image so the algorithm moves on to unexplored regions
4. **Multi-pass** — runs 3 passes with different profiles: bold strokes first (52%), then medium detail (33%), then fine refinement (15%)

### Key Parameters

| Parameter | Effect | Tuning Tips |
|-----------|--------|------------|
| `line_max_limit` | Total segment budget | Lower (2000–5000) for faster results, higher (10000+) for dense output |
| `angle_tests` | Candidate directions per step | 4 = angular crosshatch, 8–12 = fast and smooth, 16 = good quality, 36 = best but slowest |
| `line_length_px` | Length of each segment in pixels | 10–15 = fine detail, 20–30 = bolder strokes |
| `straight_bias` | Direction momentum | 0.3 = chaotic/scribble, 0.7 = flowing, 0.9 = very parallel strokes |
| `squiggle_max_deviation` | How far squiggles wander into bright areas | 12 = tight (stays in dark areas), 50 = loose (wanders freely) |
| `long_line_bias` | Random long-stroke bonus probability | 0.1 = uniform length, 0.8 = frequent dramatic long strokes |
| `directionality` | Follow natural contours | 0 = brightness-seeking only, 30–60 = follows edges/contours |
| `edge_power` | Attract strokes toward edges | 0 = none, 30–60 = moderate edge attraction |
| `max_pixel_coverage` | Max times a pixel can be inked | 1 = very sparse, 2 = normal, 4 = dense with overlapping |
| `chain_max` | Max segments chained without pen lift | 10 = many short paths, 30+ = long continuous chains |
| `tone` | Erase contrast curve | 0 = linear erasing, 0.5 = balanced, 1.0 = dark areas strongly resist erasing |
| `unsharp_amount` | Edge sharpening before processing | 0 = off, 2–3 = recommended for most images, 4+ = aggressive |
| `multi_pass` | 3-pass generation (bold → detail → refine) | On = better tonal range, Off = simpler single pass |
| `mark_mode` | Squiggle Only vs Hybrid | Hybrid competes squiggles, lines, hatch marks, and dots per seed |

### Presets

| Preset | Best For | Key Settings |
|--------|----------|-------------|
| **Quick Sketch** | Fast preview, testing settings | 3000 segs, 12 angles, single pass — fast but lower quality |
| **Portrait** | Faces and figures | Contour-following, fine detail, unsharp=3, straight_bias=0.8 |
| **Contour Portrait** | Strong edge following | High directionality=60, edge_power=30, long chains |
| **Dense Ink** | Dark, heavy coverage | 15000 segs, max_coverage=4, short strokes, low tone=0.3 |
| **Crosshatch** | Angular grid pattern | 4 directions only, long straight strokes, no chaining |
| **Loose Sketch** | Sparse, gestural look | Few long flowing strokes, high deviation=50, long_line_bias=0.8 |
| **Scribble** | Chaotic, energetic style | Low straight_bias=0.3, short steps, long wandering chains |
| **Edge Trace** | Edge emphasis | edge_power=60, tight deviation=12, unsharp=4 |
| **Hybrid Portrait** | Mixed mark types for faces | Squiggles + lines + dots, moderate settings |
| **Hybrid Dense** | Heavy hybrid coverage | Dense output with varied mark types |

### Tips

- **Start with "Quick Sketch"** to test if your image works before trying slower presets
- **Unsharp mask** is critical for low-contrast images — try `unsharp_amount=3` or higher
- **High `angle_tests`** (36) gives best quality but is ~3x slower than `angle_tests=12`
- For **portraits**, use `directionality=30` + `edge_power=15` so strokes follow facial contours
- For **very dark images**, increase `max_pixel_coverage` to 3–4 so strokes can overlap more
- **Hybrid mode** produces the most natural-looking output by mixing mark types, but is slower
- If output is **too sparse**, increase `line_density` and/or `line_max_limit`
- If output is **too uniform**, increase `tone` (0.6–0.8) so dark areas accumulate more strokes

---

## ASCII Art

**Generator:** ASCII Art

Places single-stroke Hershey font characters on a grid, with character selection based on local image brightness. Heavier characters (more strokes) go in dark areas, lighter characters in bright areas. Characters are rendered as plotter-friendly single-line strokes, not outlines.

### Key Parameters

| Parameter | Effect |
|-----------|--------|
| `cell_size_mm` | Grid cell size — smaller = more characters, finer detail |
| `min_darkness` | Skip cells brighter than this (0.1 = skip near-white) |
| `char_scale` | Character size relative to cell (0.75 = 75% of cell) |
| `rotation_mode` | Fixed angle, random, or gradient-aligned (follows edges) |

### Presets

- **Typewriter** — regular grid, no rotation
- **Scattered Type** — random rotation per character
- **Contour Text** — characters rotated to follow image edges (gradient mode)
- **Large Print** — bigger cells, bold characters

---

## Pixel Art

**Generator:** Pixel Art

Renders a raster image as a grid of discrete cells in a fixed retro palette — Game Boy, NES, SNES, PICO-8, C64, EGA/CGA, Endesga32, grayscale, and others. Each palette color is emitted on its own layer, so a multi-pen plotter can produce a full-color pixel-art print in one job.

Unlike the other image-to-lines algorithms, Pixel Art's output is *structured*: the canvas is divided into an NxM grid, every cell of one color contains the same fill strokes, and the cells themselves can be drawn as squares, diamonds, circles, rounded squares, octagons, or a staggered hex grid.

### Key Parameters

| Parameter | Effect |
|-----------|--------|
| `grid_width` | Number of pixels across (8–256). Smaller = chunkier, more abstract |
| `palette` | Color palette: `grayscale_4`, `gameboy`, `nes`, `snes`, `pico8`, `c64`, `ega`, `endesga32`, `sweetie16`, and more |
| `quantization` | How pixel colors are mapped to the palette: `nearest`, `kmeans`, `median_cut`, `octree` |
| `color_space` | `rgb` (fast) or `lab` (perceptually uniform — better for portraits) |
| `dithering` | `none`, `floyd_steinberg`, `ordered`, `atkinson` — adds patterned tone variation |
| `cell_shape` | `square` (default), `diamond`, `octagonal`, `circle`, `rounded_square`, or `hex` |
| `cell_fill_style` | How the cell interior is stroked: `solid_hatch`, `cross_hatch`, `diagonal`, `x_mark`, `outline`, `dithered_dots`, `leave_empty` |
| `fill_density` | 0–1 — line spacing within each cell |
| `cell_border` | Draw a thin outline around each cell |
| `cell_gap_mm` | Small gap between adjacent cells |

### Multi-layer output

Pixel Art is the first built-in generator to produce **multiple layers in a single run**. After clicking Generate, one layer appears per palette color used in the image — each named after its color (e.g. "NES color #04"). For a multi-pen plot, set the pen for each layer and the plotter draws them in sequence.

If you regenerate after adjusting parameters, Pixel Art **replaces** its previous layers rather than appending new ones. The replacement is tracked by a hidden `_generator_name` tag in each layer's `generator_info`, so it works regardless of which layer is currently active.

### Presets

- **Default** — grayscale_4, square cells, solid hatching
- **Game Boy Portrait** — the classic 4-shade Game Boy aesthetic with Floyd-Steinberg dithering
- **Game Boy Tiny** — chunky 32-wide grid, no dithering
- **NES Color** / **NES Crosshatched** — 54-color NES palette, solid or cross-hatched fills
- **SNES Detail** / **Genesis Detail** — higher-fidelity palettes for portrait sources
- **PICO-8 Sketch** — 16 colors with ordered (Bayer) dithering for that retro fantasy-console look
- **C64 Classic** / **CGA 4-Color** / **EGA 16-Color** — period-accurate PC palettes
- **B&W Halftone Dots** — pure black-and-white with circular cells filled with dot grids
- **B&W Hatch** / **Grayscale Fine** — monochrome with linework fills
- **Endesga Modern** / **Sweetie 16 Sketch** — modern fantasy-art palettes
- **Hex Honeycomb** — Game Boy palette in a staggered hex grid
- **Diamond Dither** — grayscale with diamond-inscribed cells
- **Outline Only** — pure cell-border line art, no fills

### Tips

- **Smaller grid + LAB color space** produces the cleanest portraits — try `grid_width=64, color_space=lab, quantization=kmeans`
- For maximum tonal range with a small palette, enable **Floyd-Steinberg dithering** — it scatters intermediate shades to give an impression of more colors
- Set `cell_gap_mm = 0.3` to visually separate adjacent cells of different pen colors, which helps the eye read the pixel structure
- For a single-pen plot, hide all but one palette layer or merge them after generation

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
