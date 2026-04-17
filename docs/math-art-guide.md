# Math Art Guide

Math Art mode generates vector paths purely from mathematical equations and algorithms —
no image input required. There are **13 math generators** covering parametric and polar
curves, tilings, flow fields, fractals, audio visualization, and more. All generators produce output in
millimeter coordinates that automatically scale to fit the canvas drawing area.

---

## Selecting a Generator

1. In the left panel, select **Math Art**
2. Choose a generator from the dropdown in the Settings Panel
3. Select a **Preset** or tune parameters manually
4. Click **Generate** (`Ctrl+G`)

The right panel updates immediately when you change the generator. Presets apply all
parameter values at once; selecting **Custom** preserves your manual edits.

---

## Parametric Curves

**Generator:** Parametric Curves

Parametric curves define `x` and `y` as functions of a parameter `t`:

```
x = f(t)
y = g(t)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `x_expr` | Expression for x(t) — e.g. `sin(3*t)` |
| `y_expr` | Expression for y(t) — e.g. `sin(4*t)` |
| `t_start` | Starting value of t (default 0) |
| `t_end` | Ending value of t (default 2π ≈ 6.2832) |
| `num_points` | Number of sampled points (100–100 000, default 5 000) |
| `scale` | Uniform scale factor applied after auto-fit |
| `rotation_deg` | Rotate the curve by this many degrees |
| `x_offset_mm`, `y_offset_mm` | Translate the curve on the canvas |

### Supported functions in expressions

`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `abs`, `sqrt`,
`log`, `log2`, `log10`, `exp`, `pow`, `floor`, `ceil`, `round`

### Supported constants

`pi` (3.14159…), `e` (2.71828…), `tau` (6.28318… = 2π)

### Presets

| Preset | x_expr | y_expr | Notes |
|--------|--------|--------|-------|
| Lissajous | `sin(3*t + pi/2)` | `sin(4*t)` | Classic ratio-3:4 figure |
| Butterfly Curve | Temple H. Fay formula | | Multi-petal butterfly |
| Spirograph (epitrochoid) | `(R+r)*cos(t) - d*cos(...)` | ... | Rolling-circles pattern |
| Hypotrochoid | | | Inner-circle spirograph |
| Farris / Mystery Curve | | | Irregular petal curve |
| Lorenz Attractor | ODE integration | x-y projection | Chaotic butterfly |

> **Note on Lorenz:** The Lorenz attractor uses SciPy ODE integration and cannot be cancelled
> mid-run. The parameter fields are read-only while Lorenz is selected.

### Tips

- Increase `num_points` for smoother curves (useful for high-frequency expressions)
- Multiply `t_end` by `pi` to use nice angular ranges: `t_end = 6*pi` gives 3 full rotations
- Combine multiple layers with different expressions for complex multi-curve compositions

---

## Polar Curves

**Generator:** Polar Curves

Polar curves define a radius `r` as a function of angle `theta`:

```
r = f(theta)
x = r * cos(theta)
y = r * sin(theta)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `r_expr` | Expression for r(theta) — e.g. `sin(4*theta)` |
| `theta_start` | Start angle (default 0) |
| `theta_end` | End angle (default 2π) |
| `num_points`, `scale`, `rotation_deg`, `x_offset_mm`, `y_offset_mm` | Same as Parametric |

### Presets

| Preset | r_expr | Notes |
|--------|--------|-------|
| Rose | `cos(n*theta)` | n petals (odd n) or 2n petals (even n) |
| Cardioid | `1 + cos(theta)` | Heart shape |
| Archimedean Spiral | `a + b*theta` | Evenly spaced spiral |
| Logarithmic Spiral | `a * exp(b*theta)` | Self-similar golden spiral |
| Limaçon | `a + b*cos(theta)` | With inner loop when b > a |

### Tips

- Rose with `n=5` gives 5 petals; `n=6` gives 12 petals
- Extend `theta_end` to `6*pi` for spirals that complete multiple turns

---

## Modular Multiplication Circles

**Generator:** Modular Multiplication

Places N evenly spaced points on a circle and draws a line from each point `p` to
point `(p × multiplier) mod N`. The resulting pattern changes dramatically with small
changes to the multiplier.

### Parameters

| Parameter | Range | Default |
|-----------|-------|---------|
| `num_points` | 2–1000 | 200 |
| `multiplier` | 0.0–500.0 | 2.0 |
| `radius_mm` | auto | auto-fit |

### Tips

- Try integer multipliers (2, 3, 4…) for symmetric patterns
- Non-integer multipliers like 2.5 or 3.14 produce more complex, asymmetric figures
- Animate through a series of multiplier values across multiple layers for a flip-book effect

---

## Perlin Noise Flow Field

**Generator:** Flow Field

Particles are placed at random positions and follow a direction determined by Perlin noise
at each location. Each particle leaves a polyline trail as it moves across the canvas.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `num_particles` | Number of particle trails (100–10 000) |
| `step_size_mm` | Distance each particle moves per step |
| `max_steps` | Maximum steps per particle |
| `noise_scale` | Frequency of the noise pattern (lower = larger features) |
| `noise_octaves` | Layers of detail added to the noise (1–8) |
| `seed` | Random seed for reproducibility |
| `angle_range` | Total angular range of the flow directions (default 2π) |

### Tips

- Low `noise_scale` (0.002–0.005) creates wide, sweeping curves
- High `noise_scale` (0.02–0.05) creates tight, chaotic patterns
- Increase `noise_octaves` for more textured, organic-looking trails
- Layer multiple flow field generations with different seeds and low opacity colors

---

## L-Systems (Fractals)

**Generator:** L-System / Fractal

L-systems use string-rewriting rules to build complex fractal patterns. A turtle graphics
interpreter converts the resulting string into drawing commands.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `axiom` | Starting string (e.g. `F`) |
| `rules` | Rewrite rules, semicolon-separated (e.g. `F=F+F-F-F+F`) |
| `iterations` | How many times to apply the rules (1–10) |
| `angle_deg` | Turning angle for `+` and `-` commands |
| `step_length_mm` | Length of each `F` / `G` draw step |
| `scale`, `rotation_deg`, `x_offset_mm`, `y_offset_mm` | Standard transforms |

### Turtle commands

| Symbol | Action |
|--------|--------|
| `F`, `G` | Draw forward one step |
| `f`, `g` | Move forward (no draw) |
| `+` | Turn right (clockwise) by `angle_deg` |
| `-` | Turn left (counter-clockwise) by `angle_deg` |
| `[` | Push position and heading (start a branch) |
| `]` | Pop position and heading (return to saved state) |

### Presets

| Preset | Angle | Iterations | Description |
|--------|-------|-----------|-------------|
| Koch Snowflake | 60° | 4 | Classic snowflake fractal |
| Sierpinski Triangle | 60° | 6 | Triangle-within-triangle recursion |
| Dragon Curve | 90° | 10 | Space-filling folded strip |
| Plant / Tree | 25° | 5 | Branching organic plant shape |
| Hilbert Curve | 90° | 5 | Space-filling curve |
| Gosper Curve | 60° | 4 | Peano-Gosper space-filling curve |

### Tips

- Keep iterations ≤ 6 for complex rules; path counts grow exponentially
- Adjust `step_length_mm` after increasing iterations to keep the output at a reasonable size
- Layer a Plant L-system on top of a Flow Field for an organic, illustrated feel

---

## Grid Patterns

**Generator:** Grid Pattern

Provides three sub-modes selected by the **mode** dropdown:

### Sine-Modulated Grid

Draws horizontal or vertical lines with a sine-wave deformation applied to each line.

| Parameter | Description |
|-----------|-------------|
| `line_count` | Number of lines |
| `line_spacing_mm` | Gap between lines |
| `amplitude_mm` | Height of the sine wave |
| `frequency` | Cycles per line |
| `phase` | Phase offset |
| `direction` | Horizontal, Vertical, or Both |

### Truchet Tiles

Divides the canvas into equal tiles and places a random quarter-circle arc in each.
Adjacent arcs form flowing organic curves.

| Parameter | Description |
|-----------|-------------|
| `tile_size_mm` | Side length of each tile |
| `seed` | Random seed for the tile pattern |

### Concentric Shapes

Draws nested copies of a shape centered on the canvas.

| Parameter | Description |
|-----------|-------------|
| `shape` | `Circle`, `Square`, or `Polygon` |
| `sides` | Number of polygon sides (3–12) |
| `spacing_mm` | Gap between each ring |
| `count` | Number of rings |

---

## Islamic Geometric Tiling

**Generator:** Grid Pattern → Islamic Tiling sub-mode

Generates star-polygon tilings in the style of Islamic geometric art. Choose from
6-point stars, 8-point stars, and 12-point stars.

| Parameter | Description |
|-----------|-------------|
| `islamic_type` | `6-Point Stars`, `8-Point Stars`, or `12-Point Stars` |
| `tile_size_mm` | Side length of each tile |
| `star_inset` | How deep the star points go (0.05–0.49) |

---

## Celtic Knots

**Generator:** Grid Pattern → Celtic Knot sub-mode

Draws an interlaced Celtic knot pattern with over/under crossings simulated by
small gap breaks.

| Parameter | Description |
|-----------|-------------|
| `knot_cols` | Number of knot columns (1–30) |
| `knot_rows` | Number of knot rows (1–30) |
| `tile_size_mm` | Size of each knot tile |
| `gap_mm` | Width of the gap at each crossing (0.1–5.0) |

---

## Text

**Generator:** Text

Renders text as plotter-ready single-stroke paths using either Hershey (stroke-based) fonts or system TrueType/OpenType fonts.

### Hershey Fonts

Hershey fonts are single-stroke vector fonts designed for plotters — each letter is drawn as a single line path with no fill.

| Parameter | Description |
|-----------|-------------|
| `text` | Text to render (supports multi-line) |
| `font_type` | Hershey or System Font |
| `hershey_font` | Simplex, Duplex, Script, or Gothic |
| `font_size_mm` | Cap height in millimeters (2–100, default 10) |
| `letter_spacing_mm` | Extra space between characters |
| `line_spacing` | Line height multiplier (0.5–3.0, default 1.2) |
| `text_align` | Left, Center, or Right |
| `rotation_deg` | Rotation angle in degrees |
| `stroke_repeat` | Trace each stroke multiple times (1–10) for a bolder look |

### System Fonts (TTF/OTF)

System fonts are rendered as outlines (and optionally filled with hatching). Browse and download Google Fonts directly from the app.

| Parameter | Description |
|-----------|-------------|
| `system_font_path` | Path to a .ttf or .otf file |
| `render_mode` | Outline, Filled, or Outline + Filled |
| `fill_type` | Hatching, Cross-hatch, or Concentric (for Filled modes) |
| `fill_spacing_mm` | Spacing between fill lines (0.2–10.0, default 0.5) |
| `fill_angle` | Hatching angle in degrees |
| `curve_tolerance_mm` | Bezier sampling tolerance |

### Tips

- **Hershey Simplex** is the fastest and most plotter-friendly — single-stroke, no fills
- Use **stroke_repeat = 2–3** for bolder text with thicker pens
- For filled text, **Concentric** fill creates the most even coverage
- Access Google Fonts via **Help › Browse Google Fonts** — downloaded fonts are cached in `~/.plottter/fonts/google/`

---

## Concentric Rings

**Generator:** Concentric Rings

Draws concentric shapes radiating from a center point, with optional Perlin noise distortion for organic, flowing effects.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `ring_count` | Number of rings (2–200, default 30) |
| `ring_spacing_mm` | Distance between rings (0.5–10.0, default 2.0) |
| `ring_shape` | Circle, Square, Triangle, Pentagon, Hexagon, or Octagon |
| `center_x_mm`, `center_y_mm` | Offset of ring center from canvas center |
| `points_per_ring` | Smoothness — more points = smoother curves (16–512, default 64) |
| `noise_scale` | Perlin noise frequency — smaller = larger noise features (0.01–1.0, default 0.05) |
| `noise_amplitude_mm` | How far noise pushes ring points from ideal position (0–20, default 3.0) |
| `noise_seed` | Random seed |
| `noise_evolution` | How much the noise pattern changes from inner to outer rings (0–1, default 0.1) |
| `amplitude_growth` | Constant, Linear, or Exponential — how amplitude changes with radius |
| `thickness_noise` | Variation in ring spacing — causes bunching/spreading (0–1, default 0) |
| `ring_gap_chance` | Probability of gaps/breaks in each ring (0–0.8, default 0) |
| `radial_lines` | Draw lines from center through all rings |
| `radial_line_count` | Number of radial connecting lines (4–64, default 8) |

### Presets

| Preset | Description |
|--------|-------------|
| Simple Circles | Clean concentric circles, no noise |
| Tight Circles | Dense rings, 1mm spacing |
| Ripples | Wavy water-ripple effect with noise |
| Organic Rings | Broken rings with gaps and noise evolution |
| Distorted Squares | Square rings distorted by noise |
| Spider Web | Circular rings with radial connecting lines |
| Topographic | Dense, heavily distorted rings — resembles a topographic map |

### Tips

- Set `noise_amplitude_mm = 0` for perfect geometric rings; increase for organic effects
- **Linear** amplitude growth makes outer rings more distorted while inner rings stay stable
- Enable `ring_gap_chance` (0.1–0.3) and `thickness_noise` together for a natural, imperfect look
- **Spider Web** preset works well layered on top of other generators

---

## Dot Grid

**Generator:** Dot Grid

Creates a grid of shapes (circles, squares, diamonds, crosses, stars, hexagons) with Perlin noise modulating size, rotation, and position.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `grid_cols`, `grid_rows` | Grid dimensions (3–100, default 20) |
| `dot_shape` | Circle, Square, Diamond, Cross, Star, or Hexagon |
| `base_size_mm` | Base shape size in mm (0.5–20, default 3.0) |
| `spacing_mm` | Grid cell spacing (1–30, default 5.0) |
| `noise_scale` | Perlin noise frequency (0.01–1.0, default 0.1) |
| `noise_strength` | How much noise affects dot size — 0 = uniform, 1 = full range |
| `noise_seed` | Random seed |
| `min_size_mm` | Minimum dot size — dots below this are skipped (creates gaps) |
| `max_size_mm` | Maximum dot size |
| `rotation_noise` | Max random rotation per dot in degrees (0–180) |
| `jitter_mm` | Random position offset from grid center (breaks rigid grid feel) |
| `filled` | Fill shapes with concentric lines for a solid appearance |
| `pen_width_mm` | Pen width for fill spacing (visible when filled is on) |

### Presets

| Preset | Description |
|--------|-------------|
| Halftone Dots | Classic halftone pattern with noise-modulated circle sizes |
| Star Field | Scattered stars with varied sizes and rotation |
| Diamond Mosaic | Filled diamonds creating a mosaic effect |
| Cross Stitch | Subtle cross pattern with slight rotation |
| Hexagon Grid | Organic hex grid with noise variation |
| Noise Landscape | Large-scale noise creates landscape-like density variation |

### Tips

- `noise_strength = 0` produces a perfectly uniform grid; increase for organic variation
- Use `jitter_mm` to break the rigid grid — values of 0.5–2.0 add subtle irregularity
- **Filled** mode with a small `pen_width_mm` (0.2–0.3) creates solid-looking shapes for plotters
- Stars and diamonds look especially good with `rotation_noise` enabled

---

## Geometric Grid

**Generator:** Geometric Grid

Creates tessellated grids (square, hexagonal, or triangular) where each cell contains shapes with noise-driven density, size, and rotation variation.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `grid_type` | Square, Hexagonal, or Triangular tessellation |
| `cell_size_mm` | Size of each grid cell (2–50, default 10.0) |
| `cell_shape` | Outline, Diagonal, Cross, Circle Inscribed, Diamond Inscribed, or Random Fill |
| `subdivisions` | Recursively subdivide dense cells (0–3 levels) |
| `cell_rotation` | Fixed rotation angle for cell contents (0–360) |
| `rotation_noise` | Random rotation variation per cell (0–90) |
| `noise_scale` | Perlin noise frequency (0.01–1.0, default 0.1) |
| `noise_seed` | Random seed |
| `density_variation` | Noise-driven cell content density — 0 = all cells drawn, 1 = many gaps |

### Presets

| Preset | Description |
|--------|-------------|
| Honeycomb | Clean hexagonal outline grid |
| Broken Tiles | Square grid with random fills and gaps |
| Triangle Mesh | Triangular tessellation with one level of subdivision |
| City Grid | Dense/sparse cross pattern resembling a city plan |
| Hex Detail | Hexagonal grid with inscribed circles and recursive subdivision |

### Tips

- **Hexagonal** grid with **Outline** cell shape creates a classic honeycomb pattern
- **Subdivisions** add detail variation — some areas have large shapes, others have fine detail
- **Random Fill** picks a different shape per cell using noise, creating organic variety
- Combine with other generators on separate layers for complex compositions

---

## Voronoi / Delaunay

**Generator:** Voronoi / Delaunay

Distributes seed points across the canvas and computes either the Voronoi diagram
(cell boundaries equidistant from neighbouring seeds) or the dual Delaunay
triangulation (triangles whose circumcircles contain no other seeds). The choice of
seed placement strategy — from uniform random to blue-noise to the phyllotaxis golden
spiral — determines the character of the output.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `render_mode` | What to draw: **Voronoi Edges**, **Delaunay Edges**, **Both**, or **Voronoi + Centroids** |
| `centroid_radius_mm` | Radius of small circle markers drawn at each seed (0.1–5.0 mm; visible in *Voronoi + Centroids* mode only) |
| `num_points` | Number of seed points for *Random* and *Phyllotaxis* methods (50–10 000, default 500) |
| `seed_method` | Seed placement strategy — see below |
| `poisson_spacing_mm` | Minimum distance between *Poisson Disk* samples (0.5–20 mm, default 3.0) |
| `grid_spacing_mm` | Cell size for *Grid Jitter* seeds (1–50 mm, default 5.0) |
| `grid_jitter` | Offset magnitude for *Grid Jitter*, as a fraction of grid spacing (0 = no jitter, 1 = full spacing, default 0.5) |
| `lloyd_iterations` | Rounds of Lloyd relaxation applied after seed generation (0 = none, higher = more regular cells) |
| `random_seed` | RNG seed for reproducibility (0–99 999, default 42) |
| `image_density` | When enabled, image brightness drives seed density — dark areas receive more seeds, bright areas fewer |
| `brightness`, `contrast`, `invert` | Preprocessing adjustments applied to the density source image |
| `x_offset_mm`, `y_offset_mm` | Translate the output on the canvas |

### Seed Strategies

| Method | Description |
|--------|-------------|
| **Random** | Uniformly random seed positions — fast and familiar |
| **Poisson Disk** | Blue-noise sampling: every seed is at least `poisson_spacing_mm` from its neighbours, giving a natural, even distribution without clustering |
| **Grid Jitter** | Regular grid with random per-point offsets — structured but not rigid |
| **Phyllotaxis** | Golden-angle spiral (sunflower seed packing) — uniform area coverage and pleasing radial symmetry |

### Lloyd Relaxation

Setting `lloyd_iterations > 0` iteratively moves each seed to the centroid of its
Voronoi cell. A few iterations (5–10) regularise cell sizes noticeably; 20+ iterations
produce near-hexagonal grids. This is independent of the initial seed method.

### Image Density Mode

Enable **Image Density** and load a source image to make seed concentration follow
image brightness. Combined with **Poisson Disk** seeding this replicates a stippling
effect: dark portrait regions attract dense, tightly packed cells while light regions
remain open.

### Presets

| Preset | Description |
|--------|-------------|
| Classic Voronoi | 500 random seeds, no relaxation |
| Relaxed Hexagons | 300 random seeds, 20 Lloyd iterations — near-hexagonal cells |
| Blue Noise Voronoi | Poisson disk seeds, 3 mm spacing |
| Delaunay Mesh | 1 000 random seeds, Delaunay edges |
| Golden Spiral | 500 phyllotaxis seeds, Voronoi edges |
| Organic Cells | 200 random seeds, 10 Lloyd iterations |
| Random Scatter | 500 random seeds, default render |
| Blue Noise | Poisson disk, 5 mm spacing |
| Jittered Grid | 8 mm grid spacing, 0.5 jitter |
| Phyllotaxis Spiral | 500 phyllotaxis seeds |

### Tips

- **Random + Lloyd 0** gives an organic, irregular feel; **Random + Lloyd 20** gives a structured, honeycomb-like look
- Combine **Voronoi Edges** on one layer with **Delaunay Edges** on another for a dual-mesh composition
- Use **Voronoi + Centroids** to visualise both the cells and their seed locations
- Increase `poisson_spacing_mm` for a sparse, airy result; decrease for a dense mosaic

---

## Penrose Tiling

**Generator:** Penrose Tiling

Generates a non-periodic Penrose tiling using Robinson triangle subdivision. Starting
from a small seed configuration, each subdivision level inflates the triangle count
by approximately φ² ≈ 2.618 (the square of the golden ratio), producing finer and
finer P3-style rhombs that tile the plane without ever repeating.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `initial_config` | Starting seed: **Sun** (10-fold decagonal wheel), **Star** (10-fold star wheel), or **Dart** (4-fold dart arrangement) |
| `subdivisions` | Subdivision depth (1–8, default 5) — each level multiplies tile count by ~2.618; depth 7–8 produces very fine tilings |
| `rotation_deg` | Rotate the entire tiling around the canvas centre (0–360°) |
| `render_mode` | What to draw — see below |
| `x_offset_mm`, `y_offset_mm` | Translate the tiling centre from canvas centre |

### Render Modes

| Mode | Description |
|------|-------------|
| **Edges Only** | Draws the rhomb outlines — clean, geometric tiling |
| **Edges + Arcs** | Rhomb outlines plus the classic arc matching-rule decorations inside each rhomb |
| **Arcs Only** | Arc decorations only — the rhombs disappear and the overlapping circular arcs reveal the hidden pentagonal symmetry of the tiling |

### Arc Decoration

Each rhomb in a Penrose tiling carries two circular arc markings — a convention due to
Conway that encodes the matching rules preventing periodic repetition. Two arcs are
inscribed in each thick (fat) rhomb and each thin (narrow) rhomb; neighbouring rhombs
that share an edge will have arcs of matching radius meeting at that edge. Plotting
**Arcs Only** produces a striking pattern of overlapping curves with 5-fold symmetry.

### Initial Configurations

| Config | Description |
|--------|-------------|
| **Sun** | 10 thick (fat) triangles arranged in a decagonal wheel — the most common starting point; produces a sun-like radial centre |
| **Star** | 10 thin triangles in a star wheel — produces a star-shaped hole at the centre |
| **Dart** | 4 thin triangles in a 4-fold arrangement — smaller seed, useful for off-centre compositions |

### Presets

| Preset | Config | Subdivisions | Mode | Description |
|--------|--------|-------------|------|-------------|
| Classic P3 | Sun | 5 | Edges Only | Standard Penrose rhomb tiling |
| Penrose Stars | Star | 4 | Edges Only | Star-centred variant |
| Arc Pattern | Sun | 5 | Arcs Only | Pure arc decoration |
| Full Decoration | Sun | 6 | Edges + Arcs | Rhombs with arc markings |
| Dense Tiling | Sun | 7 | Edges Only | Very fine tile grid |
| Dart Origin | Dart | 5 | Edges Only | 4-fold dart seed |

### Tips

- Subdivisions 5–6 is a good balance — enough detail without excessive line count
- **Arcs Only** with a light pen weight produces a delicate, lace-like pattern
- Rotate by 18° increments to align a different axis of pentagonal symmetry with the canvas
- Layer **Edges Only** (dark pen) and **Arcs Only** (light pen) as separate layers for a richly decorated composition

---

## 3D Scene

**Generator:** 3D Scene

A full 3D wireframe renderer with hidden line removal (HLR), supporting multiple primitive shapes, mesh import, lighting, and shadow effects. Each 3D layer adds one shape to a shared scene; shapes across layers occlude each other correctly.

### Shapes

| Shape | Description |
|-------|-------------|
| Sphere | Wireframe latitude/longitude lines |
| Shaded Sphere | Line density varies with light direction for shading |
| Cube | 12-edge wireframe box |
| Striped Cube | Box with face stripe detail lines |
| Cone | Cone with meridian lines |
| Cylinder | Cylinder with lateral and ring lines |
| Plane | Flat grid |
| Terrain | Heightfield from Perlin noise |
| Shard | Double pyramid shape |
| Mesh Import | Load OBJ or STL files |

### Common Parameters

| Parameter | Description |
|-----------|-------------|
| `pos_x`, `pos_y`, `pos_z` | World position (-20 to +20) |
| `rot_x`, `rot_y`, `rot_z` | Rotation in degrees |
| `uniform_scale` | Scale factor (0.01–20) |

### Camera Controls

Camera settings are shared across all 3D layers. You can orbit, zoom, and pan interactively with the mouse when a 3D layer is selected, or adjust via sliders:

- **Azimuth** (0–360) and **Elevation** (-90 to 90) — orbit angle
- **Distance** — camera distance from center
- **FOV** — field of view
- **Projection** — Perspective or Orthographic

### Render Style

The `render_style` parameter controls how surfaces are drawn:

- **Wireframe** (default) — classic wireframe edges only
- **Hatched** — fills visible surfaces with parallel hatching lines whose density varies based on light angle. Darker (shadow-facing) surfaces get denser hatching, producing a traditional pen-and-ink illustration look
- **Wireframe + Hatched** — both wireframe edges and surface hatching combined

When hatching is enabled, configure:

| Parameter | Description |
|-----------|-------------|
| `hatch_density_min` | Lines per mm for fully lit faces (0 = no lines on lit surfaces) |
| `hatch_density_max` | Lines per mm for faces in shadow |
| `hatch_angle_deg` | Direction of hatching lines |
| `hatch_cross` | Add perpendicular cross-hatching on deep shadow faces (brightness < 0.3) |

### Shadows

When shadows are enabled, the scene supports:

- **On-surface shadows** — hatching on faces pointing away from the light
- **Ground-plane shadows** — silhouette projected onto a ground plane with hatching fill
- **Shadow styles** — Thicken, Hatch, or Cross-Hatch
- **Render modes** — Combined, Shadow Only (for multi-pen plotting), Lit Only

### Mesh Import

Load OBJ and STL files for custom geometry. The renderer supports:
- Per-mesh BVH acceleration for fast hidden line removal
- Vertex deduplication and edge chaining
- Mesh decimation for very large models

### Tips

- Each 3D layer is one shape — add multiple layers for multi-object scenes
- Objects across layers correctly occlude each other via shared HLR
- Use **Orthographic** projection for a technical/architectural look
- Use **Shadow Only** render mode to plot shadows in a different pen color
- Start with low-detail primitives to iterate quickly, then increase detail for final plots

---

## Audio Waveform

**Generator:** Audio Waveform

Import an audio file and produce plotter-ready visualizations. Supports WAV natively; MP3, FLAC, OGG, and M4A require the optional `pydub` package and ffmpeg (`pip install -e ".[audio]"`). Six visualization modes are available, selected via the **Visualization** dropdown.

### Common Parameters

| Parameter | Description |
|-----------|-------------|
| `audio_file` | Path to the audio file |
| `start_sec` | Start time in the audio (0–3600 s) |
| `duration_sec` | Duration of audio to visualize (0.1–60 s, default 10) |

### Ridgeline (Joy Division)

Computes a spectrogram and draws each frequency row as a displaced horizontal line, stacked vertically. With **Hidden Line Removal** enabled, front-row peaks create opaque fills that occlude rows behind — the iconic Unknown Pleasures effect.

| Parameter | Description |
|-----------|-------------|
| `num_rows` | Number of horizontal lines (10–200, default 60) |
| `amplitude` | Vertical displacement scale (0.1–20, default 2.0) — higher values create taller peaks and more dramatic occlusion |
| `row_spacing` | Vertical gap between line baselines (0.5–5.0, default 1.2) |
| `smoothing` | Gaussian smoothing per line (0–10, default 2.0) — higher = smoother peaks |
| `freq_max` | Maximum frequency in Hz (500–20000, default 8000) |
| `fft_size` | FFT window size (512–8192, default 2048) — larger = smoother frequency resolution |
| `mirror` | Mirror the waveform symmetrically around each baseline |
| `hlr_enabled` | Enable Joy Division-style hidden line removal |

**Presets:** Joy Division, Dense Ridgeline, Wide Ridgeline, Mirror Ridgeline

**Tips:**
- For strong occlusion, keep `amplitude` notably larger than `row_spacing` (e.g., 2.0 amplitude with 1.2 spacing)
- Increase `smoothing` for cleaner, more organic peaks — lower values show more spectral detail
- Lower `freq_max` (3000–5000) focuses on bass/mid content, which often has more dramatic peaks

### Circular Waveform

Maps audio data onto a circle, modulating the radius with the signal. Three source modes: raw **Waveform** (oscillates above and below), **Envelope** (smooth amplitude bumps), or **Spectrum** (frequency domain magnitudes).

| Parameter | Description |
|-----------|-------------|
| `circle_amplitude` | Amplitude relative to radius (0.01–0.5, default 0.2) |
| `circle_points` | Number of points around the circle (360–7200, default 3600) |
| `circle_smoothing` | Gaussian smoothing (0–20, default 5.0) |
| `circle_source` | Waveform, Envelope, or Spectrum |
| `circle_closed` | Connect end back to start |

**Presets:** Circular Waveform, Circular Envelope, Circular Spectrum

### Spiral Waveform

Audio mapped onto an Archimedean spiral, like reading a vinyl record. Supports hidden line removal for when the amplitude causes adjacent turns to overlap.

| Parameter | Description |
|-----------|-------------|
| `spiral_turns` | Number of revolutions (1–30, default 8) |
| `spiral_amplitude` | Amplitude relative to spiral gap (0.01–1.0, default 0.15) |
| `spiral_points` | Total points (1000–20000, default 7200) |
| `spiral_smoothing` | Gaussian smoothing (0–20, default 8.0) |
| `spiral_source` | Waveform or Envelope |
| `spiral_direction` | Outward (center to edge) or Inward (edge to center) |
| `spiral_hlr` | Hidden line removal — hides inner turn segments occluded by outer turns |

**Presets:** Vinyl Spiral, Tight Spiral

**Tips:**
- Increase `spiral_amplitude` for more dramatic displacement — HLR handles overlapping turns cleanly
- **Envelope** source produces smoother, more organic spirals; **Waveform** shows raw oscillation detail
- **Inward** direction reads time from edge to center, like a record playing in reverse

### Spectrogram Contours

Treats the spectrogram as a 2D heightmap and extracts topographic contour lines at evenly spaced intensity levels. Produces a terrain-map-like visualization of the sound.

| Parameter | Description |
|-----------|-------------|
| `contour_levels` | Number of contour levels (3–30, default 10) |
| `contour_smoothing` | Gaussian smoothing on the spectrogram before contouring (0–5, default 1.5) |
| `contour_freq_max` | Maximum frequency in Hz (500–20000, default 8000) |
| `contour_fft_size` | FFT window size (512–8192, default 2048) |
| `contour_min_length` | Minimum contour length in mm — removes tiny fragments (0–20, default 2.0) |

**Presets:** Topographic Audio, Detailed Contours, Smooth Contours

### Frequency Bands

Splits audio into frequency bands using Butterworth bandpass filters and draws each band as a separate waveform. Useful for multi-pen plotting — plot each band with a different pen color.

| Parameter | Description |
|-----------|-------------|
| `band_count` | 3 (Bass/Mid/Treble), 4, or 5 bands |
| `band_style` | Stacked Waveforms, Stacked Envelopes, or Side by Side |
| `band_amplitude` | Vertical scale (0.1–5.0, default 1.5) |
| `band_smoothing` | Gaussian smoothing (0–10, default 3.0) |
| `band_points` | Points per band (500–10000, default 3000) |

**Presets:** Bass/Mid/Treble, Five Band Envelope, Side by Side

**Tips:**
- Use **Side by Side** for a clear comparison of frequency content across bands
- **Stacked Envelopes** produces a smoother, more abstract look than raw waveforms

### Lissajous / Stereo Phase

For stereo audio, plots the left channel as X and right channel as Y, revealing the spatial character of the mix. Mono files are handled by creating a phase-shifted copy.

| Parameter | Description |
|-----------|-------------|
| `liss_segment_sec` | Duration of audio segment (0.005–2.0 s, default 0.05) — shorter = cleaner curves |
| `liss_points` | Number of points (500–20000, default 5000) |
| `liss_smoothing` | Gaussian smoothing (0–10, default 3.0) |
| `liss_auto_segment` | Automatically find the most energetic segment |
| `liss_segment_start` | Manual segment start (visible when auto-select is off) |

**Presets:** Lissajous Clean, Lissajous Dense, Lissajous Minimal

**Tips:**
- Short segments (0.01–0.05 s) produce clean elliptical figures; longer segments (0.2+ s) create dense, textured fills
- Stereo music with wide panning produces the most interesting patterns
- Try different segments of the same song — verse, chorus, and bridge often produce very different figures

---

## Post-Generation Transforms

After any generation, the Settings Panel provides additional transform controls that apply
to the generator output before it is added to the layer:

| Control | Description |
|---------|-------------|
| **Scale** | Multiply all coordinates by a factor |
| **Rotation** | Rotate around the canvas center |
| **X / Y Translate** | Shift in mm |
| **Mirror H / V** | Flip horizontally or vertically |
| **Rotational symmetry** | Replicate n times around the center (n-fold) |
| **Tile Repeat** | Grid-repeat across the canvas (rows × columns) |

---

## Randomize

Click **Randomize** (`Ctrl+R`) or use **Generate › Randomize** to set all parameters
to random values within their valid ranges. The current random seed is displayed in the
status bar for reproducibility.

---

## Surprise Me

**Generate › Surprise Me** picks a random math generator, randomizes all its parameters,
and generates immediately. Use this for rapid inspiration or happy accidents.
