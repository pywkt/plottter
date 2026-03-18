# Math Art Guide

Math Art mode generates vector paths purely from mathematical equations and algorithms —
no image input required. All generators produce output in millimeter coordinates that
automatically scale to fit the canvas drawing area.

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
