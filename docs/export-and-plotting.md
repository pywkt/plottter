# Export & Plotting

Plottter can export artwork in four vector formats: **SVG** (primary), **HPGL**
(for vintage plotters), **G-code** (for CNC-style machines with a pen mount), and
**Mural** (for wall-mounted anchor-pin plotters). It can also plot directly to an
AxiDraw via USB.

---

## Opening the Export Dialog

- **File › Export Current Layer** (`Ctrl+E`) — export the active layer
- **File › Export All Layers** (`Ctrl+Shift+E`) — export all visible layers

The Export Dialog lets you choose:

1. **Format** — SVG, HPGL, or G-code
2. **Layer scope** — Current Layer / All Separate / All Combined
3. **Registration marks** — include or exclude corner crosshairs
4. **Stroke width** — pen line width in the SVG (does not affect HPGL/G-code geometry)
5. **Output path** — file for single/combined export; directory for batch export

---

## SVG Export

SVG is the primary export format. Every Plottter SVG is dimensioned in millimeters
so that your plotter software can scale it correctly without guessing.

### Format details

- **ViewBox:** `0 0 {width_mm} {height_mm}` — coordinates match millimeter canvas units
- **`width` / `height` attributes:** set in mm (e.g. `width="210mm"`)
- **Polylines:** each path is a `<polyline>` element with `fill="none"`
- **Stroke color:** taken from the layer color (hex string)
- **Stroke width:** configurable (default 0.3 mm)
- **Coordinate precision:** 3 decimal places
- **Registration marks:** `<g id="registration">` group of `<line>` elements (0.1 mm stroke)

### Export modes

| Mode | Output | Use when |
|------|--------|----------|
| **Current Layer** | Single SVG | One pen, one pass |
| **All Separate** | One SVG per visible layer | Multi-pen plotting (load each pen separately) |
| **All Combined** | Single SVG with all layers | Inkscape editing or manual pen management |

### Naming convention (batch export)

Files are named:

```
{project_name}_{layer_number:02d}_{layer_name}.svg
```

Example: `my_portrait_01_Shadows.svg`, `my_portrait_02_Midtones.svg`

---

## Path Optimization

Before exporting, run path optimization to minimize pen travel. This speeds up plots
significantly for dense images.

### Full optimization pipeline

**Tools › Optimize Current Layer** runs all steps in sequence:

1. **Simplify** — Ramer-Douglas-Peucker (removes redundant points while preserving shape)
2. **Filter** — removes paths shorter than the minimum length threshold
3. **Clip** — clips paths to the canvas drawing area
4. **Merge** — connects path endpoints within a configurable distance
5. **Reorder (nearest-neighbor)** — re-sequences paths to minimize pen-up travel
6. **2-opt improvement** — iterative swap pass to improve on the nearest-neighbor result

After optimization, a summary dialog shows:

- Before / after pen-up travel distance
- Percentage travel reduction
- Pen lift count

### Individual operations

**Tools menu:**

| Action | Description |
|--------|-------------|
| Simplify Paths | RDP simplification only |
| Merge Nearby Paths | Connect endpoints within threshold |
| Clip to Canvas | Remove paths outside drawing area |
| Weld Overlapping Paths | Remove duplicate overlapping segments — useful after merging layers |
| Optimize All Layers | Full pipeline on every unlocked visible layer |
| Apply Brush to Layer | Replace plain strokes with stippled, multi-stroke, or calligraphic effects |

All optimization runs in a background thread and reports progress.

### Bezier Curve Fitting

Plottter includes a Potrace-style cubic Bezier curve fitting post-processor. When enabled
on a generator (via the `smooth_curves` parameter on edge detection generators), raw jagged
contours are replaced with smooth Bezier curves. The algorithm:

1. Detects corners (angle change > 60deg)
2. Fits cubic Bezier curves between corners using least-squares
3. Recursively splits and re-fits if the error exceeds the tolerance

This produces significantly smoother output that plots more cleanly, especially for edge
detection and contour-based generators.

---

## HPGL Export

HPGL (Hewlett-Packard Graphics Language) is the native language of vintage HP plotters
(7470A, 7475A, 7550A, DraftMaster series, etc.).

### Format details

```
IN;                         Initialize plotter
SP1;                        Select pen 1
PU x1,y1;                   Pen up, move to start
PD x2,y2,x3,y3,...;        Pen down, draw through points
PU;                         Final pen up
```

### Coordinate conversion

```
hpgl_x = int(x_mm × 40)
hpgl_y = int((canvas_height_mm − y_mm) × 40)
```

The Y-axis is inverted (HPGL origin is bottom-left). Units are 0.025 mm (40 units per mm).

### Settings

| Setting | Description |
|---------|-------------|
| `pen_number` | SP command value (1-based index of the plotted layer) |

---

## G-code Export

G-code is used by CNC-style pen plotters, modified 3D printers, and laser engravers
with a servo pen mount.

### Preamble

```gcode
G90       ; absolute positioning
G21       ; millimeter units
G28       ; home all axes
M3 S0     ; servo pen up (initial state)
```

### Drawing commands

```gcode
M3 S{pen_up_angle}   ; pen up
G0 X{x} Y{y} F{travel_speed}    ; rapid travel move
M3 S{pen_down_angle} ; pen down
G1 X{x} Y{y} F{draw_speed}      ; draw move
```

### Epilogue

```gcode
M3 S0     ; servo pen up
G28       ; home
M5        ; spindle off
```

### Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `travel_speed` | 3000 mm/min | Pen-up move speed |
| `draw_speed` | 1000 mm/min | Pen-down draw speed |
| `pen_up_angle` | 0 | Servo angle for pen up (M3 S value) |
| `pen_down_angle` | 90 | Servo angle for pen down |

---

## Mural Plotter Export

Mural format is a plain-text command format for wall-mounted plotters that draw using two
anchor pins and a gondola mechanism (similar to the Makelangelo or v-plotter designs).

### Format details

```
d{total_distance}    # Distance header — total drawing distance in mm
h{drawing_height}    # Height header — drawing area height in mm
p0                   # Pen up
{x} {y}              # Move to position
p1                   # Pen down
{x} {y}              # Draw to position(s)
p0                   # Final pen up
```

### Coordinate system

- Origin: top-left corner
- Y increases downward
- Drawing width is derived from `top_distance * 0.6`
- Home position: center of the drawing area at y=350mm

### Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `top_distance` | 1025.0 mm | Distance between the two anchor pins at the top of the wall |

**Best for:** Wall-mounted string plotters, v-plotters, gondola plotters.

---

## Direct AxiDraw Control (USB)

Plottter can send artwork directly to an AxiDraw pen plotter via USB without exporting
a file first.

### Requirements

Install the official pyaxidraw Python package:

```bash
pip install pyaxidraw
```

Connect your AxiDraw via USB and power it on.

### Using the AxiDraw dialog

1. **Tools › Plot with AxiDraw…**
2. The dialog checks if pyaxidraw is installed and shows its status
3. Select your AxiDraw model (default: V3/A3)
4. Adjust speed, pen position, and delay settings
5. Click **Plot Now**

### Settings

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Model | 1–6 | 2 (V3/A3) | AxiDraw hardware model |
| Drawing speed | 1–100% | 25% | Pen-down movement speed |
| Travel speed | 1–100% | 75% | Pen-up movement speed |
| Pen-down position | 0–100% | 40% | Servo position when drawing |
| Pen-up position | 0–100% | 60% | Servo position when travelling |
| Delay after pen-down | 0–2000 ms | 0 | Wait after lowering pen |
| Delay after pen-up | 0–2000 ms | 0 | Wait after raising pen |
| Constant speed | checkbox | off | Disable acceleration (better for certain media) |
| Preview mode | checkbox | off | Simulate without device (useful for testing) |

**Preview mode** runs the full plot job without sending any commands to the device.
It is enabled automatically if pyaxidraw is not installed.

### Plot scope

The AxiDraw dialog always plots all visible layers as a single combined SVG job. To plot
one layer at a time, hide all other layers before opening the dialog.

---

## Tips for Common Plotters

### AxiDraw (V2, V3, SE series)

- Use **Tools › Plot with AxiDraw…** for direct USB control
- Or export SVG and open in Inkscape with the AxiDraw Inkscape extension
- Default pen-down position 40% works for most pen types; adjust for felt-tip vs. ballpoint
- Use speed 15–20% for fine-nibbed pens to prevent skipping

### HP DraftMaster / 7475A

- Export HPGL; send via serial port or USB-to-serial adapter
- Baud rate: 9600 (most HP plotters)
- The `SP{n}` command selects the pen carousel position (1–8)

### Axidraw via Inkscape Extension

1. Export **All Separate SVG** files
2. Open each in Inkscape
3. Load the AxiDraw extension (`Extensions › AxiDraw Utilities`)
4. Use `Plot` to send each layer to the plotter

### Generic CNC / GRBL machines

- Export G-code and send with a tool like [Universal Gcode Sender](https://winder.github.io/ugs_website/)
  or [bCNC](https://github.com/vlachoudis/bCNC)
- Adjust `travel_speed` and `draw_speed` to suit your machine's capabilities
- Set `pen_up_angle` and `pen_down_angle` to match your servo mount

---

## Canvas and Margin Setup

Before exporting, verify the canvas dimensions match your physical paper:

- **Edit › Canvas Settings** — change paper size or margin
- The margin inset (dashed rectangle on canvas) defines the safe drawing boundary
- **Tools › Clip to Canvas** removes any paths outside the margin so they don't clip unexpectedly
  when the plotter reaches the edge of its travel range

Use a margin of at least 10 mm for most plotters. Some plotters (e.g. AxiDraw V3/A3) can
travel close to the paper edge, but paper clips and tape often reduce the usable area.
