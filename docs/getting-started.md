# Getting Started with Plottter

Plottter is a free, open-source desktop application for generating pen-plotter-ready vector art
from mathematical equations and raster images. It produces multi-layer SVG output (and HPGL/G-code)
suitable for any pen plotter such as the AxiDraw, HP plotters, or CNC machines with a pen mount.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.12 or later |
| Operating system | Linux, macOS, Windows |
| Qt runtime (bundled via PyQt6) | Qt 6 |
| Optional: pyaxidraw | Latest (for direct AxiDraw USB control) |

On Linux you may also need:

```
libEGL.so.1   (part of Mesa or Chromium)
```

See the [Troubleshooting guide](troubleshooting.md#blank-window-or-egl-error-on-linux) if the app
fails to start on a headless or minimal Linux installation.

---

## Installation

### From source (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/plottter.git
cd plottter

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install Plottter and all dependencies
pip install -e ".[dev]"
```

### Dependencies installed automatically

- **PyQt6** — GUI framework
- **numpy** — numerical computation
- **opencv-python** — image processing
- **Pillow** — image I/O
- **shapely** — 2D geometry
- **scipy** — spatial indexing and ODE integration
- **svgwrite** — SVG generation
- **noise** — Perlin noise for flow fields

---

## First Launch

Start Plottter from the command line:

```bash
python -m plottter
```

Or use the installed entry point (after `pip install -e .`):

```bash
plottter
```

The application window appears with:

- **Left panel** — Mode selector and generator list
- **Center** — Interactive canvas (white paper on grey background)
- **Right panel** — Generator parameters and Generate button
- **Bottom** — Layer panel with controls
- **Status bar** — Canvas size, path count, and cursor coordinates

---

## Your First Project

### Step 1 — Create a project

When the app opens, a default project with an A4 canvas and one empty layer is ready to use.

To start from a specific paper size:

1. **File › New** (or `Ctrl+N`)
2. Select a paper preset (A4, A3, Letter, etc.) or enter custom dimensions
3. Set a margin (default 10 mm) — plotters cannot usually reach the paper edge
4. Click **OK**

### Step 2 — Generate some art

1. In the left panel, confirm **Math Art** mode is selected
2. Choose a generator from the dropdown, for example **Parametric Curves**
3. Select a preset from the dropdown at the top of the right panel — try **Lissajous**
4. Click **Generate** (or `Ctrl+G`)

The canvas shows the generated paths. Zoom with the scroll wheel; pan by holding the middle mouse
button (or `Ctrl` + drag).

### Step 3 — Export as SVG

1. **File › Export Current Layer** (`Ctrl+E`) or **Export All Layers** (`Ctrl+Shift+E`)
2. Choose **SVG** format
3. Select an output file (or directory for batch export)
4. Click **Export**

Open the resulting `.svg` file in Inkscape, your plotter software, or send it directly to the
plotter.

---

## Choosing a Paper Size

Plottter includes these built-in presets:

| Preset | Width × Height |
|--------|---------------|
| A4 | 210 × 297 mm |
| A3 | 297 × 420 mm |
| A2 | 420 × 594 mm |
| Letter | 215.9 × 279.4 mm |
| Legal | 215.9 × 355.6 mm |

Select **Custom** in the new-project dialog to enter any dimensions in mm or inches.

The **margin** setting defines the safe drawing area inset from the paper edge. A margin of 10 mm
is a safe default for most plotters. The canvas shows the margin as a dashed grey rectangle inside
the solid paper border.

---

## Saving and Loading Projects

Projects are saved in Plottter's own `.plottter` format (JSON, gzip-compressed for large files):

- **Save** — `Ctrl+S` (saves to current path, or prompts if new)
- **Save As** — `Ctrl+Shift+S`
- **Open** — `Ctrl+O`

The **File › Recent Projects** submenu lists your last 10 projects for quick access.

---

## Next Steps

| Topic | Guide |
|-------|-------|
| Math generators (parametric, polar, L-systems, grids, text, 3D) | [Math Art Guide](math-art-guide.md) |
| Convert a photo to line art (10 algorithms) | [Image-to-Lines Guide](image-to-lines-guide.md) |
| Work with layers and pen colors | [Layers & Colors](layers-and-colors.md) |
| Export SVG, HPGL, G-code, or Mural | [Export & Plotting](export-and-plotting.md) |
| Preview and simulate the plot | [Preview & Simulation](preview-and-simulation.md) |
| Tips, workflows, and brush effects | [Tips & Workflows](tips-and-workflows.md) |
| Fix common issues | [Troubleshooting](troubleshooting.md) |
