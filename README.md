![](docs/images/hatch_logo_2color.png)

# Plottter

A free, open-source desktop application for generating pen-plotter-ready vector art from mathematical equations, generative algorithms, and raster images. Produces multi-layer output in SVG, HPGL, G-code, and Mural formats for any pen plotter.

## Features

**33 generators** across math art, image-to-lines, 3D, and audio:

- **Math Art** — Parametric curves, polar curves, modular multiplication, Perlin noise flow fields (with quantized and rectilinear modes), superformula fields, L-system fractals, grid patterns, concentric rings, dot grids, geometric grids, Voronoi/Delaunay tessellation, Penrose aperiodic tiling, and text rendering
- **Image to Lines** — Edge detection (Canny), hatching (with oscillation mode), flow field/squiggle, Voronoi stippling (with TSP single-line mode), contour lines, XDoG, FDoG (coherent lines), hedcut portraits, scanline halftone, circular scribble, LIC, TAM, dot grid halftone, spiral portraits, sketch (iterative darkest-trace-erase with hybrid marks), mosaic hatching (triangles/Voronoi/rectangles/hexagons/quadtree/superpixels), and ASCII art
- **3D Scene** — Wireframe, hatched/shaded, and perspective-hatched rendering with hidden line removal, 10+ primitive shapes, OBJ/STL mesh import and slicing, camera controls, and shadow effects
- **Audio** — Import WAV, MP3, FLAC, or OGG files and generate plotter-ready visualizations: Joy Division-style ridgeline spectrograms with hidden line removal, circular and spiral waveforms, spectrogram contour maps, frequency band separation, and stereo Lissajous figures

**Multi-layer system** with per-layer pen colors and opacity, drag reorder, visibility/lock controls, and color separation (K-Means, Luminance, RGB, CMYK).

**Mask painting** with brush, rectangle, ellipse, polygon, and pen tools. AI-powered mask generation via point prompts, box prompts, or text descriptions (Replicate SAM-2). Per-project mask library for saving and reusing masks. Mask refinement with feather and grow/shrink controls.

**Export formats:** SVG (mm coordinates), HPGL (vintage plotters), G-code (CNC/servo), Mural (wall-mounted plotters), and direct AxiDraw USB control.

**Post-processing tools:** Path optimization (nearest-neighbor + 2-opt + 3-opt + Or-opt), simplification, merge, clip, weld overlapping paths, Bezier curve fitting, path tapering (fade stroke width at endpoints), path offsetting (parallel curves), and a brush system (stippled, multi-stroke, calligraphic).

**Other features:** Stroke-order animation, pen-up travel visualization, pen jitter simulation, AI result caching, auto contrast and unsharp mask preprocessing, CLI batch mode, extensible plugin system (generators, processing, and export format plugins), Google Fonts integration, and AI-powered depth maps and background removal via Replicate.

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| OS | Linux, macOS, Windows |

## Installation

```bash
git clone https://github.com/pywkt/plottter.git
cd plottter

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

### Optional dependencies

```bash
pip install -e ".[fmm]"    # Fast Marching Method for topographic contours
pip install -e ".[audio]"  # MP3/FLAC/OGG audio import (requires ffmpeg)
pip install pyaxidraw       # Direct AxiDraw USB control
```

AI features (depth maps, background removal, segmentation) require a Replicate.com API key configured in Preferences.

## Usage

```bash
plottter           # Launch the GUI
python -m plottter # Alternative launch

# CLI batch mode
plottter --generator "Parametric Curves" --preset lissajous --output out.svg
plottter --list-generators  # Show all available generators
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, first project, basic workflow |
| [Math Art Guide](docs/math-art-guide.md) | All math generators with parameters and presets |
| [Image-to-Lines Guide](docs/image-to-lines-guide.md) | 13 image conversion algorithms |
| [Layers & Colors](docs/layers-and-colors.md) | Multi-layer workflows and color separation |
| [Export & Plotting](docs/export-and-plotting.md) | SVG, HPGL, G-code, Mural export and AxiDraw control |
| [Preview & Simulation](docs/preview-and-simulation.md) | Animation, travel metrics, pen jitter |
| [Tips & Workflows](docs/tips-and-workflows.md) | Algorithm selection, optimization, brush effects, plugins |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |

## License

MIT
