![](docs/images/hatch_logo_2color.png)

# Plottter

A free, open-source desktop application for generating pen-plotter-ready vector art from mathematical equations, generative algorithms, and raster images. Produces multi-layer output in SVG, HPGL, G-code, and Mural formats for any pen plotter.

## Features

### Generators

**39 generators** across math art, image-to-lines, 3D, audio, maps, and code-driven art:

- **Math Art** — Parametric curves, polar curves, modular multiplication, Perlin noise flow fields (quantized and rectilinear modes), superformula fields, L-system fractals, grid patterns, concentric rings, dot grids, geometric grids, Voronoi/Delaunay tessellation, Penrose aperiodic tiling, and text rendering.
- **Image to Lines** — Edge detection (Canny), hatching (with oscillation mode), flow field/squiggle, Voronoi stippling (with TSP single-line mode), contour lines, XDoG, FDoG (coherent lines), hedcut portraits, scanline halftone, circular scribble, LIC, TAM, dot grid halftone, spiral portraits, sketch (iterative darkest-trace-erase), mosaic hatching (triangles/Voronoi/rectangles/hexagons/quadtree/superpixels), and ASCII art — plus four multi-layer colour generators:
  - **Paired Wave Shading** — split-line two-pen colour shading.
  - **Pixel Art** — pixel-as-shape rendering with 15+ retro palettes (NES, Game Boy, SNES, PICO-8, C64, EGA/CGA, Endesga32, grayscale), square / diamond / circle / hex cells, and dithering.
  - **Pointillist** — optical-mixing dot art, palette-driven with Mitchell best-candidate sampling; gives each dot its own paper to escape ink-on-ink interaction.
  - **CRT TV** — retro-CRT effect with shadow-mask / aperture-grille / slot-mask subpixel layouts, scanlines, vignette, and barrel distortion (Trinitron-style bars or NES-style triad dots).
- **3D Scene** — Wireframe, hatched/shaded, and perspective-hatched rendering with hidden line removal, 10+ primitive shapes, OBJ/STL mesh import and slicing, camera controls, and shadow effects.
- **Audio** — Import WAV, MP3, FLAC, or OGG and generate plotter-ready visualizations: Joy Division-style ridgeline spectrograms with hidden line removal, circular and spiral waveforms, spectrogram contour maps, frequency band separation, and stereo Lissajous figures.
- **Map** — Turn any real-world location's OpenStreetMap data into multi-layer art — roads (major/minor with stroke emphasis), water bodies (multipolygon-aware, hatch/cross-hatch fill), waterways, rail/transit, parks, buildings, and coastline, each its own pen layer. Interactive pan/zoom positioning frames the exact crop before generating, and selecting a map layer later restores the full settings to tweak and regenerate in place. Disk-cached fetches, no API key required, ODbL attribution baked in, CLI batch mode supported.
- **Code-driven (plugin)** — The **TurtleToy** plugin runs JavaScript pasted from [turtletoy.net](https://turtletoy.net) 1:1, including adjustable-variable comment syntax (`const x = 5; // min=0, max=10`) that surfaces as live sliders/dropdowns. Requires the optional `quickjs` package.

### Layers & color separation

- Per-layer pen colors and opacity, drag reorder, and visibility/lock controls.
- Color separation: K-Means, Luminance, RGB, CMYK, AI Layer Separation, and **Custom Palette** (perceptually-correct Lab-space matching with Floyd-Steinberg / ordered / Atkinson dithering).
- Built-in pen palettes — Basic 6, Copic 12, Sakura Metallic 5, Grayscale 5, RYBK 4 (traditional painter primaries), CMYKOG 6 (Hexachrome-style extended gamut), Risograph 6 — plus a palette editor for saving your own.
- "Skip near-white layer" drops the background after AI background removal so K-Means / Luminance / Custom Palette don't leave a stray white-pen layer.

### Mask painting

- Brush, rectangle, ellipse, polygon, and pen tools.
- AI-powered mask generation via point prompts, box prompts, or text descriptions (Replicate SAM-2).
- Per-project mask library for saving and reusing masks.
- Refinement with feather and grow/shrink controls.

### Export & plotting

- **SVG** — mm coordinates.
- **HPGL** — vintage plotters.
- **G-code** — CNC/servo.
- **Mural** — wall-mounted plotters.
- **AxiDraw** — direct USB control.
- **Wireless plotting** — offload jobs over the network to a Raspberry Pi running the companion `plottter-daemon`. The device owns the plot, so your computer is free during long runs and the plotter can live in another room.

### Post-processing

- **Path optimization** — nearest-neighbor + 2-opt + 3-opt + Or-opt.
- **Simplification, merge, and clip.**
- **Graph-aware junction joiner** — splits paths at T-junctions and traces Eulerian chains across the connectivity graph, dramatically reducing pen-lift count on dense road networks (auto-enabled for Map layers).
- **Remove Duplicate Segments.**
- **Bezier curve fitting.**
- **Path tapering** — fade stroke width at endpoints.
- **Path offsetting** — parallel curves.
- **Brush system** — stippled, multi-stroke, calligraphic.
- **numba JIT acceleration** (optional) — speeds up the optimize / merge / weld inner loops; install with `pip install -e ".[fast]"`.
- **Remote optimization** — offload the full Optimize pipeline over SSH to a faster machine (or via Tailscale SSH, with zero open ports on the remote), keeping the GUI snappy on a low-power computer.

### Other features

- Stroke-order animation, pen-up travel visualization, and pen jitter simulation.
- **Ink Preview** canvas mode — subtractive colour mixing where layers overlap, previewing how the actual plot will look.
- Configurable preview pen width with a "Marker" preset.
- Auto contrast and unsharp mask preprocessing.
- AI result caching, plus AI-powered depth maps and background removal via Replicate.
- CLI batch mode.
- Extensible plugin system — generator, processing, and export-format plugins.
- Google Fonts integration.

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| pip | 21.3+ (for PEP 660 editable installs) |
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
pip install -e ".[fmm]"    # Fast Marching Method — topographic & "Edge Hug" contours
pip install -e ".[audio]"  # MP3/FLAC/OGG audio import (requires ffmpeg)
pip install quickjs        # Enables the TurtleToy plugin (run JavaScript sketches)
pip install https://cdn.evilmadscientist.com/dl/ad/public/AxiDraw_API.zip  # Direct AxiDraw USB control
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
| [Map Guide](docs/map-guide.md) | OpenStreetMap-based map art: fetch, interactive pan/zoom positioning, edit-in-place, categories, fills, labels (water/park/place/road), attribution, CLI |
| [Layers & Colors](docs/layers-and-colors.md) | Multi-layer workflows and color separation |
| [Export & Plotting](docs/export-and-plotting.md) | SVG, HPGL, G-code, Mural export and AxiDraw control |
| [Preview & Simulation](docs/preview-and-simulation.md) | Animation, travel metrics, pen jitter |
| [Tips & Workflows](docs/tips-and-workflows.md) | Algorithm selection, optimization, brush effects, plugins |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [Performance](docs/performance.md) | Benchmark results for the path-processing pipeline; `[fast]` extra install instructions |
| [Remote Optimization](docs/remote-optimization.md) | Offload the Optimize pipeline to a fast machine over SSH (or Tailscale SSH with no open ports) |
| [Remote Plotting](docs/remote-plotting.md) | Plot wirelessly to a networked Raspberry Pi (companion `plottter-daemon`) so your computer is free during long plots |

## License

MIT
