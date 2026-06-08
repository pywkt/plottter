![](docs/images/hatch_logo_2color.png)

# Plottter

A free, open-source desktop application for generating pen-plotter-ready vector art from mathematical equations, generative algorithms, and raster images. Produces multi-layer output in SVG, HPGL, G-code, and Mural formats for any pen plotter.

## Features

**39 generators** across math art, image-to-lines, 3D, audio, maps, and code-driven art:

- **Math Art** — Parametric curves, polar curves, modular multiplication, Perlin noise flow fields (with quantized and rectilinear modes), superformula fields, L-system fractals, grid patterns, concentric rings, dot grids, geometric grids, Voronoi/Delaunay tessellation, Penrose aperiodic tiling, and text rendering
- **Image to Lines** — Edge detection (Canny), hatching (with oscillation mode), flow field/squiggle, Voronoi stippling (with TSP single-line mode), contour lines, XDoG, FDoG (coherent lines), hedcut portraits, scanline halftone, circular scribble, LIC, TAM, dot grid halftone, spiral portraits, sketch (iterative darkest-trace-erase with hybrid marks), mosaic hatching (triangles/Voronoi/rectangles/hexagons/quadtree/superpixels), ASCII art, **Paired Wave Shading** (split-line two-pen colour shading), **Pixel Art** (multi-layer pixel-as-shape rendering with 15+ retro palettes — NES, Game Boy, SNES, PICO-8, C64, EGA/CGA, Endesga32, grayscale — plus square / diamond / circle / hex cell shapes and dithering), **Pointillist** (multi-layer optical-mixing dot art — palette-driven with Mitchell best-candidate sampling, escapes ink-on-ink interaction by giving each dot its own paper), and **CRT TV** (multi-layer retro-CRT effect with shadow-mask / aperture-grille / slot-mask subpixel layouts, scanlines, vignette, and barrel distortion — Trinitron-style vertical bars or NES-style triad dots, palette-driven)
- **3D Scene** — Wireframe, hatched/shaded, and perspective-hatched rendering with hidden line removal, 10+ primitive shapes, OBJ/STL mesh import and slicing, camera controls, and shadow effects
- **Audio** — Import WAV, MP3, FLAC, or OGG files and generate plotter-ready visualizations: Joy Division-style ridgeline spectrograms with hidden line removal, circular and spiral waveforms, spectrogram contour maps, frequency band separation, and stereo Lissajous figures
- **Map** — Type a real-world location and turn its OpenStreetMap data into multi-layer pen-plotter art: roads (major/minor with stroke emphasis), water bodies (multipolygon-aware, with hatch/cross-hatch fill), waterways, rail/transit, parks and green space, buildings, and coastline — each its own pen layer. Interactive **pan/zoom positioning** lets you frame the exact crop before generating; selecting a map layer later restores the full settings so you can tweak and regenerate in place. Disk-cached fetches, no API key required, ODbL attribution baked in. CLI batch mode supported.
- **Code-driven (plugin)** — **TurtleToy** plugin lets you paste JavaScript from [turtletoy.net](https://turtletoy.net) and run it 1:1, including adjustable-variable comment syntax (`const x = 5; // min=0, max=10`) that surfaces as live sliders/dropdowns. Requires the optional `quickjs` package.

**Multi-layer system** with per-layer pen colors and opacity, drag reorder, visibility/lock controls, and color separation (K-Means, Luminance, RGB, CMYK, **Custom Palette** with perceptually-correct Lab-space matching + Floyd-Steinberg / ordered / Atkinson dithering, AI Layer Separation). Built-in pen palettes — Basic 6, Copic 12, Sakura Metallic 5, Grayscale 5, RYBK 4 (traditional painter primaries), CMYKOG 6 (Hexachrome-style extended gamut), Risograph 6 — plus a palette editor for saving your own. A "Skip near-white layer" option drops the background layer after AI background removal so K-Means / Luminance / Custom Palette don't leave a stray white-pen layer.

**Mask painting** with brush, rectangle, ellipse, polygon, and pen tools. AI-powered mask generation via point prompts, box prompts, or text descriptions (Replicate SAM-2). Per-project mask library for saving and reusing masks. Mask refinement with feather and grow/shrink controls.

**Export formats:** SVG (mm coordinates), HPGL (vintage plotters), G-code (CNC/servo), Mural (wall-mounted plotters), and direct AxiDraw USB control. **Wireless plotting** offloads jobs over the network to a Raspberry Pi running the companion `plottter-daemon` — the device owns the plot, so your computer is free during long runs and the plotter can live in another room.

**Post-processing tools:** Path optimization (nearest-neighbor + 2-opt + 3-opt + Or-opt), simplification, merge, **graph-aware junction joiner** (splits paths at T-junctions and traces Eulerian chains across the connectivity graph — dramatically reduces pen-lift count on dense road networks; auto-enabled for Map layers), Remove Duplicate Segments, clip, Bezier curve fitting, path tapering (fade stroke width at endpoints), path offsetting (parallel curves), and a brush system (stippled, multi-stroke, calligraphic). Optional **numba JIT** acceleration for the optimize / merge / weld inner loops — install with `pip install -e ".[fast]"`. **Remote optimization** offloads the full Optimize pipeline over SSH to a faster machine on your network (or via Tailscale SSH, with zero open ports on the remote) — keeps the GUI snappy on a low-power computer.

**Other features:** Stroke-order animation, pen-up travel visualization, pen jitter simulation, **Ink Preview** canvas mode (subtractive colour mixing where layers overlap, so you can preview how the actual plot will look), configurable preview pen width with a "Marker" preset, AI result caching, auto contrast and unsharp mask preprocessing, CLI batch mode, extensible plugin system (generators, processing, and export format plugins), Google Fonts integration, and AI-powered depth maps and background removal via Replicate.

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
pip install -e ".[fmm]"    # Fast Marching Method for topographic contours
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
