# Interface Overview

This guide describes every part of the Plottter main window so you can orient yourself quickly
and understand how the panels work together.

---

## Main Window Layout

```
┌────────────────────────────────────────────────────────────────────────┐
│  File  Edit  View  Generate  Tools  Help          [toolbar icons]      │
├───────────────┬────────────────────────────────┬───────────────────────┤
│  Mode Panel   │                                │  Settings Panel       │
│  (left top)   │       Canvas Area              │  (right panel)        │
├───────────────┤       (center)                 │                       │
│  Layer Panel  │                                │                       │
│  (left bottom)│                                │                       │
├───────────────┴────────────────────────────────┴───────────────────────┤
│  Status bar: dimensions │ path count │ travel distance │ cursor pos    │
└────────────────────────────────────────────────────────────────────────┘
```

The three main areas are separated by draggable splitters. Drag the vertical dividers left or right
to resize the panels. Plottter saves these proportions when you quit.

---

## Canvas Area

The canvas displays your artwork on a simulated sheet of paper.

### Paper and margin boundaries

- **Solid black rectangle** — the physical paper edge
- **Dashed grey rectangle** — the safe drawing margin (set in the new-project dialog)

Paths that stray outside the margin boundary will be clipped when you use **Tools › Clip to
Canvas**.

### Navigation

| Action | How |
|--------|-----|
| Zoom in / out | Scroll wheel |
| Zoom in | `Ctrl+=` |
| Zoom out | `Ctrl+-` |
| Fit canvas to window | `Ctrl+0` |
| Pan | Middle-click drag, or `Ctrl` + left-click drag |

### Grid overlay

Press `G` (or **View › Toggle Grid**) to show a 10 mm grid. The grid is purely visual and does not
affect export.

### Registration marks

Press `R` (or **View › Toggle Registration Marks**) to show corner crosshairs. These marks appear
in the exported SVG to help you align multi-pen plots.

### Pen-up travel visualization

Press `T` (or **View › Toggle Travel Lines**) to show dashed grey lines between path endpoints.
Each dashed segment is a pen-up travel move. Minimizing these moves speeds up plots — use
**Tools › Optimize** to reorder paths automatically.

---

## Mode Panel (Left Top)

The **Mode Panel** selects the type of art generation:

| Mode | Description |
|------|-------------|
| **Math Art** | Generate paths from parametric equations, polar curves, L-systems, flow fields, and tiling patterns |
| **Image to Lines** | Convert a raster image to plotter paths via edge detection, hatching, stippling, or flow fields |
| **Color Separation** | Separate an image by hue, luminance, or channel and assign each color to a separate layer |
| **Mask Paint** | Paint regions on the canvas with a brush to create custom masks for per-region line art |

Selecting a mode updates the **Settings Panel** on the right with relevant controls.

---

## Settings Panel (Right Panel)

The Settings Panel is a scrollable area that changes depending on the active mode and generator.

### Common controls (top)

- **Generator / Algorithm** dropdown — selects which generator to use
- **Preset** dropdown — applies a named set of parameters; choosing a preset fills all fields
- **Target layer** dropdown — which layer receives the generated paths

### Parameter controls

Each generator exposes its own parameters as interactive controls:

| Parameter type | Widget |
|---------------|--------|
| Floating-point number | Spin box with step buttons |
| Integer | Integer spin box |
| Math expression | Text field (validated against the safe evaluator) |
| Multiple-choice | Dropdown |
| Boolean flag | Checkbox |
| Image file | File picker button + thumbnail preview |

### Action buttons

- **Generate** (`Ctrl+G`) — run the selected generator and add paths to the target layer
- **Randomize** (`Ctrl+R`) — randomize all parameters within their valid ranges (displays a seed)
- **Cancel** — available during generation; stops the background thread (note: some operations
  like the Lorenz ODE cannot be interrupted mid-run)

### Progress bar

A progress bar appears during generation and disappears when complete.

---

## Layer Panel (Left Bottom / Bottom)

The Layer Panel lists all layers in the project and provides controls for managing them.

### Layer list

Each row shows:

- **Color swatch** — click to open a color picker (choose the pen color for this layer)
- **Name** — double-click to rename inline
- **Eye icon / checkbox** — toggle layer visibility
- **Lock icon / checkbox** — when locked, the layer cannot be edited
- **Path count badge** — number of polylines in the layer

Drag rows to reorder layers. The order affects export (layer 01 is exported first) and the
animation playback sequence.

### Layer buttons

| Button | Action |
|--------|--------|
| **+** | Add a new empty layer |
| **−** | Delete the selected layer |
| **Duplicate** | Copy the selected layer (new UUID, copied paths) |
| **Merge** | Merge selected layers into one |
| **↑ / ↓** | Move the selected layer up or down |

### Context menu (right-click a layer)

- Rename
- Change Color
- Export Layer (opens the Export dialog for that layer only)

---

## Status Bar

The status bar at the bottom of the window shows:

| Section | Description |
|---------|-------------|
| Canvas dimensions | e.g. `210 × 297 mm (A4)` |
| Total paths | Number of polylines across all visible layers |
| Estimated travel | Total pen-down distance in mm |
| Cursor position | Current mouse position in mm (updates as you move over the canvas) |

---

## Menu Bar

### File

| Item | Shortcut | Description |
|------|----------|-------------|
| New | `Ctrl+N` | Create a new project |
| Open | `Ctrl+O` | Open a `.plottter` file |
| Save | `Ctrl+S` | Save the current project |
| Save As | `Ctrl+Shift+S` | Save to a new path |
| Recent Projects | — | Submenu of last 10 projects |
| Export Current Layer | `Ctrl+E` | Export the active layer |
| Export All Layers | `Ctrl+Shift+E` | Export all visible layers |
| Quit | `Ctrl+Q` | Exit (prompts if unsaved) |

### Edit

| Item | Shortcut | Description |
|------|----------|-------------|
| Undo | `Ctrl+Z` | Undo the last action |
| Redo | `Ctrl+Y` | Redo the last undone action |
| Canvas Settings | — | Edit paper size and margins |

### View

| Item | Shortcut | Description |
|------|----------|-------------|
| Zoom In | `Ctrl+=` | Zoom in |
| Zoom Out | `Ctrl+-` | Zoom out |
| Fit to Window | `Ctrl+0` | Fit the entire canvas |
| Toggle Grid | `G` | Show or hide the 10 mm grid |
| Toggle Registration Marks | `R` | Show or hide corner crosshairs |
| Toggle Travel Lines | `T` | Show or hide pen-up travel moves |

### Generate

| Item | Description |
|------|-------------|
| Generate | Run the current generator |
| Randomize | Randomize all parameters |
| Surprise Me | Pick a random math generator with random parameters |

### Tools

| Item | Description |
|------|-------------|
| Optimize Current Layer | Full optimization pipeline on the active layer |
| Optimize All Layers | Full optimization on every unlocked layer |
| Simplify Paths | RDP simplification only |
| Merge Nearby Paths | Connect nearby endpoints |
| Clip to Canvas | Remove paths outside the margin boundary |
| Plot with AxiDraw… | Send to AxiDraw plotter via USB |
| Manage Plugins… | Load custom generator plugins |

### Help

| Item | Description |
|------|-------------|
| About Plottter | App name, version, and credits |
| Keyboard Shortcuts | Table of all keyboard shortcuts |

---

## Toolbar

The toolbar provides quick access to the most common actions:

New · Open · Save · Export · | · Undo · Redo

---

## Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New project |
| `Ctrl+O` | Open project |
| `Ctrl+S` | Save project |
| `Ctrl+Shift+S` | Save As |
| `Ctrl+Q` | Quit |
| `Ctrl+E` | Export current layer |
| `Ctrl+Shift+E` | Export all layers |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+=` | Zoom in |
| `Ctrl+-` | Zoom out |
| `Ctrl+0` | Fit to window |
| `Ctrl+G` | Generate |
| `Ctrl+R` | Randomize |
| `G` | Toggle grid |
| `T` | Toggle travel lines |
| `R` | Toggle registration marks |
