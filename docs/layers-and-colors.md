# Layers & Colors

Plottter uses a multi-layer system that maps directly to multi-pen plotting: each layer
holds one pen's worth of paths and carries a designated pen color. Layers are managed in
the **Layer Panel** at the bottom-left of the main window.

---

## The Layer Panel

Each row in the layer list represents one layer:

| Element | Description |
|---------|-------------|
| **Color swatch** | Click to change the pen color (opens a color picker) |
| **Name** | Double-click to rename inline |
| **Eye icon** | Click to toggle visibility; hidden layers are not exported |
| **Lock icon** | Click to lock; locked layers cannot be edited or optimized |
| **Path count** | Number of polylines currently on this layer |

Layers are ordered top-to-bottom in the panel. The top layer is drawn last (on top) in the
preview; the bottom layer is drawn first (behind everything else).

---

## Creating and Removing Layers

| Action | How |
|--------|-----|
| Add empty layer | Click **+** (Add Layer button) |
| Delete layer | Select it and click **−** |
| Duplicate layer | Select it and click **Duplicate** |
| Rename | Double-click the name, or right-click → Rename |
| Reorder | Drag rows up or down |
| Move up / down | Select and click **↑** / **↓** |

Undo and redo (`Ctrl+Z` / `Ctrl+Y`) work for all layer operations.

---

## Assigning Pen Colors

Click the **color swatch** next to any layer to open the color picker. The hex color string
you choose is stored in the layer and appears in exported SVG files as the `stroke` attribute.

When plotting with a physical pen plotter, load the matching physical pen and assign it to
the correct layer. For example:

```
Layer 1  →  #000000 (black)  →  Pen 1 (black ink)
Layer 2  →  #FF0000 (red)    →  Pen 2 (red ink)
Layer 3  →  #0000FF (blue)   →  Pen 3 (blue ink)
```

Export each layer separately (File › Export All Layers) to get one SVG per pen.

---

## Merging and Duplicating

### Merge

Select two or more layers in the panel (Ctrl-click to multi-select), then click **Merge**.
All paths from the selected layers are combined into a single new layer. The original layers
are removed.

Alternatively, right-click and choose **Merge Selected** from the context menu.

### Duplicate

Click **Duplicate** to create a copy of the selected layer with a new ID. The duplicate
contains copies of all paths. This is useful for:

- Making a backup before applying a destructive optimization
- Creating multiple color variations of the same paths

---

## Visibility and Lock

- **Visible** layers are shown on the canvas and included in exports
- **Hidden** layers (`eye` toggled off) are excluded from all exports
- **Locked** layers cannot have paths added, removed, or optimized — useful for reference layers

The status bar path count and travel metrics count only visible, unlocked layers.

---

## Registration Marks

Registration marks are shared across all layers. They are controlled per-project:

- **Edit › Canvas Settings** — choose style: `corners`, `center`, or `both`
- **View › Toggle Registration Marks** (`R`) — toggle preview on the canvas
- Marks appear in all exported SVGs as thin 3 mm crosshairs at `stroke-width: 0.1mm`

Use registration marks to align multiple pen passes precisely. Print the marks from the first
layer, then align subsequent layers to them before clamping the paper.

---

## Color Separation

Color Separation mode automatically splits an image into multiple layers, one per color cluster
or channel. Access it by selecting **Color Separation** in the mode panel.

### Methods

#### K-Means Clustering

Groups pixels by perceptual color similarity (k-means in LAB color space). Produces the most
visually meaningful separation for full-color images.

| Parameter | Description |
|-----------|-------------|
| `num_colors` | Number of color clusters (layers) to create (2–16) |

After separation, Plottter creates one layer per cluster, named e.g. `Cluster 1 — #FF6B35`.
The layer color is set to the cluster's centroid color.

**Best for:** Full-color photos, illustrations with distinct color regions.

#### Luminance Bands

Splits a grayscale version of the image into brightness bands. The darkest band contains
the darkest pixels; the lightest band contains the lightest.

| Parameter | Description |
|-----------|-------------|
| `num_bands` | Number of brightness bands (2–8) |
| Threshold sliders | Drag to adjust band boundaries |

**Best for:** Black-and-white photos, images where tone matters more than hue.

#### RGB Channels

Splits the image into its Red, Green, and Blue channels. Produces three layers with
respective hex colors `#FF0000`, `#00FF00`, `#0000FF`.

**Best for:** RGB process printing simulation, artistic RGB decomposition.

#### CMYK Channels

Converts RGB to CMYK and splits into Cyan, Magenta, Yellow, and Key (Black) channels.
Produces four layers with standard process colors.

**Best for:** Simulating CMYK print-style separations for risograph-style plotter art.

---

## Generating Lines from Separation Layers

After separation, each layer holds a binary mask showing which pixels belong to it.
You can then generate line art for each layer independently:

1. Select a separation layer in the Layer Panel
2. Open the Settings Panel (it will show the Image-to-Lines controls with the layer's mask)
3. Choose a line generation algorithm (Hatching recommended for color separation)
4. Click **Generate Lines**

Plottter processes each layer in sequence, showing a progress bar. When complete, each
layer contains line-art paths that represent its color region.

Alternatively, use **Generate Lines (All Layers)** to process all separation layers at once
with the same algorithm settings.

---

## Tips for Multi-Layer Workflows

- Use **K-Means with 3–4 clusters** for most full-color photos
- For portraits: **Luminance with 3 bands** (shadows / midtones / highlights) combined with
  hatching produces classic engraving-style results
- Assign light colors (cyan, magenta) to layers that will be plotted with lighter-pressure pens;
  dark colors (black) to layers that will use more pressure
- Export all layers to separate SVGs and load them into Inkscape for final layout adjustments
  before sending to the plotter
