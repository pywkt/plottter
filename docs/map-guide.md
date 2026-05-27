# Map Generator Guide

Map mode turns a real-world location into pen-plotter-ready vector art — roads, water,
parks, rail, buildings, and coastline — with **one layer per feature category** so each
can be plotted in a different pen color.

All geographic data comes from [OpenStreetMap](https://www.openstreetmap.org/). No API
key is required.

---

## The Two-Step Workflow

Map generation is intentionally split into two separate operations, with an optional
positioning step in between:

```
Type a location → Fetch Map Data  →  (data cached to disk)
                                            │
    Adjust parameters ──────────────────────┤
                                            │
                              [Position Map]  ←  optional pan/zoom
                                            │
                                            ▼
                              Generate  →  Layers on canvas
```

**Step 1 — Fetch** hits the network once and saves the raw OSM geometry to
`~/.plottter/maps/`. This can take several seconds for large radii or busy Overpass
servers.

**Step 1.5 — Position (optional)** lets you pan and zoom interactively to frame the
exact portion of the downloaded area you want to plot. See
[Positioning the Map](#positioning-the-map) below.

**Step 2 — Generate** is fast and fully offline. It reprojects the cached data using
the current pan/zoom position, applies your current parameter choices (fill style, road
detail, which categories are on), clips to the printable area, and emits layers. You
can tweak parameters and regenerate as many times as you like without any further
network requests.

If you close and reopen the project, clicking **Fetch Map Data** again for the same
location and radius reads from the disk cache rather than hitting the network. The
saved pan/zoom position is restored once the data is loaded.

---

## Fetching Map Data

1. Select **Map** in the mode panel.
2. Type a location into the **Location** field — a city name, address, landmark, or any
   place recognisable by Nominatim (OpenStreetMap's geocoder):
   - `Kyoto, Japan`
   - `Brooklyn, NY`
   - `Canal Saint-Martin, Paris`
   - `1600 Pennsylvania Ave, Washington DC`
3. Set the **Radius (km)** parameter to control how much area to download (default 1.5 km).
4. Click **Fetch Map Data**.

The status label updates as each phase completes:

| Status text | Meaning |
|-------------|---------|
| `Geocoding…` | Resolving the place name to a latitude/longitude |
| `Downloading roads, water…` | Querying the Overpass API for OSM features |
| `Loaded: 1,240 features` | Fetch complete; data is cached and ready |
| *(red text)* | An error occurred — see the message for details |

Once the status shows a feature count, click **Generate** (`Ctrl+G`) to produce layers,
or click **Position Map** to frame the area interactively first.

---

## Positioning the Map

After fetching, you can pan and zoom the map preview interactively to frame exactly
which portion of the downloaded area to plot. This step is optional — if you skip it,
Generate fits all downloaded features inside the printable area automatically.

### Entering position mode

Click **Position Map** in the Map settings panel. The canvas switches to an interactive
overlay showing the fetched geometry as faded preview lines. A dashed rectangle marks
the printable area — everything inside that rectangle is what Generate will output.

To exit position mode without generating, click **Position Map** again to deactivate.

### Pan and zoom

| Action | Effect |
|--------|--------|
| **Drag** (left mouse button) | Pan the map |
| **Scroll wheel** | Zoom in or out; the geographic point under the cursor stays fixed |

Pan and zoom are **bounded by the extent of the fetched data**. You cannot scroll past
the edge of the downloaded area, and you cannot zoom out past the fit-to-canvas level.
If you want to reposition beyond the current boundary, increase **Radius (km)** and
click **Fetch Map Data** again to download a larger area.

### Reset to fit

Click **Reset to fit** to return to the default view where all downloaded features are
scaled to fill the printable area. Use this to recover from an unusable pan/zoom state.

### How cropping works

The dashed printable-area rectangle is the exact crop boundary used by Generate.
When you click **Generate**, the map data is reprojected using your current pan/zoom
position and clipped to the printable area. Features outside the rectangle are
discarded — the generated layers contain exactly what the preview shows, no more.

### Saving position with the project

The pan/zoom position is stored in the project file automatically as you adjust it.
When you reopen the project and click **Fetch Map Data** (which reads from the disk
cache), the saved position is restored and the canvas preview reflects it immediately.

---

## Location and Radius

### Radius mode (default)

`radius_km` draws a square of side `2 × radius_km` centred on the geocoded point.
This gives predictable, repeatable framing regardless of how Nominatim sizes the place
feature itself, which makes it well-suited for art prints.

| Radius | Good for |
|--------|----------|
| 0.5 km | A dense neighbourhood, a small town centre |
| 1.5 km | A typical city district (default) |
| 3–5 km | A large urban area or inner city |
| 8–10 km | A whole small city or island |

Large radii (> 5 km) with buildings enabled can be slow to fetch and generate.

### Place bbox mode

Set **Extent Mode** to `place_bbox` to use the bounding box Nominatim associates with
the place feature itself. This is useful for "the whole borough of Manhattan" or "the
island of Capri" where the natural boundary matters more than a fixed radius.

---

## Feature Categories

Each enabled, non-empty category becomes one output layer. Empty categories are silently
skipped so you never get blank layers.

| Toggle | Layer name | Geometry | What it draws | Default color |
|--------|-----------|----------|---------------|---------------|
| **Include Roads** | Roads (major) / Roads (minor) | lines | Road network, split by tier | black / dark grey |
| **Include Rail** | Rail | lines | Rail, light rail, subway, tram, monorail | brown |
| **Include Water (areas)** | Water | areas | Lakes, ponds, sea polygons | blue |
| **Include Waterways (lines)** | Waterways | lines | Rivers, streams, canals | blue |
| **Include Parks / Green Space** | Parks | areas | Parks, gardens, forests, meadows, cemeteries | green |
| **Include Buildings** | Buildings | areas | Building footprints | tan/brown |
| **Include Coastline** | Coastline | lines | Coastline ways | blue |

Areas are drawn as outlines by default; see [Area Fill](#area-fill) below.

### Road detail level

The **Road Detail** parameter controls which OSM highway tags feed the road layers:

| Setting | Roads included |
|---------|---------------|
| `major_only` | Motorway, trunk, primary, secondary (+ `_link` variants) |
| `standard` | Adds tertiary, residential, living street, unclassified (default) |
| `all_streets` | Also adds service roads, tracks, footways, paths, cycleways |

`major_only` produces clean, fast output for large areas. `all_streets` captures the
full pedestrian network but can be slow at larger radii.

### Major Road Strokes

Set **Major Road Strokes** to 2–4 to draw parallel offset copies of major roads. This
gives a double-stroke "transit map" emphasis effect.

---

## Area Fill

Area categories (Water, Parks, Buildings) are outlines by default. Change **Area Fill**
to add fill:

| Setting | Effect |
|---------|--------|
| `none` | Polygon outline only (default) |
| `hatch` | Parallel fill lines at **Fill Angle** degrees, spaced **Fill Spacing (mm)** apart, plus the outline |
| `cross_hatch` | Two perpendicular sets of hatch lines, plus the outline |

Fill is applied uniformly to all area categories. To hatch water but leave parks as
outline, generate twice with different settings and combine the layers manually.

**Fill Spacing (mm):** Distance between hatch lines (0.3–10 mm, default 2.0 mm). A
spacing close to your pen nib width produces a solid-looking fill; larger spacing gives
an open, airy texture.

**Fill Angle (deg):** Direction of hatch lines (0–180°, default 45°). 0° = horizontal,
90° = vertical.

---

## Presets

Four built-in presets cover common use cases:

| Preset | What it enables | Best for |
|--------|----------------|----------|
| **Minimal Streets** | Roads only, no fill | Fast, clean street maps; large-format prints |
| **Roads + Water** | Roads, water (hatched), waterways, coastline | Cities near rivers, lakes, or sea |
| **Full City** | All categories, no fill | Dense urban exploration; small radii |
| **Transit Map** | Major roads (double-stroke) + rail only | Schematic poster-style transit art |

Apply a preset from the **Preset** dropdown, then adjust individual parameters as needed.

---

## Parameters Reference

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `radius_km` | float | 0.2–10.0 | 1.5 | Radius of the map area in km (radius mode) |
| `extent_mode` | choice | `radius`, `place_bbox` | `radius` | How to frame the map |
| `road_detail` | choice | `major_only`, `standard`, `all_streets` | `standard` | Which road tiers to include |
| `include_roads` | bool | — | true | Draw road network |
| `include_rail` | bool | — | true | Draw rail/tram ways |
| `include_water` | bool | — | true | Draw water area polygons |
| `include_waterways` | bool | — | true | Draw rivers and canals |
| `include_parks` | bool | — | true | Draw green-space polygons |
| `include_buildings` | bool | — | **false** | Draw building footprints (off by default — slow for dense areas) |
| `include_coastline` | bool | — | true | Draw coastline ways |
| `area_fill` | choice | `none`, `hatch`, `cross_hatch` | `none` | Fill style for area polygons |
| `fill_spacing_mm` | float | 0.3–10.0 | 2.0 | Hatch line spacing (hatch/cross_hatch only) |
| `fill_angle_deg` | float | 0–180 | 45 | Hatch line angle in degrees (hatch/cross_hatch only) |
| `major_road_strokes` | int | 1–4 | 1 | Parallel strokes for major roads (>1 = emphasis) |
| `simplify_mm` | float | 0.0–2.0 | 0.15 | Douglas–Peucker tolerance; higher = smoother, fewer points |
| `min_feature_mm` | float | 0.0–10.0 | 0.8 | Drop polyline fragments shorter than this |
| `include_attribution` | bool | — | true | Emit ODbL attribution credit (see below) |

---

## Colors

Default layer colors match the feature category (blue for water, green for parks, brown
for rail and buildings, black/grey for roads). After generation, use the **Layer panel**
to change any layer's pen color — the map generator does not expose per-category color
parameters.

---

## Attribution (ODbL)

OpenStreetMap data is published under the
[Open Database Licence (ODbL)](https://www.openstreetmap.org/copyright). The licence
requires that any work produced from OSM data **visibly credits "© OpenStreetMap
contributors"**.

When **Include Attribution** is checked (the default), the generator appends a small
credit layer named **Attribution** containing the text "© OpenStreetMap contributors"
rendered as vector lines in the bottom margin. This credit is included in every SVG,
HPGL, and G-code export, and is also stored in the project metadata under
`map_attribution`.

**Do not uncheck Include Attribution** if you intend to publish or distribute the
plotted output. See the full licence at
<https://www.openstreetmap.org/copyright>.

---

## Overpass Endpoint Preference

The generator queries the Overpass API to download OSM geometry. The default endpoint is
`https://overpass-api.de/api/interpreter`. If you encounter repeated errors (429 Too
Many Requests or 504 Gateway Timeout) this usually means the default server is under
load.

To switch to a mirror:

1. Open **Preferences** (Edit → Preferences on Linux/Windows, Plottter → Preferences on
   macOS).
2. Find the **Overpass API Endpoint** field.
3. Enter an alternative endpoint URL, for example:
   - `https://overpass.kumi.systems/api/interpreter`
4. Click **OK**. All subsequent fetch operations use the new endpoint.

The setting is saved in application preferences and persists across sessions. Clearing
the field restores the default.

No API key is required for any Overpass endpoint.

---

## CLI Usage

Map generation works in headless batch mode. Pass the location and radius as extra
parameters on the command line:

```bash
plottter --generator "Map" \
         --param location="Kyoto, Japan" \
         --param radius_km=2.0 \
         --output kyoto.svg
```

When `location` is provided and no cached map data is available, the CLI fetches from
the network inline before generating. This requires an internet connection. The fetched
data is cached to `~/.plottter/maps/` so subsequent CLI runs with the same arguments
are fast and offline.

To include buildings:

```bash
plottter --generator "Map" \
         --param location="Paris, France" \
         --param include_buildings=true \
         --param area_fill=hatch \
         --output paris.svg
```

---

## Tips

**Start small.** Use radius 1.5 km or less for your first map to keep fetch and
generation fast. Increase once you know what the output looks like for your chosen city.

**Position before generating.** Use **Position Map** to zoom into a dense
neighbourhood or to place a landmark precisely inside the printable area. Pan/zoom
costs nothing — Generate is fully offline and regenerates from the same cached data.

**Buildings are slow.** Building footprints are the densest category. Enable them only
when you need the detail, and prefer small radii (≤ 2 km).

**Re-generate freely.** Changing any parameter and clicking Generate costs nothing —
the cached data is reused. Experiment with road detail, fill, and stroke count without
ever re-fetching.

**Layer colors.** After generation, select any layer in the Layer panel and change its
pen color. Roads in black, water in blue, parks in green is a classic starting palette.

**Hatch fill for printing.** Cross-hatch at 1.0–1.5 mm spacing produces a dense, almost
solid look for water bodies that reads well in single-pen plots. Wider spacing (3–5 mm)
gives a lighter texture.

**Coastal cities.** Enable **Include Coastline** for locations on the sea. Coastline ways
from Overpass appear as open lines; they frame the land edge without a filled polygon.

**Combine with other generators.** Map layers are just regular layers — you can add a
text label on top, overlay a math-art pattern as a background, or apply post-processing
(simplify, optimize) to the map paths independently.

---

## Known Limitations

- **Rail/transit** shows physical track ways only (no named route lines or colored
  routes by line).
- **Coastline** is drawn as open line segments, not as a filled land polygon.
- **Labels and street names** are not included; use the Text generator for place-name
  overlays.
- **True-scale output** (e.g. exactly 1:10 000 with a scale bar) is not yet supported —
  the map is fit-to-canvas.
- Re-opening a saved project does not restore the map data automatically; clicking
  **Fetch Map Data** again (which reads from the disk cache) is required before
  regenerating. The pan/zoom position **is** saved with the project and is restored
  automatically once you click Fetch Map Data.
