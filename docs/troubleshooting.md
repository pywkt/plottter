# Troubleshooting

Common issues and how to fix them.

---

## Installation Issues

### `ModuleNotFoundError: No module named 'PyQt6'` (or other dependency)

**Cause:** Plottter was not installed with its dependencies, or the wrong Python environment
is active.

**Fix:**
```bash
# Make sure the virtual environment is active
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Re-install in editable mode with all dependencies
pip install -e ".[dev]"
```

---

### Blank window or EGL error on Linux

**Symptom:** The app starts but shows a blank white window, or crashes with:
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```
or
```
libEGL.so.1: cannot open shared object file
```

**Cause:** PyQt6 requires `libEGL.so.1` which may not be present on minimal Linux installs
or headless CI environments.

**Fix (using Chromium's bundled library):**
```bash
ln -s /opt/pw-browsers/chromium-1208/chrome-linux64/libEGL.so \
  .venv/lib/python3.12/site-packages/PyQt6/Qt6/lib/libEGL.so.1
```
Then launch with:
```bash
LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/PyQt6/Qt6/lib python -m plottter
```

---

### `ImportError: No module named 'cv2'`

**Cause:** `opencv-python` is not installed, or a headless build is installed instead.

**Fix:**
```bash
pip install opencv-python
```
If you already have `opencv-python-headless`, remove it first:
```bash
pip uninstall opencv-python-headless
pip install opencv-python
```

---

## Canvas and Preview Issues

### Blank canvas after Generate

**Symptom:** Clicking Generate completes (progress bar reaches 100%) but the canvas shows nothing.

**Possible causes and fixes:**

1. **Wrong target layer** — Check the **Target Layer** dropdown in the Settings Panel. If it
   points to a hidden or locked layer, paths are added but not shown. Toggle layer visibility.

2. **All paths outside canvas bounds** — Open the Layer Panel and check the path count badge.
   If it shows > 0 but nothing is visible, run **Tools › Clip to Canvas** to see if paths
   exist outside the drawing area.

3. **Expression evaluates to all-same-point** — If a parametric expression is constant (e.g.
   `x_expr = "0"`), all points collapse to one location. Open the generator and verify your
   expressions vary with `t`.

4. **Canvas is zoomed far out** — Try **Fit to Window** (`Ctrl+0`).

---

### Slow preview with many paths

**Symptom:** Canvas redraws slowly when there are thousands of paths.

**Fixes:**

1. Run **Tools › Optimize Current Layer** — simplification reduces point counts significantly
2. Run **Tools › Simplify Paths** with tolerance 0.5–1.0 mm
3. Hide layers you are not currently editing (eye icon in Layer Panel)
4. The app uses viewport culling automatically — try zooming in to a smaller area

---

### Canvas shows paths in wrong position

**Symptom:** Paths appear far from center, or outside the paper boundary.

**Cause:** Generator output was not auto-fitted, or post-generation transforms moved paths
outside bounds.

**Fix:** Run **Tools › Clip to Canvas** to remove out-of-bounds paths. Then re-generate
with default scale/offset settings to restore auto-fit behavior.

---

## Generation Issues

### Generate button does nothing (stays grey)

**Cause:** A generation thread is already running, or required inputs are missing.

**Fix:**
- Wait for the current generation to complete (check progress bar)
- For Image-to-Lines mode: make sure an image is loaded before clicking Generate
- If the button stays greyed indefinitely, try restarting the app

---

### Lorenz Attractor cannot be cancelled

**This is expected behavior.** The Lorenz preset uses SciPy ODE integration (`odeint`) which
cannot be interrupted. The cancellation flag is checked, but the blocking ODE call runs to
completion. For most hardware this takes < 5 seconds.

---

### Expression error: "Disallowed construct"

**Symptom:** Typing an expression like `t.real` or `__import__('os')` shows an error banner.

**Cause:** Plottter uses a restricted expression evaluator that only allows safe math operations.
Attribute access, imports, and most Python built-ins are blocked.

**Allowed:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `abs`, `sqrt`, `log`,
`log2`, `log10`, `exp`, `pow`, `floor`, `ceil`, `round`, `pi`, `e`, `tau`

---

## Export Issues

### SVG not loading in plotter software

**Symptom:** The exported SVG opens in Inkscape but the plotter software (e.g. AxiDraw
Inkscape extension) cannot read coordinates correctly.

**Possible causes:**

1. **Units mismatch** — Plottter SVGs use millimeter units. Some older plotter software
   assumes pixels (96 dpi). Import the SVG into Inkscape and use **File › Document Properties**
   to verify the document size matches your canvas dimensions in mm.

2. **Nested groups** — some plotter software cannot handle deeply nested `<g>` elements.
   Try the "Flatten Beziers / Simplify" option in Inkscape's AxiDraw extension settings.

3. **Fill attribute** — all Plottter polylines use `fill="none"`. If your plotter software
   only recognizes `fill` (not `stroke`) paths, they may appear invisible.

---

### Registration mark alignment issues

**Symptom:** The second pen pass does not align with the first.

**Possible causes:**

1. **Paper shifted** — if paper is not clamped, it can shift between passes. Use painter's tape
   or a paper jig.

2. **Registration marks not enabled** — check **Edit › Canvas Settings** and ensure
   registration marks are set to `corners` or `both`.

3. **Different canvas dimensions** — verify both SVGs have identical `width` and `height`
   attributes (open in a text editor or check Inkscape document properties).

---

### HPGL file has no output

**Symptom:** The `.hpgl` file is created but very small (< 100 bytes) or the plotter does
nothing.

**Cause:** The exported layer has no paths, or all paths have fewer than 2 points.

**Fix:** Check the path count badge in the Layer Panel. Run the generator again if needed,
then export.

---

## AI Features

### AI features not working (mask generation, background removal, depth maps)

**Symptom:** AI buttons in Mask Paint mode are greyed out, or clicking them shows "API key not
configured" / produces no result.

**Cause:** The Replicate API key has not been set, or the key is invalid.

**Fix:**
1. Open **Edit › Preferences**
2. Paste your Replicate API key into the **Replicate API Key** field and click **OK**
3. Verify the key is valid at [replicate.com](https://replicate.com) — keys start with `r8_`

> **No additional packages required.** Plottter uses the Replicate REST API directly. You do not
> need to run `pip install` anything extra to enable AI features.

### AI generation is slow or times out

**Cause:** Replicate runs models on cold-start cloud GPUs; the first request after a period of
inactivity may take 30–60 seconds.

**Fix:** Wait for the first request to complete — subsequent requests in the same session are
faster. If a request times out, click the AI button again to retry.

### AI results cache

Plottter caches AI results (depth maps, background removal, masks) on disk in `~/.plottter/ai_cache/` with subdirectories for each operation type. When you load the same source image again, cached results are restored automatically without an API call. A "(cached)" indicator appears next to the feature when a cached result is available.

To manage the cache:
1. Open **Edit › Preferences**
2. The **AI Cache** section shows the current cache size and directory
3. Click **Clear Cache** to remove all cached results

---

## AxiDraw USB Issues

### "pyaxidraw is NOT installed"

The AxiDraw Python API is not on PyPI — install it directly from Evil Mad Scientist Labs' hosted zip:

```bash
pip install https://cdn.evilmadscientist.com/dl/ad/public/AxiDraw_API.zip
```

For more information: <https://axidraw.com/doc/py_api/>

---

### "AxiDraw device not found"

**Symptom:** Clicking **Plot Now** shows "AxiDraw device not found. Make sure the device is
connected via USB and powered on."

**Fix:**
1. Check the USB cable — try a different port
2. Power cycle the AxiDraw (unplug, wait 5 seconds, reconnect)
3. On Linux, check USB permissions:
   ```bash
   ls -l /dev/ttyUSB*  # or /dev/ttyACM*
   sudo usermod -aG dialout $USER  # add yourself to the dialout group
   # log out and back in
   ```
4. Use **Preview mode** to verify settings before connecting a device

---

## Memory Usage with Large Images

High-resolution images (> 4000 × 4000 px) can use significant memory during processing.

**Fixes:**
1. Downsample before loading: use an image editor to reduce to 2000 × 2000 px
2. The **Crop to Canvas** preprocessing checkbox reduces the working image to match the canvas
   aspect ratio and a fixed resolution (5 px/mm × canvas dimensions)
3. Use **Stipple** with a lower `num_points` (2 000–5 000) for large images

---

## Known Limitations

| Limitation | Details |
|-----------|---------|
| Lorenz attractor cannot be cancelled | ODE solver runs to completion; usually < 5 seconds |
| L-system iteration > 7 may be slow | String length grows exponentially; keep ≤ 6 for complex rules |
| Stipple with 50 000 points takes ~60 seconds | Reduce `iterations` to 10 for faster results |
| pyaxidraw is a separate install | Not included in Plottter's default dependencies |
| CLI mode does not support image generators | Image-to-lines requires a loaded image, which the CLI does not handle |
| Undo does not cover canvas pan/zoom | Pan and zoom are view-only and not in the undo stack |

---

## Getting Help

If you encounter an issue not covered here:

1. Check the [GitHub Issues page](https://github.com/pywkt/plottter/issues) for known bugs
2. Search for your error message — it may already have a resolution
3. Open a new issue with:
   - Your operating system and Python version (`python --version`)
   - Steps to reproduce the problem
   - Any error messages from the terminal
