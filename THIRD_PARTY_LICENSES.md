# Third-Party Licenses

Plottter bundles a handful of third-party assets.  This file is the index;
full licence text lives next to each asset in the source tree.

## Hershey + EMS single-stroke fonts

**Location:** `src/plottter/fonts/hershey/data/`
**Used by:** the Text generator, OSM map labels, ASCII Art generator,
calligraphy plugin, and calibration plots.

| Subdirectory | Origin | Licence |
|---|---|---|
| `ems/` | [Inkscape Hershey Text](https://gitlab.com/oskay/hershey-text) by Windell H. Oskay (Evil Mad Scientist) | SIL Open Font License 1.1 — `data/OFL.txt` |
| `hershey/` | Same upstream; glyph data ultimately derives from Dr. A. V. Hershey's 1967 NBS vector fonts via James Hurt (Cognition Inc.) and Marty McGuire's SVG conversion | Hershey terms (free use, attribution required) + SIL OFL 1.1 — `data/OFL.txt` |
| `symbols/` | [Inkscape Stroke Font](https://github.com/Shriinivas/inkscapestrokefont) by Shriinivas Khandkar — Hershey-derived symbol sets | Hershey terms + SIL OFL 1.1 — `data/OFL-inkscapestrokefont.txt` |
| `custom/` | Same Shriinivas extension; original designs not derived from Hershey | SIL OFL 1.1 — `data/OFL-inkscapestrokefont.txt` |

See `src/plottter/fonts/hershey/data/SOURCES.md` for the full per-font
provenance and the design-rationale notes from each upstream.

### Required attribution

The Hershey distribution terms ask that re-distributors preserve the
following credit (paraphrased from the metadata embedded in every
Hershey SVG):

> The Hershey Fonts were originally created by Dr. A. V. Hershey while
> working at the U.S. National Bureau of Standards.  The format of the
> font data was created by James Hurt of Cognition Inc.  SVG conversion
> by Marty McGuire (Thingiverse #6168); SVG-font packaging by Windell H.
> Oskay (Evil Mad Scientist Laboratories, 2019).
