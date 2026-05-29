# Font Sources & Licensing

All vendored font files are public-domain or OFL-licensed.  This directory
ships with the licence text in `OFL.txt` (and `OFL-inkscapestrokefont.txt`
for the Shriinivas extension's redistribution).

## ems/

The nine **EMS** fonts come from the
[Inkscape *Hershey Text* extension](https://gitlab.com/oskay/hershey-text)
maintained by Windell H. Oskay (Evil Mad Scientist Laboratories).
Designed as modern, plotter-quality replacements for the original Hershey
set — see the package README for design notes.

* License: SIL Open Font License 1.1 (OFL.txt)
* Upstream commit / files: `svg_fonts/EMS*.svg`

## hershey/

The nine **Hershey** SVG fonts come from the same Inkscape *Hershey Text*
package.  Glyph data ultimately derives from Dr. A. V. Hershey's original
vector fonts created at the U.S. National Bureau of Standards in 1967.
James Hurt at Cognition Inc. recoded the data; Marty McGuire converted
the result to SVG ([Thingiverse #6168](https://www.thingiverse.com/thing:6168));
Windell Oskay re-normalised it into proper SVG-font files in 2019.

* License: Hershey distribution terms — free use, commercial or
  otherwise, provided the originator credits are preserved (full text
  in the SVG metadata of each file)
* OFL companion: SIL OFL 1.1 (OFL.txt)

## symbols/

The eleven **symbol** fonts (math upper/lower, music, meteorology,
astrology, markers, Greek 1-stroke / medium, Cyrillic, Japanese,
Symbolic) come from the
[*Inkscape Stroke Font* extension](https://github.com/Shriinivas/inkscapestrokefont)
by Shriinivas Khandkar, which redistributes Hershey-derived data in the
same SVG-font format used by the EMS extension.

* License: Hershey distribution terms + SIL OFL 1.1
  (OFL-inkscapestrokefont.txt)

## custom/

The three **Custom** fonts (Script, Square, Square Italic) are original
designs bundled with the Shriinivas extension.

* License: SIL OFL 1.1 (OFL-inkscapestrokefont.txt)

---

## Why we ship the SVGs in-tree

These files are small (≈3 MB total) and rarely change.  Vendoring them
keeps Plottter a single `pip install` away from a working install, with
no network round-trip to fetch fonts on first use.
