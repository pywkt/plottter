"""SVG-font parser for Hershey + EMS single-stroke font files.

The vendored SVGs follow the SVG 1.1 ``<font>``/``<font-face>``/``<glyph>``
convention used by Inkscape's Hershey Text 3.x extension.  Glyph path data
is restricted to ``M x y`` (pen-up move) and ``L x y`` (pen-down line) —
no curves — which makes parsing a few dozen lines of stdlib code.

Font-face coordinates are baseline-relative with **y pointing up**;
``units-per-em`` is normally 1000 for the EMS/Hershey set.  Callers that
want millimetre coordinates should scale by ``size_mm / cap_height`` and
flip y for screen rendering.

This module performs **no normalisation** — every font keeps its native
coordinate system.  The compatibility shim in :mod:`plottter.fonts.hershey`
handles the legacy 21-unit cap-height convention used by older code.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_SVG_NS = {"s": "http://www.w3.org/2000/svg"}

# Tokeniser for ``d="..."`` attributes — only M/L commands + numbers.
_PATH_TOKEN = re.compile(r"[MLml]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


#: A single glyph: advance width in font units + list of stroke polylines.
#: Each polyline is a list of ``(x, y)`` tuples in font units (y-up).
@dataclass(frozen=True)
class Glyph:
    advance: float
    strokes: tuple[tuple[tuple[float, float], ...], ...]


#: Font metrics declared in ``<font-face>``.  All in font units, y-up.
@dataclass(frozen=True)
class FontMetrics:
    units_per_em: float
    ascent: float
    descent: float
    cap_height: float
    x_height: float
    default_advance: float  # ``<font horiz-adv-x>`` fallback when a glyph omits it


#: A parsed font: metrics + glyphs keyed by single-character ``unicode`` attr.
@dataclass(frozen=True)
class Font:
    name: str
    family: str
    metrics: FontMetrics
    glyphs: dict[str, Glyph] = field(hash=False)

    def glyph(self, char: str) -> Glyph | None:
        """Return the glyph for *char* (no fallback).  ``None`` if missing."""
        return self.glyphs.get(char)


def _parse_path(d: str) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Parse an ``M x y L x y …`` path into a tuple of polylines.

    Empty paths (whitespace-only, the ``space`` glyph) yield an empty tuple.
    Single-point sub-paths are dropped — they are pen-up moves with no
    following ``L`` and would render as zero-length strokes.

    Lowercase ``m``/``l`` (relative) variants are tolerated even though the
    vendored files never use them; this keeps the parser robust against
    third-party SVG fonts a user may drop in.
    """
    if not d or not d.strip():
        return ()
    tokens = _PATH_TOKEN.findall(d)
    polylines: list[tuple[tuple[float, float], ...]] = []
    current: list[tuple[float, float]] = []
    pen = (0.0, 0.0)  # for relative m/l
    i, n = 0, len(tokens)
    last_cmd = ""
    while i < n:
        tok = tokens[i]
        if tok in ("M", "m", "L", "l"):
            cmd = tok
            i += 1
        else:
            # Implicit repetition: after M/m subsequent coord pairs are L/l;
            # after L/l subsequent pairs are L/l.
            cmd = "L" if last_cmd in ("M", "L") else "l" if last_cmd in ("m", "l") else "L"
        if i + 1 >= n:
            break
        x, y = float(tokens[i]), float(tokens[i + 1])
        i += 2
        if cmd in ("m", "l"):
            x, y = pen[0] + x, pen[1] + y
        if cmd in ("M", "m"):
            if len(current) >= 2:
                polylines.append(tuple(current))
            current = [(x, y)]
        else:  # L / l
            current.append((x, y))
        pen = (x, y)
        last_cmd = cmd
    if len(current) >= 2:
        polylines.append(tuple(current))
    return tuple(polylines)


def load_svg_font(path: str | Path, *, name: str | None = None) -> Font:
    """Parse an SVG font file and return a :class:`Font`.

    Raises :class:`ValueError` if the file lacks a ``<font-face>`` element
    or contains no glyphs.

    ``name`` defaults to the file stem; this is the key used in the
    catalog and the value stored on layer presets.
    """
    path = Path(path)
    root = ET.parse(path).getroot()

    font_el = root.find(".//s:font", _SVG_NS)
    face_el = root.find(".//s:font-face", _SVG_NS)
    if face_el is None:
        raise ValueError(f"{path}: missing <font-face> element")

    units_per_em = float(face_el.get("units-per-em", "1000"))
    metrics = FontMetrics(
        units_per_em=units_per_em,
        ascent=float(face_el.get("ascent", units_per_em * 0.8)),
        descent=float(face_el.get("descent", -units_per_em * 0.2)),
        cap_height=float(face_el.get("cap-height", units_per_em * 0.5)),
        x_height=float(face_el.get("x-height", units_per_em * 0.3)),
        default_advance=float(font_el.get("horiz-adv-x", units_per_em * 0.5)) if font_el is not None else units_per_em * 0.5,
    )

    glyphs: dict[str, Glyph] = {}
    for g_el in root.findall(".//s:glyph", _SVG_NS):
        ch = g_el.get("unicode")
        if not ch or len(ch) != 1:
            # Multi-codepoint ligatures and the empty space glyph end up here;
            # we keep single-char entries only.  Space is handled below.
            continue
        advance = float(g_el.get("horiz-adv-x", metrics.default_advance))
        glyphs[ch] = Glyph(advance=advance, strokes=_parse_path(g_el.get("d", "")))

    if not glyphs:
        raise ValueError(f"{path}: no <glyph unicode=...> elements found")

    # Some fonts emit the space glyph as ``<glyph unicode=" " />`` with no
    # ``d`` attribute — the loop above already catches that.  Synthesise one
    # if missing so callers always get an advance for whitespace.  This must
    # run *after* the empty-font check so a truly empty font still raises.
    if " " not in glyphs:
        glyphs[" "] = Glyph(advance=metrics.default_advance, strokes=())

    return Font(
        name=name or path.stem,
        family=(face_el.get("font-family") or path.stem),
        metrics=metrics,
        glyphs=glyphs,
    )
