"""Hershey font catalog: registry, categories, legacy aliases, lazy loading.

The catalog is the single source of truth for which fonts ship and how
they are presented in the UI.  Fonts are loaded lazily on first use —
parsing all 32 SVGs eagerly costs ~30ms but matters at import time.

Three classes of names exist:

* **Canonical name** — the catalog key (e.g. ``"EMSReadability"``).
  This is what gets stored in saved projects and presets.
* **Display name** — human-readable label for the UI dropdown
  (e.g. ``"EMS Readability"``).  Never persisted.
* **Legacy alias** — the four old names (``"Simplex"``, ``"Duplex"``,
  ``"Script"``, ``"Gothic"``) preserved so projects created before this
  change keep loading.  Each maps to a real font.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .loader import Font, load_svg_font

_DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class FontEntry:
    """Catalog metadata for one font."""

    name: str          # canonical key
    display: str       # UI label
    category: str      # group: "EMS Modern" | "Hershey Sans" | …
    path: Path         # SVG file
    description: str = ""

    @property
    def relative_path(self) -> Path:
        return self.path.relative_to(_DATA_DIR)


# ---------------------------------------------------------------------------
# The registry — ordered so the UI dropdown lists fonts in a sensible order.
# Each (subdir, filename, canonical_name, display, category, description).
# ---------------------------------------------------------------------------
_REGISTRY: tuple[FontEntry, ...] = tuple(
    FontEntry(name=name, display=display, category=cat, path=_DATA_DIR / sub / fname, description=desc)
    for sub, fname, name, display, cat, desc in [
        # ----- EMS Modern (recommended for new work) -------------------------
        ("ems", "EMSReadability.svg",       "EMSReadability",       "EMS Readability",        "EMS Modern", "Designed for maximum legibility at small sizes — best default for map labels."),
        ("ems", "EMSReadabilityItalic.svg", "EMSReadabilityItalic", "EMS Readability Italic", "EMS Modern", "Italic companion to EMS Readability."),
        ("ems", "EMSOsmotron.svg",          "EMSOsmotron",          "EMS Osmotron",           "EMS Modern", "Rounded modern single-stroke sans."),
        ("ems", "EMSNixish.svg",            "EMSNixish",            "EMS Nixish",             "EMS Modern", "Geometric grotesque."),
        ("ems", "EMSNixishItalic.svg",      "EMSNixishItalic",      "EMS Nixish Italic",      "EMS Modern", "Italic companion to EMS Nixish."),
        ("ems", "EMSAllure.svg",            "EMSAllure",            "EMS Allure",             "EMS Modern", "Connected modern cursive script."),
        ("ems", "EMSFelix.svg",             "EMSFelix",             "EMS Felix",              "EMS Modern", "Friendly hand-drawn feel."),
        ("ems", "EMSElfin.svg",             "EMSElfin",             "EMS Elfin",              "EMS Modern", "Tall, narrow proportions."),
        ("ems", "EMSTech.svg",              "EMSTech",              "EMS Tech",               "EMS Modern", "Engineering / drafting style."),

        # ----- Hershey originals --------------------------------------------
        ("hershey", "HersheySans1.svg",          "HersheySans1",          "Hershey Sans 1-stroke",       "Hershey Sans",   "Original Hershey single-stroke sans."),
        ("hershey", "HersheySansMed.svg",        "HersheySansMed",        "Hershey Sans Medium",         "Hershey Sans",   "Medium-weight Hershey sans (multi-stroke)."),
        ("hershey", "HersheySerifMed.svg",       "HersheySerifMed",       "Hershey Serif Medium",        "Hershey Serif",  "Classic Hershey serif."),
        ("hershey", "HersheySerifMedItalic.svg", "HersheySerifMedItalic", "Hershey Serif Medium Italic", "Hershey Serif",  "Italic Hershey serif."),
        ("hershey", "HersheySerifBold.svg",      "HersheySerifBold",      "Hershey Serif Bold",          "Hershey Serif",  "Bold Hershey serif."),
        ("hershey", "HersheySerifBoldItalic.svg","HersheySerifBoldItalic","Hershey Serif Bold Italic",   "Hershey Serif",  "Bold italic Hershey serif."),
        ("hershey", "HersheyScript1.svg",        "HersheyScript1",        "Hershey Script 1-stroke",     "Hershey Script", "Original Hershey cursive."),
        ("hershey", "HersheyScriptMed.svg",      "HersheyScriptMed",      "Hershey Script Medium",       "Hershey Script", "Heavier Hershey cursive."),
        ("hershey", "HersheyGothEnglish.svg",    "HersheyGothEnglish",    "Hershey Gothic English",      "Hershey Gothic", "Blackletter / Old English."),

        # ----- Symbol fonts (extras from inkscapestrokefont set) ------------
        ("symbols", "HersheySymbolic.svg",       "HersheySymbolic",       "Hershey Symbolic",       "Symbols", "General symbol set incl. degree sign."),
        ("symbols", "HersheyMathLower.svg",      "HersheyMathLower",      "Hershey Math (lower)",   "Symbols", "Math operators — lower set."),
        ("symbols", "HersheyMathUpper.svg",      "HersheyMathUpper",      "Hershey Math (upper)",   "Symbols", "Math operators — upper set."),
        ("symbols", "HersheyMusic.svg",          "HersheyMusic",          "Hershey Music",          "Symbols", "Musical notation glyphs."),
        ("symbols", "HersheyMeteorology.svg",    "HersheyMeteorology",    "Hershey Meteorology",    "Symbols", "Weather symbols."),
        ("symbols", "HersheyAstrology.svg",      "HersheyAstrology",      "Hershey Astrology",      "Symbols", "Zodiac and planetary symbols."),
        ("symbols", "HersheyMarkers.svg",        "HersheyMarkers",        "Hershey Markers",        "Symbols", "Geometric markers / bullets."),
        ("symbols", "HersheyGreek1.svg",         "HersheyGreek1",         "Hershey Greek 1-stroke", "Symbols", "Greek alphabet (single stroke)."),
        ("symbols", "HersheyGreekMed.svg",       "HersheyGreekMed",       "Hershey Greek Medium",   "Symbols", "Greek alphabet (medium weight)."),
        ("symbols", "HersheyCyrillic.svg",       "HersheyCyrillic",       "Hershey Cyrillic",       "Symbols", "Cyrillic alphabet."),
        ("symbols", "HersheyJapanese.svg",       "HersheyJapanese",       "Hershey Japanese",       "Symbols", "Hiragana / Katakana."),

        # ----- Custom designs from Shriinivas' stroke-font extension --------
        ("custom",  "CustomScript.svg",          "CustomScript",          "Custom Script",          "Custom", "Hand-designed cursive."),
        ("custom",  "CustomSquareNormal.svg",    "CustomSquareNormal",    "Custom Square",          "Custom", "Hand-designed geometric."),
        ("custom",  "CustomSquareItalic.svg",    "CustomSquareItalic",    "Custom Square Italic",   "Custom", "Italic of Custom Square."),
    ]
)


# Legacy names from the old hand-coded _hershey.py.  Each maps to a real font
# so old projects open with an upgraded appearance rather than crashing.
_LEGACY_ALIASES: dict[str, str] = {
    "Simplex": "HersheySans1",
    "Duplex":  "HersheySansMed",
    "Script":  "HersheyScript1",
    "Gothic":  "HersheyGothEnglish",
}

#: Canonical name of the default font — chosen for max legibility at small sizes.
DEFAULT_FONT_NAME: str = "EMSReadability"


# ---------------------------------------------------------------------------
# Lookup + lazy-load API
# ---------------------------------------------------------------------------

_BY_NAME: dict[str, FontEntry] = {entry.name: entry for entry in _REGISTRY}
_LOAD_LOCK = Lock()
_LOADED: dict[str, Font] = {}


def resolve_name(name: str) -> str:
    """Translate a legacy alias to its canonical name.  Pass-through otherwise."""
    return _LEGACY_ALIASES.get(name, name)


def list_entries() -> tuple[FontEntry, ...]:
    """Return all catalog entries in display order."""
    return _REGISTRY


def list_names() -> list[str]:
    """Canonical names of every shipped font, in display order."""
    return [e.name for e in _REGISTRY]


def list_categories() -> list[str]:
    """Distinct category labels in display order."""
    seen: list[str] = []
    for e in _REGISTRY:
        if e.category not in seen:
            seen.append(e.category)
    return seen


def entries_by_category() -> dict[str, list[FontEntry]]:
    """Map category → list of entries, preserving registry order."""
    out: dict[str, list[FontEntry]] = {}
    for e in _REGISTRY:
        out.setdefault(e.category, []).append(e)
    return out


def get_entry(name: str) -> FontEntry:
    """Look up a :class:`FontEntry` by canonical name or legacy alias.

    Falls back to the default font when *name* is unknown — old projects
    that reference a deleted font still open, they just render in the
    default face.
    """
    canon = resolve_name(name)
    entry = _BY_NAME.get(canon)
    if entry is None:
        entry = _BY_NAME[DEFAULT_FONT_NAME]
    return entry


def choices_for_param(*, include_legacy_aliases: bool = True) -> tuple[list[str], dict[str, str]]:
    """Return ``(choices, descriptions)`` for a :class:`ChoiceParam` dropdown.

    The values are canonical font names (what gets stored in saved projects);
    descriptions include the category prefix so the UI tooltip groups visually.

    Legacy aliases (``"Simplex"``/``"Duplex"``/``"Script"``/``"Gothic"``) are
    appended at the end so projects saved before the catalog change keep
    validating against the choice list.  They are explicitly labelled as
    legacy in the description so users can migrate.
    """
    choices: list[str] = []
    descriptions: dict[str, str] = {}
    for entry in _REGISTRY:
        choices.append(entry.name)
        descriptions[entry.name] = f"{entry.category} — {entry.description}" if entry.description else entry.category
    if include_legacy_aliases:
        for alias, target in _LEGACY_ALIASES.items():
            choices.append(alias)
            descriptions[alias] = f"Legacy alias for {target}"
    return choices, descriptions


def load_font(name: str) -> Font:
    """Return the parsed :class:`Font` for *name*, loading + caching on first call.

    Thread-safe — the GUI loads fonts on the main thread but generators run
    in worker threads, so concurrent first-touch is possible.
    """
    entry = get_entry(name)
    cached = _LOADED.get(entry.name)
    if cached is not None:
        return cached
    with _LOAD_LOCK:
        cached = _LOADED.get(entry.name)
        if cached is None:
            cached = load_svg_font(entry.path, name=entry.name)
            _LOADED[entry.name] = cached
    return cached
