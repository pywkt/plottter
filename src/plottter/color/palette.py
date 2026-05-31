"""PenPalette dataclass and user-palette persistence helpers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


@dataclass(frozen=True)
class PenPalette:
    """A named list of pen colors for the Custom Palette separator.

    Frozen so palettes are hashable / safe to pass around. Mutation
    happens by constructing a new PenPalette with the updated fields.

    Attributes:
        name: Human-readable name. Used as the title in the picker UI
            and (slug-form) as the filename when persisted.
        colors: Tuple of "#RRGGBB" hex strings, uppercase. Order matters
            — it becomes the output layer order from palette_separate().
            Must be non-empty.
        description: Optional one-line description for the picker tooltip.
        source: Optional URL or note documenting where the colors came
            from (e.g. "Copic.com 2024 catalogue, B-row").
    """

    name: str
    colors: tuple[str, ...]
    description: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not self.colors:
            raise ValueError(f"PenPalette {self.name!r} must have at least one colour")
        for c in self.colors:
            if not _HEX_RE.match(c):
                raise ValueError(f"PenPalette {self.name!r}: invalid hex {c!r}")
        # Normalise to uppercase for stable equality + JSON round-trip.
        object.__setattr__(self, "colors", tuple(c.upper() for c in self.colors))

    @property
    def count(self) -> int:
        return len(self.colors)


def palette_to_dict(p: PenPalette) -> dict:
    return {
        "name": p.name,
        "colors": list(p.colors),
        "description": p.description,
        "source": p.source,
    }


def palette_from_dict(d: dict) -> PenPalette:
    return PenPalette(
        name=str(d["name"]),
        colors=tuple(str(c) for c in d["colors"]),
        description=str(d.get("description", "")),
        source=str(d.get("source", "")),
    )


def palette_slug(name: str) -> str:
    """Filename-safe slug for `name`. 'My Watercolours!' → 'my-watercolours'."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "palette"


def palette_dir() -> Path:
    """Return ~/.plottter/palettes/, creating it if needed."""
    p = Path.home() / ".plottter" / "palettes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_user_palette(p: PenPalette) -> Path:
    """Persist a user palette as `<palette_dir>/<slug>.json`. Returns the path."""
    fp = palette_dir() / f"{palette_slug(p.name)}.json"
    fp.write_text(json.dumps(palette_to_dict(p), indent=2))
    return fp


def load_user_palettes() -> list[PenPalette]:
    """Load every `*.json` in palette_dir(). Skips malformed files (logged)."""
    out: list[PenPalette] = []
    for fp in sorted(palette_dir().glob("*.json")):
        try:
            out.append(palette_from_dict(json.loads(fp.read_text())))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Bad palette %s: %s", fp, e)
    return out
