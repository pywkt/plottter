"""Canvas dataclass representing paper size and drawing area."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_PRESETS: dict[str, tuple[float, float]] = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "A1": (594.0, 841.0),
    "A0": (841.0, 1189.0),
    "Letter": (215.9, 279.4),
    "Legal": (215.9, 355.6),
}


@dataclass
class Canvas:
    width_mm: float
    height_mm: float
    margin_mm: float = 10.0
    paper_preset: str = "Custom"

    def drawing_area(self) -> tuple[float, float, float, float]:
        """Return (left, top, right, bottom) of the drawable area in mm."""
        m = self.margin_mm
        return (m, m, self.width_mm - m, self.height_mm - m)

    @classmethod
    def from_preset(cls, name: str, margin: float = 10.0) -> "Canvas":
        """Create a Canvas from a named paper preset."""
        if name not in PAPER_PRESETS:
            raise ValueError(f"Unknown paper preset: {name!r}. Valid: {list(PAPER_PRESETS)}")
        width, height = PAPER_PRESETS[name]
        return cls(width_mm=width, height_mm=height, margin_mm=margin, paper_preset=name)
