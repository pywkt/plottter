"""Enumerations and constants shared across canvas_widget sub-modules."""
from __future__ import annotations

import enum

# Mask resolution: pixels per millimetre (must match image-pipeline PX_PER_MM = 5)
_MASK_PX_PER_MM: int = 5


class MaskTool(str, enum.Enum):
    """Active mask-painting tool."""

    BRUSH = "brush"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"
    PEN = "pen"


class ShapeDrawTool(str, enum.Enum):
    """Active shape-drawing tool."""

    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    FREEHAND = "freehand"
    LINE = "line"
