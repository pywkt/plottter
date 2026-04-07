"""canvas_widget package — zoomable/pannable vector canvas for plottter."""
from .widget import CanvasWidget
from .enums import MaskTool, ShapeDrawTool, _MASK_PX_PER_MM

__all__ = ["CanvasWidget", "MaskTool", "ShapeDrawTool", "_MASK_PX_PER_MM"]
