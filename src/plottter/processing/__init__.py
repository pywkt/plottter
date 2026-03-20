"""Path post-processing and optimization utilities."""

from plottter.processing.simplify import simplify_paths, simplify_polyline
from plottter.processing.filter import filter_short_paths
from plottter.processing.clip import clip_to_bounds
from plottter.processing.merge import merge_nearby_paths, merge_fragments
from plottter.processing.optimize import (
    reorder_paths,
    optimize_2opt,
    optimize_or_opt,
    calculate_travel_distance,
)
from plottter.processing.weld import weld_overlapping_paths
from plottter.processing.curves import fit_curves
from plottter.processing.rasterize import rasterize_layer
from plottter.processing.brush import apply_brush
from plottter.processing.taper import taper_paths

__all__ = [
    "simplify_paths",
    "simplify_polyline",
    "filter_short_paths",
    "clip_to_bounds",
    "merge_nearby_paths",
    "merge_fragments",
    "reorder_paths",
    "optimize_2opt",
    "optimize_or_opt",
    "calculate_travel_distance",
    "weld_overlapping_paths",
    "fit_curves",
    "rasterize_layer",
    "apply_brush",
    "taper_paths",
]
