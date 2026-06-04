"""Shared optimization pipeline.

The full Tools → Optimize Current Layer flow is run from three places:
the local QThread worker (``_OptimizeWorker``), the headless CLI
(``plottter --optimize``), and the remote-optimize worker that ships
paths over SSH to a faster machine. All three call
``run_optimization_pipeline`` so they stay in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from plottter.models import Polyline


@dataclass
class OptimizationResult:
    paths: list[Polyline]
    before_travel: float
    after_travel: float
    before_lifts: int
    after_lifts: int


# Default settings mirror ``_OptimizeWorker.__init__`` defaults — callers
# pass a partial dict and missing keys fall back to these.
_DEFAULT_SETTINGS: dict = {
    "run_weld": False,
    "weld_tolerance": 0.1,
    "run_simplify": True,
    "simplify_tolerance": 0.1,
    "run_filter": True,
    "filter_min_length": 0.5,
    "run_clip": True,
    "run_merge": True,
    "merge_threshold": 0.5,
    "run_join": False,
    "join_threshold": 0.1,
    "run_2opt": True,
    "run_3opt": False,
    "run_or_opt": True,
    "num_starts": 5,
}


def run_optimization_pipeline(
    paths: list[Polyline],
    settings: dict | None = None,
    clip_bounds: tuple[float, float, float, float] | None = None,
    generator_info: dict | None = None,
    progress_callback: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> OptimizationResult:
    """Run the full optimization pipeline on ``paths``.

    Parameters mirror :class:`_OptimizeWorker`. ``progress_callback`` receives
    integer 0..100. ``cancelled`` is polled at safe checkpoints; if it
    returns ``True``, the function returns early with whatever paths it
    has so far. The travel metrics in the returned result are still
    accurate for that partial state.

    Map layers (``generator_info["_generator_name"] == "Map"``) get the
    Join step force-enabled — this matches the per-layer policy in
    ``_OptimizeWorker`` so the CLI / remote paths behave identically.
    """
    from plottter.processing import (
        weld_overlapping_paths,
        simplify_paths,
        filter_short_paths,
        clip_to_bounds,
        merge_nearby_paths,
        join_at_junctions,
        reorder_paths,
        optimize_2opt,
        optimize_3opt,
        optimize_or_opt,
        calculate_travel_distance,
    )

    s = dict(_DEFAULT_SETTINGS)
    if settings:
        s.update(settings)

    # Force Join on for map layers (parity with _OptimizeWorker).
    is_map = (
        isinstance(generator_info, dict)
        and generator_info.get("_generator_name") == "Map"
    )
    if is_map and not s.get("run_join"):
        s["run_join"] = True

    def _is_cancelled() -> bool:
        return bool(cancelled()) if cancelled is not None else False

    def _emit(value: int) -> None:
        if progress_callback is not None:
            progress_callback(value)

    before_travel = calculate_travel_distance(paths)
    before_lifts = len(paths)

    # --- Preprocessing (0–10%) ---
    _emit(0)
    if s["run_weld"]:
        paths = weld_overlapping_paths(paths, tolerance_mm=s["weld_tolerance"])
    if s["run_simplify"]:
        paths = simplify_paths(paths, tolerance_mm=s["simplify_tolerance"])
    if s["run_filter"]:
        paths = filter_short_paths(paths, min_length_mm=s["filter_min_length"])
    if s["run_clip"] and clip_bounds is not None:
        paths = clip_to_bounds(paths, clip_bounds)
    if s["run_merge"]:
        paths = merge_nearby_paths(paths, threshold_mm=s["merge_threshold"])
    if s["run_join"]:
        paths = join_at_junctions(paths, threshold_mm=s["join_threshold"])
    _emit(10)

    if _is_cancelled():
        return OptimizationResult(
            paths=paths,
            before_travel=before_travel,
            after_travel=calculate_travel_distance(paths),
            before_lifts=before_lifts,
            after_lifts=len(paths),
        )

    # --- Nearest-neighbour reordering (10–35%) ---
    paths = reorder_paths(
        paths,
        num_starts=s["num_starts"],
        progress_callback=lambda f: _emit(10 + int(f * 25)),
        cancelled=_is_cancelled,
    )
    _emit(35)

    if s["run_2opt"] and not _is_cancelled():
        paths = optimize_2opt(
            paths,
            progress_callback=lambda f: _emit(35 + int(f * 20)),
            cancelled=_is_cancelled,
        )
        _emit(55)

    if s["run_3opt"] and not _is_cancelled():
        paths = optimize_3opt(
            paths,
            progress_callback=lambda f: _emit(55 + int(f * 20)),
            cancelled=_is_cancelled,
        )
        _emit(75)

    if s["run_or_opt"] and not _is_cancelled():
        paths = optimize_or_opt(
            paths,
            progress_callback=lambda f: _emit(75 + int(f * 25)),
            cancelled=_is_cancelled,
        )
        _emit(100)

    return OptimizationResult(
        paths=paths,
        before_travel=before_travel,
        after_travel=calculate_travel_distance(paths),
        before_lifts=before_lifts,
        after_lifts=len(paths),
    )
