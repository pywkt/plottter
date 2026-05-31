"""Background QThread workers for MainWindow operations."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from plottter.models.path import Polyline


class _WeldWorker(QThread):
    """QThread that runs weld_overlapping_paths on a layer's paths."""

    finished = pyqtSignal(list, int, int)  # (new_paths, before_count, after_count)
    progress = pyqtSignal(int, int)        # (current_index, total)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        paths: list[Polyline],
        tolerance_mm: float = 0.1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._tolerance_mm = tolerance_mm
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from plottter.processing.weld import weld_overlapping_paths

            before_count = len(self._paths)
            new_paths = weld_overlapping_paths(
                self._paths,
                tolerance_mm=self._tolerance_mm,
                cancelled_callback=lambda: self._cancelled,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
            )
            if self._cancelled:
                self.cancelled.emit()
            else:
                after_count = len(new_paths)
                self.finished.emit(new_paths, before_count, after_count)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _OptimizeWorker(QThread):
    """QThread that runs the full path optimization pipeline on a layer's paths."""

    finished = pyqtSignal(list, float, float, int, int)  # (new_paths, before_travel, after_travel, before_lifts, after_lifts)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100 within-layer progress

    def __init__(
        self,
        paths: list[Polyline],
        run_weld: bool = False,
        weld_tolerance: float = 0.1,
        run_simplify: bool = True,
        simplify_tolerance: float = 0.1,
        run_filter: bool = True,
        filter_min_length: float = 0.5,
        run_clip: bool = True,
        clip_bounds: tuple[float, float, float, float] | None = None,
        run_merge: bool = True,
        merge_threshold: float = 0.5,
        run_join: bool = False,
        join_threshold: float = 0.1,
        run_2opt: bool = True,
        run_3opt: bool = False,
        run_or_opt: bool = True,
        num_starts: int = 5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._run_weld = run_weld
        self._weld_tolerance = weld_tolerance
        self._run_simplify = run_simplify
        self._simplify_tolerance = simplify_tolerance
        self._run_filter = run_filter
        self._filter_min_length = filter_min_length
        self._run_clip = run_clip
        self._clip_bounds = clip_bounds
        self._run_merge = run_merge
        self._merge_threshold = merge_threshold
        self._run_join = run_join
        self._join_threshold = join_threshold
        self._run_2opt = run_2opt
        self._run_3opt = run_3opt
        self._run_or_opt = run_or_opt
        self._num_starts = num_starts
        self._cancelled = False

    def request_stop(self) -> None:
        """Request cancellation.  The worker will stop at the next safe checkpoint."""
        self._cancelled = True

    def run(self) -> None:
        try:
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

            paths = self._paths
            before_travel = calculate_travel_distance(paths)
            before_lifts = len(paths)

            # --- Preprocessing steps (0-10%) ---
            self.progress.emit(0)
            if self._run_weld:
                paths = weld_overlapping_paths(paths, tolerance_mm=self._weld_tolerance)
            if self._run_simplify:
                paths = simplify_paths(paths, tolerance_mm=self._simplify_tolerance)
            if self._run_filter:
                paths = filter_short_paths(paths, min_length_mm=self._filter_min_length)
            if self._run_clip and self._clip_bounds is not None:
                paths = clip_to_bounds(paths, self._clip_bounds)
            if self._run_merge:
                paths = merge_nearby_paths(paths, threshold_mm=self._merge_threshold)
            if self._run_join:
                paths = join_at_junctions(paths, threshold_mm=self._join_threshold)
            self.progress.emit(10)

            if self._cancelled:
                after_travel = calculate_travel_distance(paths)
                self.finished.emit(paths, before_travel, after_travel, before_lifts, len(paths))
                return

            # --- Nearest-neighbour reordering (10-35%) ---
            def _nn_progress(f: float) -> None:
                self.progress.emit(10 + int(f * 25))

            paths = reorder_paths(
                paths,
                num_starts=self._num_starts,
                progress_callback=_nn_progress,
                cancelled=lambda: self._cancelled,
            )
            self.progress.emit(35)

            if self._run_2opt and not self._cancelled:
                # --- 2-opt improvement (35-55%) ---
                def _2opt_progress(f: float) -> None:
                    self.progress.emit(35 + int(f * 20))

                paths = optimize_2opt(
                    paths,
                    progress_callback=_2opt_progress,
                    cancelled=lambda: self._cancelled,
                )
                self.progress.emit(55)

            if self._run_3opt and not self._cancelled:
                # --- 3-opt improvement (55-75%) ---
                def _3opt_progress(f: float) -> None:
                    self.progress.emit(55 + int(f * 20))

                paths = optimize_3opt(
                    paths,
                    progress_callback=_3opt_progress,
                    cancelled=lambda: self._cancelled,
                )
                self.progress.emit(75)

            if self._run_or_opt and not self._cancelled:
                # --- Or-opt improvement (75-100%) ---
                def _oropt_progress(f: float) -> None:
                    self.progress.emit(75 + int(f * 25))

                paths = optimize_or_opt(
                    paths,
                    progress_callback=_oropt_progress,
                    cancelled=lambda: self._cancelled,
                )
                self.progress.emit(100)

            after_travel = calculate_travel_distance(paths)
            after_lifts = len(paths)
            self.finished.emit(paths, before_travel, after_travel, before_lifts, after_lifts)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _BrushWorker(QThread):
    """QThread that runs apply_brush on a layer's paths."""

    finished = pyqtSignal(list)  # (new_paths,)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100

    def __init__(
        self,
        paths: list[Polyline],
        brush_type: str,
        params: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._brush_type = brush_type
        self._params = params

    def run(self) -> None:
        try:
            from plottter.processing.brush import apply_brush
            self.progress.emit(10)
            result = apply_brush(self._paths, self._brush_type, self._params)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _TaperWorker(QThread):
    """QThread that runs taper_paths on a layer's paths."""

    finished = pyqtSignal(list)  # (new_paths,)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100

    def __init__(
        self,
        paths: list[Polyline],
        params: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._params = params

    def run(self) -> None:
        try:
            from plottter.processing.taper import taper_paths
            self.progress.emit(10)
            result = taper_paths(self._paths, **self._params)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _OffsetWorker(QThread):
    """QThread that runs offset_paths on a layer's paths."""

    finished = pyqtSignal(list)  # (new_paths,)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100

    def __init__(
        self,
        paths: list[Polyline],
        params: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._params = params

    def run(self) -> None:
        try:
            from plottter.processing.offset import offset_paths
            self.progress.emit(10)
            result = offset_paths(self._paths, **self._params)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
