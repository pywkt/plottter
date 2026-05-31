"""Remove duplicate overlapping path segments (weld/union)."""

from __future__ import annotations

from typing import Callable

from plottter.models.path import Polyline
from plottter.processing._jit import njit


@njit(cache=True)
def _segments_match_jit(
    a0x, a0y, a1x, a1y,   # segment A endpoints
    b0x, b0y, b1x, b1y,   # segment B endpoints
    tol_sq,                # tolerance squared
):
    """Return True if segment (a0→a1) duplicates segment (b0→b1) within *tol_sq*.

    Checks both same-direction and reversed-direction duplicates.  Uses
    short-circuit evaluation: only computes the second distance if the first
    is within tolerance, halving the average work for non-duplicate pairs.

    This function is JIT-compiled by numba when available (``cache=True`` so
    compilation is reused across runs).  When numba is absent the decorator is
    a no-op and the function runs as ordinary Python.
    """
    # Same direction: a0≈b0 AND a1≈b1
    d00x = a0x - b0x
    d00y = a0y - b0y
    if d00x * d00x + d00y * d00y <= tol_sq:
        d11x = a1x - b1x
        d11y = a1y - b1y
        if d11x * d11x + d11y * d11y <= tol_sq:
            return True
    # Reversed direction: a0≈b1 AND a1≈b0
    d01x = a0x - b1x
    d01y = a0y - b1y
    if d01x * d01x + d01y * d01y <= tol_sq:
        d10x = a1x - b0x
        d10y = a1y - b0y
        if d10x * d10x + d10y * d10y <= tol_sq:
            return True
    return False


def weld_overlapping_paths(
    paths: list[Polyline],
    tolerance_mm: float = 0.1,
    cancelled_callback: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Polyline]:
    """Remove duplicate overlapping segments across polylines.

    When a segment (consecutive point pair) in one polyline is nearly identical
    to a segment in a previously-processed polyline (within *tolerance_mm*),
    the duplicate is removed from the later polyline.  Polylines that become
    empty after de-duplication are discarded.  Polylines split by removals
    produce multiple shorter fragments.

    The first path to claim a segment "wins" (processing order = input order).
    Both same-direction and reversed-direction duplicates are detected.

    A grid-based spatial index on segment midpoints is used for efficient
    candidate lookup, giving O(n) average-case performance rather than O(n²).

    The per-candidate segment comparison is handled by the JIT-compiled
    ``_segments_match_jit`` kernel (falls back to pure Python when numba is
    absent).

    Args:
        paths: Input list of polylines.
        tolerance_mm: Max endpoint distance for two segments to be considered
            duplicates.  Default 0.1 mm.
        cancelled_callback: Optional zero-argument callable returning True when
            the operation should abort.  Checked once per input path.
        progress_callback: Optional callable receiving (current_index, total)
            for progress reporting.  Called once per input path.

    Returns:
        New list of polylines with duplicate segments removed.  If cancelled,
        returns the partially processed result accumulated so far.
    """
    if len(paths) < 2:
        return list(paths)

    tol_sq = tolerance_mm * tolerance_mm
    # Grid cell size equals tolerance so neighbouring cells cover the full
    # search radius.  Use max(tolerance, 1e-9) to avoid division by zero.
    cell_size = max(tolerance_mm, 1e-9)
    total = len(paths)

    # Spatial index: grid_key → list of canonical segment indices
    # Each canonical segment stored as (p0, p1).
    _grid: dict[tuple[int, int], list[int]] = {}
    _canonical: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def _grid_key(x: float, y: float) -> tuple[int, int]:
        return (int(x / cell_size), int(y / cell_size))

    def _add_to_index(idx: int, p0: tuple[float, float], p1: tuple[float, float]) -> None:
        mx = (p0[0] + p1[0]) * 0.5
        my = (p0[1] + p1[1]) * 0.5
        key = _grid_key(mx, my)
        if key not in _grid:
            _grid[key] = []
        _grid[key].append(idx)

    def _is_duplicate(
        s0: tuple[float, float], s1: tuple[float, float]
    ) -> bool:
        """Check whether (s0→s1) duplicates any canonical segment.

        The grid lookup stays in Python (dict-of-int-keys is not JIT-able).
        The per-candidate comparison delegates to the JIT-compiled
        ``_segments_match_jit`` kernel.
        """
        mx = (s0[0] + s1[0]) * 0.5
        my = (s0[1] + s1[1]) * 0.5
        gx, gy = _grid_key(mx, my)
        # Check the 3×3 neighbourhood of grid cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (gx + dx, gy + dy)
                for cidx in _grid.get(cell, ()):
                    c0, c1 = _canonical[cidx]
                    if _segments_match_jit(
                        s0[0], s0[1], s1[0], s1[1],
                        c0[0], c0[1], c1[0], c1[1],
                        tol_sq,
                    ):
                        return True
        return False

    result: list[Polyline] = []
    _was_cancelled = False

    for i, path in enumerate(paths):
        if cancelled_callback is not None and cancelled_callback():
            # Preserve all unprocessed paths so no data is lost on cancel.
            result.extend(paths[i:])
            _was_cancelled = True
            break
        if progress_callback is not None:
            progress_callback(i, total)

        if len(path) < 2:
            continue

        n_segs = len(path) - 1

        # Determine which segments to keep (not duplicates of canonical)
        keep: list[bool] = []
        for k in range(n_segs):
            s0: tuple[float, float] = path[k]
            s1: tuple[float, float] = path[k + 1]
            keep.append(not _is_duplicate(s0, s1))

        # Add newly-kept segments to canonical set
        for k, kept in enumerate(keep):
            if kept:
                p0: tuple[float, float] = path[k]
                p1: tuple[float, float] = path[k + 1]
                idx = len(_canonical)
                _canonical.append((p0, p1))
                _add_to_index(idx, p0, p1)

        # Reassemble kept segments into one or more polyline fragments
        fragments = _reassemble(path, keep)
        result.extend(fragments)

    # Only emit completion progress when not cancelled; emitting max value
    # when cancelled would make the QProgressDialog auto-close and look like
    # the operation succeeded.
    if progress_callback is not None and not _was_cancelled:
        progress_callback(total, total)

    return result


def _reassemble(path: Polyline, keep: list[bool]) -> list[Polyline]:
    """Split *path* into contiguous fragments where *keep[i]* is True."""
    fragments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] | None = None

    for s_idx, kept in enumerate(keep):
        if kept:
            if current is None:
                current = [path[s_idx], path[s_idx + 1]]
            else:
                current.append(path[s_idx + 1])
        else:
            if current is not None and len(current) >= 2:
                fragments.append(current)
            current = None

    if current is not None and len(current) >= 2:
        fragments.append(current)

    return fragments
