"""Per-layer cached ``QPainterPath`` rendering geometry (canvas-performance §6).

The canvas previously redrew every layer by iterating its polylines point-by-point
in Python on each paint event. ``LayerPathCache`` instead builds one
``QPainterPath`` per layer (in **mm** coordinates) and reuses it across frames,
letting Qt do the per-point work in C++ and clip in the viewport transform.

See ``specs/canvas-performance.md`` §6 for the full design. This module covers
§6.1 (the per-layer data model) and §6.5 (the travel-line path cache); GUI
wiring lives in the canvas widget.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from numpy.random import default_rng

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QPainter, QPainterPath, QPixmap

from plottter.models.layer import Layer

if TYPE_CHECKING:
    from plottter.gui.canvas_widget import CanvasWidget
    from plottter.models.project import Project


@dataclass
class Chunk:
    """A spatial slice of a layer's geometry with two draw granularities.

    ``path`` is the chunk's combined geometry (one ``drawPath`` for the fast
    hairline-stroker case, pen ≤ 1 device px). ``polys`` holds the same
    geometry as per-polyline ``(bbox, path)`` pairs for the wide-pen case,
    where Qt's stroke cost is superlinear in per-call path size and tight
    per-polyline culling pays for itself.

    Bboxes are ``(min_x, min_y, max_x, max_y)`` in mm over the chunk's actual
    (possibly jittered) coordinates. Stored as plain floats — NOT ``QRectF`` —
    because dot chunks can be zero-height (0.01 mm segments) and
    ``QRectF.intersects`` treats empty rects as intersecting nothing, which
    would cull every dot. Callers test overlap with simple interval comparisons.
    """

    bbox: tuple[float, float, float, float]
    path: QPainterPath
    polys: list[tuple[tuple[float, float, float, float], QPainterPath]]


#: Polylines per chunk to aim for when bucketing. Qt's wide-pen stroke cost
#: (device width > 1 px) is superlinear in the size of a single drawPath call
#: — measured ~83× cliff at exactly 1 px, and a 54k-point path strokes no
#: faster than a 120k-point one. Small spatially-bucketed chunks keep each
#: call cheap AND let the zoomed-in draw cull off-screen geometry, while at
#: hairline widths (fit zoom) the per-call overhead is negligible (~27 ms vs
#: 26 ms single-path at 120k points).
_TARGET_POLYLINES_PER_CHUNK = 24
_MAX_GRID = 32


def _build_chunks(
    polylines: list, disp=None  # type: ignore[no-untyped-def]
) -> list[Chunk]:
    """Bucket *polylines* into spatial grid chunks (shared chunking core).

    ``disp`` is an optional ``(n_points, 2)`` displacement array applied in
    polyline order (the baked-jitter variant); bboxes are computed from the
    displaced coordinates so culling needs no jitter margin. Polylines are
    never split across chunks (assignment is by bbox centre; the chunk bbox is
    the union of member bboxes), so no AA seams can appear inside a stroke.
    """
    if not polylines:
        return []

    # Displaced coordinates + per-polyline bbox, single pass.
    coords: list[list[tuple[float, float]]] = []
    bboxes: list[tuple[float, float, float, float]] = []
    k = 0
    for polyline in polylines:
        if disp is None:
            pts = [(float(x), float(y)) for x, y in polyline]
        else:
            pts = [
                (x + disp[k + i, 0], y + disp[k + i, 1])
                for i, (x, y) in enumerate(polyline)
            ]
            k += len(polyline)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        coords.append(pts)
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))

    overall_min_x = min(b[0] for b in bboxes)
    overall_min_y = min(b[1] for b in bboxes)
    overall_max_x = max(b[2] for b in bboxes)
    overall_max_y = max(b[3] for b in bboxes)
    span_x = max(overall_max_x - overall_min_x, 1e-9)
    span_y = max(overall_max_y - overall_min_y, 1e-9)

    n = len(polylines)
    grid = max(1, min(_MAX_GRID, int((n / _TARGET_POLYLINES_PER_CHUNK) ** 0.5 + 0.999)))

    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (bx0, by0, bx1, by1) in enumerate(bboxes):
        cx = (bx0 + bx1) / 2.0
        cy = (by0 + by1) / 2.0
        gx = min(grid - 1, int((cx - overall_min_x) / span_x * grid))
        gy = min(grid - 1, int((cy - overall_min_y) / span_y * grid))
        buckets.setdefault((gx, gy), []).append(i)

    chunks: list[Chunk] = []
    for cell in sorted(buckets):  # sorted → deterministic chunk order
        members = buckets[cell]
        path = QPainterPath()
        polys: list[tuple[tuple[float, float, float, float], QPainterPath]] = []
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for i in members:
            pts = coords[i]
            path.moveTo(pts[0][0], pts[0][1])
            poly_path = QPainterPath()
            poly_path.moveTo(pts[0][0], pts[0][1])
            for x, y in pts[1:]:
                path.lineTo(x, y)
                poly_path.lineTo(x, y)
            bx0, by0, bx1, by1 = bboxes[i]
            polys.append(((bx0, by0, bx1, by1), poly_path))
            min_x = min(min_x, bx0)
            min_y = min(min_y, by0)
            max_x = max(max_x, bx1)
            max_y = max(max_y, by1)
        chunks.append(
            Chunk(bbox=(min_x, min_y, max_x, max_y), path=path, polys=polys)
        )
    return chunks


def build_layer_chunks(layer: Layer) -> list[Chunk]:
    """Build spatially-bucketed chunks of a layer's polylines (mm coords).

    Same geometry as :func:`build_layer_path` (polylines shorter than two
    points are skipped), split into grid chunks so zoomed-in draws can cull and
    Qt's superlinear wide-pen stroker never sees one huge path.
    """
    return _build_chunks([pl for pl in layer.paths if len(pl) >= 2])


def build_jittered_layer_chunks(layer: Layer, sigma_mm: float) -> list[Chunk]:
    """Chunked variant of :func:`build_jittered_layer_path` (spec §6.4).

    The displacement array is generated exactly as in
    :func:`build_jittered_layer_path` (same seed, same polyline order), so the
    jittered geometry is identical — only the draw-call structure differs.
    """
    polylines = [pl for pl in layer.paths if len(pl) >= 2]
    n_points = sum(len(pl) for pl in polylines)
    if n_points == 0:
        return []
    rng = default_rng(zlib.crc32(layer.id.encode()))
    disp = rng.normal(0.0, sigma_mm, (n_points, 2))
    return _build_chunks(polylines, disp)


@dataclass
class _Entry:
    """Cached chunk lists plus the layer state they were built from.

    ``chunks`` is the un-jittered geometry. ``jittered_chunks`` is the
    baked-jitter variant (spec §6.4), built on demand and keyed by
    ``jitter_key = (enabled, intensity)``; both reset when jitter is toggled
    or its intensity changes.
    """

    chunks: list[Chunk]
    point_count: int
    jittered_chunks: list[Chunk] | None = None
    jitter_key: tuple[bool, float] | None = None


def build_layer_path(layer: Layer) -> QPainterPath:
    """Build one ``QPainterPath`` (mm coords) from a layer's polylines.

    Each polyline with at least two points becomes a disconnected subpath:
    ``moveTo`` the first point, then ``lineTo`` each subsequent point.
    Polylines with fewer than two points are skipped (nothing to stroke).
    """
    path = QPainterPath()
    for polyline in layer.paths:
        if len(polyline) < 2:
            continue
        x0, y0 = polyline[0]
        path.moveTo(x0, y0)
        for x, y in polyline[1:]:
            path.lineTo(x, y)
    return path


def build_jittered_layer_path(layer: Layer, sigma_mm: float) -> QPainterPath:
    """Build a baked-jitter ``QPainterPath`` (mm coords) for a layer (spec §6.4).

    Every drawn point is displaced by a deterministic per-layer normal sample:
    ``default_rng(zlib.crc32(layer.id))`` seeds the generator, and each of the
    ``n_points`` drawn points gets an independent ``(dx, dy)`` drawn from
    ``normal(0, sigma_mm)``. The displacement lives in **mm** (zoom-independent)
    and is fully determined by the layer id, so the result is identical across
    rebuilds and frames — no shimmer-on-pan. Same point-skipping rules as
    :func:`build_layer_path` (polylines shorter than two points are skipped).
    """
    polylines = [pl for pl in layer.paths if len(pl) >= 2]
    n_points = sum(len(pl) for pl in polylines)
    path = QPainterPath()
    if n_points == 0:
        return path
    rng = default_rng(zlib.crc32(layer.id.encode()))
    disp = rng.normal(0.0, sigma_mm, (n_points, 2))
    k = 0
    for polyline in polylines:
        x0, y0 = polyline[0]
        path.moveTo(x0 + disp[k, 0], y0 + disp[k, 1])
        k += 1
        for x, y in polyline[1:]:
            path.lineTo(x + disp[k, 0], y + disp[k, 1])
            k += 1
    return path


class LayerPathCache:
    """Cached chunked ``QPainterPath`` geometry per layer, in mm coordinates.

    ``get`` lazily builds an entry the first time a layer is drawn (or after it
    has been invalidated / its point count has changed) and returns the same
    cached chunk list on subsequent calls. ``invalidate`` / ``invalidate_all``
    drop entries so the next ``get`` rebuilds.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def get(
        self, layer: Layer, jitter: tuple[bool, float] | None = None
    ) -> list[Chunk]:
        """Return the cached chunks for ``layer``, building lazily if stale.

        ``jitter`` is ``(enabled, intensity)`` or ``None``. When enabled, the
        baked-jitter variant (spec §6.4) is returned: a deterministic mm-space
        displacement keyed by ``(True, intensity)``, built on demand and kept
        alongside the un-jittered chunks so toggling jitter never discards them.
        """
        entry = self._entries.get(layer.id)
        point_count = layer.total_point_count()
        if entry is None or entry.point_count != point_count:
            entry = _Entry(chunks=build_layer_chunks(layer), point_count=point_count)
            self._entries[layer.id] = entry
        if jitter is None or not jitter[0]:
            return entry.chunks
        key = (True, jitter[1])
        if entry.jitter_key != key or entry.jittered_chunks is None:
            sigma_mm = 0.15 * jitter[1]
            entry.jittered_chunks = build_jittered_layer_chunks(layer, sigma_mm)
            entry.jitter_key = key
        return entry.jittered_chunks

    def invalidate(self, layer_id: str) -> None:
        """Drop the cached entry for ``layer_id`` (no-op if absent)."""
        self._entries.pop(layer_id, None)

    def invalidate_all(self) -> None:
        """Drop all cached entries."""
        self._entries.clear()

    def invalidate_jitter(self) -> None:
        """Drop every baked-jitter variant, keeping un-jittered chunks (§6.4).

        Called when jitter is toggled or its intensity changes — cheap, since
        the (expensive) un-jittered geometry survives and only the on-demand
        jittered variant is rebuilt on the next ``get``.
        """
        for entry in self._entries.values():
            entry.jittered_chunks = None
            entry.jitter_key = None


def build_travel_path(project: "Project") -> QPainterPath:
    """Build one ``QPainterPath`` (mm coords) of pen-up travel moves.

    A travel move is the straight hop between the end of one drawn polyline and
    the start of the next, across all **visible** layers in project order — the
    same traversal the legacy per-frame loop performed. Each hop is its own
    disconnected subpath (``moveTo`` the previous end, ``lineTo`` the next
    start). Empty polylines contribute no endpoint and are skipped.
    """
    path = QPainterPath()
    last_end: tuple[float, float] | None = None
    for layer in project.layers:
        if not layer.visible:
            continue
        for polyline in layer.paths:
            if not polyline:
                continue
            if last_end is not None:
                path.moveTo(last_end[0], last_end[1])
                path.lineTo(polyline[0][0], polyline[0][1])
            last_end = polyline[-1]
    return path


class TravelPathCache:
    """Single cached travel-line ``QPainterPath`` in mm coordinates (§6.5).

    Lazily (re)built on first ``get`` after construction or invalidation. The
    travel path spans every visible layer, so any layer mutation, removal,
    addition, reorder, or visibility toggle invalidates it.
    """

    def __init__(self) -> None:
        self._path: QPainterPath | None = None

    def get(self, project: "Project") -> QPainterPath:
        """Return the cached travel path, building it lazily if invalidated."""
        if self._path is None:
            self._path = build_travel_path(project)
        return self._path

    def invalidate(self) -> None:
        """Drop the cached travel path so the next ``get`` rebuilds."""
        self._path = None


@dataclass
class _SceneEntry:
    """A baked scene pixmap plus the view state it was rendered for (§7.2).

    ``pixmap`` holds grid + reg marks + layer paths + travel on a transparent
    background, covering the viewport plus 0.5 viewport of slop per side and
    rendered at ``devicePixelRatio`` (tagged via ``setDevicePixelRatio``).
    ``origin_mm`` is the mm coordinate of the pixmap's top-left corner.
    """

    pixmap: QPixmap
    zoom: float
    origin_mm: tuple[float, float]
    scene_revision: int
    excluded_layer_id: str | None


def _current_excluded_layer_id(widget: "CanvasWidget") -> str | None:
    """The layer id to leave out of the scene pixmap (§7.5).

    While drag-to-move is active the active layer is drawn live on top of the
    blit, so it is excluded from the baked pixmap; otherwise nothing is.
    """
    if widget._drag_move_active:
        return widget._controller.active_layer_id
    return None


class ScenePixmapCache:
    """Single baked scene pixmap for the canvas (canvas-performance §7).

    Renders the expensive *static* content — grid, registration marks, layer
    paths and travel lines — into one transparent pixmap that can be blitted in
    a single ``drawPixmap`` per frame instead of re-stroking every path. The
    pixmap covers the current viewport plus 0.5 viewport of slop on each side
    (so small pans stay within the cached region) and is rendered at the
    widget's ``devicePixelRatio``.

    This class owns only the cache entry and its (re)build / validity logic;
    blitting into ``paintEvent`` is wired up in a later phase.
    """

    def __init__(self) -> None:
        self._entry: _SceneEntry | None = None
        #: Number of full pixmap (re)builds since construction. A pure test/
        #: bench hook (spec §11): lets a test assert that a small pan stayed
        #: within the slop region and reused the cached pixmap instead of
        #: rebuilding. Never read by production code.
        self.rebuild_count: int = 0

    @property
    def entry(self) -> _SceneEntry | None:
        """The current cached entry, or ``None`` if never built / invalidated."""
        return self._entry

    def invalidate(self) -> None:
        """Drop the cached pixmap so the next paint rebuilds (§7.4)."""
        self._entry = None

    def static_matches(self, widget: "CanvasWidget") -> bool:
        """Whether the baked *static content* still matches the widget (§7.3).

        Checks everything that determines what pixels were painted — the
        ``scene_revision``, the excluded drag layer, and the device pixel ratio
        — but **not** the view geometry (zoom / pan). When this is true and only
        the zoom has drifted, the pixmap can be reused as a soft scaled preview
        (§7.3 zoom-mismatch branch); when it is false the content itself changed
        and a fresh rebuild is required.
        """
        entry = self._entry
        if entry is None:
            return False
        if entry.scene_revision != widget.scene_revision:
            return False
        if entry.excluded_layer_id != _current_excluded_layer_id(widget):
            return False
        # A DPR change has no setter to bump ``scene_revision`` (spec §7.4 folds
        # it into a paint-time comparison), so guard it here too — a pixmap baked
        # at the wrong device ratio would blit blurry.
        if entry.pixmap.devicePixelRatio() != widget.devicePixelRatioF():
            return False
        return True

    def covers_viewport(self, widget: "CanvasWidget") -> bool:
        """Whether the live viewport lies fully inside the cached region (§7.2).

        Evaluated at the *cached* zoom, so it answers "did the pan stay within
        the slop?" independently of any zoom drift. Used both by :meth:`is_valid`
        (same-zoom blit) and to decide a pan-beyond-slop rebuild.
        """
        entry = self._entry
        if entry is None:
            return False
        dpr = entry.pixmap.devicePixelRatio()
        cov_w_mm = (entry.pixmap.width() / dpr) / entry.zoom
        cov_h_mm = (entry.pixmap.height() / dpr) / entry.zoom
        x0, y0 = entry.origin_mm
        vp_tl = widget.pixel_to_mm(QPointF(0.0, 0.0))
        vp_br = widget.pixel_to_mm(QPointF(float(widget.width()), float(widget.height())))
        return (
            vp_tl[0] >= x0
            and vp_tl[1] >= y0
            and vp_br[0] <= x0 + cov_w_mm
            and vp_br[1] <= y0 + cov_h_mm
        )

    def is_valid(self, widget: "CanvasWidget") -> bool:
        """Whether the cached pixmap can be blitted 1:1 for the current view (§7.2).

        Valid iff the static content still matches (:meth:`static_matches`), the
        zoom is unchanged, and the viewport lies inside the covered region
        (:meth:`covers_viewport`).
        """
        entry = self._entry
        if entry is None:
            return False
        if entry.zoom != widget._zoom:
            return False
        return self.static_matches(widget) and self.covers_viewport(widget)

    def rebuild(self, widget: "CanvasWidget") -> _SceneEntry | None:
        """Render the static scene into a fresh slop-padded pixmap (§7.1, §7.2).

        The pixmap is 2× the viewport per dimension (viewport + 0.5 slop each
        side), allocated at ``devicePixelRatio`` and filled transparent, then
        painted with grid → reg marks → layer paths → travel using the widget's
        own ``_draw_*`` helpers so the output is pixel-identical to drawing them
        live. Returns the stored entry (or ``None`` for a zero-size widget).
        """
        w = widget.width()
        h = widget.height()
        if w <= 0 or h <= 0:
            self._entry = None
            return None

        self.rebuild_count += 1
        zoom = widget._zoom
        dpr = widget.devicePixelRatioF()
        excluded_layer_id = _current_excluded_layer_id(widget)

        # Covered region: viewport (w×h) plus 0.5·viewport slop on each side →
        # 2w×2h logical px, top-left at (-0.5w, -0.5h) in widget pixels.
        origin_mm = widget.pixel_to_mm(QPointF(-0.5 * w, -0.5 * h))

        pixmap = QPixmap(round(2 * w * dpr), round(2 * h * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)

        # The widget's draw helpers read ``_pan_offset`` (directly and via
        # ``mm_to_pixel``). Swap in the pan that maps ``origin_mm`` to the
        # pixmap's top-left so a mm point p lands at (p − origin_mm)·zoom, then
        # restore — the swap is synchronous and never re-entrant.
        saved_pan = widget._pan_offset
        widget._pan_offset = QPointF(-origin_mm[0] * zoom, -origin_mm[1] * zoom)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            project = widget._controller.current_project
            canvas = project.canvas
            if widget._show_grid:
                widget._draw_grid(painter, canvas)
            if widget._show_reg_marks and project.registration_marks:
                widget._draw_registration_marks(painter, canvas, project.reg_mark_style)
            for layer in project.layers:
                if not layer.visible or layer.id == excluded_layer_id:
                    continue
                # The pixmap is 2w×2h logical px — pass its size so chunk
                # culling matches the pixmap's coverage, not the widget's.
                widget._draw_layer(painter, layer, device_size_px=(2.0 * w, 2.0 * h))
            if widget._show_travel:
                widget._draw_travel_lines(painter, project)
        finally:
            painter.end()
            widget._pan_offset = saved_pan

        entry = _SceneEntry(
            pixmap=pixmap,
            zoom=zoom,
            origin_mm=origin_mm,
            scene_revision=widget.scene_revision,
            excluded_layer_id=excluded_layer_id,
        )
        self._entry = entry
        return entry
