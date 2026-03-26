"""Scene: the main rendering orchestrator.

Usage
-----
from plottter.scene3d import Scene, Camera
from plottter.scene3d.shapes import Sphere, Cube

scene = Scene()
sphere = Sphere(radius=1.0)
cube = Cube(center=[2, 0, 0], size=1.5)
scene.add(sphere)
scene.add(cube)
scene.compile()  # builds BVH

camera = Camera.default(aspect=1.0)
polylines = scene.render(camera, canvas_w_mm=100, canvas_h_mm=100)
# polylines is list[list[tuple[float, float]]] — Plottter Polylines in mm
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .bvh import BVH
from .path3d import Path3D
from .ray import Ray, EPSILON
from .vector3 import vec3, normalize

if TYPE_CHECKING:
    from .shapes.base import Shape
    from .camera import Camera


def _path_outside_frustum(
    pts: np.ndarray,
    vp_matrix: np.ndarray,
    x_min_ndc: float = -1.0,
    x_max_ndc: float = 1.0,
    y_min_ndc: float = -1.0,
    y_max_ndc: float = 1.0,
) -> bool:
    """Return True if the path is guaranteed to be entirely outside the view frustum.

    Uses clip-space coordinate tests.  If all points are on the wrong side of any
    single clip plane the path is definitely off-screen and can be skipped before
    chopping, saving all the ray-casting work for that path.

    This is a *conservative* test — False means "might be visible" (never skips
    a path that has any on-screen portion).

    Parameters
    ----------
    pts:       (N, 3) float64 array of path points.
    vp_matrix: (4, 4) view-projection matrix (row-vector convention: p @ M).
    x_min_ndc: Left NDC boundary (default -1); adjusted when offset_mm is non-zero.
    x_max_ndc: Right NDC boundary (default 1).
    y_min_ndc: Bottom NDC boundary (default -1).
    y_max_ndc: Top NDC boundary (default 1).

    Returns
    -------
    True  → path is definitely off-screen, safe to skip.
    False → path may be visible (default — do not cull).
    """
    n = len(pts)
    if n == 0:
        return True

    # Compute homogeneous clip coordinates: shape (N, 4)
    homogeneous = np.empty((n, 4), dtype=np.float64)
    homogeneous[:, :3] = pts
    homogeneous[:, 3] = 1.0
    clip = homogeneous @ vp_matrix  # (N, 4)

    w = clip[:, 3]
    x = clip[:, 0]
    y = clip[:, 1]

    # All points behind the camera (w ≤ 0)
    if np.all(w <= 0.0):
        return True
    # All points past the right clip plane: x > x_max_ndc * w
    if np.all(x > x_max_ndc * w):
        return True
    # All points past the left clip plane: x < x_min_ndc * w
    if np.all(x < x_min_ndc * w):
        return True
    # All points above the top clip plane: y > y_max_ndc * w
    if np.all(y > y_max_ndc * w):
        return True
    # All points below the bottom clip plane: y < y_min_ndc * w
    if np.all(y < y_min_ndc * w):
        return True

    return False


class Scene:
    """A 3D scene containing shapes, with rendering via hidden line removal.

    Parameters
    ----------
    hlr_enabled:  If True, perform hidden line removal (default True).
    chop_step:    Segment length for HLR ray casting (smaller = more accurate).
    simplify_tol: RDP simplification tolerance after rendering (in world units).
    """

    def __init__(
        self,
        hlr_enabled: bool = True,
        chop_step: float = 0.05,
        simplify_tol: float = 0.001,
    ) -> None:
        self.shapes: list["Shape"] = []
        self.hlr_enabled = hlr_enabled
        self.chop_step = chop_step
        self.simplify_tol = simplify_tol
        self._bvh: BVH | None = None
        self._compiled = False

    def add(self, shape: "Shape") -> "Scene":
        self.shapes.append(shape)
        self._compiled = False
        return self

    def compile(self) -> "Scene":
        """Build the BVH acceleration structure."""
        self._bvh = BVH(self.shapes)
        self._bvh.build()
        self._compiled = True
        return self

    def render(
        self,
        camera: "Camera",
        canvas_w_mm: float,
        canvas_h_mm: float,
        render_shapes: list["Shape"] | None = None,
        progress_callback=None,
        cancelled_callback=None,
        offset_mm: tuple[float, float] = (0.0, 0.0),
        light_dir: "tuple[float, float, float] | None" = None,
        extra_render_paths: "list[Path3D] | None" = None,
        hlr_quality: str = "Fine",
    ):
        """Render shapes to 2D polylines in mm coordinates.

        Parameters
        ----------
        camera:             Camera defining the view and projection.
        canvas_w_mm:        Canvas width in millimeters.
        canvas_h_mm:        Canvas height in millimeters.
        render_shapes:      If given, only render these shapes' paths. All shapes
                            are still used for occlusion testing.
        progress_callback:  Optional callable(progress: float) for reporting 0..1 progress.
        cancelled_callback: Optional callable() → bool; return True to abort rendering.
                            Checked every 100 segments during HLR.
        offset_mm:          (x_off, y_off) canvas offset in mm.  When non-zero, the
                            projection is adjusted so that clipping happens in the
                            correct coordinate space: geometry that falls outside the
                            standard NDC range but is visible on the offset canvas is
                            correctly included, and geometry off the offset canvas is
                            correctly excluded.
        light_dir:          Optional (lx, ly, lz) directional light vector.  When set,
                            shadow visibility is computed by casting a ray from each
                            visible segment midpoint toward the light source.  A segment
                            is "in shadow" if any scene geometry blocks that ray.
        extra_render_paths: Additional Path3D paths to render with camera HLR but
                            *without* shadow ray casting.  Intended for ground-plane
                            shadow geometry (task 29.4): these paths should be occluded
                            by scene objects but should not themselves cast shadows.

        Returns
        -------
        When ``light_dir`` is ``None`` (default):
            ``list[list[tuple[float, float]]]`` — all visible polylines in mm.
        When ``light_dir`` is provided:
            ``tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]``
            — a pair ``(lit_polylines, shadow_polylines)`` where ``lit_polylines``
            are visible segments that are NOT in shadow and ``shadow_polylines`` are
            visible segments that ARE in shadow.  Both coordinate sets are in mm
            relative to the canvas origin with the offset already applied.
        """
        if not self._compiled:
            self.compile()

        shapes_to_render = render_shapes if render_shapes is not None else self.shapes

        # Build the view-projection matrix
        vp_matrix = camera.view_proj_matrix()

        # Collect all 3D paths from the shapes to render
        all_paths: list[Path3D] = []
        for shape in shapes_to_render:
            all_paths.extend(shape.paths())

        if not all_paths and not extra_render_paths:
            return [] if light_dir is None else ([], [])

        if not self.hlr_enabled:
            # No HLR: just project all paths directly (no shadow computation)
            projected = self._project_paths(all_paths, vp_matrix, canvas_w_mm, canvas_h_mm, offset_mm)
            extra_projected: list[list[tuple[float, float]]] = []
            if extra_render_paths:
                extra_projected = self._project_paths(extra_render_paths, vp_matrix, canvas_w_mm, canvas_h_mm, offset_mm)
            all_projected = projected + extra_projected
            return all_projected if light_dir is None else (all_projected, [])

        # HLR: chop paths into segments and ray-cast each for visibility
        return self._render_with_hlr(
            all_paths, camera, vp_matrix, canvas_w_mm, canvas_h_mm,
            progress_callback, cancelled_callback, offset_mm, light_dir,
            extra_render_paths=extra_render_paths,
            hlr_quality=hlr_quality,
        )

    def _project_paths(
        self,
        paths: list[Path3D],
        vp_matrix: np.ndarray,
        canvas_w_mm: float,
        canvas_h_mm: float,
        offset_mm: tuple[float, float] = (0.0, 0.0),
    ) -> list[list[tuple[float, float]]]:
        """Project 3D paths to 2D without HLR."""
        result: list[list[tuple[float, float]]] = []
        for path in paths:
            for polyline in path.project_segments(vp_matrix, canvas_w_mm, canvas_h_mm, offset_mm):
                if len(polyline) >= 2:
                    result.append(polyline)
        return result

    def _render_with_hlr(
        self,
        paths: list[Path3D],
        camera: "Camera",
        vp_matrix: np.ndarray,
        canvas_w_mm: float,
        canvas_h_mm: float,
        progress_callback=None,
        cancelled_callback=None,
        offset_mm: tuple[float, float] = (0.0, 0.0),
        light_dir: "tuple[float, float, float] | None" = None,
        extra_render_paths: "list[Path3D] | None" = None,
        hlr_quality: str = "Fine",
    ):
        """Render with hidden line removal via ray casting.

        Optimizations applied:
        - **Frustum culling**: skip paths where all points are outside the same
          clip plane — saves all chopping and ray-cast work for off-screen geometry.
        - **Raw segment collection**: segment pairs are collected as plain NumPy
          arrays rather than creating one Path3D object per sub-segment, reducing
          Python object-allocation overhead significantly for dense meshes.
        - **Batch precomputation**: midpoints, ray directions, distances, and
          inverse directions are computed in vectorized NumPy calls before the
          per-ray BVH loop starts.
        - **Early-exit occlusion**: BVH.intersect_any() now dispatches to
          Shape.intersect_any() which for Mesh delegates to
          TriangleBVH.intersect_any() (iterative, early-exit on first hit).
        - **Cancellation support**: cancelled_callback is checked every 100
          segments so long renders can be aborted from the UI.
        - **Shadow ray casting**: when ``light_dir`` is provided, each visible
          segment is additionally tested against the BVH from the midpoint toward
          the light source.  Visible segments are split into lit and shadowed lists
          which are returned as separate polyline collections.

        Returns
        -------
        When ``light_dir`` is ``None``:
            ``list[list[tuple[float, float]]]`` — all visible polylines.
        When ``light_dir`` is provided:
            ``tuple[list[...], list[...]]`` — ``(lit_polylines, shadow_polylines)``.
        """
        bvh = self._bvh
        eye = np.asarray(camera.eye, dtype=np.float64)
        chop_step = self.chop_step
        x_off, y_off = offset_mm

        # Normalise the light direction for shadow ray casting (if provided).
        light_dir_norm: np.ndarray | None = None
        if light_dir is not None:
            ld_arr = np.asarray(light_dir, dtype=np.float64)
            ld_len = float(np.linalg.norm(ld_arr))
            if ld_len > EPSILON:
                light_dir_norm = ld_arr / ld_len

        # Compute offset-adjusted NDC bounds for frustum culling.
        # These match the bounds used in Path3D.project_segments().
        if canvas_w_mm > 0.0:
            x_min_ndc = -1.0 - 2.0 * x_off / canvas_w_mm
            x_max_ndc =  1.0 - 2.0 * x_off / canvas_w_mm
        else:
            x_min_ndc, x_max_ndc = -1.0, 1.0
        if canvas_h_mm > 0.0:
            y_min_ndc = -1.0 + 2.0 * y_off / canvas_h_mm
            y_max_ndc =  1.0 + 2.0 * y_off / canvas_h_mm
        else:
            y_min_ndc, y_max_ndc = -1.0, 1.0

        # ── 1. Frustum-cull paths and collect raw segment pairs ──────────────
        # We accumulate segment endpoints as plain arrays to avoid creating one
        # Path3D object per sub-segment (and the associated list allocation).
        # Segments from the same original path are adjacent in the list so that
        # _reassemble_polylines can still chain them correctly.
        #
        # seg_normals stores the face_normal of the originating Path3D for each
        # segment (or None if no face normal is set on that path).  This is used
        # in the shadow classification step below.
        seg_starts: list[np.ndarray] = []
        seg_ends: list[np.ndarray] = []
        seg_normals: list[np.ndarray | None] = []
        # For coarse-to-fine: track (path_id, start_idx, end_idx) per source path.
        # end_idx is exclusive (like a Python range).
        seg_path_ranges: list[tuple[int, int, int]] = []
        _path_id = 0

        for path in paths:
            pts = np.asarray(path.points, dtype=np.float64)
            if len(pts) < 2:
                continue

            # Frustum cull: skip path if all points are on the wrong side of
            # any single clip plane — saves chopping + ray-casting for those paths.
            if _path_outside_frustum(pts, vp_matrix, x_min_ndc, x_max_ndc, y_min_ndc, y_max_ndc):
                continue

            # Chop this path into sub-segments as raw numpy pairs.
            path_normal = path.face_normal  # may be None
            _path_start = len(seg_starts)
            for i in range(len(pts) - 1):
                a = pts[i]
                b = pts[i + 1]
                d = float(np.linalg.norm(b - a))
                if d < 1e-10:
                    continue
                # Edges shorter than chop_step get exactly one segment (the whole
                # edge), which is the same result as Path3D.chop() but without
                # the Python object overhead.
                n = max(1, int(d / chop_step))
                step_vec = (b - a) / n
                for j in range(n):
                    seg_starts.append(a + step_vec * j)
                    seg_ends.append(a + step_vec * (j + 1))
                    seg_normals.append(path_normal)
            _path_end = len(seg_starts)
            if _path_end > _path_start:
                seg_path_ranges.append((_path_id, _path_start, _path_end))
            _path_id += 1

        if not seg_starts and not extra_render_paths:
            return [] if light_dir_norm is None else ([], [])

        lit_segments: list[tuple[np.ndarray, np.ndarray]] = []
        shadow_segments: list[tuple[np.ndarray, np.ndarray]] = []

        if seg_starts:
            # ── 2. Batch precompute ray data ─────────────────────────────────────
            n_segs = len(seg_starts)
            seg_a = np.stack(seg_starts)          # (N, 3)
            seg_b = np.stack(seg_ends)            # (N, 3)
            mids = (seg_a + seg_b) * 0.5          # (N, 3)

            # Build per-segment normal array (None entries → zero vector)
            seg_normals_arr = np.zeros((n_segs, 3), dtype=np.float64)
            for _k, _sn in enumerate(seg_normals):
                if _sn is not None:
                    seg_normals_arr[_k] = _sn

            # Offset midpoints along face normal to lift the test point slightly
            # off the surface, preventing self-occlusion at grazing angles.
            # Only applied where face normals are valid (non-zero magnitude).
            norm_mags = np.linalg.norm(seg_normals_arr, axis=1)  # (N,)
            has_normal = norm_mags > 0.1
            offset_mids = mids.copy()
            if np.any(has_normal):
                offset_mids[has_normal] = (
                    mids[has_normal]
                    + seg_normals_arr[has_normal] * (chop_step * 0.1)
                )

            dirs = offset_mids - eye              # (N, 3) raw direction vectors
            dists = np.linalg.norm(dirs, axis=1)  # (N,)

            # Mark degenerate rays (midpoint coincides with eye) as invalid
            valid = dists >= EPSILON

            # Normalized direction for valid rays; zeros for invalid (won't be used)
            dirs_norm = np.zeros_like(dirs)
            dirs_norm[valid] = dirs[valid] / dists[valid, np.newaxis]

            # t_max per segment: segment midpoint distance minus half a chop step
            # (so that the segment itself doesn't self-occlude)
            t_maxs = np.maximum(EPSILON, dists - chop_step * 0.5)

            # Precompute shadow ray origins (offset from midpoint along light dir to
            # avoid self-intersection) if shadow computation is requested.
            shadow_origins: np.ndarray | None = None
            if light_dir_norm is not None:
                # Offset origin by half a chop_step along the light direction.
                # This ensures the shadow ray doesn't hit the segment's own surface.
                shadow_origins = mids + chop_step * 0.5 * light_dir_norm  # (N, 3)

            # ── 3. Per-ray visibility + shadow test ───────────────────────────────
            # Determine coarse-to-fine stride from hlr_quality.
            # "Fine"   → stride 1 (test every segment, identical to legacy behavior).
            # "Normal" → stride 4 (coarse pre-pass every 4th segment per path).
            # "Fast"   → stride 8 (coarse pre-pass every 8th segment per path).
            # Coarse-to-fine is only applied when n_segs > 200 and bvh is present.
            if hlr_quality == "Fast":
                _coarse_stride = 8
            elif hlr_quality == "Normal":
                _coarse_stride = 4
            else:
                _coarse_stride = 1  # "Fine": no optimization

            _use_coarse = _coarse_stride > 1 and n_segs > 200 and bvh is not None

            if not _use_coarse:
                # ── Original per-segment loop (Fine quality or small scene) ──────
                for i in range(n_segs):
                    if i % 100 == 0:
                        if cancelled_callback is not None and cancelled_callback():
                            return [] if light_dir_norm is None else ([], [])
                        if progress_callback is not None:
                            progress_callback(i / n_segs)

                    if not valid[i]:
                        continue

                    is_visible = True
                    if bvh is not None:
                        ray = Ray(origin=eye, direction=dirs_norm[i])
                        is_visible = not bvh.intersect_any(ray, t_max=float(t_maxs[i]))

                    if is_visible:
                        if light_dir_norm is not None:
                            in_shadow = False
                            fn = seg_normals[i]
                            if fn is not None:
                                if float(np.dot(fn, light_dir_norm)) < 0:
                                    in_shadow = True
                            if not in_shadow and shadow_origins is not None and bvh is not None:
                                shadow_ray = Ray(
                                    origin=shadow_origins[i],
                                    direction=light_dir_norm,
                                )
                                in_shadow = bvh.intersect_any(shadow_ray, t_max=1e9)
                            if in_shadow:
                                shadow_segments.append((seg_a[i], seg_b[i]))
                            else:
                                lit_segments.append((seg_a[i], seg_b[i]))
                        else:
                            lit_segments.append((seg_a[i], seg_b[i]))
            else:
                # ── Coarse-to-fine per-path loop ─────────────────────────────────
                # Phase 1: Coarse pass — test every Nth segment per source path.
                # Phase 2: Per block between coarse samples:
                #   - If both endpoints agree: intermediate segments inherit.
                #   - If endpoints differ (transition): fine-test intermediates.
                # Phase 3: Record all visible segments (with shadow logic).
                #
                # Visibility is encoded in vis_result:
                #   1 = visible, 0 = hidden/invalid, -1 = unresolved (shouldn't remain)
                vis_result = np.zeros(n_segs, dtype=np.int8)  # 0 = hidden/invalid
                vis_result[~valid] = 0  # invalid rays are hidden

                _ray_count = 0
                _coarse_cancelled = False

                def _cast_vis(idx: int) -> bool:
                    """Cast a visibility ray for segment idx; returns True if visible.

                    Sets _coarse_cancelled when the cancelled_callback fires so the
                    outer loop can propagate early exit identically to the Fine path.
                    """
                    nonlocal _ray_count, _coarse_cancelled
                    _ray_count += 1
                    if _ray_count % 100 == 0:
                        if cancelled_callback is not None and cancelled_callback():
                            _coarse_cancelled = True
                            return False
                        if progress_callback is not None:
                            progress_callback(_ray_count / n_segs)
                    r = Ray(origin=eye, direction=dirs_norm[idx])
                    return not bvh.intersect_any(r, t_max=float(t_maxs[idx]))

                for _pid, _ps, _pe in seg_path_ranges:
                    if _coarse_cancelled:
                        return [] if light_dir_norm is None else ([], [])
                    _n_path = _pe - _ps
                    if _n_path == 0:
                        continue

                    # Coarse sample positions within the path (0-based offsets).
                    # Always include the last segment so the final block is bounded.
                    _coarse_offsets: list[int] = list(range(0, _n_path, _coarse_stride))
                    _last = _n_path - 1
                    if _coarse_offsets[-1] != _last:
                        _coarse_offsets.append(_last)

                    # Phase 1: coarse ray casts
                    _coarse_vis: dict[int, int] = {}
                    for _k in _coarse_offsets:
                        if _coarse_cancelled:
                            return [] if light_dir_norm is None else ([], [])
                        _i = _ps + _k
                        if not valid[_i]:
                            _coarse_vis[_k] = 0
                        else:
                            _coarse_vis[_k] = 1 if _cast_vis(_i) else 0
                        vis_result[_i] = _coarse_vis[_k]

                    # Phase 2: inheritance or fine test for intermediate segments
                    for _bi in range(len(_coarse_offsets) - 1):
                        _k1 = _coarse_offsets[_bi]
                        _k2 = _coarse_offsets[_bi + 1]
                        _same = _coarse_vis[_k1] == _coarse_vis[_k2]
                        _inherit_val = _coarse_vis[_k1]

                        for _k in range(_k1 + 1, _k2):
                            if _coarse_cancelled:
                                return [] if light_dir_norm is None else ([], [])
                            _i = _ps + _k
                            if not valid[_i]:
                                vis_result[_i] = 0
                            elif _same:
                                vis_result[_i] = _inherit_val
                            else:
                                vis_result[_i] = 1 if _cast_vis(_i) else 0

                # Phase 3: record all visible segments with shadow logic
                for i in range(n_segs):
                    if vis_result[i] != 1:
                        continue
                    if light_dir_norm is not None:
                        in_shadow = False
                        fn = seg_normals[i]
                        if fn is not None:
                            if float(np.dot(fn, light_dir_norm)) < 0:
                                in_shadow = True
                        if not in_shadow and shadow_origins is not None and bvh is not None:
                            shadow_ray = Ray(
                                origin=shadow_origins[i],
                                direction=light_dir_norm,
                            )
                            in_shadow = bvh.intersect_any(shadow_ray, t_max=1e9)
                        if in_shadow:
                            shadow_segments.append((seg_a[i], seg_b[i]))
                        else:
                            lit_segments.append((seg_a[i], seg_b[i]))
                    else:
                        lit_segments.append((seg_a[i], seg_b[i]))

        # ── 4. Process extra_render_paths (ground shadow geometry) ───────────
        # These paths receive camera HLR (occluded by scene objects) but do NOT
        # receive shadow ray tests — they are shadow geometry and should not cast
        # shadows themselves.  Visible extra segments are added to lit_segments.
        if extra_render_paths:
            extra_starts: list[np.ndarray] = []
            extra_ends: list[np.ndarray] = []

            for path in extra_render_paths:
                pts = np.asarray(path.points, dtype=np.float64)
                if len(pts) < 2:
                    continue
                if _path_outside_frustum(pts, vp_matrix, x_min_ndc, x_max_ndc, y_min_ndc, y_max_ndc):
                    continue
                for i in range(len(pts) - 1):
                    a = pts[i]
                    b = pts[i + 1]
                    d = float(np.linalg.norm(b - a))
                    if d < 1e-10:
                        continue
                    n = max(1, int(d / chop_step))
                    step_vec = (b - a) / n
                    for j in range(n):
                        extra_starts.append(a + step_vec * j)
                        extra_ends.append(a + step_vec * (j + 1))

            if extra_starts:
                ex_n = len(extra_starts)
                ex_a = np.stack(extra_starts)
                ex_b = np.stack(extra_ends)
                ex_mids = (ex_a + ex_b) * 0.5
                ex_dirs = ex_mids - eye
                ex_dists = np.linalg.norm(ex_dirs, axis=1)
                ex_valid = ex_dists >= EPSILON
                ex_dirs_norm = np.zeros_like(ex_dirs)
                ex_dirs_norm[ex_valid] = ex_dirs[ex_valid] / ex_dists[ex_valid, np.newaxis]
                ex_t_maxs = np.maximum(EPSILON, ex_dists - chop_step * 0.5)

                for i in range(ex_n):
                    if i % 100 == 0 and cancelled_callback is not None and cancelled_callback():
                        return [] if light_dir_norm is None else ([], [])
                    if not ex_valid[i]:
                        continue
                    is_visible = True
                    if bvh is not None:
                        ray = Ray(origin=eye, direction=ex_dirs_norm[i])
                        is_visible = not bvh.intersect_any(ray, t_max=float(ex_t_maxs[i]))
                    if is_visible:
                        lit_segments.append((ex_a[i], ex_b[i]))

        if light_dir_norm is not None:
            # Return separate lit and shadow polyline lists
            if not lit_segments and not shadow_segments:
                return ([], [])
            lit_polys = self._reassemble_polylines(
                lit_segments, vp_matrix, canvas_w_mm, canvas_h_mm, offset_mm
            )
            shadow_polys = self._reassemble_polylines(
                shadow_segments, vp_matrix, canvas_w_mm, canvas_h_mm, offset_mm
            )
            return (lit_polys, shadow_polys)

        # No shadow computation — return all visible segments as a single list
        if not lit_segments:
            return []
        return self._reassemble_polylines(
            lit_segments, vp_matrix, canvas_w_mm, canvas_h_mm, offset_mm
        )

    def _compute_visible_faces(
        self,
        light_dir: "tuple[float, float, float]",
        camera: "Camera",
        canvas_w_mm: float,
        canvas_h_mm: float,
        progress_callback=None,
    ) -> "list[tuple[list[tuple[float, float]], float]]":
        """Compute visible, front-facing surface triangles with shading.

        For each shape that implements ``surface_triangles()``:
        - skip back-facing triangles (face normal points away from camera),
        - skip occluded triangles (BVH ray cast from eye to centroid is blocked),
        - compute diffuse brightness from the light direction,
        - project the 3 vertices to 2D canvas coordinates (mm).

        Parameters
        ----------
        light_dir:         (lx, ly, lz) directional light vector (need not be
                           normalised — it is normalised internally).
        camera:            Camera defining the viewpoint and projection.
        canvas_w_mm:       Canvas width in millimetres.
        canvas_h_mm:       Canvas height in millimetres.
        progress_callback: Optional ``callable(progress: float)`` for 0..1
                           progress reporting during face processing.

        Returns
        -------
        List of ``(projected_vertices, brightness)`` tuples for every visible
        face.  ``projected_vertices`` is a list of three ``(x_mm, y_mm)``
        tuples (the projected triangle corners).  ``brightness`` is in [0, 1].
        """
        if not self._compiled:
            self.compile()

        # Normalise light direction.
        ld_arr = np.asarray(light_dir, dtype=np.float64)
        ld_len = float(np.linalg.norm(ld_arr))
        if ld_len < EPSILON:
            return []
        light_dir_norm = ld_arr / ld_len

        eye = np.asarray(camera.eye, dtype=np.float64)
        vp_matrix = camera.view_proj_matrix()
        bvh = self._bvh

        # Collect all triangles from shapes that support surface_triangles().
        all_triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for shape in self.shapes:
            tris = shape.surface_triangles()
            if not tris:
                continue
            for tri in tris:
                v0 = np.asarray(tri[0], dtype=np.float64)
                v1 = np.asarray(tri[1], dtype=np.float64)
                v2 = np.asarray(tri[2], dtype=np.float64)
                all_triangles.append((v0, v1, v2))

        n_tris = len(all_triangles)
        if n_tris == 0:
            return []

        result: list[tuple[list[tuple[float, float]], float]] = []

        for i, (v0, v1, v2) in enumerate(all_triangles):
            if i % 100 == 0 and progress_callback is not None:
                progress_callback(i / n_tris)

            # 1. Compute face normal via cross product.
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal_len = float(np.linalg.norm(normal))
            if normal_len < EPSILON:
                continue  # degenerate triangle
            normal = normal / normal_len

            # 2. View direction: from centroid toward the camera eye.
            centroid = (v0 + v1 + v2) / 3.0
            to_eye = eye - centroid
            to_eye_len = float(np.linalg.norm(to_eye))
            if to_eye_len < EPSILON:
                continue  # eye is at the centroid
            view_dir = to_eye / to_eye_len

            # 3. Back-face cull: skip if face normal points away from camera.
            if float(np.dot(normal, view_dir)) <= 0.0:
                continue

            # 4. Occlusion test: ray from eye toward centroid.
            if bvh is not None:
                ray_dir = -view_dir  # from eye toward centroid
                ray = Ray(origin=eye, direction=ray_dir)
                t_max = float(to_eye_len) - self.chop_step * 0.5
                if t_max > EPSILON and bvh.intersect_any(ray, t_max=t_max):
                    continue  # occluded by other geometry

            # 5. Diffuse brightness.
            brightness = float(max(0.0, np.dot(normal, light_dir_norm)))

            # 6. Project 3 vertices to 2D mm.
            projected: list[tuple[float, float]] = []
            skip = False
            for vert in (v0, v1, v2):
                h = np.array([vert[0], vert[1], vert[2], 1.0], dtype=np.float64)
                clip = h @ vp_matrix
                w = clip[3]
                if abs(w) < EPSILON:
                    skip = True
                    break
                ndc_x = clip[0] / w
                ndc_y = clip[1] / w
                x_mm = (ndc_x + 1.0) * 0.5 * canvas_w_mm
                y_mm = (1.0 - (ndc_y + 1.0) * 0.5) * canvas_h_mm
                projected.append((float(x_mm), float(y_mm)))

            if skip or len(projected) != 3:
                continue

            result.append((projected, brightness))

        if progress_callback is not None:
            progress_callback(1.0)

        return result

    def _reassemble_polylines(
        self,
        segments: list[tuple[np.ndarray, np.ndarray]],
        vp_matrix: np.ndarray,
        canvas_w_mm: float,
        canvas_h_mm: float,
        offset_mm: tuple[float, float] = (0.0, 0.0),
    ) -> list[list[tuple[float, float]]]:
        """Join consecutive visible segments back into polylines and project to 2D."""
        if not segments:
            return []

        # Build chains of consecutive segments
        chains: list[list[np.ndarray]] = []
        current_chain: list[np.ndarray] = list(segments[0])

        for i in range(1, len(segments)):
            prev_b = segments[i - 1][1]
            curr_a = segments[i][0]
            # If the start of this segment matches the end of the previous, extend the chain
            if np.linalg.norm(curr_a - prev_b) < self.chop_step * 0.01:
                current_chain.append(segments[i][1])
            else:
                chains.append(current_chain)
                current_chain = list(segments[i])
        chains.append(current_chain)

        # Project each chain to 2D
        result: list[list[tuple[float, float]]] = []
        for chain in chains:
            if len(chain) < 2:
                continue
            path = Path3D(chain)
            if self.simplify_tol > 0:
                path = path.simplify(self.simplify_tol)
            polyline = path.to_polyline(vp_matrix, canvas_w_mm, canvas_h_mm, offset_mm)
            if polyline and len(polyline) >= 2:
                result.append(polyline)
        return result
