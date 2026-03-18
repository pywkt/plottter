"""3D path class with HLR support methods.

A Path3D is a sequence of 3D points. The key operations are:
- chop(): break into short sub-segments for ray-casting
- filter_visible(): remove segments occluded by the scene
- simplify(): RDP simplification to reduce point count
- project(): convert 3D → 2D using a projection matrix
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .vector3 import Vec3

if TYPE_CHECKING:
    from .ray import Ray, Hit
    from .shapes.base import Shape


# Six frustum clip planes in homogeneous clip space.
# Each row is (a, b, c, d) meaning  a*x + b*y + c*z + d*w >= 0.
#   left:   x >= -w  →  x + w >= 0
#   right:  x <=  w  → -x + w >= 0
#   bottom: y >= -w  →  y + w >= 0
#   top:    y <=  w  → -y + w >= 0
#   near:   z >= -w  →  z + w >= 0  (OpenGL NDC: z in [-1, 1])
#   far:    z <=  w  → -z + w >= 0
_FRUSTUM_PLANES: NDArray[np.float64] = np.array([
    [ 1,  0,  0,  1],
    [-1,  0,  0,  1],
    [ 0,  1,  0,  1],
    [ 0, -1,  0,  1],
    [ 0,  0,  1,  1],
    [ 0,  0, -1,  1],
], dtype=np.float64)


def _clip_segment_homogeneous(
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    planes: NDArray[np.float64] = _FRUSTUM_PLANES,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Clip a segment in homogeneous clip space to the view frustum.

    Implements the Cyrus-Beck / Liang-Barsky algorithm against all 6 frustum
    planes.  Works correctly when either or both endpoints are behind the
    camera (w <= 0), because clipping is done before the perspective divide.

    Parameters
    ----------
    p0, p1: Homogeneous clip-space points (x, y, z, w).
    planes:  (6, 4) array of clip planes; defaults to the standard NDC frustum.

    Returns
    -------
    (clipped_p0, clipped_p1) if the segment intersects the frustum, or
    None if the segment is entirely outside.
    """
    d = p1 - p0
    t_enter = 0.0
    t_exit = 1.0

    for plane in planes:
        f0 = float(np.dot(plane, p0))
        fd = float(np.dot(plane, d))

        if abs(fd) < 1e-12:
            # Segment is parallel to this plane.
            if f0 < -1e-10:
                return None  # entirely on the outside
        elif fd > 0:
            # Segment is entering the positive half-space.
            t = -f0 / fd
            if t > t_exit:
                return None
            if t > t_enter:
                t_enter = t
        else:
            # Segment is exiting the positive half-space.
            t = -f0 / fd
            if t < t_enter:
                return None
            if t < t_exit:
                t_exit = t

    if t_enter > t_exit + 1e-12:
        return None

    return p0 + t_enter * d, p0 + t_exit * d


def _clip_to_mm(
    p: NDArray[np.float64],
    canvas_w_mm: float,
    canvas_h_mm: float,
    x_off: float = 0.0,
    y_off: float = 0.0,
) -> tuple[float, float]:
    """Perspective-divide a clip-space point and convert to mm canvas coords.

    Parameters
    ----------
    p:            Homogeneous clip-space point (x, y, z, w).
    canvas_w_mm:  Canvas width in millimeters.
    canvas_h_mm:  Canvas height in millimeters.
    x_off:        X offset in mm applied after NDC-to-mm mapping.
    y_off:        Y offset in mm applied after NDC-to-mm mapping.
    """
    w = p[3]
    ndc_x = p[0] / w
    ndc_y = p[1] / w
    x_mm = (ndc_x + 1.0) * 0.5 * canvas_w_mm + x_off
    # NDC +1 = top of screen → 0 mm; NDC -1 = bottom → canvas_h_mm
    y_mm = (1.0 - (ndc_y + 1.0) * 0.5) * canvas_h_mm + y_off
    return float(x_mm), float(y_mm)


def _is_inside_frustum(
    p: NDArray[np.float64],
    x_min_ndc: float = -1.0,
    x_max_ndc: float = 1.0,
    y_min_ndc: float = -1.0,
    y_max_ndc: float = 1.0,
) -> bool:
    """Return True if a homogeneous clip-space point is inside the (adjusted) frustum.

    Parameters
    ----------
    p:           Homogeneous clip-space point (x, y, z, w).
    x_min_ndc:   Left NDC boundary (default -1).
    x_max_ndc:   Right NDC boundary (default 1).
    y_min_ndc:   Bottom NDC boundary (default -1).
    y_max_ndc:   Top NDC boundary (default 1).
    """
    w = p[3]
    if w <= 1e-6:
        return False
    # Allow a tiny epsilon so boundary points are treated as inside.
    eps = w * 1e-6
    return (
        p[0] >= (x_min_ndc * w - eps) and p[0] <= (x_max_ndc * w + eps) and
        p[1] >= (y_min_ndc * w - eps) and p[1] <= (y_max_ndc * w + eps) and
        p[2] >= -(w + eps) and p[2] <= (w + eps)
    )


class Path3D:
    """A 3D polyline: an ordered sequence of Vec3 points."""

    def __init__(
        self,
        points: list[Vec3] | NDArray[np.float64],
        face_normal: NDArray[np.float64] | None = None,
    ) -> None:
        if isinstance(points, np.ndarray):
            self.points: list[Vec3] = list(points)
        else:
            self.points = list(points)
        # Surface normal of the face that generated this edge/path.
        # Used by the HLR pipeline for face-normal-based shadow classification.
        # None means shadow classification falls back to ray casting only.
        self.face_normal: NDArray[np.float64] | None = face_normal

    def __len__(self) -> int:
        return len(self.points)

    def chop(self, step: float) -> list["Path3D"]:
        """Break into short sub-segments, each of length <= step.

        Returns a list of 2-point Path3D objects (one per segment).
        If the path has <2 points, returns empty list.
        """
        if len(self.points) < 2:
            return []
        segments: list[Path3D] = []
        pts = np.array(self.points, dtype=np.float64)
        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            d = np.linalg.norm(b - a)
            if d < 1e-10:
                continue
            n = max(1, int(d / step))
            for j in range(n):
                t0 = j / n
                t1 = (j + 1) / n
                p0 = a + (b - a) * t0
                p1 = a + (b - a) * t1
                seg = Path3D([p0, p1])
                seg.face_normal = self.face_normal
                segments.append(seg)
        return segments

    def filter_outside(self, shape: "Shape") -> list["Path3D"]:
        """Keep only the 2-point segments where the midpoint is outside shape.

        This is used for CSG difference — return parts of the path not inside shape.
        """
        result = []
        pts = np.array(self.points, dtype=np.float64)
        if len(pts) < 2:
            return []
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            mid = (a + b) * 0.5
            # Test if mid is outside by doing a ray cast and checking distance
            # Simple heuristic: if a ray from mid along +Y hits the shape, we're inside
            from .ray import Ray
            from .vector3 import vec3
            test_ray = Ray(origin=mid, direction=vec3(0, 1, 0))
            hit = shape.intersect(test_ray)
            if hit is None:
                result.append(Path3D([a, b]))  # outside = keep
        return result

    def filter_inside(self, shape: "Shape") -> list["Path3D"]:
        """Keep only segments where the midpoint is inside shape (CSG intersection)."""
        result = []
        pts = np.array(self.points, dtype=np.float64)
        if len(pts) < 2:
            return []
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            mid = (a + b) * 0.5
            from .ray import Ray
            from .vector3 import vec3
            test_ray = Ray(origin=mid, direction=vec3(0, 1, 0))
            hit = shape.intersect(test_ray)
            if hit is not None:
                result.append(Path3D([a, b]))  # inside = keep
        return result

    def simplify(self, tolerance: float) -> "Path3D":
        """Ramer-Douglas-Peucker simplification in 3D.

        Parameters
        ----------
        tolerance: Maximum perpendicular distance to keep a point (in world units).
        """
        pts = self.points
        if len(pts) <= 2:
            return Path3D(list(pts))
        result = _rdp_3d(pts, tolerance)
        return Path3D(result)

    def project(
        self,
        view_proj: NDArray[np.float64],
        canvas_w_mm: float,
        canvas_h_mm: float,
        offset_mm: tuple[float, float] = (0.0, 0.0),
    ) -> list[tuple[float, float]] | None:
        """Project 3D points to 2D canvas coordinates (mm).

        Parameters
        ----------
        view_proj:    Combined view × projection matrix (4×4).
        canvas_w_mm:  Canvas width in millimeters.
        canvas_h_mm:  Canvas height in millimeters.
        offset_mm:    (x_off, y_off) offset in mm applied during NDC-to-mm mapping.

        Returns
        -------
        List of (x, y) tuples in mm for the first contiguous visible sub-sequence,
        or None if all points are outside the view frustum.

        Note: use project_segments() to get all visible sub-sequences as separate lists.
        """
        segs = self.project_segments(view_proj, canvas_w_mm, canvas_h_mm, offset_mm)
        return segs[0] if segs else None

    def project_segments(
        self,
        view_proj: NDArray[np.float64],
        canvas_w_mm: float,
        canvas_h_mm: float,
        offset_mm: tuple[float, float] = (0.0, 0.0),
    ) -> list[list[tuple[float, float]]]:
        """Project 3D points to 2D and return all contiguous visible sub-sequences.

        Segments that cross the view frustum boundary are clipped to the boundary
        using the Liang-Barsky algorithm in homogeneous clip space.  This ensures
        that shapes at the canvas edge render with a clean clip rather than a broken
        gap, and that geometry behind the camera is handled correctly before the
        perspective divide.

        When ``offset_mm`` is non-zero, the frustum is adjusted so that clipping
        happens in the correct coordinate space for the offset canvas.  The NDC-to-mm
        mapping is shifted by (x_off, y_off), meaning geometry that falls outside
        the standard NDC range [-1, 1] but is visible on the offset canvas is
        correctly included.

        Parameters
        ----------
        view_proj:    Combined view × projection matrix (4×4).
        canvas_w_mm:  Canvas width in millimeters.
        canvas_h_mm:  Canvas height in millimeters.
        offset_mm:    (x_off, y_off) offset in mm.  The projection maps NDC to
                      ``[x_off, canvas_w_mm + x_off]`` × ``[y_off, canvas_h_mm + y_off]``
                      and clips to ``[0, canvas_w_mm]`` × ``[0, canvas_h_mm]``.

        Returns
        -------
        List of sub-paths (each a list of (x_mm, y_mm) tuples).  Empty list when
        every segment is outside the view frustum.
        """
        x_off, y_off = offset_mm

        # Compute adjusted NDC bounds corresponding to the canvas [0, W] × [0, H].
        #
        # After offset, the NDC-to-mm mapping is:
        #   x_mm = (ndc_x + 1) * 0.5 * W + x_off
        #   y_mm = (1 - (ndc_y + 1) * 0.5) * H + y_off
        #
        # Solving for the NDC values that map to the canvas edges:
        #   x_mm = 0  → ndc_x = -1 - 2*x_off/W  (left edge)
        #   x_mm = W  → ndc_x =  1 - 2*x_off/W  (right edge)
        #   y_mm = 0  → ndc_y =  1 + 2*y_off/H  (top edge, y-axis flipped)
        #   y_mm = H  → ndc_y = -1 + 2*y_off/H  (bottom edge)
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

        # Build offset-adjusted frustum planes.
        # Each row (a, b, c, d) encodes a*x + b*y + c*z + d*w >= 0.
        # With offset (0, 0) this reduces to the standard _FRUSTUM_PLANES constant.
        if x_off == 0.0 and y_off == 0.0:
            frustum_planes = _FRUSTUM_PLANES
        else:
            frustum_planes = np.array([
                [ 1,  0,  0, -x_min_ndc],  # left:   ndc_x >= x_min_ndc
                [-1,  0,  0,  x_max_ndc],  # right:  ndc_x <= x_max_ndc
                [ 0,  1,  0, -y_min_ndc],  # bottom: ndc_y >= y_min_ndc
                [ 0, -1,  0,  y_max_ndc],  # top:    ndc_y <= y_max_ndc
                [ 0,  0,  1,  1],          # near
                [ 0,  0, -1,  1],          # far
            ], dtype=np.float64)

        pts = np.array(self.points, dtype=np.float64)
        n = len(pts)
        if n == 0:
            return []

        # Transform to homogeneous clip space in one batch.
        homogeneous = np.ones((n, 4), dtype=np.float64)
        homogeneous[:, :3] = pts
        clip = homogeneous @ view_proj  # (N, 4)

        result: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []

        for i in range(n - 1):
            p0 = clip[i]
            p1 = clip[i + 1]

            clipped = _clip_segment_homogeneous(p0, p1, frustum_planes)
            if clipped is None:
                # Segment entirely outside the frustum — split the polyline.
                if len(current) >= 2:
                    result.append(current)
                current = []
                continue

            c0, c1 = clipped
            pt0 = _clip_to_mm(c0, canvas_w_mm, canvas_h_mm, x_off, y_off)
            pt1 = _clip_to_mm(c1, canvas_w_mm, canvas_h_mm, x_off, y_off)

            if len(current) == 0:
                # Start a new sub-path.
                current = [pt0, pt1]
            elif _is_inside_frustum(p0, x_min_ndc, x_max_ndc, y_min_ndc, y_max_ndc):
                # p0 is inside the adjusted frustum: this segment connects directly
                # to the previous one (c0 == p0, so pt0 == current[-1]).
                current.append(pt1)
            else:
                # p0 is outside: the previous segment ended at a frustum boundary
                # that is a different point from where this segment enters.
                # Flush the current sub-path and start a fresh one.
                if len(current) >= 2:
                    result.append(current)
                current = [pt0, pt1]

        if len(current) >= 2:
            result.append(current)
        return result

    def to_polyline(
        self,
        view_proj: NDArray[np.float64],
        canvas_w_mm: float,
        canvas_h_mm: float,
        offset_mm: tuple[float, float] = (0.0, 0.0),
    ) -> list[tuple[float, float]] | None:
        """Project and return a 2D Polyline in mm coordinates, or None if invisible.

        When the path crosses the frustum boundary multiple times, only the first
        contiguous visible sub-sequence is returned.  Use project_segments() to
        retrieve all visible sub-sequences.
        """
        return self.project(view_proj, canvas_w_mm, canvas_h_mm, offset_mm)


def _rdp_3d(points: list[Vec3], tolerance: float) -> list[Vec3]:
    """Recursive Ramer-Douglas-Peucker in 3D."""
    if len(points) <= 2:
        return list(points)
    pts = np.array(points, dtype=np.float64)
    start, end = pts[0], pts[-1]
    d = end - start
    d_len = np.linalg.norm(d)
    if d_len < 1e-12:
        # Degenerate: all points at same location
        dists = np.linalg.norm(pts - start, axis=1)
    else:
        d_unit = d / d_len
        # Distance from each point to the line start→end
        vecs = pts - start
        proj = np.dot(vecs, d_unit)[:, np.newaxis] * d_unit
        perp = vecs - proj
        dists = np.linalg.norm(perp, axis=1)

    max_idx = int(np.argmax(dists))
    max_dist = dists[max_idx]

    if max_dist <= tolerance:
        return [points[0], points[-1]]

    left = _rdp_3d(points[:max_idx + 1], tolerance)
    right = _rdp_3d(points[max_idx:], tolerance)
    return left[:-1] + right
