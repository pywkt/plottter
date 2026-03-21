"""Scene3DGenerator — 3D line art renderer integrated into the Plottter generator system."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
    StringParam,
)
from plottter.models import Canvas, Polyline


# ---------------------------------------------------------------------------
# Shadow hatching helpers (task 29.3)
# ---------------------------------------------------------------------------

def _offset_polyline_2d(
    poly: list[tuple[float, float]],
    offset: float,
) -> list[tuple[float, float]]:
    """Return a parallel offset of a 2D polyline by *offset* mm.

    Positive offset is to the left of the direction of travel; negative offset
    is to the right.  Returns an empty list if the polyline has fewer than
    2 points.  Short or degenerate segments are skipped when computing normals.
    """
    if len(poly) < 2:
        return []

    result: list[tuple[float, float]] = []
    n = len(poly)

    for i in range(n):
        x, y = poly[i]
        normals: list[tuple[float, float]] = []

        # Normal from the previous segment
        if i > 0:
            px, py = poly[i - 1]
            dx, dy = x - px, y - py
            d = math.hypot(dx, dy)
            if d > 1e-10:
                normals.append((-dy / d, dx / d))

        # Normal from the next segment
        if i < n - 1:
            nx, ny = poly[i + 1]
            dx, dy = nx - x, ny - y
            d = math.hypot(dx, dy)
            if d > 1e-10:
                normals.append((-dy / d, dx / d))

        if not normals:
            result.append((x, y))
            continue

        avg_nx = sum(nn[0] for nn in normals) / len(normals)
        avg_ny = sum(nn[1] for nn in normals) / len(normals)
        avg_d = math.hypot(avg_nx, avg_ny)
        if avg_d > 1e-10:
            avg_nx /= avg_d
            avg_ny /= avg_d

        result.append((x + avg_nx * offset, y + avg_ny * offset))

    return result


def _hatch_shadow_polylines(
    shadow_polys: list[list[tuple[float, float]]],
    density: float,
    shadow_style: str,
    hatch_angle_deg: float = 45.0,
) -> list[list[tuple[float, float]]]:
    """Generate additional hatching lines for shadowed polylines.

    Combined with the original shadow polylines (which are kept as-is in the
    output), the hatching lines give shadowed regions a visually darker,
    denser appearance suitable for pen-plotter output.

    Parameters
    ----------
    shadow_polys:    Visible-but-shadowed polylines from Scene.render().
    density:         Shadow density (controls number/spacing of extra lines).
    shadow_style:    One of "Thicken", "Hatch", or "Cross-Hatch".
    hatch_angle_deg: Angle of hatching lines in degrees (Hatch / Cross-Hatch).

    Returns
    -------
    A list of additional polylines to append to the generator output alongside
    the original shadow polylines.
    """
    result: list[list[tuple[float, float]]] = []
    if not shadow_polys or density <= 0:
        return result

    # spacing between parallel lines / between tick centres along the path
    spacing = max(0.05, 1.0 / density)
    # number of offset copies on *each* side for Thicken mode
    n_offsets = max(1, round(density))

    if shadow_style == "Thicken":
        for poly in shadow_polys:
            if len(poly) < 2:
                continue
            for k in range(1, n_offsets + 1):
                dist = k * spacing
                left = _offset_polyline_2d(poly, dist)
                if len(left) >= 2:
                    result.append(left)
                right = _offset_polyline_2d(poly, -dist)
                if len(right) >= 2:
                    result.append(right)

    elif shadow_style in ("Hatch", "Cross-Hatch"):
        angles_deg = [hatch_angle_deg]
        if shadow_style == "Cross-Hatch":
            angles_deg.append(hatch_angle_deg + 90.0)

        tick_half = spacing * 0.75  # half-length of each tick mark

        for angle_deg in angles_deg:
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            for poly in shadow_polys:
                if len(poly) < 2:
                    continue
                dist_accum = 0.0
                next_tick = 0.0  # place first tick at path start

                for i in range(len(poly) - 1):
                    ax, ay = poly[i]
                    bx, by = poly[i + 1]
                    seg_dx, seg_dy = bx - ax, by - ay
                    seg_len = math.hypot(seg_dx, seg_dy)
                    if seg_len < 1e-10:
                        dist_accum += seg_len
                        continue

                    while next_tick <= dist_accum + seg_len:
                        t = (next_tick - dist_accum) / seg_len
                        t = max(0.0, min(1.0, t))
                        cx = ax + t * seg_dx
                        cy = ay + t * seg_dy
                        result.append([
                            (cx - cos_a * tick_half, cy - sin_a * tick_half),
                            (cx + cos_a * tick_half, cy + sin_a * tick_half),
                        ])
                        next_tick += spacing

                    dist_accum += seg_len

    return result


# ---------------------------------------------------------------------------
# Ground-plane shadow helpers (task 29.4)
# ---------------------------------------------------------------------------

def _project_to_ground_z(
    px: float, py: float, pz: float,
    lx: float, ly: float, lz: float,
    ground_z: float,
) -> "tuple[float, float, float] | None":
    """Project a 3D point onto the ground plane ``z = ground_z`` along the shadow direction.

    The shadow direction is *opposite* to the light direction: a ray from ``P``
    in direction ``(-lx, -ly, -lz)`` hits the ground plane at

        t = (pz - ground_z) / lz
        shadow = (px - lx*t,  py - ly*t,  ground_z)

    Returns ``None`` when the projection is invalid (light is nearly horizontal,
    or the point is below the ground plane so the shadow falls "upward").
    """
    if abs(lz) < 1e-6:
        return None          # Horizontal light — shadow goes to infinity
    t = (pz - ground_z) / lz
    if t < 0.0:
        return None          # Point is below the ground plane
    return (px - lx * t, py - ly * t, ground_z)


def _compute_ground_shadow_paths(
    shape,
    light_dir_norm: "np.ndarray",
    ground_z: float,
    shadow_density: float,
    shadow_hatch_angle_deg: float,
) -> list:
    """Return Path3D objects on the ground plane representing the shadow of *shape*.

    Two components are returned:
    1. Convex hull outline — a closed polyline tracing the silhouette boundary of
       the projected shadow.
    2. Hatching fill — parallel lines clipped to the convex hull of the projected
       points, giving the shadow a filled appearance.

    Projected wireframe edges are intentionally excluded so the shadow looks like
    a proper shadow shape rather than a duplicate wireframe object.

    The returned paths are 3D (z = ground_z) and should be passed through the
    normal HLR pipeline so they are correctly occluded by scene objects.
    """
    from plottter.scene3d.path3d import Path3D as _Path3D

    try:
        from shapely.geometry import MultiPoint  # type: ignore[import]
    except ImportError:
        MultiPoint = None

    lx = float(light_dir_norm[0])
    ly = float(light_dir_norm[1])
    lz = float(light_dir_norm[2])

    if lz < 1e-6:
        # Light is horizontal or from below — skip
        return []

    all_shadow_pts_xy: list[tuple[float, float]] = []

    # Collect all projected shadow points (but do NOT add wireframe edges as paths)
    for path in shape.paths():
        pts = np.asarray(path.points, dtype=np.float64)
        if len(pts) < 1:
            continue
        for pt in pts:
            result = _project_to_ground_z(
                float(pt[0]), float(pt[1]), float(pt[2]),
                lx, ly, lz, ground_z
            )
            if result is not None:
                sx, sy, _ = result
                all_shadow_pts_xy.append((sx, sy))

    if len(all_shadow_pts_xy) < 3 or MultiPoint is None:
        return []

    hull = MultiPoint(all_shadow_pts_xy).convex_hull
    if hull.is_empty or hull.area < 1e-6:
        return []

    shadow_path3ds: list = []

    # 1. Add convex hull outline as a closed polyline.
    # Degenerate hulls (LineString, Point) have area == 0 and are already
    # rejected by the hull.area < 1e-6 guard above, so hull.exterior is safe.
    hull_coords = list(hull.exterior.coords)
    if len(hull_coords) >= 2:
        hull_pts_3d = [np.array([x, y, ground_z], dtype=np.float64) for x, y in hull_coords]
        shadow_path3ds.append(_Path3D(hull_pts_3d))

    # 2. Add hatching fill using the pre-computed hull
    hatch_paths = _generate_shadow_ground_hatching(
        all_shadow_pts_xy, ground_z, shadow_density, shadow_hatch_angle_deg,
        precomputed_hull=hull,
    )
    shadow_path3ds.extend(hatch_paths)

    return shadow_path3ds


def _generate_shadow_ground_hatching(
    pts_xy: list[tuple[float, float]],
    ground_z: float,
    density: float,
    hatch_angle_deg: float,
    precomputed_hull=None,
) -> list:
    """Fill the convex hull of the shadow projection with parallel hatching lines.

    Returns Path3D objects at z = ground_z ready for the HLR pipeline.
    If *precomputed_hull* is provided (a Shapely geometry), it is used directly
    instead of re-computing it from *pts_xy*.
    """
    from plottter.scene3d.path3d import Path3D as _Path3D

    try:
        from shapely.geometry import (  # type: ignore[import]
            MultiPoint,
            LineString,
            MultiLineString,
            GeometryCollection,
        )
    except ImportError:
        return []

    if len(pts_xy) < 3:
        return []

    spacing = max(0.01, 1.0 / max(density, 1e-6))

    if precomputed_hull is not None:
        hull = precomputed_hull
    else:
        hull = MultiPoint(pts_xy).convex_hull
    if hull.is_empty or hull.area < 1e-6:
        return []

    hatch_angle_rad = math.radians(hatch_angle_deg)
    cos_a = math.cos(hatch_angle_rad)
    sin_a = math.sin(hatch_angle_rad)
    # Perpendicular direction (used to step parallel lines)
    perp_x, perp_y = -sin_a, cos_a

    minx, miny, maxx, maxy = hull.bounds
    corners = [(minx, miny), (maxx, miny), (minx, maxy), (maxx, maxy)]
    perp_projs = [cx * perp_x + cy * perp_y for cx, cy in corners]
    t_start, t_end = min(perp_projs), max(perp_projs)

    # Extend lines far enough to span the bounding box before clipping
    extent = max(maxx - minx, maxy - miny) * 2.0 + 1.0

    def _extract_linestrings(geom):
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            return [coords] if len(coords) >= 2 else []
        if isinstance(geom, (MultiLineString, GeometryCollection)):
            segs = []
            for g in geom.geoms:
                segs.extend(_extract_linestrings(g))
            return segs
        return []

    result: list = []
    t = t_start
    while t <= t_end:
        cx = t * perp_x
        cy = t * perp_y
        line = LineString([
            (cx - cos_a * extent, cy - sin_a * extent),
            (cx + cos_a * extent, cy + sin_a * extent),
        ])
        clipped = hull.intersection(line)
        if not clipped.is_empty:
            for seg_coords in _extract_linestrings(clipped):
                pts_3d = [np.array([x, y, ground_z], dtype=np.float64) for x, y in seg_coords]
                if len(pts_3d) >= 2:
                    result.append(_Path3D(pts_3d))
        t += spacing

    return result


# ---------------------------------------------------------------------------
# Per-face hatching visibility (task 52.4)
# ---------------------------------------------------------------------------

def _compute_hatching_faces(
    shape,
    scene,
    light_dir_norm: "np.ndarray",
    camera: Any,
    canvas_w_mm: float,
    canvas_h_mm: float,
    offset_mm: tuple[float, float] = (0.0, 0.0),
) -> "list[tuple[list[tuple[float, float]], float]]":
    """Return visible, front-facing surface triangles of *shape* with brightness.

    Unlike ``Scene._compute_visible_faces()``, this function:
    - Only processes triangles from *shape* (not sibling shapes that are present
      solely for occlusion).
    - Offsets the occlusion ray origin along the face normal so that centroids
      which are geometrically *inside* the surface (e.g. triangles on a sphere
      where the centroid lies at radius < sphere_radius) do not self-occlude.
    - Applies *offset_mm* directly in the 2D projection so the hatching aligns
      with the wireframe output.

    Parameters
    ----------
    shape:           The TransformedShape (or bare shape) to generate faces for.
    scene:           The compiled Scene whose BVH is used for occlusion testing.
    light_dir_norm:  Normalised directional light vector (unit length).
    camera:          Camera defining the viewpoint and projection.
    canvas_w_mm:     Canvas width in millimetres.
    canvas_h_mm:     Canvas height in millimetres.
    offset_mm:       (x_off, y_off) canvas offset applied to 2D projections.

    Returns
    -------
    List of ``(projected_vertices, brightness)`` tuples.
    """
    from plottter.scene3d.ray import Ray, EPSILON as _EPS

    eye = np.asarray(camera.eye, dtype=np.float64)
    vp_matrix = camera.view_proj_matrix()
    bvh = scene._bvh
    x_off, y_off = offset_mm

    # 1e-2 world-unit offset along the face normal when shooting the occlusion ray.
    # This ensures that faces on curved surfaces (sphere, cylinder, cone) whose
    # triangle centroids lie slightly inside the surface do not self-occlude.
    _NORMAL_OFFSET = 1e-2

    tris = shape.surface_triangles()
    if not tris:
        return []

    result: list[tuple[list[tuple[float, float]], float]] = []

    for raw_v0, raw_v1, raw_v2 in tris:
        v0 = np.asarray(raw_v0, dtype=np.float64)
        v1 = np.asarray(raw_v1, dtype=np.float64)
        v2 = np.asarray(raw_v2, dtype=np.float64)

        # Face normal via cross product.
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal_len = float(np.linalg.norm(normal))
        if normal_len < _EPS:
            continue  # degenerate triangle
        normal = normal / normal_len

        # View direction: from centroid toward camera eye.
        centroid = (v0 + v1 + v2) / 3.0
        to_eye = eye - centroid
        to_eye_len = float(np.linalg.norm(to_eye))
        if to_eye_len < _EPS:
            continue
        view_dir = to_eye / to_eye_len

        # Back-face cull.
        if float(np.dot(normal, view_dir)) <= 0.0:
            continue

        # Occlusion test: cast from a point offset along the face normal so
        # that curved-surface centroids (which lie inside the surface) do not
        # incorrectly self-occlude.  We use bvh.intersect() (not intersect_any)
        # so we can check the hit shape — shapes that use bbox-based intersection
        # (Cylinder, Cone) would otherwise always self-occlude since the bbox
        # boundary is at the surface and the offset ray immediately re-enters it.
        ray_origin = centroid + normal * _NORMAL_OFFSET
        ray = Ray(origin=ray_origin, direction=view_dir)
        t_max = to_eye_len - _NORMAL_OFFSET
        if t_max > _EPS and bvh is not None:
            occluder = bvh.intersect(ray)
            if occluder is not None and _EPS < occluder.t < t_max and occluder.shape is not shape:
                continue  # occluded by a different shape

        # Diffuse brightness.
        brightness = float(max(0.0, np.dot(normal, light_dir_norm)))

        # Project all 3 vertices to 2D canvas coordinates (mm) + offset.
        projected: list[tuple[float, float]] = []
        skip = False
        for vert in (v0, v1, v2):
            h = np.array([vert[0], vert[1], vert[2], 1.0], dtype=np.float64)
            clip = h @ vp_matrix
            w = clip[3]
            if abs(w) < _EPS:
                skip = True
                break
            ndc_x = clip[0] / w
            ndc_y = clip[1] / w
            x_mm = (ndc_x + 1.0) * 0.5 * canvas_w_mm + x_off
            y_mm = (1.0 - (ndc_y + 1.0) * 0.5) * canvas_h_mm + y_off
            projected.append((float(x_mm), float(y_mm)))

        if skip or len(projected) != 3:
            continue

        result.append((projected, brightness))

    return result


_SHAPE_TYPES = [
    "Sphere",
    "Shaded Sphere",
    "Cube",
    "Striped Cube",
    "Cone",
    "Cylinder",
    "Plane",
    "Terrain",
    "Shard",
    "Mesh Import",
]


@register_generator
class Scene3DGenerator(Generator):
    """3D line art renderer with hidden line removal.

    Each layer represents one 3D shape. Multiple layers share a common
    camera and occlude each other via the BVH-accelerated HLR pipeline.
    """

    name = "3D Scene"
    category = "3d"

    def get_parameters(self) -> list[Parameter]:
        return [
            # ── Shape type ───────────────────────────────────────────
            ChoiceParam(
                name="shape_type",
                label="Shape Type",
                choices=_SHAPE_TYPES,
                default="Sphere",
                description="The 3D primitive to render for this layer",
            ),
            # ── World transform ──────────────────────────────────────
            FloatParam(name="pos_x", label="Position X", min=-20.0, max=20.0, step=0.1, default=0.0,
                       description="X offset in 3D world units"),
            FloatParam(name="pos_y", label="Position Y", min=-20.0, max=20.0, step=0.1, default=0.0,
                       description="Y offset in 3D world units"),
            FloatParam(name="pos_z", label="Position Z", min=-20.0, max=20.0, step=0.1, default=0.0,
                       description="Z offset in 3D world units"),
            FloatParam(name="rot_x", label="Rotation X (°)", min=-360.0, max=360.0, step=1.0, default=0.0,
                       description="Rotation around the X axis in degrees"),
            FloatParam(name="rot_y", label="Rotation Y (°)", min=-360.0, max=360.0, step=1.0, default=0.0,
                       description="Rotation around the Y axis in degrees"),
            FloatParam(name="rot_z", label="Rotation Z (°)", min=-360.0, max=360.0, step=1.0, default=0.0,
                       description="Rotation around the Z axis in degrees"),
            FloatParam(name="uniform_scale", label="Scale", min=0.01, max=20.0, step=0.1, default=1.0,
                       description="Uniform scale applied to the shape"),
            # ── 2D canvas offset (applied after projection) ───────────
            FloatParam(name="x_offset_mm", label="X Offset (mm)", min=-500.0, max=500.0, step=0.5, default=0.0,
                       randomizable=False,
                       description="Horizontal offset of the rendered output on the canvas page (mm)"),
            FloatParam(name="y_offset_mm", label="Y Offset (mm)", min=-500.0, max=500.0, step=0.5, default=0.0,
                       randomizable=False,
                       description="Vertical offset of the rendered output on the canvas page (mm)"),
            # ── Sphere ───────────────────────────────────────────────
            FloatParam(
                name="sphere_radius", label="Radius", min=0.1, max=10.0, step=0.1, default=1.0,
                visible_when={"shape_type": ["Sphere", "Shaded Sphere"]},
                description="Sphere radius in world units",
            ),
            IntParam(
                name="sphere_lat_lines", label="Latitude Lines", min=3, max=30, step=1, default=8,
                visible_when={"shape_type": ["Sphere"]},
                description="Number of latitude (horizontal) circles",
            ),
            IntParam(
                name="sphere_lng_lines", label="Longitude Lines", min=3, max=30, step=1, default=8,
                visible_when={"shape_type": ["Sphere"]},
                description="Number of longitude (vertical) arcs",
            ),
            # ── Shaded Sphere ────────────────────────────────────────
            FloatParam(
                name="shaded_light_x", label="Light X", min=-1.0, max=1.0, step=0.1, default=1.0,
                visible_when={"shape_type": ["Shaded Sphere"]},
                description="Light direction X component (will be normalized)",
            ),
            FloatParam(
                name="shaded_light_y", label="Light Y", min=-1.0, max=1.0, step=0.1, default=1.0,
                visible_when={"shape_type": ["Shaded Sphere"]},
                description="Light direction Y component (will be normalized)",
            ),
            FloatParam(
                name="shaded_light_z", label="Light Z", min=-1.0, max=1.0, step=0.1, default=-1.0,
                visible_when={"shape_type": ["Shaded Sphere"]},
                description="Light direction Z component (will be normalized)",
            ),
            IntParam(
                name="shaded_min_lines", label="Min Lines", min=3, max=50, step=1, default=8,
                visible_when={"shape_type": ["Shaded Sphere"]},
                description="Minimum number of shading lines (lit side)",
            ),
            IntParam(
                name="shaded_max_lines", label="Max Lines", min=3, max=100, step=1, default=30,
                visible_when={"shape_type": ["Shaded Sphere"]},
                description="Maximum number of shading lines (shadow side)",
            ),
            # ── Cube ─────────────────────────────────────────────────
            FloatParam(
                name="cube_size", label="Size", min=0.1, max=10.0, step=0.1, default=1.5,
                visible_when={"shape_type": ["Cube", "Striped Cube"]},
                description="Cube side length in world units",
            ),
            IntParam(
                name="cube_stripes", label="Stripe Count", min=1, max=20, step=1, default=5,
                visible_when={"shape_type": ["Striped Cube"]},
                description="Number of stripes per face (Striped Cube only)",
            ),
            # ── Cone ─────────────────────────────────────────────────
            FloatParam(
                name="cone_radius", label="Base Radius", min=0.1, max=10.0, step=0.1, default=1.0,
                visible_when={"shape_type": ["Cone"]},
                description="Radius of the cone base",
            ),
            FloatParam(
                name="cone_height", label="Height", min=0.1, max=10.0, step=0.1, default=2.0,
                visible_when={"shape_type": ["Cone"]},
                description="Cone height from base to apex",
            ),
            IntParam(
                name="cone_lines", label="Lines", min=3, max=30, step=1, default=12,
                visible_when={"shape_type": ["Cone"]},
                description="Number of lateral lines from apex to base",
            ),
            # ── Cylinder ─────────────────────────────────────────────
            FloatParam(
                name="cyl_radius", label="Radius", min=0.1, max=10.0, step=0.1, default=1.0,
                visible_when={"shape_type": ["Cylinder"]},
                description="Cylinder radius",
            ),
            FloatParam(
                name="cyl_height", label="Height", min=0.1, max=10.0, step=0.1, default=2.0,
                visible_when={"shape_type": ["Cylinder"]},
                description="Cylinder height",
            ),
            IntParam(
                name="cyl_lines", label="Lines", min=3, max=30, step=1, default=12,
                visible_when={"shape_type": ["Cylinder"]},
                description="Number of vertical lines",
            ),
            # ── Plane / Terrain ──────────────────────────────────────
            FloatParam(
                name="plane_size", label="Size", min=0.5, max=20.0, step=0.5, default=4.0,
                visible_when={"shape_type": ["Plane", "Terrain"]},
                description="Half-extent of the plane in each direction",
            ),
            IntParam(
                name="plane_steps", label="Grid Steps", min=2, max=40, step=1, default=10,
                visible_when={"shape_type": ["Plane", "Terrain"]},
                description="Number of grid divisions",
            ),
            # ── Shard ────────────────────────────────────────────────
            FloatParam(
                name="shard_radius", label="Radius", min=0.1, max=10.0, step=0.1, default=1.0,
                visible_when={"shape_type": ["Shard"]},
                description="Equatorial radius of the shard",
            ),
            FloatParam(
                name="shard_height", label="Height", min=0.1, max=10.0, step=0.1, default=2.0,
                visible_when={"shape_type": ["Shard"]},
                description="Total height of the shard (split above/below center)",
            ),
            IntParam(
                name="shard_sides", label="Sides", min=3, max=12, step=1, default=6,
                visible_when={"shape_type": ["Shard"]},
                description="Number of sides on the equatorial polygon",
            ),
            # ── Mesh Import ──────────────────────────────────────────
            StringParam(
                name="mesh_file", label="Mesh File (.obj/.stl)", default="",
                visible_when={"shape_type": ["Mesh Import"]},
                description="Absolute path to an OBJ or STL mesh file",
            ),
            BoolParam(
                name="mesh_all_edges", label="Draw All Edges", default=False,
                visible_when={"shape_type": ["Mesh Import"]},
                description="Draw all triangle edges; when off, only hard/boundary edges are drawn",
            ),
            FloatParam(
                name="mesh_decimate", label="Decimation", min=0.0, max=1.0, step=0.05, default=1.0,
                visible_when={"shape_type": ["Mesh Import"]},
                description=(
                    "Reduce mesh complexity before rendering. "
                    "1.0 = no decimation (full detail). "
                    "0.5 = reduce to ~50% of original face count. "
                    "Recommended for meshes with 50 000+ triangles."
                ),
            ),
            # ── HLR / render quality ──────────────────────────────────
            BoolParam(
                name="hlr_enabled", label="Hidden Line Removal", default=True,
                description="Remove lines occluded by other shapes in the scene",
            ),
            FloatParam(
                name="chop_step", label="HLR Accuracy", min=0.005, max=0.5, step=0.005, default=0.05,
                description="Path segment length for HLR ray casting — smaller = more accurate but slower",
            ),
            # ── Shadow / lighting ─────────────────────────────────────
            BoolParam(
                name="shadow_enabled", label="Enable Shadows", default=False,
                description="Enable shadow computation — adds directional light shading to the scene",
            ),
            FloatParam(
                name="light_azimuth", label="Light Azimuth (°)", min=0.0, max=360.0, step=5.0, default=45.0,
                visible_when={"shadow_enabled": [True]},
                description="Horizontal angle of the light direction in degrees",
            ),
            FloatParam(
                name="light_elevation", label="Light Elevation (°)", min=-90.0, max=90.0, step=5.0, default=45.0,
                visible_when={"shadow_enabled": [True]},
                description="Vertical angle of the light direction above the horizon in degrees",
            ),
            FloatParam(
                name="shadow_density", label="Shadow Density", min=0.1, max=2.0, step=0.1, default=1.0,
                visible_when={"shadow_enabled": [True]},
                description="Density of shadow hatching lines — higher = darker shadows",
            ),
            ChoiceParam(
                name="shadow_style",
                label="Shadow Style",
                choices=["Thicken", "Hatch", "Cross-Hatch"],
                default="Thicken",
                visible_when={"shadow_enabled": [True]},
                description=(
                    "How to render shadow areas: "
                    "Thicken adds parallel offset lines alongside shadow edges; "
                    "Hatch overlays angled tick marks; "
                    "Cross-Hatch uses two perpendicular sets of ticks"
                ),
            ),
            FloatParam(
                name="shadow_hatch_angle", label="Hatch Angle (°)", min=0.0, max=180.0, step=5.0, default=45.0,
                visible_when={"shadow_enabled": [True]},
                description="Angle of hatching lines for Hatch/Cross-Hatch shadow styles",
            ),
            # ── Ground-plane shadow (task 29.4) ──────────────────────
            BoolParam(
                name="shadow_ground_plane", label="Ground Plane Shadow", default=False,
                visible_when={"shadow_enabled": [True]},
                description=(
                    "Project the shape's shadow onto a ground plane to create "
                    "the classic 'shadow on the floor' effect"
                ),
            ),
            FloatParam(
                name="ground_plane_z", label="Ground Plane Z", min=-10.0, max=10.0, step=0.1, default=-2.0,
                visible_when={"shadow_enabled": [True], "shadow_ground_plane": [True]},
                description=(
                    "Z height of the ground plane — shadows are projected "
                    "onto this surface.  Objects above this Z value cast a shadow."
                ),
            ),
            # ── Shadow render mode (task 29.5) ────────────────────────
            ChoiceParam(
                name="shadow_render_mode",
                label="Render Mode",
                choices=["Combined", "Shadow Only", "Lit Only"],
                default="Combined",
                visible_when={"shadow_enabled": [True]},
                description=(
                    "Combined renders everything (default). "
                    "Shadow Only renders only shadow hatching and ground shadow — "
                    "useful for plotting shadows in a separate pen color. "
                    "Lit Only renders only the visible lit wireframe edges without shadow hatching — "
                    "useful for plotting wireframes in a separate pen color."
                ),
            ),
            # ── Render style / hatched fill (task 52.4) ──────────────────
            ChoiceParam(
                name="render_style",
                label="Render Style",
                choices=["Wireframe", "Hatched", "Wireframe + Hatched"],
                default="Wireframe",
                description=(
                    "Wireframe: render only visible edges (default). "
                    "Hatched: fill visible faces with brightness-mapped hatching lines. "
                    "Wireframe + Hatched: combine wireframe edges and hatching fill."
                ),
            ),
            FloatParam(
                name="hatch_density_min",
                label="Hatch Density Min",
                min=0.0, max=5.0, step=0.1, default=0.5,
                visible_when={"render_style": ["Hatched", "Wireframe + Hatched"]},
                description="Hatching density for fully lit faces — 0 = no lines on lit faces",
            ),
            FloatParam(
                name="hatch_density_max",
                label="Hatch Density Max",
                min=1.0, max=20.0, step=0.5, default=4.0,
                visible_when={"render_style": ["Hatched", "Wireframe + Hatched"]},
                description="Hatching density for faces in shadow",
            ),
            FloatParam(
                name="hatch_angle_deg",
                label="Hatch Angle (°)",
                min=0.0, max=180.0, step=5.0, default=45.0,
                visible_when={"render_style": ["Hatched", "Wireframe + Hatched"]},
                description="Angle of hatching lines in degrees",
            ),
            BoolParam(
                name="hatch_cross",
                label="Cross-Hatch",
                default=False,
                visible_when={"render_style": ["Hatched", "Wireframe + Hatched"]},
                description="Add perpendicular cross-hatching for darker areas (brightness < 0.3)",
            ),
        ]

    # ------------------------------------------------------------------
    # Shape construction helpers
    # ------------------------------------------------------------------

    def build_shape(self, params: dict[str, Any]):  # type: ignore[return]
        """Build a shape from params in local (untransformed) space.

        This is a public method used externally to reconstruct shapes from
        saved generator_info params for sibling-shape injection in HLR.
        """
        from plottter.scene3d.shapes import (
            Cone,
            Cube,
            Cylinder,
            Plane,
            Shard,
            ShadedSphere,
            Sphere,
            StripedCube,
            TerrainPlane,
            Mesh,
        )

        shape_type = params.get("shape_type", "Sphere")

        if shape_type == "Sphere":
            return Sphere(
                radius=float(params.get("sphere_radius", 1.0)),
                lat_lines=int(params.get("sphere_lat_lines", 8)),
                lng_lines=int(params.get("sphere_lng_lines", 8)),
            )

        if shape_type == "Shaded Sphere":
            lx = float(params.get("shaded_light_x", 1.0))
            ly = float(params.get("shaded_light_y", 1.0))
            lz = float(params.get("shaded_light_z", -1.0))
            light = np.array([lx, ly, lz], dtype=np.float64)
            norm = np.linalg.norm(light)
            if norm > 1e-9:
                light = light / norm
            else:
                # All-zero direction is invalid — fall back to a sensible default
                light = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
            return ShadedSphere(
                radius=float(params.get("sphere_radius", 1.0)),
                light_dir=light,
                min_lines=int(params.get("shaded_min_lines", 8)),
                max_lines=int(params.get("shaded_max_lines", 30)),
            )

        if shape_type == "Cube":
            return Cube(size=float(params.get("cube_size", 1.5)))

        if shape_type == "Striped Cube":
            return StripedCube(
                size=float(params.get("cube_size", 1.5)),
                stripe_count=int(params.get("cube_stripes", 5)),
            )

        if shape_type == "Cone":
            height = float(params.get("cone_height", 2.0))
            radius = float(params.get("cone_radius", 1.0))
            lines = int(params.get("cone_lines", 12))
            return Cone(
                apex=np.array([0.0, height / 2.0, 0.0]),
                base=np.array([0.0, -height / 2.0, 0.0]),
                radius=radius,
                lines=lines,
            )

        if shape_type == "Cylinder":
            height = float(params.get("cyl_height", 2.0))
            radius = float(params.get("cyl_radius", 1.0))
            lines = int(params.get("cyl_lines", 12))
            return Cylinder(
                bottom=np.array([0.0, -height / 2.0, 0.0]),
                top=np.array([0.0, height / 2.0, 0.0]),
                radius=radius,
                lines=lines,
            )

        if shape_type == "Plane":
            return Plane(
                size=float(params.get("plane_size", 4.0)),
                steps=int(params.get("plane_steps", 10)),
            )

        if shape_type == "Terrain":
            return TerrainPlane(
                size=float(params.get("plane_size", 4.0)),
                steps=int(params.get("plane_steps", 10)),
            )

        if shape_type == "Shard":
            return Shard(
                radius=float(params.get("shard_radius", 1.0)),
                height=float(params.get("shard_height", 2.0)),
                sides=int(params.get("shard_sides", 6)),
            )

        if shape_type == "Mesh Import":
            mesh_file = params.get("mesh_file", "").strip()
            if not mesh_file:
                return None
            decimate = float(params.get("mesh_decimate", 1.0))
            return Mesh(
                file_path=mesh_file,
                draw_all_edges=bool(params.get("mesh_all_edges", False)),
                decimate=decimate,
            )

        # Fallback: sphere
        return Sphere(radius=1.0)

    def build_transformed_shape(self, params: dict[str, Any]):
        """Build a shape and wrap it in a TransformedShape with position/rotation/scale."""
        from plottter.scene3d.shapes.transformed import TransformedShape
        from plottter.scene3d.matrix4 import (
            scale as mat_scale,
            rotate_xyz,
            translate as mat_translate,
            multiply,
        )

        shape = self.build_shape(params)
        if shape is None:
            return None

        s = float(params.get("uniform_scale", 1.0))
        rx = math.radians(float(params.get("rot_x", 0.0)))
        ry = math.radians(float(params.get("rot_y", 0.0)))
        rz = math.radians(float(params.get("rot_z", 0.0)))
        tx = float(params.get("pos_x", 0.0))
        ty = float(params.get("pos_y", 0.0))
        tz = float(params.get("pos_z", 0.0))

        # Compose: scale → rotate → translate
        m = multiply(multiply(mat_scale(s, s, s), rotate_xyz(rx, ry, rz)), mat_translate(tx, ty, tz))
        return TransformedShape(shape, m)

    # ------------------------------------------------------------------
    # Generator ABC
    # ------------------------------------------------------------------

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        from plottter.scene3d import Scene, Camera

        # Build current shape with its transform
        shape_type = params.get("shape_type", "Sphere")
        if shape_type == "Mesh Import" and not params.get("mesh_file", "").strip():
            return []

        current_shape = self.build_transformed_shape(params)

        # Auto-suggest decimation for large meshes (informational only — never auto-decimate).
        if shape_type == "Mesh Import" and current_shape is not None:
            # Unwrap TransformedShape to reach the Mesh object
            from plottter.scene3d.shapes.transformed import TransformedShape
            from plottter.scene3d.shapes.mesh import Mesh as MeshShape
            inner = current_shape.shape if isinstance(current_shape, TransformedShape) else current_shape
            if isinstance(inner, MeshShape):
                face_count = inner.face_count
                if face_count > 50_000 and float(params.get("mesh_decimate", 1.0)) >= 1.0:
                    # Log a hint; the Decimation parameter description in get_parameters()
                    # already guides the user — this is just a runtime reminder.
                    import warnings
                    warnings.warn(
                        f"Mesh has {face_count:,} triangles. "
                        "Consider enabling Decimation (mesh_decimate < 1.0) to speed up rendering.",
                        stacklevel=2,
                    )

        if cancelled_callback is not None and cancelled_callback():
            return []

        # Build camera from injected camera dict
        cam_dict = params.get("_camera", {})
        aspect = canvas.width_mm / canvas.height_mm if canvas.height_mm > 0 else 1.0
        camera = Camera(
            projection=cam_dict.get("projection", "perspective"),
            fov_deg=float(cam_dict.get("fov", 45.0)),
            aspect=aspect,
        )
        camera.set_orbit(
            azimuth_deg=float(cam_dict.get("azimuth", 30.0)),
            elevation_deg=float(cam_dict.get("elevation", 20.0)),
            distance=float(cam_dict.get("distance", 8.0)),
            center=[
                float(cam_dict.get("look_at_x", 0.0)),
                float(cam_dict.get("look_at_y", 0.0)),
                float(cam_dict.get("look_at_z", 0.0)),
            ],
        )

        # Build scene
        hlr_enabled = bool(params.get("hlr_enabled", True))
        chop_step = float(params.get("chop_step", 0.05))
        scene = Scene(hlr_enabled=hlr_enabled, chop_step=chop_step)

        # Add sibling shapes for occlusion (not rendered, just block rays)
        for sibling in params.get("_sibling_3d_shapes", []):
            scene.add(sibling)

        # Add current shape (to be rendered)
        scene.add(current_shape)
        scene.compile()

        if cancelled_callback is not None and cancelled_callback():
            return []

        # Apply canvas offset in the render pipeline so frustum clipping happens
        # in the correct coordinate space (not as a post-projection translation).
        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))

        # Compute scene-level light direction from azimuth/elevation (if shadows enabled).
        light_dir: tuple[float, float, float] | None = None
        if bool(params.get("shadow_enabled", False)):
            az_rad = math.radians(float(params.get("light_azimuth", 45.0)))
            el_rad = math.radians(float(params.get("light_elevation", 45.0)))
            lx = math.cos(el_rad) * math.cos(az_rad)
            ly = math.cos(el_rad) * math.sin(az_rad)
            lz = math.sin(el_rad)
            light_dir = (lx, ly, lz)

            # If the current shape is a ShadedSphere, sync its light direction
            # with the scene light so procedural shading matches the shadow direction.
            from plottter.scene3d.shapes.transformed import TransformedShape
            inner = (
                current_shape.shape
                if isinstance(current_shape, TransformedShape)
                else current_shape
            )
            from plottter.scene3d.shapes.sphere import ShadedSphere
            if isinstance(inner, ShadedSphere):
                inner.set_light_dir(np.array([lx, ly, lz], dtype=np.float64))

        # Read shadow render mode (task 29.5).
        shadow_render_mode = "Combined"
        if light_dir is not None:
            shadow_render_mode = str(params.get("shadow_render_mode", "Combined"))

        # Compute ground-plane shadow paths (task 29.4).
        # These are 3D paths on the ground plane that are fed into the HLR
        # pipeline as "extra_render_paths" — they receive camera occlusion
        # testing (so objects in front hide parts of the shadow) but do NOT
        # receive shadow ray testing (ground shadow lines should not cast
        # their own shadows).
        ground_shadow_paths: list = []
        if (
            light_dir is not None
            and bool(params.get("shadow_ground_plane", False))
        ):
            ground_z = float(params.get("ground_plane_z", -2.0))
            shadow_density = float(params.get("shadow_density", 1.0))
            shadow_hatch_angle = float(params.get("shadow_hatch_angle", 45.0))
            ld_arr = np.array(list(light_dir), dtype=np.float64)
            ld_len = float(np.linalg.norm(ld_arr))
            if ld_len > 1e-9:
                ld_norm = ld_arr / ld_len
                ground_shadow_paths = _compute_ground_shadow_paths(
                    current_shape, ld_norm, ground_z, shadow_density, shadow_hatch_angle
                )

        # For "Combined" mode, include ground shadow in the main render call
        # (ground shadow lands in lit_polylines since it isn't shadow-tested).
        # For "Shadow Only" mode, render ground shadow separately so we can
        # include it without the regular lit wireframe edges.
        # For "Lit Only" mode, skip ground shadow entirely.
        extra_for_main = ground_shadow_paths if (shadow_render_mode == "Combined" and ground_shadow_paths) else None

        # Render only this layer's shape.
        # When light_dir is set, render() returns (lit_polylines, shadow_polylines).
        # Shadow polylines receive additional hatching via _hatch_shadow_polylines (task 29.3).
        render_result = scene.render(
            camera,
            canvas_w_mm=canvas.width_mm,
            canvas_h_mm=canvas.height_mm,
            render_shapes=[current_shape],
            progress_callback=(lambda p: progress_callback(int(p * 100))) if progress_callback else None,
            cancelled_callback=cancelled_callback,
            offset_mm=(x_off, y_off),
            light_dir=light_dir,
            extra_render_paths=extra_for_main,
        )

        if isinstance(render_result, tuple):
            lit_polylines, shadow_polylines = render_result
            # Generate additional hatching lines for the shadowed segments so
            # they appear visually darker/denser than the lit segments.
            shadow_style = str(params.get("shadow_style", "Thicken"))
            shadow_density = float(params.get("shadow_density", 1.0))
            shadow_hatch_angle = float(params.get("shadow_hatch_angle", 45.0))
            hatch_lines = _hatch_shadow_polylines(
                shadow_polylines, shadow_density, shadow_style, shadow_hatch_angle
            )

            if shadow_render_mode == "Lit Only":
                # Only the visible lit wireframe edges — no shadow hatching, no ground shadow.
                polylines = lit_polylines
            elif shadow_render_mode == "Shadow Only":
                # On-surface shadow hatching only, plus ground shadow rendered separately.
                polylines = shadow_polylines + hatch_lines
                if ground_shadow_paths:
                    # Render the ground shadow paths on their own (no regular shape paths,
                    # just HLR occlusion testing against the full scene BVH).
                    ground_render = scene.render(
                        camera,
                        canvas_w_mm=canvas.width_mm,
                        canvas_h_mm=canvas.height_mm,
                        render_shapes=[],
                        progress_callback=None,
                        cancelled_callback=cancelled_callback,
                        offset_mm=(x_off, y_off),
                        light_dir=None,
                        extra_render_paths=ground_shadow_paths,
                    )
                    if isinstance(ground_render, list):
                        polylines = polylines + ground_render
            else:
                # Combined (default): lit edges + shadow edges + hatching lines.
                # Ground shadow is already inside lit_polylines (via extra_for_main).
                polylines = lit_polylines + shadow_polylines + hatch_lines
        else:
            polylines = render_result

        # ── Hatched fill (task 52.4) ──────────────────────────────────────────
        # When render_style includes "Hatched", compute visible faces and fill
        # them with brightness-mapped hatching lines.
        render_style = params.get("render_style", "Wireframe")
        hatch_polylines: list[Polyline] = []

        if render_style in ("Hatched", "Wireframe + Hatched"):
            from plottter.scene3d.hatching import (
                _fill_triangle_with_hatching,
                brightness_to_density,
            )

            hatch_density_min = float(params.get("hatch_density_min", 0.5))
            hatch_density_max = float(params.get("hatch_density_max", 4.0))
            hatch_angle = float(params.get("hatch_angle_deg", 45.0))
            hatch_cross = bool(params.get("hatch_cross", False))

            # Use the scene light direction if shadows are enabled; otherwise use
            # a sensible default directional light for brightness computation.
            if light_dir is not None:
                face_light_dir: tuple[float, float, float] = light_dir
            else:
                # Upper-right directional light (will be normalised inside _compute_visible_faces)
                face_light_dir = (1.0, 1.0, -1.0)

            face_light_norm = np.array(face_light_dir, dtype=np.float64)
            fln_len = float(np.linalg.norm(face_light_norm))
            if fln_len > 1e-9:
                face_light_norm = face_light_norm / fln_len

            visible_faces = _compute_hatching_faces(
                current_shape, scene, face_light_norm, camera,
                canvas.width_mm, canvas.height_mm,
                offset_mm=(x_off, y_off),
            )

            for verts_2d, brightness in visible_faces:
                density = brightness_to_density(
                    brightness, hatch_density_min, hatch_density_max
                )
                # Cross-hatch only for deep-shadow faces (brightness < 0.3).
                cross_for_face = hatch_cross and (brightness < 0.3)
                face_lines = _fill_triangle_with_hatching(
                    verts_2d, density, hatch_angle, cross_for_face
                )
                hatch_polylines.extend(face_lines)

        if render_style == "Wireframe":
            return polylines
        elif render_style == "Hatched":
            return hatch_polylines
        else:  # "Wireframe + Hatched"
            return polylines + hatch_polylines

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Sphere",
                params={
                    "shape_type": "Sphere",
                    "sphere_radius": 1.5,
                    "sphere_lat_lines": 10,
                    "sphere_lng_lines": 10,
                },
            ),
            Preset(
                name="Shaded Sphere",
                params={
                    "shape_type": "Shaded Sphere",
                    "sphere_radius": 1.5,
                    "shaded_min_lines": 6,
                    "shaded_max_lines": 35,
                },
            ),
            Preset(
                name="Cube",
                params={
                    "shape_type": "Striped Cube",
                    "cube_size": 2.0,
                    "cube_stripes": 6,
                },
            ),
            Preset(
                name="Terrain",
                params={
                    "shape_type": "Terrain",
                    "plane_size": 8.0,
                    "plane_steps": 16,
                    "pos_y": -2.0,
                },
            ),
            Preset(
                name="Geometric Still Life",
                params={
                    "shape_type": "Sphere",
                    "sphere_radius": 1.0,
                    "sphere_lat_lines": 8,
                    "sphere_lng_lines": 8,
                },
            ),
            # ── Shadow presets (task 29.5) ────────────────────────────
            Preset(
                name="Dramatic Shadows",
                params={
                    "shape_type": "Sphere",
                    "sphere_radius": 1.5,
                    "sphere_lat_lines": 10,
                    "sphere_lng_lines": 10,
                    "shadow_enabled": True,
                    "light_azimuth": 45.0,
                    "light_elevation": 20.0,
                    "shadow_density": 2.0,
                    "shadow_style": "Thicken",
                    "shadow_ground_plane": True,
                    "ground_plane_z": -2.0,
                    "shadow_render_mode": "Combined",
                },
            ),
            Preset(
                name="Architectural",
                params={
                    "shape_type": "Cube",
                    "cube_size": 2.0,
                    "shadow_enabled": True,
                    "light_azimuth": 225.0,
                    "light_elevation": 45.0,
                    "shadow_density": 1.0,
                    "shadow_style": "Hatch",
                    "shadow_hatch_angle": 45.0,
                    "shadow_ground_plane": True,
                    "ground_plane_z": -1.0,
                    "shadow_render_mode": "Combined",
                },
            ),
            Preset(
                name="Subtle Shading",
                params={
                    "shape_type": "Shaded Sphere",
                    "sphere_radius": 1.5,
                    "shaded_min_lines": 6,
                    "shaded_max_lines": 30,
                    "shadow_enabled": True,
                    "light_azimuth": 45.0,
                    "light_elevation": 70.0,
                    "shadow_density": 0.5,
                    "shadow_style": "Thicken",
                    "shadow_ground_plane": False,
                    "shadow_render_mode": "Combined",
                },
            ),
            # ── Hatching presets (task 52.4) ──────────────────────────────
            Preset(
                name="Hatched Sphere",
                params={
                    "shape_type": "Sphere",
                    "sphere_radius": 1.5,
                    "sphere_lat_lines": 10,
                    "sphere_lng_lines": 10,
                    "render_style": "Hatched",
                    "hatch_density_min": 0.5,
                    "hatch_density_max": 4.0,
                    "hatch_angle_deg": 45.0,
                    "hatch_cross": False,
                },
            ),
            Preset(
                name="Cross-Hatched Cube",
                params={
                    "shape_type": "Cube",
                    "cube_size": 2.0,
                    "render_style": "Wireframe + Hatched",
                    "hatch_density_min": 0.5,
                    "hatch_density_max": 4.0,
                    "hatch_angle_deg": 30.0,
                    "hatch_cross": True,
                },
            ),
            Preset(
                name="Pen & Ink Portrait",
                params={
                    "shape_type": "Mesh Import",
                    "mesh_file": "",
                    "render_style": "Hatched",
                    "hatch_density_min": 0.0,
                    "hatch_density_max": 6.0,
                    "hatch_angle_deg": 45.0,
                    "hatch_cross": False,
                },
            ),
        ]
