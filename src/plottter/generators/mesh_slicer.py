"""Plane-mesh intersection (mesh slicing) utilities.

Provides:
- _slice_mesh:       Intersect a triangle mesh with a plane → raw 3D segments.
- _chain_segments:   Chain segments into closed polylines (tolerance-based).
- _project_to_2d:    Drop the dominant normal axis to get 2D contours.
- slice_mesh:        All-in-one public function.
- _slice_mesh_multi: Slice along num_slices evenly-spaced planes → list of lists of 2D polylines.
- MeshSlicerGenerator: Generator class for multi-plane mesh slicing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from plottter.generators import register_generator
from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    FileParam,
    FloatParam,
    Generator,
    IntParam,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline


# ---------------------------------------------------------------------------
# Core intersection
# ---------------------------------------------------------------------------

def _slice_mesh(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    plane_origin: NDArray[np.float64],
    plane_normal: NDArray[np.float64],
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Intersect a triangle mesh with a plane and return 3D line segments.

    For each triangle that straddles the plane, computes the two edge-crossing
    intersection points and returns them as a (P1, P2) tuple.

    Parameters
    ----------
    vertices:       (N, 3) float64 vertex positions.
    faces:          (M, 3) int32 vertex indices (triangles).
    plane_origin:   A point on the plane, shape (3,).
    plane_normal:   Plane normal vector, shape (3,).  Need not be unit length.

    Returns
    -------
    List of (P1, P2) tuples where P1 and P2 are 3D numpy float64 arrays.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    origin = np.asarray(plane_origin, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    # Signed distances for all vertices: d[i] = dot(vertices[i] - origin, normal)
    vdists = (vertices - origin) @ normal  # (N,)

    # Per-face signed distances: shape (M, 3)
    d = vdists[faces]
    # Per-face vertex positions: shape (M, 3, 3)
    v = vertices[faces]

    segments: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    for i in range(len(faces)):
        da, db, dc = float(d[i, 0]), float(d[i, 1]), float(d[i, 2])

        # Skip if all three vertices are on the same side (all positive or all negative)
        if (da > 0 and db > 0 and dc > 0) or (da < 0 and db < 0 and dc < 0):
            continue

        va, vb, vc = v[i, 0], v[i, 1], v[i, 2]

        # Find intersection points on crossing edges
        pts: list[NDArray[np.float64]] = []
        edge_dists = [(da, db, va, vb), (db, dc, vb, vc), (dc, da, vc, va)]
        for d1, d2, p1, p2 in edge_dists:
            if (d1 > 0) == (d2 > 0):
                # Both on the same side (both positive, or both non-positive):
                # edge does not strictly cross the plane.
                # Vertices with d=0 are captured via their adjacent crossing edge.
                continue
            denom = d1 - d2
            if abs(denom) < 1e-14:
                continue
            t = d1 / denom
            pts.append(p1 + t * (p2 - p1))

        if len(pts) == 2:
            segments.append((pts[0], pts[1]))

    return segments


# ---------------------------------------------------------------------------
# Segment chaining
# ---------------------------------------------------------------------------

def _chain_edges_by_index(
    edges: list[tuple[int, int]],
) -> list[list[int]]:
    """Chain (a, b) integer edge pairs into polylines.

    Same greedy algorithm as ``Mesh._chain_edges``:
    walks from branch/endpoint vertices first, then handles remaining loops.

    Returns a list of vertex-index chains.  Closed loops have the same
    start and end index.
    """
    if not edges:
        return []

    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    unvisited: set[tuple[int, int]] = set()
    for a, b in edges:
        unvisited.add((min(a, b), max(a, b)))

    chains: list[list[int]] = []

    def walk(start: int, nxt: int) -> list[int]:
        key = (min(start, nxt), max(start, nxt))
        if key not in unvisited:
            return []
        unvisited.discard(key)
        chain = [start, nxt]
        prev, curr = start, nxt
        while True:
            if len(adj[curr]) != 2:
                break
            next_node: int | None = None
            for n in adj[curr]:
                if n != prev:
                    k = (min(curr, n), max(curr, n))
                    if k in unvisited:
                        next_node = n
                        break
            if next_node is None:
                break
            k = (min(curr, next_node), max(curr, next_node))
            unvisited.discard(k)
            chain.append(next_node)
            prev, curr = curr, next_node
        return chain

    # Walk from branch/endpoint vertices (degree ≠ 2)
    branch_vertices = [v for v, nbrs in adj.items() if len(nbrs) != 2]
    for bv in branch_vertices:
        for neighbor in adj[bv]:
            key = (min(bv, neighbor), max(bv, neighbor))
            if key in unvisited:
                chain = walk(bv, neighbor)
                if len(chain) >= 2:
                    chains.append(chain)

    # Handle remaining unvisited edges (closed loops)
    while unvisited:
        key = next(iter(unvisited))
        a, b = key
        chain = walk(a, b)
        if len(chain) >= 2:
            chains.append(chain)

    return chains


def _chain_segments(
    segments: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    tol: float = 1e-6,
) -> list[list[NDArray[np.float64]]]:
    """Chain 3D line segments into polylines by matching shared endpoints.

    Uses a tolerance-based spatial hash to identify matching endpoints, then
    runs the same greedy chaining algorithm as ``Mesh._chain_edges``.

    Parameters
    ----------
    segments:   List of (P1, P2) endpoint pairs in 3D.
    tol:        Coordinate matching tolerance.  Points within this distance
                are considered identical.

    Returns
    -------
    List of polylines, each a list of 3D numpy arrays.
    Closed loops have the same first and last point.
    """
    if not segments:
        return []

    # Map floating-point 3D points → integer indices using a grid hash
    scale = 1.0 / tol
    point_to_idx: dict[tuple[int, int, int], int] = {}
    points: list[NDArray[np.float64]] = []

    def get_idx(pt: NDArray[np.float64]) -> int:
        key = (
            int(round(float(pt[0]) * scale)),
            int(round(float(pt[1]) * scale)),
            int(round(float(pt[2]) * scale)),
        )
        if key not in point_to_idx:
            idx = len(points)
            point_to_idx[key] = idx
            points.append(pt.copy())
        return point_to_idx[key]

    # Build integer edge list
    edges: list[tuple[int, int]] = []
    for p1, p2 in segments:
        i1 = get_idx(p1)
        i2 = get_idx(p2)
        if i1 != i2:
            edges.append((i1, i2))

    # Chain edges
    chains_idx = _chain_edges_by_index(edges)

    # Convert index chains back to 3D point lists
    result: list[list[NDArray[np.float64]]] = []
    for chain in chains_idx:
        result.append([points[i] for i in chain])
    return result


# ---------------------------------------------------------------------------
# 2D projection
# ---------------------------------------------------------------------------

def _project_to_2d(
    contours_3d: list[list[NDArray[np.float64]]],
    plane_normal: NDArray[np.float64],
) -> list[list[tuple[float, float]]]:
    """Project 3D contour points to 2D by dropping the dominant normal axis.

    For a Z-slicing plane (normal = [0, 0, 1]), drops Z and keeps (X, Y).
    More generally, drops the axis with the largest absolute component in the
    normalised plane normal.

    Parameters
    ----------
    contours_3d:    List of polylines, each a list of (3,) numpy arrays.
    plane_normal:   Plane normal vector; used to determine which axis to drop.

    Returns
    -------
    List of polylines, each a list of (x, y) float tuples.
    """
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    drop_axis = int(np.argmax(np.abs(n)))
    keep_axes = [i for i in range(3) if i != drop_axis]

    result: list[list[tuple[float, float]]] = []
    for contour_3d in contours_3d:
        contour_2d = [
            (float(pt[keep_axes[0]]), float(pt[keep_axes[1]]))
            for pt in contour_3d
        ]
        result.append(contour_2d)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def slice_mesh(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    plane_origin: NDArray[np.float64],
    plane_normal: NDArray[np.float64],
    tol: float = 1e-6,
    project: bool = True,
) -> list[list[tuple[float, float]]] | list[list[NDArray[np.float64]]]:
    """Slice a triangle mesh with a plane and return closed 2D (or 3D) contours.

    Combines :func:`_slice_mesh`, :func:`_chain_segments`, and optionally
    :func:`_project_to_2d` into a single convenience function.

    Parameters
    ----------
    vertices:       (N, 3) float64 vertex positions.
    faces:          (M, 3) int32 vertex indices.
    plane_origin:   A point on the cutting plane.
    plane_normal:   Normal vector of the cutting plane.
    tol:            Endpoint-matching tolerance for chaining.
    project:        If True (default), project contours to 2D using the plane
                    normal to determine the drop axis.  If False, return 3D.

    Returns
    -------
    List of closed polylines.  When ``project=True``, each polyline is a list
    of ``(x, y)`` float tuples.  When ``project=False``, a list of 3D numpy
    arrays.
    """
    segments = _slice_mesh(vertices, faces, plane_origin, plane_normal)
    contours_3d = _chain_segments(segments, tol=tol)
    if project:
        return _project_to_2d(contours_3d, plane_normal)
    return contours_3d


# ---------------------------------------------------------------------------
# Multi-plane slicing
# ---------------------------------------------------------------------------

_AXIS_MAP: dict[str, int] = {"X": 0, "Y": 1, "Z": 2}


def _slice_mesh_multi(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    axis: str,
    num_slices: int,
    z_min: float,
    z_max: float,
    tol: float = 1e-6,
) -> list[list[list[tuple[float, float]]]]:
    """Generate ``num_slices`` evenly-spaced planes and slice the mesh with each.

    Planes are distributed between ``z_min`` and ``z_max`` exclusive (endpoints
    are not included so that degenerate tip slices are avoided).

    Parameters
    ----------
    vertices:   (N, 3) float64 vertex positions.
    faces:      (M, 3) int32 vertex indices.
    axis:       Slicing axis — one of "X", "Y", "Z".
    num_slices: Number of evenly-spaced cutting planes.
    z_min:      Lower bound along ``axis``.
    z_max:      Upper bound along ``axis``.
    tol:        Endpoint-matching tolerance for segment chaining.

    Returns
    -------
    List of length ``num_slices``.  Each element is a list of 2D polylines
    (``list[list[tuple[float, float]]]``) produced by slicing at one plane.
    """
    axis_idx = _AXIS_MAP.get(axis.upper(), 2)
    normal = np.zeros(3, dtype=np.float64)
    normal[axis_idx] = 1.0

    # num_slices interior planes between z_min and z_max (endpoints excluded)
    positions: NDArray[np.float64] = np.linspace(z_min, z_max, num_slices + 2)[1:-1]

    result: list[list[list[tuple[float, float]]]] = []
    for pos in positions:
        origin = np.zeros(3, dtype=np.float64)
        origin[axis_idx] = float(pos)
        contours_2d: list[list[tuple[float, float]]] = slice_mesh(  # type: ignore[assignment]
            vertices, faces, origin, normal, tol=tol, project=True
        )
        result.append(contours_2d)

    return result


# ---------------------------------------------------------------------------
# MeshSlicerGenerator
# ---------------------------------------------------------------------------


@register_generator
class MeshSlicerGenerator(Generator):
    """Multi-plane mesh slicer.

    Loads an STL or OBJ file, slices it with ``num_slices`` evenly-spaced
    planes, and stacks the resulting 2D contours vertically for display.
    """

    name = "Mesh Slicer"
    category = "3d"

    def get_parameters(self) -> list[Parameter]:
        return [
            FileParam(
                name="mesh_file",
                label="Mesh File",
                default="",
                filter="Mesh Files (*.stl *.obj);;All Files (*)",
                description="Path to an OBJ or STL mesh file",
            ),
            ChoiceParam(
                name="slice_axis",
                label="Slice Axis",
                choices=["X", "Y", "Z"],
                default="Z",
                description="Axis along which to slice the mesh",
            ),
            IntParam(
                name="num_slices",
                label="Number of Slices",
                min=5,
                max=200,
                step=1,
                default=40,
                description="Number of parallel cutting planes",
            ),
            ChoiceParam(
                name="view_mode",
                label="View Mode",
                choices=["Stacked", "Plan View"],
                default="Stacked",
                description=(
                    "Stacked: slices are offset vertically for a side-profile look. "
                    "Plan View: all contours overlaid at the same position (topographic map)."
                ),
            ),
            FloatParam(
                name="slice_spacing_mm",
                label="Slice Spacing (mm)",
                min=0.5,
                max=10.0,
                step=0.1,
                default=2.0,
                visible_when={"view_mode": ["Stacked"]},
                description=(
                    "Vertical spacing between contour lines when displayed — "
                    "controls how spread out the slices appear"
                ),
            ),
            FloatParam(
                name="scale",
                label="Scale",
                min=0.1,
                max=100.0,
                step=0.1,
                default=10.0,
                description="Scale factor for the mesh",
            ),
            FloatParam(
                name="rot_x",
                label="Rotation X (°)",
                min=-360.0,
                max=360.0,
                step=1.0,
                default=0.0,
                description="Rotate the mesh around the X axis before slicing",
            ),
            FloatParam(
                name="rot_y",
                label="Rotation Y (°)",
                min=-360.0,
                max=360.0,
                step=1.0,
                default=0.0,
                description="Rotate the mesh around the Y axis before slicing",
            ),
            FloatParam(
                name="rot_z",
                label="Rotation Z (°)",
                min=-360.0,
                max=360.0,
                step=1.0,
                default=0.0,
                description="Rotate the mesh around the Z axis before slicing",
            ),
            BoolParam(
                name="flip_vertical",
                label="Flip Vertical",
                default=False,
                description="Flip the output vertically — useful when the mesh appears upside down",
            ),
        ]

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        import os

        from plottter.scene3d.loaders import load_obj, load_stl

        mesh_file = params.get("mesh_file", "").strip()
        if not mesh_file or not os.path.isfile(mesh_file):
            return []

        # Load mesh
        ext = os.path.splitext(mesh_file)[1].lower()
        if ext == ".stl":
            vertices, faces = load_stl(mesh_file)
        elif ext == ".obj":
            vertices, faces = load_obj(mesh_file)
        else:
            return []

        if len(faces) == 0:
            return []

        # Apply rotation to vertices before slicing
        rot_x = float(params.get("rot_x", 0.0))
        rot_y = float(params.get("rot_y", 0.0))
        rot_z = float(params.get("rot_z", 0.0))
        if rot_x != 0.0 or rot_y != 0.0 or rot_z != 0.0:
            import math
            # Rotation matrices applied in order: X → Y → Z
            if rot_x != 0.0:
                a = math.radians(rot_x)
                c, s = math.cos(a), math.sin(a)
                rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
                vertices = vertices @ rx.T
            if rot_y != 0.0:
                a = math.radians(rot_y)
                c, s = math.cos(a), math.sin(a)
                ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
                vertices = vertices @ ry.T
            if rot_z != 0.0:
                a = math.radians(rot_z)
                c, s = math.cos(a), math.sin(a)
                rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
                vertices = vertices @ rz.T

        slice_axis = str(params.get("slice_axis", "Z"))
        num_slices = max(1, int(params.get("num_slices", 40)))
        view_mode = str(params.get("view_mode", "Stacked"))
        slice_spacing_mm = float(params.get("slice_spacing_mm", 2.0))
        scale = float(params.get("scale", 10.0))
        plan_view = view_mode == "Plan View"

        # Compute mesh bounds along the slice axis
        axis_idx = _AXIS_MAP.get(slice_axis.upper(), 2)
        z_min = float(vertices[:, axis_idx].min())
        z_max = float(vertices[:, axis_idx].max())

        if z_max <= z_min:
            return []

        if progress_callback:
            progress_callback(10)
        if cancelled_callback and cancelled_callback():
            return []

        # Generate multi-plane slices
        all_slices = _slice_mesh_multi(
            vertices, faces,
            axis=slice_axis,
            num_slices=num_slices,
            z_min=z_min,
            z_max=z_max,
        )

        if progress_callback:
            progress_callback(70)
        if cancelled_callback and cancelled_callback():
            return []

        # Build polylines: scale mesh coordinates, then stack vertically (or overlay in plan view)
        raw_polylines: list[Polyline] = []
        for slice_idx, contours in enumerate(all_slices):
            y_offset = 0.0 if plan_view else slice_idx * slice_spacing_mm
            for contour in contours:
                if len(contour) < 2:
                    continue
                poly: Polyline = [
                    (pt[0] * scale, pt[1] * scale + y_offset)
                    for pt in contour
                ]
                raw_polylines.append(poly)

        if not raw_polylines:
            return []

        # Flip vertical if requested
        flip_vertical = bool(params.get("flip_vertical", False))
        if flip_vertical:
            all_ys_raw = [pt[1] for poly in raw_polylines for pt in poly]
            y_center = (min(all_ys_raw) + max(all_ys_raw)) / 2.0
            raw_polylines = [
                [(pt[0], 2.0 * y_center - pt[1]) for pt in poly]
                for poly in raw_polylines
            ]

        # Center the result on the canvas
        all_xs = [pt[0] for poly in raw_polylines for pt in poly]
        all_ys = [pt[1] for poly in raw_polylines for pt in poly]
        cx = (min(all_xs) + max(all_xs)) / 2.0
        cy = (min(all_ys) + max(all_ys)) / 2.0
        dx = canvas.width_mm / 2.0 - cx
        dy = canvas.height_mm / 2.0 - cy

        polylines: list[Polyline] = [
            [(pt[0] + dx, pt[1] + dy) for pt in poly]
            for poly in raw_polylines
        ]

        if progress_callback:
            progress_callback(100)

        return polylines

    def get_presets(self) -> list[Preset]:
        return [
            Preset(
                name="Default",
                params={
                    "slice_axis": "Z",
                    "num_slices": 40,
                    "view_mode": "Stacked",
                    "slice_spacing_mm": 2.0,
                    "scale": 10.0,
                },
                description="Default mesh slicing settings",
            ),
            Preset(
                name="Dense Slices",
                params={
                    "slice_axis": "Z",
                    "num_slices": 100,
                    "view_mode": "Stacked",
                    "slice_spacing_mm": 1.0,
                    "scale": 8.0,
                },
                description="Many thin slices for high detail",
            ),
            Preset(
                name="Sparse Slices",
                params={
                    "slice_axis": "Z",
                    "num_slices": 15,
                    "view_mode": "Stacked",
                    "slice_spacing_mm": 5.0,
                    "scale": 15.0,
                },
                description="Few slices with wide spacing",
            ),
            Preset(
                name="Topographic Map",
                params={
                    "slice_axis": "Z",
                    "num_slices": 40,
                    "view_mode": "Plan View",
                    "slice_spacing_mm": 2.0,
                    "scale": 10.0,
                },
                description="Z-axis slices overlaid like a topographic map viewed from above",
            ),
            Preset(
                name="Side Profile",
                params={
                    "slice_axis": "Z",
                    "num_slices": 30,
                    "view_mode": "Stacked",
                    "slice_spacing_mm": 2.0,
                    "scale": 10.0,
                },
                description="Z-axis slices stacked vertically at 2 mm spacing",
            ),
            Preset(
                name="Cross Sections",
                params={
                    "slice_axis": "X",
                    "num_slices": 20,
                    "view_mode": "Stacked",
                    "slice_spacing_mm": 3.0,
                    "scale": 10.0,
                },
                description="X-axis cross sections stacked at 3 mm spacing",
            ),
        ]
