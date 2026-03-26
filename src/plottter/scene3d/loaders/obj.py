"""Minimal OBJ file loader.

Parses vertex (v) and face (f) lines. Triangulates quads.
Ignores normals, UVs, materials, and groups.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _weld_vertices(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Deduplicate vertices within the given tolerance.

    Some OBJ exporters emit one vertex per face corner (e.g. 24 vertices for a
    cube instead of 8), which breaks edge-adjacency detection.  This pass merges
    vertices that are closer than *tol* and removes any degenerate faces.

    Parameters
    ----------
    vertices: (N, 3) float64 array — may contain duplicates.
    faces:    (M, 3) int32 array of triangle face indices.
    tol:      Spatial tolerance; vertices closer than this are merged.

    Returns
    -------
    Deduplicated (vertices, faces) with remapped indices.
    Degenerate faces (where two or more vertices collapsed to the same point)
    are removed.
    """
    if len(vertices) == 0:
        return vertices, faces

    inv_tol = 1.0 / tol
    grid_coords = np.round(vertices * inv_tol).astype(np.int64)

    canonical: dict[tuple[int, int, int], int] = {}
    new_verts: list[NDArray[np.float64]] = []
    remap = np.empty(len(vertices), dtype=np.int32)

    for i in range(len(vertices)):
        key = (int(grid_coords[i, 0]), int(grid_coords[i, 1]), int(grid_coords[i, 2]))
        if key not in canonical:
            canonical[key] = len(new_verts)
            new_verts.append(vertices[i])
        remap[i] = canonical[key]

    new_vertices = np.array(new_verts, dtype=np.float64)
    new_faces = remap[faces]

    # Remove degenerate triangles where any two vertices collapsed together.
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[valid]

    return new_vertices, new_faces


def load_obj(
    path: str | Path,
    weld_tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Load an OBJ file and return (vertices, faces).

    Parameters
    ----------
    path:     Path to the .obj file.
    weld_tol: Spatial tolerance for vertex deduplication.  Vertices closer than
              this distance are merged so that edge sharing can be detected
              correctly.  Pass 0.0 to disable welding.

    Returns
    -------
    vertices: (N, 3) float64 array of deduplicated vertex positions.
    faces:    (M, 3) int32 array of triangle face indices (0-based).
    """
    _MAX_VERTICES = 10_000_000

    path = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "v":
                # Vertex: v x y z [w]
                if len(parts) < 4:
                    raise ValueError(
                        f"OBJ line {lineno}: vertex 'v' requires at least 3 coordinates, "
                        f"got {len(parts) - 1}: {line!r}"
                    )
                try:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError as exc:
                    raise ValueError(
                        f"OBJ line {lineno}: could not parse vertex coordinates: {line!r}"
                    ) from exc
                if len(vertices) >= _MAX_VERTICES:
                    raise ValueError(
                        f"OBJ file exceeds the safety limit of {_MAX_VERTICES} vertices. "
                        f"The file may be malformed or unexpectedly large."
                    )

            elif parts[0] == "f":
                # Face: f v1[/vt1[/vn1]] ...
                # Parse vertex indices (1-based in OBJ → 0-based)
                if len(parts) < 4:
                    raise ValueError(
                        f"OBJ line {lineno}: face 'f' requires at least 3 vertex indices, "
                        f"got {len(parts) - 1}: {line!r}"
                    )
                indices = []
                for p in parts[1:]:
                    # Handle v/vt/vn and v//vn formats
                    raw_idx = p.split("/")[0]
                    try:
                        idx = int(raw_idx)
                    except ValueError as exc:
                        raise ValueError(
                            f"OBJ line {lineno}: could not parse face index {raw_idx!r}: {line!r}"
                        ) from exc
                    # Support negative indices (relative to end of vertex list)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx -= 1  # convert to 0-based
                    if idx < 0 or idx >= len(vertices):
                        raise ValueError(
                            f"OBJ line {lineno}: face index {p!r} resolves to {idx} which is "
                            f"out of range (0..{len(vertices) - 1}): {line!r}"
                        )
                    indices.append(idx)

                # Triangulate polygon (fan triangulation from first vertex)
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])

    if not vertices:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

    verts_array = np.array(vertices, dtype=np.float64)
    faces_array = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)

    if weld_tol > 0.0 and len(verts_array) > 0:
        verts_array, faces_array = _weld_vertices(verts_array, faces_array, tol=weld_tol)

    return verts_array, faces_array
