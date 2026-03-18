"""Minimal STL file loader.

Supports both binary and ASCII STL formats.
Returns a triangle mesh as (vertices, faces) arrays with deduplicated vertices.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _weld_vertices(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Deduplicate vertices within the given tolerance.

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

    # Round each coordinate to a tolerance grid and use as hash key.
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
    new_faces = remap[faces]  # remap[faces] broadcasts correctly for (M,3) input

    # Remove degenerate triangles where any two vertices collapsed together.
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[valid]

    return new_vertices, new_faces


def load_stl(
    path: str | Path,
    weld_tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Load an STL file and return (vertices, faces).

    Parameters
    ----------
    path:     Path to the .stl file.
    weld_tol: Spatial tolerance for vertex deduplication. Vertices closer than
              this distance are merged into a single vertex so that edge sharing
              can be detected correctly. Pass 0.0 to disable welding.

    Returns
    -------
    vertices: (N, 3) float64 array of deduplicated vertex positions.
    faces:    (M, 3) int32 array of triangle face indices (0-based).
    """
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(80)

    # Detect binary vs ASCII
    # Binary STL starts with an 80-byte header followed by a 4-byte triangle count
    # ASCII STL starts with "solid"
    is_ascii = header[:5].lower().startswith(b"solid")

    if is_ascii:
        vertices, faces = _load_stl_ascii(path)
    else:
        vertices, faces = _load_stl_binary(path)

    if weld_tol > 0.0 and len(vertices) > 0:
        vertices, faces = _weld_vertices(vertices, faces, tol=weld_tol)

    return vertices, faces


def _load_stl_binary(path: Path) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    with open(path, "rb") as f:
        f.read(80)  # skip header
        count_data = f.read(4)
        if len(count_data) < 4:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
        n_triangles = struct.unpack("<I", count_data)[0]

        _MAX_TRIANGLES = 10_000_000
        if n_triangles > _MAX_TRIANGLES:
            raise ValueError(
                f"STL file claims {n_triangles} triangles, which exceeds the safety limit of "
                f"{_MAX_TRIANGLES}. The file may be malformed."
            )

        raw = f.read(n_triangles * 50)  # 50 bytes per triangle: 3 floats normal + 9 floats verts + 2 bytes attr
        triangles = np.frombuffer(
            raw,
            dtype=np.dtype([
                ("normal", np.float32, (3,)),
                ("v0", np.float32, (3,)),
                ("v1", np.float32, (3,)),
                ("v2", np.float32, (3,)),
                ("attr", np.uint16),
            ])
        )
        if len(triangles) != n_triangles:
            raise ValueError(
                f"STL file header claims {n_triangles} triangles but only {len(triangles)} "
                f"could be read. The file is likely truncated or corrupt."
            )

    n = len(triangles)
    vertices = np.zeros((n * 3, 3), dtype=np.float64)
    vertices[0::3] = triangles["v0"].astype(np.float64)
    vertices[1::3] = triangles["v1"].astype(np.float64)
    vertices[2::3] = triangles["v2"].astype(np.float64)

    faces = np.arange(n * 3, dtype=np.int32).reshape(n, 3)
    return vertices, faces


_MAX_TRIANGLES = 10_000_000


def _load_stl_ascii(path: Path) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        current_verts: list[list[float]] = []
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip().lower()
            if line.startswith("vertex"):
                parts = line.split()
                try:
                    current_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"STL ASCII parse error on line {lineno}: "
                        f"expected 'vertex x y z', got {raw_line.rstrip()!r}"
                    ) from exc
            elif line.startswith("endfacet"):
                if len(current_verts) == 3:
                    base = len(vertices)
                    vertices.extend(current_verts)
                    faces.append([base, base + 1, base + 2])
                    if len(faces) >= _MAX_TRIANGLES:
                        raise ValueError(
                            f"STL ASCII file exceeds the safety limit of {_MAX_TRIANGLES} triangles. "
                            f"The file may be malformed or unexpectedly large."
                        )
                current_verts = []

    if not vertices:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

    return (
        np.array(vertices, dtype=np.float64),
        np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32),
    )
