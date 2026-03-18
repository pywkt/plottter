"""Mesh decimation via vertex clustering.

Provides ``decimate_mesh(vertices, faces, target_ratio)`` — a fast, dependency-free
mesh simplification function suitable for reducing detail before plotter rendering.

Algorithm: vertex clustering (spatial binning)
    1. Divide the mesh bounding box into a uniform 3-D grid.
    2. All vertices that fall in the same grid cell are merged into their centroid.
    3. Faces are rebuilt by mapping old vertex indices to cluster representatives.
    4. Degenerate faces (two or more vertices in the same cluster) are discarded.

This is not as accurate as Quadric Error Metrics (QEM) but it is O(N) and produces
good results for plotter-style line art where smooth curves matter more than sharp
feature preservation.  For a plotter output, the user only needs enough triangles to
represent major silhouettes — fine surface detail is invisible when drawn with a pen.

Complexity: O(N + M) where N = number of vertices, M = number of faces.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Minimum face count — never decimate below this (prevents degenerate output)
# ---------------------------------------------------------------------------
_MIN_FACES = 100


def decimate_mesh(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    target_ratio: float,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Decimate a triangle mesh by vertex clustering.

    Parameters
    ----------
    vertices:
        ``(N, 3)`` float64 array of vertex positions.
    faces:
        ``(M, 3)`` int32 array of triangle vertex indices.
    target_ratio:
        Desired ``output_faces / input_faces`` ratio in the range ``(0, 1]``.
        Values ≥ 1.0 return the original arrays unchanged.
        Values that would produce fewer than :data:`_MIN_FACES` are clamped so
        the output has at least that many faces (if the input does).

    Returns
    -------
    new_vertices:
        ``(N', 3)`` float64 array — cluster centroids.
    new_faces:
        ``(M', 3)`` int32 array — non-degenerate faces remapped to cluster indices.

    Notes
    -----
    The returned arrays are always **new** NumPy arrays; the inputs are never
    modified in-place.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    n_verts = len(vertices)
    n_faces = len(faces)

    # Nothing to do.
    if target_ratio >= 1.0 or n_faces == 0 or n_verts == 0:
        return vertices, faces

    # Target face count — always keep at least _MIN_FACES.
    target_faces = max(_MIN_FACES, int(n_faces * max(0.0, target_ratio)))
    if target_faces >= n_faces:
        return vertices, faces

    # ------------------------------------------------------------------ grid
    # Choose grid resolution so the expected number of cluster centroids is
    # approximately ``n_verts * (target_faces / n_faces)``.
    #
    # With a uniform G×G×G grid and N vertices distributed uniformly we get
    # ~G³ non-empty cells, so:   G³ ≈ n_verts · reduction  →  G ≈ ∛(n_verts·r)
    reduction = target_faces / n_faces
    grid_size = max(2, int(round((n_verts * reduction) ** (1.0 / 3.0))))

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = vmax - vmin

    # For flat/degenerate meshes a dimension may be zero — avoid division by zero.
    extent = np.where(extent < 1e-12, 1.0, extent)

    # Map every vertex to a (gx, gy, gz) cell index in [0, grid_size).
    grid_coords = ((vertices - vmin) / extent * grid_size).clip(0, grid_size - 1).astype(np.int32)

    # Flatten 3-D cell index to a single integer so np.unique can sort it.
    g2 = grid_size * grid_size
    cell_ids: NDArray[np.int64] = (
        grid_coords[:, 0].astype(np.int64) * g2
        + grid_coords[:, 1].astype(np.int64) * grid_size
        + grid_coords[:, 2].astype(np.int64)
    )

    # unique_cells: sorted unique cell IDs
    # inverse:      for each vertex, the index into unique_cells (= cluster id)
    _unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
    n_clusters = len(_unique_cells)

    # ---------------------------------------------------------------- centroids
    # Compute centroid of all vertices assigned to each cluster.
    new_vertices = np.zeros((n_clusters, 3), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.float64)

    np.add.at(new_vertices, inverse, vertices)
    np.add.at(counts, inverse, 1.0)
    new_vertices /= counts[:, np.newaxis]

    # ------------------------------------------------------------------- faces
    # Remap face vertex indices from original vertices → cluster indices.
    new_face_indices = inverse[faces]  # (M, 3) cluster indices

    # Discard degenerate triangles — two or more corners in the same cluster.
    v0, v1, v2 = new_face_indices[:, 0], new_face_indices[:, 1], new_face_indices[:, 2]
    valid_mask = (v0 != v1) & (v1 != v2) & (v0 != v2)
    new_faces = new_face_indices[valid_mask].astype(np.int32)

    return new_vertices, new_faces
