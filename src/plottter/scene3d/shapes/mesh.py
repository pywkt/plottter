"""Mesh shape — triangle soup from OBJ/STL or vertex/face arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..path3d import Path3D
from ..bbox import BBox
from ..ray import Ray, Hit, EPSILON
from ..vector3 import Vec3
from ..triangle_bvh import TriangleBVH
from .base import Shape


class Mesh(Shape):
    """Triangle mesh loaded from OBJ/STL or provided directly.

    Parameters
    ----------
    vertices:           (N, 3) float64 array of vertex positions.
    faces:              (M, 3) int array of vertex indices (triangles).
    file_path:          Optional path to an OBJ or STL file to load.
    draw_all_edges:     If True, draw all triangle edges.
                        If False (default False), draw only "hard" edges — boundary
                        edges (shared by exactly 1 triangle) plus crease edges (shared
                        by 2 triangles whose face normals differ by more than
                        ``crease_angle_deg``).
    crease_angle_deg:   Dihedral angle threshold (degrees) for crease edge detection.
                        Edges shared by two faces whose normals differ by more than this
                        angle are considered hard/crease edges and are drawn even when
                        ``draw_all_edges=False``. Default 30°.
    backface_cull:      If True (default), skip back-facing triangles during HLR.
    simplify_edges_tol: RDP tolerance (world units) for simplifying chained edge
                        polylines before HLR. 0.0 (default) disables simplification.
    decimate:           Decimation ratio in (0, 1].  1.0 (default) = no decimation.
                        Values < 1.0 reduce the mesh to approximately
                        ``decimate × original_face_count`` faces using vertex
                        clustering before building the BVH and edge lists.
    """

    def __init__(
        self,
        vertices: NDArray[np.float64] | None = None,
        faces: NDArray[np.int32] | None = None,
        file_path: str | Path | None = None,
        draw_all_edges: bool = False,
        crease_angle_deg: float = 30.0,
        backface_cull: bool = True,
        simplify_edges_tol: float = 0.0,
        decimate: float = 1.0,
    ) -> None:
        self.draw_all_edges = draw_all_edges
        self._crease_angle_deg = crease_angle_deg
        self._simplify_edges_tol = simplify_edges_tol
        if file_path is not None:
            file_path = Path(file_path)
            if file_path.suffix.lower() == ".obj":
                from ..loaders.obj import load_obj
                self.vertices, self.faces = load_obj(file_path)
            elif file_path.suffix.lower() == ".stl":
                from ..loaders.stl import load_stl
                self.vertices, self.faces = load_stl(file_path)
            else:
                raise ValueError(f"Unsupported mesh format: {file_path.suffix}")
        else:
            if vertices is None or faces is None:
                raise ValueError("Must provide either file_path or vertices+faces")
            self.vertices = np.asarray(vertices, dtype=np.float64)
            self.faces = np.asarray(faces, dtype=np.int32)

        # Optional mesh decimation — reduce triangle count before building BVH.
        if decimate < 1.0 and len(self.faces) > 0:
            from ..decimate import decimate_mesh
            self.vertices, self.faces = decimate_mesh(self.vertices, self.faces, decimate)

        # Store original (post-decimation) face count for info/UI reporting.
        self.face_count: int = int(len(self.faces))

        # Build per-mesh BVH for O(log N) triangle intersection
        self._tri_bvh = TriangleBVH()
        if len(self.faces) > 0:
            self._tri_bvh.build(self.vertices, self.faces, backface_cull=backface_cull)

    def _face_normals(self) -> NDArray[np.float64]:
        """Compute unit normals for each triangle face."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        # Avoid division by zero for degenerate triangles
        norms = np.where(norms < 1e-12, 1.0, norms)
        return normals / norms

    def _build_edge_to_faces(self) -> dict[tuple[int, int], list[int]]:
        """Build edge → list-of-face-indices adjacency mapping."""
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for fi in range(len(self.faces)):
            face = self.faces[fi]
            for i in range(3):
                a, b = int(face[i]), int(face[(i + 1) % 3])
                key = (min(a, b), max(a, b))
                if key not in edge_to_faces:
                    edge_to_faces[key] = []
                edge_to_faces[key].append(fi)
        return edge_to_faces

    def _compute_edge_normals(self) -> dict[tuple[int, int], NDArray[np.float64]]:
        """Compute an averaged face normal for each edge.

        For boundary edges (1 adjacent face), returns that face's normal.
        For crease/interior edges (2+ adjacent faces), returns the average of
        the adjacent face normals (normalized).  Used to set ``face_normal`` on
        ``Path3D`` objects so the HLR pipeline can perform face-normal-based
        shadow classification.
        """
        if len(self.faces) == 0:
            return {}
        face_norms = self._face_normals()
        edge_to_faces = self._build_edge_to_faces()
        result: dict[tuple[int, int], NDArray[np.float64]] = {}
        for edge, face_indices in edge_to_faces.items():
            avg = face_norms[face_indices].mean(axis=0)  # (3,)
            nlen = float(np.linalg.norm(avg))
            result[edge] = avg / nlen if nlen > 1e-12 else avg
        return result

    def _edges(self) -> list[tuple[int, int]]:
        """Return edges to draw.

        If ``draw_all_edges`` is True, returns every triangle edge.
        Otherwise returns only "hard" edges:
          - Boundary edges: shared by exactly 1 triangle (always drawn).
          - Crease edges: shared by 2 triangles whose face-normal angle exceeds
            ``crease_angle_deg`` (drawn because they represent a sharp feature).
          - Non-manifold edges: shared by 3+ triangles (drawn conservatively).
        """
        edge_to_faces = self._build_edge_to_faces()

        if self.draw_all_edges:
            return list(edge_to_faces.keys())

        # Hard-edge detection
        cos_thresh = np.cos(np.radians(self._crease_angle_deg))
        normals = self._face_normals() if len(self.faces) > 0 else None

        draw_edges: list[tuple[int, int]] = []
        for edge, face_indices in edge_to_faces.items():
            if len(face_indices) == 1:
                # Boundary edge — always draw
                draw_edges.append(edge)
            elif len(face_indices) == 2 and normals is not None:
                # Crease edge if dihedral angle exceeds threshold
                n0 = normals[face_indices[0]]
                n1 = normals[face_indices[1]]
                dot = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
                if dot < cos_thresh:
                    draw_edges.append(edge)
            else:
                # Non-manifold edge (3+ faces) — draw conservatively
                draw_edges.append(edge)

        return draw_edges

    def _chain_edges(self, edges: list[tuple[int, int]]) -> list[list[int]]:
        """Chain a list of (vertex_a, vertex_b) edge pairs into polylines.

        Walks the edge graph greedily: starts at branch/endpoint vertices (degree ≠ 2),
        follows connected edges until hitting another branch or a dead-end.  Remaining
        unvisited edges form closed loops and are handled separately.

        Returns
        -------
        List of vertex-index chains.  Each chain is a list of vertex indices where
        consecutive entries share an edge.  Closed loops have the same start and end
        vertex index.
        """
        if not edges:
            return []

        # Build adjacency: vertex → sorted list of neighbor vertex indices
        adj: dict[int, list[int]] = {}
        for a, b in edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        # Track unvisited edges as (min, max) pairs
        unvisited: set[tuple[int, int]] = set()
        for a, b in edges:
            unvisited.add((min(a, b), max(a, b)))

        chains: list[list[int]] = []

        def walk(start: int, nxt: int) -> list[int]:
            """Walk a chain starting with edge start→nxt.  Removes visited edges."""
            key = (min(start, nxt), max(start, nxt))
            if key not in unvisited:
                return []
            unvisited.discard(key)
            chain = [start, nxt]
            prev, curr = start, nxt
            while True:
                # Only continue when curr is a simple pass-through vertex (degree 2)
                if len(adj[curr]) != 2:
                    break
                # Find the unvisited neighbor that is not the previous vertex
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

        # Walk chains starting from branch / endpoint vertices first (degree ≠ 2)
        branch_vertices = [v for v, nbrs in adj.items() if len(nbrs) != 2]
        for v in branch_vertices:
            for neighbor in adj[v]:
                key = (min(v, neighbor), max(v, neighbor))
                if key in unvisited:
                    chain = walk(v, neighbor)
                    if len(chain) >= 2:
                        chains.append(chain)

        # Handle remaining unvisited edges (closed loops — all vertices have degree 2)
        while unvisited:
            key = next(iter(unvisited))
            a, b = key
            chain = walk(a, b)
            if len(chain) >= 2:
                chains.append(chain)

        return chains

    def paths(self) -> list[Path3D]:
        edges = self._edges()
        chains = self._chain_edges(edges)
        edge_normals = self._compute_edge_normals()
        result: list[Path3D] = []
        for chain in chains:
            pts = [self.vertices[i] for i in chain]
            p = Path3D(pts)
            if self._simplify_edges_tol > 0.0 and len(p) > 2:
                p = p.simplify(self._simplify_edges_tol)
            # Compute chain face normal as weighted average of constituent edge normals.
            chain_normals = []
            for k in range(len(chain) - 1):
                a, b = chain[k], chain[k + 1]
                key = (min(a, b), max(a, b))
                if key in edge_normals:
                    chain_normals.append(edge_normals[key])
            if chain_normals:
                avg = np.mean(chain_normals, axis=0)
                nlen = float(np.linalg.norm(avg))
                if nlen > 1e-12:
                    p.face_normal = avg / nlen
            result.append(p)
        return result

    def intersect(self, ray: Ray) -> Hit | None:
        """Find closest triangle intersection using the per-mesh BVH."""
        hit = self._tri_bvh.intersect(ray)
        if hit is not None:
            hit.shape = self
        return hit

    def intersect_any(self, ray: Ray, t_max: float) -> bool:
        """Fast occlusion test using TriangleBVH early-exit traversal."""
        return self._tri_bvh.intersect_any(ray, t_max)

    def bbox(self) -> BBox:
        if len(self.vertices) == 0:
            from ..vector3 import vec3
            return BBox(vec3(0, 0, 0), vec3(0, 0, 0))
        return BBox(self.vertices.min(axis=0), self.vertices.max(axis=0)).pad()
