"""Tests for Penrose P2 tiling generator (Tasks 46.1 and 46.2)."""

from __future__ import annotations

import cmath
import math

import pytest

from plottter.generators.penrose import (
    PHI,
    PSI,
    TYPE_THICK,
    TYPE_THIN,
    _clip_to_canvas,
    _edge_key,
    _initial_config,
    _subdivide,
    _triangles_to_edges,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_phi_value(self):
        assert abs(PHI - (1 + math.sqrt(5)) / 2) < 1e-10

    def test_psi_value(self):
        assert abs(PSI - 1 / PHI) < 1e-10

    def test_phi_psi_product(self):
        assert abs(PHI * PSI - 1.0) < 1e-10

    def test_phi_squared(self):
        # PHI² = PHI + 1 (golden ratio identity)
        assert abs(PHI ** 2 - (PHI + 1)) < 1e-10


# ---------------------------------------------------------------------------
# Thick (Type 1) subdivision
# ---------------------------------------------------------------------------

class TestSubdivideThick:
    """Type 1 (thick) triangle subdivides into exactly 3 sub-triangles."""

    def _make_thick(self) -> tuple:
        # A simple thick triangle with vertices at convenient complex coords
        A = 0 + 0j
        B = 1 + 0j
        C = cmath.exp(1j * 2 * math.pi / 5)  # 72° from positive real axis
        return (TYPE_THICK, A, B, C)

    def test_thick_produces_three_subtriangles(self):
        result = _subdivide([self._make_thick()])
        assert len(result) == 3

    def test_thick_subtriangle_types(self):
        result = _subdivide([self._make_thick()])
        types = [t for t, *_ in result]
        assert types.count(TYPE_THICK) == 2
        assert types.count(TYPE_THIN) == 1

    def test_thick_all_fields_are_tuples_of_four(self):
        result = _subdivide([self._make_thick()])
        for tri in result:
            assert len(tri) == 4

    def test_thick_type_tags_are_integers(self):
        result = _subdivide([self._make_thick()])
        for tri in result:
            assert isinstance(tri[0], int)


# ---------------------------------------------------------------------------
# Thin (Type 0) subdivision
# ---------------------------------------------------------------------------

class TestSubdivideThin:
    """Type 0 (thin) triangle subdivides into exactly 2 sub-triangles."""

    def _make_thin(self) -> tuple:
        A = 0 + 0j
        B = 1 + 0j
        C = cmath.exp(1j * math.pi / 5)  # 36° from positive real axis
        return (TYPE_THIN, A, B, C)

    def test_thin_produces_two_subtriangles(self):
        result = _subdivide([self._make_thin()])
        assert len(result) == 2

    def test_thin_subtriangle_types(self):
        result = _subdivide([self._make_thin()])
        types = [t for t, *_ in result]
        assert TYPE_THIN in types
        assert TYPE_THICK in types

    def test_thin_produces_one_thin_one_thick(self):
        result = _subdivide([self._make_thin()])
        types = [t for t, *_ in result]
        assert types.count(TYPE_THIN) == 1
        assert types.count(TYPE_THICK) == 1


# ---------------------------------------------------------------------------
# Mixed input
# ---------------------------------------------------------------------------

class TestSubdivideMixed:
    def test_empty_input(self):
        assert _subdivide([]) == []

    def test_multiple_triangles(self):
        thick = (TYPE_THICK, 0+0j, 1+0j, cmath.exp(1j * 2 * math.pi / 5))
        thin = (TYPE_THIN, 0+0j, 1+0j, cmath.exp(1j * math.pi / 5))
        result = _subdivide([thick, thin])
        assert len(result) == 5  # 3 from thick + 2 from thin


# ---------------------------------------------------------------------------
# Initial configurations
# ---------------------------------------------------------------------------

class TestInitialConfig:
    def test_sun_produces_ten_triangles(self):
        tris = _initial_config("Sun", 1.0)
        assert len(tris) == 10

    def test_sun_all_thick(self):
        tris = _initial_config("Sun", 1.0)
        assert all(t == TYPE_THICK for t, *_ in tris)

    def test_star_produces_ten_triangles(self):
        tris = _initial_config("Star", 1.0)
        assert len(tris) == 10

    def test_star_all_thin(self):
        tris = _initial_config("Star", 1.0)
        assert all(t == TYPE_THIN for t, *_ in tris)

    def test_dart_produces_four_triangles(self):
        tris = _initial_config("Dart", 1.0)
        assert len(tris) == 4

    def test_dart_all_thin(self):
        tris = _initial_config("Dart", 1.0)
        assert all(t == TYPE_THIN for t, *_ in tris)

    def test_unknown_config_returns_empty(self):
        tris = _initial_config("Unknown", 1.0)
        assert tris == []

    def test_radius_scales_vertices(self):
        tris_r1 = _initial_config("Sun", 1.0)
        tris_r2 = _initial_config("Sun", 2.0)
        # Non-origin vertices should be scaled by 2×
        for (_, A1, B1, C1), (_, A2, B2, C2) in zip(tris_r1, tris_r2):
            assert abs(B2 - 2 * B1) < 1e-10
            assert abs(C2 - 2 * C1) < 1e-10


# ---------------------------------------------------------------------------
# Vertex validity
# ---------------------------------------------------------------------------

class TestVertexValidity:
    """All vertices must be finite complex numbers."""

    def _check_all_finite(self, tris: list) -> None:
        for tri in tris:
            t, A, B, C = tri
            for v in (A, B, C):
                assert cmath.isfinite(v), f"Non-finite vertex {v} in triangle {tri}"

    def test_sun_initial_vertices_finite(self):
        self._check_all_finite(_initial_config("Sun", 1.0))

    def test_star_initial_vertices_finite(self):
        self._check_all_finite(_initial_config("Star", 1.0))

    def test_dart_initial_vertices_finite(self):
        self._check_all_finite(_initial_config("Dart", 1.0))

    def test_sun_after_five_subdivisions_finite(self):
        tris = _initial_config("Sun", 1.0)
        for _ in range(5):
            tris = _subdivide(tris)
        self._check_all_finite(tris)

    def test_star_after_five_subdivisions_finite(self):
        tris = _initial_config("Star", 1.0)
        for _ in range(5):
            tris = _subdivide(tris)
        self._check_all_finite(tris)

    def test_type_tags_valid(self):
        tris = _initial_config("Sun", 1.0)
        for _ in range(4):
            tris = _subdivide(tris)
        for tri in tris:
            assert tri[0] in (TYPE_THIN, TYPE_THICK)


# ---------------------------------------------------------------------------
# Growth rate ≈ PHI²
# ---------------------------------------------------------------------------

class TestCountGrowth:
    """Triangle count grows by approximately PHI² ≈ 2.618 per subdivision."""

    def test_sun_growth_converges_to_phi_squared(self):
        tris = _initial_config("Sun", 1.0)
        counts = [len(tris)]
        for _ in range(6):
            tris = _subdivide(tris)
            counts.append(len(tris))

        # From level 2 onward growth ratio converges tightly to PHI²
        for i in range(2, len(counts)):
            ratio = counts[i] / counts[i - 1]
            assert abs(ratio - PHI ** 2) < 0.1, (
                f"Growth ratio {ratio:.4f} at step {i} is not close to "
                f"PHI² = {PHI**2:.4f}"
            )

    def test_sun_level1_ratio_is_three(self):
        # With all-thick initial config, first subdivision triples count (3 per thick)
        tris = _initial_config("Sun", 1.0)
        n0 = len(tris)
        tris = _subdivide(tris)
        n1 = len(tris)
        assert n1 == n0 * 3

    def test_all_configs_grow(self):
        for config in ("Sun", "Star", "Dart"):
            tris = _initial_config(config, 1.0)
            n0 = len(tris)
            tris = _subdivide(tris)
            n1 = len(tris)
            assert n1 > n0, f"{config}: count did not grow after subdivision"

    def test_exact_counts_sun(self):
        # Sun starts at 10 thick triangles; count sequence is deterministic
        # Level 0: 10, Level 1: 30, Level 2: 80, Level 3: 210
        tris = _initial_config("Sun", 1.0)
        assert len(tris) == 10
        tris = _subdivide(tris)
        assert len(tris) == 30
        tris = _subdivide(tris)
        assert len(tris) == 80
        tris = _subdivide(tris)
        assert len(tris) == 210


# ---------------------------------------------------------------------------
# Generator registration and integration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_registered(self):
        from plottter.generators import GENERATORS
        assert "Penrose Tiling" in GENERATORS

    def test_category_is_math(self):
        from plottter.generators.penrose import PenroseGenerator
        assert PenroseGenerator.category == "math"

    def test_has_required_parameters(self):
        from plottter.generators.penrose import PenroseGenerator
        gen = PenroseGenerator()
        names = {p.name for p in gen.get_parameters()}
        assert "initial_config" in names
        assert "depth" in names
        assert "radius_mm" in names
        assert "rotation_deg" in names
        assert "draw_mode" in names
        assert "deduplicate" in names

    def test_has_presets(self):
        from plottter.generators.penrose import PenroseGenerator
        presets = PenroseGenerator().get_presets()
        assert len(presets) >= 1

    def test_generate_returns_nonempty_list(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        result = gen.generate(
            {
                "initial_config": "Sun",
                "depth": 2,
                "radius_mm": 80.0,
                "rotation_deg": 0.0,
                "draw_mode": "All edges",
                "deduplicate": True,
            },
            canvas,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_polylines_have_two_points(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        result = gen.generate(
            {
                "initial_config": "Sun",
                "depth": 2,
                "radius_mm": 80.0,
                "rotation_deg": 0.0,
                "draw_mode": "All edges",
                "deduplicate": True,
            },
            canvas,
        )
        for poly in result:
            assert len(poly) == 2

    def test_generate_star_config(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        result = gen.generate(
            {
                "initial_config": "Star",
                "depth": 3,
                "radius_mm": 80.0,
                "rotation_deg": 45.0,
                "draw_mode": "Thin only",
                "deduplicate": False,
            },
            canvas,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_dart_config(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        result = gen.generate(
            {
                "initial_config": "Dart",
                "depth": 3,
                "radius_mm": 80.0,
                "rotation_deg": 0.0,
                "draw_mode": "All edges",
                "deduplicate": True,
            },
            canvas,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_deduplicate_reduces_edge_count(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        base_params = {
            "initial_config": "Sun",
            "depth": 3,
            "radius_mm": 80.0,
            "rotation_deg": 0.0,
            "draw_mode": "All edges",
        }
        with_dedup = gen.generate({**base_params, "deduplicate": True}, canvas)
        without_dedup = gen.generate({**base_params, "deduplicate": False}, canvas)
        assert len(with_dedup) < len(without_dedup)

    def test_depth_zero_returns_initial_edges(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        result = gen.generate(
            {
                "initial_config": "Sun",
                "depth": 0,
                "radius_mm": 80.0,
                "rotation_deg": 0.0,
                "draw_mode": "All edges",
                "deduplicate": False,
            },
            canvas,
        )
        # 10 triangles × 3 edges each = 30 edges (no dedup)
        assert len(result) == 30


# ---------------------------------------------------------------------------
# Task 46.2 — _triangles_to_edges: rhomb grouping and deduplication
# ---------------------------------------------------------------------------

def _count_rhombs(triangles) -> int:
    """Count rhombs by finding same-type triangle pairs sharing a long edge."""
    from plottter.generators.penrose import _long_edge_keys

    long_edge_map: dict = {}
    for i, (t, A, B, C) in enumerate(triangles):
        for key in _long_edge_keys(t, A, B, C):
            long_edge_map.setdefault(key, []).append(i)

    count = 0
    for indices in long_edge_map.values():
        if len(indices) == 2:
            i, j = indices
            if triangles[i][0] == triangles[j][0]:
                count += 1
    return count


class TestTrianglesToEdges:
    """Tests for _triangles_to_edges (task 46.2-A)."""

    def test_returns_list(self):
        tris = _initial_config("Sun", 1.0)
        edges = _triangles_to_edges(tris)
        assert isinstance(edges, list)

    def test_nonempty_for_sun_depth0(self):
        tris = _initial_config("Sun", 1.0)
        edges = _triangles_to_edges(tris)
        assert len(edges) > 0

    # (a) Edge count consistent with rhomb count: <= 4 * n_rhombs
    def test_edge_count_consistent_with_rhomb_count(self):
        tris = _initial_config("Sun", 1.0)
        for _ in range(3):
            tris = _subdivide(tris)
        edges = _triangles_to_edges(tris)
        n_rhombs = _count_rhombs(tris)
        # Each rhomb contributes at most 4 edges; deduplication can only reduce this
        assert len(edges) <= 4 * n_rhombs
        # At least 1 edge per rhomb (some edges must appear)
        assert len(edges) >= n_rhombs

    def test_edge_count_less_than_4_per_rhomb_due_to_sharing(self):
        """Deduplication across adjacent rhombs reduces the total."""
        tris = _initial_config("Sun", 1.0)
        for _ in range(3):
            tris = _subdivide(tris)
        edges = _triangles_to_edges(tris)
        n_rhombs = _count_rhombs(tris)
        # For a non-trivial tiling many rhombs share outer edges → strictly less
        assert len(edges) < 4 * n_rhombs

    # (b) Deduplication removes exactly the shared edges — no duplicate in output
    def test_no_duplicate_edges(self):
        tris = _initial_config("Sun", 1.0)
        for _ in range(3):
            tris = _subdivide(tris)
        edges = _triangles_to_edges(tris)
        keys = [_edge_key(v1, v2) for v1, v2 in edges]
        assert len(keys) == len(set(keys)), "Duplicate edges in output"

    def test_no_duplicate_edges_star_depth4(self):
        tris = _initial_config("Star", 1.0)
        for _ in range(4):
            tris = _subdivide(tris)
        edges = _triangles_to_edges(tris)
        keys = [_edge_key(v1, v2) for v1, v2 in edges]
        assert len(keys) == len(set(keys)), "Duplicate edges in Star output"

    def test_internal_diagonals_not_in_output(self):
        """Shared long edges (rhomb diagonals) must not appear in the output."""
        from plottter.generators.penrose import _long_edge_keys

        tris = _initial_config("Sun", 1.0)
        for _ in range(2):
            tris = _subdivide(tris)
        edges = _triangles_to_edges(tris)
        output_keys = {_edge_key(v1, v2) for v1, v2 in edges}

        # Build shared long edge keys
        long_edge_map: dict = {}
        for i, (t, A, B, C) in enumerate(tris):
            for key in _long_edge_keys(t, A, B, C):
                long_edge_map.setdefault(key, []).append(i)

        for key, indices in long_edge_map.items():
            if len(indices) == 2:
                i, j = indices
                if tris[i][0] == tris[j][0]:
                    assert key not in output_keys, (
                        "Shared long edge (rhomb diagonal) appears in output"
                    )

    # (d) No zero-length edges
    def test_no_zero_length_edges(self):
        tris = _initial_config("Sun", 1.0)
        for _ in range(3):
            tris = _subdivide(tris)
        edges = _triangles_to_edges(tris)
        for v1, v2 in edges:
            assert abs(v1 - v2) > 1e-10, f"Zero-length edge: {v1} → {v2}"

    def test_no_zero_length_edges_dart(self):
        tris = _initial_config("Dart", 1.0)
        for _ in range(4):
            tris = _subdivide(tris)
        edges = _triangles_to_edges(tris)
        for v1, v2 in edges:
            assert abs(v1 - v2) > 1e-10, f"Zero-length edge: {v1} → {v2}"

    def test_each_edge_is_complex_pair(self):
        tris = _initial_config("Sun", 1.0)
        tris = _subdivide(tris)
        edges = _triangles_to_edges(tris)
        for item in edges:
            v1, v2 = item
            assert isinstance(v1, complex)
            assert isinstance(v2, complex)


# ---------------------------------------------------------------------------
# Task 46.2 — _clip_to_canvas
# ---------------------------------------------------------------------------

class TestClipToCanvas:
    """Tests for _clip_to_canvas (task 46.2-B/C)."""

    # Canvas: 200×150 mm, margin 10 mm → drawing area [10, 10, 190, 140]

    def _canvas_params(self):
        return dict(canvas_w=200.0, canvas_h=150.0, margin=10.0)

    # (c) All output edges within canvas bounds
    def test_edge_fully_inside_passes_through(self):
        edges = [((20.0, 20.0), (80.0, 60.0))]
        clipped = _clip_to_canvas(edges, **self._canvas_params())
        assert len(clipped) == 1
        (x0, y0), (x1, y1) = clipped[0]
        assert x0 == pytest.approx(20.0)
        assert y0 == pytest.approx(20.0)
        assert x1 == pytest.approx(80.0)
        assert y1 == pytest.approx(60.0)

    def test_edge_fully_outside_discarded(self):
        # Edge in the margin zone (x < 10)
        edges = [((1.0, 20.0), (5.0, 50.0))]
        clipped = _clip_to_canvas(edges, **self._canvas_params())
        assert len(clipped) == 0

    def test_edge_above_canvas_discarded(self):
        edges = [((50.0, 1.0), (100.0, 5.0))]
        clipped = _clip_to_canvas(edges, **self._canvas_params())
        assert len(clipped) == 0

    def test_horizontal_crossing_edge_clipped(self):
        # Horizontal line from x=0 to x=200 at y=75 (centre)
        edges = [((0.0, 75.0), (200.0, 75.0))]
        clipped = _clip_to_canvas(edges, **self._canvas_params())
        assert len(clipped) == 1
        (x0, y0), (x1, y1) = clipped[0]
        assert x0 == pytest.approx(10.0)
        assert x1 == pytest.approx(190.0)
        assert y0 == pytest.approx(75.0)
        assert y1 == pytest.approx(75.0)

    def test_all_output_within_bounds(self):
        """All clipped edge endpoints must lie within the drawing area."""
        tris = _initial_config("Sun", 1.0)
        for _ in range(3):
            tris = _subdivide(tris)
        # Transform to mm (centre 100, 75; scale 80)
        cx, cy, scale = 100.0, 75.0, 80.0
        mm_edges = [
            ((cx + v1.real * scale, cy - v1.imag * scale),
             (cx + v2.real * scale, cy - v2.imag * scale))
            for v1, v2 in _triangles_to_edges(tris)
        ]
        clipped = _clip_to_canvas(mm_edges, 200.0, 150.0, 10.0)
        xmin, ymin, xmax, ymax = 10.0, 10.0, 190.0, 140.0
        for (x0, y0), (x1, y1) in clipped:
            assert xmin - 1e-6 <= x0 <= xmax + 1e-6
            assert ymin - 1e-6 <= y0 <= ymax + 1e-6
            assert xmin - 1e-6 <= x1 <= xmax + 1e-6
            assert ymin - 1e-6 <= y1 <= ymax + 1e-6

    def test_no_zero_length_after_clipping(self):
        """Clipped output must contain no zero-length segments."""
        tris = _initial_config("Sun", 1.0)
        for _ in range(3):
            tris = _subdivide(tris)
        cx, cy, scale = 100.0, 75.0, 80.0
        mm_edges = [
            ((cx + v1.real * scale, cy - v1.imag * scale),
             (cx + v2.real * scale, cy - v2.imag * scale))
            for v1, v2 in _triangles_to_edges(tris)
        ]
        clipped = _clip_to_canvas(mm_edges, 200.0, 150.0, 10.0)
        for (x0, y0), (x1, y1) in clipped:
            assert (x1 - x0) ** 2 + (y1 - y0) ** 2 > 1e-20, (
                f"Zero-length segment at ({x0},{y0})"
            )

    def test_empty_input_returns_empty(self):
        assert _clip_to_canvas([], 200.0, 150.0, 10.0) == []

    def test_returns_polylines(self):
        edges = [((20.0, 20.0), (80.0, 60.0))]
        clipped = _clip_to_canvas(edges, **self._canvas_params())
        assert len(clipped) == 1
        assert len(clipped[0]) == 2  # 2-point polyline


# ---------------------------------------------------------------------------
# Task 46.2 — Rhombs draw mode integration
# ---------------------------------------------------------------------------

class TestRhombsDrawMode:
    """Integration tests for the 'Rhombs' draw mode in PenroseGenerator."""

    def _gen_rhombs(self, config="Sun", depth=3, radius_mm=80.0, rotation_deg=0.0):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        return gen.generate(
            {
                "initial_config": config,
                "depth": depth,
                "radius_mm": radius_mm,
                "rotation_deg": rotation_deg,
                "draw_mode": "Rhombs",
                "deduplicate": True,
            },
            canvas,
        )

    def test_returns_nonempty_list(self):
        result = self._gen_rhombs()
        assert isinstance(result, list)
        assert len(result) > 0

    # (c) All output edges within canvas bounds
    def test_all_edges_within_canvas(self):
        from plottter.models.canvas import Canvas
        canvas = Canvas.from_preset("A4", margin=10.0)
        x1, y1, x2, y2 = canvas.drawing_area()
        result = self._gen_rhombs()
        for poly in result:
            for x, y in poly:
                assert x1 - 1e-6 <= x <= x2 + 1e-6, f"x={x} out of [{x1},{x2}]"
                assert y1 - 1e-6 <= y <= y2 + 1e-6, f"y={y} out of [{y1},{y2}]"

    # (d) No zero-length edges
    def test_no_zero_length_polylines(self):
        result = self._gen_rhombs()
        for poly in result:
            (x0, y0), (x1, y1) = poly
            dist_sq = (x1 - x0) ** 2 + (y1 - y0) ** 2
            assert dist_sq > 1e-20, f"Zero-length polyline at ({x0},{y0})"

    def test_each_polyline_has_two_points(self):
        result = self._gen_rhombs()
        for poly in result:
            assert len(poly) == 2

    def test_rhombs_star_config(self):
        result = self._gen_rhombs(config="Star", depth=3)
        assert len(result) > 0

    def test_rhombs_with_rotation(self):
        r0 = self._gen_rhombs(rotation_deg=0.0)
        r45 = self._gen_rhombs(rotation_deg=45.0)
        # Different rotations produce different edge sets
        assert len(r0) == len(r45)  # same count
        # But coordinates differ
        coords0 = {(round(p[0], 3), round(p[1], 3)) for poly in r0 for p in poly}
        coords45 = {(round(p[0], 3), round(p[1], 3)) for poly in r45 for p in poly}
        assert coords0 != coords45
