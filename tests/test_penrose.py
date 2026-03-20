"""Tests for Penrose P2 tiling generator (Tasks 46.1, 46.2, and 46.3)."""

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
# Generator registration and integration (Task 46.3)
# ---------------------------------------------------------------------------

def _make_params(**overrides) -> dict:
    """Return default Task 46.3 parameters, optionally overridden."""
    defaults = {
        "initial_config": "Sun",
        "subdivisions": 2,
        "rotation_deg": 0.0,
        "render_mode": "Edges Only",
        "x_offset_mm": 0.0,
        "y_offset_mm": 0.0,
    }
    defaults.update(overrides)
    return defaults


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
        assert "subdivisions" in names
        assert "rotation_deg" in names
        assert "render_mode" in names
        assert "x_offset_mm" in names
        assert "y_offset_mm" in names

    def test_subdivisions_param_range(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.generators.base import IntParam
        gen = PenroseGenerator()
        sub_param = next(p for p in gen.get_parameters() if p.name == "subdivisions")
        assert isinstance(sub_param, IntParam)
        assert sub_param.min == 1
        assert sub_param.max == 8
        assert sub_param.default == 5

    def test_render_mode_choices(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.generators.base import ChoiceParam
        gen = PenroseGenerator()
        rm_param = next(p for p in gen.get_parameters() if p.name == "render_mode")
        assert isinstance(rm_param, ChoiceParam)
        assert "Edges Only" in rm_param.choices
        assert "Edges + Arcs" in rm_param.choices
        assert "Arcs Only" in rm_param.choices
        assert rm_param.default == "Edges Only"

    def test_rotation_param_range(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.generators.base import FloatParam
        gen = PenroseGenerator()
        rot_param = next(p for p in gen.get_parameters() if p.name == "rotation_deg")
        assert isinstance(rot_param, FloatParam)
        assert rot_param.min == 0.0
        assert rot_param.max == 360.0

    def test_has_presets(self):
        from plottter.generators.penrose import PenroseGenerator
        presets = PenroseGenerator().get_presets()
        assert len(presets) >= 1

    def test_generate_returns_nonempty_list(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        result = gen.generate(_make_params(), canvas)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_edges_only_polylines_have_two_points(self):
        """'Edges Only' mode produces 2-point straight-edge polylines."""
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        result = gen.generate(_make_params(render_mode="Edges Only"), canvas)
        for poly in result:
            assert len(poly) == 2

    def test_generate_star_config(self):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        result = gen.generate(
            _make_params(initial_config="Star", subdivisions=3, rotation_deg=45.0),
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
            _make_params(initial_config="Dart", subdivisions=3),
            canvas,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    # (b) Higher subdivisions produce more tiles
    def test_higher_subdivisions_produce_more_polylines(self):
        """Higher subdivision depth produces more rhombs → more edge polylines."""
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        result2 = gen.generate(_make_params(subdivisions=2), canvas)
        result3 = gen.generate(_make_params(subdivisions=3), canvas)
        assert len(result3) > len(result2)

    # (c) "Arcs Only" produces curved (multi-point) polylines
    def test_arcs_only_produces_curved_polylines(self):
        """'Arcs Only' mode must yield polylines with more than 2 points."""
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        result = gen.generate(_make_params(render_mode="Arcs Only"), canvas)
        assert len(result) > 0
        assert any(len(poly) > 2 for poly in result), (
            "Expected at least one arc polyline with >2 points"
        )

    def test_edges_plus_arcs_has_more_polylines_than_edges_only(self):
        """'Edges + Arcs' must include at least as many polylines as 'Edges Only'."""
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        edges_only = gen.generate(_make_params(render_mode="Edges Only"), canvas)
        edges_arcs = gen.generate(_make_params(render_mode="Edges + Arcs"), canvas)
        assert len(edges_arcs) > len(edges_only)

    # (d) Rotation rotates the pattern
    def test_rotation_changes_coordinates(self):
        """Different rotation_deg values must produce different coordinates."""
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        r0  = gen.generate(_make_params(rotation_deg=0.0),  canvas)
        r45 = gen.generate(_make_params(rotation_deg=45.0), canvas)
        coords0  = {(round(p[0], 3), round(p[1], 3)) for poly in r0  for p in poly}
        coords45 = {(round(p[0], 3), round(p[1], 3)) for poly in r45 for p in poly}
        assert coords0 != coords45

    # (a) Parameter changes produce visibly different output
    def test_different_initial_configs_produce_different_output(self):
        """'Sun' and 'Star' initial configs must yield different edge sets."""
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        sun  = gen.generate(_make_params(initial_config="Sun"),  canvas)
        star = gen.generate(_make_params(initial_config="Star"), canvas)
        coords_sun  = {(round(p[0], 3), round(p[1], 3)) for poly in sun  for p in poly}
        coords_star = {(round(p[0], 3), round(p[1], 3)) for poly in star for p in poly}
        assert coords_sun != coords_star

    def test_offset_shifts_center(self):
        """x_offset_mm and y_offset_mm shift the tiling centre."""
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        r_center = gen.generate(_make_params(), canvas)
        r_offset = gen.generate(_make_params(x_offset_mm=20.0, y_offset_mm=10.0), canvas)
        coords_c = {(round(p[0], 3), round(p[1], 3)) for poly in r_center for p in poly}
        coords_o = {(round(p[0], 3), round(p[1], 3)) for poly in r_offset for p in poly}
        assert coords_c != coords_o


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
# Task 46.3 — Arc decoration helpers
# ---------------------------------------------------------------------------

class TestArcComplex:
    """Tests for the _arc_complex helper."""

    def test_arc_returns_correct_number_of_points(self):
        from plottter.generators.penrose import _arc_complex
        center = 0 + 0j
        p1 = 1 + 0j
        p2 = 0 + 1j
        inside = 0.5 + 0.5j
        pts = _arc_complex(center, p1, p2, inside, n_segments=8)
        assert len(pts) == 9  # n_segments + 1

    def test_arc_starts_and_ends_at_p1_p2(self):
        from plottter.generators.penrose import _arc_complex
        center = 0 + 0j
        p1 = 1 + 0j
        p2 = 0 + 1j
        inside = 0.5 + 0.5j
        pts = _arc_complex(center, p1, p2, inside)
        assert abs(pts[0] - p1) < 1e-9
        assert abs(pts[-1] - p2) < 1e-9

    def test_arc_points_on_circle(self):
        from plottter.generators.penrose import _arc_complex
        center = 1 + 2j
        r = 3.0
        p1 = center + r
        p2 = center + r * cmath.exp(1j * math.pi / 2)
        inside = center + r * cmath.exp(1j * math.pi / 4)
        pts = _arc_complex(center, p1, p2, inside)
        for z in pts:
            assert abs(abs(z - center) - r) < 1e-6

    def test_arc_midpoint_near_inside_ref(self):
        from plottter.generators.penrose import _arc_complex
        center = 0 + 0j
        r = 1.0
        # Arc from 0° to 90°, inside ref at 45° → CCW arc
        p1 = r * cmath.exp(1j * 0)
        p2 = r * cmath.exp(1j * math.pi / 2)
        inside_ccw = r * cmath.exp(1j * math.pi / 4)  # 45° = midpoint of CCW arc
        pts = _arc_complex(center, p1, p2, inside_ccw)
        mid = pts[len(pts) // 2]
        # Midpoint should be near 45°
        angle = cmath.phase(mid - center) % (2 * math.pi)
        expected = math.pi / 4
        assert abs(angle - expected) < 0.2

    def test_arc_degenerate_center_returns_empty(self):
        from plottter.generators.penrose import _arc_complex
        center = 1 + 0j
        p1 = center  # zero distance → degenerate
        p2 = 2 + 0j
        inside = 1.5 + 0.5j
        pts = _arc_complex(center, p1, p2, inside)
        assert pts == []


class TestGenerateRhombArcs:
    """Tests for _generate_rhomb_arcs."""

    def test_returns_list(self):
        from plottter.generators.penrose import _generate_rhomb_arcs
        tris = _initial_config("Sun", 1.0)
        for _ in range(2):
            tris = _subdivide(tris)
        arcs = _generate_rhomb_arcs(tris)
        assert isinstance(arcs, list)

    def test_nonempty_for_sun_depth2(self):
        from plottter.generators.penrose import _generate_rhomb_arcs
        tris = _initial_config("Sun", 1.0)
        for _ in range(2):
            tris = _subdivide(tris)
        arcs = _generate_rhomb_arcs(tris)
        assert len(arcs) > 0

    def test_each_arc_has_correct_point_count(self):
        from plottter.generators.penrose import _generate_rhomb_arcs
        tris = _initial_config("Sun", 1.0)
        for _ in range(2):
            tris = _subdivide(tris)
        arcs = _generate_rhomb_arcs(tris, n_arc_segments=8)
        for arc in arcs:
            assert len(arc) == 9  # n_segments + 1

    def test_arc_count_grows_with_subdivisions(self):
        from plottter.generators.penrose import _generate_rhomb_arcs
        tris2 = _initial_config("Sun", 1.0)
        for _ in range(2):
            tris2 = _subdivide(tris2)
        tris3 = _initial_config("Sun", 1.0)
        for _ in range(3):
            tris3 = _subdivide(tris3)
        assert len(_generate_rhomb_arcs(tris3)) > len(_generate_rhomb_arcs(tris2))

    def test_two_arcs_per_rhomb(self):
        """Each rhomb produces exactly two arcs."""
        from plottter.generators.penrose import _generate_rhomb_arcs, _long_edge_keys
        tris = _initial_config("Sun", 1.0)
        for _ in range(2):
            tris = _subdivide(tris)

        # Count rhombs directly
        long_edge_map: dict = {}
        for i, (t, A, B, C) in enumerate(tris):
            for key in _long_edge_keys(t, A, B, C):
                long_edge_map.setdefault(key, []).append(i)
        n_rhombs = sum(
            1 for idxs in long_edge_map.values()
            if len(idxs) == 2 and tris[idxs[0]][0] == tris[idxs[1]][0]
        )

        arcs = _generate_rhomb_arcs(tris)
        assert len(arcs) == 2 * n_rhombs

    def test_arc_points_are_complex(self):
        from plottter.generators.penrose import _generate_rhomb_arcs
        tris = _initial_config("Sun", 1.0)
        tris = _subdivide(tris)
        for arc in _generate_rhomb_arcs(tris):
            for z in arc:
                assert isinstance(z, complex)


# ---------------------------------------------------------------------------
# Task 46.2 — Rhombs draw mode integration (updated for 46.3 API)
# ---------------------------------------------------------------------------

class TestEdgesOnlyMode:
    """Integration tests for the 'Edges Only' render mode in PenroseGenerator."""

    def _gen_edges(self, config="Sun", subdivisions=3, rotation_deg=0.0):
        from plottter.generators.penrose import PenroseGenerator
        from plottter.models.canvas import Canvas
        gen = PenroseGenerator()
        canvas = Canvas.from_preset("A4", margin=10.0)
        return gen.generate(
            _make_params(
                initial_config=config,
                subdivisions=subdivisions,
                rotation_deg=rotation_deg,
                render_mode="Edges Only",
            ),
            canvas,
        )

    def test_returns_nonempty_list(self):
        result = self._gen_edges()
        assert isinstance(result, list)
        assert len(result) > 0

    # (c) All output edges within canvas bounds
    def test_all_edges_within_canvas(self):
        from plottter.models.canvas import Canvas
        canvas = Canvas.from_preset("A4", margin=10.0)
        x1, y1, x2, y2 = canvas.drawing_area()
        result = self._gen_edges()
        for poly in result:
            for x, y in poly:
                assert x1 - 1e-6 <= x <= x2 + 1e-6, f"x={x} out of [{x1},{x2}]"
                assert y1 - 1e-6 <= y <= y2 + 1e-6, f"y={y} out of [{y1},{y2}]"

    # (d) No zero-length edges
    def test_no_zero_length_polylines(self):
        result = self._gen_edges()
        for poly in result:
            (x0, y0), (x1, y1) = poly
            dist_sq = (x1 - x0) ** 2 + (y1 - y0) ** 2
            assert dist_sq > 1e-20, f"Zero-length polyline at ({x0},{y0})"

    def test_each_polyline_has_two_points(self):
        result = self._gen_edges()
        for poly in result:
            assert len(poly) == 2

    def test_edges_star_config(self):
        result = self._gen_edges(config="Star", subdivisions=3)
        assert len(result) > 0

    def test_edges_with_rotation(self):
        r0 = self._gen_edges(rotation_deg=0.0)
        r45 = self._gen_edges(rotation_deg=45.0)
        # Different rotations produce different edge sets
        assert len(r0) == len(r45)  # same count
        # But coordinates differ
        coords0 = {(round(p[0], 3), round(p[1], 3)) for poly in r0 for p in poly}
        coords45 = {(round(p[0], 3), round(p[1], 3)) for poly in r45 for p in poly}
        assert coords0 != coords45


# ---------------------------------------------------------------------------
# Task 46.4 — Presets, registration, and high-depth generation
# ---------------------------------------------------------------------------

_EXPECTED_PRESET_NAMES = {
    "Classic P3",
    "Penrose Stars",
    "Arc Pattern",
    "Full Decoration",
    "Dense Tiling",
    "Dart Origin",
}


def _small_canvas_penrose():
    """80×80 mm canvas for fast preset testing."""
    from plottter.models.canvas import Canvas
    return Canvas(width_mm=80.0, height_mm=80.0, margin_mm=5.0)


class TestTask464Presets:
    """Task 46.4: preset definitions, registration, and high-depth generation."""

    def setup_method(self):
        from plottter.generators.penrose import PenroseGenerator
        self.gen = PenroseGenerator()
        self.canvas = _small_canvas_penrose()

    # --- preset inventory ---

    def test_exact_six_presets(self):
        presets = self.gen.get_presets()
        assert len(presets) == 6, f"Expected 6 presets, got {len(presets)}"

    def test_all_expected_preset_names_present(self):
        preset_names = {p.name for p in self.gen.get_presets()}
        for name in _EXPECTED_PRESET_NAMES:
            assert name in preset_names, f"Preset '{name}' is missing"

    # --- preset parameter correctness ---

    def test_classic_p3_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Classic P3"]
        assert p.params["initial_config"] == "Sun"
        assert p.params["subdivisions"] == 5
        assert p.params["render_mode"] == "Edges Only"

    def test_penrose_stars_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Penrose Stars"]
        assert p.params["initial_config"] == "Star"
        assert p.params["subdivisions"] == 4
        assert p.params["render_mode"] == "Edges Only"

    def test_arc_pattern_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Arc Pattern"]
        assert p.params["initial_config"] == "Sun"
        assert p.params["subdivisions"] == 5
        assert p.params["render_mode"] == "Arcs Only"

    def test_full_decoration_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Full Decoration"]
        assert p.params["initial_config"] == "Sun"
        assert p.params["subdivisions"] == 6
        assert p.params["render_mode"] == "Edges + Arcs"

    def test_dense_tiling_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Dense Tiling"]
        assert p.params["initial_config"] == "Sun"
        assert p.params["subdivisions"] == 7
        assert p.params["render_mode"] == "Edges Only"

    def test_dart_origin_params(self):
        presets = {p.name: p for p in self.gen.get_presets()}
        p = presets["Dart Origin"]
        assert p.params["initial_config"] == "Dart"
        assert p.params["subdivisions"] == 5
        assert p.params["render_mode"] == "Edges Only"

    # (f) all presets generate valid output

    def test_all_presets_generate_nonempty_list(self):
        """(f) Every preset must return a non-empty list of Polylines."""
        for preset in self.gen.get_presets():
            result = self.gen.generate(preset.params, self.canvas)
            assert isinstance(result, list), (
                f"Preset {preset.name!r}: expected list, got {type(result)}"
            )
            assert len(result) > 0, (
                f"Preset {preset.name!r}: expected non-empty output"
            )

    def test_all_preset_polylines_have_at_least_two_points(self):
        """Every Polyline from every preset must have >= 2 points."""
        for preset in self.gen.get_presets():
            result = self.gen.generate(preset.params, self.canvas)
            for pl in result:
                assert len(pl) >= 2, (
                    f"Preset {preset.name!r}: polyline has fewer than 2 points"
                )

    def test_preset_output_within_canvas_bounds(self):
        """Preset output coordinates must lie within the canvas drawing area."""
        x1, y1, x2, y2 = self.canvas.drawing_area()
        tol = 1e-6
        for preset in self.gen.get_presets():
            result = self.gen.generate(preset.params, self.canvas)
            for pl in result:
                for x, y in pl:
                    assert x >= x1 - tol, f"Preset {preset.name!r}: x={x:.3f} < x_min={x1}"
                    assert x <= x2 + tol, f"Preset {preset.name!r}: x={x:.3f} > x_max={x2}"
                    assert y >= y1 - tol, f"Preset {preset.name!r}: y={y:.3f} < y_min={y1}"
                    assert y <= y2 + tol, f"Preset {preset.name!r}: y={y:.3f} > y_max={y2}"

    # (g) generator is registered and accessible

    def test_generator_registered_and_accessible(self):
        """(g) 'Penrose Tiling' must be in GENERATORS and return a usable instance."""
        from plottter.generators import GENERATORS
        assert "Penrose Tiling" in GENERATORS
        cls = GENERATORS["Penrose Tiling"]
        instance = cls()
        assert len(instance.get_parameters()) > 0

    # (h) high subdivision depth (7) completes without error

    def test_high_subdivision_depth_7_edges_only(self):
        """(h) Depth 7, Edges Only must complete and return non-empty output."""
        result = self.gen.generate(
            _make_params(subdivisions=7, render_mode="Edges Only"),
            self.canvas,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_high_subdivision_depth_7_arcs_only(self):
        """(h) Depth 7, Arcs Only must also complete without error."""
        result = self.gen.generate(
            _make_params(subdivisions=7, render_mode="Arcs Only"),
            self.canvas,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    # (a) subdivision increases triangle count by expected factor

    def test_subdivision_count_grows_by_expected_factor(self):
        """(a) Each subdivision multiplies triangle count by ~PHI² after warm-up."""
        tris = _initial_config("Sun", 1.0)
        for _ in range(3):
            tris = _subdivide(tris)
        n_before = len(tris)
        tris = _subdivide(tris)
        n_after = len(tris)
        ratio = n_after / n_before
        assert abs(ratio - PHI ** 2) < 0.15, (
            f"Growth ratio {ratio:.4f} expected near PHI² ≈ {PHI**2:.4f}"
        )

    # (b) edge deduplication removes correct number of edges

    def test_edge_deduplication_reduces_count(self):
        """(b) Deduplicated edge count must be < 4 per rhomb with no duplicates."""
        from plottter.generators.penrose import _long_edge_keys
        tris = _initial_config("Sun", 1.0)
        for _ in range(3):
            tris = _subdivide(tris)
        long_edge_map: dict = {}
        for i, (t, A, B, C) in enumerate(tris):
            for key in _long_edge_keys(t, A, B, C):
                long_edge_map.setdefault(key, []).append(i)
        n_rhombs = sum(
            1 for idxs in long_edge_map.values()
            if len(idxs) == 2 and tris[idxs[0]][0] == tris[idxs[1]][0]
        )
        edges = _triangles_to_edges(tris)
        assert len(edges) < 4 * n_rhombs, (
            f"Edge count {len(edges)} should be < 4 × n_rhombs ({4 * n_rhombs})"
        )
        keys = [_edge_key(v1, v2) for v1, v2 in edges]
        assert len(keys) == len(set(keys)), "Duplicate edges in deduplicated output"

    # (c) all edges within canvas bounds after clipping

    def test_all_edges_within_canvas_after_clipping(self):
        """(c) generate() output must have all points within canvas drawing area."""
        from plottter.models.canvas import Canvas
        canvas = Canvas.from_preset("A4", margin=10.0)
        x1, y1, x2, y2 = canvas.drawing_area()
        result = self.gen.generate(
            _make_params(subdivisions=4, render_mode="Edges Only"),
            canvas,
        )
        tol = 1e-6
        for pl in result:
            for x, y in pl:
                assert x >= x1 - tol and x <= x2 + tol
                assert y >= y1 - tol and y <= y2 + tol

    # (d) arc mode produces curved polylines with >2 points per arc

    def test_arc_mode_produces_multipoint_polylines(self):
        """(d) 'Arcs Only' must yield at least one polyline with more than 2 points."""
        result = self.gen.generate(
            _make_params(subdivisions=3, render_mode="Arcs Only"),
            self.canvas,
        )
        assert len(result) > 0
        assert any(len(pl) > 2 for pl in result), (
            "'Arcs Only' must produce at least one curved (>2 point) polyline"
        )

    # (e) different initial configs produce different triangle counts

    def test_different_configs_produce_different_counts(self):
        """(e) Sun and Dart start from different seed sizes → different counts."""
        sun = _initial_config("Sun", 1.0)
        dart = _initial_config("Dart", 1.0)
        for _ in range(3):
            sun = _subdivide(sun)
            dart = _subdivide(dart)
        assert len(sun) != len(dart), (
            f"Sun ({len(sun)}) and Dart ({len(dart)}) should have different counts"
        )
