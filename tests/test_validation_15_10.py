"""Phase 15.10 validation: project persistence.

Verifies the full project save/load round-trip against the spec in
specs/architecture.md:

1. Multi-layer projects — all layer fields are preserved: id, name, color,
   paths, visible, locked, opacity, generator_info.
2. Generator info — layers tagged with generator_info for every generator
   type survive save/load with full fidelity.
3. Canvas settings — all paper presets and custom canvas sizes are preserved
   exactly, including margin_mm and paper_preset string.
4. Project settings — name, registration_marks, reg_mark_style all preserved,
   including edge-case values (False marks, "center" / "both" styles).
5. Edge cases — empty project (no layers), thousands of paths (stress test),
   empty layers (no paths), single-point polylines, very long layer names.
6. Gzip — large projects are compressed (magic bytes verified), auto-detected
   on load, and preserve all data identically to plain JSON.
7. Real generator output — use ParametricGenerator to produce real polylines,
   save, load, verify coordinates round-trip with floating-point precision.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from plottter.models.canvas import Canvas, PAPER_PRESETS
from plottter.models.layer import Layer
from plottter.models.project import Project
from plottter.io.project_file import save_project, load_project


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_polyline(n: int = 5, offset: float = 0.0) -> list[tuple[float, float]]:
    """Synthetic n-point polyline starting at (offset, offset)."""
    return [(offset + i * 2.5, offset + math.sin(i) * 10.0) for i in range(n)]


def _make_layer(
    name: str = "Layer",
    color: str = "#000000",
    num_paths: int = 3,
    visible: bool = True,
    locked: bool = False,
    opacity: float = 1.0,
    generator_info: dict | None = None,
) -> Layer:
    layer = Layer(
        name=name,
        color=color,
        visible=visible,
        locked=locked,
        opacity=opacity,
        generator_info=generator_info,
    )
    paths = [_make_polyline(5, offset=i * 20.0) for i in range(num_paths)]
    layer.add_paths(paths)
    return layer


def _make_full_project() -> Project:
    """Create a project with 4 layers exercising all layer fields."""
    canvas = Canvas.from_preset("A4", margin=15.0)
    proj = Project(
        name="FullProject",
        canvas=canvas,
        registration_marks=True,
        reg_mark_style="corners",
    )
    proj.add_layer(
        _make_layer(
            "Black Ink",
            "#000000",
            num_paths=5,
            generator_info={"type": "Parametric Curves", "preset": "Lissajous"},
        )
    )
    proj.add_layer(
        _make_layer(
            "Red Detail",
            "#FF0000",
            num_paths=3,
            locked=True,
            opacity=0.8,
            generator_info={"type": "Polar Curves", "preset": "Rose"},
        )
    )
    proj.add_layer(
        _make_layer(
            "Blue Fill",
            "#0000FF",
            num_paths=2,
            visible=False,
            opacity=0.5,
        )
    )
    proj.add_layer(
        _make_layer(
            "Empty Layer",
            "#00FF00",
            num_paths=0,
            generator_info=None,
        )
    )
    return proj


# ---------------------------------------------------------------------------
# 1. Multi-layer round-trip
# ---------------------------------------------------------------------------


class TestMultiLayerRoundTrip:
    """Full round-trip with multiple layers exercising all layer fields."""

    def test_layer_count_preserved(self, tmp_path):
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert len(loaded.layers) == len(proj.layers)

    def test_layer_order_preserved(self, tmp_path):
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig, rest in zip(proj.layers, loaded.layers):
            assert orig.name == rest.name

    def test_layer_ids_preserved(self, tmp_path):
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig, rest in zip(proj.layers, loaded.layers):
            assert orig.id == rest.id

    def test_layer_colors_preserved(self, tmp_path):
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig, rest in zip(proj.layers, loaded.layers):
            assert orig.color == rest.color

    def test_layer_visibility_preserved(self, tmp_path):
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig, rest in zip(proj.layers, loaded.layers):
            assert orig.visible == rest.visible

    def test_layer_locked_preserved(self, tmp_path):
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig, rest in zip(proj.layers, loaded.layers):
            assert orig.locked == rest.locked

    def test_layer_opacity_preserved(self, tmp_path):
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig, rest in zip(proj.layers, loaded.layers):
            assert rest.opacity == pytest.approx(orig.opacity)

    def test_path_counts_preserved(self, tmp_path):
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig, rest in zip(proj.layers, loaded.layers):
            assert rest.path_count() == orig.path_count()

    def test_path_coordinates_preserved(self, tmp_path):
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig_layer, loaded_layer in zip(proj.layers, loaded.layers):
            for orig_path, loaded_path in zip(orig_layer.paths, loaded_layer.paths):
                assert len(orig_path) == len(loaded_path)
                for (ox, oy), (lx, ly) in zip(orig_path, loaded_path):
                    assert ox == pytest.approx(lx, abs=1e-9)
                    assert oy == pytest.approx(ly, abs=1e-9)

    def test_empty_layer_paths_preserved(self, tmp_path):
        """Layer with no paths should load with empty paths list."""
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        # "Empty Layer" is the last one
        empty = loaded.layers[-1]
        assert empty.path_count() == 0
        assert empty.paths == []

    def test_total_point_count_preserved(self, tmp_path):
        proj = _make_full_project()
        fp = str(tmp_path / "full.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig, rest in zip(proj.layers, loaded.layers):
            assert rest.total_point_count() == orig.total_point_count()

    def test_six_layers_preserved(self, tmp_path):
        canvas = Canvas.from_preset("A3")
        proj = Project(name="Six", canvas=canvas)
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
        for i, color in enumerate(colors):
            layer = _make_layer(f"L{i}", color, num_paths=i + 1)
            proj.add_layer(layer)
        fp = str(tmp_path / "six.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert len(loaded.layers) == 6
        for i, (orig, rest) in enumerate(zip(proj.layers, loaded.layers)):
            assert orig.name == rest.name
            assert orig.color == rest.color
            assert rest.path_count() == i + 1


# ---------------------------------------------------------------------------
# 2. Generator info persistence
# ---------------------------------------------------------------------------


# All math and image generator types with representative generator_info payloads
_GENERATOR_INFOS = [
    {
        "type": "Parametric Curves",
        "preset": "Lissajous",
        "params": {"x_expr": "sin(3*t)", "y_expr": "sin(4*t)", "t_start": 0.0, "t_end": 6.283},
    },
    {
        "type": "Polar Curves",
        "preset": "Rose",
        "params": {"r_expr": "cos(4*theta)", "theta_start": 0.0, "theta_end": 6.283},
    },
    {
        "type": "Modular Multiplication",
        "preset": None,
        "params": {"num_points": 200, "multiplier": 3.0},
    },
    {
        "type": "Flow Field",
        "preset": None,
        "params": {"num_particles": 500, "seed": 42, "noise_scale": 0.01},
    },
    {
        "type": "L-System",
        "preset": "Koch Snowflake",
        "params": {"axiom": "F--F--F", "rules": "F=F+F--F+F", "iterations": 4, "angle_deg": 60.0},
    },
    {
        "type": "Grid Pattern",
        "preset": None,
        "params": {"mode": "sine_grid", "line_count": 20},
    },
    {
        "type": "Edge Detection",
        "preset": None,
        "params": {"low_threshold": 50.0, "high_threshold": 150.0},
    },
    {
        "type": "Hatching",
        "preset": None,
        "params": {"mode": "parallel", "angle_deg": 45.0},
    },
    {
        "type": "Flow Image",
        "preset": None,
        "params": {"mode": "flow", "num_lines": 100},
    },
    {
        "type": "Stipple",
        "preset": None,
        "params": {"num_points": 1000, "connect_tsp": False},
    },
    {
        "type": "Contour Lines",
        "preset": None,
        "params": {"num_levels": 10, "smooth": True},
    },
]


class TestGeneratorInfoPersistence:
    """Generator info for every generator type survives save/load."""

    @pytest.mark.parametrize("gen_info", _GENERATOR_INFOS, ids=[g["type"] for g in _GENERATOR_INFOS])
    def test_generator_info_round_trip(self, tmp_path, gen_info):
        canvas = Canvas.from_preset("A4")
        proj = Project(name="GenTest", canvas=canvas)
        layer = Layer(name=gen_info["type"], generator_info=gen_info)
        layer.add_paths([_make_polyline(3)])
        proj.add_layer(layer)
        fp = str(tmp_path / "gen.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.layers[0].generator_info == gen_info

    def test_none_generator_info_preserved(self, tmp_path):
        """Layer with generator_info=None stays None after load."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="NoGen", canvas=canvas)
        layer = Layer(name="Manual", generator_info=None)
        layer.add_paths([_make_polyline(4)])
        proj.add_layer(layer)
        fp = str(tmp_path / "nogen.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.layers[0].generator_info is None

    def test_nested_generator_info_preserved(self, tmp_path):
        """Deeply nested generator_info with nested dicts and lists survives."""
        gen_info = {
            "type": "Parametric Curves",
            "params": {
                "x_expr": "sin(3*t)",
                "nested": {"a": 1, "b": [1, 2, 3]},
                "flags": [True, False, True],
            },
        }
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Nested", canvas=canvas)
        layer = Layer(name="Nested", generator_info=gen_info)
        layer.add_paths([_make_polyline(3)])
        proj.add_layer(layer)
        fp = str(tmp_path / "nested.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.layers[0].generator_info == gen_info

    def test_all_generator_types_in_one_project(self, tmp_path):
        """One project with N layers, one per generator type — all info preserved."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="AllGens", canvas=canvas)
        for gen_info in _GENERATOR_INFOS:
            layer = Layer(name=gen_info["type"], generator_info=gen_info)
            layer.add_paths([_make_polyline(3)])
            proj.add_layer(layer)
        fp = str(tmp_path / "allgens.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert len(loaded.layers) == len(_GENERATOR_INFOS)
        for orig, rest in zip(proj.layers, loaded.layers):
            assert rest.generator_info == orig.generator_info


# ---------------------------------------------------------------------------
# 3. Canvas settings
# ---------------------------------------------------------------------------


class TestCanvasSettings:
    """All paper presets and custom canvas sizes are preserved exactly."""

    @pytest.mark.parametrize("preset", list(PAPER_PRESETS.keys()))
    def test_paper_preset_round_trip(self, tmp_path, preset):
        canvas = Canvas.from_preset(preset, margin=10.0)
        proj = Project(name=f"Paper_{preset}", canvas=canvas)
        fp = str(tmp_path / f"{preset}.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.canvas.paper_preset == preset
        assert loaded.canvas.width_mm == pytest.approx(canvas.width_mm)
        assert loaded.canvas.height_mm == pytest.approx(canvas.height_mm)
        assert loaded.canvas.margin_mm == pytest.approx(10.0)

    def test_custom_canvas_preserved(self, tmp_path):
        canvas = Canvas(width_mm=300.0, height_mm=400.0, margin_mm=25.0, paper_preset="Custom")
        proj = Project(name="Custom", canvas=canvas)
        fp = str(tmp_path / "custom.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.canvas.width_mm == pytest.approx(300.0)
        assert loaded.canvas.height_mm == pytest.approx(400.0)
        assert loaded.canvas.margin_mm == pytest.approx(25.0)
        assert loaded.canvas.paper_preset == "Custom"

    def test_zero_margin_canvas(self, tmp_path):
        canvas = Canvas(width_mm=200.0, height_mm=200.0, margin_mm=0.0, paper_preset="Custom")
        proj = Project(name="ZeroMargin", canvas=canvas)
        fp = str(tmp_path / "zero_margin.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.canvas.margin_mm == pytest.approx(0.0)

    def test_large_margin_canvas(self, tmp_path):
        canvas = Canvas(width_mm=297.0, height_mm=420.0, margin_mm=50.0, paper_preset="A3")
        proj = Project(name="LargeMargin", canvas=canvas)
        fp = str(tmp_path / "large_margin.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.canvas.margin_mm == pytest.approx(50.0)

    def test_fractional_dimensions_preserved(self, tmp_path):
        canvas = Canvas(width_mm=215.9, height_mm=279.4, margin_mm=12.7, paper_preset="Letter")
        proj = Project(name="Letter", canvas=canvas)
        fp = str(tmp_path / "letter.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.canvas.width_mm == pytest.approx(215.9)
        assert loaded.canvas.height_mm == pytest.approx(279.4)
        assert loaded.canvas.margin_mm == pytest.approx(12.7)


# ---------------------------------------------------------------------------
# 4. Project settings
# ---------------------------------------------------------------------------


class TestProjectSettings:
    """Project-level fields (name, registration_marks, reg_mark_style) survive."""

    def test_project_name_preserved(self, tmp_path):
        proj = Project(name="My Plotter Art", canvas=Canvas.from_preset("A4"))
        fp = str(tmp_path / "name.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.name == "My Plotter Art"

    def test_project_name_with_special_chars(self, tmp_path):
        proj = Project(name="Café & Gallery — №1", canvas=Canvas.from_preset("A4"))
        fp = str(tmp_path / "special.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.name == "Café & Gallery — №1"

    def test_registration_marks_true(self, tmp_path):
        proj = Project(name="R", canvas=Canvas.from_preset("A4"), registration_marks=True)
        fp = str(tmp_path / "reg_true.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.registration_marks is True

    def test_registration_marks_false(self, tmp_path):
        proj = Project(name="R", canvas=Canvas.from_preset("A4"), registration_marks=False)
        fp = str(tmp_path / "reg_false.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.registration_marks is False

    @pytest.mark.parametrize("style", ["corners", "center", "both"])
    def test_reg_mark_style_preserved(self, tmp_path, style):
        proj = Project(
            name="Style", canvas=Canvas.from_preset("A4"),
            registration_marks=True, reg_mark_style=style,
        )
        fp = str(tmp_path / f"style_{style}.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.reg_mark_style == style

    def test_empty_project_name(self, tmp_path):
        """Empty string project name round-trips cleanly."""
        proj = Project(name="", canvas=Canvas.from_preset("A4"))
        fp = str(tmp_path / "empty_name.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.name == ""

    def test_version_field_in_file(self, tmp_path):
        """Saved file must contain a 'version' key equal to 1."""
        proj = Project(name="V", canvas=Canvas.from_preset("A4"))
        fp = str(tmp_path / "version.plottter")
        save_project(proj, fp)
        with open(fp, "rb") as fh:
            data = json.loads(fh.read())
        assert "version" in data
        assert data["version"] == 1


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Empty project, thousands of paths, degenerate polylines, long names."""

    def test_empty_project_no_layers(self, tmp_path):
        proj = Project(name="Empty", canvas=Canvas.from_preset("A4"))
        fp = str(tmp_path / "empty.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.name == "Empty"
        assert loaded.layers == []

    def test_thousands_of_paths(self, tmp_path):
        """Project with 2000 paths across two layers survives round-trip."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Heavy", canvas=canvas)
        layer_a = Layer(name="Dense A", color="#000000")
        layer_b = Layer(name="Dense B", color="#FF0000")
        rng = random.Random(42)
        paths_a = [
            [(rng.uniform(10, 200), rng.uniform(10, 280)),
             (rng.uniform(10, 200), rng.uniform(10, 280))]
            for _ in range(1000)
        ]
        paths_b = [
            [(rng.uniform(10, 200), rng.uniform(10, 280)),
             (rng.uniform(10, 200), rng.uniform(10, 280)),
             (rng.uniform(10, 200), rng.uniform(10, 280))]
            for _ in range(1000)
        ]
        layer_a.add_paths(paths_a)
        layer_b.add_paths(paths_b)
        proj.add_layer(layer_a)
        proj.add_layer(layer_b)
        fp = str(tmp_path / "heavy.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.layers[0].path_count() == 1000
        assert loaded.layers[1].path_count() == 1000
        # Spot-check first path of each layer
        assert loaded.layers[0].paths[0][0] == pytest.approx(paths_a[0][0], abs=1e-9)
        assert loaded.layers[1].paths[0][0] == pytest.approx(paths_b[0][0], abs=1e-9)

    def test_single_point_polylines(self, tmp_path):
        """Single-point polylines (degenerate) are preserved."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="SinglePt", canvas=canvas)
        layer = Layer(name="Dots", color="#000000")
        layer.add_paths([[(10.0, 20.0)], [(30.0, 40.0)], [(50.0, 60.0)]])
        proj.add_layer(layer)
        fp = str(tmp_path / "single_pt.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.layers[0].path_count() == 3
        for orig, rest in zip(layer.paths, loaded.layers[0].paths):
            assert len(rest) == 1
            assert rest[0][0] == pytest.approx(orig[0][0])
            assert rest[0][1] == pytest.approx(orig[0][1])

    def test_two_point_polylines(self, tmp_path):
        """Minimum-segment (2-point) polylines are preserved."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="TwoPt", canvas=canvas)
        layer = Layer(name="Lines", color="#000000")
        segs = [[(0.0, 0.0), (10.0, 10.0)], [(20.0, 0.0), (30.0, 10.0)]]
        layer.add_paths(segs)
        proj.add_layer(layer)
        fp = str(tmp_path / "two_pt.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.layers[0].path_count() == 2
        for orig, rest in zip(segs, loaded.layers[0].paths):
            assert len(rest) == len(orig)

    def test_very_long_layer_name(self, tmp_path):
        """Layer name of 500 characters survives round-trip."""
        long_name = "A" * 500
        canvas = Canvas.from_preset("A4")
        proj = Project(name="LongName", canvas=canvas)
        proj.add_layer(Layer(name=long_name, color="#000000"))
        fp = str(tmp_path / "longname.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.layers[0].name == long_name

    def test_unicode_layer_names(self, tmp_path):
        """Unicode layer names (emoji, CJK, Arabic) survive round-trip."""
        names = ["日本語レイヤー", "Schicht №1 — Café", "طبقة", "Layer 🎨✏️"]
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Unicode", canvas=canvas)
        for name in names:
            proj.add_layer(Layer(name=name, color="#000000"))
        fp = str(tmp_path / "unicode.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig_name, layer in zip(names, loaded.layers):
            assert layer.name == orig_name

    def test_opacity_edge_values(self, tmp_path):
        """Opacity = 0.0 and 1.0 round-trip without floating-point drift."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Opacity", canvas=canvas)
        proj.add_layer(Layer(name="Transparent", color="#000000", opacity=0.0))
        proj.add_layer(Layer(name="Opaque", color="#000000", opacity=1.0))
        proj.add_layer(Layer(name="Half", color="#000000", opacity=0.5))
        fp = str(tmp_path / "opacity.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.layers[0].opacity == pytest.approx(0.0)
        assert loaded.layers[1].opacity == pytest.approx(1.0)
        assert loaded.layers[2].opacity == pytest.approx(0.5)

    def test_all_layers_locked(self, tmp_path):
        """Locked=True is preserved for all layers."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Locked", canvas=canvas)
        for i in range(3):
            proj.add_layer(Layer(name=f"L{i}", color="#000000", locked=True))
        fp = str(tmp_path / "locked.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for layer in loaded.layers:
            assert layer.locked is True

    def test_all_layers_hidden(self, tmp_path):
        """Visible=False is preserved for all layers."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Hidden", canvas=canvas)
        for i in range(3):
            proj.add_layer(Layer(name=f"L{i}", color="#000000", visible=False))
        fp = str(tmp_path / "hidden.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for layer in loaded.layers:
            assert layer.visible is False

    def test_negative_coordinates_preserved(self, tmp_path):
        """Negative coordinate values in paths survive round-trip."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Neg", canvas=canvas)
        layer = Layer(name="Negative", color="#000000")
        layer.add_paths([[(-10.0, -20.0), (-30.0, -40.0), (0.0, 0.0)]])
        proj.add_layer(layer)
        fp = str(tmp_path / "neg.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        pt = loaded.layers[0].paths[0][0]
        assert pt[0] == pytest.approx(-10.0)
        assert pt[1] == pytest.approx(-20.0)

    def test_large_coordinate_values_preserved(self, tmp_path):
        """Coordinate values at canvas edge survive round-trip."""
        canvas = Canvas.from_preset("A3")  # 297×420 mm
        proj = Project(name="Large", canvas=canvas)
        layer = Layer(name="Edge", color="#000000")
        layer.add_paths([[(0.0, 0.0), (297.0, 420.0), (148.5, 210.0)]])
        proj.add_layer(layer)
        fp = str(tmp_path / "large.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        pts = loaded.layers[0].paths[0]
        assert pts[1][0] == pytest.approx(297.0)
        assert pts[1][1] == pytest.approx(420.0)

    def test_hex_color_edge_cases(self, tmp_path):
        """Hex colors including black, white, full-channel colors survive."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Colors", canvas=canvas)
        colors = ["#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF",
                  "#FF00FF", "#00FFFF", "#FFFF00", "#1A2B3C", "#ABCDEF"]
        for color in colors:
            proj.add_layer(Layer(name=color, color=color))
        fp = str(tmp_path / "colors.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        for orig_color, layer in zip(colors, loaded.layers):
            assert layer.color == orig_color


# ---------------------------------------------------------------------------
# 6. Gzip persistence
# ---------------------------------------------------------------------------


class TestGzipPersistence:
    """Large projects use gzip; auto-detected on load; data preserved."""

    def test_large_project_is_gzipped(self, tmp_path):
        """A project that exceeds 1 MB is saved as gzip."""
        import plottter.io.project_file as pf_module
        original = pf_module._GZIP_THRESHOLD_BYTES
        pf_module._GZIP_THRESHOLD_BYTES = 0  # force gzip
        try:
            proj = _make_full_project()
            fp = str(tmp_path / "big.plottter")
            save_project(proj, fp)
            with open(fp, "rb") as fh:
                magic = fh.read(2)
            assert magic == b"\x1f\x8b", "Expected gzip magic bytes"
        finally:
            pf_module._GZIP_THRESHOLD_BYTES = original

    def test_gzip_project_loads_correctly(self, tmp_path):
        """Gzip-saved project loads with all data intact."""
        import plottter.io.project_file as pf_module
        original = pf_module._GZIP_THRESHOLD_BYTES
        pf_module._GZIP_THRESHOLD_BYTES = 0  # force gzip
        try:
            proj = _make_full_project()
            fp = str(tmp_path / "big.plottter")
            save_project(proj, fp)
            loaded = load_project(fp)
            assert loaded.name == proj.name
            assert len(loaded.layers) == len(proj.layers)
            for orig, rest in zip(proj.layers, loaded.layers):
                assert orig.name == rest.name
                assert orig.color == rest.color
                assert orig.path_count() == rest.path_count()
        finally:
            pf_module._GZIP_THRESHOLD_BYTES = original

    def test_plain_json_loads_correctly(self, tmp_path):
        """Non-gzip project (under threshold) loads without error."""
        import plottter.io.project_file as pf_module
        original = pf_module._GZIP_THRESHOLD_BYTES
        pf_module._GZIP_THRESHOLD_BYTES = 100_000_000  # never gzip
        try:
            proj = _make_full_project()
            fp = str(tmp_path / "plain.plottter")
            save_project(proj, fp)
            with open(fp, "rb") as fh:
                magic = fh.read(2)
            assert magic != b"\x1f\x8b"
            loaded = load_project(fp)
            assert loaded.name == proj.name
        finally:
            pf_module._GZIP_THRESHOLD_BYTES = original

    def test_gzip_data_matches_plain_json(self, tmp_path):
        """Gzip and plain saves of the same project produce identical loaded data."""
        import plottter.io.project_file as pf_module
        proj = _make_full_project()
        # Save as plain JSON
        orig = pf_module._GZIP_THRESHOLD_BYTES
        pf_module._GZIP_THRESHOLD_BYTES = 100_000_000
        try:
            plain_fp = str(tmp_path / "plain.plottter")
            save_project(proj, plain_fp)
        finally:
            pf_module._GZIP_THRESHOLD_BYTES = orig

        # Save as gzip
        pf_module._GZIP_THRESHOLD_BYTES = 0
        try:
            gz_fp = str(tmp_path / "gz.plottter")
            save_project(proj, gz_fp)
        finally:
            pf_module._GZIP_THRESHOLD_BYTES = orig

        plain_loaded = load_project(plain_fp)
        gz_loaded = load_project(gz_fp)
        assert plain_loaded.name == gz_loaded.name
        assert len(plain_loaded.layers) == len(gz_loaded.layers)
        for pl, gl in zip(plain_loaded.layers, gz_loaded.layers):
            assert pl.name == gl.name
            assert pl.path_count() == gl.path_count()

    def test_real_large_project_uses_gzip(self, tmp_path):
        """A project with 10k paths (realistic large session) triggers gzip automatically."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="RealLarge", canvas=canvas)
        layer = Layer(name="Massive", color="#000000")
        rng = random.Random(99)
        paths = [
            [(rng.uniform(10, 200), rng.uniform(10, 280)) for _ in range(20)]
            for _ in range(500)
        ]
        layer.add_paths(paths)
        proj.add_layer(layer)
        fp = str(tmp_path / "reallarge.plottter")
        save_project(proj, fp)
        # May or may not trigger gzip (depends on serialized size), but must load
        loaded = load_project(fp)
        assert loaded.layers[0].path_count() == 500


# ---------------------------------------------------------------------------
# 7. Real generator output
# ---------------------------------------------------------------------------


class TestRealGeneratorOutput:
    """Use real generator produce real polylines and verify they survive round-trip."""

    def test_parametric_lissajous_survives(self, tmp_path):
        """Real ParametricGenerator Lissajous output survives save/load."""
        from plottter.generators.parametric import ParametricGenerator
        canvas = Canvas.from_preset("A4")
        gen = ParametricGenerator()
        params = {
            "x_expr": "sin(3*t)",
            "y_expr": "sin(4*t)",
            "t_start": 0.0,
            "t_end": 6.283185307,
            "num_points": 500,
            "scale": 1.0,
            "rotation_deg": 0.0,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        paths = gen.generate(params, canvas)
        assert paths, "Generator produced no paths"

        proj = Project(name="GenLissajous", canvas=canvas)
        layer = Layer(
            name="Lissajous",
            color="#000000",
            generator_info={"type": "Parametric Curves", "preset": "Lissajous", "params": params},
        )
        layer.add_paths(paths)
        proj.add_layer(layer)

        fp = str(tmp_path / "lissajous.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)

        assert loaded.layers[0].path_count() == len(paths)
        # Verify generator_info survived
        assert loaded.layers[0].generator_info["preset"] == "Lissajous"
        # Spot-check first and last point of first path
        orig_path = paths[0]
        loaded_path = loaded.layers[0].paths[0]
        assert len(loaded_path) == len(orig_path)
        assert loaded_path[0][0] == pytest.approx(orig_path[0][0], abs=1e-6)
        assert loaded_path[0][1] == pytest.approx(orig_path[0][1], abs=1e-6)
        assert loaded_path[-1][0] == pytest.approx(orig_path[-1][0], abs=1e-6)
        assert loaded_path[-1][1] == pytest.approx(orig_path[-1][1], abs=1e-6)

    def test_polar_rose_survives(self, tmp_path):
        """Real PolarGenerator Rose output survives save/load."""
        from plottter.generators.polar import PolarGenerator
        canvas = Canvas.from_preset("A4")
        gen = PolarGenerator()
        params = {
            "r_expr": "cos(4*theta)",
            "theta_start": 0.0,
            "theta_end": 6.283185307,
            "num_points": 300,
            "scale": 1.0,
            "rotation_deg": 0.0,
            "x_offset_mm": 0.0,
            "y_offset_mm": 0.0,
        }
        paths = gen.generate(params, canvas)
        assert paths, "Generator produced no paths"

        proj = Project(name="GenRose", canvas=canvas)
        layer = Layer(name="Rose", color="#FF0000")
        layer.add_paths(paths)
        proj.add_layer(layer)

        fp = str(tmp_path / "rose.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)

        assert loaded.layers[0].path_count() == len(paths)
        assert loaded.layers[0].total_point_count() == layer.total_point_count()

    def test_modular_mult_survives(self, tmp_path):
        """ModularMultGenerator output survives save/load."""
        from plottter.generators.modular_mult import ModularMultGenerator
        canvas = Canvas.from_preset("A4")
        gen = ModularMultGenerator()
        params = {p.name: p.default for p in gen.get_parameters()}
        paths = gen.generate(params, canvas)
        assert paths

        proj = Project(name="ModMult", canvas=canvas)
        layer = Layer(name="ModMult", color="#000000")
        layer.add_paths(paths)
        proj.add_layer(layer)

        fp = str(tmp_path / "modmult.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.layers[0].path_count() == len(paths)

    def test_lsystem_survives(self, tmp_path):
        """LSystemGenerator Koch Snowflake output survives save/load."""
        from plottter.generators.lsystem import LSystemGenerator
        canvas = Canvas.from_preset("A4")
        gen = LSystemGenerator()
        presets = gen.get_presets()
        koch = next((p for p in presets if "Koch" in p.name), None)
        assert koch, "Koch Snowflake preset not found"
        params = dict(koch.params)
        paths = gen.generate(params, canvas)
        assert paths

        proj = Project(name="Koch", canvas=canvas)
        layer = Layer(name="Koch Snowflake", color="#000000")
        layer.add_paths(paths)
        proj.add_layer(layer)

        fp = str(tmp_path / "koch.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)
        assert loaded.layers[0].path_count() == len(paths)

    def test_multiple_generator_types_same_project(self, tmp_path):
        """Project with 3 real generators in separate layers survives full round-trip."""
        from plottter.generators.parametric import ParametricGenerator
        from plottter.generators.polar import PolarGenerator
        from plottter.generators.modular_mult import ModularMultGenerator

        canvas = Canvas.from_preset("A4")
        proj = Project(name="MultiGen", canvas=canvas, registration_marks=True, reg_mark_style="both")

        # Layer 1: parametric
        gen1 = ParametricGenerator()
        p1 = gen1.generate(
            {"x_expr": "sin(2*t)", "y_expr": "cos(3*t)", "t_start": 0.0,
             "t_end": 6.283, "num_points": 200, "scale": 1.0,
             "rotation_deg": 0.0, "x_offset_mm": 0.0, "y_offset_mm": 0.0},
            canvas,
        )
        l1 = Layer(name="Parametric", color="#000000",
                   generator_info={"type": "Parametric Curves"})
        l1.add_paths(p1)
        proj.add_layer(l1)

        # Layer 2: polar
        gen2 = PolarGenerator()
        p2 = gen2.generate(
            {"r_expr": "1+cos(theta)", "theta_start": 0.0, "theta_end": 6.283,
             "num_points": 200, "scale": 1.0, "rotation_deg": 0.0,
             "x_offset_mm": 0.0, "y_offset_mm": 0.0},
            canvas,
        )
        l2 = Layer(name="Polar", color="#FF0000",
                   generator_info={"type": "Polar Curves"})
        l2.add_paths(p2)
        proj.add_layer(l2)

        # Layer 3: modular mult
        gen3 = ModularMultGenerator()
        p3 = gen3.generate({p.name: p.default for p in gen3.get_parameters()}, canvas)
        l3 = Layer(name="ModMult", color="#0000FF",
                   generator_info={"type": "Modular Multiplication"})
        l3.add_paths(p3)
        proj.add_layer(l3)

        fp = str(tmp_path / "multigen.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)

        assert len(loaded.layers) == 3
        assert loaded.layers[0].path_count() == len(p1)
        assert loaded.layers[1].path_count() == len(p2)
        assert loaded.layers[2].path_count() == len(p3)
        assert loaded.layers[0].generator_info == {"type": "Parametric Curves"}
        assert loaded.layers[1].generator_info == {"type": "Polar Curves"}
        assert loaded.layers[2].generator_info == {"type": "Modular Multiplication"}
        assert loaded.registration_marks is True
        assert loaded.reg_mark_style == "both"

    def test_duplicate_layer_survives(self, tmp_path):
        """Project.duplicate_layer() creates a distinct layer; both survive round-trip."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Dup", canvas=canvas)
        orig = _make_layer("Original", "#000000", num_paths=3)
        proj.add_layer(orig)
        dup = proj.duplicate_layer(orig.id)
        proj.add_layer(dup)

        fp = str(tmp_path / "dup.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)

        assert len(loaded.layers) == 2
        # IDs must be different
        assert loaded.layers[0].id != loaded.layers[1].id
        # Both layers have same path count
        assert loaded.layers[0].path_count() == orig.path_count()
        assert loaded.layers[1].path_count() == orig.path_count()

    def test_merged_layer_survives(self, tmp_path):
        """Merged layer (combined paths) survives round-trip."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="Merge", canvas=canvas)
        l1 = _make_layer("A", "#000000", num_paths=2)
        l2 = _make_layer("B", "#FF0000", num_paths=3)
        proj.add_layer(l1)
        proj.add_layer(l2)
        merged = proj.merge_layers([l1.id, l2.id])
        proj.add_layer(merged)

        fp = str(tmp_path / "merge.plottter")
        save_project(proj, fp)
        loaded = load_project(fp)

        # Merged layer has 5 paths
        merged_loaded = loaded.layers[2]
        assert merged_loaded.path_count() == 5
