"""Tests for SVG export functionality."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import pytest

from plottter.export.svg import (
    export_all_layers_svg,
    export_combined_svg,
    export_layer_svg,
)
from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.project import Project

SVG_NS = "http://www.w3.org/2000/svg"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canvas() -> Canvas:
    return Canvas.from_preset("A4", margin=10.0)


@pytest.fixture
def layer_with_paths() -> Layer:
    layer = Layer(name="Test Layer", color="#FF0000")
    layer.add_paths([
        [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)],
        [(5.0, 5.0), (100.0, 100.0)],
    ])
    return layer


@pytest.fixture
def project(canvas: Canvas, layer_with_paths: Layer) -> Project:
    proj = Project(name="TestProject", canvas=canvas)
    proj.add_layer(layer_with_paths)
    return proj


@pytest.fixture
def default_settings() -> dict:
    return {
        "registration_marks": True,
        "stroke_width": 0.3,
        "reg_mark_style": "corners",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_svg(filepath: str) -> ET.Element:
    tree = ET.parse(filepath)
    return tree.getroot()


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _find_all(root: ET.Element, local_tag: str) -> list[ET.Element]:
    return root.findall(f".//{{{SVG_NS}}}{local_tag}")


# ---------------------------------------------------------------------------
# export_layer_svg
# ---------------------------------------------------------------------------


def test_svg_is_valid_xml(tmp_path, canvas, layer_with_paths, default_settings):
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, default_settings)
    # Should not raise
    root = _parse_svg(out)
    assert root is not None


def test_svg_viewbox_matches_canvas(tmp_path, canvas, layer_with_paths, default_settings):
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, default_settings)
    root = _parse_svg(out)
    vb = root.get("viewBox")
    assert vb == f"0 0 {canvas.width_mm} {canvas.height_mm}"


def test_svg_width_height_attrs(tmp_path, canvas, layer_with_paths, default_settings):
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, default_settings)
    root = _parse_svg(out)
    assert root.get("width") == f"{canvas.width_mm}mm"
    assert root.get("height") == f"{canvas.height_mm}mm"


def test_polylines_present_for_each_path(tmp_path, canvas, layer_with_paths, default_settings):
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, default_settings)
    root = _parse_svg(out)
    polylines = _find_all(root, "polyline")
    assert len(polylines) == len(layer_with_paths.paths)


def test_stroke_color_matches_layer(tmp_path, canvas, layer_with_paths, default_settings):
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, default_settings)
    root = _parse_svg(out)
    # Layer group should carry the layer color as stroke
    groups = _find_all(root, "g")
    layer_groups = [g for g in groups if g.get("id", "").startswith("layer_")]
    assert layer_groups, "No layer group found in SVG"
    assert layer_groups[0].get("stroke") == layer_with_paths.color


def test_fill_is_none(tmp_path, canvas, layer_with_paths, default_settings):
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, default_settings)
    root = _parse_svg(out)
    groups = _find_all(root, "g")
    layer_groups = [g for g in groups if g.get("id", "").startswith("layer_")]
    assert layer_groups[0].get("fill") == "none"


def test_stroke_width_setting_respected(tmp_path, canvas, layer_with_paths):
    settings = {"registration_marks": False, "stroke_width": 1.5}
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, settings)
    root = _parse_svg(out)
    groups = _find_all(root, "g")
    layer_groups = [g for g in groups if g.get("id", "").startswith("layer_")]
    sw = layer_groups[0].get("stroke-width", "")
    assert "1.500" in sw


def test_registration_marks_present_when_enabled(tmp_path, canvas, layer_with_paths):
    settings = {"registration_marks": True, "stroke_width": 0.3, "reg_mark_style": "corners"}
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, settings)
    root = _parse_svg(out)
    # Registration group should exist
    reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
    assert reg_groups, "Registration mark group not found"
    lines = _find_all(reg_groups[0], "line")
    assert len(lines) > 0, "No lines in registration group"


def test_registration_marks_absent_when_disabled(tmp_path, canvas, layer_with_paths):
    settings = {"registration_marks": False, "stroke_width": 0.3}
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, settings)
    root = _parse_svg(out)
    reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
    assert not reg_groups, "Registration group should not be present"


def test_corners_style_has_four_crosshairs(tmp_path, canvas, layer_with_paths):
    settings = {"registration_marks": True, "stroke_width": 0.3, "reg_mark_style": "corners"}
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, settings)
    root = _parse_svg(out)
    reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
    lines = _find_all(reg_groups[0], "line")
    # 4 corners × 2 lines per cross = 8 lines
    assert len(lines) == 8


def test_center_style_has_one_crosshair(tmp_path, canvas, layer_with_paths):
    settings = {"registration_marks": True, "stroke_width": 0.3, "reg_mark_style": "center"}
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, settings)
    root = _parse_svg(out)
    reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
    lines = _find_all(reg_groups[0], "line")
    # 1 center × 2 lines per cross = 2 lines
    assert len(lines) == 2


def test_both_style_has_five_crosshairs(tmp_path, canvas, layer_with_paths):
    settings = {"registration_marks": True, "stroke_width": 0.3, "reg_mark_style": "both"}
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer_with_paths, canvas, out, settings)
    root = _parse_svg(out)
    reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
    lines = _find_all(reg_groups[0], "line")
    # (4 corners + 1 center) × 2 = 10 lines
    assert len(lines) == 10


def test_single_point_path_skipped(tmp_path, canvas):
    """Paths with fewer than 2 points should be silently skipped."""
    layer = Layer(name="Sparse", color="#000000")
    layer.add_paths([[(5.0, 5.0)]])  # single-point path
    settings = {"registration_marks": False, "stroke_width": 0.3}
    out = str(tmp_path / "out.svg")
    export_layer_svg(layer, canvas, out, settings)
    root = _parse_svg(out)
    polylines = _find_all(root, "polyline")
    assert len(polylines) == 0


# ---------------------------------------------------------------------------
# export_all_layers_svg
# ---------------------------------------------------------------------------


def test_batch_export_creates_correct_number_of_files(tmp_path, project, default_settings):
    out_dir = str(tmp_path / "batch")
    export_all_layers_svg(project, out_dir, default_settings)
    svg_files = [f for f in os.listdir(out_dir) if f.endswith(".svg")]
    visible_count = sum(1 for lyr in project.layers if lyr.visible)
    assert len(svg_files) == visible_count


def test_batch_export_file_naming_pattern(tmp_path, project, default_settings):
    out_dir = str(tmp_path / "batch")
    export_all_layers_svg(project, out_dir, default_settings)
    svg_files = os.listdir(out_dir)
    # Verify naming: project_name_01_layer_name.svg
    assert any("TestProject_01_" in f for f in svg_files)


def test_batch_export_hidden_layers_excluded(tmp_path, canvas, default_settings):
    proj = Project(name="Proj", canvas=canvas)
    visible = Layer(name="Visible", color="#000000")
    hidden = Layer(name="Hidden", color="#FF0000", visible=False)
    hidden.add_paths([[(0.0, 0.0), (1.0, 1.0)]])
    proj.add_layer(visible)
    proj.add_layer(hidden)

    out_dir = str(tmp_path / "batch")
    export_all_layers_svg(proj, out_dir, default_settings)
    svg_files = [f for f in os.listdir(out_dir) if f.endswith(".svg")]
    assert len(svg_files) == 1  # only the visible layer


def test_batch_export_output_dir_created(tmp_path, project, default_settings):
    out_dir = str(tmp_path / "new" / "subdir")
    assert not os.path.exists(out_dir)
    export_all_layers_svg(project, out_dir, default_settings)
    assert os.path.isdir(out_dir)


def test_batch_export_files_are_valid_xml(tmp_path, project, default_settings):
    out_dir = str(tmp_path / "batch")
    export_all_layers_svg(project, out_dir, default_settings)
    for svg_file in os.listdir(out_dir):
        if svg_file.endswith(".svg"):
            root = _parse_svg(os.path.join(out_dir, svg_file))
            assert root is not None


# ---------------------------------------------------------------------------
# export_combined_svg
# ---------------------------------------------------------------------------


def test_combined_export_is_valid_xml(tmp_path, project, default_settings):
    out = str(tmp_path / "combined.svg")
    export_combined_svg(project, out, default_settings)
    root = _parse_svg(out)
    assert root is not None


def test_combined_export_has_layer_groups_for_visible_layers(tmp_path, project, default_settings):
    out = str(tmp_path / "combined.svg")
    export_combined_svg(project, out, default_settings)
    root = _parse_svg(out)
    groups = _find_all(root, "g")
    layer_groups = [g for g in groups if g.get("id", "").startswith("layer_")]
    visible_count = sum(1 for lyr in project.layers if lyr.visible)
    assert len(layer_groups) == visible_count


def test_combined_export_excludes_hidden_layers(tmp_path, canvas, default_settings):
    proj = Project(name="Combo", canvas=canvas)
    visible = Layer(name="Visible", color="#000000")
    visible.add_paths([[(0.0, 0.0), (10.0, 10.0)]])
    hidden = Layer(name="Hidden", color="#FF0000", visible=False)
    hidden.add_paths([[(1.0, 1.0), (2.0, 2.0)]])
    proj.add_layer(visible)
    proj.add_layer(hidden)

    out = str(tmp_path / "combined.svg")
    export_combined_svg(proj, out, default_settings)
    root = _parse_svg(out)
    groups = _find_all(root, "g")
    layer_groups = [g for g in groups if g.get("id", "").startswith("layer_")]
    assert len(layer_groups) == 1
    assert layer_groups[0].get("id") == "layer_Visible"


def test_combined_export_registration_marks(tmp_path, project, default_settings):
    out = str(tmp_path / "combined.svg")
    export_combined_svg(project, out, default_settings)
    root = _parse_svg(out)
    reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
    assert reg_groups


def test_combined_export_polyline_coordinates(tmp_path, canvas):
    """Check that coordinate values appear in the polyline points attribute."""
    layer = Layer(name="L", color="#000000")
    layer.add_paths([[(1.5, 2.5), (3.0, 4.0)]])
    proj = Project(name="P", canvas=canvas)
    proj.add_layer(layer)
    settings = {"registration_marks": False, "stroke_width": 0.3}
    out = str(tmp_path / "combined.svg")
    export_combined_svg(proj, out, settings)
    with open(out) as f:
        content = f.read()
    assert "1.500" in content
    assert "2.500" in content
    assert "3.000" in content
    assert "4.000" in content
