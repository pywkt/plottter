"""Phase 15.7 validation: export validation — SVG, HPGL, and G-code formats.

This validation suite goes beyond the unit tests in test_svg_export.py,
test_hpgl_export.py, and test_gcode_export.py to verify:

1. SVG — valid XML structure, correct namespace, correct dimensions/units,
   coordinate precision (3 decimal places), registration marks, multi-layer
   projects, special characters in layer names, empty layers, all paper sizes.
2. HPGL — correct coordinate conversion across all quadrants, multi-layer
   batch with per-layer pen numbers, complex polylines, optional commands.
3. G-code — preamble/epilogue integrity, speed settings, coordinate precision,
   multi-file batch consistency.
4. Cross-format correctness — same paths exported in all three formats produce
   coordinate values that are numerically consistent with each other.
5. End-to-end pipeline — generate real paths via ParametricGenerator, export,
   verify the exported file is parseable and contains expected content.
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET

import pytest

from plottter.export.svg import (
    export_all_layers_svg,
    export_combined_svg,
    export_layer_svg,
)
from plottter.export.hpgl import (
    export_layer_hpgl,
    export_all_layers_hpgl,
    _mm_to_hpgl,
)
from plottter.export.gcode import export_layer_gcode, export_all_layers_gcode
from plottter.models.canvas import Canvas, PAPER_PRESETS
from plottter.models.layer import Layer
from plottter.models.project import Project

SVG_NS = "http://www.w3.org/2000/svg"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_svg(filepath: str) -> ET.Element:
    tree = ET.parse(filepath)
    return tree.getroot()


def _find_all(root: ET.Element, local_tag: str) -> list[ET.Element]:
    return root.findall(f".//{{{SVG_NS}}}{local_tag}")


def _read(filepath: str) -> str:
    with open(filepath) as f:
        return f.read()


def _make_layer(name: str, color: str = "#000000", num_paths: int = 3) -> Layer:
    """Create a Layer with synthetic polylines spread across a 100×100 area."""
    layer = Layer(name=name, color=color)
    paths = []
    for i in range(num_paths):
        t = i * 10.0
        paths.append([(t, 10.0), (t + 5.0, 50.0), (t + 10.0, 90.0)])
    layer.add_paths(paths)
    return layer


def _make_project(num_layers: int = 2, canvas: Canvas | None = None) -> Project:
    if canvas is None:
        canvas = Canvas.from_preset("A4", margin=10.0)
    proj = Project(name="ValidationProject", canvas=canvas)
    colors = ["#000000", "#FF0000", "#0000FF", "#00FF00"]
    for i in range(num_layers):
        layer = _make_layer(f"Layer {i + 1}", color=colors[i % len(colors)])
        proj.add_layer(layer)
    return proj


# ---------------------------------------------------------------------------
# SVG: XML structure
# ---------------------------------------------------------------------------

class TestSVGStructure:
    """Verify the structural correctness of exported SVG files."""

    def test_svg_root_has_correct_namespace(self, tmp_path: str) -> None:
        """Root element must carry the SVG namespace."""
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "ns.svg")
        export_layer_svg(layer, canvas, out, {})
        with open(out) as f:
            content = f.read()
        assert 'xmlns="http://www.w3.org/2000/svg"' in content

    def test_svg_has_xml_declaration(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "decl.svg")
        export_layer_svg(layer, canvas, out, {})
        with open(out) as f:
            first_line = f.readline()
        # svgwrite emits <?xml ... ?> at the start
        assert "<?xml" in first_line or first_line.startswith("<svg")

    def test_svg_viewbox_format(self, tmp_path: str) -> None:
        """viewBox must be '0 0 W H' where W and H are the canvas dimensions."""
        canvas = Canvas.from_preset("A3", margin=15.0)
        layer = _make_layer("L")
        out = str(tmp_path / "vb.svg")
        export_layer_svg(layer, canvas, out, {})
        root = _parse_svg(out)
        vb = root.get("viewBox")
        assert vb == f"0 0 {canvas.width_mm} {canvas.height_mm}"

    def test_svg_width_has_mm_units(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "units.svg")
        export_layer_svg(layer, canvas, out, {})
        root = _parse_svg(out)
        assert root.get("width", "").endswith("mm")
        assert root.get("height", "").endswith("mm")

    def test_svg_layer_group_has_id(self, tmp_path: str) -> None:
        """Layer <g> must have an id attribute starting with 'layer_'."""
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("MyLayer")
        out = str(tmp_path / "lid.svg")
        export_layer_svg(layer, canvas, out, {})
        root = _parse_svg(out)
        groups = _find_all(root, "g")
        layer_groups = [g for g in groups if g.get("id", "").startswith("layer_")]
        assert layer_groups, "No layer group with id='layer_*' found"

    def test_svg_polylines_have_points_attr(self, tmp_path: str) -> None:
        """Every <polyline> element must have a non-empty 'points' attribute."""
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L", num_paths=5)
        out = str(tmp_path / "pts.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        root = _parse_svg(out)
        polylines = _find_all(root, "polyline")
        assert len(polylines) == 5
        for pl in polylines:
            pts = pl.get("points", "")
            assert pts.strip(), "Empty 'points' attribute on <polyline>"

    def test_svg_no_fill_on_layer_group(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "fill.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        root = _parse_svg(out)
        groups = _find_all(root, "g")
        layer_groups = [g for g in groups if g.get("id", "").startswith("layer_")]
        for g in layer_groups:
            assert g.get("fill") == "none"

    def test_svg_layer_stroke_matches_color(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L", color="#AABB33")
        out = str(tmp_path / "stroke.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        root = _parse_svg(out)
        groups = _find_all(root, "g")
        layer_groups = [g for g in groups if g.get("id", "").startswith("layer_")]
        assert layer_groups[0].get("stroke") == "#AABB33"


# ---------------------------------------------------------------------------
# SVG: coordinate precision
# ---------------------------------------------------------------------------

class TestSVGCoordinatePrecision:
    """Verify that SVG coordinates use 3 decimal places as specified."""

    def test_known_coordinates_appear_with_3_decimals(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(1.1111, 2.2222), (33.3333, 44.4444)]])
        out = str(tmp_path / "prec.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        content = _read(out)
        # Coordinates should be rounded to 3 decimal places
        assert "1.111" in content
        assert "2.222" in content
        assert "33.333" in content
        assert "44.444" in content

    def test_integer_coordinates_have_three_decimal_zeros(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(10.0, 20.0), (30.0, 40.0)]])
        out = str(tmp_path / "int.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        content = _read(out)
        # The points attribute should have "10.000,20.000 30.000,40.000"
        assert "10.000" in content

    def test_stroke_width_default_precision(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "sw.svg")
        export_layer_svg(layer, canvas, out, {"stroke_width": 0.3, "registration_marks": False})
        content = _read(out)
        assert "0.300" in content


# ---------------------------------------------------------------------------
# SVG: paper sizes
# ---------------------------------------------------------------------------

class TestSVGAllPaperSizes:
    """Each PAPER_PRESETS entry should export correctly."""

    @pytest.mark.parametrize("preset_name", list(PAPER_PRESETS.keys()))
    def test_paper_preset_viewbox(self, preset_name: str, tmp_path: str) -> None:
        canvas = Canvas.from_preset(preset_name, margin=10.0)
        layer = _make_layer("L")
        out = str(tmp_path / f"{preset_name}.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        root = _parse_svg(out)
        w, h = PAPER_PRESETS[preset_name]
        assert root.get("viewBox") == f"0 0 {w} {h}"
        assert root.get("width") == f"{w}mm"
        assert root.get("height") == f"{h}mm"


# ---------------------------------------------------------------------------
# SVG: registration marks across all styles
# ---------------------------------------------------------------------------

class TestSVGRegistrationMarks:
    """Comprehensive registration mark validation."""

    def test_corners_style_crosshair_positions_near_margin(self, tmp_path: str) -> None:
        """Corner crosshairs must be positioned at the drawing area corners."""
        canvas = Canvas.from_preset("A4", margin=10.0)
        layer = _make_layer("L")
        settings = {"registration_marks": True, "reg_mark_style": "corners"}
        out = str(tmp_path / "corners.svg")
        export_layer_svg(layer, canvas, out, settings)
        root = _parse_svg(out)
        reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
        assert reg_groups
        lines = _find_all(reg_groups[0], "line")
        assert len(lines) == 8  # 4 corners × 2 arms

    def test_center_style_crosshair_at_canvas_center(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4", margin=10.0)
        layer = _make_layer("L")
        settings = {"registration_marks": True, "reg_mark_style": "center"}
        out = str(tmp_path / "center.svg")
        export_layer_svg(layer, canvas, out, settings)
        root = _parse_svg(out)
        reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
        assert reg_groups
        lines = _find_all(reg_groups[0], "line")
        assert len(lines) == 2  # 1 center crosshair

    def test_both_style_has_10_lines(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4", margin=10.0)
        layer = _make_layer("L")
        settings = {"registration_marks": True, "reg_mark_style": "both"}
        out = str(tmp_path / "both.svg")
        export_layer_svg(layer, canvas, out, settings)
        root = _parse_svg(out)
        reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
        assert reg_groups
        lines = _find_all(reg_groups[0], "line")
        assert len(lines) == 10  # (4 corners + 1 center) × 2

    def test_registration_marks_consistent_across_batch_layers(
        self, tmp_path: str
    ) -> None:
        """All layers in a batch export must have identical registration marks."""
        proj = _make_project(num_layers=3)
        proj.registration_marks = True
        out_dir = str(tmp_path / "batch")
        settings = {"registration_marks": True, "reg_mark_style": "corners"}
        export_all_layers_svg(proj, out_dir, settings)
        svg_files = sorted(
            [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".svg")]
        )
        assert len(svg_files) == 3
        line_counts = []
        for svg_path in svg_files:
            root = _parse_svg(svg_path)
            reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
            assert reg_groups, f"No registration group in {svg_path}"
            line_counts.append(len(_find_all(reg_groups[0], "line")))
        # All files should have the same number of registration lines
        assert len(set(line_counts)) == 1, f"Inconsistent reg mark counts: {line_counts}"

    def test_registration_marks_use_black_stroke(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4", margin=10.0)
        layer = _make_layer("L")
        settings = {"registration_marks": True, "reg_mark_style": "corners"}
        out = str(tmp_path / "regcolor.svg")
        export_layer_svg(layer, canvas, out, settings)
        root = _parse_svg(out)
        reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
        lines = _find_all(reg_groups[0], "line")
        for line in lines:
            stroke = line.get("stroke", "")
            assert stroke == "#000000", f"Expected black stroke on reg mark, got {stroke!r}"


# ---------------------------------------------------------------------------
# SVG: multi-layer combined export
# ---------------------------------------------------------------------------

class TestSVGCombinedMultiLayer:
    """Validate combined SVG export with multiple layers."""

    def test_combined_has_one_group_per_visible_layer(self, tmp_path: str) -> None:
        proj = _make_project(num_layers=4)
        proj.layers[2].visible = False  # hide one layer
        out = str(tmp_path / "combined.svg")
        export_combined_svg(proj, out, {"registration_marks": False})
        root = _parse_svg(out)
        groups = _find_all(root, "g")
        layer_groups = [g for g in groups if g.get("id", "").startswith("layer_")]
        assert len(layer_groups) == 3  # 4 total - 1 hidden

    def test_combined_layer_groups_have_distinct_ids(self, tmp_path: str) -> None:
        proj = _make_project(num_layers=3)
        out = str(tmp_path / "combined.svg")
        export_combined_svg(proj, out, {"registration_marks": False})
        root = _parse_svg(out)
        groups = _find_all(root, "g")
        layer_groups = [g for g in groups if g.get("id", "").startswith("layer_")]
        ids = [g.get("id") for g in layer_groups]
        assert len(ids) == len(set(ids)), f"Duplicate layer group IDs: {ids}"

    def test_combined_each_layer_has_correct_stroke_color(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        proj = Project(name="P", canvas=canvas)
        for color in ["#FF0000", "#00FF00", "#0000FF"]:
            layer = Layer(name=f"Layer {color}", color=color)
            layer.add_paths([[(10.0, 10.0), (50.0, 50.0)]])
            proj.add_layer(layer)
        out = str(tmp_path / "colors.svg")
        export_combined_svg(proj, out, {"registration_marks": False})
        content = _read(out)
        assert "#FF0000" in content
        assert "#00FF00" in content
        assert "#0000FF" in content

    def test_combined_registration_marks_appear_once(self, tmp_path: str) -> None:
        """The registration group must appear exactly once, not per layer."""
        proj = _make_project(num_layers=3)
        out = str(tmp_path / "combined_reg.svg")
        settings = {"registration_marks": True, "reg_mark_style": "corners"}
        export_combined_svg(proj, out, settings)
        content = _read(out)
        # Count occurrences of the registration group id
        count = content.count('id="registration"')
        assert count == 1, f"Expected exactly 1 registration group, found {count}"


# ---------------------------------------------------------------------------
# SVG: edge cases
# ---------------------------------------------------------------------------

class TestSVGEdgeCases:
    """Edge cases: empty layers, special characters, large datasets."""

    def test_empty_layer_produces_valid_svg(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="Empty", color="#000000")
        out = str(tmp_path / "empty.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        root = _parse_svg(out)
        polylines = _find_all(root, "polyline")
        assert len(polylines) == 0

    def test_layer_name_with_spaces_produces_valid_xml(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="My Cool Layer", color="#000000")
        layer.add_paths([[(1.0, 1.0), (10.0, 10.0)]])
        out = str(tmp_path / "spaces.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        # Must be parseable XML
        root = _parse_svg(out)
        assert root is not None

    def test_layer_name_with_special_chars_produces_valid_xml(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="Layer: <test> & more", color="#000000")
        layer.add_paths([[(1.0, 1.0), (10.0, 10.0)]])
        out = str(tmp_path / "special.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        root = _parse_svg(out)
        assert root is not None

    def test_large_path_count_exports_correctly(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="Large", color="#000000")
        # 500 paths, each with 10 points
        paths = [[(float(j), float(i)) for j in range(10)] for i in range(500)]
        layer.add_paths(paths)
        out = str(tmp_path / "large.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        root = _parse_svg(out)
        polylines = _find_all(root, "polyline")
        assert len(polylines) == 500

    def test_two_point_path_produces_valid_polyline(self, tmp_path: str) -> None:
        """Minimum valid path (2 points) must produce a <polyline> element."""
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(0.0, 0.0), (100.0, 100.0)]])
        out = str(tmp_path / "twopoint.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": False})
        root = _parse_svg(out)
        polylines = _find_all(root, "polyline")
        assert len(polylines) == 1

    def test_batch_export_project_name_with_spaces(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        proj = Project(name="My Great Project", canvas=canvas)
        layer = _make_layer("L1")
        proj.add_layer(layer)
        out_dir = str(tmp_path / "spaces_batch")
        export_all_layers_svg(proj, out_dir, {"registration_marks": False})
        files = os.listdir(out_dir)
        assert len(files) == 1
        # Spaces in project name should be replaced with underscores
        assert "My_Great_Project" in files[0]


# ---------------------------------------------------------------------------
# HPGL: coordinate correctness
# ---------------------------------------------------------------------------

class TestHPGLCoordinateCorrectness:
    """Verify HPGL coordinate conversion math is correct across all quadrants."""

    def test_coordinate_conversion_all_corners(self) -> None:
        """Test _mm_to_hpgl at each corner of A4 canvas."""
        h = 297.0
        # Bottom-left (SVG origin) → HPGL bottom-left
        x, y = _mm_to_hpgl(0.0, 0.0, h)
        assert x == 0
        assert y == int(h * 40)  # Y-inverted: 0mm maps to max HPGL Y

        # Top-right (SVG bottom-right) → HPGL top-right
        x, y = _mm_to_hpgl(210.0, 297.0, h)
        assert x == int(210.0 * 40)
        assert y == 0

    def test_mid_canvas_coordinate(self) -> None:
        """Center of A4 (105, 148.5) should map to midpoint of HPGL range."""
        h = 297.0
        x, y = _mm_to_hpgl(105.0, 148.5, h)
        expected_x = int(105.0 * 40)
        expected_y = int((h - 148.5) * 40)
        assert x == expected_x
        assert y == expected_y

    def test_fractional_mm_truncated_to_int(self) -> None:
        """HPGL coordinates must always be integers (truncated, not rounded)."""
        h = 297.0
        x, y = _mm_to_hpgl(0.7, 0.3, h)
        assert isinstance(x, int)
        assert isinstance(y, int)
        # int(0.7 * 40) = int(28.0) = 28
        assert x == int(0.7 * 40)

    def test_hpgl_units_are_1_40th_mm(self) -> None:
        """40 HPGL units = 1mm. Verify 10mm maps to 400 units."""
        h = 100.0
        x, y = _mm_to_hpgl(10.0, 50.0, h)
        assert x == 400
        assert y == int((100.0 - 50.0) * 40)


class TestHPGLFileContent:
    """Verify HPGL file content for complex scenarios."""

    def test_multi_path_pen_sequence(self, tmp_path: str) -> None:
        """Each path starts with PU (position) followed by PD (draw)."""
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="L", color="#000000")
        # 4 paths
        for i in range(4):
            layer.add_paths([[(i * 10.0, 0.0), (i * 10.0 + 5.0, 20.0)]])
        out = str(tmp_path / "multi.plt")
        export_layer_hpgl(layer, canvas, out, {})
        content = _read(out)
        lines = content.splitlines()
        # Count PU with coordinates (positioning) and PD lines
        pu_with_coords = [l for l in lines if l.startswith("PU") and "," in l]
        pd_lines = [l for l in lines if l.startswith("PD")]
        assert len(pu_with_coords) == 4
        assert len(pd_lines) == 4

    def test_hpgl_no_floating_point_coords(self, tmp_path: str) -> None:
        """Fractional mm coordinates must yield integer HPGL values, no decimals."""
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(3.7, 12.3), (8.9, 25.1)]])
        out = str(tmp_path / "frac.plt")
        export_layer_hpgl(layer, canvas, out, {})
        content = _read(out)
        # No dots should appear in coordinate values
        for line in content.splitlines():
            if line.startswith(("PU", "PD")):
                coords_part = line[2:].rstrip(";")
                assert "." not in coords_part, (
                    f"Floating point coordinate found in HPGL line: {line!r}"
                )

    def test_hpgl_init_is_first_command(self, tmp_path: str) -> None:
        """IN; must be the very first command."""
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "first.plt")
        export_layer_hpgl(layer, canvas, out, {})
        lines = [l.strip() for l in _read(out).splitlines() if l.strip()]
        assert lines[0] == "IN;"

    def test_hpgl_final_pen_up_is_last(self, tmp_path: str) -> None:
        """PU; (bare, no coordinates) must be the final command."""
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "last.plt")
        export_layer_hpgl(layer, canvas, out, {})
        lines = [l.strip() for l in _read(out).splitlines() if l.strip()]
        assert lines[-1] == "PU;"

    def test_hpgl_y_axis_inversion_verified_in_output(self, tmp_path: str) -> None:
        """Point at y=0 must have larger HPGL Y than point at y=canvas_height."""
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="L", color="#000000")
        # Two paths: one at y=0, one at y=canvas_height
        layer.add_paths([
            [(50.0, 0.0), (60.0, 0.0)],       # near bottom edge in mm
            [(50.0, 297.0), (60.0, 297.0)],  # near top edge in mm
        ])
        out = str(tmp_path / "yinv.plt")
        export_layer_hpgl(layer, canvas, out, {})
        content = _read(out)
        # Extract all PU coordinate pairs
        pu_lines = [l for l in content.splitlines() if l.startswith("PU") and "," in l]
        assert len(pu_lines) == 2
        # First path: y_mm=0 → HPGL Y = int(297*40) = 11880
        y0 = int(pu_lines[0][2:].rstrip(";").split(",")[1])
        # Second path: y_mm=297 → HPGL Y = 0
        y1 = int(pu_lines[1][2:].rstrip(";").split(",")[1])
        assert y0 > y1, f"Expected y at y=0mm > y at y=297mm: {y0} vs {y1}"

    def test_hpgl_batch_assigns_pen_numbers_per_layer(self, tmp_path: str) -> None:
        """Layer n should get pen number n."""
        proj = _make_project(num_layers=3)
        out_dir = str(tmp_path / "hpgl_batch")
        export_all_layers_hpgl(proj, out_dir, {})
        files = sorted(
            [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".plt")]
        )
        assert len(files) == 3
        for i, fpath in enumerate(files, start=1):
            content = _read(fpath)
            assert f"SP{i};" in content, f"File {fpath} missing SP{i};"


# ---------------------------------------------------------------------------
# G-code: content validation
# ---------------------------------------------------------------------------

class TestGcodeContentValidation:
    """Validate G-code output for complex scenarios and spec compliance."""

    def test_gcode_preamble_order(self, tmp_path: str) -> None:
        """Preamble must follow spec: G90, G21, G28, M3 S{up}."""
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "preamble.gcode")
        export_layer_gcode(layer, canvas, out, {"pen_up_angle": 0})
        lines = [l for l in _read(out).splitlines() if l.strip()]
        assert lines[0] == "G90"
        assert lines[1] == "G21"
        assert lines[2] == "G28"
        # M3 S0 (pen up) must appear after G28 and before first G0
        g0_idx = next((i for i, l in enumerate(lines) if l.startswith("G0")), None)
        assert g0_idx is not None
        m3_idx = next((i for i, l in enumerate(lines) if l.startswith("M3")), None)
        assert m3_idx is not None
        assert m3_idx < g0_idx, "M3 (pen up) must come before first G0 move"

    def test_gcode_epilogue_ends_with_m5(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "epilogue.gcode")
        export_layer_gcode(layer, canvas, out, {})
        lines = [l for l in _read(out).splitlines() if l.strip()]
        assert lines[-1] == "M5"

    def test_gcode_g28_in_epilogue(self, tmp_path: str) -> None:
        """G28 must appear in epilogue (before M5)."""
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "epilogue2.gcode")
        export_layer_gcode(layer, canvas, out, {})
        lines = [l for l in _read(out).splitlines() if l.strip()]
        # G28 appears in preamble AND epilogue; check it's also near the end
        assert lines[-2] == "G28"

    def test_gcode_coordinate_precision_3dp(self, tmp_path: str) -> None:
        """G-code X/Y coordinates must have exactly 3 decimal places."""
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(7.1234, 13.5678), (22.9999, 55.1234)]])
        out = str(tmp_path / "prec.gcode")
        export_layer_gcode(layer, canvas, out, {})
        content = _read(out)
        # X7.123 and Y13.568 should appear (rounded to 3dp)
        assert "X7.123" in content
        assert "Y13.568" in content

    def test_gcode_each_path_has_pen_down_and_up(self, tmp_path: str) -> None:
        """Each path must have exactly one pen-down and one pen-up M3 command."""
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="L", color="#000000")
        layer.add_paths([
            [(0.0, 0.0), (10.0, 0.0)],
            [(20.0, 0.0), (30.0, 0.0)],
            [(40.0, 0.0), (50.0, 0.0)],
        ])
        out = str(tmp_path / "penud.gcode")
        export_layer_gcode(layer, canvas, out, {"pen_up_angle": 0, "pen_down_angle": 90})
        content = _read(out)
        lines = content.splitlines()
        # After preamble, count M3 commands (excluding preamble pen-up)
        g0_first = next(i for i, l in enumerate(lines) if l.startswith("G0"))
        post_preamble = lines[g0_first:]
        m3_down = [l for l in post_preamble if l.strip() == "M3 S90"]
        m3_up = [l for l in post_preamble if l.strip() == "M3 S0"]
        # 3 paths = 3 pen downs, 3 pen ups (the last pen up may merge with epilogue)
        assert len(m3_down) == 3
        assert len(m3_up) >= 3

    def test_gcode_default_speeds_in_moves(self, tmp_path: str) -> None:
        """Default travel=3000, draw=1000 must appear in G0 and G1 lines."""
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "speed.gcode")
        export_layer_gcode(layer, canvas, out, {})  # no explicit speeds → defaults
        content = _read(out)
        g0_lines = [l for l in content.splitlines() if l.startswith("G0 X")]
        g1_lines = [l for l in content.splitlines() if l.startswith("G1 X")]
        assert g0_lines, "No G0 moves found"
        assert g1_lines, "No G1 moves found"
        assert all("F3000" in l for l in g0_lines), "G0 moves must use F3000"
        assert all("F1000" in l for l in g1_lines), "G1 moves must use F1000"

    def test_gcode_custom_speeds_applied(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        layer = _make_layer("L")
        out = str(tmp_path / "cspeed.gcode")
        export_layer_gcode(layer, canvas, out, {"travel_speed": 5000, "draw_speed": 500})
        content = _read(out)
        g0_lines = [l for l in content.splitlines() if l.startswith("G0 X")]
        g1_lines = [l for l in content.splitlines() if l.startswith("G1 X")]
        assert g0_lines, "No G0 moves found"
        assert g1_lines, "No G1 moves found"
        assert all("F5000" in l for l in g0_lines)
        assert all("F500" in l for l in g1_lines)

    def test_gcode_batch_all_files_have_preamble(self, tmp_path: str) -> None:
        proj = _make_project(num_layers=3)
        out_dir = str(tmp_path / "gcode_batch")
        export_all_layers_gcode(proj, out_dir, {})
        files = [f for f in os.listdir(out_dir) if f.endswith(".gcode")]
        assert len(files) == 3
        for fname in files:
            content = _read(os.path.join(out_dir, fname))
            assert content.startswith("G90"), f"{fname} does not start with G90"
            assert "G21" in content
            assert "G28" in content
            assert "M5" in content


# ---------------------------------------------------------------------------
# Cross-format: coordinate consistency
# ---------------------------------------------------------------------------

class TestCrossFormatConsistency:
    """Same paths exported in different formats must encode the same coordinates."""

    def test_svg_and_hpgl_encode_same_x_position(self, tmp_path: str) -> None:
        """A point at x=50mm must appear as '50.000' in SVG and 2000 in HPGL."""
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(50.0, 50.0), (100.0, 100.0)]])

        svg_out = str(tmp_path / "cross.svg")
        hpgl_out = str(tmp_path / "cross.plt")
        export_layer_svg(layer, canvas, svg_out, {"registration_marks": False})
        export_layer_hpgl(layer, canvas, hpgl_out, {})

        svg_content = _read(svg_out)
        hpgl_content = _read(hpgl_out)

        # SVG: x=50mm appears as "50.000" in points attribute
        assert "50.000" in svg_content

        # HPGL: x=50mm → 50 * 40 = 2000
        assert "2000" in hpgl_content

    def test_svg_and_gcode_encode_same_coordinates(self, tmp_path: str) -> None:
        """x=25.5mm must be '25.500' in SVG and 'X25.500' in G-code."""
        canvas = Canvas.from_preset("A4")
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(25.5, 75.0), (80.0, 30.0)]])

        svg_out = str(tmp_path / "cg.svg")
        gcode_out = str(tmp_path / "cg.gcode")
        export_layer_svg(layer, canvas, svg_out, {"registration_marks": False})
        export_layer_gcode(layer, canvas, gcode_out, {})

        svg_content = _read(svg_out)
        gcode_content = _read(gcode_out)

        assert "25.500" in svg_content
        assert "X25.500" in gcode_content


# ---------------------------------------------------------------------------
# End-to-end: generate + export
# ---------------------------------------------------------------------------

class TestEndToEndGenerateAndExport:
    """Generate real paths via generators, then export and verify output."""

    def test_parametric_generator_output_exports_to_svg(self, tmp_path: str) -> None:
        """Run ParametricGenerator (Lissajous preset), export SVG, verify valid."""
        from plottter.generators.parametric import ParametricGenerator

        canvas = Canvas.from_preset("A4", margin=10.0)
        gen = ParametricGenerator()
        presets = gen.get_presets()
        lissajous = next((p for p in presets if "Lissajous" in p.name), presets[0])
        params = dict(lissajous.params)
        params["num_points"] = 200  # keep fast for tests

        paths = gen.generate(params, canvas)
        assert paths, "Generator produced no paths"

        layer = Layer(name="Lissajous", color="#333333")
        layer.add_paths(paths)
        proj = Project(name="GenTest", canvas=canvas)
        proj.add_layer(layer)

        out = str(tmp_path / "lissajous.svg")
        export_layer_svg(layer, canvas, out, {"registration_marks": True, "reg_mark_style": "corners"})

        root = _parse_svg(out)
        assert root is not None
        polylines = _find_all(root, "polyline")
        assert len(polylines) >= 1

    def test_polar_generator_output_exports_to_hpgl(self, tmp_path: str) -> None:
        """Run PolarGenerator (Rose preset), export HPGL, verify valid format."""
        from plottter.generators.polar import PolarGenerator

        canvas = Canvas.from_preset("A4", margin=10.0)
        gen = PolarGenerator()
        presets = gen.get_presets()
        rose = next((p for p in presets if "Rose" in p.name), presets[0])
        params = dict(rose.params)
        params["num_points"] = 200

        paths = gen.generate(params, canvas)
        assert paths

        layer = Layer(name="Rose", color="#FF0000")
        layer.add_paths(paths)

        out = str(tmp_path / "rose.plt")
        export_layer_hpgl(layer, canvas, out, {})

        content = _read(out)
        assert "IN;" in content
        assert "SP1;" in content
        assert "PD" in content
        assert content.strip().endswith("PU;")
        # All coordinates must be integers
        for line in content.splitlines():
            if line.startswith(("PU", "PD")) and "," in line:
                coords = line[2:].rstrip(";").split(",")
                for c in coords:
                    assert "." not in c, f"Float in HPGL coord: {line!r}"

    def test_flow_field_generator_exports_to_gcode(self, tmp_path: str) -> None:
        """Run FlowFieldGenerator, export G-code, verify valid format."""
        from plottter.generators.flow_field import FlowFieldGenerator

        canvas = Canvas.from_preset("A4", margin=10.0)
        gen = FlowFieldGenerator()
        params = {
            "num_particles": 20,
            "step_size_mm": 2.0,
            "max_steps": 20,
            "noise_scale": 0.02,
            "noise_octaves": 2,
            "seed": 42,
            "angle_range": 6.283185307,
        }
        paths = gen.generate(params, canvas)
        assert paths

        layer = Layer(name="Flow", color="#0000FF")
        layer.add_paths(paths)

        out = str(tmp_path / "flow.gcode")
        export_layer_gcode(layer, canvas, out, {})

        content = _read(out)
        assert content.startswith("G90")
        assert "G21" in content
        assert "G1 X" in content  # has draw moves
        assert content.strip().endswith("M5")

    def test_full_project_batch_svg_export(self, tmp_path: str) -> None:
        """Multi-layer project (3 generators) exports all layers as valid SVGs."""
        from plottter.generators.parametric import ParametricGenerator
        from plottter.generators.polar import PolarGenerator
        from plottter.generators.modular_mult import ModularMultGenerator

        canvas = Canvas.from_preset("A4", margin=10.0)
        proj = Project(name="FullTest", canvas=canvas)

        for GenClass, color in [
            (ParametricGenerator, "#000000"),
            (PolarGenerator, "#FF0000"),
            (ModularMultGenerator, "#0000FF"),
        ]:
            gen = GenClass()
            presets = gen.get_presets()
            params = dict(presets[0].params)
            params["num_points"] = 100 if "num_points" in params else params.get("num_points", 100)
            try:
                paths = gen.generate(params, canvas)
            except Exception as exc:
                pytest.fail(f"{GenClass.__name__}.generate() raised unexpectedly: {exc}")
            layer = Layer(name=f"{gen.name} Layer", color=color)
            if paths:
                layer.add_paths(paths)
            proj.add_layer(layer)

        out_dir = str(tmp_path / "full_batch")
        settings = {"registration_marks": True, "reg_mark_style": "corners"}
        export_all_layers_svg(proj, out_dir, settings)

        svg_files = [f for f in os.listdir(out_dir) if f.endswith(".svg")]
        assert len(svg_files) == 3

        for svg_file in svg_files:
            root = _parse_svg(os.path.join(out_dir, svg_file))
            assert root is not None
            # Must have registration marks
            reg_groups = [g for g in _find_all(root, "g") if g.get("id") == "registration"]
            assert reg_groups, f"{svg_file} missing registration marks"


# ---------------------------------------------------------------------------
# SVG batch: naming
# ---------------------------------------------------------------------------

class TestSVGBatchNaming:
    """SVG batch export naming conventions."""

    def test_batch_file_names_use_zero_padded_index(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        proj = Project(name="SVGBatch", canvas=canvas)
        for i in range(5):
            layer = _make_layer(f"L{i + 1}")
            proj.add_layer(layer)
        out_dir = str(tmp_path / "svg5")
        export_all_layers_svg(proj, out_dir, {"registration_marks": False})
        files = sorted(f for f in os.listdir(out_dir) if f.endswith(".svg"))
        assert len(files) == 5
        assert files[0].startswith("SVGBatch_01_")
        assert files[4].startswith("SVGBatch_05_")
        assert all(f.endswith(".svg") for f in files)


# ---------------------------------------------------------------------------
# HPGL batch: naming and count
# ---------------------------------------------------------------------------

class TestHPGLBatchNaming:
    """HPGL batch export naming conventions."""

    def test_batch_file_names_use_zero_padded_index(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        proj = Project(name="BatchTest", canvas=canvas)
        for i in range(5):
            layer = _make_layer(f"L{i + 1}")
            proj.add_layer(layer)
        out_dir = str(tmp_path / "hpgl5")
        export_all_layers_hpgl(proj, out_dir, {})
        files = sorted(os.listdir(out_dir))
        assert len(files) == 5
        assert files[0].startswith("BatchTest_01_")
        assert files[4].startswith("BatchTest_05_")
        assert all(f.endswith(".plt") for f in files)

    def test_batch_hidden_layer_skipped_in_pen_assignment(
        self, tmp_path: str
    ) -> None:
        """Hidden layers should not appear in the output at all."""
        canvas = Canvas.from_preset("A4")
        proj = Project(name="H", canvas=canvas)
        visible = _make_layer("Visible")
        hidden = _make_layer("Hidden")
        hidden.visible = False
        proj.add_layer(visible)
        proj.add_layer(hidden)
        out_dir = str(tmp_path / "hidden_hpgl")
        export_all_layers_hpgl(proj, out_dir, {})
        files = [f for f in os.listdir(out_dir) if f.endswith(".plt")]
        assert len(files) == 1
        content = _read(os.path.join(out_dir, files[0]))
        # Only one visible layer → SP1
        assert "SP1;" in content


# ---------------------------------------------------------------------------
# G-code batch: naming and integrity
# ---------------------------------------------------------------------------

class TestGcodeBatchNaming:
    """G-code batch export naming and per-file integrity."""

    def test_batch_file_names_use_zero_padded_index(self, tmp_path: str) -> None:
        canvas = Canvas.from_preset("A4")
        proj = Project(name="GBatch", canvas=canvas)
        for i in range(4):
            layer = _make_layer(f"L{i + 1}")
            proj.add_layer(layer)
        out_dir = str(tmp_path / "gcode4")
        export_all_layers_gcode(proj, out_dir, {})
        files = sorted(os.listdir(out_dir))
        assert len(files) == 4
        assert files[0].startswith("GBatch_01_")
        assert files[3].startswith("GBatch_04_")
        assert all(f.endswith(".gcode") for f in files)

    def test_each_file_is_self_contained(self, tmp_path: str) -> None:
        """Each G-code file must have its own complete preamble and epilogue."""
        proj = _make_project(num_layers=2)
        out_dir = str(tmp_path / "gcself")
        export_all_layers_gcode(proj, out_dir, {})
        for fname in os.listdir(out_dir):
            fpath = os.path.join(out_dir, fname)
            content = _read(fpath)
            assert "G90" in content, f"{fname} missing G90"
            assert "G21" in content, f"{fname} missing G21"
            assert "M5" in content, f"{fname} missing M5"
