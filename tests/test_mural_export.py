"""Tests for Mural plotter export (Phase 16.26)."""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.project import Project
from plottter.export.mural import (
    export_layer_mural,
    export_all_layers_mural,
    _build_mural_content,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def canvas() -> Canvas:
    return Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0, paper_preset="A4")


@pytest.fixture
def layer_with_paths() -> Layer:
    layer = Layer(name="Test Layer", color="#FF0000")
    layer.add_paths([
        [(10.0, 20.0), (50.0, 20.0), (50.0, 80.0)],
        [(100.0, 100.0), (150.0, 150.0)],
    ])
    return layer


@pytest.fixture
def project_with_layers(canvas: Canvas) -> Project:
    project = Project(name="Test Project", canvas=canvas)
    layer1 = Layer(name="Layer 1", color="#000000")
    layer1.add_paths([[(0.0, 0.0), (10.0, 10.0)]])
    layer2 = Layer(name="Layer 2", color="#FF0000")
    layer2.add_paths([[(20.0, 20.0), (30.0, 30.0)]])
    project.add_layer(layer1)
    project.add_layer(layer2)
    return project


TOP_DISTANCE = 1025.0  # mm between anchor pins
WIDTH = TOP_DISTANCE * 0.6  # 615 mm


# ---------------------------------------------------------------------------
# Header format
# ---------------------------------------------------------------------------

class TestMuralHeader:
    def test_first_line_is_distance_header(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        content, _ = _build_mural_content(layer_with_paths, canvas, TOP_DISTANCE)
        lines = content.splitlines()
        assert lines[0].startswith("d")

    def test_second_line_is_height_header(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        content, _ = _build_mural_content(layer_with_paths, canvas, TOP_DISTANCE)
        lines = content.splitlines()
        assert lines[1].startswith("h")

    def test_height_header_matches_canvas_height(
        self, layer_with_paths: Layer, canvas: Canvas
    ) -> None:
        content, _ = _build_mural_content(layer_with_paths, canvas, TOP_DISTANCE)
        lines = content.splitlines()
        h_value = float(lines[1][1:])
        assert h_value == pytest.approx(canvas.height_mm, abs=0.01)

    def test_distance_header_is_non_negative(
        self, layer_with_paths: Layer, canvas: Canvas
    ) -> None:
        content, _ = _build_mural_content(layer_with_paths, canvas, TOP_DISTANCE)
        d_value = float(content.splitlines()[0][1:])
        assert d_value >= 0.0


# ---------------------------------------------------------------------------
# Pen commands and coordinate format
# ---------------------------------------------------------------------------

class TestMuralCommands:
    def test_pen_up_before_each_path(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        content, _ = _build_mural_content(layer_with_paths, canvas, TOP_DISTANCE)
        lines = content.splitlines()[2:]  # skip headers
        p0_indices = [i for i, ln in enumerate(lines) if ln == "p0"]
        # Should have p0 before each path plus one at the end
        assert len(p0_indices) >= 2

    def test_pen_down_after_move_to_start(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        content, _ = _build_mural_content(layer_with_paths, canvas, TOP_DISTANCE)
        lines = content.splitlines()[2:]
        p1_indices = [i for i, ln in enumerate(lines) if ln == "p1"]
        assert len(p1_indices) == 2  # one p1 per path

    def test_file_ends_with_pen_up(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        content, _ = _build_mural_content(layer_with_paths, canvas, TOP_DISTANCE)
        last_non_empty = content.rstrip().splitlines()[-1]
        assert last_non_empty == "p0"

    def test_coordinate_lines_have_two_space_separated_floats(
        self, layer_with_paths: Layer, canvas: Canvas
    ) -> None:
        content, _ = _build_mural_content(layer_with_paths, canvas, TOP_DISTANCE)
        for line in content.splitlines()[2:]:
            if line in ("p0", "p1"):
                continue
            parts = line.split()
            assert len(parts) == 2, f"Expected 2 parts in coord line: {line!r}"
            float(parts[0])
            float(parts[1])

    def test_sequence_p0_move_p1_draw_p0(self, canvas: Canvas) -> None:
        """Verify the canonical draw-sequence for a single-path layer."""
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)]])
        content, _ = _build_mural_content(layer, canvas, TOP_DISTANCE)
        body = content.splitlines()[2:]
        assert body[0] == "p0"                  # pen up
        assert body[1] not in ("p0", "p1")      # move to start
        assert body[2] == "p1"                  # pen down
        assert body[3] not in ("p0", "p1")      # draw point
        assert body[4] not in ("p0", "p1")      # draw point
        assert body[5] == "p0"                  # pen up at end


# ---------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------

class TestMuralCoordinates:
    def test_x_offset_applied(self, canvas: Canvas) -> None:
        """Drawing coordinates should be offset so canvas is centred on home."""
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(0.0, 0.0), (canvas.width_mm, canvas.height_mm)]])
        content, _ = _build_mural_content(layer, canvas, TOP_DISTANCE)
        body = content.splitlines()[2:]
        coord_lines = [ln for ln in body if ln not in ("p0", "p1")]
        x_offset = (WIDTH - canvas.width_mm) / 2.0
        first_x = float(coord_lines[0].split()[0])
        assert first_x == pytest.approx(x_offset, abs=0.2)

    def test_y_offset_applied(self, canvas: Canvas) -> None:
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(0.0, 0.0), (10.0, 10.0)]])
        content, _ = _build_mural_content(layer, canvas, TOP_DISTANCE)
        body = content.splitlines()[2:]
        coord_lines = [ln for ln in body if ln not in ("p0", "p1")]
        y_offset = 350.0 - canvas.height_mm / 2.0
        first_y = float(coord_lines[0].split()[1])
        assert first_y == pytest.approx(y_offset, abs=0.2)

    def test_coordinates_rounded_to_one_decimal(self, canvas: Canvas) -> None:
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(1.234, 5.678), (9.999, 8.001)]])
        content, _ = _build_mural_content(layer, canvas, TOP_DISTANCE)
        for line in content.splitlines()[2:]:
            if line in ("p0", "p1"):
                continue
            for part in line.split():
                # Should have at most 1 decimal place
                if "." in part:
                    decimals = len(part.split(".")[1])
                    assert decimals <= 1, f"Too many decimal places in: {part}"

    def test_different_top_distance_shifts_offset(self, canvas: Canvas) -> None:
        """Larger pin distance → larger drawing width → different x offset."""
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(0.0, 0.0), (10.0, 10.0)]])
        content_1025, _ = _build_mural_content(layer, canvas, 1025.0)
        content_800, _ = _build_mural_content(layer, canvas, 800.0)

        def first_coord_x(content: str) -> float:
            for line in content.splitlines()[2:]:
                if line not in ("p0", "p1"):
                    return float(line.split()[0])
            raise ValueError("No coordinate line found")

        x1 = first_coord_x(content_1025)
        x2 = first_coord_x(content_800)
        assert x1 != x2


# ---------------------------------------------------------------------------
# Distance header accuracy
# ---------------------------------------------------------------------------

class TestMuralDistanceHeader:
    def test_distance_matches_euclidean_sum(self, canvas: Canvas) -> None:
        """Total distance header should equal sum of Euclidean segments."""
        layer = Layer(name="L", color="#000000")
        pts = [(10.0, 10.0), (20.0, 10.0), (20.0, 25.0)]
        layer.add_paths([pts])
        content, _ = _build_mural_content(layer, canvas, TOP_DISTANCE)
        reported = float(content.splitlines()[0][1:])

        # Compute expected: home→first_coord + path segments
        width = TOP_DISTANCE * 0.6
        home_x, home_y = width / 2, 350.0
        x_off = (width - canvas.width_mm) / 2
        y_off = 350.0 - canvas.height_mm / 2

        mx = [x_off + pt[0] for pt in pts]
        my = [y_off + pt[1] for pt in pts]

        expected = math.sqrt((mx[0] - home_x) ** 2 + (my[0] - home_y) ** 2)
        for i in range(len(mx) - 1):
            expected += math.sqrt((mx[i + 1] - mx[i]) ** 2 + (my[i + 1] - my[i]) ** 2)

        assert reported == pytest.approx(expected, abs=0.2)

    def test_empty_layer_distance_is_zero(self, canvas: Canvas) -> None:
        layer = Layer(name="L", color="#000000")
        content, _ = _build_mural_content(layer, canvas, TOP_DISTANCE)
        d_value = float(content.splitlines()[0][1:])
        assert d_value == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Out-of-bounds warnings
# ---------------------------------------------------------------------------

class TestMuralWarnings:
    def test_no_warnings_for_valid_coordinates(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        _, warnings = _build_mural_content(layer_with_paths, canvas, TOP_DISTANCE)
        assert warnings == []

    def test_warning_for_negative_x(self, canvas: Canvas) -> None:
        """A very large canvas width that pushes coords out of the drawing area."""
        big_canvas = Canvas(width_mm=700.0, height_mm=100.0, margin_mm=0.0)
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(0.0, 0.0), (700.0, 50.0)]])
        _, warnings = _build_mural_content(layer, big_canvas, TOP_DISTANCE)
        # Width = 615, so a 700mm-wide canvas will push left edge negative
        assert len(warnings) > 0

    def test_warning_for_negative_y(self, canvas: Canvas) -> None:
        """A path that, after y-offset, lands above y=0."""
        layer = Layer(name="L", color="#000000")
        # y_offset = 350 - 297/2 ≈ 201.5, so a point at y_mm=0 → Mural y=201.5 (fine)
        # To trigger negative y we need a negative canvas y (below top of canvas in mm)
        layer.add_paths([[(-300.0, -300.0), (-250.0, -250.0)]])
        _, warnings = _build_mural_content(layer, canvas, TOP_DISTANCE)
        # This pushes mural_y below 0
        assert len(warnings) > 0

    def test_warnings_describe_coordinates(self, canvas: Canvas) -> None:
        big_canvas = Canvas(width_mm=700.0, height_mm=100.0, margin_mm=0.0)
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(0.0, 0.0), (700.0, 50.0)]])
        _, warnings = _build_mural_content(layer, big_canvas, TOP_DISTANCE)
        assert any("drawing area" in w for w in warnings)


# ---------------------------------------------------------------------------
# Single-point paths skipped
# ---------------------------------------------------------------------------

class TestMuralEdgeCases:
    def test_single_point_path_skipped(self, canvas: Canvas) -> None:
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(5.0, 5.0)]])
        content, _ = _build_mural_content(layer, canvas, TOP_DISTANCE)
        # No p1 (pen-down) should appear
        assert "p1" not in content

    def test_empty_layer_produces_only_headers_and_final_p0(self, canvas: Canvas) -> None:
        layer = Layer(name="L", color="#000000")
        content, _ = _build_mural_content(layer, canvas, TOP_DISTANCE)
        lines = [ln for ln in content.splitlines() if ln]
        assert lines[0].startswith("d")
        assert lines[1].startswith("h")
        assert lines[2] == "p0"
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

class TestMuralFileExport:
    def test_creates_file(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mural", delete=False) as f:
            path = f.name
        try:
            export_layer_mural(layer_with_paths, canvas, path, {"top_distance": TOP_DISTANCE})
            assert os.path.isfile(path)
        finally:
            os.unlink(path)

    def test_returns_warnings_list(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mural", delete=False) as f:
            path = f.name
        try:
            result = export_layer_mural(layer_with_paths, canvas, path, {"top_distance": TOP_DISTANCE})
            assert isinstance(result, list)
        finally:
            os.unlink(path)

    def test_default_top_distance(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        """Not specifying top_distance should default to 1025."""
        with tempfile.NamedTemporaryFile(suffix=".mural", delete=False) as f:
            path = f.name
        try:
            export_layer_mural(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            assert content.startswith("d")
        finally:
            os.unlink(path)

    def test_file_content_matches_build_output(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        expected_content, _ = _build_mural_content(layer_with_paths, canvas, TOP_DISTANCE)
        with tempfile.NamedTemporaryFile(suffix=".mural", delete=False) as f:
            path = f.name
        try:
            export_layer_mural(layer_with_paths, canvas, path, {"top_distance": TOP_DISTANCE})
            with open(path) as f:
                actual = f.read()
            assert actual == expected_content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

class TestMuralBatchExport:
    def test_creates_correct_number_of_files(
        self, project_with_layers: Project
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_mural(project_with_layers, tmpdir, {"top_distance": TOP_DISTANCE})
            files = [f for f in os.listdir(tmpdir) if f.endswith(".mural")]
            assert len(files) == 2

    def test_file_naming_pattern(self, project_with_layers: Project) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_mural(project_with_layers, tmpdir, {"top_distance": TOP_DISTANCE})
            files = sorted(os.listdir(tmpdir))
            assert files[0].startswith("Test_Project_01_")
            assert files[1].startswith("Test_Project_02_")
            assert all(f.endswith(".mural") for f in files)

    def test_hidden_layers_excluded(self, canvas: Canvas) -> None:
        project = Project(name="Proj", canvas=canvas)
        visible = Layer(name="Visible", color="#000000")
        visible.add_paths([[(0.0, 0.0), (10.0, 10.0)]])
        hidden = Layer(name="Hidden", color="#FF0000", visible=False)
        hidden.add_paths([[(5.0, 5.0), (15.0, 15.0)]])
        project.add_layer(visible)
        project.add_layer(hidden)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_mural(project, tmpdir, {"top_distance": TOP_DISTANCE})
            files = [f for f in os.listdir(tmpdir) if f.endswith(".mural")]
            assert len(files) == 1

    def test_creates_output_dir_if_missing(
        self, project_with_layers: Project
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "subdir", "output")
            export_all_layers_mural(project_with_layers, target, {"top_distance": TOP_DISTANCE})
            assert os.path.isdir(target)

    def test_returns_aggregated_warnings(self, canvas: Canvas) -> None:
        project = Project(name="Proj", canvas=canvas)
        big_canvas = Canvas(width_mm=700.0, height_mm=100.0, margin_mm=0.0)
        project_big = Project(name="Big", canvas=big_canvas)
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(0.0, 0.0), (700.0, 50.0)]])
        project_big.add_layer(layer)
        with tempfile.TemporaryDirectory() as tmpdir:
            warnings = export_all_layers_mural(project_big, tmpdir, {"top_distance": TOP_DISTANCE})
            assert isinstance(warnings, list)
            assert len(warnings) > 0
