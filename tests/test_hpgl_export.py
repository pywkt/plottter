"""Tests for HPGL export (Phase 11.4)."""

from __future__ import annotations

import os
import tempfile

import pytest

from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.project import Project
from plottter.export.hpgl import (
    export_layer_hpgl,
    export_all_layers_hpgl,
    _mm_to_hpgl,
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


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

class TestHpglCoordinates:
    def test_basic_conversion(self, canvas: Canvas) -> None:
        hx, hy = _mm_to_hpgl(10.0, 20.0, canvas.height_mm)
        assert hx == 400   # 10 * 40
        assert hy == int((297.0 - 20.0) * 40)  # Y inverted

    def test_zero_origin(self, canvas: Canvas) -> None:
        hx, hy = _mm_to_hpgl(0.0, 0.0, canvas.height_mm)
        assert hx == 0
        assert hy == int(297.0 * 40)  # Y-inverted: bottom-left origin

    def test_top_left_corner(self, canvas: Canvas) -> None:
        hx, hy = _mm_to_hpgl(0.0, canvas.height_mm, canvas.height_mm)
        assert hx == 0
        assert hy == 0  # top of canvas maps to 0 in HPGL (Y inverted)

    def test_values_are_integers(self, canvas: Canvas) -> None:
        hx, hy = _mm_to_hpgl(5.5, 12.3, canvas.height_mm)
        assert isinstance(hx, int)
        assert isinstance(hy, int)

    def test_units_are_40_per_mm(self, canvas: Canvas) -> None:
        hx, _ = _mm_to_hpgl(1.0, 0.0, canvas.height_mm)
        assert hx == 40

    def test_y_axis_inverted(self, canvas: Canvas) -> None:
        """Higher Y in mm → smaller HPGL Y value."""
        _, hy_low = _mm_to_hpgl(0.0, 10.0, canvas.height_mm)
        _, hy_high = _mm_to_hpgl(0.0, 200.0, canvas.height_mm)
        assert hy_high < hy_low


# ---------------------------------------------------------------------------
# File content structure
# ---------------------------------------------------------------------------

class TestHpglContent:
    def test_contains_in_command(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            assert "IN;" in content
        finally:
            os.unlink(path)

    def test_contains_sp_command(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer_with_paths, canvas, path, {"pen_number": 3})
            with open(path) as f:
                content = f.read()
            assert "SP3;" in content
        finally:
            os.unlink(path)

    def test_default_pen_number_is_1(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            assert "SP1;" in content
        finally:
            os.unlink(path)

    def test_ends_with_pen_up(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read().strip()
            assert content.endswith("PU;")
        finally:
            os.unlink(path)

    def test_pen_up_move_before_each_path(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        """Each path should start with a PU move to the first point."""
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            # 2 paths → 2 PU{x},{y} commands (plus final bare PU;)
            pu_commands = [line for line in content.splitlines() if line.startswith("PU") and "," in line]
            assert len(pu_commands) == 2
        finally:
            os.unlink(path)

    def test_pen_down_sequence_correct(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            pd_commands = [line for line in content.splitlines() if line.startswith("PD")]
            assert len(pd_commands) == 2  # one PD per path
        finally:
            os.unlink(path)

    def test_coordinate_values_are_integers(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        """HPGL coordinates must be integer plotter units."""
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            for line in content.splitlines():
                if line.startswith("PU") and "," in line:
                    coords_str = line[2:].rstrip(";")
                    for part in coords_str.split(","):
                        assert float(part) == int(part), f"Non-integer coordinate in: {line}"
                elif line.startswith("PD"):
                    coords_str = line[2:].rstrip(";")
                    for part in coords_str.split(","):
                        assert float(part) == int(part), f"Non-integer coordinate in: {line}"
        finally:
            os.unlink(path)

    def test_multipoint_polyline_uses_pd_with_multiple_pairs(
        self, canvas: Canvas
    ) -> None:
        """A 3-point path → PD with 2 coordinate pairs."""
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]])
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer, canvas, path, {})
            with open(path) as f:
                content = f.read()
            pd_line = next(line for line in content.splitlines() if line.startswith("PD"))
            # Two coordinate pairs = three comma-separated values pairs (x1,y1,x2,y2)
            coords_str = pd_line[2:].rstrip(";")
            values = coords_str.split(",")
            assert len(values) == 4  # 2 points × 2 coords each
        finally:
            os.unlink(path)

    def test_optional_speed_command(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer_with_paths, canvas, path, {"speed": 30})
            with open(path) as f:
                content = f.read()
            assert "VS30;" in content
        finally:
            os.unlink(path)

    def test_no_speed_command_when_not_set(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            assert "VS" not in content
        finally:
            os.unlink(path)

    def test_optional_force_command(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer_with_paths, canvas, path, {"force": 12})
            with open(path) as f:
                content = f.read()
            assert "FS12;" in content
        finally:
            os.unlink(path)

    def test_single_point_path_skipped(self, canvas: Canvas) -> None:
        """Paths with fewer than 2 points produce no PD command."""
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(5.0, 5.0)]])  # single point
        with tempfile.NamedTemporaryFile(suffix=".plt", delete=False, mode="w") as f:
            path = f.name
        try:
            export_layer_hpgl(layer, canvas, path, {})
            with open(path) as f:
                content = f.read()
            assert "PD" not in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

class TestHpglBatchExport:
    def test_creates_correct_number_of_files(
        self, project_with_layers: Project
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_hpgl(project_with_layers, tmpdir, {})
            files = [f for f in os.listdir(tmpdir) if f.endswith(".plt")]
            assert len(files) == 2

    def test_file_naming_pattern(self, project_with_layers: Project) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_hpgl(project_with_layers, tmpdir, {})
            files = sorted(os.listdir(tmpdir))
            assert files[0].startswith("Test_Project_01_")
            assert files[1].startswith("Test_Project_02_")
            assert all(f.endswith(".plt") for f in files)

    def test_each_file_has_correct_pen_number(
        self, project_with_layers: Project
    ) -> None:
        """Layer 1 gets SP1, layer 2 gets SP2, etc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_hpgl(project_with_layers, tmpdir, {})
            files = sorted(os.listdir(tmpdir))
            with open(os.path.join(tmpdir, files[0])) as f:
                content0 = f.read()
            with open(os.path.join(tmpdir, files[1])) as f:
                content1 = f.read()
            assert "SP1;" in content0
            assert "SP2;" in content1

    def test_hidden_layers_excluded(self, canvas: Canvas) -> None:
        project = Project(name="Proj", canvas=canvas)
        visible = Layer(name="Visible", color="#000000")
        visible.add_paths([[(0.0, 0.0), (10.0, 10.0)]])
        hidden = Layer(name="Hidden", color="#FF0000", visible=False)
        hidden.add_paths([[(5.0, 5.0), (15.0, 15.0)]])
        project.add_layer(visible)
        project.add_layer(hidden)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_hpgl(project, tmpdir, {})
            files = [f for f in os.listdir(tmpdir) if f.endswith(".plt")]
            assert len(files) == 1

    def test_creates_output_dir_if_missing(
        self, project_with_layers: Project
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "subdir", "output")
            export_all_layers_hpgl(project_with_layers, target, {})
            assert os.path.isdir(target)
