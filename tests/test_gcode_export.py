"""Tests for G-code export (Phase 11.5)."""

from __future__ import annotations

import os
import tempfile

import pytest

from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.project import Project
from plottter.export.gcode import export_layer_gcode, export_all_layers_gcode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def canvas() -> Canvas:
    return Canvas(width_mm=210.0, height_mm=297.0, margin_mm=10.0, paper_preset="A4")


@pytest.fixture
def layer_with_paths() -> Layer:
    layer = Layer(name="Test Layer", color="#000000")
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
# Preamble and epilogue
# ---------------------------------------------------------------------------

class TestGcodePreamble:
    def test_starts_with_g90(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            lines = content.splitlines()
            assert lines[0] == "G90"
        finally:
            os.unlink(path)

    def test_g21_present(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            assert "G21" in content
        finally:
            os.unlink(path)

    def test_g28_homing_in_preamble(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            lines = content.splitlines()
            # G28 should appear in first few lines (preamble)
            assert "G28" in lines[:4]
        finally:
            os.unlink(path)

    def test_ends_with_g28_and_m5(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            lines = [l for l in content.splitlines() if l.strip()]
            assert lines[-2] == "G28"
            assert lines[-1] == "M5"
        finally:
            os.unlink(path)

    def test_preamble_pen_up_present(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        """Preamble ends with M3 S0 (default pen-up angle)."""
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {"pen_up_angle": 0})
            with open(path) as f:
                content = f.read()
            lines = content.splitlines()
            # M3 S0 should appear before the first G0 move
            g0_idx = next(i for i, l in enumerate(lines) if l.startswith("G0"))
            m3_lines_before_g0 = [l for l in lines[:g0_idx] if l.startswith("M3")]
            assert len(m3_lines_before_g0) >= 1
            assert "M3 S0" in m3_lines_before_g0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Pen up / down servo commands
# ---------------------------------------------------------------------------

class TestGcodeServoCommands:
    def test_pen_down_uses_correct_angle(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {"pen_down_angle": 90})
            with open(path) as f:
                content = f.read()
            assert "M3 S90" in content
        finally:
            os.unlink(path)

    def test_pen_up_uses_correct_angle(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {"pen_up_angle": 0})
            with open(path) as f:
                content = f.read()
            assert "M3 S0" in content
        finally:
            os.unlink(path)

    def test_custom_servo_angles(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(
                layer_with_paths, canvas, path,
                {"pen_up_angle": 15, "pen_down_angle": 75}
            )
            with open(path) as f:
                content = f.read()
            assert "M3 S15" in content
            assert "M3 S75" in content
        finally:
            os.unlink(path)

    def test_pen_down_between_g0_and_g1(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        """Pattern per path: G0 (rapid move) → M3 S{down} → G1 (draw) → M3 S{up}."""
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("G0 X"):
                    # Next non-empty command should lower the pen
                    next_cmd = lines[i + 1]
                    assert next_cmd.startswith("M3"), (
                        f"Expected M3 pen-down after G0, got: {next_cmd}"
                    )
                    break
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Move commands
# ---------------------------------------------------------------------------

class TestGcodeMoves:
    def test_rapid_moves_use_g0(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            g0_lines = [l for l in content.splitlines() if l.startswith("G0 X")]
            assert len(g0_lines) >= 1
        finally:
            os.unlink(path)

    def test_draw_moves_use_g1(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {})
            with open(path) as f:
                content = f.read()
            g1_lines = [l for l in content.splitlines() if l.startswith("G1 X")]
            assert len(g1_lines) >= 1
        finally:
            os.unlink(path)

    def test_travel_speed_in_g0(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {"travel_speed": 3000})
            with open(path) as f:
                content = f.read()
            g0_lines = [l for l in content.splitlines() if l.startswith("G0 X")]
            assert all("F3000" in l for l in g0_lines)
        finally:
            os.unlink(path)

    def test_draw_speed_in_g1(self, layer_with_paths: Layer, canvas: Canvas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer_with_paths, canvas, path, {"draw_speed": 1000})
            with open(path) as f:
                content = f.read()
            g1_lines = [l for l in content.splitlines() if l.startswith("G1 X")]
            assert all("F1000" in l for l in g1_lines)
        finally:
            os.unlink(path)

    def test_coordinate_precision_3_decimal_places(
        self, canvas: Canvas
    ) -> None:
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(1.1111, 2.2222), (3.3333, 4.4444)]])
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer, canvas, path, {})
            with open(path) as f:
                content = f.read()
            # Coordinates should appear with 3 decimal places
            assert "X1.111" in content
            assert "Y2.222" in content
        finally:
            os.unlink(path)

    def test_single_point_path_skipped(self, canvas: Canvas) -> None:
        """Paths with fewer than 2 points produce no G0/G1 moves."""
        layer = Layer(name="L", color="#000000")
        layer.add_paths([[(5.0, 5.0)]])
        with tempfile.NamedTemporaryFile(suffix=".gcode", delete=False) as f:
            path = f.name
        try:
            export_layer_gcode(layer, canvas, path, {})
            with open(path) as f:
                content = f.read()
            assert "G0 X" not in content
            assert "G1 X" not in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

class TestGcodeBatchExport:
    def test_creates_correct_number_of_files(
        self, project_with_layers: Project
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_gcode(project_with_layers, tmpdir, {})
            files = [f for f in os.listdir(tmpdir) if f.endswith(".gcode")]
            assert len(files) == 2

    def test_file_naming_pattern(self, project_with_layers: Project) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_gcode(project_with_layers, tmpdir, {})
            files = sorted(os.listdir(tmpdir))
            assert files[0].startswith("Test_Project_01_")
            assert files[1].startswith("Test_Project_02_")
            assert all(f.endswith(".gcode") for f in files)

    def test_each_file_has_preamble(self, project_with_layers: Project) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_gcode(project_with_layers, tmpdir, {})
            for fname in os.listdir(tmpdir):
                with open(os.path.join(tmpdir, fname)) as f:
                    content = f.read()
                assert "G90" in content
                assert "G21" in content

    def test_hidden_layers_excluded(self, canvas: Canvas) -> None:
        project = Project(name="Proj", canvas=canvas)
        visible = Layer(name="Visible", color="#000000")
        visible.add_paths([[(0.0, 0.0), (10.0, 10.0)]])
        hidden = Layer(name="Hidden", color="#FF0000", visible=False)
        hidden.add_paths([[(5.0, 5.0), (15.0, 15.0)]])
        project.add_layer(visible)
        project.add_layer(hidden)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_all_layers_gcode(project, tmpdir, {})
            files = [f for f in os.listdir(tmpdir) if f.endswith(".gcode")]
            assert len(files) == 1

    def test_creates_output_dir_if_missing(
        self, project_with_layers: Project
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "subdir", "output")
            export_all_layers_gcode(project_with_layers, target, {})
            assert os.path.isdir(target)
