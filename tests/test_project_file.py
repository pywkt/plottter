"""Tests for project file save/load round-trips."""

import gzip
import json
import os
import tempfile

import pytest

from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.project import Project
from plottter.io.project_file import save_project, load_project, _GZIP_THRESHOLD_BYTES


def _make_project(with_paths: bool = True) -> Project:
    canvas = Canvas.from_preset("A4", margin=15.0)
    layer = Layer(
        name="Ink Layer",
        color="#1A2B3C",
        visible=True,
        locked=False,
        opacity=0.9,
        generator_info={"type": "parametric", "preset": "Lissajous"},
    )
    if with_paths:
        layer.add_paths([
            [(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)],
            [(5.0, 5.0), (5.0, 15.0)],
        ])
    project = Project(
        name="My Art",
        canvas=canvas,
        layers=[layer],
        registration_marks=True,
        reg_mark_style="corners",
    )
    return project


class TestRoundTrip:
    def test_basic_fields_preserved(self, tmp_path):
        proj = _make_project()
        filepath = str(tmp_path / "test.plottter")
        save_project(proj, filepath)
        loaded = load_project(filepath)

        assert loaded.name == proj.name
        assert loaded.registration_marks == proj.registration_marks
        assert loaded.reg_mark_style == proj.reg_mark_style

    def test_canvas_preserved(self, tmp_path):
        proj = _make_project()
        filepath = str(tmp_path / "test.plottter")
        save_project(proj, filepath)
        loaded = load_project(filepath)

        assert loaded.canvas.width_mm == pytest.approx(proj.canvas.width_mm)
        assert loaded.canvas.height_mm == pytest.approx(proj.canvas.height_mm)
        assert loaded.canvas.margin_mm == pytest.approx(proj.canvas.margin_mm)
        assert loaded.canvas.paper_preset == proj.canvas.paper_preset

    def test_layer_preserved(self, tmp_path):
        proj = _make_project()
        filepath = str(tmp_path / "test.plottter")
        save_project(proj, filepath)
        loaded = load_project(filepath)

        assert len(loaded.layers) == 1
        orig = proj.layers[0]
        saved = loaded.layers[0]
        assert saved.id == orig.id
        assert saved.name == orig.name
        assert saved.color == orig.color
        assert saved.visible == orig.visible
        assert saved.locked == orig.locked
        assert saved.opacity == pytest.approx(orig.opacity)
        assert saved.generator_info == orig.generator_info

    def test_paths_preserved(self, tmp_path):
        proj = _make_project(with_paths=True)
        filepath = str(tmp_path / "test.plottter")
        save_project(proj, filepath)
        loaded = load_project(filepath)

        orig_paths = proj.layers[0].paths
        load_paths = loaded.layers[0].paths
        assert len(load_paths) == len(orig_paths)
        for orig_p, load_p in zip(orig_paths, load_paths):
            assert len(orig_p) == len(load_p)
            for (ox, oy), (lx, ly) in zip(orig_p, load_p):
                assert ox == pytest.approx(lx)
                assert oy == pytest.approx(ly)

    def test_version_field_in_json(self, tmp_path):
        proj = _make_project()
        filepath = str(tmp_path / "test.plottter")
        save_project(proj, filepath)
        with open(filepath, "rb") as fh:
            data = json.loads(fh.read())
        assert "version" in data
        assert data["version"] == 1

    def test_multiple_layers_round_trip(self, tmp_path):
        canvas = Canvas.from_preset("A3")
        proj = Project(name="Multi", canvas=canvas)
        for i in range(3):
            proj.add_layer(Layer(name=f"Layer {i}", color=f"#0{i*3:02x}000"))
        filepath = str(tmp_path / "multi.plottter")
        save_project(proj, filepath)
        loaded = load_project(filepath)
        assert len(loaded.layers) == 3
        for i, l in enumerate(loaded.layers):
            assert l.name == f"Layer {i}"


class TestGzip:
    def test_gzip_file_detected_and_loaded(self, tmp_path):
        """Force gzip by monkey-patching the threshold."""
        import plottter.io.project_file as pf_module
        original = pf_module._GZIP_THRESHOLD_BYTES
        pf_module._GZIP_THRESHOLD_BYTES = 0  # force gzip for any payload
        try:
            proj = _make_project()
            filepath = str(tmp_path / "gzip.plottter")
            save_project(proj, filepath)
            # File should be gzip-compressed
            with open(filepath, "rb") as fh:
                magic = fh.read(2)
            assert magic == b"\x1f\x8b", "Expected gzip magic bytes"
            # Load should still work
            loaded = load_project(filepath)
            assert loaded.name == proj.name
        finally:
            pf_module._GZIP_THRESHOLD_BYTES = original

    def test_plain_json_still_loads(self, tmp_path):
        proj = _make_project()
        filepath = str(tmp_path / "plain.plottter")
        # Write as plain JSON explicitly
        import plottter.io.project_file as pf_module
        original = pf_module._GZIP_THRESHOLD_BYTES
        pf_module._GZIP_THRESHOLD_BYTES = 10_000_000  # never gzip
        try:
            save_project(proj, filepath)
            with open(filepath, "rb") as fh:
                magic = fh.read(2)
            assert magic != b"\x1f\x8b", "Expected plain JSON"
            loaded = load_project(filepath)
            assert loaded.name == proj.name
        finally:
            pf_module._GZIP_THRESHOLD_BYTES = original


class TestForwardCompatibility:
    def test_unknown_fields_ignored(self, tmp_path):
        """A file with extra unknown fields should still load without crashing."""
        proj = _make_project()
        filepath = str(tmp_path / "future.plottter")
        save_project(proj, filepath)
        # Inject an unknown top-level key
        with open(filepath, "rb") as fh:
            data = json.loads(fh.read())
        data["unknown_future_key"] = "some value"
        data["layers"][0]["new_layer_field"] = 42
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        # Should not raise
        loaded = load_project(filepath)
        assert loaded.name == proj.name

    def test_missing_optional_fields_use_defaults(self, tmp_path):
        """A minimal JSON file without optional fields should load with defaults."""
        minimal = {
            "version": 1,
            "name": "Minimal",
            "canvas": {"width_mm": 100.0, "height_mm": 150.0},
            "layers": [],
        }
        filepath = str(tmp_path / "minimal.plottter")
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(minimal, fh)
        loaded = load_project(filepath)
        assert loaded.name == "Minimal"
        assert loaded.canvas.margin_mm == 10.0  # default
        assert loaded.registration_marks is True  # default
        assert loaded.reg_mark_style == "corners"  # default
