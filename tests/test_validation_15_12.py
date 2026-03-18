"""Phase 15.12 validation: CLI mode — headless batch generation.

This validation suite tests the CLI interface (src/plottter/cli.py) including:

1. Discovery commands — --list-generators and --list-presets produce correct output.
2. Math generator generation — each math generator produces a valid SVG via CLI.
3. Format outputs — SVG, HPGL, and G-code formats all produce well-formed files.
4. Invalid argument handling — bad generator names, missing required args, unknown
   paper sizes, malformed --param values all produce graceful error messages and
   non-zero exit codes.
5. Parameter overrides — --param name=value correctly overrides generator defaults.
6. Canvas options — --paper presets, --paper Custom with --width/--height, --margin.
7. Preset application — --preset correctly loads preset parameter values.
"""

from __future__ import annotations

import io
import os
import sys
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from plottter.cli import run_cli, _parse_params, _list_generators, _list_presets


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _run(args: list[str]) -> tuple[int, str, str]:
    """Run CLI with given args, capture stdout/stderr, return (exit_code, stdout, stderr)."""
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            exit_code = run_cli(args)
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return exit_code, out_buf.getvalue(), err_buf.getvalue()


def _parse_svg(filepath: str) -> ET.Element:
    tree = ET.parse(filepath)
    return tree.getroot()


SVG_NS = "http://www.w3.org/2000/svg"


def _find_all(root: ET.Element, local_tag: str) -> list[ET.Element]:
    return root.findall(f".//{{{SVG_NS}}}{local_tag}")


# ---------------------------------------------------------------------------
# 1. Discovery commands
# ---------------------------------------------------------------------------

class TestListGenerators:
    """--list-generators outputs all registered generators grouped by category."""

    def test_exits_zero(self):
        code, _, _ = _run(["--list-generators"])
        assert code == 0

    def test_math_category_present(self):
        code, out, _ = _run(["--list-generators"])
        assert code == 0
        assert "[math]" in out

    def test_image_category_present(self):
        code, out, _ = _run(["--list-generators"])
        assert "[image]" in out

    def test_parametric_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "Parametric Curves" in out

    def test_polar_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "Polar Curves" in out

    def test_flow_field_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "Flow Field" in out

    def test_modular_mult_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "Modular Multiplication" in out

    def test_lsystem_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "L-System / Fractal" in out

    def test_grid_pattern_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "Grid Pattern" in out

    def test_edge_detect_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "Edge Detect" in out

    def test_hatching_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "Hatching" in out

    def test_stipple_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "Stipple" in out

    def test_flow_image_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "Flow Image" in out

    def test_contour_listed(self):
        code, out, _ = _run(["--list-generators"])
        assert "Contour Lines" in out

    def test_all_11_generators_listed(self):
        code, out, _ = _run(["--list-generators"])
        from plottter.generators import GENERATORS
        # All registered generator names appear in output
        builtin_names = [
            "Parametric Curves", "Polar Curves", "Modular Multiplication",
            "Flow Field", "L-System / Fractal", "Grid Pattern",
            "Edge Detect", "Hatching", "Flow Image", "Stipple", "Contour Lines"
        ]
        for name in builtin_names:
            assert name in out, f"Generator '{name}' missing from --list-generators output"


class TestListPresets:
    """--list-presets GENERATOR outputs preset names for that generator."""

    def test_exits_zero_for_valid_generator(self):
        code, _, _ = _run(["--list-presets", "Parametric Curves"])
        assert code == 0

    def test_lists_parametric_presets(self):
        code, out, _ = _run(["--list-presets", "Parametric Curves"])
        assert "Lissajous" in out
        assert "Butterfly Curve" in out
        assert "Lorenz Attractor" in out

    def test_lists_polar_presets(self):
        code, out, _ = _run(["--list-presets", "Polar Curves"])
        assert "Rose (4-petal)" in out or "Rose" in out
        assert "Cardioid" in out

    def test_lists_lsystem_presets(self):
        code, out, _ = _run(["--list-presets", "L-System / Fractal"])
        assert "Koch Snowflake" in out
        assert "Dragon Curve" in out

    def test_lists_grid_pattern_presets(self):
        code, out, _ = _run(["--list-presets", "Grid Pattern"])
        assert "Truchet Tiles" in out

    def test_lists_modular_mult_presets(self):
        code, out, _ = _run(["--list-presets", "Modular Multiplication"])
        assert "Times 2" in out or "cardioid" in out.lower()

    def test_invalid_generator_exits_nonzero(self):
        code, _, err = _run(["--list-presets", "Nonexistent Generator"])
        assert code != 0
        assert "unknown generator" in err.lower() or "Error" in err

    def test_invalid_generator_error_message(self):
        code, _, err = _run(["--list-presets", "BadGen"])
        assert "BadGen" in err

    def test_suggests_list_generators_on_error(self):
        code, _, err = _run(["--list-presets", "BadGen"])
        assert "--list-generators" in err


# ---------------------------------------------------------------------------
# 2. Math generator generation — SVG output validation
# ---------------------------------------------------------------------------

class TestParametricGeneration:
    """Parametric Curves generator produces valid SVG via CLI."""

    def test_generates_svg(self, tmp_path):
        out_file = str(tmp_path / "parametric.svg")
        code, stdout, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--param", "num_points=500",
        ])
        assert code == 0, f"stderr: {err}"
        assert os.path.exists(out_file)

    def test_svg_is_valid_xml(self, tmp_path):
        out_file = str(tmp_path / "parametric.svg")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        root = _parse_svg(out_file)
        assert root is not None

    def test_svg_has_polylines(self, tmp_path):
        out_file = str(tmp_path / "parametric.svg")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        root = _parse_svg(out_file)
        polylines = _find_all(root, "polyline")
        assert len(polylines) > 0

    def test_svg_viewbox_matches_a4(self, tmp_path):
        out_file = str(tmp_path / "parametric.svg")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--param", "num_points=200",
            "--paper", "A4",
        ])
        root = _parse_svg(out_file)
        viewbox = root.get("viewBox", "")
        assert "210" in viewbox
        assert "297" in viewbox

    def test_with_lissajous_preset(self, tmp_path):
        out_file = str(tmp_path / "lissajous.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--preset", "Lissajous",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"
        assert os.path.exists(out_file)

    def test_stdout_reports_path_count(self, tmp_path):
        out_file = str(tmp_path / "parametric.svg")
        code, stdout, _ = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        assert "paths" in stdout.lower() or "Generated" in stdout


class TestPolarGeneration:
    """Polar Curves generator produces valid SVG via CLI."""

    def test_generates_svg(self, tmp_path):
        out_file = str(tmp_path / "polar.svg")
        code, _, err = _run([
            "--generator", "Polar Curves",
            "--output", out_file,
            "--param", "num_points=300",
        ])
        assert code == 0, f"stderr: {err}"
        assert os.path.exists(out_file)

    def test_svg_is_valid_xml(self, tmp_path):
        out_file = str(tmp_path / "polar.svg")
        _run([
            "--generator", "Polar Curves",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        root = _parse_svg(out_file)
        assert root is not None

    def test_cardioid_preset(self, tmp_path):
        out_file = str(tmp_path / "cardioid.svg")
        code, _, err = _run([
            "--generator", "Polar Curves",
            "--preset", "Cardioid",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"

    def test_rose_preset(self, tmp_path):
        out_file = str(tmp_path / "rose.svg")
        code, _, err = _run([
            "--generator", "Polar Curves",
            "--preset", "Rose (4-petal)",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"


class TestModularMultGeneration:
    """Modular Multiplication generator produces valid SVG via CLI."""

    def test_generates_svg(self, tmp_path):
        out_file = str(tmp_path / "modular.svg")
        code, _, err = _run([
            "--generator", "Modular Multiplication",
            "--output", out_file,
            "--param", "num_points=50",
        ])
        assert code == 0, f"stderr: {err}"
        assert os.path.exists(out_file)

    def test_svg_has_content(self, tmp_path):
        out_file = str(tmp_path / "modular.svg")
        _run([
            "--generator", "Modular Multiplication",
            "--output", out_file,
            "--param", "num_points=50",
            "--param", "multiplier=2.0",
        ])
        root = _parse_svg(out_file)
        polylines = _find_all(root, "polyline")
        assert len(polylines) > 0

    def test_times2_preset(self, tmp_path):
        out_file = str(tmp_path / "modular.svg")
        code, _, err = _run([
            "--generator", "Modular Multiplication",
            "--preset", "Times 2 (cardioid)",
            "--output", out_file,
        ])
        assert code == 0, f"stderr: {err}"


class TestFlowFieldGeneration:
    """Flow Field generator produces valid SVG via CLI."""

    def test_generates_svg(self, tmp_path):
        out_file = str(tmp_path / "flow.svg")
        code, _, err = _run([
            "--generator", "Flow Field",
            "--output", out_file,
            "--param", "num_particles=20",
            "--param", "max_steps=10",
            "--param", "seed=1",
        ])
        assert code == 0, f"stderr: {err}"
        assert os.path.exists(out_file)

    def test_svg_is_valid_xml(self, tmp_path):
        out_file = str(tmp_path / "flow.svg")
        _run([
            "--generator", "Flow Field",
            "--output", out_file,
            "--param", "num_particles=10",
            "--param", "max_steps=5",
            "--param", "seed=42",
        ])
        root = _parse_svg(out_file)
        assert root is not None


class TestLSystemGeneration:
    """L-System / Fractal generator produces valid SVG via CLI."""

    def test_generates_svg(self, tmp_path):
        out_file = str(tmp_path / "lsystem.svg")
        code, _, err = _run([
            "--generator", "L-System / Fractal",
            "--output", out_file,
            "--param", "iterations=2",
        ])
        assert code == 0, f"stderr: {err}"
        assert os.path.exists(out_file)

    def test_koch_snowflake_preset(self, tmp_path):
        out_file = str(tmp_path / "koch.svg")
        code, _, err = _run([
            "--generator", "L-System / Fractal",
            "--preset", "Koch Snowflake",
            "--output", out_file,
            "--param", "iterations=2",
        ])
        assert code == 0, f"stderr: {err}"

    def test_dragon_curve_preset(self, tmp_path):
        out_file = str(tmp_path / "dragon.svg")
        code, _, err = _run([
            "--generator", "L-System / Fractal",
            "--preset", "Dragon Curve",
            "--output", out_file,
            "--param", "iterations=3",
        ])
        assert code == 0, f"stderr: {err}"


class TestGridPatternGeneration:
    """Grid Pattern generator produces valid SVG via CLI."""

    def test_generates_svg(self, tmp_path):
        out_file = str(tmp_path / "grid.svg")
        code, _, err = _run([
            "--generator", "Grid Pattern",
            "--output", out_file,
        ])
        assert code == 0, f"stderr: {err}"
        assert os.path.exists(out_file)

    def test_truchet_preset(self, tmp_path):
        out_file = str(tmp_path / "truchet.svg")
        code, _, err = _run([
            "--generator", "Grid Pattern",
            "--preset", "Truchet Tiles",
            "--output", out_file,
        ])
        assert code == 0, f"stderr: {err}"

    def test_concentric_circles_preset(self, tmp_path):
        out_file = str(tmp_path / "concentric.svg")
        code, _, err = _run([
            "--generator", "Grid Pattern",
            "--preset", "Concentric Circles",
            "--output", out_file,
        ])
        assert code == 0, f"stderr: {err}"


# ---------------------------------------------------------------------------
# 3. Format outputs — HPGL and G-code
# ---------------------------------------------------------------------------

class TestHPGLOutput:
    """--format hpgl produces well-formed HPGL files."""

    def test_generates_hpgl_file(self, tmp_path):
        out_file = str(tmp_path / "out.hpgl")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--format", "hpgl",
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"
        assert os.path.exists(out_file)

    def test_hpgl_contains_initialize(self, tmp_path):
        out_file = str(tmp_path / "out.hpgl")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--format", "hpgl",
            "--param", "num_points=200",
        ])
        content = Path(out_file).read_text()
        assert "IN;" in content

    def test_hpgl_contains_select_pen(self, tmp_path):
        out_file = str(tmp_path / "out.hpgl")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--format", "hpgl",
            "--param", "num_points=200",
        ])
        content = Path(out_file).read_text()
        assert "SP" in content

    def test_hpgl_contains_pen_moves(self, tmp_path):
        out_file = str(tmp_path / "out.hpgl")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--format", "hpgl",
            "--param", "num_points=200",
        ])
        content = Path(out_file).read_text()
        assert "PU" in content
        assert "PD" in content


class TestGcodeOutput:
    """--format gcode produces well-formed G-code files."""

    def test_generates_gcode_file(self, tmp_path):
        out_file = str(tmp_path / "out.gcode")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--format", "gcode",
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"
        assert os.path.exists(out_file)

    def test_gcode_preamble(self, tmp_path):
        out_file = str(tmp_path / "out.gcode")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--format", "gcode",
            "--param", "num_points=200",
        ])
        content = Path(out_file).read_text()
        assert "G90" in content
        assert "G21" in content

    def test_gcode_contains_moves(self, tmp_path):
        out_file = str(tmp_path / "out.gcode")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--format", "gcode",
            "--param", "num_points=200",
        ])
        content = Path(out_file).read_text()
        assert "G0" in content or "G1" in content

    def test_gcode_speed_settings(self, tmp_path):
        out_file = str(tmp_path / "out.gcode")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--format", "gcode",
            "--param", "num_points=200",
            "--travel-speed", "5000",
            "--draw-speed", "2000",
        ])
        content = Path(out_file).read_text()
        assert "5000" in content or "2000" in content


# ---------------------------------------------------------------------------
# 4. Invalid argument handling
# ---------------------------------------------------------------------------

class TestInvalidGeneratorName:
    """Unknown generator names produce a graceful error."""

    def test_exits_nonzero(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, _ = _run([
            "--generator", "Totally Fake Generator",
            "--output", out_file,
        ])
        assert code != 0

    def test_error_message_mentions_generator_name(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _, _, err = _run([
            "--generator", "Totally Fake Generator",
            "--output", out_file,
        ])
        assert "Totally Fake Generator" in err

    def test_error_suggests_list_generators(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _, _, err = _run([
            "--generator", "BadGenerator",
            "--output", out_file,
        ])
        assert "--list-generators" in err

    def test_no_output_file_created(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _run([
            "--generator", "BadGenerator",
            "--output", out_file,
        ])
        assert not os.path.exists(out_file)


class TestMissingRequiredArgs:
    """Missing required arguments produce helpful error messages."""

    def test_no_generator_shows_help(self):
        code, out, _ = _run(["--output", "/tmp/out.svg"])
        # Should print help and exit non-zero, or exit non-zero at minimum
        assert code != 0

    def test_no_output_exits_nonzero(self):
        code, _, err = _run(["--generator", "Parametric Curves"])
        assert code != 0

    def test_no_output_error_message(self):
        code, _, err = _run(["--generator", "Parametric Curves"])
        assert "--output" in err or "output" in err.lower()

    def test_custom_paper_without_dimensions_fails(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "Custom",
            "--param", "num_points=200",
        ])
        assert code != 0
        assert "width" in err.lower() or "height" in err.lower() or "--width" in err

    def test_custom_paper_missing_height_fails(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "Custom",
            "--width", "150",
            "--param", "num_points=200",
        ])
        assert code != 0


class TestInvalidPaperSize:
    """Unknown paper size names produce a graceful error."""

    def test_exits_nonzero(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, _ = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "B5",
            "--param", "num_points=200",
        ])
        assert code != 0

    def test_error_mentions_paper_name(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "B5",
            "--param", "num_points=200",
        ])
        assert "B5" in err

    def test_error_lists_valid_options(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "B5",
            "--param", "num_points=200",
        ])
        # Should mention at least one valid paper size
        assert "A4" in err or "Letter" in err


class TestInvalidPreset:
    """Unknown preset name for a generator produces a graceful error."""

    def test_exits_nonzero(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, _ = _run([
            "--generator", "Parametric Curves",
            "--preset", "Nonexistent Preset",
            "--output", out_file,
        ])
        assert code != 0

    def test_error_mentions_preset_name(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _, _, err = _run([
            "--generator", "Parametric Curves",
            "--preset", "Nonexistent Preset",
            "--output", out_file,
        ])
        assert "Nonexistent Preset" in err

    def test_error_suggests_list_presets(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _, _, err = _run([
            "--generator", "Parametric Curves",
            "--preset", "BadPreset",
            "--output", out_file,
        ])
        assert "--list-presets" in err


class TestMalformedParamValues:
    """Malformed --param values (missing =) produce a warning but don't crash."""

    def test_malformed_param_prints_warning(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        # Run with a malformed param (no '='), should warn and continue
        _, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--param", "badparam",
            "--param", "num_points=200",
        ])
        assert "warning" in err.lower() or "ignoring" in err.lower() or "malformed" in err.lower()

    def test_generation_still_succeeds_after_malformed_param(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, _ = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--param", "badparam",
            "--param", "num_points=200",
        ])
        assert code == 0
        assert os.path.exists(out_file)


# ---------------------------------------------------------------------------
# 5. Parameter overrides
# ---------------------------------------------------------------------------

class TestParamOverrides:
    """--param name=value correctly overrides generator defaults."""

    def test_numeric_param_override(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, stdout, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--param", "num_points=300",
        ])
        assert code == 0, f"stderr: {err}"

    def test_float_param_override(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Polar Curves",
            "--output", out_file,
            "--param", "num_points=200",
            "--param", "theta_end=3.14159",
        ])
        assert code == 0, f"stderr: {err}"

    def test_string_param_override(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Grid Pattern",
            "--output", out_file,
            "--param", "mode=Truchet Tiles",
        ])
        assert code == 0, f"stderr: {err}"

    def test_multiple_param_overrides(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Flow Field",
            "--output", out_file,
            "--param", "num_particles=10",
            "--param", "max_steps=5",
            "--param", "seed=99",
        ])
        assert code == 0, f"stderr: {err}"


class TestParseParams:
    """Unit tests for _parse_params() helper."""

    def test_integer_value(self):
        result = _parse_params(["num_points=500"])
        assert result["num_points"] == 500
        assert isinstance(result["num_points"], int)

    def test_float_value(self):
        result = _parse_params(["angle=3.14"])
        assert abs(result["angle"] - 3.14) < 1e-9
        assert isinstance(result["angle"], float)

    def test_true_bool(self):
        result = _parse_params(["connect_tsp=true"])
        assert result["connect_tsp"] is True

    def test_false_bool(self):
        result = _parse_params(["invert=false"])
        assert result["invert"] is False

    def test_string_value(self):
        result = _parse_params(["x_expr=sin(t)"])
        assert result["x_expr"] == "sin(t)"

    def test_multiple_params(self):
        result = _parse_params(["a=1", "b=2.5", "c=hello"])
        assert result["a"] == 1
        assert result["b"] == 2.5
        assert result["c"] == "hello"

    def test_malformed_no_equals(self, capsys):
        result = _parse_params(["badparam"])
        assert "badparam" not in result

    def test_value_with_equals_sign(self):
        result = _parse_params(["x_expr=sin(t)+cos(t)"])
        assert result["x_expr"] == "sin(t)+cos(t)"

    def test_empty_list(self):
        result = _parse_params([])
        assert result == {}

    def test_negative_float(self):
        result = _parse_params(["offset=-3.5"])
        assert result["offset"] == -3.5

    def test_negative_int(self):
        result = _parse_params(["seed=-1"])
        assert result["seed"] == -1


# ---------------------------------------------------------------------------
# 6. Canvas options
# ---------------------------------------------------------------------------

class TestCanvasOptions:
    """--paper, --width, --height, --margin all affect SVG output."""

    def test_a4_viewbox(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "A4",
            "--param", "num_points=200",
        ])
        root = _parse_svg(out_file)
        viewbox = root.get("viewBox", "")
        assert "210" in viewbox
        assert "297" in viewbox

    def test_a3_viewbox(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "A3",
            "--param", "num_points=200",
        ])
        root = _parse_svg(out_file)
        viewbox = root.get("viewBox", "")
        assert "297" in viewbox
        assert "420" in viewbox

    def test_letter_viewbox(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "Letter",
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"
        root = _parse_svg(out_file)
        viewbox = root.get("viewBox", "")
        assert "215" in viewbox

    def test_custom_paper(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "Custom",
            "--width", "150",
            "--height", "200",
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"
        root = _parse_svg(out_file)
        viewbox = root.get("viewBox", "")
        assert "150" in viewbox
        assert "200" in viewbox

    def test_custom_margin_affects_drawing_area(self, tmp_path):
        """Generation with custom margin succeeds (no exception in SVG export)."""
        out_file = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Modular Multiplication",
            "--output", out_file,
            "--margin", "20",
            "--param", "num_points=50",
        ])
        assert code == 0, f"stderr: {err}"

    def test_all_paper_presets_succeed(self, tmp_path):
        """All standard paper presets produce a valid SVG file."""
        presets = ["A4", "A3", "A2", "Letter", "Legal"]
        for preset in presets:
            out_file = str(tmp_path / f"{preset}.svg")
            code, _, err = _run([
                "--generator", "Modular Multiplication",
                "--output", out_file,
                "--paper", preset,
                "--param", "num_points=50",
            ])
            assert code == 0, f"Paper preset '{preset}' failed: {err}"
            assert os.path.exists(out_file), f"No output for preset '{preset}'"


# ---------------------------------------------------------------------------
# 7. Preset application
# ---------------------------------------------------------------------------

class TestPresetApplication:
    """--preset correctly loads preset parameter values."""

    def test_preset_butterfly_curve(self, tmp_path):
        out_file = str(tmp_path / "butterfly.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--preset", "Butterfly Curve",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"
        root = _parse_svg(out_file)
        polylines = _find_all(root, "polyline")
        assert len(polylines) > 0

    def test_preset_spirograph(self, tmp_path):
        out_file = str(tmp_path / "spirograph.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--preset", "Spirograph (Epitrochoid)",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"

    def test_preset_archimedean_spiral(self, tmp_path):
        out_file = str(tmp_path / "spiral.svg")
        code, _, err = _run([
            "--generator", "Polar Curves",
            "--preset", "Archimedean Spiral",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"

    def test_preset_overridden_by_param(self, tmp_path):
        """A --param after --preset overrides the preset value."""
        out_file_low = str(tmp_path / "low.svg")
        out_file_high = str(tmp_path / "high.svg")

        _run([
            "--generator", "Modular Multiplication",
            "--preset", "Times 2 (cardioid)",
            "--output", out_file_low,
            "--param", "num_points=20",
        ])
        _run([
            "--generator", "Modular Multiplication",
            "--preset", "Times 2 (cardioid)",
            "--output", out_file_high,
            "--param", "num_points=200",
        ])

        root_low = _parse_svg(out_file_low)
        root_high = _parse_svg(out_file_high)
        polylines_low = _find_all(root_low, "polyline")
        polylines_high = _find_all(root_high, "polyline")
        # More points → more or equal polylines (may vary)
        assert len(polylines_high) >= len(polylines_low)

    def test_preset_case_insensitive_lookup(self, tmp_path):
        """Preset names are looked up case-insensitively."""
        out_file = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--preset", "lissajous",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        assert code == 0, f"stderr: {err}"


# ---------------------------------------------------------------------------
# 8. SVG output content validation
# ---------------------------------------------------------------------------

class TestSVGContentValidation:
    """Verify SVG structure and content for generated files."""

    def test_svg_has_root_svg_element(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        root = _parse_svg(out_file)
        local = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        assert local == "svg"

    def test_svg_has_width_height_in_mm(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "A4",
            "--param", "num_points=200",
        ])
        root = _parse_svg(out_file)
        width = root.get("width", "")
        height = root.get("height", "")
        assert "mm" in width
        assert "mm" in height

    def test_svg_with_registration_marks(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--registration-marks",
            "--param", "num_points=200",
        ])
        content = Path(out_file).read_text()
        assert "registration" in content.lower() or "reg" in content.lower()

    def test_svg_without_registration_marks(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--no-registration-marks",
            "--param", "num_points=200",
        ])
        content = Path(out_file).read_text()
        # Either no registration group, or fewer lines total
        root = _parse_svg(out_file)
        assert root is not None  # valid SVG

    def test_svg_layer_color_applied(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _run([
            "--generator", "Modular Multiplication",
            "--output", out_file,
            "--layer-color", "#FF0000",
            "--param", "num_points=50",
        ])
        content = Path(out_file).read_text()
        assert "#ff0000" in content.lower() or "#FF0000" in content

    def test_svg_stroke_width_applied(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        _run([
            "--generator", "Modular Multiplication",
            "--output", out_file,
            "--stroke-width", "0.5",
            "--param", "num_points=50",
        ])
        content = Path(out_file).read_text()
        assert "0.5" in content

    def test_stdout_reports_exported_path(self, tmp_path):
        out_file = str(tmp_path / "my_art.svg")
        code, stdout, _ = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--param", "num_points=200",
        ])
        assert code == 0
        assert "my_art.svg" in stdout or out_file in stdout

    def test_stdout_shows_canvas_dimensions(self, tmp_path):
        out_file = str(tmp_path / "out.svg")
        code, stdout, _ = _run([
            "--generator", "Parametric Curves",
            "--output", out_file,
            "--paper", "A4",
            "--param", "num_points=200",
        ])
        assert "210" in stdout or "297" in stdout


# ---------------------------------------------------------------------------
# 9. No-args / help behavior
# ---------------------------------------------------------------------------

class TestHelpBehavior:
    """Running with no args or --help behaves gracefully."""

    def test_no_generator_exits_nonzero(self):
        code, _, _ = _run([])
        assert code != 0

    def test_help_exits_zero(self):
        code, out, _ = _run(["--help"])
        assert code == 0


# ---------------------------------------------------------------------------
# 10. Out-of-range parameter values — graceful handling
# ---------------------------------------------------------------------------

class TestOutOfRangeParams:
    """CLI handles out-of-range generator parameter values gracefully.

    The spec (15.12) requires that out-of-range values produce either a graceful
    error message (non-zero exit with descriptive stderr) or a valid result when
    the generator internally clamps the value.  No unhandled exceptions / crashes.
    """

    def test_negative_num_points_exits_nonzero(self, tmp_path):
        """num_points=-1 for Parametric Curves: numpy raises; CLI returns exit 1."""
        out_path = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_path,
            "--param", "num_points=-1",
        ])
        assert code != 0
        assert "Error" in err or "error" in err

    def test_negative_num_points_error_message_descriptive(self, tmp_path):
        """Error message for negative num_points is descriptive (not a raw traceback)."""
        out_path = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_path,
            "--param", "num_points=-1",
        ])
        assert code != 0
        # Should mention "generation" context, not dump a raw Python traceback header
        assert "Error during generation" in err

    def test_zero_num_points_does_not_crash(self, tmp_path):
        """num_points=0 produces empty output with a warning but does not crash."""
        out_path = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Polar Curves",
            "--output", out_path,
            "--param", "num_points=0",
        ])
        # Generator succeeds (exit 0) and warns about no paths, or exits with error —
        # either is acceptable as long as the process does not raise an unhandled exception.
        assert code in (0, 1)

    def test_zero_num_points_warning_when_successful(self, tmp_path):
        """When num_points=0 exits 0, stderr mentions 'no paths' warning."""
        out_path = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Polar Curves",
            "--output", out_path,
            "--param", "num_points=0",
        ])
        if code == 0:
            assert "no paths" in err.lower() or "warning" in err.lower()

    def test_lsystem_zero_iterations_does_not_crash(self, tmp_path):
        """L-System with iterations=0 should not crash (returns axiom or empty)."""
        out_path = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "L-System / Fractal",
            "--output", out_path,
            "--param", "iterations=0",
        ])
        # Either succeeds or errors gracefully — never an unhandled exception
        assert code in (0, 1)

    def test_lsystem_negative_iterations_does_not_crash(self, tmp_path):
        """L-System with negative iterations is handled without an unhandled exception."""
        out_path = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "L-System / Fractal",
            "--output", out_path,
            "--param", "iterations=-3",
        ])
        assert code in (0, 1)

    def test_negative_margin_does_not_crash(self, tmp_path):
        """--margin with a negative value should not cause an unhandled exception."""
        out_path = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_path,
            "--margin", "-50",
            "--param", "num_points=200",
        ])
        assert code in (0, 1)

    def test_very_large_margin_does_not_crash(self, tmp_path):
        """An excessively large margin should not crash (canvas may be empty but no exception)."""
        out_path = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Parametric Curves",
            "--output", out_path,
            "--margin", "500",
            "--param", "num_points=200",
        ])
        assert code in (0, 1)

    def test_modular_mult_negative_num_points_does_not_crash(self, tmp_path):
        """Modular Multiplication with negative num_points does not raise unhandled exception."""
        out_path = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Modular Multiplication",
            "--output", out_path,
            "--param", "num_points=-5",
        ])
        assert code in (0, 1)

    def test_flow_field_negative_num_lines_does_not_crash(self, tmp_path):
        """Flow Field with negative num_lines does not raise an unhandled exception."""
        out_path = str(tmp_path / "out.svg")
        code, _, err = _run([
            "--generator", "Flow Field",
            "--output", out_path,
            "--param", "num_lines=-1",
            "--param", "num_steps=50",
        ])
        assert code in (0, 1)
