"""Headless CLI mode for Plottter.

Usage examples:
  plottter --generator "Parametric Curves" --preset lissajous --output out.svg
  plottter --generator "Polar Curves" --preset rose --output rose.svg --paper A3
  plottter --generator "Parametric Curves" --output out.svg \\
             --param x_expr="sin(3*t)" --param y_expr="cos(2*t)"
  plottter --list-generators
  plottter --list-presets "Parametric Curves"
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plottter",
        description="Plottter — headless batch generation for pen plotters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Discovery flags
    parser.add_argument(
        "--list-generators",
        action="store_true",
        help="List all available generator names and exit.",
    )
    parser.add_argument(
        "--list-presets",
        metavar="GENERATOR",
        help="List all presets for GENERATOR and exit.",
    )

    # Generation options
    parser.add_argument(
        "--generator", "-g",
        metavar="NAME",
        help="Generator name (use --list-generators to see options).",
    )
    parser.add_argument(
        "--preset", "-p",
        metavar="NAME",
        help="Preset name to use as base parameters.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Output file path (or directory for --format svg --all-layers).",
    )
    parser.add_argument(
        "--param",
        action="append",
        metavar="name=value",
        dest="params",
        default=[],
        help="Override a generator parameter. Repeatable. E.g. --param num_points=5000",
    )

    # Canvas options
    parser.add_argument(
        "--paper",
        metavar="SIZE",
        default="A4",
        help="Paper size preset: A4 (default), A3, A2, Letter, Legal, or Custom.",
    )
    parser.add_argument(
        "--width",
        type=float,
        metavar="MM",
        help="Custom paper width in mm (requires --paper Custom).",
    )
    parser.add_argument(
        "--height",
        type=float,
        metavar="MM",
        help="Custom paper height in mm (requires --paper Custom).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=10.0,
        metavar="MM",
        help="Page margin in mm (default: 10).",
    )

    # Export options
    parser.add_argument(
        "--format", "-f",
        choices=["svg", "hpgl", "gcode"],
        default="svg",
        help="Output format: svg (default), hpgl, gcode.",
    )
    parser.add_argument(
        "--layer-color",
        metavar="HEX",
        default="#000000",
        help="Layer/pen color as hex string (default: #000000).",
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=0.3,
        metavar="MM",
        help="SVG stroke width in mm (default: 0.3).",
    )
    parser.add_argument(
        "--registration-marks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include registration marks in export (default: on).",
    )
    parser.add_argument(
        "--reg-mark-style",
        choices=["corners", "center", "both"],
        default="corners",
        help="Registration mark style (default: corners).",
    )
    # G-code specific
    parser.add_argument(
        "--travel-speed",
        type=int,
        default=3000,
        metavar="MM_MIN",
        help="G-code rapid travel speed mm/min (default: 3000).",
    )
    parser.add_argument(
        "--draw-speed",
        type=int,
        default=1000,
        metavar="MM_MIN",
        help="G-code drawing speed mm/min (default: 1000).",
    )
    parser.add_argument(
        "--pen-up-angle",
        type=int,
        default=0,
        metavar="DEG",
        help="G-code servo angle for pen-up (default: 0).",
    )
    parser.add_argument(
        "--pen-down-angle",
        type=int,
        default=90,
        metavar="DEG",
        help="G-code servo angle for pen-down (default: 90).",
    )

    return parser


def _parse_params(param_list: list[str]) -> dict[str, Any]:
    """Parse 'name=value' strings into a dict, auto-converting numeric values."""
    result: dict[str, Any] = {}
    for item in param_list:
        if "=" not in item:
            print(f"Warning: ignoring malformed --param '{item}' (expected name=value)", file=sys.stderr)
            continue
        name, _, raw_value = item.partition("=")
        name = name.strip()
        raw_value = raw_value.strip()
        # Try int, then float, then bool, then string
        if raw_value.lower() == "true":
            result[name] = True
        elif raw_value.lower() == "false":
            result[name] = False
        else:
            try:
                result[name] = int(raw_value)
            except ValueError:
                try:
                    result[name] = float(raw_value)
                except ValueError:
                    result[name] = raw_value
    return result


def _list_generators() -> None:
    """Print all registered generator names grouped by category."""
    from plottter.generators import GENERATORS  # noqa — triggers registration
    by_category: dict[str, list[str]] = {}
    for name, cls in GENERATORS.items():
        by_category.setdefault(cls.category, []).append(name)
    for category in sorted(by_category):
        print(f"\n[{category}]")
        for name in sorted(by_category[category]):
            print(f"  {name}")


def _list_presets(generator_name: str) -> None:
    """Print all preset names for the given generator."""
    from plottter.generators import GENERATORS  # noqa — triggers registration
    cls = GENERATORS.get(generator_name)
    if cls is None:
        print(f"Error: unknown generator '{generator_name}'.", file=sys.stderr)
        print("Run --list-generators to see available names.", file=sys.stderr)
        sys.exit(1)
    gen = cls()
    presets = gen.get_presets()
    if not presets:
        print(f"No presets defined for '{generator_name}'.")
        return
    print(f"Presets for '{generator_name}':")
    for p in presets:
        print(f"  {p.name}")


def run_cli(argv: list[str] | None = None) -> int:
    """Entry point for CLI mode.

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Discovery commands (no generation needed) ---
    if args.list_generators:
        _list_generators()
        return 0

    if args.list_presets:
        _list_presets(args.list_presets)
        return 0

    # --- Validate required args for generation ---
    if not args.generator:
        parser.print_help()
        return 1

    if not args.output:
        print("Error: --output is required for generation.", file=sys.stderr)
        return 1

    # --- Resolve generator ---
    from plottter.generators import GENERATORS  # noqa — triggers registration
    cls = GENERATORS.get(args.generator)
    if cls is None:
        print(f"Error: unknown generator '{args.generator}'.", file=sys.stderr)
        print("Run --list-generators to see available names.", file=sys.stderr)
        return 1

    gen = cls()

    # --- Build params: start with parameter defaults, apply preset, apply overrides ---
    params: dict[str, Any] = {}

    # Fill in defaults from get_parameters()
    for param in gen.get_parameters():
        params[param.name] = param.default

    # Apply preset if specified
    if args.preset:
        preset_map = {p.name.lower(): p for p in gen.get_presets()}
        preset = preset_map.get(args.preset.lower())
        if preset is None:
            # Try exact match first
            preset_map_exact = {p.name: p for p in gen.get_presets()}
            preset = preset_map_exact.get(args.preset)
        if preset is None:
            print(f"Error: preset '{args.preset}' not found for generator '{args.generator}'.", file=sys.stderr)
            print(f"Run --list-presets \"{args.generator}\" to see available presets.", file=sys.stderr)
            return 1
        params.update(preset.params)

    # Apply --param overrides
    cli_overrides = _parse_params(args.params)
    params.update(cli_overrides)

    # --- Build canvas ---
    from plottter.models import Canvas, PAPER_PRESETS
    paper_size = args.paper
    if paper_size == "Custom":
        if args.width is None or args.height is None:
            print("Error: --width and --height are required when --paper Custom is used.", file=sys.stderr)
            return 1
        canvas = Canvas(
            width_mm=args.width,
            height_mm=args.height,
            margin_mm=args.margin,
            paper_preset="Custom",
        )
    else:
        if paper_size not in PAPER_PRESETS:
            print(f"Error: unknown paper size '{paper_size}'.", file=sys.stderr)
            print(f"Available: {', '.join(PAPER_PRESETS)}, Custom", file=sys.stderr)
            return 1
        canvas = Canvas.from_preset(paper_size, margin=args.margin)

    # --- Run generator ---
    print(f"Generating with '{args.generator}' on {canvas.width_mm}×{canvas.height_mm}mm canvas…")

    def _progress(pct: float) -> None:
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

    try:
        paths = gen.generate(params, canvas, progress_callback=_progress)
    except Exception as exc:
        print(f"\nError during generation: {exc}", file=sys.stderr)
        return 1

    print()  # newline after progress bar

    if not paths:
        print("Warning: generator produced no paths.", file=sys.stderr)

    total_paths = len(paths)
    total_points = sum(len(p) for p in paths)
    print(f"  Generated {total_paths} paths, {total_points} points.")

    # --- Create layer ---
    from plottter.models import Layer
    layer = Layer(name="Layer 1", color=args.layer_color, paths=paths)

    # --- Export ---
    output_path = args.output
    fmt = args.format

    export_settings: dict[str, Any] = {
        "registration_marks": args.registration_marks,
        "reg_mark_style": args.reg_mark_style,
        "stroke_width": args.stroke_width,
        "travel_speed": args.travel_speed,
        "draw_speed": args.draw_speed,
        "pen_up_angle": args.pen_up_angle,
        "pen_down_angle": args.pen_down_angle,
    }

    try:
        if fmt == "svg":
            from plottter.export.svg import export_layer_svg
            export_layer_svg(layer, canvas, output_path, export_settings)
        elif fmt == "hpgl":
            from plottter.export.hpgl import export_layer_hpgl
            export_layer_hpgl(layer, canvas, output_path, export_settings)
        elif fmt == "gcode":
            from plottter.export.gcode import export_layer_gcode
            export_layer_gcode(layer, canvas, output_path, export_settings)
    except Exception as exc:
        print(f"Error during export: {exc}", file=sys.stderr)
        return 1

    print(f"  Exported to '{output_path}'.")
    return 0
