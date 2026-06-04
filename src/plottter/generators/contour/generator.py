"""ContourGenerator class."""

from __future__ import annotations

from typing import Any

import numpy as np

from plottter.generators import register_generator
from plottter.generators._helpers import _px_to_mm, compute_image_rect
from plottter.generators.base import (
    Generator,
    Parameter,
    Preset,
)
from plottter.models import Canvas, Polyline

from ._smoothing import _chaikin_smooth
from ._isolines import _trace_isolines, _extract_contours_with_hierarchy, _compute_thresholds
from ._line_art import _trace_line_art, _trace_skeleton
from ._fmm import (
    _compute_fmm_field, _trace_fmm_topographic, _fmm_displacement,
    _compute_fmm_wave_y_positions, _fmm_wave, _fmm_radial,
)
from ._fills import _fill_polygon_hatch, _fill_polygon_concentric


@register_generator
class ContourGenerator(Generator):
    """Topographic contour lines traced from image brightness thresholds.

    Provides four operating modes:

    **Contour Levels** (default) — produces concentric isoline rings at
    evenly spaced (or custom) brightness levels, similar to topographic maps.
    Suitable for photographs and continuous-tone images.

    **Line Art Trace** — directly traces the outlines of black regions in a
    binary image using a single threshold.  Optimised for clean B&W line
    drawings (icons, logos, sketches, technical illustrations) where strokes
    should be reproduced faithfully.  Chaikin corner-cutting smoothing
    converts pixel-stepped boundaries into smooth plotter-friendly curves.

    **Skeleton** — reduces thick ink strokes to single-pixel centerlines via
    morphological thinning (scikit-image ``skeletonize``).  Produces one
    plotter stroke per original drawn line instead of two boundary outlines,
    making it ideal for handwriting, calligraphy, and bold graphic marks.

    **FMM Topographic** — uses the Fast Marching Method to propagate a wave
    from a source point across a speed map derived from the image (dark pixels
    = slow, light pixels = fast).  Isocontours of the resulting travel-time
    field bunch up in dark regions (wave slows down) and spread apart in light
    regions, producing portrait-style topographic line art.  Requires
    ``scikit-fmm`` for full quality; falls back to a
    ``scipy.ndimage.distance_transform_edt`` approximation when unavailable.
    """

    name = "Contour Lines"
    category = "image"

    def get_parameters(self) -> list[Parameter]:
        from .parameters import build_parameters
        return build_parameters()

    def get_presets(self) -> list[Preset]:
        from .presets import build_presets
        return build_presets()

    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        """Generate contour polylines from ``params["_source_image"]``.

        Dispatches to _trace_isolines (Contour Levels mode), _trace_line_art
        (Line Art Trace mode), or _trace_skeleton (Skeleton mode) depending on
        the ``mode`` parameter.
        """
        image = params.get("_source_image")
        if image is None:
            return []

        from plottter.io.image_import import (
            adjust_brightness,
            adjust_contrast,
            apply_blur,
            invert_image,
            to_grayscale,
        )

        # Preprocessing
        img = image.copy()
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        blur_radius = float(params.get("blur_radius", 1.0))
        do_invert = bool(params.get("invert", False))

        if brightness != 0.0:
            img = adjust_brightness(img, brightness)
        if contrast != 0.0:
            img = adjust_contrast(img, contrast)
        if blur_radius > 0.0:
            img = apply_blur(img, blur_radius)
        if do_invert:
            img = invert_image(img)

        gray = to_grayscale(img)  # uint8 H×W

        simplify_mm = float(params.get("simplify_mm", 0.3))
        min_contour_px = int(params.get("min_contour_px", 10))
        mode = str(params.get("mode", "Contour Levels"))

        draw_x1, draw_y1, draw_x2, draw_y2 = canvas.drawing_area()
        img_h, img_w = gray.shape

        img_x1, img_y1, img_x2, img_y2 = compute_image_rect(
            str(params.get("image_fit_mode", "fill")),
            img_w, img_h, draw_x1, draw_y1, draw_x2, draw_y2,
            custom_w_mm=params.get("image_width_mm"),
            custom_h_mm=params.get("image_height_mm"),
            offset_x_mm=float(params.get("image_offset_x_mm", 0.0)),
            offset_y_mm=float(params.get("image_offset_y_mm", 0.0)),
        )

        if mode == "Line Art Trace":
            if progress_callback:
                progress_callback(10)
            if cancelled_callback and cancelled_callback():
                return []

            trace_threshold = int(params.get("trace_threshold", 128))
            smooth_iterations = int(params.get("smooth_iterations", 0))
            fill_style = str(params.get("fill", "None"))
            fill_spacing_mm = float(params.get("fill_spacing_mm", 0.3))
            fill_angle = float(params.get("fill_angle", 45.0))
            adaptive_thresh = bool(params.get("adaptive_threshold", False))
            adaptive_c = float(params.get("adaptive_c", 5.0))
            trace_supersample = int(params.get("trace_supersample", 1))

            if fill_style == "None":
                # Original outline-only behaviour
                result = _trace_line_art(
                    gray,
                    trace_threshold,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    simplify_mm,
                    min_contour_px,
                    smooth_iterations,
                    adaptive_thresh,
                    adaptive_c,
                    trace_supersample,
                )
            else:
                # Extract contours with hierarchy for fill generation
                contour_pairs = _extract_contours_with_hierarchy(
                    gray,
                    trace_threshold,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    simplify_mm,
                    min_contour_px,
                    smooth_iterations,
                    adaptive_thresh,
                    adaptive_c,
                    trace_supersample,
                )

                if progress_callback:
                    progress_callback(40)
                if cancelled_callback and cancelled_callback():
                    return []

                result = []
                # Add outline polylines
                for outer, holes in contour_pairs:
                    result.append(outer)

                if progress_callback:
                    progress_callback(60)

                # Add fill lines for each shape
                total = len(contour_pairs)
                for idx, (outer, holes) in enumerate(contour_pairs):
                    if cancelled_callback and cancelled_callback():
                        break
                    if progress_callback and total > 0:
                        progress_callback(60 + int(idx / total * 35))

                    if fill_style == "Solid":
                        fill_lines = _fill_polygon_hatch(
                            outer, holes, 0.0, fill_spacing_mm
                        )
                        result.extend(fill_lines)
                    elif fill_style == "Hatching":
                        fill_lines = _fill_polygon_hatch(
                            outer, holes, fill_angle, fill_spacing_mm
                        )
                        result.extend(fill_lines)
                    elif fill_style == "Cross-hatch":
                        # Two perpendicular passes
                        fill_lines = _fill_polygon_hatch(
                            outer, holes, fill_angle, fill_spacing_mm
                        )
                        result.extend(fill_lines)
                        fill_lines2 = _fill_polygon_hatch(
                            outer, holes, fill_angle + 90.0, fill_spacing_mm
                        )
                        result.extend(fill_lines2)
                    elif fill_style == "Concentric":
                        fill_lines = _fill_polygon_concentric(
                            outer, holes, fill_spacing_mm
                        )
                        result.extend(fill_lines)

            if bool(params.get("smooth_curves", False)):
                from plottter.processing.curves import fit_curves as _fit_curves
                _tol = float(params.get("curve_tolerance_mm", 0.5))
                result = _fit_curves(result, _tol)

            if progress_callback:
                progress_callback(100)
            x_off = float(params.get("x_offset_mm", 0.0))
            y_off = float(params.get("y_offset_mm", 0.0))
            if x_off != 0.0 or y_off != 0.0:
                result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
            return result

        if mode == "Skeleton":
            if progress_callback:
                progress_callback(10)
            if cancelled_callback and cancelled_callback():
                return []

            trace_threshold = int(params.get("trace_threshold", 128))
            smooth_iterations = int(params.get("smooth_iterations", 0))
            cleanup_kernel = int(params.get("cleanup_kernel", 0))
            merge_gap_mm = float(params.get("merge_gap_mm", 0.5))
            adaptive_thresh = bool(params.get("adaptive_threshold", False))
            adaptive_c = float(params.get("adaptive_c", 5.0))

            result = _trace_skeleton(
                gray,
                trace_threshold,
                cleanup_kernel,
                img_w,
                img_h,
                img_x1,
                img_y1,
                img_x2,
                img_y2,
                simplify_mm,
                min_contour_px,
                smooth_iterations,
                merge_gap_mm,
                adaptive_thresh,
                adaptive_c,
            )

            if bool(params.get("smooth_curves", False)):
                from plottter.processing.curves import fit_curves as _fit_curves
                _tol = float(params.get("curve_tolerance_mm", 0.5))
                result = _fit_curves(result, _tol)

            if progress_callback:
                progress_callback(100)
            x_off = float(params.get("x_offset_mm", 0.0))
            y_off = float(params.get("y_offset_mm", 0.0))
            if x_off != 0.0 or y_off != 0.0:
                result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
            return result

        if mode == "FMM Topographic":
            if progress_callback:
                progress_callback(5)
            if cancelled_callback and cancelled_callback():
                return []

            fmm_num_contours = int(params.get("fmm_num_contours", 20))
            fmm_source_point = str(params.get("fmm_source_point", "Center"))
            fmm_source_x_pct = float(params.get("fmm_source_x_pct", 50.0))
            fmm_source_y_pct = float(params.get("fmm_source_y_pct", 50.0))
            fmm_gamma = float(params.get("fmm_gamma", 1.0))
            fmm_speed_floor = float(params.get("fmm_speed_floor", 0.01))
            fmm_contour_spacing = str(params.get("fmm_contour_spacing", "Linear"))
            fmm_min_contour_length_mm = float(params.get("fmm_min_contour_length_mm", 2.0))
            smooth_iterations = int(params.get("smooth_iterations", 0))
            # speed_map_override is always None — the source image (which may be
            # an AI depth map selected via the image source controls) is used
            # directly as the brightness-based speed map.
            speed_map_override: np.ndarray | None = None

            fmm_render_mode = str(params.get("fmm_render_mode", "Contours"))

            if fmm_render_mode == "Displacement":
                T, _sy, _sx = _compute_fmm_field(
                    gray, fmm_source_point, fmm_gamma, fmm_speed_floor, speed_map_override,
                    fmm_source_x_pct, fmm_source_y_pct,
                )
                if progress_callback:
                    progress_callback(10)
                result = _fmm_displacement(
                    T,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    num_lines=int(params.get("fmm_num_lines", 100)),
                    displacement_mm=float(params.get("fmm_displacement_mm", 5.0)),
                    line_angle_deg=float(params.get("fmm_line_angle", 0.0)),
                    progress_callback=progress_callback,
                    cancelled_callback=cancelled_callback,
                )
            elif fmm_render_mode == "Wave":
                T, _sy, _sx = _compute_fmm_field(
                    gray, fmm_source_point, fmm_gamma, fmm_speed_floor, speed_map_override,
                    fmm_source_x_pct, fmm_source_y_pct,
                )
                if progress_callback:
                    progress_callback(10)
                result = _fmm_wave(
                    T,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    num_lines=int(params.get("fmm_num_lines", 100)),
                    amplitude_mm=float(params.get("fmm_amplitude_mm", 3.0)),
                    frequency=float(params.get("fmm_frequency", 10.0)),
                    line_spacing=str(params.get("fmm_line_spacing", "Uniform")),
                    min_spacing_mm=float(params.get("fmm_min_spacing_mm", 0.5)),
                    max_spacing_mm=float(params.get("fmm_max_spacing_mm", 5.0)),
                    group_size=int(params.get("fmm_group_size", 3)),
                    group_gap_mm=float(params.get("fmm_group_gap_mm", 4.0)),
                    group_intra_spacing_mm=float(params.get("fmm_group_intra_spacing_mm", 0.5)),
                    displacement_variation=float(params.get("fmm_displacement_variation", 0.0)),
                    seed=int(params.get("seed", 0)),
                    progress_callback=progress_callback,
                    cancelled_callback=cancelled_callback,
                )
            elif fmm_render_mode == "Radial":
                T, sy, sx = _compute_fmm_field(
                    gray, fmm_source_point, fmm_gamma, fmm_speed_floor, speed_map_override,
                    fmm_source_x_pct, fmm_source_y_pct,
                )
                if progress_callback:
                    progress_callback(10)
                # Geometric ray origin: for a Custom source, use the unclamped
                # percentage so the origin can sit outside the image (negative
                # or >100%). The FMM field's seed (sy, sx) stays clamped to the
                # grid; only the ray geometry uses the off-image origin.
                if fmm_source_point == "Custom":
                    origin_sx = fmm_source_x_pct / 100.0 * (img_w - 1)
                    origin_sy = fmm_source_y_pct / 100.0 * (img_h - 1)
                else:
                    origin_sx, origin_sy = float(sx), float(sy)
                img_rect_w = img_x2 - img_x1
                step_size_mm = float(params.get("fmm_step_size_mm", 0.5))
                px_per_mm = img_w / img_rect_w if img_rect_w > 0 else 1.0
                step_size_px = max(0.1, step_size_mm * px_per_mm)
                result = _fmm_radial(
                    T,
                    origin_sy,
                    origin_sx,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    num_radials=int(params.get("fmm_num_radials", 120)),
                    step_size_px=step_size_px,
                    progress_callback=progress_callback,
                    cancelled_callback=cancelled_callback,
                )
            else:
                # Default: "Contours" mode — isocontour extraction
                result = _trace_fmm_topographic(
                    gray,
                    img_w,
                    img_h,
                    img_x1,
                    img_y1,
                    img_x2,
                    img_y2,
                    fmm_num_contours,
                    fmm_source_point,
                    fmm_gamma,
                    fmm_speed_floor,
                    fmm_contour_spacing,
                    fmm_min_contour_length_mm,
                    simplify_mm,
                    smooth_iterations,
                    progress_callback,
                    cancelled_callback,
                    speed_map_override=speed_map_override,
                    source_x_pct=fmm_source_x_pct,
                    source_y_pct=fmm_source_y_pct,
                )

            if bool(params.get("smooth_curves", False)):
                from plottter.processing.curves import fit_curves as _fit_curves
                _tol = float(params.get("curve_tolerance_mm", 0.5))
                result = _fit_curves(result, _tol)

            if progress_callback:
                progress_callback(100)
            x_off = float(params.get("x_offset_mm", 0.0))
            y_off = float(params.get("y_offset_mm", 0.0))
            if x_off != 0.0 or y_off != 0.0:
                result = [[(x + x_off, y + y_off) for x, y in path] for path in result]
            return result

        # --- Contour Levels mode (original behaviour) ---
        num_levels = int(params.get("num_levels", 8))
        spacing = str(params.get("spacing", "linear"))
        smooth_iterations = int(params.get("smooth_iterations", 0))

        # Compute threshold values in [1, 254] range
        thresholds = _compute_thresholds(num_levels, spacing)

        all_polylines: list[Polyline] = []
        for i, thr in enumerate(thresholds):
            if cancelled_callback and cancelled_callback():
                break
            if progress_callback:
                progress_callback(int(i / len(thresholds) * 95))

            polys = _trace_isolines(
                gray,
                int(thr),
                img_w,
                img_h,
                img_x1,
                img_y1,
                img_x2,
                img_y2,
                simplify_mm,
                min_contour_px,
                smooth_iterations,
            )
            all_polylines.extend(polys)

        if bool(params.get("smooth_curves", False)):
            from plottter.processing.curves import fit_curves as _fit_curves
            _tol = float(params.get("curve_tolerance_mm", 0.5))
            all_polylines = _fit_curves(all_polylines, _tol)

        if progress_callback:
            progress_callback(100)

        x_off = float(params.get("x_offset_mm", 0.0))
        y_off = float(params.get("y_offset_mm", 0.0))
        if x_off != 0.0 or y_off != 0.0:
            all_polylines = [[(x + x_off, y + y_off) for x, y in path] for path in all_polylines]
        return all_polylines
