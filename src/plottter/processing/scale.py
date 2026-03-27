"""Scale paths from one canvas drawing area to another."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plottter.models import Canvas
    from plottter.models.path import Polyline


def scale_paths_to_canvas(
    paths: list[Polyline],
    old_canvas: Canvas,
    new_canvas: Canvas,
) -> list[Polyline]:
    """Scale polylines from old_canvas drawing area to new_canvas drawing area.

    Each point (x, y) is transformed using:
        new_x = new_margin + (x - old_margin) * (new_draw_w / old_draw_w)
        new_y = new_margin + (y - old_margin) * (new_draw_h / old_draw_h)

    If the old drawing area has zero width or height, points are returned as-is.
    """
    old_left, old_top, old_right, old_bottom = old_canvas.drawing_area()
    new_left, new_top, new_right, new_bottom = new_canvas.drawing_area()

    old_draw_w = old_right - old_left
    old_draw_h = old_bottom - old_top
    new_draw_w = new_right - new_left
    new_draw_h = new_bottom - new_top

    if old_draw_w == 0.0 or old_draw_h == 0.0:
        return [list(poly) for poly in paths]

    sx = new_draw_w / old_draw_w
    sy = new_draw_h / old_draw_h

    result: list[Polyline] = []
    for polyline in paths:
        scaled = [
            (new_left + (x - old_left) * sx, new_top + (y - old_top) * sy)
            for x, y in polyline
        ]
        result.append(scaled)
    return result
