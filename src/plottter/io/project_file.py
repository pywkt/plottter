"""Save and load Plottter project files (.plottter JSON, optionally gzip-compressed)."""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any

from plottter.models.canvas import Canvas
from plottter.models.layer import Layer
from plottter.models.path import Polyline
from plottter.models.project import Project

_FORMAT_VERSION = 1
_GZIP_THRESHOLD_BYTES = 1_000_000  # 1 MB


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_project(project: Project, filepath: str) -> None:
    """Serialize *project* to JSON and write to *filepath*.

    The file is gzip-compressed if the serialized size exceeds 1 MB.
    The file extension is not enforced here so callers can pass any path.
    """
    data = _project_to_dict(project)
    payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
    if len(payload) > _GZIP_THRESHOLD_BYTES:
        with gzip.open(filepath, "wb") as fh:
            fh.write(payload)
    else:
        with open(filepath, "wb") as fh:
            fh.write(payload)


def load_project(filepath: str) -> Project:
    """Deserialize a project from *filepath* (auto-detects gzip vs plain JSON)."""
    raw = _read_bytes(filepath)
    data = json.loads(raw)
    return _dict_to_project(data)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _project_to_dict(project: Project) -> dict[str, Any]:
    masks = [
        {"name": name, "data": base64.b64encode(png_bytes).decode("ascii")}
        for name, png_bytes in project.masks.items()
    ]
    return {
        "version": _FORMAT_VERSION,
        "name": project.name,
        "canvas": _canvas_to_dict(project.canvas),
        "layers": [_layer_to_dict(l) for l in project.layers],
        "registration_marks": project.registration_marks,
        "reg_mark_style": project.reg_mark_style,
        "metadata": project.metadata,
        "masks": masks,
    }


def _canvas_to_dict(canvas: Canvas) -> dict[str, Any]:
    return {
        "width_mm": canvas.width_mm,
        "height_mm": canvas.height_mm,
        "margin_mm": canvas.margin_mm,
        "paper_preset": canvas.paper_preset,
    }


def _layer_to_dict(layer: Layer) -> dict[str, Any]:
    return {
        "id": layer.id,
        "name": layer.name,
        "color": layer.color,
        "paths": [[[pt[0], pt[1]] for pt in path] for path in layer.paths],
        "visible": layer.visible,
        "locked": layer.locked,
        "opacity": layer.opacity,
        "generator_info": layer.generator_info,
    }


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------


def _dict_to_project(data: dict[str, Any]) -> Project:
    canvas = _dict_to_canvas(data["canvas"])
    layers = [_dict_to_layer(l) for l in data.get("layers", [])]
    masks: dict[str, bytes] = {
        entry["name"]: base64.b64decode(entry["data"])
        for entry in data.get("masks", [])
    }
    return Project(
        name=data.get("name", "Untitled"),
        canvas=canvas,
        layers=layers,
        registration_marks=data.get("registration_marks", True),
        reg_mark_style=data.get("reg_mark_style", "corners"),
        metadata=dict(data.get("metadata", {})),
        masks=masks,
    )


def _dict_to_canvas(data: dict[str, Any]) -> Canvas:
    return Canvas(
        width_mm=float(data["width_mm"]),
        height_mm=float(data["height_mm"]),
        margin_mm=float(data.get("margin_mm", 10.0)),
        paper_preset=data.get("paper_preset", "Custom"),
    )


def _dict_to_layer(data: dict[str, Any]) -> Layer:
    raw_paths = data.get("paths", [])
    paths: list[Polyline] = [
        [(float(pt[0]), float(pt[1])) for pt in path] for path in raw_paths
    ]
    layer = Layer(
        name=data.get("name", "Layer"),
        color=data.get("color", "#000000"),
        paths=paths,
        visible=data.get("visible", True),
        locked=data.get("locked", False),
        opacity=data.get("opacity", 1.0),
        generator_info=data.get("generator_info"),
    )
    # Restore the original UUID if present
    if "id" in data:
        layer.id = data["id"]
    return layer


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------


def _read_bytes(filepath: str) -> bytes:
    """Read file, auto-detecting gzip by magic bytes."""
    with open(filepath, "rb") as fh:
        header = fh.read(2)
        fh.seek(0)
        if header == b"\x1f\x8b":
            with gzip.open(filepath, "rb") as gz:
                return gz.read()
        return fh.read()
