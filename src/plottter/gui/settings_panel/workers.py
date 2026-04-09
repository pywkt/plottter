"""Worker QThread classes used by SettingsPanel."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal as _pyqtSignal
from PyQt6.QtWidgets import QLabel


class _AiBgWorker(QThread):
    """Runs ReplicateClient.remove_background() off the main GUI thread."""

    progress = _pyqtSignal(int)
    finished = _pyqtSignal(object)  # emits RGBA np.ndarray
    error = _pyqtSignal(str)

    def __init__(
        self, api_key: str, image: "np.ndarray", cache_dir: "str | None" = None, parent: Any = None
    ) -> None:
        super().__init__(parent)
        self._api_key = api_key
        self._image = image
        self._cache_dir = cache_dir

    def run(self) -> None:
        try:
            from plottter.ai.replicate_client import ReplicateClient

            client = ReplicateClient(api_key=self._api_key, cache_dir=self._cache_dir)
            result = client.remove_background(
                self._image, progress_callback=self.progress.emit
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _AiSegmentWorker(QThread):
    """Runs ReplicateClient.segment_image() off the main GUI thread."""

    progress = _pyqtSignal(int)
    finished = _pyqtSignal(list)  # emits list[(mask, hex_color)]
    error = _pyqtSignal(str)

    def __init__(
        self, api_key: str, image: "np.ndarray", num_segments: int, parent: Any = None
    ) -> None:
        super().__init__(parent)
        self._api_key = api_key
        self._image = image
        self._num_segments = num_segments

    def run(self) -> None:
        try:
            from plottter.ai.replicate_client import ReplicateClient

            client = ReplicateClient(api_key=self._api_key)
            results = client.segment_image(
                self._image,
                num_segments=self._num_segments,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class _AiMaskWorker(QThread):
    """Runs AI point/box/text mask generation off the main GUI thread.

    Args:
        api_key: Replicate API key.
        image: RGB source image (H×W×3 uint8) to segment.
        mode: ``'point'``, ``'box'``, or ``'text'``.
        positive_points: (x_mm, y_mm) foreground points (point mode).
        negative_points: (x_mm, y_mm) background points (point mode).
        box_xyxy_mm: Bounding box as (x1, y1, x2, y2) in mm (box mode).
        text_prompt: Natural-language object description (text mode).
        canvas_width_mm: Canvas width in mm (used for mm→pixel conversion).
        canvas_height_mm: Canvas height in mm (used for mm→pixel conversion).
    """

    progress = _pyqtSignal(int)
    finished = _pyqtSignal(object)  # emits binary mask (H×W uint8)
    error = _pyqtSignal(str)

    def __init__(
        self,
        api_key: str,
        image: "np.ndarray",
        mode: str,
        positive_points: list | None = None,
        negative_points: list | None = None,
        box_xyxy_mm: tuple | None = None,
        text_prompt: str = "",
        canvas_width_mm: float = 0.0,
        canvas_height_mm: float = 0.0,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._api_key = api_key
        self._image = image
        self._mode = mode
        self._positive_points = positive_points or []
        self._negative_points = negative_points or []
        self._box_xyxy_mm = box_xyxy_mm
        self._text_prompt = text_prompt
        self._canvas_width_mm = canvas_width_mm
        self._canvas_height_mm = canvas_height_mm

    def run(self) -> None:
        try:
            from plottter.ai.replicate_client import ReplicateClient

            client = ReplicateClient(api_key=self._api_key)
            img_h, img_w = self._image.shape[:2]

            # Scale factor: mm → image pixels
            # Assumes the image fills the full canvas area.
            sx = img_w / self._canvas_width_mm if self._canvas_width_mm > 0 else 1.0
            sy = img_h / self._canvas_height_mm if self._canvas_height_mm > 0 else 1.0

            if self._mode == "point":
                pos_px = [(int(x * sx), int(y * sy)) for x, y in self._positive_points]
                neg_px = [(int(x * sx), int(y * sy)) for x, y in self._negative_points]
                mask = client.segment_by_point(
                    self._image,
                    pos_px,
                    neg_px if neg_px else None,
                    progress_callback=self.progress.emit,
                )
            elif self._mode == "box":
                if self._box_xyxy_mm is None:
                    raise ValueError("No bounding box specified.")
                x1, y1, x2, y2 = self._box_xyxy_mm
                box_px = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                mask = client.segment_by_box(
                    self._image,
                    box_px,
                    progress_callback=self.progress.emit,
                )
            elif self._mode == "text":
                if not self._text_prompt.strip():
                    raise ValueError("Text prompt must not be empty.")
                mask = client.segment_by_text(
                    self._image,
                    self._text_prompt,
                    progress_callback=self.progress.emit,
                )
            else:
                raise ValueError(f"Unknown AI mask mode: {self._mode!r}")

            self.finished.emit(mask)
        except Exception as exc:
            self.error.emit(str(exc))


class _DepthMapWorker(QThread):
    """Runs ReplicateClient.estimate_depth() off the main GUI thread."""

    progress = _pyqtSignal(int)
    finished = _pyqtSignal(object)  # emits float32 np.ndarray (H×W)
    error = _pyqtSignal(str)

    def __init__(
        self, api_key: str, cache_dir: "str | None", image: "np.ndarray"
    ) -> None:
        super().__init__()  # no parent — prevent Qt parent-child destruction
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._image = image

    def run(self) -> None:
        try:
            from plottter.ai.replicate_client import ReplicateClient

            client = ReplicateClient(api_key=self._api_key, cache_dir=self._cache_dir)
            result = client.estimate_depth(
                self._image, progress_callback=self.progress.emit
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _VisibilityTrackedLabel(QLabel):
    """QLabel whose isVisible() returns the intended visibility set via setVisible().

    Qt's isVisible() requires the entire parent hierarchy to be shown, which makes
    it unsuitable for unit tests that don't show the top-level window. This subclass
    tracks the explicit show/hide state so isVisible() works independently of the
    parent chain's mapped state.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._explicitly_visible: bool = True  # QWidget default: not hidden until told

    def setVisible(self, visible: bool) -> None:  # type: ignore[override]
        self._explicitly_visible = visible
        super().setVisible(visible)

    def isVisible(self) -> bool:  # type: ignore[override]
        return self._explicitly_visible


class _WireframeWorker(QThread):
    """Render all 3D layers without HLR for a fast wireframe preview.

    Collects all 3D layers' shape params, builds the scene with
    ``hlr_enabled=False``, renders, and emits the projected 2D polylines.

    IMPORTANT: Custom result signals are named ``result_ready`` / ``render_error``
    so they do NOT shadow ``QThread.finished`` — Qt needs the built-in
    ``finished`` signal to work correctly for thread lifecycle management.
    """

    result_ready = _pyqtSignal(list)  # list[Polyline]
    render_error = _pyqtSignal(str)

    def __init__(
        self,
        layer_params_list: list[dict],
        camera_dict: dict,
        canvas_w_mm: float,
        canvas_h_mm: float,
    ) -> None:
        super().__init__()  # no parent — prevent Qt parent-child destruction of running thread
        self._layer_params_list = layer_params_list
        self._camera_dict = camera_dict
        self._canvas_w_mm = canvas_w_mm
        self._canvas_h_mm = canvas_h_mm
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation — checked between expensive steps."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            from plottter.scene3d import Scene, Camera
            from plottter.generators.scene3d_generator import Scene3DGenerator
            from plottter.generators.mesh_slicer import MeshSlicerGenerator

            if self._cancelled:
                return

            cam_dict = self._camera_dict
            aspect = self._canvas_w_mm / max(self._canvas_h_mm, 1e-6)

            camera = Camera(
                projection=cam_dict.get("projection", "perspective"),
                fov_deg=float(cam_dict.get("fov", 45.0)),
                aspect=aspect,
            )
            camera.set_orbit(
                azimuth_deg=float(cam_dict.get("azimuth", 30.0)),
                elevation_deg=float(cam_dict.get("elevation", 20.0)),
                distance=float(cam_dict.get("distance", 8.0)),
                center=[
                    float(cam_dict.get("look_at_x", 0.0)),
                    float(cam_dict.get("look_at_y", 0.0)),
                    float(cam_dict.get("look_at_z", 0.0)),
                ],
            )

            if self._cancelled:
                return

            # Separate mesh-slicer layers (identified by "mesh_file" key) from
            # Scene3DGenerator shape layers.
            scene3d_params: list[dict] = []
            mesh_slicer_params: list[dict] = []
            for params in self._layer_params_list:
                if "mesh_file" in params:
                    mesh_slicer_params.append(params)
                else:
                    scene3d_params.append(params)

            all_polylines: list = []

            # --- Scene3DGenerator layers (shapes: sphere, cube, …) ---
            if scene3d_params:
                gen = Scene3DGenerator()
                scene = Scene(hlr_enabled=False)
                for params in scene3d_params:
                    if self._cancelled:
                        return
                    shape = gen.build_transformed_shape(params)
                    if shape is not None:
                        scene.add(shape)

                if scene.shapes:
                    if self._cancelled:
                        return
                    polylines = scene.render(
                        camera,
                        canvas_w_mm=self._canvas_w_mm,
                        canvas_h_mm=self._canvas_h_mm,
                    )
                    all_polylines.extend(polylines)

            # --- MeshSlicerGenerator layers (fast non-HLR slice projection) ---
            for params in mesh_slicer_params:
                if self._cancelled:
                    return
                polylines = MeshSlicerGenerator.preview_wireframe(
                    params, camera, self._canvas_w_mm, self._canvas_h_mm,
                )
                all_polylines.extend(polylines)

            if not self._cancelled:
                self.result_ready.emit(all_polylines)
        except Exception as exc:  # noqa: BLE001
            if not self._cancelled:
                self.render_error.emit(str(exc))
