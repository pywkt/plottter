"""Phase 16.57 validation: Make AI depth map a reusable image source.

Verifies:
1. SettingsPanel has 'AI Depth Map' radio button in the image source group.
2. Selecting 'AI Depth Map' shows the depth source widget and hides file/layer widgets.
3. _apply_depth_map() converts float32 depth to uint8 RGB and sets _raw_image.
4. _on_depth_map_ready() caches the depth map and applies it to _raw_image.
5. Switching back to 'File' source restores the original _raw_image.
6. _get_settings_snapshot() saves 'image_source_type' = 'depth_map'.
7. _apply_settings_snapshot() restores 'depth_map' source type (selects the radio).
8. fmm_source parameter no longer has 'AI Depth Map' choice.
9. fmm_depth_invert parameter no longer exists in ContourGenerator.
10. No auto-created depth map preview layer from generation metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_project() -> Project:
    canvas = Canvas.from_preset("A4", margin=10.0)
    proj = Project(name="DepthTest", canvas=canvas)
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


@pytest.fixture
def controller(qapp):
    from plottter.gui.project_controller import ProjectController
    return ProjectController(_make_project())


@pytest.fixture
def settings_panel(controller, qtbot):
    from plottter.gui.settings_panel import SettingsPanel
    panel = SettingsPanel(controller)
    qtbot.addWidget(panel)
    panel.show()
    return panel


# ---------------------------------------------------------------------------
# 1. AI Depth Map radio button exists
# ---------------------------------------------------------------------------


class TestDepthMapRadioButton:
    """SettingsPanel has an 'AI Depth Map' radio button in the image source group."""

    def test_depth_radio_button_exists(self, settings_panel) -> None:
        assert hasattr(settings_panel, "_src_type_depth_radio"), (
            "SettingsPanel must have _src_type_depth_radio attribute"
        )
        assert settings_panel._src_type_depth_radio.text() == "AI Depth Map"

    def test_depth_src_widget_exists(self, settings_panel) -> None:
        assert hasattr(settings_panel, "_depth_src_widget"), (
            "SettingsPanel must have _depth_src_widget"
        )

    def test_gen_depth_btn_exists(self, settings_panel) -> None:
        assert hasattr(settings_panel, "_gen_depth_btn")

    def test_depth_status_label_exists(self, settings_panel) -> None:
        assert hasattr(settings_panel, "_depth_status_label")

    def test_depth_invert_check_exists(self, settings_panel) -> None:
        assert hasattr(settings_panel, "_depth_invert_check")


# ---------------------------------------------------------------------------
# 2. Selecting 'AI Depth Map' shows/hides correct widgets
# ---------------------------------------------------------------------------


class TestDepthMapSourceVisibility:
    """Selecting the AI Depth Map radio shows depth widget, hides file/layer."""

    def test_file_source_selected_by_default(self, settings_panel) -> None:
        assert settings_panel._src_type_file_radio.isChecked()
        assert not settings_panel._src_type_depth_radio.isChecked()

    def test_depth_src_widget_hidden_by_default(self, settings_panel) -> None:
        assert not settings_panel._depth_src_widget.isVisible()

    def test_depth_widget_shown_when_depth_radio_selected(
        self, settings_panel, qtbot
    ) -> None:
        settings_panel.on_mode_changed("Image to Lines")
        settings_panel._src_type_depth_radio.setChecked(True)
        assert settings_panel._depth_src_widget.isVisible()
        assert not settings_panel._file_src_widget.isVisible()

    def test_file_widget_shown_when_file_radio_reselected(
        self, settings_panel, qtbot
    ) -> None:
        settings_panel.on_mode_changed("Image to Lines")
        settings_panel._src_type_depth_radio.setChecked(True)
        settings_panel._src_type_file_radio.setChecked(True)
        assert settings_panel._file_src_widget.isVisible()
        assert not settings_panel._depth_src_widget.isVisible()


# ---------------------------------------------------------------------------
# 3. _apply_depth_map converts float32 to uint8 RGB
# ---------------------------------------------------------------------------


class TestApplyDepthMap:
    """_apply_depth_map() converts float32 depth to 3-channel uint8 and sets _raw_image."""

    def test_apply_depth_map_sets_raw_image(self, settings_panel) -> None:
        depth = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        settings_panel._apply_depth_map(depth)

        assert settings_panel._raw_image is not None
        assert settings_panel._raw_image.shape == (1, 3, 3)
        assert settings_panel._raw_image.dtype == np.uint8

    def test_apply_depth_map_channel_values(self, settings_panel) -> None:
        depth = np.array([[0.0, 1.0]], dtype=np.float32)
        settings_panel._apply_depth_map(depth)

        img = settings_panel._raw_image
        # 0.0 → value 0 in all channels
        assert img[0, 0, 0] == 0
        assert img[0, 0, 1] == 0
        assert img[0, 0, 2] == 0
        # 1.0 → value 255 in all channels
        assert img[0, 1, 0] == 255
        assert img[0, 1, 1] == 255
        assert img[0, 1, 2] == 255

    def test_apply_depth_map_all_channels_equal(self, settings_panel) -> None:
        depth = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
        settings_panel._apply_depth_map(depth)
        img = settings_panel._raw_image
        np.testing.assert_array_equal(img[:, :, 0], img[:, :, 1])
        np.testing.assert_array_equal(img[:, :, 0], img[:, :, 2])


# ---------------------------------------------------------------------------
# 4. _on_depth_map_ready() caches and applies depth map
# ---------------------------------------------------------------------------


class TestOnDepthMapReady:
    """_on_depth_map_ready() caches the depth map and applies it."""

    def test_depth_map_cached_after_ready(self, settings_panel) -> None:
        settings_panel._image_source_path = "/test/image.jpg"
        depth = np.ones((4, 4), dtype=np.float32) * 0.5
        settings_panel._on_depth_map_ready(depth)

        key = "/test/image.jpg"
        assert key in settings_panel._depth_map_cache
        cached = settings_panel._depth_map_cache[key]
        np.testing.assert_array_equal(cached, depth)

    def test_depth_map_applied_to_raw_image_after_ready(self, settings_panel) -> None:
        settings_panel._image_source_path = "/test/image.jpg"
        depth = np.ones((4, 4), dtype=np.float32) * 0.5
        settings_panel._on_depth_map_ready(depth)

        assert settings_panel._raw_image is not None
        assert settings_panel._raw_image.shape == (4, 4, 3)

    def test_status_label_updated_after_ready(self, settings_panel) -> None:
        settings_panel._image_source_path = "/test/image.jpg"
        depth = np.zeros((4, 4), dtype=np.float32)
        settings_panel._on_depth_map_ready(depth)
        assert "ready" in settings_panel._depth_status_label.text().lower()


# ---------------------------------------------------------------------------
# 5. Switching back to 'File' restores original _raw_image
# ---------------------------------------------------------------------------


class TestSwitchBackToFile:
    """Switching from depth_map back to file mode restores the original image."""

    def test_original_image_restored_on_file_switch(self, settings_panel) -> None:
        # Set up an original image and switch to depth map mode
        original = np.zeros((8, 8, 3), dtype=np.uint8)
        original[:, :, 0] = 100  # distinctive red channel
        settings_panel._raw_image = original
        settings_panel._src_type_depth_radio.setChecked(True)
        # original should be saved
        assert settings_panel._original_raw_image is not None

        # Apply a depth map (replacing _raw_image)
        depth = np.ones((8, 8), dtype=np.float32) * 0.5
        settings_panel._apply_depth_map(depth)
        # _raw_image now has the depth map
        assert settings_panel._raw_image[0, 0, 0] != 100

        # Switch back to file mode
        settings_panel._src_type_file_radio.setChecked(True)
        # Original should be restored
        assert settings_panel._raw_image is not None
        assert settings_panel._raw_image[0, 0, 0] == 100


# ---------------------------------------------------------------------------
# 6–7. Snapshot save/restore for 'depth_map' source type
# ---------------------------------------------------------------------------


class TestSnapshotDepthMapSourceType:
    """_get_settings_snapshot saves image_source_type; _apply_settings_snapshot restores it."""

    def test_snapshot_saves_depth_map_source_type(self, settings_panel, controller) -> None:
        # Switch to Image to Lines mode first so snapshot is valid
        settings_panel.on_mode_changed("Image to Lines")
        settings_panel._image_source_type = "depth_map"
        settings_panel._depth_invert_check.setChecked(True)

        snapshot = settings_panel._get_settings_snapshot()
        assert snapshot is not None
        assert snapshot.get("image_source_type") == "depth_map"
        assert snapshot.get("depth_map_invert") is True

    def test_snapshot_restores_depth_map_radio(self, settings_panel) -> None:
        snapshot = {
            "mode": "Image to Lines",
            "generator_name": "Edge Detect / Contour",
            "params": {},
            "transforms": {},
            "image_source_type": "depth_map",
            "depth_map_invert": False,
        }
        settings_panel.on_mode_changed("Image to Lines")
        settings_panel._apply_settings_snapshot(snapshot)
        assert settings_panel._src_type_depth_radio.isChecked(), (
            "_apply_settings_snapshot should restore the AI Depth Map radio button"
        )

    def test_snapshot_restores_depth_map_invert_state(self, settings_panel) -> None:
        snapshot = {
            "mode": "Image to Lines",
            "generator_name": "Edge Detect / Contour",
            "params": {},
            "transforms": {},
            "image_source_type": "depth_map",
            "depth_map_invert": True,
        }
        settings_panel.on_mode_changed("Image to Lines")
        settings_panel._apply_settings_snapshot(snapshot)
        assert settings_panel._depth_invert_check.isChecked(), (
            "_apply_settings_snapshot should restore depth_map_invert=True"
        )

    def test_snapshot_restores_file_source_when_saved(self, settings_panel) -> None:
        snapshot = {
            "mode": "Image to Lines",
            "generator_name": "Edge Detect / Contour",
            "params": {},
            "transforms": {},
            "image_source_type": "file",
        }
        settings_panel.on_mode_changed("Image to Lines")
        # First switch to depth_map to ensure restore works
        settings_panel._src_type_depth_radio.setChecked(True)
        settings_panel._apply_settings_snapshot(snapshot)
        assert settings_panel._src_type_file_radio.isChecked(), (
            "_apply_settings_snapshot should restore the File radio button"
        )


# ---------------------------------------------------------------------------
# 8–9. ContourGenerator no longer has deprecated depth map params
# ---------------------------------------------------------------------------


class TestContourGeneratorParams:
    """ContourGenerator no longer has AI Depth Map fmm_source or fmm_depth_invert."""

    def test_fmm_source_param_removed(self) -> None:
        from plottter.generators.contour import ContourGenerator
        gen = ContourGenerator()
        fmm_source = next(
            (p for p in gen.get_parameters() if p.name == "fmm_source"), None
        )
        assert fmm_source is None, (
            "fmm_source param was fully removed — it had only one choice and was never read"
        )

    def test_fmm_depth_invert_not_in_parameters(self) -> None:
        from plottter.generators.contour import ContourGenerator
        gen = ContourGenerator()
        param_names = {p.name for p in gen.get_parameters()}
        assert "fmm_depth_invert" not in param_names, (
            "fmm_depth_invert was removed in task 16.57"
        )

    def test_depth_portrait_preset_has_no_fmm_source_key(self) -> None:
        from plottter.generators.contour import ContourGenerator
        gen = ContourGenerator()
        presets = {p.name: p for p in gen.get_presets()}
        assert "FMM Depth Portrait" in presets
        assert "fmm_source" not in presets["FMM Depth Portrait"].params, (
            "fmm_source key was removed from presets along with the parameter"
        )

    def test_depth_landscape_preset_has_no_fmm_source_key(self) -> None:
        from plottter.generators.contour import ContourGenerator
        gen = ContourGenerator()
        presets = {p.name: p for p in gen.get_presets()}
        assert "FMM Depth Landscape" in presets
        assert "fmm_source" not in presets["FMM Depth Landscape"].params, (
            "fmm_source key was removed from presets along with the parameter"
        )


# ---------------------------------------------------------------------------
# 10. No auto-created depth map preview layer from generation metadata
# ---------------------------------------------------------------------------


class TestNoAutoDepthMapLayer:
    """_on_generation_metadata should not create a depth map preview layer."""

    def test_generation_metadata_does_not_create_layer(
        self, settings_panel, controller
    ) -> None:
        """Calling _on_generation_metadata with a depth_map should NOT add a layer."""
        initial_layer_count = len(controller.current_project.layers)
        depth = np.ones((8, 8), dtype=np.float32) * 0.5
        # Call with the old-style metadata dict that used to trigger layer creation
        settings_panel._on_generation_metadata({"depth_map": depth}, "dummy-layer-id")
        assert len(controller.current_project.layers) == initial_layer_count, (
            "_on_generation_metadata must not auto-create a depth map preview layer"
        )
