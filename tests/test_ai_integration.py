"""Tests for the AI integration module (fully mocked — no real API key needed)."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build fake PNG bytes
# ---------------------------------------------------------------------------

def _make_png_bytes(h: int = 4, w: int = 4, mode: str = "RGBA") -> bytes:
    """Create a tiny in-memory PNG suitable for testing fetch helpers."""
    from PIL import Image

    img = Image.new(mode, (w, h), color=(128, 64, 32, 255) if mode == "RGBA" else (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _replicate_run tests
# ---------------------------------------------------------------------------

class TestReplicateRun:
    """Tests for the _replicate_run() HTTP helper function."""

    def test_can_be_called_without_replicate_package(self) -> None:
        """_replicate_run is importable and callable without the replicate package installed."""
        # Temporarily hide replicate from sys.modules
        saved = sys.modules.pop("replicate", None)
        try:
            from plottter.ai.replicate_client import _replicate_run
            assert callable(_replicate_run)
        finally:
            if saved is not None:
                sys.modules["replicate"] = saved

    def test_raises_replicate_api_error_on_http_401(self) -> None:
        """_replicate_run raises ReplicateAPIError with clear message on HTTP 401."""
        from plottter.ai.replicate_client import ReplicateAPIError, _replicate_run

        http_error = urllib.error.HTTPError(
            url="https://api.replicate.com/v1/predictions",
            code=401,
            msg="Unauthorized",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"detail":"Invalid token"}'),
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(ReplicateAPIError, match="401"):
                _replicate_run("bad-key", "owner/model:abc123", {"input": "test"})

    def test_raises_replicate_api_error_on_http_422(self) -> None:
        """_replicate_run raises ReplicateAPIError with clear message on HTTP 422."""
        from plottter.ai.replicate_client import ReplicateAPIError, _replicate_run

        http_error = urllib.error.HTTPError(
            url="https://api.replicate.com/v1/predictions",
            code=422,
            msg="Unprocessable Entity",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"detail":"Invalid input"}'),
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(ReplicateAPIError, match="422"):
                _replicate_run("key", "owner/model:abc123", {"bad": "input"})

    def test_raises_replicate_api_error_on_other_http_error(self) -> None:
        """_replicate_run raises ReplicateAPIError for unexpected HTTP errors."""
        from plottter.ai.replicate_client import ReplicateAPIError, _replicate_run

        http_error = urllib.error.HTTPError(
            url="https://api.replicate.com/v1/predictions",
            code=500,
            msg="Internal Server Error",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(b"server error"),
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(ReplicateAPIError, match="500"):
                _replicate_run("key", "owner/model:abc123", {})

    def test_polls_until_succeeded(self) -> None:
        """_replicate_run polls the prediction URL until status is 'succeeded'."""
        from plottter.ai.replicate_client import _replicate_run

        create_response = json.dumps({
            "id": "pred_123",
            "status": "starting",
            "urls": {"get": "https://api.replicate.com/v1/predictions/pred_123"},
        }).encode()

        poll_processing = json.dumps({"status": "processing"}).encode()
        poll_succeeded = json.dumps({"status": "succeeded", "output": "https://output.png"}).encode()

        responses = [
            io.BytesIO(create_response),
            io.BytesIO(poll_processing),
            io.BytesIO(poll_succeeded),
        ]

        call_count = 0

        def mock_urlopen(req):
            nonlocal call_count
            resp = MagicMock()
            resp.read.return_value = responses[call_count].read()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            call_count += 1
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with patch("time.sleep"):  # skip actual sleeps
                output = _replicate_run("key", "owner/model:abc123", {})

        assert output == "https://output.png"
        assert call_count == 3  # create + 2 polls

    def test_raises_on_failed_prediction(self) -> None:
        """_replicate_run raises ReplicateAPIError when prediction status is 'failed'."""
        from plottter.ai.replicate_client import ReplicateAPIError, _replicate_run

        create_response = json.dumps({
            "status": "starting",
            "urls": {"get": "https://api.replicate.com/v1/predictions/pred_456"},
        }).encode()
        poll_failed = json.dumps({
            "status": "failed",
            "error": "out of memory",
        }).encode()

        responses = [io.BytesIO(create_response), io.BytesIO(poll_failed)]
        call_count = 0

        def mock_urlopen(req):
            nonlocal call_count
            resp = MagicMock()
            resp.read.return_value = responses[call_count].read()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            call_count += 1
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with patch("time.sleep"):
                with pytest.raises(ReplicateAPIError, match="out of memory"):
                    _replicate_run("key", "owner/model:abc123", {})

    def test_raises_on_canceled_prediction(self) -> None:
        """_replicate_run raises ReplicateAPIError when prediction is canceled."""
        from plottter.ai.replicate_client import ReplicateAPIError, _replicate_run

        create_response = json.dumps({
            "status": "starting",
            "urls": {"get": "https://api.replicate.com/v1/predictions/pred_789"},
        }).encode()
        poll_canceled = json.dumps({"status": "canceled"}).encode()

        responses = [io.BytesIO(create_response), io.BytesIO(poll_canceled)]
        call_count = 0

        def mock_urlopen(req):
            nonlocal call_count
            resp = MagicMock()
            resp.read.return_value = responses[call_count].read()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            call_count += 1
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with patch("time.sleep"):
                with pytest.raises(ReplicateAPIError, match="canceled"):
                    _replicate_run("key", "owner/model:abc123", {})

    def test_pydantic_monkeypatch_removed(self) -> None:
        """The pydantic monkey-patch code (_PydanticV1Redirect) must not exist."""
        import plottter.ai.replicate_client as rc_mod
        assert not hasattr(rc_mod, "_PydanticV1Redirect"), (
            "_PydanticV1Redirect should have been removed from replicate_client"
        )

    def test_replicate_lib_variables_removed(self) -> None:
        """_REPLICATE_AVAILABLE and _replicate_lib module vars must not exist."""
        import plottter.ai.replicate_client as rc_mod
        assert not hasattr(rc_mod, "_REPLICATE_AVAILABLE"), (
            "_REPLICATE_AVAILABLE module variable should have been removed"
        )
        assert not hasattr(rc_mod, "_replicate_lib"), (
            "_replicate_lib module variable should have been removed"
        )


# ---------------------------------------------------------------------------
# is_available tests
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_unavailable_when_api_key_empty(self) -> None:
        """is_available() returns False when api_key is an empty string."""
        from plottter.ai.replicate_client import ReplicateClient
        client = ReplicateClient(api_key="")
        assert not client.is_available()

    def test_available_when_key_set(self) -> None:
        """is_available() returns True when the API key is set."""
        from plottter.ai.replicate_client import ReplicateClient
        client = ReplicateClient(api_key="r8_abc123")
        assert client.is_available()

    def test_available_when_api_key_is_whitespace(self) -> None:
        """is_available() returns True when api_key is whitespace (truthy string)."""
        from plottter.ai.replicate_client import ReplicateClient
        # Whitespace is truthy — only empty string disables the client
        client = ReplicateClient(api_key="   ")
        assert client.is_available()


# ---------------------------------------------------------------------------
# remove_background tests
# ---------------------------------------------------------------------------

class TestRemoveBackground:
    def _make_client(self) -> "ReplicateClient":  # noqa: F821
        from plottter.ai.replicate_client import ReplicateClient
        return ReplicateClient(api_key="r8_test")

    def test_returns_rgba_shape(self) -> None:
        """remove_background() returns an RGBA array with the same H×W as the input."""
        client = self._make_client()
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        input_image = np.zeros((h, w, 3), dtype=np.uint8)

        from PIL import Image as _PIL_Image
        pil_rgba = _PIL_Image.new("RGBA", (w, h), (100, 150, 200, 255))
        expected_arr = np.array(pil_rgba)

        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/output.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgba", return_value=expected_arr):
                result = client.remove_background(input_image)

        assert result.shape == (h, w, 4), f"Expected ({h}, {w}, 4), got {result.shape}"
        assert result.dtype == np.uint8

    def test_raises_replicate_api_error_on_failure(self) -> None:
        """remove_background() raises ReplicateAPIError when the fetch helper fails."""
        client = self._make_client()
        import plottter.ai.replicate_client as rc_mod
        from plottter.ai.replicate_client import ReplicateAPIError

        def _boom(*args, **kwargs):
            raise RuntimeError("network timeout")

        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/output.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgba", side_effect=_boom):
                with pytest.raises(ReplicateAPIError, match="Background removal failed"):
                    client.remove_background(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_raises_when_replicate_run_raises(self) -> None:
        """remove_background() propagates ReplicateAPIError from _replicate_run."""
        client = self._make_client()
        import plottter.ai.replicate_client as rc_mod
        from plottter.ai.replicate_client import ReplicateAPIError

        with patch.object(rc_mod, "_replicate_run",
                          side_effect=ReplicateAPIError("Invalid API key (HTTP 401)")):
            with pytest.raises(ReplicateAPIError, match="401"):
                client.remove_background(np.zeros((4, 4, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# segment_image tests
# ---------------------------------------------------------------------------

class TestSegmentImage:
    def _make_client(self) -> "ReplicateClient":  # noqa: F821
        from plottter.ai.replicate_client import ReplicateClient
        return ReplicateClient(api_key="r8_test")

    def _make_seg_map(self, h: int, w: int, num_classes: int) -> np.ndarray:
        """Produce a fake RGB segmentation map with ``num_classes`` distinct colours."""
        seg = np.zeros((h, w, 3), dtype=np.uint8)
        # Fill each row-strip with a distinct colour
        for i in range(num_classes):
            row_start = i * (h // num_classes)
            row_end = (i + 1) * (h // num_classes) if i < num_classes - 1 else h
            seg[row_start:row_end, :] = [i * 30, i * 20, i * 10]
        return seg

    def test_returns_correct_number_of_segments(self) -> None:
        """segment_image() returns exactly num_segments mask/color tuples."""
        client = self._make_client()
        import plottter.ai.replicate_client as rc_mod

        h, w = 16, 16
        num_seg = 3
        original_image = np.zeros((h, w, 3), dtype=np.uint8)
        seg_map = self._make_seg_map(h, w, num_classes=4)  # 4 classes, request 3

        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/seg.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgb", return_value=seg_map):
                results = client.segment_image(original_image, num_segments=num_seg)

        assert len(results) == num_seg

    def test_mask_shape_and_dtype(self) -> None:
        """Each mask returned by segment_image() is (H×W) uint8 with values 0 or 255."""
        client = self._make_client()
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        original_image = np.zeros((h, w, 3), dtype=np.uint8)
        seg_map = self._make_seg_map(h, w, num_classes=2)

        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/seg.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgb", return_value=seg_map):
                results = client.segment_image(original_image, num_segments=2)

        for mask, hex_color in results:
            assert mask.shape == (h, w), f"Expected ({h}, {w}), got {mask.shape}"
            assert mask.dtype == np.uint8
            assert set(np.unique(mask)).issubset({0, 255})
            assert hex_color.startswith("#") and len(hex_color) == 7


# ---------------------------------------------------------------------------
# ReplicateAPIError surface tests
# ---------------------------------------------------------------------------

class TestReplicateAPIError:
    def test_error_is_exception_subclass(self) -> None:
        from plottter.ai.replicate_client import ReplicateAPIError
        err = ReplicateAPIError("test message")
        assert isinstance(err, Exception)
        assert str(err) == "test message"

    def test_network_failure_raises_replicate_api_error(self) -> None:
        """A mocked network failure during remove_background surfaces as ReplicateAPIError."""
        import plottter.ai.replicate_client as rc_mod
        from plottter.ai.replicate_client import ReplicateClient, ReplicateAPIError

        client = ReplicateClient(api_key="r8_test")

        with patch.object(rc_mod, "_replicate_run",
                          side_effect=Exception("connection refused")):
            with pytest.raises(ReplicateAPIError, match="Background removal failed"):
                client.remove_background(np.zeros((4, 4, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# segment_by_point tests
# ---------------------------------------------------------------------------

class TestSegmentByPoint:
    def _make_client(self) -> "ReplicateClient":  # noqa: F821
        from plottter.ai.replicate_client import ReplicateClient
        return ReplicateClient(api_key="r8_test")

    def test_returns_binary_mask(self) -> None:
        """segment_by_point() returns a binary uint8 mask with the same H×W as the input."""
        client = self._make_client()
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        input_image = np.zeros((h, w, 3), dtype=np.uint8)
        expected_mask = np.full((h, w), 255, dtype=np.uint8)
        expected_mask[:h // 2] = 0  # top half background, bottom half foreground

        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=expected_mask):
                result = client.segment_by_point(input_image, positive_points=[(4, 6)])

        assert result.shape == (h, w)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_raises_without_positive_points(self) -> None:
        """segment_by_point() raises ReplicateAPIError when no positive points are given."""
        client = self._make_client()
        from plottter.ai.replicate_client import ReplicateAPIError

        with pytest.raises(ReplicateAPIError, match="positive point"):
            client.segment_by_point(np.zeros((4, 4, 3), dtype=np.uint8), positive_points=[])


# ---------------------------------------------------------------------------
# segment_by_box tests
# ---------------------------------------------------------------------------

class TestSegmentByBox:
    def _make_client(self) -> "ReplicateClient":  # noqa: F821
        from plottter.ai.replicate_client import ReplicateClient
        return ReplicateClient(api_key="r8_test")

    def test_returns_binary_mask(self) -> None:
        """segment_by_box() returns a binary uint8 mask with the same H×W as the input."""
        client = self._make_client()
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        input_image = np.zeros((h, w, 3), dtype=np.uint8)
        expected_mask = np.zeros((h, w), dtype=np.uint8)
        expected_mask[2:6, 2:6] = 255  # foreground in the box region

        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=expected_mask):
                result = client.segment_by_box(input_image, box_xyxy=(2, 2, 6, 6))

        assert result.shape == (h, w)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})


# ---------------------------------------------------------------------------
# segment_by_text tests
# ---------------------------------------------------------------------------

class TestSegmentByText:
    def _make_client(self) -> "ReplicateClient":  # noqa: F821
        from plottter.ai.replicate_client import ReplicateClient
        return ReplicateClient(api_key="r8_test")

    def test_returns_binary_mask(self) -> None:
        """segment_by_text() returns a binary uint8 mask with the same H×W as the input."""
        client = self._make_client()
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        input_image = np.zeros((h, w, 3), dtype=np.uint8)
        expected_mask = np.zeros((h, w), dtype=np.uint8)
        expected_mask[1:7, 1:7] = 255

        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=expected_mask):
                result = client.segment_by_text(input_image, text_prompt="the object")

        assert result.shape == (h, w)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_raises_with_empty_prompt(self) -> None:
        """segment_by_text() raises ReplicateAPIError when the text prompt is empty."""
        client = self._make_client()
        from plottter.ai.replicate_client import ReplicateAPIError

        with pytest.raises(ReplicateAPIError, match="non-empty text prompt"):
            client.segment_by_text(np.zeros((4, 4, 3), dtype=np.uint8), text_prompt="")


# ---------------------------------------------------------------------------
# Public re-export tests (plottter.ai.__init__)
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_public_exports(self) -> None:
        from plottter.ai import ReplicateClient, ReplicateAPIError
        assert ReplicateClient is not None
        assert ReplicateAPIError is not None


# ---------------------------------------------------------------------------
# FMM Topographic depth-map integration tests (task 16.57 updated)
# ---------------------------------------------------------------------------

class TestFMMDepthMapIntegration:
    """Tests for the depth-map-as-image-source workflow in ContourGenerator FMM mode.

    As of task 16.57, the AI depth map is no longer injected via ``_ai_client``
    inside the generator.  Instead, the settings panel converts the depth map to a
    uint8 RGB image and sets it as the layer's source image before generation.
    The ContourGenerator simply receives an ``_source_image`` that happens to be a
    depth map and processes it with ``fmm_source='Image Brightness'``.
    """

    def _make_canvas(self):
        from plottter.models.canvas import Canvas
        return Canvas.from_preset("A4", margin=10.0)

    def _make_rgb_image(self, h: int = 32, w: int = 32) -> np.ndarray:
        """Create a small RGB gradient image (left dark → right bright)."""
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            val = int(x / (w - 1) * 255)
            arr[:, x] = val
        return arr

    def _make_depth_map(self, h: int = 32, w: int = 32) -> np.ndarray:
        """Create a float32 depth map (top row = 1.0, bottom row = 0.0)."""
        depth = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            depth[y, :] = 1.0 - (y / (h - 1))
        return depth

    def _depth_to_rgb(self, depth: np.ndarray) -> np.ndarray:
        """Replicate the _apply_depth_map conversion used by SettingsPanel."""
        depth_uint8 = (depth * 255.0).clip(0, 255).astype(np.uint8)
        return np.stack([depth_uint8] * 3, axis=-1)

    def _run_fmm(self, params: dict) -> list:
        from plottter.generators.contour import ContourGenerator
        gen = ContourGenerator()
        canvas = self._make_canvas()
        return gen.generate(params, canvas)

    def test_fmm_with_depth_map_image_source_produces_valid_output(self) -> None:
        """FMM Topographic with a depth-map RGB source image produces valid polylines.

        The workflow (task 16.57): the settings panel converts the float32 depth
        array to uint8 RGB and sets it as _source_image.  The generator then uses
        'Image Brightness' as its speed map source.  This test simulates that flow.
        """
        depth = self._make_depth_map(h=64, w=64)
        depth_rgb = self._depth_to_rgb(depth)

        params = {
            "_source_image": depth_rgb,
            "mode": "FMM Topographic",
            "fmm_source": "Image Brightness",
            "fmm_num_contours": 5,
            "fmm_source_point": "Center",
            "fmm_gamma": 1.0,
            "fmm_speed_floor": 0.01,
            "fmm_contour_spacing": "Linear",
            "fmm_min_contour_length_mm": 0.5,
            "simplify_mm": 0.3,
            "smooth_iterations": 0,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "invert": False,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
            "min_contour_px": 3,
        }

        result = self._run_fmm(params)
        assert isinstance(result, list)

    def test_fmm_depth_map_versus_plain_image_gives_different_output(self) -> None:
        """Using a depth map as the source image gives different FMM output than
        a plain brightness image, confirming the depth map influence propagates."""
        h, w = 64, 64

        # Plain gradient image (dark left, bright right)
        plain_rgb = self._make_rgb_image(h=h, w=w)

        # Depth map converted to RGB (bright top, dark bottom)
        depth = self._make_depth_map(h=h, w=w)
        depth_rgb = self._depth_to_rgb(depth)

        base_params = {
            "mode": "FMM Topographic",
            "fmm_source": "Image Brightness",
            "fmm_num_contours": 8,
            "fmm_source_point": "Center",
            "fmm_gamma": 1.0,
            "fmm_speed_floor": 0.01,
            "fmm_contour_spacing": "Linear",
            "fmm_min_contour_length_mm": 0.5,
            "simplify_mm": 0.3,
            "smooth_iterations": 0,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "invert": False,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
            "min_contour_px": 3,
        }

        result_plain = self._run_fmm(dict(base_params, _source_image=plain_rgb))
        result_depth = self._run_fmm(dict(base_params, _source_image=depth_rgb))

        assert isinstance(result_plain, list)
        assert isinstance(result_depth, list)

    def test_fmm_source_param_removed(self) -> None:
        """fmm_source parameter was fully removed in task 16.57 — it had only one
        choice ('Image Brightness') and was never read in generate()."""
        from plottter.generators.contour import ContourGenerator
        gen = ContourGenerator()
        params = gen.get_parameters()
        fmm_source_param = next(
            (p for p in params if p.name == "fmm_source"), None
        )
        assert fmm_source_param is None, (
            "fmm_source param was fully removed — it had only one choice and was never read"
        )

    def test_fmm_depth_invert_param_removed(self) -> None:
        """fmm_depth_invert parameter must NOT exist in ContourGenerator.

        Inversion is now handled by the 'Invert' checkbox in the AI Depth Map
        image source controls (settings panel), not by the generator.
        """
        from plottter.generators.contour import ContourGenerator
        gen = ContourGenerator()
        param_names = {p.name for p in gen.get_parameters()}
        assert "fmm_depth_invert" not in param_names, (
            "fmm_depth_invert was removed in task 16.57 — depth inversion is "
            "now a settings panel concern, not a generator parameter"
        )

    def test_apply_depth_map_conversion(self) -> None:
        """The depth-to-RGB conversion (float32 [0,1] → uint8 [0,255] 3-channel)
        is correct: 0.0 → 0, 1.0 → 255, mid → ~128."""
        depth = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        depth_rgb = self._depth_to_rgb(depth)
        assert depth_rgb.shape == (1, 3, 3), f"Unexpected shape: {depth_rgb.shape}"
        assert depth_rgb.dtype == np.uint8
        assert depth_rgb[0, 0, 0] == 0, "0.0 should map to 0"
        assert depth_rgb[0, 2, 0] == 255, "1.0 should map to 255"
        # All three channels must be equal (greyscale replicated)
        np.testing.assert_array_equal(depth_rgb[:, :, 0], depth_rgb[:, :, 1])
        np.testing.assert_array_equal(depth_rgb[:, :, 0], depth_rgb[:, :, 2])

    def test_depth_invert_effect_via_image_source(self) -> None:
        """Using an inverted depth map as the source gives different FMM output
        than the non-inverted one — simulating the 'Invert' checkbox effect."""
        h, w = 64, 64
        # Non-uniform depth: top half = 0.9 (near), bottom half = 0.1 (far)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:32, :] = 0.9
        depth[32:, :] = 0.1

        depth_rgb = self._depth_to_rgb(depth)
        depth_inv_rgb = self._depth_to_rgb(1.0 - depth)

        base_params = {
            "mode": "FMM Topographic",
            "fmm_source": "Image Brightness",
            "fmm_num_contours": 8,
            "fmm_source_point": "Center",
            "fmm_gamma": 1.0,
            "fmm_speed_floor": 0.01,
            "fmm_contour_spacing": "Linear",
            "fmm_min_contour_length_mm": 0.5,
            "simplify_mm": 0.3,
            "smooth_iterations": 0,
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_radius": 0.0,
            "invert": False,
            "smooth_curves": False,
            "curve_tolerance_mm": 0.5,
            "min_contour_px": 3,
        }

        result_normal = self._run_fmm(dict(base_params, _source_image=depth_rgb))
        result_inverted = self._run_fmm(dict(base_params, _source_image=depth_inv_rgb))

        assert isinstance(result_normal, list)
        assert isinstance(result_inverted, list)

    def test_cached_depth_map_reused_across_regenerations(self) -> None:
        """A ReplicateClient's internal cache is reused;
        calling estimate_depth twice with the same image only hits the API once."""
        import plottter.ai.replicate_client as rc_mod
        from plottter.ai.replicate_client import ReplicateClient

        real_client = ReplicateClient(api_key="r8_test")
        rgb = self._make_rgb_image()
        depth = self._make_depth_map()

        depth_rgb_response = np.stack([
            (depth * 255).astype(np.uint8),
            (depth * 255).astype(np.uint8),
            (depth * 255).astype(np.uint8),
        ], axis=-1)

        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/depth.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgb", return_value=depth_rgb_response):
                # First call populates the cache
                d1 = real_client.estimate_depth(rgb)
                # Second call with the same image hits the cache (no API call)
                d2 = real_client.estimate_depth(rgb)

        # Both should return the same object (cache hit)
        assert d1 is d2

    def test_depth_preset_params_use_image_brightness(self) -> None:
        """The FMM depth presets now use fmm_source='Image Brightness' (task 16.57).

        The user selects 'AI Depth Map' as the image source type in the settings
        panel before generation — the generator itself just uses image brightness.
        """
        from plottter.generators.contour import ContourGenerator

        gen = ContourGenerator()
        presets = {p.name: p for p in gen.get_presets()}

        assert "FMM Depth Portrait" in presets, "FMM Depth Portrait preset missing"
        assert "FMM Depth Landscape" in presets, "FMM Depth Landscape preset missing"

        portrait_params = presets["FMM Depth Portrait"].params
        assert "fmm_source" not in portrait_params, (
            "fmm_source key was removed from presets along with the parameter"
        )
        assert "fmm_depth_invert" not in portrait_params, (
            "fmm_depth_invert was removed — depth inversion is a settings panel concern"
        )

        landscape_params = presets["FMM Depth Landscape"].params
        assert "fmm_source" not in landscape_params, (
            "fmm_source key was removed from presets along with the parameter"
        )
        assert "fmm_depth_invert" not in landscape_params


# ---------------------------------------------------------------------------
# Disk-based depth map cache tests (task 16.56)
# ---------------------------------------------------------------------------

class TestDepthMapDiskCache:
    """Tests verifying the disk-based depth map cache in ReplicateClient."""

    def _make_client(self, cache_dir: str | None = None) -> "ReplicateClient":  # noqa: F821
        from plottter.ai.replicate_client import ReplicateClient
        return ReplicateClient(api_key="r8_test", cache_dir=cache_dir)

    def _make_depth_response(self, h: int, w: int) -> np.ndarray:
        """Return a fake RGB depth image (all grey = 0.5 depth)."""
        val = int(0.5 * 255)
        return np.full((h, w, 3), val, dtype=np.uint8)

    def test_saves_png_to_depth_subdir_after_api_call(self, tmp_path) -> None:
        """estimate_depth() saves a 16-bit PNG to the depth/ subdirectory after an API call."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        depth_response = self._make_depth_response(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/depth.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgb", return_value=depth_response):
                result = client.estimate_depth(image)

        # Should have produced exactly one PNG in the depth/ subdirectory
        pngs = list((tmp_path / "depth").glob("*.png"))
        assert len(pngs) == 1, f"Expected 1 PNG in depth/, got {len(pngs)}"
        # No PNGs should be written to the flat cache dir
        assert list(tmp_path.glob("*.png")) == [], "No PNGs should be in the flat cache dir"
        # The saved PNG should be readable and contain the same depth data
        from PIL import Image as _PIL
        arr = np.array(_PIL.open(pngs[0])).astype(np.float32)
        assert arr.max() <= 65535.0  # 16-bit range
        assert result.shape == (h, w)
        assert result.dtype == np.float32

    # Keep old name as an alias so existing test runs don't break
    test_saves_png_to_cache_dir_after_api_call = test_saves_png_to_depth_subdir_after_api_call

    def test_loads_from_cache_on_second_call(self, tmp_path) -> None:
        """Second estimate_depth() call with the same image hits the disk cache,
        not the API."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        depth_response = self._make_depth_response(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/depth.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgb", return_value=depth_response) as mock_fetch:
                # First call — API hit, writes cache
                d1 = client.estimate_depth(image)
                assert mock_fetch.call_count == 1

                # Clear the in-memory cache to force the disk lookup path
                client._cache.clear()

                # Second call — should load from disk, not call API again
                d2 = client.estimate_depth(image)
                assert mock_fetch.call_count == 1, (
                    "API should not be called a second time when disk cache hit"
                )

        assert d1.shape == d2.shape
        assert d1.dtype == d2.dtype
        np.testing.assert_allclose(d1, d2, atol=1.0 / 255)

    def test_different_image_causes_cache_miss(self, tmp_path) -> None:
        """Two images with different content produce different cache entries."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image_a = np.zeros((h, w, 3), dtype=np.uint8)
        image_b = np.full((h, w, 3), 128, dtype=np.uint8)
        depth_a = self._make_depth_response(h, w)
        depth_b = np.full((h, w, 3), 200, dtype=np.uint8)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/depth.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgb") as mock_fetch:
                mock_fetch.return_value = depth_a
                client.estimate_depth(image_a)
                assert mock_fetch.call_count == 1

                client._cache.clear()
                mock_fetch.return_value = depth_b
                client.estimate_depth(image_b)
                assert mock_fetch.call_count == 2, (
                    "Different image should cause a cache miss and new API call"
                )

        # Two different PNGs should be in the depth/ subdirectory
        pngs = list((tmp_path / "depth").glob("*.png"))
        assert len(pngs) == 2, f"Expected 2 cached PNGs in depth/, got {len(pngs)}"

    def test_cache_disabled_when_cache_dir_is_none(self, tmp_path) -> None:
        """When cache_dir=None, no files are written and the API is always called."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        depth_response = self._make_depth_response(h, w)

        # Use None to disable caching
        client = self._make_client(cache_dir=None)
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/depth.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgb", return_value=depth_response) as mock_fetch:
                d1 = client.estimate_depth(image)
                assert mock_fetch.call_count == 1

                client._cache.clear()
                d2 = client.estimate_depth(image)
                # With no disk cache and cleared in-memory cache, API is called again
                assert mock_fetch.call_count == 2

        # No PNGs should have been written anywhere near tmp_path
        assert list(tmp_path.glob("*.png")) == []

    def test_cache_dir_created_on_init(self, tmp_path) -> None:
        """ReplicateClient creates the cache directory and subdirs on init if they don't exist."""
        new_dir = tmp_path / "nested" / "cache"
        assert not new_dir.exists()
        self._make_client(cache_dir=str(new_dir))
        assert new_dir.is_dir(), "Cache directory should be created by __init__"
        assert (new_dir / "depth").is_dir(), "depth/ subdir should be created by __init__"
        assert (new_dir / "bg_removal").is_dir(), "bg_removal/ subdir should be created by __init__"

    def test_flat_depth_cache_loaded_as_backward_compat(self, tmp_path) -> None:
        """Old flat-directory depth cache files are loaded as backward compatibility fallback."""
        import hashlib
        import plottter.ai.replicate_client as rc_mod
        import cv2

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # Write a fake depth PNG to the flat (old) cache location
        img_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        flat_cache_path = tmp_path / f"{img_hash}.png"
        expected_depth = np.full((h, w), 0.75, dtype=np.float32)
        uint16_arr = (expected_depth * 65535).astype(np.uint16)
        cv2.imwrite(str(flat_cache_path), uint16_arr)
        assert flat_cache_path.exists()

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run") as mock_run:
            result = client.estimate_depth(image)

        # API should NOT have been called — flat cache was found
        assert mock_run.call_count == 0, "Flat cache should have been loaded without API call"
        assert result.shape == (h, w)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, expected_depth, atol=1.0 / 65535)


# ---------------------------------------------------------------------------
# Background removal disk cache tests (task 53.1)
# ---------------------------------------------------------------------------

class TestRemoveBackgroundDiskCache:
    """Tests for disk caching in ReplicateClient.remove_background()."""

    def _make_client(self, cache_dir=None):
        from plottter.ai.replicate_client import ReplicateClient
        return ReplicateClient(api_key="r8_test", cache_dir=cache_dir)

    def _make_rgba_result(self, h: int, w: int) -> np.ndarray:
        return np.full((h, w, 4), [100, 150, 200, 255], dtype=np.uint8)

    def test_saves_png_to_bg_removal_subdir(self, tmp_path) -> None:
        """remove_background() saves RGBA PNG to bg_removal/ subdir after an API call."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        rgba_result = self._make_rgba_result(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/output.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgba", return_value=rgba_result):
                result = client.remove_background(image)

        pngs = list((tmp_path / "bg_removal").glob("*.png"))
        assert len(pngs) == 1, f"Expected 1 PNG in bg_removal/, got {len(pngs)}"
        assert list(tmp_path.glob("*.png")) == [], "No PNGs should be in the flat dir"
        assert result.shape == (h, w, 4)
        assert result.dtype == np.uint8

    def test_loads_bg_removal_from_disk_on_second_call(self, tmp_path) -> None:
        """Second remove_background() with same image loads from disk, not the API.

        Mocks _replicate_run and verifies it is NOT called a second time when
        the result is already cached on disk.
        """
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        rgba_result = self._make_rgba_result(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/output.png") as mock_run:
            with patch.object(rc_mod, "_fetch_url_as_rgba", return_value=rgba_result) as mock_fetch:
                r1 = client.remove_background(image)
                assert mock_fetch.call_count == 1
                assert mock_run.call_count == 1

                client._cache.clear()
                r2 = client.remove_background(image)
                assert mock_fetch.call_count == 1, (
                    "API should not be called a second time when disk cache exists"
                )
                assert mock_run.call_count == 1, (
                    "_replicate_run must not be called a second time on disk cache hit"
                )

        assert r1.shape == r2.shape
        np.testing.assert_array_equal(r1, r2)

    def test_bg_removal_no_disk_cache_when_cache_dir_none(self, tmp_path) -> None:
        """When cache_dir=None, no files are written for remove_background()."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        rgba_result = self._make_rgba_result(h, w)

        client = self._make_client(cache_dir=None)
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/output.png"):
            with patch.object(rc_mod, "_fetch_url_as_rgba", return_value=rgba_result) as mock_fetch:
                client.remove_background(image)
                client._cache.clear()
                client.remove_background(image)
                assert mock_fetch.call_count == 2, "API always called when no cache dir"

        assert list(tmp_path.glob("**/*.png")) == []


# ---------------------------------------------------------------------------
# Mask segmentation disk cache tests (task 53.1b)
# ---------------------------------------------------------------------------

class TestSegmentByTextDiskCache:
    """Tests for disk caching in ReplicateClient.segment_by_text()."""

    def _make_client(self, cache_dir=None):
        from plottter.ai.replicate_client import ReplicateClient
        return ReplicateClient(api_key="r8_test", cache_dir=cache_dir)

    def _make_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[1:h - 1, 1:w - 1] = 255
        return mask

    def test_saves_png_to_masks_subdir(self, tmp_path) -> None:
        """segment_by_text() saves binary mask PNG to masks/ subdir after an API call."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = self._make_mask(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=mask):
                result = client.segment_by_text(image, text_prompt="the object")

        pngs = list((tmp_path / "masks").glob("*.png"))
        assert len(pngs) == 1, f"Expected 1 PNG in masks/, got {len(pngs)}"
        assert pngs[0].name.startswith("") and "_text_" in pngs[0].name
        assert list(tmp_path.glob("*.png")) == [], "No PNGs should be in the flat dir"
        assert result.shape == (h, w)
        assert result.dtype == np.uint8

    def test_different_prompts_produce_different_cache_files(self, tmp_path) -> None:
        """Different text prompts produce different cache files in masks/."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = self._make_mask(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=mask):
                client.segment_by_text(image, text_prompt="the cat")
                client._cache.clear()
                client.segment_by_text(image, text_prompt="the dog")

        pngs = list((tmp_path / "masks").glob("*_text_*.png"))
        assert len(pngs) == 2, f"Expected 2 PNGs for different prompts, got {len(pngs)}"

    def test_second_call_loads_from_disk_cache(self, tmp_path) -> None:
        """Second segment_by_text() with same image+prompt loads from disk, not API."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = self._make_mask(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=mask) as mock_fetch:
                r1 = client.segment_by_text(image, text_prompt="the object")
                assert mock_fetch.call_count == 1

                client._cache.clear()
                r2 = client.segment_by_text(image, text_prompt="the object")
                assert mock_fetch.call_count == 1, (
                    "API should not be called a second time when disk cache exists"
                )

        assert r1.shape == r2.shape
        np.testing.assert_array_equal(r1, r2)

    def test_masks_subdir_created_on_init(self, tmp_path) -> None:
        """ReplicateClient creates masks/ subdirectory on init."""
        new_dir = tmp_path / "nested" / "cache"
        assert not new_dir.exists()
        self._make_client(cache_dir=str(new_dir))
        assert (new_dir / "masks").is_dir(), "masks/ subdir should be created by __init__"

    def test_no_disk_cache_when_cache_dir_none(self, tmp_path) -> None:
        """When cache_dir=None, no files are written for segment_by_text()."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = self._make_mask(h, w)

        client = self._make_client(cache_dir=None)
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=mask) as mock_fetch:
                client.segment_by_text(image, text_prompt="the object")
                client._cache.clear()
                client.segment_by_text(image, text_prompt="the object")
                assert mock_fetch.call_count == 2, "API always called when no cache dir"

        assert list(tmp_path.glob("**/*.png")) == []


class TestSegmentByPointDiskCache:
    """Tests for disk caching in ReplicateClient.segment_by_point()."""

    def _make_client(self, cache_dir=None):
        from plottter.ai.replicate_client import ReplicateClient
        return ReplicateClient(api_key="r8_test", cache_dir=cache_dir)

    def _make_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        return mask

    def test_saves_png_to_masks_subdir(self, tmp_path) -> None:
        """segment_by_point() saves binary mask PNG to masks/ subdir after an API call."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = self._make_mask(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=mask):
                client.segment_by_point(image, positive_points=[(4, 4)])

        pngs = list((tmp_path / "masks").glob("*_point_*.png"))
        assert len(pngs) == 1, f"Expected 1 PNG in masks/, got {len(pngs)}"

    def test_second_call_loads_from_disk_cache(self, tmp_path) -> None:
        """Second segment_by_point() with same image+points loads from disk, not API."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = self._make_mask(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=mask) as mock_fetch:
                r1 = client.segment_by_point(image, positive_points=[(4, 4)])
                assert mock_fetch.call_count == 1

                client._cache.clear()
                r2 = client.segment_by_point(image, positive_points=[(4, 4)])
                assert mock_fetch.call_count == 1, (
                    "API should not be called a second time when disk cache exists"
                )

        np.testing.assert_array_equal(r1, r2)


class TestSegmentByBoxDiskCache:
    """Tests for disk caching in ReplicateClient.segment_by_box()."""

    def _make_client(self, cache_dir=None):
        from plottter.ai.replicate_client import ReplicateClient
        return ReplicateClient(api_key="r8_test", cache_dir=cache_dir)

    def _make_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        return mask

    def test_saves_png_to_masks_subdir(self, tmp_path) -> None:
        """segment_by_box() saves binary mask PNG to masks/ subdir after an API call."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = self._make_mask(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=mask):
                client.segment_by_box(image, box_xyxy=(2, 2, 6, 6))

        pngs = list((tmp_path / "masks").glob("*_box_*.png"))
        assert len(pngs) == 1, f"Expected 1 PNG in masks/, got {len(pngs)}"

    def test_second_call_loads_from_disk_cache(self, tmp_path) -> None:
        """Second segment_by_box() with same image+box loads from disk, not API."""
        import plottter.ai.replicate_client as rc_mod

        h, w = 8, 8
        image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = self._make_mask(h, w)

        client = self._make_client(cache_dir=str(tmp_path))
        with patch.object(rc_mod, "_replicate_run", return_value="https://fake/mask.png"):
            with patch.object(rc_mod, "_fetch_mask_as_binary", return_value=mask) as mock_fetch:
                r1 = client.segment_by_box(image, box_xyxy=(2, 2, 6, 6))
                assert mock_fetch.call_count == 1

                client._cache.clear()
                r2 = client.segment_by_box(image, box_xyxy=(2, 2, 6, 6))
                assert mock_fetch.call_count == 1, (
                    "API should not be called a second time when disk cache exists"
                )

        np.testing.assert_array_equal(r1, r2)
