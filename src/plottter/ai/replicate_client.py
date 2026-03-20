"""Replicate.com API wrapper for AI-powered image processing.

This module uses the Replicate REST API directly with no third-party SDK.
All network calls should be run in a QThread so the GUI stays responsive.
Check ``ReplicateClient.is_available()`` before calling any method; it returns
``False`` when the API key is not set. The API key is configured in Preferences.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Model identifiers (update these when better models become available)
# ---------------------------------------------------------------------------
MODEL_REMOVE_BG = "lucataco/remove-bg:95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1"
MODEL_DEPTH = "zedge/zoedepth:fd85428545f04150f59856dab2a51a7be2ca5003a331920b0e4303b17b411332"
# SAM-2 automatic mask generation (returns combined_mask + individual_masks)
MODEL_SEGMENT = "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"
# SAM-2 video model — also accepts single images with click_coordinates/click_labels
MODEL_SAM2 = "meta/sam-2-video:33432afdfc06a10da6b4018932893d39b0159f838b6d11dd1236dff85cc5ec1d"
# Grounded SAM for text-prompted segmentation (input: mask_prompt)
MODEL_GROUNDED_SAM = "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c"


class ReplicateAPIError(Exception):
    """Raised on network failures, invalid key, rate limits, or model errors."""


class ReplicateClient:
    """Thin wrapper around the Replicate REST API for Plottter AI features.

    Args:
        api_key: Replicate API key (from QSettings ``"replicate/api_key"``).
            Pass an empty string to construct a disabled client.
    """

    def __init__(self, api_key: str, cache_dir: str | None = None) -> None:
        self._api_key = api_key
        # In-memory response cache keyed by (operation, image object id)
        self._cache: dict[tuple, object] = {}
        # Disk-based depth map cache directory (None = disabled)
        self._cache_dir: str | None = cache_dir
        if cache_dir is not None:
            pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the API key is set."""
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # Background removal
    # ------------------------------------------------------------------

    def remove_background(
        self,
        image: np.ndarray,
        progress_callback: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Remove background from an RGB image using a SAM/segmentation model.

        Args:
            image: RGB input image (H x W x 3, uint8).
            progress_callback: Optional callable receiving progress 0–100.

        Returns:
            RGBA image (H x W x 4, uint8) where background pixels have alpha=0.

        Raises:
            ReplicateAPIError: On network failure, invalid key, or model error.
        """
        cache_key = ("remove_bg", id(image), image.shape)
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        try:
            if progress_callback:
                progress_callback(10)

            img_b64 = _image_to_data_uri(image)
            output = _replicate_run(self._api_key, MODEL_REMOVE_BG, {"image": img_b64})

            if progress_callback:
                progress_callback(70)

            result = _fetch_url_as_rgba(output)
            # Resize to match input dimensions if necessary
            if result.shape[:2] != image.shape[:2]:
                from PIL import Image as _PIL_Image
                pil = _PIL_Image.fromarray(result, mode="RGBA")
                pil = pil.resize(
                    (image.shape[1], image.shape[0]),
                    _PIL_Image.LANCZOS,
                )
                result = np.array(pil)

            self._cache[cache_key] = result
            if progress_callback:
                progress_callback(100)
            return result

        except ReplicateAPIError:
            raise
        except Exception as exc:
            raise ReplicateAPIError(f"Background removal failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Depth estimation
    # ------------------------------------------------------------------

    def estimate_depth(
        self,
        image: np.ndarray,
        progress_callback: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Estimate a monocular depth map from an RGB image.

        Args:
            image: RGB input image (H x W x 3, uint8).
            progress_callback: Optional callable receiving progress 0–100.

        Returns:
            Single-channel float32 array (H x W), normalized 0.0–1.0 where
            1.0 is closest to the camera.

        Raises:
            ReplicateAPIError: On network failure, invalid key, or model error.
        """
        cache_key = ("depth", id(image), image.shape)
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        # --- Disk cache lookup ---
        disk_cache_path: str | None = None
        if self._cache_dir is not None:
            img_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
            disk_cache_path = os.path.join(self._cache_dir, f"{img_hash}.png")
            if os.path.exists(disk_cache_path):
                try:
                    from PIL import Image as _PIL_Image
                    pil = _PIL_Image.open(disk_cache_path)
                    arr = np.array(pil).astype(np.float32)
                    # Cache always stores 16-bit PNG (depth * 65535); normalise back.
                    if arr.max() > 1.0:
                        arr = arr / 65535.0
                    if arr.shape != tuple(image.shape[:2]):
                        pil = pil.resize(
                            (image.shape[1], image.shape[0]),
                            _PIL_Image.LANCZOS,
                        )
                        arr = np.array(pil).astype(np.float32)
                        if arr.max() > 1.0:
                            arr = arr / 65535.0
                    self._cache[cache_key] = arr
                    if progress_callback:
                        progress_callback(100)
                    return arr
                except Exception:
                    pass  # Corrupt cache file — fall through to API call

        try:
            if progress_callback:
                progress_callback(10)

            img_b64 = _image_to_data_uri(image)
            output = _replicate_run(self._api_key, MODEL_DEPTH, {"image": img_b64})

            if progress_callback:
                progress_callback(70)

            # Depth model returns a grayscale image URL; convert to float32
            depth_rgb = _fetch_url_as_rgb(output)
            # Use luminance channel as depth proxy
            depth_gray = (
                0.299 * depth_rgb[:, :, 0].astype(np.float32)
                + 0.587 * depth_rgb[:, :, 1].astype(np.float32)
                + 0.114 * depth_rgb[:, :, 2].astype(np.float32)
            ) / 255.0

            # Resize to match input
            if depth_gray.shape != image.shape[:2]:
                from PIL import Image as _PIL_Image
                pil = _PIL_Image.fromarray((depth_gray * 255).astype(np.uint8))
                pil = pil.resize(
                    (image.shape[1], image.shape[0]),
                    _PIL_Image.LANCZOS,
                )
                depth_gray = np.array(pil).astype(np.float32) / 255.0

            # --- Disk cache write (16-bit grayscale PNG for precision) ---
            if disk_cache_path is not None:
                try:
                    from PIL import Image as _PIL_Image
                    import cv2 as _cv2
                    uint16_arr = (depth_gray * 65535).astype(np.uint16)
                    _cv2.imwrite(str(disk_cache_path), uint16_arr)
                except Exception:
                    pass  # Cache write failure is non-fatal

            self._cache[cache_key] = depth_gray
            if progress_callback:
                progress_callback(100)
            return depth_gray

        except ReplicateAPIError:
            raise
        except Exception as exc:
            raise ReplicateAPIError(f"Depth estimation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Semantic segmentation
    # ------------------------------------------------------------------

    def segment_image(
        self,
        image: np.ndarray,
        num_segments: int = 4,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[tuple[np.ndarray, str]]:
        """Segment an image into semantic regions using a segmentation model.

        Args:
            image: RGB input image (H x W x 3, uint8).
            num_segments: Maximum number of segments to return (largest first).
            progress_callback: Optional callable receiving progress 0–100.

        Returns:
            List of ``(binary_mask, hex_color)`` tuples, one per segment.
            Each *binary_mask* is a uint8 array (H x W) with 255 for foreground
            and 0 for background.  *hex_color* is the representative colour of
            that region as a hex string (e.g. ``"#FF0000"``).

        Raises:
            ReplicateAPIError: On network failure, invalid key, or model error.
        """
        cache_key = ("segment", id(image), image.shape, num_segments)
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        try:
            if progress_callback:
                progress_callback(10)

            img_b64 = _image_to_data_uri(image)
            output = _replicate_run(self._api_key, MODEL_SEGMENT, {"image": img_b64})

            if progress_callback:
                progress_callback(70)

            # meta/sam-2 returns {"combined_mask": url, "individual_masks": [url, ...]}
            individual_urls = output.get("individual_masks", []) if isinstance(output, dict) else []
            if not individual_urls and isinstance(output, dict):
                # Fallback: use combined mask
                combined = output.get("combined_mask")
                if combined:
                    individual_urls = [combined]

            if not individual_urls:
                # Legacy / other model: single URL to palette segmentation map
                seg_rgb = _fetch_url_as_rgb(output)
                if seg_rgb.shape[:2] != image.shape[:2]:
                    from PIL import Image as _PIL_Image
                    pil = _PIL_Image.fromarray(seg_rgb)
                    pil = pil.resize(
                        (image.shape[1], image.shape[0]),
                        _PIL_Image.NEAREST,
                    )
                    seg_rgb = np.array(pil)
                results = _extract_segments(image, seg_rgb, num_segments)
            else:
                results = _extract_segments_from_masks(
                    image, individual_urls[:num_segments],
                )

            self._cache[cache_key] = results
            if progress_callback:
                progress_callback(100)
            return results

        except ReplicateAPIError:
            raise
        except Exception as exc:
            raise ReplicateAPIError(f"Image segmentation failed: {exc}") from exc


    # ------------------------------------------------------------------
    # Point-prompted segmentation (SAM-2)
    # ------------------------------------------------------------------

    def segment_by_point(
        self,
        image: np.ndarray,
        positive_points: list[tuple[int, int]],
        negative_points: list[tuple[int, int]] | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Segment the object under the given point prompts using SAM-2.

        Args:
            image: RGB input image (H x W x 3, uint8).
            positive_points: List of (x, y) pixel coords marking the object.
            negative_points: Optional list of (x, y) pixel coords to exclude.
            progress_callback: Optional callable receiving progress 0–100.

        Returns:
            Binary mask (H x W, uint8) where foreground pixels are 255.

        Raises:
            ReplicateAPIError: On network failure, invalid key, or model error.
        """
        if not positive_points:
            raise ReplicateAPIError("At least one positive point is required.")

        neg = negative_points or []
        cache_key = (
            "sam2_point",
            id(image),
            image.shape,
            tuple(positive_points),
            tuple(neg),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        try:
            if progress_callback:
                progress_callback(10)

            # meta/sam-2-video requires a downloadable file URL (not a data URI)
            img_url = _upload_file(self._api_key, image)
            all_points = list(positive_points) + list(neg)
            labels = [1] * len(positive_points) + [0] * len(neg)

            # meta/sam-2-video expects click_coordinates as "[x,y],[x,y],..."
            # and click_labels as "1,0,..." strings.
            coords_str = ",".join(f"[{x},{y}]" for x, y in all_points)
            labels_str = ",".join(str(l) for l in labels)

            output = _replicate_run(
                self._api_key,
                MODEL_SAM2,
                {
                    "input_video": img_url,
                    "click_coordinates": coords_str,
                    "click_labels": labels_str,
                },
            )

            if progress_callback:
                progress_callback(70)

            mask = _extract_first_mask(output, image.shape[:2])

            self._cache[cache_key] = mask
            if progress_callback:
                progress_callback(100)
            return mask

        except ReplicateAPIError:
            raise
        except Exception as exc:
            raise ReplicateAPIError(f"Point segmentation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Box-prompted segmentation (SAM-2)
    # ------------------------------------------------------------------

    def segment_by_box(
        self,
        image: np.ndarray,
        box_xyxy: tuple[int, int, int, int],
        progress_callback: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Segment the object inside the given bounding box using SAM-2.

        Args:
            image: RGB input image (H x W x 3, uint8).
            box_xyxy: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
            progress_callback: Optional callable receiving progress 0–100.

        Returns:
            Binary mask (H x W, uint8) where foreground pixels are 255.

        Raises:
            ReplicateAPIError: On network failure, invalid key, or model error.
        """
        cache_key = ("sam2_box", id(image), image.shape, box_xyxy)
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        try:
            if progress_callback:
                progress_callback(10)

            # meta/sam-2-video requires a downloadable file URL (not a data URI)
            img_url = _upload_file(self._api_key, image)
            # meta/sam-2-video doesn't have a "box" param — approximate
            # by clicking the center as a foreground point.
            x1, y1, x2, y2 = box_xyxy
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            coords_str = f"[{cx},{cy}]"
            labels_str = "1"

            output = _replicate_run(
                self._api_key,
                MODEL_SAM2,
                {
                    "input_video": img_url,
                    "click_coordinates": coords_str,
                    "click_labels": labels_str,
                },
            )

            if progress_callback:
                progress_callback(70)

            mask = _extract_first_mask(output, image.shape[:2])

            self._cache[cache_key] = mask
            if progress_callback:
                progress_callback(100)
            return mask

        except ReplicateAPIError:
            raise
        except Exception as exc:
            raise ReplicateAPIError(f"Box segmentation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Text-prompted segmentation (Grounded SAM)
    # ------------------------------------------------------------------

    def segment_by_text(
        self,
        image: np.ndarray,
        text_prompt: str,
        progress_callback: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Segment the region matching the text description using Grounded SAM.

        Args:
            image: RGB input image (H x W x 3, uint8).
            text_prompt: Natural-language description of the object to segment
                (e.g. ``"the camera"`` or ``"person on the left"``).
            progress_callback: Optional callable receiving progress 0–100.

        Returns:
            Binary mask (H x W, uint8) where foreground pixels are 255.

        Raises:
            ReplicateAPIError: On network failure, invalid key, or model error.
        """
        if not text_prompt.strip():
            raise ReplicateAPIError("A non-empty text prompt is required.")

        cache_key = ("grounded_sam", id(image), image.shape, text_prompt.strip())
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        try:
            if progress_callback:
                progress_callback(10)

            img_b64 = _image_to_data_uri(image)
            output = _replicate_run(
                self._api_key,
                MODEL_GROUNDED_SAM,
                {
                    "image": img_b64,
                    "mask_prompt": text_prompt.strip(),
                },
            )

            if progress_callback:
                progress_callback(70)

            # schananas/grounded_sam returns a list of mask image URLs
            mask = _extract_first_mask(output, image.shape[:2])

            self._cache[cache_key] = mask
            if progress_callback:
                progress_callback(100)
            return mask

        except ReplicateAPIError:
            raise
        except Exception as exc:
            raise ReplicateAPIError(f"Text segmentation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# REST API helpers
# ---------------------------------------------------------------------------

def _upload_file(api_key: str, image: np.ndarray) -> str:
    """Upload an image to the Replicate file API and return the file URL.

    Some models (e.g. ``meta/sam-2-video``) require a downloadable URL
    rather than an inline data URI.  This uploads the image as a PNG and
    returns the ``urls.get`` URL which can be used as a prediction input.

    The uploaded file expires after 24 hours on Replicate's servers.
    """
    import base64
    from PIL import Image as _PIL_Image

    arr = image.astype(np.uint8)
    if arr.ndim == 2:
        pil = _PIL_Image.fromarray(arr, mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil = _PIL_Image.fromarray(arr, mode="RGB")
    elif arr.ndim == 3 and arr.shape[2] == 4:
        pil = _PIL_Image.fromarray(arr, mode="RGBA").convert("RGB")
    else:
        pil = _PIL_Image.fromarray(arr).convert("RGB")

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    file_bytes = buf.getvalue()

    # Build multipart/form-data body
    boundary = "----ReplicateUpload"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="content"; filename="image.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        "https://api.replicate.com/v1/files",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        raise ReplicateAPIError(f"File upload failed (HTTP {exc.code}): {body_text}") from exc

    return result["urls"]["get"]


def _replicate_run(api_key: str, model: str, input_data: dict) -> object:
    """Call the Replicate REST API, poll for completion, and return the output.

    Args:
        api_key: Replicate API key.
        model: Model identifier in the form ``"owner/name:version_hash"``.
        input_data: Dict of model input parameters.

    Returns:
        The ``output`` field from the completed prediction (concrete Python
        objects decoded from JSON — strings, lists, dicts, etc.).

    Raises:
        ReplicateAPIError: On HTTP errors, prediction failure, or cancellation.
    """
    # Extract version hash — format: "owner/name:version_hash"
    version_hash = model.split(":")[-1] if ":" in model else model

    # POST to create a new prediction
    body = json.dumps({"version": version_hash, "input": input_data}).encode()
    req = urllib.request.Request(
        "https://api.replicate.com/v1/predictions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            response: dict = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        if exc.code == 401:
            raise ReplicateAPIError(
                f"Invalid API key (HTTP 401): {body_text}"
            ) from exc
        elif exc.code == 422:
            raise ReplicateAPIError(
                f"Invalid input (HTTP 422): {body_text}"
            ) from exc
        else:
            raise ReplicateAPIError(f"HTTP {exc.code}: {body_text}") from exc

    # Poll until terminal status
    poll_url: str = response["urls"]["get"]
    while True:
        poll_req = urllib.request.Request(
            poll_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(poll_req) as resp:
                response = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise ReplicateAPIError(
                f"HTTP {exc.code} while polling: {body_text}"
            ) from exc

        status = response.get("status")
        if status == "succeeded":
            return response["output"]
        elif status == "failed":
            raise ReplicateAPIError(response.get("error") or "Prediction failed")
        elif status == "canceled":
            raise ReplicateAPIError("Prediction was canceled")

        time.sleep(1)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _image_to_data_uri(image: np.ndarray) -> str:
    """Encode a numpy array (grayscale, RGB, or RGBA) as a base64 PNG data URI."""
    import base64
    from PIL import Image as _PIL_Image

    arr = image.astype(np.uint8)
    if arr.ndim == 2:
        # Grayscale → convert to RGB
        pil = _PIL_Image.fromarray(arr, mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA → convert to RGB
        pil = _PIL_Image.fromarray(arr, mode="RGBA").convert("RGB")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil = _PIL_Image.fromarray(arr, mode="RGB")
    else:
        pil = _PIL_Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _fetch_url_as_rgba(url: str | object) -> np.ndarray:
    """Download *url* (string or file-like) and return as RGBA uint8 array."""
    from PIL import Image as _PIL_Image

    if hasattr(url, "read"):
        data = url.read()  # type: ignore[union-attr]
        pil = _PIL_Image.open(io.BytesIO(data)).convert("RGBA")
    else:
        with urllib.request.urlopen(str(url)) as resp:
            data = resp.read()
        pil = _PIL_Image.open(io.BytesIO(data)).convert("RGBA")
    return np.array(pil)


def _fetch_url_as_rgb(url: str | object) -> np.ndarray:
    """Download *url* (string or file-like) and return as RGB uint8 array."""
    from PIL import Image as _PIL_Image

    if hasattr(url, "read"):
        data = url.read()  # type: ignore[union-attr]
        pil = _PIL_Image.open(io.BytesIO(data)).convert("RGB")
    else:
        with urllib.request.urlopen(str(url)) as resp:
            data = resp.read()
        pil = _PIL_Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(pil)


def _extract_segments(
    original: np.ndarray,
    seg_map: np.ndarray,
    num_segments: int,
) -> list[tuple[np.ndarray, str]]:
    """Convert an RGB segmentation palette image to binary masks + hex colours.

    Each unique colour in *seg_map* represents one semantic class.  We keep the
    largest *num_segments* classes and for each one return a binary mask (uint8,
    255 = foreground) and the median colour of the *original* image pixels
    within that mask.
    """
    h, w = seg_map.shape[:2]
    # Encode each pixel as a single int32 key: R | G<<8 | B<<16
    keys = (
        seg_map[:, :, 0].astype(np.int32)
        | (seg_map[:, :, 1].astype(np.int32) << 8)
        | (seg_map[:, :, 2].astype(np.int32) << 16)
    )
    unique_keys, counts = np.unique(keys, return_counts=True)

    # Sort by area (largest first), then limit to num_segments
    order = np.argsort(-counts)
    unique_keys = unique_keys[order[:num_segments]]

    results: list[tuple[np.ndarray, str]] = []
    for key in unique_keys:
        mask = (keys == key).astype(np.uint8) * 255
        # Representative colour: median of original image pixels in this mask
        pixels = original[mask == 255]
        if pixels.size == 0:
            hex_color = "#808080"
        else:
            r, g, b = (
                int(np.median(pixels[:, 0])),
                int(np.median(pixels[:, 1])),
                int(np.median(pixels[:, 2])),
            )
            hex_color = f"#{r:02X}{g:02X}{b:02X}"
        results.append((mask, hex_color))

    return results


def _fetch_mask_as_binary(
    url: str | object,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Download a mask image and return as a binary uint8 array (H x W, 0/255).

    The mask is thresholded at mid-grey (128) so that any non-background pixel
    in the model's output becomes foreground (255).  The result is resized to
    *target_shape* (height, width) if necessary.
    """
    from PIL import Image as _PIL_Image

    if hasattr(url, "read"):
        data = url.read()  # type: ignore[union-attr]
        pil = _PIL_Image.open(io.BytesIO(data)).convert("L")
    else:
        with urllib.request.urlopen(str(url)) as resp:
            data = resp.read()
        pil = _PIL_Image.open(io.BytesIO(data)).convert("L")

    h, w = target_shape
    if pil.size != (w, h):
        pil = pil.resize((w, h), _PIL_Image.NEAREST)

    arr = np.array(pil)
    binary = (arr >= 128).astype(np.uint8) * 255
    return binary


def _extract_first_mask(
    output: object,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Extract the first usable mask from a model output.

    *output* may be a single URL string, a list of URLs (e.g. sam-2-video
    returns a sequence of mask frames), or a dict with ``combined_mask``.
    Returns a binary uint8 array (H x W, 0/255).
    """
    url: object = output
    if isinstance(output, dict):
        url = output.get("combined_mask") or output.get("individual_masks", [None])[0]
    elif isinstance(output, (list, tuple)):
        url = output[0] if output else None
    if url is None:
        raise ReplicateAPIError("Model returned no mask output.")
    return _fetch_mask_as_binary(url, target_shape)


def _extract_segments_from_masks(
    original: np.ndarray,
    mask_urls: list[str],
) -> list[tuple[np.ndarray, str]]:
    """Build segment tuples from a list of individual mask image URLs.

    Each mask is downloaded, binarised, and paired with the median colour of
    the original image pixels inside that mask region.
    """
    results: list[tuple[np.ndarray, str]] = []
    target_shape = original.shape[:2]
    for url in mask_urls:
        mask = _fetch_mask_as_binary(url, target_shape)
        pixels = original[mask == 255]
        if pixels.size == 0:
            hex_color = "#808080"
        else:
            r = int(np.median(pixels[:, 0]))
            g = int(np.median(pixels[:, 1]))
            b = int(np.median(pixels[:, 2]))
            hex_color = f"#{r:02X}{g:02X}{b:02X}"
        results.append((mask, hex_color))
    return results
