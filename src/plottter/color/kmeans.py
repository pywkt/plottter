"""K-means color clustering for color separation."""

from __future__ import annotations

import numpy as np


def _rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image (H×W×3, uint8) to LAB color space (H×W×3, float32).

    Uses OpenCV for the conversion which gives perceptually uniform LAB values.
    Falls back to a simple sRGB→XYZ→LAB pipeline if OpenCV is unavailable.
    """
    try:
        import cv2

        img_float = image_rgb.astype(np.float32) / 255.0
        bgr = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab
    except ImportError:
        # Minimal sRGB → XYZ → LAB fallback (no cv2)
        img = image_rgb.astype(np.float32) / 255.0
        # sRGB linearisation
        mask = img > 0.04045
        lin = np.where(mask, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
        # sRGB → XYZ D65
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=np.float32)
        xyz = lin.reshape(-1, 3) @ M.T
        # Normalise by D65 white point
        xyz[:, 0] /= 0.95047
        xyz[:, 2] /= 1.08883
        epsilon = 0.008856
        kappa = 903.3
        fx = np.where(xyz > epsilon, xyz ** (1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)
        L = 116.0 * fx[:, 1] - 16.0
        a = 500.0 * (fx[:, 0] - fx[:, 1])
        b = 200.0 * (fx[:, 1] - fx[:, 2])
        lab = np.stack([L, a, b], axis=1).reshape(image_rgb.shape).astype(np.float32)
        return lab


def _lab_to_hex(lab_center: np.ndarray) -> str:
    """Convert a LAB cluster centre to the nearest hex RGB color string."""
    try:
        import cv2

        lab_img = np.array([[[lab_center[0], lab_center[1], lab_center[2]]]], dtype=np.float32)
        bgr = cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
        r, g, b = int(bgr[0, 0, 2]), int(bgr[0, 0, 1]), int(bgr[0, 0, 0])
    except ImportError:
        # Approximate inverse: just return gray based on L
        v = int(np.clip(lab_center[0] * 2.55, 0, 255))
        r, g, b = v, v, v
    return f"#{r:02X}{g:02X}{b:02X}"


def kmeans_separate(
    image: np.ndarray,
    num_colors: int,
    sample_size: int = 10000,
    iterations: int = 20,
) -> list[tuple[np.ndarray, str]]:
    """Cluster an RGB image into *num_colors* groups by LAB-space color similarity.

    Parameters
    ----------
    image:
        RGB image array of shape (H, W, 3), dtype uint8.
    num_colors:
        Number of clusters (pens), in the range 2–8.
    sample_size:
        Maximum number of pixels to use for clustering (subsampling large images).
    iterations:
        K-means iterations.

    Returns
    -------
    list of (binary_mask, hex_color) tuples — one per cluster.
    ``binary_mask`` is a boolean array of shape (H, W) where True indicates
    pixels belonging to that cluster.  ``hex_color`` is the cluster-centre
    colour rendered as a hex string like ``"#FF0000"``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H×W×3), got shape {image.shape}")
    num_colors = max(2, min(8, num_colors))

    h, w = image.shape[:2]
    lab = _rgb_to_lab(image)
    pixels = lab.reshape(-1, 3)  # (N, 3)

    # Subsample for clustering
    n_pixels = pixels.shape[0]
    if n_pixels > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_pixels, size=sample_size, replace=False)
        sample = pixels[indices]
    else:
        sample = pixels

    # K-means initialisation: spread initial centres with k-means++ heuristic
    rng = np.random.default_rng(42)
    centre_indices = [int(rng.integers(0, len(sample)))]
    for _ in range(1, num_colors):
        dists = np.min(
            np.array([np.sum((sample - sample[ci]) ** 2, axis=1) for ci in centre_indices]),
            axis=0,
        )
        total = dists.sum()
        if total == 0:
            probs = np.ones(len(sample)) / len(sample)
        else:
            probs = dists / total
        centre_indices.append(int(rng.choice(len(sample), p=probs)))
    centres = sample[centre_indices].copy()  # (K, 3)

    # Iterate
    for _ in range(iterations):
        # Assign each sample pixel to the nearest centre
        dists = np.array([np.sum((sample - c) ** 2, axis=1) for c in centres])  # (K, N)
        labels = np.argmin(dists, axis=0)  # (N,)
        # Update centres
        new_centres = np.array([
            sample[labels == k].mean(axis=0) if np.any(labels == k) else centres[k]
            for k in range(num_colors)
        ])
        if np.allclose(centres, new_centres, atol=1e-6):
            break
        centres = new_centres

    # Assign every pixel in the full image to a cluster
    dists_all = np.array([np.sum((pixels - c) ** 2, axis=1) for c in centres])  # (K, H*W)
    all_labels = np.argmin(dists_all, axis=0).reshape(h, w)  # (H, W)

    results: list[tuple[np.ndarray, str]] = []
    for k in range(num_colors):
        mask = all_labels == k  # (H, W) boolean
        hex_color = _lab_to_hex(centres[k])
        results.append((mask, hex_color))

    return results
