"""Edge-aware reassignment for partitioning colour separators.

Nearest-colour assignment (K-Means, Custom Palette) sends every pixel to the
closest pen colour in Lab space.  That is wrong for the *blended* pixels
along every edge: anti-aliasing, JPEG ringing and resampling all produce
pixels that are a mix of two neighbouring colours — black ink on a white
page gives grays, black on teal gives dark teals.  A mid-gray is nearer to a
dark teal pen than to black or white, so with a dark pen in the palette the
teal layer receives a ghost of every black line in the drawing.

The fix treats each pixel as a possible mix of *two* pens: the candidates are
the pen colours themselves plus the straight Lab segment between every pair.
A pixel is assigned to its nearest candidate; when that is a segment, it goes
to the endpoint it is closer to.  Pure colours are unaffected (they sit on a
pen point), while blended pixels land on one of the two colours they were
blended from instead of on an unrelated pen that happens to be nearby.
"""

from __future__ import annotations

import numpy as np

from plottter.color.kmeans import _rgb_to_lab

#: Pixels processed per block — bounds the float32 temporaries at ~1M × 3.
_BLOCK = 1_000_000


def _hex_to_lab(hex_color: str) -> np.ndarray:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return _rgb_to_lab(np.array([[[r, g, b]]], dtype=np.uint8))[0, 0].astype(np.float32)


def edge_aware_labels(pixels_lab: np.ndarray, centres_lab: np.ndarray) -> np.ndarray:
    """Assign each Lab pixel to a centre, treating pairs of centres as mixes.

    Parameters
    ----------
    pixels_lab:
        (N, 3) float32 Lab pixels.
    centres_lab:
        (K, 3) float32 Lab pen colours.

    Returns
    -------
    (N,) int32 label per pixel in ``range(K)``.
    """
    centres = np.asarray(centres_lab, dtype=np.float32)
    k = len(centres)
    n = len(pixels_lab)
    out = np.zeros(n, dtype=np.int32)
    if k == 0:
        return out
    for start in range(0, n, _BLOCK):
        px = np.asarray(pixels_lab[start : start + _BLOCK], dtype=np.float32)
        best_d = np.full(len(px), np.inf, dtype=np.float32)
        best_l = np.zeros(len(px), dtype=np.int32)
        # Points first so ties with a segment endpoint resolve to the point.
        for i in range(k):
            d = np.sum((px - centres[i]) ** 2, axis=1)
            upd = d < best_d
            best_d[upd] = d[upd]
            best_l[upd] = i
        for i in range(k):
            for j in range(i + 1, k):
                a = centres[i]
                ab = centres[j] - a
                len2 = float(ab @ ab)
                if len2 == 0.0:
                    continue
                t = np.clip(((px - a) @ ab) / len2, 0.0, 1.0)
                d = np.sum((px - (a + t[:, None] * ab)) ** 2, axis=1)
                upd = d < best_d
                best_d[upd] = d[upd]
                best_l[upd] = np.where(t[upd] < 0.5, i, j)
        out[start : start + len(px)] = best_l
    return out


def edge_aware_reassign(
    image_rgb: np.ndarray,
    results: list[tuple[np.ndarray, str]],
) -> list[tuple[np.ndarray, str]]:
    """Re-partition *image_rgb* over the layer colours in *results*.

    *results* is the ``[(mask, hex), ...]`` list a partitioning separator
    produced.  The hex colours are taken as the pen set; each mask is rebuilt
    with :func:`edge_aware_labels` and returned in the same order, with the
    same hex and the same mask dtype (bool, or uint8 0/255 as Custom Palette
    emits).  With fewer than two layers there is nothing to disambiguate and
    the input is returned unchanged.
    """
    if len(results) < 2:
        return results
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H×W×3), got shape {image_rgb.shape}")
    h, w = image_rgb.shape[:2]
    centres = np.stack([_hex_to_lab(hex_color) for _, hex_color in results])
    pixels = _rgb_to_lab(image_rgb).reshape(-1, 3)
    labels = edge_aware_labels(pixels, centres).reshape(h, w)

    out: list[tuple[np.ndarray, str]] = []
    for k, (mask, hex_color) in enumerate(results):
        new_mask = labels == k
        if mask.dtype != np.bool_:
            new_mask = np.where(new_mask, np.uint8(255), np.uint8(0))
        out.append((new_mask, hex_color))
    return out
