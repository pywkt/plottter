"""Tests for the boundary conversion between color-separation output and the
line generators' luminance-input convention.

Before the fix, CMYK/RGB channel images were fed directly into generators
that interpret their source as luminance — producing colour-negated plots
(lots of cyan ink where the source had no cyan, etc.).  The fix inverts
ink-coverage channels at the color-sep boundary so every generator gets
luminance regardless of separation method.

These tests pin the converter's contract and the downstream effect when
the converted image is fed through a real line generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.gui.settings_panel._colorsep import _separation_mask_to_luminance


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_boolean_mask_passes_through_luminance_with_outside_whitened():
    """K-Means / Luminance separation outputs a boolean cluster mask.  The
    helper must take the source grayscale and force every NON-cluster pixel
    to pure white — leaving cluster pixels at their original brightness."""
    src = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    mask = np.array([[True, False, True], [False, True, False]])
    out = _separation_mask_to_luminance(mask, src)
    expected = np.array([[10, 255, 30], [255, 50, 255]], dtype=np.uint8)
    assert np.array_equal(out, expected)


def test_boolean_mask_with_rgb_source_uses_grayscale_of_source():
    """If the source is RGB, the helper greys it first then applies the mask."""
    src_rgb = np.full((2, 2, 3), 200, dtype=np.uint8)
    src_rgb[0, 0] = [255, 0, 0]
    mask = np.array([[True, True], [True, True]])
    out = _separation_mask_to_luminance(mask, src_rgb)
    assert out.shape == (2, 2)
    assert out.dtype == np.uint8


def test_cmyk_channel_image_is_inverted_to_luminance():
    """A uint8 CMYK/RGB channel image (255 = lots of this ink wanted) must
    come back inverted, so high channel intensity reads as DARK to a
    luminance-based generator (the convention every line generator uses)."""
    # Bright cyan channel (lots of cyan ink wanted) at pixel (0, 0),
    # no cyan at (1, 1).
    channel = np.array([[255, 128], [64, 0]], dtype=np.uint8)
    out = _separation_mask_to_luminance(channel, src_img=channel)
    expected = np.array([[0, 127], [191, 255]], dtype=np.uint8)
    assert np.array_equal(out, expected)


def test_full_white_channel_becomes_full_black_luminance():
    """A 'full ink wanted everywhere' channel reads as 'pure black' to the
    luminance-based generator, so it will draw the densest possible ink."""
    channel = np.full((10, 10), 255, dtype=np.uint8)
    out = _separation_mask_to_luminance(channel, src_img=channel)
    assert (out == 0).all()


def test_full_black_channel_becomes_full_white_luminance():
    """A 'no ink wanted anywhere' channel reads as 'pure white' so the
    generator skips drawing entirely."""
    channel = np.zeros((10, 10), dtype=np.uint8)
    out = _separation_mask_to_luminance(channel, src_img=channel)
    assert (out == 255).all()


# ---------------------------------------------------------------------------
# End-to-end: red square through CMYK separation + Paired Wave generator
# ---------------------------------------------------------------------------


def test_paired_wave_draws_ink_where_cmyk_channel_is_strong():
    """The fix's real test: feed a CMYK channel image of a red square
    through the converter + Paired Wave Shading, and verify the generator
    puts lots of paired-line spread INSIDE the red region (where the
    magenta channel is bright) — not the inverse."""
    from plottter.color import cmyk_separate
    from plottter.generators.paired_wave_shading import PairedWaveShadingGenerator
    from plottter.models import Canvas

    # 200×200 white image with a 100×100 red square dead centre.
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    img[50:150, 50:150] = [255, 0, 0]
    results = cmyk_separate(img)
    # Magenta channel: high inside the red square, zero outside.
    _c, (m_channel, _), _y, _k = results
    assert m_channel[100, 100] == 255   # inside red
    assert m_channel[10, 10] == 0       # outside (white)

    # Run the converter that color sep applies before passing to a generator.
    lum = _separation_mask_to_luminance(m_channel, src_img=img)
    # After inversion the bright-channel area must now be dark.
    assert lum[100, 100] == 0
    assert lum[10, 10] == 255

    gen = PairedWaveShadingGenerator()
    params = {
        "line_spacing_mm": 5.0,
        "max_deviation_mm": 4.0,
        "min_deviation_mm": 0.0,
        "sample_interval_mm": 0.5,
        "tone_gamma": 1.0,
        "smoothing_mm": 0.0,
        # Never skip — we need pair samples everywhere so we can measure
        # the gap difference between inside-square and outside-square.
        "skip_white_above": 255,
        "image_fit_mode": "fit",
        "_source_image": lum,
    }
    paths = gen.generate(params, Canvas(width_mm=200, height_mm=200, margin_mm=0))

    # For every paired (top, bot), measure the gap at samples INSIDE the
    # red square (x ≈ 100 mm) and at samples OUTSIDE (x ≈ 25 mm).
    pair_iter = zip(paths[::2], paths[1::2])
    gaps_inside: list[float] = []
    gaps_outside: list[float] = []
    for top, bot in pair_iter:
        for (tx, ty), (bx, by) in zip(top, bot):
            assert tx == pytest.approx(bx)
            if 70 < tx < 130 and 70 < ty < 130:
                gaps_inside.append(abs(ty - by))
            elif (tx < 40 or tx > 160) and (ty < 40 or ty > 160):
                gaps_outside.append(abs(ty - by))

    assert gaps_inside and gaps_outside
    mean_in = sum(gaps_inside) / len(gaps_inside)
    mean_out = sum(gaps_outside) / len(gaps_outside)
    assert mean_in > 3.0, f"Expected wide pair inside red square, got mean {mean_in}"
    assert mean_out < 0.5, f"Expected nearly-touching pair outside, got mean {mean_out}"
