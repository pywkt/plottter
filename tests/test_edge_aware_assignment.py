"""Edge-aware reassignment for partitioning colour separators.

Blended edge pixels (anti-aliasing, JPEG ringing) are mixes of two
neighbouring colours.  Nearest-colour assignment sends a mid-gray between
black ink and white paper to a *dark teal* pen, so the teal layer ends up
carrying a ghost of every black line.  ``edge_aware_reassign`` treats each
pair of pens as a candidate segment in Lab and sends a pixel to the nearer
endpoint of its nearest segment, so blends only ever land on one of the two
colours they came from.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from plottter.color import edge_aware_reassign, kmeans_separate, palette_separate
from plottter.color.edge_aware import edge_aware_labels
from plottter.color.palette import PenPalette

TEAL = "#205868"
RED = "#B06058"
BLACK = "#000000"
WHITE = "#FFFFFF"


def _hex(h):
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def _line_art(seed=0):
    """White page, a teal rectangle and a red rectangle, thin black lines
    everywhere; rendered 4× and box-downsampled so every edge is blended."""
    import cv2

    s = 4
    h, w = 200, 300
    img = np.full((h * s, w * s, 3), 255, np.uint8)
    cv2.rectangle(img, (20 * s, 40 * s), (120 * s, 180 * s), _hex(TEAL), -1)
    cv2.rectangle(img, (170 * s, 30 * s), (280 * s, 120 * s), _hex(RED), -1)
    rng = np.random.default_rng(seed)
    for _ in range(40):
        p1 = tuple(int(v) for v in rng.integers(0, (w * s, h * s)))
        p2 = tuple(int(v) for v in rng.integers(0, (w * s, h * s)))
        cv2.line(img, p1, p2, (0, 0, 0), 2 * s)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _gray_fraction(mask, img):
    """Share of a mask that is near-neutral (blended black/white) pixels."""
    m = mask.astype(bool)
    if not m.any():
        return 0.0
    px = img[m].astype(int)
    spread = px.max(axis=1) - px.min(axis=1)
    return float((spread < 12).mean())


# ----------------------------------------------------------------------
# edge_aware_labels
# ----------------------------------------------------------------------


def test_pure_colours_stay_on_their_pen():
    centres = np.array([[0, 0, 0], [100, 0, 0], [40, -20, -10]], np.float32)
    labels = edge_aware_labels(centres.copy(), centres)
    assert labels.tolist() == [0, 1, 2]


def test_blend_goes_to_nearer_endpoint_not_third_colour():
    black = np.array([0.0, 0.0, 0.0], np.float32)
    white = np.array([100.0, 0.0, 0.0], np.float32)
    teal = np.array([40.0, -12.0, -8.0], np.float32)  # dark, low chroma
    centres = np.stack([black, white, teal])
    # Mid-grays: nearest *point* is teal, but they lie on the black–white segment.
    grays = np.array([[30, 0, 0], [45, 0, 0], [55, 0, 0], [70, 0, 0]], np.float32)
    assert np.argmin(((grays[1] - centres) ** 2).sum(axis=1)) == 2  # sanity: teal wins by point
    labels = edge_aware_labels(grays, centres)
    assert labels.tolist() == [0, 0, 1, 1]


def test_blend_between_black_and_red_avoids_teal():
    black = np.array([0.0, 0.0, 0.0], np.float32)
    red = np.array([55.0, 35.0, 20.0], np.float32)
    teal = np.array([36.0, -10.0, -9.0], np.float32)
    centres = np.stack([black, red, teal])
    px = np.array([black * 0.7 + red * 0.3, black * 0.3 + red * 0.7], np.float32)
    assert edge_aware_labels(px, centres).tolist() == [0, 1]


def test_single_centre():
    assert edge_aware_labels(np.zeros((5, 3), np.float32), np.zeros((1, 3), np.float32)).tolist() == [0] * 5


def test_blocking_matches_unblocked():
    from plottter.color import edge_aware as mod

    rng = np.random.default_rng(1)
    px = rng.uniform(-50, 100, (5000, 3)).astype(np.float32)
    centres = rng.uniform(-50, 100, (5, 3)).astype(np.float32)
    full = edge_aware_labels(px, centres)
    with patch.object(mod, "_BLOCK", 700):
        blocked = edge_aware_labels(px, centres)
    assert np.array_equal(full, blocked)


# ----------------------------------------------------------------------
# edge_aware_reassign on separator output
# ----------------------------------------------------------------------


def test_palette_ghost_lines_leave_the_teal_layer():
    img = _line_art()
    pal = PenPalette(name="t", colors=[TEAL, RED, BLACK, WHITE])
    before = palette_separate(img, pal)
    after = edge_aware_reassign(img, before)

    teal_before = before[0][0] > 0
    teal_after = after[0][0] > 0
    # Nearest-colour put a meaningful share of neutral blends on teal.
    assert _gray_fraction(teal_before, img) > 0.08
    # Edge-aware leaves (almost) none, and only *removes* pixels from teal.
    assert _gray_fraction(teal_after, img) < 0.01
    assert (teal_after & ~teal_before).sum() < 0.002 * teal_before.sum()
    # The solid teal interior is untouched.
    interior = np.zeros(img.shape[:2], bool)
    interior[60:160, 40:100] = True
    pure = interior & (np.abs(img.astype(int) - np.array(_hex(TEAL))).sum(axis=2) < 6)
    assert pure.sum() > 1000
    assert teal_after[pure].all()


def test_reassign_preserves_order_hex_dtype_and_partition():
    img = _line_art()
    pal = PenPalette(name="t", colors=[TEAL, RED, BLACK, WHITE])
    before = palette_separate(img, pal)
    after = edge_aware_reassign(img, before)
    assert [h for _, h in after] == [TEAL, RED, BLACK, WHITE]
    assert all(m.dtype == np.uint8 for m, _ in after)
    total = sum((m > 0).astype(int) for m, _ in after)
    assert np.all(total == 1)  # still exhaustive and mutually exclusive


def test_reassign_on_kmeans_bool_masks():
    img = _line_art()
    # Five clusters: white, black, teal, red and one for the gray blends
    # (with four, K-Means merges teal and red rather than give up the grays).
    before = kmeans_separate(img, 5)
    after = edge_aware_reassign(img, before)
    assert all(m.dtype == np.bool_ for m, _ in after)
    assert [h for _, h in after] == [h for _, h in before]
    total = sum(m.astype(int) for m, _ in after)
    assert np.all(total == 1)
    # The cluster nearest the teal pen sheds its neutral ghosts.
    i = int(np.argmin([
        sum((a - b) ** 2 for a, b in zip(_hex(h), _hex(TEAL))) for _, h in before
    ]))
    gf_before = _gray_fraction(before[i][0], img)
    assert gf_before > 0.02, (before[i][1], gf_before)
    assert _gray_fraction(after[i][0], img) < gf_before / 3


def test_fewer_than_two_layers_is_a_no_op():
    img = _line_art()
    only = [(np.ones(img.shape[:2], bool), BLACK)]
    assert edge_aware_reassign(img, only) is only


def test_rejects_non_rgb():
    with pytest.raises(ValueError):
        edge_aware_reassign(np.zeros((4, 4), np.uint8), [(np.ones((4, 4), bool), BLACK), (np.zeros((4, 4), bool), WHITE)])


# ----------------------------------------------------------------------
# Panel wiring
# ----------------------------------------------------------------------


def _make_project():
    from plottter.models import Canvas, Layer, Project

    proj = Project(name="Test", canvas=Canvas.from_preset("A4"))
    proj.add_layer(Layer(name="Layer 1", color="#000000"))
    return proj


@pytest.fixture
def panel(qapp, qtbot):
    from plottter.gui.project_controller import ProjectController
    from plottter.gui.settings_panel import SettingsPanel

    sp = SettingsPanel(ProjectController(_make_project()))
    qtbot.addWidget(sp)
    sp.on_mode_changed("Color Separation")
    return sp


def test_checkbox_default_on_and_visible_only_for_colour_partitions(panel):
    assert panel._edge_aware_check.isChecked()
    for method, visible in (
        ("K-Means", True), ("Custom Palette", True), ("Luminance", False),
        ("RGB", False), ("CMYK", False), ("AI Layer Separation", False),
    ):
        panel._color_sep_method_combo.setCurrentText(method)
        assert not panel._edge_aware_check.isHidden() == visible, method


def test_checkbox_persists_to_qsettings(panel):
    from PyQt6.QtCore import QSettings

    panel._edge_aware_check.setChecked(False)
    assert QSettings("Plottter", "Plottter").value("colorsep/edge_aware") == "false"
    panel._edge_aware_check.setChecked(True)
    assert QSettings("Plottter", "Plottter").value("colorsep/edge_aware") == "true"


@pytest.mark.parametrize("method,dither,expect_call", [
    ("K-Means", "none", True),
    ("Custom Palette", "none", True),
    ("Custom Palette", "floyd-steinberg", False),
    ("Luminance", "none", False),
])
def test_compute_separation_applies_reassign_when_enabled(method, dither, expect_call):
    from plottter.gui.settings_panel._colorsep import _compute_separation

    img = _line_art()
    pal = PenPalette(name="t", colors=[TEAL, RED, BLACK, WHITE])
    with patch("plottter.color.edge_aware_reassign", wraps=edge_aware_reassign) as spy:
        payload = _compute_separation(
            method, source=img, params={}, max_px=0, num=4, thresholds=None,
            enabled_channels={}, k_amount=1.0, dither=dither, palette=pal, edge_aware=True,
        )
    assert spy.called == expect_call
    assert len(payload["results"]) == 4
    with patch("plottter.color.edge_aware_reassign", wraps=edge_aware_reassign) as spy:
        _compute_separation(
            method, source=img, params={}, max_px=0, num=4, thresholds=None,
            enabled_channels={}, k_amount=1.0, dither=dither, palette=pal, edge_aware=False,
        )
    assert not spy.called
