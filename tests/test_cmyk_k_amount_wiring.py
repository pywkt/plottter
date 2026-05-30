"""End-to-end test for the K Amount auto-refresh on the Color Separation panel.

Bug: K Amount fed into ``cmyk_separate`` at separation time, but the
panel cached the resulting channel masks in ``_layer_masks`` and Generate
Lines only read from there.  Moving the K Amount spinbox after Separate
left the cache stale, so the next Generate Lines used the old K values.

Fix: ``_on_cmyk_k_amount_changed`` re-runs ``cmyk_separate`` with the
current spinbox value and swaps the cached masks in-place.  These tests
prove the wiring without spinning up the full GUI panel by exercising
the handler directly against a stub that mimics its mixin contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from plottter.gui.settings_panel._colorsep import _ColorSepMixin


class _StubSpin:
    """Minimal stand-in for a QDoubleSpinBox that the handler reads."""

    def __init__(self, value: float) -> None:
        self._v = value

    def value(self) -> float:
        return self._v

    def set(self, v: float) -> None:
        self._v = v


class _StubCheckBox:
    """Minimal stand-in for the channel-include checkboxes."""

    def __init__(self, checked: bool = True) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _Panel(_ColorSepMixin):
    """Just the mixin — no Qt parent.  We construct the state by hand so
    we can call the handler under controlled conditions."""

    def __init__(self, raw_rgb: np.ndarray, k_amount: float) -> None:
        self._cmyk_k_amount_spin = _StubSpin(k_amount)
        self._channel_checks: dict = {
            "Cyan": _StubCheckBox(),
            "Magenta": _StubCheckBox(),
            "Yellow": _StubCheckBox(),
            "Key (Black)": _StubCheckBox(),
        }
        self._cmyk_raw_rgb = raw_rgb
        self._last_sep_method = "CMYK"
        # Seed the mask cache as if Separate had just run at k_amount=1.0.
        from plottter.color import cmyk_separate
        results = cmyk_separate(raw_rgb, k_amount=1.0)
        self._separated_layer_ids = ["c-id", "m-id", "y-id", "k-id"]
        self._layer_masks = {
            lid: (results[i][0], np.zeros((1, 1), dtype=np.uint8))
            for i, lid in enumerate(self._separated_layer_ids)
        }


def _mid_gray_image() -> np.ndarray:
    """Mid-gray image generates K ≈ 128 at full k_amount — visible target."""
    return np.full((4, 4, 3), 128, dtype=np.uint8)


def test_changing_k_amount_updates_cached_k_mask_in_place():
    """The handler must replace the K layer's cached mask with the new
    cmyk_separate output at the current spinbox value, without touching
    the layer ID list (so any user-generated paths on the layer stay)."""
    panel = _Panel(_mid_gray_image(), k_amount=1.0)
    k_mask_before = panel._layer_masks["k-id"][0].copy()
    layer_ids_before = list(panel._separated_layer_ids)
    assert k_mask_before.max() > 100, "preseed K must be substantial"

    panel._cmyk_k_amount_spin.set(0.3)
    panel._on_cmyk_k_amount_changed(0.3)

    k_mask_after = panel._layer_masks["k-id"][0]
    # Mask shrank to ≈ 30% of original
    assert k_mask_after.max() < k_mask_before.max() // 2
    assert k_mask_after.max() == pytest.approx(k_mask_before.max() * 0.3, abs=2)
    # Layer-ID list untouched — user's paths on those layers survive.
    assert panel._separated_layer_ids == layer_ids_before


def test_changing_k_amount_to_zero_zeros_the_k_mask():
    panel = _Panel(_mid_gray_image(), k_amount=1.0)
    panel._cmyk_k_amount_spin.set(0.0)
    panel._on_cmyk_k_amount_changed(0.0)
    assert panel._layer_masks["k-id"][0].max() == 0


def test_changing_k_amount_leaves_cmy_masks_unchanged():
    panel = _Panel(_mid_gray_image(), k_amount=1.0)
    cmy_before = [panel._layer_masks[lid][0].copy() for lid in ("c-id", "m-id", "y-id")]
    panel._cmyk_k_amount_spin.set(0.1)
    panel._on_cmyk_k_amount_changed(0.1)
    cmy_after = [panel._layer_masks[lid][0] for lid in ("c-id", "m-id", "y-id")]
    for before, after in zip(cmy_before, cmy_after):
        assert np.array_equal(before, after)


def test_handler_no_op_when_no_separation_yet():
    """If the user never clicked Separate, the spinbox handler must do
    nothing — no crash, no stale-state mutation."""
    panel = _Panel(_mid_gray_image(), k_amount=1.0)
    panel._separated_layer_ids = []
    panel._layer_masks = {}
    panel._on_cmyk_k_amount_changed(0.5)
    assert panel._layer_masks == {}


def test_handler_no_op_when_last_method_was_not_cmyk():
    """If the user separated via K-Means (say) and then moves the CMYK
    K Amount spin, the K Amount has no business touching the cluster
    masks of an unrelated separation."""
    panel = _Panel(_mid_gray_image(), k_amount=1.0)
    panel._last_sep_method = "K-Means"
    snapshot = {lid: m[0].copy() for lid, m in panel._layer_masks.items()}
    panel._on_cmyk_k_amount_changed(0.0)
    for lid, original in snapshot.items():
        assert np.array_equal(panel._layer_masks[lid][0], original)


def test_handler_respects_unchecked_channels():
    """If the user separated with only the K box checked, the layer-id
    list has 1 entry.  The refresh must still pair correctly with the
    filtered cmyk_separate output."""
    panel = _Panel(_mid_gray_image(), k_amount=1.0)
    # Simulate the Separate step having dropped C/M/Y
    panel._channel_checks["Cyan"] = _StubCheckBox(checked=False)
    panel._channel_checks["Magenta"] = _StubCheckBox(checked=False)
    panel._channel_checks["Yellow"] = _StubCheckBox(checked=False)
    panel._separated_layer_ids = ["k-id"]
    panel._layer_masks = {"k-id": panel._layer_masks["k-id"]}

    panel._cmyk_k_amount_spin.set(0.5)
    panel._on_cmyk_k_amount_changed(0.5)

    k_max = int(panel._layer_masks["k-id"][0].max())
    # ~64 expected for mid-gray K=128 × 0.5
    assert 55 < k_max < 75
