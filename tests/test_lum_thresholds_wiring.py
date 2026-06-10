"""Wiring tests for the Luminance custom-threshold controls.

luminance_separate() already accepted explicit band boundaries; the panel
now exposes them. These tests exercise _gather_lum_thresholds (the bit that
feeds luminance_separate) directly against a stub that mimics the mixin
contract, without spinning up the full Qt panel.
"""

from __future__ import annotations

import numpy as np

from plottter.color import luminance_separate
from plottter.gui.settings_panel._colorsep import _ColorSepMixin


class _StubSpin:
    def __init__(self, value: int) -> None:
        self._v = value

    def value(self) -> int:
        return self._v


class _StubCheck:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _Panel(_ColorSepMixin):
    def __init__(self, *, custom: bool, bands: int, thresholds: list[int]) -> None:
        self._lum_custom_check = _StubCheck(custom)
        self._color_sep_num_colors_spin = _StubSpin(bands)
        self._lum_threshold_spins = [_StubSpin(t) for t in thresholds]


def test_returns_none_when_custom_off():
    p = _Panel(custom=False, bands=3, thresholds=[85, 170])
    assert p._gather_lum_thresholds() is None


def test_returns_none_when_no_spins():
    p = _Panel(custom=True, bands=3, thresholds=[])
    assert p._gather_lum_thresholds() is None


def test_returns_values_when_custom_on():
    p = _Panel(custom=True, bands=3, thresholds=[85, 170])
    assert p._gather_lum_thresholds() == [85.0, 170.0]


def test_sorts_out_of_order_values():
    p = _Panel(custom=True, bands=3, thresholds=[170, 85])
    assert p._gather_lum_thresholds() == [85.0, 170.0]


def test_falls_back_when_count_stale():
    # 4 bands needs 3 thresholds, but only 2 spins present -> even spacing.
    p = _Panel(custom=True, bands=4, thresholds=[85, 170])
    assert p._gather_lum_thresholds() is None


def test_gathered_thresholds_drive_luminance_separate():
    # The gathered list must be a valid argument to luminance_separate.
    p = _Panel(custom=True, bands=3, thresholds=[85, 170])
    thr = p._gather_lum_thresholds()
    img = np.tile(np.arange(256, dtype=np.uint8), (4, 1))  # 0..255 ramp
    results = luminance_separate(img, num_bands=3, thresholds=thr)
    assert len(results) == 3
    # Bands partition every pixel exactly once.
    total = np.zeros(img.shape, dtype=bool)
    for mask, _hex in results:
        assert not np.any(total & mask)
        total |= mask
    assert np.all(total)
