"""Tests for TurtleToy dynamic parameters and _dynamic_overrides (task 133.2).

Covers:
- get_dynamic_parameters() returns correct Parameter types from parsed code
- _dynamic_overrides applied for int/float variable → segment length changes
- _dynamic_overrides applied for choice variable → behaviour changes
- _dynamic_overrides applied for string variable → behaviour changes
- Unknown override keys are silently ignored (no error)
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from plottter.models.canvas import Canvas

# ---------------------------------------------------------------------------
# Load the plugin module directly (it lives in plugins/, not src/).
# ---------------------------------------------------------------------------

_PLUGIN_PATH = Path(__file__).parent.parent / "plugins" / "turtletoy.py"
_MODULE_NAME = "plottter_plugin_turtletoy"


def _load_plugin() -> Any:
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_tt = _load_plugin()
TurtleToyGenerator = _tt.TurtleToyGenerator

# Skip entire module if quickjs is not installed
pytestmark = pytest.mark.skipif(
    not _tt._QUICKJS_AVAILABLE,
    reason="quickjs package not installed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CANVAS = Canvas.from_preset("A4", margin=10.0)


def _run(code: str, overrides: dict[str, Any] | None = None) -> list[list[tuple[float, float]]]:
    """Run *code* with fixed_scale fit_mode (1 world unit = 1 mm)."""
    params: dict[str, Any] = {
        "code": code,
        "fit_mode": "fixed_scale",
        "mm_per_unit": 1.0,
        "seed": 0,
        "max_steps": 100_000,
        "timeout_seconds": 30.0,
    }
    if overrides:
        params["_dynamic_overrides"] = overrides
    return TurtleToyGenerator().generate(params, _CANVAS)


def _seg_len(polylines: list[list[tuple[float, float]]]) -> float:
    """Return the Euclidean length of the first segment in the first polyline."""
    assert polylines, "Expected at least one polyline"
    pl = polylines[0]
    assert len(pl) >= 2, "Expected at least 2 points in polyline"
    dx = pl[1][0] - pl[0][0]
    dy = pl[1][1] - pl[0][1]
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------
# Sketch fixtures
# ---------------------------------------------------------------------------

# A sketch with one integer adjustable variable that controls forward distance.
_DISTANCE_SKETCH = """\
const distance = 50; // min=10, max=200
const t = new Turtle();
t.forward(distance);
"""

# A sketch with a choice variable that selects between two forward distances.
_CHOICE_SKETCH = """\
const mode = 'small'; // (small, large)
const t = new Turtle();
if (mode === 'large') {
    t.forward(100.0);
} else {
    t.forward(50.0);
}
"""

# A sketch with a string variable whose length determines the forward distance.
_STRING_SKETCH = """\
const tag = "hi"; // type=string, A label
const dist = tag.length * 20.0;
const t = new Turtle();
t.forward(dist);
"""


# ---------------------------------------------------------------------------
# Tests: get_dynamic_parameters
# ---------------------------------------------------------------------------


class TestGetDynamicParameters:
    """get_dynamic_parameters() must reflect the adjustable vars in the code."""

    def _params(self, code: str) -> dict[str, Any]:
        gen = TurtleToyGenerator()
        plist = gen.get_dynamic_parameters({"code": code})
        return {p.name: p for p in plist}

    def test_int_var_returns_int_param(self) -> None:
        from plottter.generators.base import IntParam

        params = self._params(_DISTANCE_SKETCH)
        assert "distance" in params
        assert isinstance(params["distance"], IntParam)

    def test_int_param_default_min_max(self) -> None:
        from plottter.generators.base import IntParam

        params = self._params(_DISTANCE_SKETCH)
        p = params["distance"]
        assert isinstance(p, IntParam)
        assert p.default == 50
        assert p.min == 10
        assert p.max == 200

    def test_choice_var_returns_choice_param(self) -> None:
        from plottter.generators.base import ChoiceParam

        params = self._params(_CHOICE_SKETCH)
        assert "mode" in params
        p = params["mode"]
        assert isinstance(p, ChoiceParam)
        assert p.choices == ["small", "large"]
        assert p.default == "small"  # unquoted identifier from (small, large)

    def test_string_var_returns_string_param(self) -> None:
        from plottter.generators.base import StringParam

        params = self._params(_STRING_SKETCH)
        assert "tag" in params
        assert isinstance(params["tag"], StringParam)

    def test_empty_code_returns_empty_list(self) -> None:
        gen = TurtleToyGenerator()
        result = gen.get_dynamic_parameters({"code": ""})
        assert result == []

    def test_no_adjustable_vars_returns_empty_list(self) -> None:
        code = "const t = new Turtle();\nt.forward(10);\n"
        gen = TurtleToyGenerator()
        result = gen.get_dynamic_parameters({"code": code})
        assert result == []


# ---------------------------------------------------------------------------
# Tests: integer override → segment length
# ---------------------------------------------------------------------------


class TestIntegerOverride:
    """Overriding an int variable must change the turtle's forward distance."""

    def test_default_distance_50(self) -> None:
        """Without override, forward(50) → segment of ≈50 mm."""
        polylines = _run(_DISTANCE_SKETCH)
        assert abs(_seg_len(polylines) - 50.0) < 1e-6

    def test_override_distance_100(self) -> None:
        """With override distance=100, forward(100) → segment of exactly 100 mm."""
        polylines = _run(_DISTANCE_SKETCH, overrides={"distance": 100})
        assert abs(_seg_len(polylines) - 100.0) < 1e-6, (
            f"Expected segment length 100, got {_seg_len(polylines)}"
        )

    def test_override_distance_different_value(self) -> None:
        """Override to 150 → segment of exactly 150 mm."""
        polylines = _run(_DISTANCE_SKETCH, overrides={"distance": 150})
        assert abs(_seg_len(polylines) - 150.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests: choice override
# ---------------------------------------------------------------------------


class TestChoiceOverride:
    """Overriding a choice variable must change execution behaviour."""

    def test_default_choice_small(self) -> None:
        """Without override, mode='small' → forward(50) → 50 mm segment."""
        polylines = _run(_CHOICE_SKETCH)
        assert abs(_seg_len(polylines) - 50.0) < 1e-6

    def test_override_choice_large(self) -> None:
        """With override mode='large' → forward(100) → 100 mm segment."""
        polylines = _run(_CHOICE_SKETCH, overrides={"mode": "large"})
        assert abs(_seg_len(polylines) - 100.0) < 1e-6, (
            f"Expected segment length 100 (choice='large'), got {_seg_len(polylines)}"
        )


# ---------------------------------------------------------------------------
# Tests: string override
# ---------------------------------------------------------------------------


class TestStringOverride:
    """Overriding a string variable must change execution behaviour."""

    def test_default_string_hi(self) -> None:
        """Without override, tag='hi' (len=2) → forward(40) → 40 mm segment."""
        polylines = _run(_STRING_SKETCH)
        assert abs(_seg_len(polylines) - 40.0) < 1e-6

    def test_override_string_hello(self) -> None:
        """With override tag='hello' (len=5) → forward(100) → 100 mm segment."""
        polylines = _run(_STRING_SKETCH, overrides={"tag": "hello"})
        assert abs(_seg_len(polylines) - 100.0) < 1e-6, (
            f"Expected segment length 100 (tag='hello'), got {_seg_len(polylines)}"
        )


# ---------------------------------------------------------------------------
# Tests: unknown override keys silently ignored
# ---------------------------------------------------------------------------


class TestUnknownOverrideKeys:
    """Override keys not in the parsed code must be silently ignored."""

    def test_unknown_key_does_not_raise(self) -> None:
        """Passing an unknown key in _dynamic_overrides must not raise."""
        polylines = _run(_DISTANCE_SKETCH, overrides={"nonexistent_var": 999})
        # Behaviour unchanged — distance is still 50
        assert abs(_seg_len(polylines) - 50.0) < 1e-6

    def test_mixed_known_unknown_keys(self) -> None:
        """Known key is applied; unknown key is silently dropped."""
        polylines = _run(
            _DISTANCE_SKETCH,
            overrides={"distance": 75, "not_a_var": "abc"},
        )
        assert abs(_seg_len(polylines) - 75.0) < 1e-6

    def test_empty_overrides_is_a_noop(self) -> None:
        """Passing an empty dict for _dynamic_overrides must be a no-op."""
        polylines = _run(_DISTANCE_SKETCH, overrides={})
        assert abs(_seg_len(polylines) - 50.0) < 1e-6
