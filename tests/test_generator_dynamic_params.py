"""Tests for Generator.get_dynamic_parameters() default hook (phase 133.1)."""

from __future__ import annotations

from plottter.generators.parametric import ParametricGenerator


def test_default_returns_empty_list():
    """get_dynamic_parameters() default implementation returns []."""
    gen = ParametricGenerator()
    result = gen.get_dynamic_parameters({})
    assert result == []


def test_default_ignores_static_values():
    """Default hook returns [] regardless of what static_param_values contains."""
    gen = ParametricGenerator()
    result = gen.get_dynamic_parameters({"foo": 1, "bar": "baz"})
    assert result == []
