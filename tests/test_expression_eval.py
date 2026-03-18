"""Tests for the safe expression evaluator."""

from __future__ import annotations

import math

import pytest

from plottter.generators.expression_eval import ExpressionError, SafeEvaluator


@pytest.fixture
def evaluator() -> SafeEvaluator:
    return SafeEvaluator()


# ---------------------------------------------------------------------------
# Valid expressions
# ---------------------------------------------------------------------------


def test_simple_arithmetic(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("2 + 3 * 4", [])
    assert fn() == 14.0


def test_variable_substitution(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("t * 2", ["t"])
    assert fn(t=3) == 6.0


def test_variable_in_complex_expression(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("t * t + 1", ["t"])
    assert fn(t=4) == 17.0


def test_power_operator(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("t ** 2", ["t"])
    assert fn(t=5) == 25.0


def test_negative_literal(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("-1 * t", ["t"])
    assert fn(t=3) == -3.0


def test_unary_minus_on_variable(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("-t", ["t"])
    assert fn(t=7) == -7.0


def test_floordiv(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("t // 2", ["t"])
    assert fn(t=7) == 3.0


def test_modulo(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("t % 3", ["t"])
    assert fn(t=7) == 1.0


# ---------------------------------------------------------------------------
# Whitelisted functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,variable,value,expected",
    [
        ("sin(t)", "t", 0.0, 0.0),
        ("cos(t)", "t", 0.0, 1.0),
        ("tan(t)", "t", 0.0, 0.0),
        ("asin(t)", "t", 0.0, 0.0),
        ("acos(t)", "t", 1.0, 0.0),
        ("atan(t)", "t", 0.0, 0.0),
        ("abs(t)", "t", -5.0, 5.0),
        ("sqrt(t)", "t", 9.0, 3.0),
        ("log(t)", "t", math.e, 1.0),
        ("log2(t)", "t", 8.0, 3.0),
        ("log10(t)", "t", 100.0, 2.0),
        ("exp(t)", "t", 0.0, 1.0),
        ("pow(t, 3)", "t", 2.0, 8.0),
        ("floor(t)", "t", 3.7, 3.0),
        ("ceil(t)", "t", 3.2, 4.0),
        ("round(t)", "t", 3.6, 4.0),
    ],
)
def test_whitelisted_function(
    evaluator: SafeEvaluator,
    expr: str,
    variable: str,
    value: float,
    expected: float,
) -> None:
    fn = evaluator.compile(expr, [variable])
    result = fn(**{variable: value})
    assert abs(result - expected) < 1e-9


def test_atan2(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("atan2(y, x)", ["x", "y"])
    result = fn(x=1.0, y=1.0)
    assert abs(result - math.pi / 4) < 1e-9


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constant_pi(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("pi", [])
    assert abs(fn() - math.pi) < 1e-12


def test_constant_e(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("e", [])
    assert abs(fn() - math.e) < 1e-12


def test_constant_tau(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("tau", [])
    assert abs(fn() - math.tau) < 1e-12


def test_constant_in_expression(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("sin(pi / 2)", [])
    assert abs(fn() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Security: rejected constructs
# ---------------------------------------------------------------------------


def test_rejects_import(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError):
        evaluator.compile("__import__('os')", [])


def test_rejects_class_access(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError):
        evaluator.compile("().__class__", [])


def test_rejects_eval(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError):
        evaluator.compile("eval('1+1')", [])


def test_rejects_attribute_access(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError):
        evaluator.compile("t.real", ["t"])


def test_rejects_list_comprehension(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError):
        evaluator.compile("[x for x in range(10)]", ["x"])


def test_rejects_unknown_function(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError):
        evaluator.compile("open('/etc/passwd')", [])


def test_rejects_unknown_name(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError):
        evaluator.compile("undefined_var", [])


def test_rejects_string_constant(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError):
        evaluator.compile("'hello'", [])


def test_rejects_lambda(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError):
        evaluator.compile("lambda x: x", [])


def test_rejects_keyword_args_in_call(evaluator: SafeEvaluator) -> None:
    # sin(x=0) is invalid in the evaluator
    with pytest.raises(ExpressionError):
        evaluator.compile("sin(x=0)", [])


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------


def test_syntax_error_message(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError, match="Syntax error"):
        evaluator.compile("t +* 2", ["t"])


def test_empty_expression(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError, match="empty"):
        evaluator.compile("", [])


def test_attribute_error_message(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError, match="Attribute access"):
        evaluator.compile("t.imag", ["t"])


def test_unknown_function_error_message(evaluator: SafeEvaluator) -> None:
    with pytest.raises(ExpressionError, match="whitelist"):
        evaluator.compile("exec('pass')", [])


# ---------------------------------------------------------------------------
# Complex valid expressions
# ---------------------------------------------------------------------------


def test_lissajous_x_expression(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("sin(3*t + pi/2)", ["t"])
    # At t=0: sin(pi/2) = 1
    assert abs(fn(t=0.0) - 1.0) < 1e-9


def test_nested_functions(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("sin(cos(t))", ["t"])
    result = fn(t=0.0)
    assert abs(result - math.sin(1.0)) < 1e-9


def test_multiple_variables(evaluator: SafeEvaluator) -> None:
    fn = evaluator.compile("a * t + b", ["a", "t", "b"])
    assert fn(a=2.0, t=3.0, b=1.0) == 7.0


def test_complex_butterfly_expression(evaluator: SafeEvaluator) -> None:
    """Butterfly curve expression should compile and produce finite values."""
    fn = evaluator.compile(
        "sin(t)*(exp(cos(t))-2*cos(4*t)-pow(sin(t/12),5))",
        ["t"],
    )
    result = fn(t=0.0)
    assert math.isfinite(result)
