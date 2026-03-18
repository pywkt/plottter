"""Safe mathematical expression evaluator based on AST parsing."""

from __future__ import annotations

import ast
import math
from typing import Any, Callable


class ExpressionError(Exception):
    """Raised when an expression is invalid or unsafe."""


_ALLOWED_FUNCS: dict[str, Any] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "abs": abs,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
}

_ALLOWED_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_ALLOWED_BIN_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.FloorDiv,
    ast.Mod,
)

_ALLOWED_UNARY_OPS = (ast.USub, ast.UAdd)


class SafeEvaluator:
    """Parse and compile math expressions in a restricted AST environment."""

    def compile(self, expr: str, variables: list[str]) -> Callable[..., float]:
        """Compile an expression string into a callable.

        The returned callable accepts keyword arguments for each variable
        and returns a float.

        Raises ExpressionError if the expression is invalid or unsafe.
        """
        stripped = expr.strip()
        if not stripped:
            raise ExpressionError("Expression cannot be empty")

        try:
            tree = ast.parse(stripped, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Syntax error in expression: {exc}") from exc

        allowed_names = set(_ALLOWED_FUNCS) | set(_ALLOWED_CONSTS) | set(variables)
        self._validate_node(tree.body, allowed_names)

        namespace: dict[str, Any] = {**_ALLOWED_FUNCS, **_ALLOWED_CONSTS}
        code = compile(tree, "<expression>", "eval")

        def _evaluate(**kwargs: float) -> float:
            local_ns = {**namespace, **kwargs}
            result = eval(code, {"__builtins__": {}}, local_ns)  # noqa: S307
            return float(result)

        return _evaluate

    def _validate_node(self, node: ast.AST, allowed_names: set[str]) -> None:
        """Recursively validate an AST node against the whitelist."""
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ExpressionError(
                    f"Only numeric constants are allowed, got {type(node.value).__name__!r}"
                )

        elif isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ExpressionError(
                    f"Name {node.id!r} is not allowed. "
                    f"Allowed names: {', '.join(sorted(allowed_names))}"
                )

        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BIN_OPS):
                raise ExpressionError(
                    f"Operator {type(node.op).__name__} is not allowed"
                )
            self._validate_node(node.left, allowed_names)
            self._validate_node(node.right, allowed_names)

        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OPS):
                raise ExpressionError(
                    f"Unary operator {type(node.op).__name__} is not allowed"
                )
            self._validate_node(node.operand, allowed_names)

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError(
                    "Only direct function calls are allowed "
                    "(e.g. sin(x) — not obj.method(x))"
                )
            if node.func.id not in _ALLOWED_FUNCS:
                raise ExpressionError(
                    f"Function {node.func.id!r} is not in the whitelist. "
                    f"Allowed functions: {', '.join(sorted(_ALLOWED_FUNCS))}"
                )
            if node.keywords:
                raise ExpressionError("Keyword arguments in function calls are not allowed")
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    raise ExpressionError("Starred (*args) arguments are not allowed")
                self._validate_node(arg, allowed_names)

        elif isinstance(node, ast.Attribute):
            raise ExpressionError(
                "Attribute access is not allowed (e.g. 't.real' is forbidden)"
            )

        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise ExpressionError("Comprehensions are not allowed in expressions")

        elif isinstance(node, ast.Lambda):
            raise ExpressionError("Lambda expressions are not allowed")

        elif isinstance(node, ast.IfExp):
            # Allow ternary: value_if_true if condition else value_if_false
            self._validate_node(node.test, allowed_names)
            self._validate_node(node.body, allowed_names)
            self._validate_node(node.orelse, allowed_names)

        elif isinstance(node, ast.Compare):
            self._validate_node(node.left, allowed_names)
            for comparator in node.comparators:
                self._validate_node(comparator, allowed_names)

        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                self._validate_node(value, allowed_names)

        else:
            raise ExpressionError(
                f"Expression construct {type(node).__name__!r} is not allowed"
            )
