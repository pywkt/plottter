"""Optional numba JIT shim.

When numba is installed (``pip install -e ".[fast]"``), ``njit`` is the real
``numba.njit`` decorator and ``JIT_ENABLED`` is ``True``.

When numba is *not* installed, ``njit`` is a transparent no-op that accepts
both the bare-decorator form::

    @njit
    def f(x): ...

and the decorator-factory form::

    @njit(cache=True, fastmath=True)
    def f(x): ...

In either case the decorated function is returned unchanged.
"""

from __future__ import annotations

try:
    from numba import njit  # type: ignore[import-untyped]

    JIT_ENABLED: bool = True
except ImportError:
    JIT_ENABLED = False

    def njit(func=None, **kwargs):  # type: ignore[misc]
        """No-op replacement for numba.njit.

        Supports both ``@njit`` and ``@njit(cache=True, ...)`` patterns.
        """
        if func is not None:
            # Used as @njit directly — func is the decorated callable.
            return func
        # Used as @njit(...) — return a decorator.
        def decorator(f):
            return f

        return decorator


__all__ = ["JIT_ENABLED", "njit"]
