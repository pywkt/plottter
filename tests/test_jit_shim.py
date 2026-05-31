"""Tests for the optional numba JIT shim (processing/_jit.py)."""

import sys
import types
import importlib
import pytest


# ---------------------------------------------------------------------------
# Helper: reload _jit with numba absent from sys.modules
# ---------------------------------------------------------------------------

def _import_jit_without_numba():
    """Import _jit in an environment where numba is not available."""
    # Remove any cached import of the module under test.
    for key in list(sys.modules):
        if key == "plottter.processing._jit" or key == "_jit":
            del sys.modules[key]

    # Block numba by inserting a finder that always raises ImportError.
    class _BlockNumba:
        def find_module(self, name, path=None):
            if name == "numba" or name.startswith("numba."):
                return self
            return None

        def load_module(self, name):
            raise ImportError(f"numba blocked for testing: {name}")

    blocker = _BlockNumba()
    sys.meta_path.insert(0, blocker)
    try:
        import plottter.processing._jit as jit_mod
        # Force a fresh reload so the import machinery runs again.
        importlib.reload(jit_mod)
        return jit_mod
    finally:
        sys.meta_path.remove(blocker)
        # Clean up so subsequent imports are unaffected.
        for key in list(sys.modules):
            if key == "plottter.processing._jit":
                del sys.modules[key]


# ---------------------------------------------------------------------------
# Tests that always apply (numba may or may not be installed)
# ---------------------------------------------------------------------------

class TestJitImportAlwaysSucceeds:
    def test_import_succeeds(self):
        """Importing _jit must never raise."""
        import plottter.processing._jit  # noqa: F401

    def test_jit_enabled_is_bool(self):
        """JIT_ENABLED must be a plain bool."""
        from plottter.processing._jit import JIT_ENABLED
        assert isinstance(JIT_ENABLED, bool)

    def test_njit_is_callable(self):
        """njit must be callable regardless of whether numba is present."""
        from plottter.processing._jit import njit
        assert callable(njit)


# ---------------------------------------------------------------------------
# Tests for the fallback shim (numba absent)
# ---------------------------------------------------------------------------

class TestFallbackShim:
    @pytest.fixture(autouse=True)
    def _jit_no_numba(self):
        """Reload _jit with numba blocked; expose as self.jit."""
        self.jit = _import_jit_without_numba()

    def test_jit_enabled_false(self):
        assert self.jit.JIT_ENABLED is False

    def test_bare_decorator_returns_same_function(self):
        """@njit (no arguments) must return the original function unchanged."""
        njit = self.jit.njit

        @njit
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_decorator_factory_returns_same_function(self):
        """@njit(cache=True) must return the original function unchanged."""
        njit = self.jit.njit

        @njit(cache=True)
        def mul(a, b):
            return a * b

        assert mul(4, 5) == 20

    def test_decorator_factory_with_multiple_kwargs(self):
        """@njit(cache=True, fastmath=True) must also work."""
        njit = self.jit.njit

        @njit(cache=True, fastmath=True)
        def sub(a, b):
            return a - b

        assert sub(10, 3) == 7

    def test_decorated_function_identity(self):
        """The shim must return the *exact same* function object."""
        njit = self.jit.njit

        def original(x):
            return x * 2

        wrapped_bare = njit(original)
        assert wrapped_bare is original

        def original2(x):
            return x + 1

        wrapped_factory = njit(cache=True)(original2)
        assert wrapped_factory is original2


# ---------------------------------------------------------------------------
# Tests when numba IS installed (skipped otherwise)
# ---------------------------------------------------------------------------

class TestWithNumba:
    @pytest.fixture(autouse=True)
    def _check_numba(self):
        pytest.importorskip("numba", reason="numba not installed")

    def test_jit_enabled_true(self):
        from plottter.processing._jit import JIT_ENABLED
        assert JIT_ENABLED is True

    def test_njit_is_numba_njit(self):
        from plottter.processing._jit import njit
        import numba
        assert njit is numba.njit
