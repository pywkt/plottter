"""Import-and-smoke test for scripts/benchmark_processing.py.

Verifies that:
  1. The benchmark script imports without errors (``pytest --collect-only`` CI check).
  2. The synthetic workload generators return the expected number of paths.
  3. The pipeline runner completes on a tiny workload without raising.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the benchmark module from scripts/ without requiring it to be a package
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parent.parent / "scripts" / "benchmark_processing.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_processing", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Cache so it's only loaded once per test session
@pytest.fixture(scope="module")
def bm():
    return _load_benchmark()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBenchmarkImport:
    """The script imports without errors."""

    def test_module_loads(self):
        mod = _load_benchmark()
        assert mod is not None

    def test_jit_active_is_bool(self, bm):
        assert isinstance(bm.JIT_ACTIVE, bool)

    def test_mode_label_is_str(self, bm):
        assert isinstance(bm.MODE_LABEL, str)
        assert bm.MODE_LABEL in ("JIT (numba)", "pure Python")


class TestSyntheticGenerators:
    """Generators return the requested number of paths."""

    def test_stipple_count(self, bm):
        paths = bm.make_stipple_scatter(n_paths=50)
        assert len(paths) == 50

    def test_map_count(self, bm):
        paths = bm.make_map_network(n_paths=100)
        assert len(paths) == 100

    def test_flow_count(self, bm):
        paths = bm.make_flow_scatter(n_paths=200)
        assert len(paths) == 200

    def test_stipple_has_points(self, bm):
        paths = bm.make_stipple_scatter(n_paths=10)
        for p in paths:
            assert len(p) >= 2

    def test_map_has_points(self, bm):
        paths = bm.make_map_network(n_paths=20)
        for p in paths:
            assert len(p) >= 2

    def test_flow_has_points(self, bm):
        paths = bm.make_flow_scatter(n_paths=20)
        for p in paths:
            assert len(p) >= 2


class TestPipelineRunnerSmoke:
    """run_pipeline completes on a tiny input and returns expected keys."""

    def test_stipple_pipeline(self, bm):
        paths = bm.make_stipple_scatter(n_paths=30)
        result = bm.run_pipeline(paths, "stipple-small")
        assert "timings" in result
        assert "counts" in result
        assert "total" in result["timings"]
        for stage in ("weld", "simplify", "merge", "reorder", "2opt", "or_opt"):
            assert stage in result["timings"]
        assert result["counts"]["input"] == 30
        assert result["counts"]["output"] >= 1

    def test_map_pipeline(self, bm):
        paths = bm.make_map_network(n_paths=40)
        result = bm.run_pipeline(paths, "map-small")
        assert result["timings"]["total"] > 0.0

    def test_all_timings_nonneg(self, bm):
        paths = bm.make_flow_scatter(n_paths=50)
        result = bm.run_pipeline(paths, "flow-small")
        for stage, t in result["timings"].items():
            assert t >= 0.0, f"Stage {stage!r} returned negative time {t}"
