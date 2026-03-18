"""Tests for the user preset persistence layer (Phase 26.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plottter.generators.base import Preset
from plottter.presets import (
    delete_user_preset,
    load_user_presets,
    rename_user_preset,
    save_user_preset,
)
from plottter.presets.user_presets import _generator_filename


# ---------------------------------------------------------------------------
# Helper: tmp directory for isolation
# ---------------------------------------------------------------------------


@pytest.fixture()
def presets_dir(tmp_path: Path) -> Path:
    d = tmp_path / "presets"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# _generator_filename
# ---------------------------------------------------------------------------


def test_generator_filename_basic():
    assert _generator_filename("Flow Field") == "flow_field"


def test_generator_filename_mixed_case():
    assert _generator_filename("FlowField") == "flowfield"


def test_generator_filename_special_chars():
    assert _generator_filename("My-Generator v2!") == "my_generator_v2"


def test_generator_filename_already_clean():
    assert _generator_filename("stipple") == "stipple"


# ---------------------------------------------------------------------------
# save_user_preset / load_user_presets
# ---------------------------------------------------------------------------


def test_save_creates_directory_and_file(tmp_path: Path):
    d = tmp_path / "new_dir"
    assert not d.exists()
    preset = Preset(name="My Preset", params={"foo": 1, "bar": "baz"})
    save_user_preset("Flow Field", preset, presets_dir=d)
    assert d.exists()
    assert (d / "flow_field.json").exists()


def test_save_and_load_round_trip(presets_dir: Path):
    preset = Preset(name="Test Preset", params={"alpha": 0.5, "beta": 42})
    save_user_preset("stipple", preset, presets_dir=presets_dir)

    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 1
    assert loaded[0].name == "Test Preset"
    assert loaded[0].params == {"alpha": 0.5, "beta": 42}


def test_save_multiple_presets(presets_dir: Path):
    p1 = Preset(name="First", params={"x": 1})
    p2 = Preset(name="Second", params={"x": 2})
    save_user_preset("stipple", p1, presets_dir=presets_dir)
    save_user_preset("stipple", p2, presets_dir=presets_dir)

    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 2
    names = {p.name for p in loaded}
    assert names == {"First", "Second"}


def test_save_overwrites_duplicate_name(presets_dir: Path):
    p1 = Preset(name="Same", params={"v": 1})
    p2 = Preset(name="Same", params={"v": 99})
    save_user_preset("stipple", p1, presets_dir=presets_dir)
    save_user_preset("stipple", p2, presets_dir=presets_dir)

    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 1
    assert loaded[0].params == {"v": 99}


def test_load_nonexistent_returns_empty(presets_dir: Path):
    result = load_user_presets("does_not_exist", presets_dir=presets_dir)
    assert result == []


def test_load_corrupt_json_returns_empty(presets_dir: Path):
    bad_file = presets_dir / "flow_field.json"
    bad_file.write_text("NOT { valid json !!!", encoding="utf-8")
    result = load_user_presets("Flow Field", presets_dir=presets_dir)
    assert result == []


def test_load_non_list_json_returns_empty(presets_dir: Path):
    bad_file = presets_dir / "flow_field.json"
    bad_file.write_text('{"name": "oops"}', encoding="utf-8")
    result = load_user_presets("Flow Field", presets_dir=presets_dir)
    assert result == []


def test_load_params_not_a_dict(presets_dir: Path):
    """params value is not a dict (e.g. 42) → entry is skipped, no TypeError raised."""
    data = [{"name": "x", "params": 42}]
    path = presets_dir / "stipple.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    result = load_user_presets("stipple", presets_dir=presets_dir)
    assert result == []


def test_load_skips_malformed_entries(presets_dir: Path):
    data = [
        {"name": "Good", "params": {"a": 1}},
        {"broken": True},  # missing name/params
        {"name": "Also Good", "params": {"b": 2}},
    ]
    path = presets_dir / "stipple.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 2
    assert {p.name for p in loaded} == {"Good", "Also Good"}


# ---------------------------------------------------------------------------
# delete_user_preset
# ---------------------------------------------------------------------------


def test_delete_removes_only_specified_preset(presets_dir: Path):
    save_user_preset("stipple", Preset(name="Keep", params={}), presets_dir=presets_dir)
    save_user_preset("stipple", Preset(name="Remove", params={}), presets_dir=presets_dir)

    delete_user_preset("stipple", "Remove", presets_dir=presets_dir)

    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 1
    assert loaded[0].name == "Keep"


def test_delete_nonexistent_preset_is_noop(presets_dir: Path):
    save_user_preset("stipple", Preset(name="Keep", params={}), presets_dir=presets_dir)
    # Should not raise.
    delete_user_preset("stipple", "NoSuchPreset", presets_dir=presets_dir)
    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 1


def test_delete_on_missing_file_is_noop(presets_dir: Path):
    # Should not raise.
    delete_user_preset("no_generator", "Whatever", presets_dir=presets_dir)


# ---------------------------------------------------------------------------
# rename_user_preset
# ---------------------------------------------------------------------------


def test_rename_changes_name_preserves_params(presets_dir: Path):
    save_user_preset(
        "stipple", Preset(name="Old Name", params={"p": 7}), presets_dir=presets_dir
    )
    rename_user_preset("stipple", "Old Name", "New Name", presets_dir=presets_dir)

    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 1
    assert loaded[0].name == "New Name"
    assert loaded[0].params == {"p": 7}


def test_rename_nonexistent_is_noop(presets_dir: Path):
    save_user_preset("stipple", Preset(name="Exists", params={}), presets_dir=presets_dir)
    rename_user_preset("stipple", "Ghost", "Whatever", presets_dir=presets_dir)
    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 1
    assert loaded[0].name == "Exists"


def test_rename_nonexistent_old_preserves_new_name(presets_dir: Path):
    """old_name absent but new_name present → new_name must be preserved (no-op)."""
    save_user_preset("stipple", Preset(name="Target", params={"v": 5}), presets_dir=presets_dir)
    # "Ghost" doesn't exist; "Target" is new_name — it must NOT be deleted.
    rename_user_preset("stipple", "Ghost", "Target", presets_dir=presets_dir)
    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 1
    assert loaded[0].name == "Target"
    assert loaded[0].params == {"v": 5}


def test_rename_same_name_is_noop(presets_dir: Path):
    """Renaming a preset to the same name must leave it unchanged."""
    save_user_preset("stipple", Preset(name="A", params={"v": 42}), presets_dir=presets_dir)
    rename_user_preset("stipple", "A", "A", presets_dir=presets_dir)
    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 1
    assert loaded[0].name == "A"
    assert loaded[0].params == {"v": 42}


def test_rename_overwrites_if_new_name_exists(presets_dir: Path):
    save_user_preset("stipple", Preset(name="A", params={"v": 1}), presets_dir=presets_dir)
    save_user_preset("stipple", Preset(name="B", params={"v": 2}), presets_dir=presets_dir)

    # Renaming A to B should overwrite B and keep only one entry.
    rename_user_preset("stipple", "A", "B", presets_dir=presets_dir)

    loaded = load_user_presets("stipple", presets_dir=presets_dir)
    assert len(loaded) == 1
    assert loaded[0].name == "B"
    assert loaded[0].params == {"v": 1}  # A's params survive under B's name


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


def test_package_exports():
    import plottter.presets as pp

    assert callable(pp.load_user_presets)
    assert callable(pp.save_user_preset)
    assert callable(pp.delete_user_preset)
    assert callable(pp.rename_user_preset)
