"""Tests for PenPalette dataclass and persistence helpers."""
import json
import pytest
from pathlib import Path

from plottter.color.palette import (
    PenPalette,
    palette_from_dict,
    palette_to_dict,
    palette_slug,
    save_user_palette,
    load_user_palettes,
    palette_dir,
)


# ---------------------------------------------------------------------------
# PenPalette construction and validation
# ---------------------------------------------------------------------------

class TestPenPaletteConstruction:
    def test_basic_construction(self):
        p = PenPalette(name="Test", colors=("#ff0000", "#00ff00"))
        assert p.name == "Test"
        assert len(p.colors) == 2

    def test_hex_normalised_to_uppercase(self):
        p = PenPalette(name="Test", colors=("#ff0000", "#00ff00", "#0000ff"))
        assert p.colors == ("#FF0000", "#00FF00", "#0000FF")

    def test_already_uppercase_unchanged(self):
        p = PenPalette(name="Test", colors=("#ABCDEF",))
        assert p.colors == ("#ABCDEF",)

    def test_description_and_source_defaults(self):
        p = PenPalette(name="X", colors=("#000000",))
        assert p.description == ""
        assert p.source == ""

    def test_description_and_source_stored(self):
        p = PenPalette(name="X", colors=("#000000",), description="desc", source="http://x")
        assert p.description == "desc"
        assert p.source == "http://x"

    def test_count_property(self):
        p = PenPalette(name="X", colors=("#111111", "#222222", "#333333"))
        assert p.count == 3

    def test_frozen_immutable(self):
        p = PenPalette(name="X", colors=("#000000",))
        with pytest.raises((AttributeError, TypeError)):
            p.name = "Y"  # type: ignore[misc]

    def test_equality(self):
        p1 = PenPalette(name="X", colors=("#ff0000",))
        p2 = PenPalette(name="X", colors=("#FF0000",))
        assert p1 == p2

    def test_hashable(self):
        p = PenPalette(name="X", colors=("#000000",))
        s = {p}
        assert p in s


class TestPenPaletteValidation:
    def test_empty_colors_raises(self):
        with pytest.raises(ValueError, match="at least one colour"):
            PenPalette(name="Bad", colors=())

    def test_invalid_hex_no_hash_raises(self):
        with pytest.raises(ValueError, match="invalid hex"):
            PenPalette(name="Bad", colors=("ff0000",))

    def test_invalid_hex_short_raises(self):
        with pytest.raises(ValueError, match="invalid hex"):
            PenPalette(name="Bad", colors=("#fff",))

    def test_invalid_hex_long_raises(self):
        with pytest.raises(ValueError, match="invalid hex"):
            PenPalette(name="Bad", colors=("#ff00001",))

    def test_invalid_hex_non_hex_chars_raises(self):
        with pytest.raises(ValueError, match="invalid hex"):
            PenPalette(name="Bad", colors=("#GGGGGG",))

    def test_valid_all_zeros(self):
        p = PenPalette(name="X", colors=("#000000",))
        assert p.colors == ("#000000",)

    def test_valid_all_fs(self):
        p = PenPalette(name="X", colors=("#ffffff",))
        assert p.colors == ("#FFFFFF",)


# ---------------------------------------------------------------------------
# palette_slug
# ---------------------------------------------------------------------------

class TestPaletteSlug:
    def test_simple_name(self):
        assert palette_slug("Basic 6") == "basic-6"

    def test_special_chars(self):
        assert palette_slug("My Watercolours!") == "my-watercolours"

    def test_leading_trailing_stripped(self):
        assert palette_slug("  hello  ") == "hello"

    def test_empty_fallback(self):
        assert palette_slug("") == "palette"
        assert palette_slug("!!!") == "palette"

    def test_numbers_preserved(self):
        assert palette_slug("Set 42") == "set-42"

    def test_multiple_spaces_collapse(self):
        assert palette_slug("A  B  C") == "a-b-c"


# ---------------------------------------------------------------------------
# palette_to_dict / palette_from_dict round-trip
# ---------------------------------------------------------------------------

class TestPaletteSerialisation:
    def test_round_trip(self):
        original = PenPalette(
            name="Round Trip",
            colors=("#ff0000", "#00ff00"),
            description="test",
            source="http://example.com",
        )
        d = palette_to_dict(original)
        restored = palette_from_dict(d)
        assert restored == original

    def test_to_dict_structure(self):
        p = PenPalette(name="X", colors=("#aabbcc",))
        d = palette_to_dict(p)
        assert d["name"] == "X"
        assert d["colors"] == ["#AABBCC"]
        assert d["description"] == ""
        assert d["source"] == ""

    def test_from_dict_missing_optional_fields(self):
        d = {"name": "Min", "colors": ["#123456"]}
        p = palette_from_dict(d)
        assert p.name == "Min"
        assert p.colors == ("#123456",)
        assert p.description == ""
        assert p.source == ""

    def test_from_dict_hex_normalised(self):
        d = {"name": "X", "colors": ["#abcdef"]}
        p = palette_from_dict(d)
        assert p.colors == ("#ABCDEF",)

    def test_json_round_trip(self):
        original = PenPalette(name="JSON Test", colors=("#010203", "#040506"))
        json_str = json.dumps(palette_to_dict(original))
        restored = palette_from_dict(json.loads(json_str))
        assert restored == original


# ---------------------------------------------------------------------------
# save_user_palette / load_user_palettes with monkeypatched palette_dir
# ---------------------------------------------------------------------------

class TestSaveLoadPalettes:
    def test_save_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plottter.color.palette.palette_dir", lambda: tmp_path
        )
        p = PenPalette(name="My Palette", colors=("#ff0000", "#0000ff"))
        fp = save_user_palette(p)
        assert fp.exists()
        assert fp.suffix == ".json"

    def test_saved_content_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plottter.color.palette.palette_dir", lambda: tmp_path
        )
        p = PenPalette(name="My Palette", colors=("#ff0000",))
        fp = save_user_palette(p)
        data = json.loads(fp.read_text())
        assert data["name"] == "My Palette"
        assert data["colors"] == ["#FF0000"]

    def test_save_load_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plottter.color.palette.palette_dir", lambda: tmp_path
        )
        original = PenPalette(
            name="My Watercolours",
            colors=("#1e3a5f", "#c13b4f", "#e2b23a"),
            description="Mungyo set",
            source="",
        )
        save_user_palette(original)
        loaded = load_user_palettes()
        assert len(loaded) == 1
        assert loaded[0] == original

    def test_load_multiple_palettes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plottter.color.palette.palette_dir", lambda: tmp_path
        )
        p1 = PenPalette(name="Alpha", colors=("#aaaaaa",))
        p2 = PenPalette(name="Beta", colors=("#bbbbbb",))
        save_user_palette(p1)
        save_user_palette(p2)
        loaded = load_user_palettes()
        assert len(loaded) == 2
        names = {p.name for p in loaded}
        assert names == {"Alpha", "Beta"}

    def test_load_skips_malformed_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plottter.color.palette.palette_dir", lambda: tmp_path
        )
        # Write a bad file
        bad = tmp_path / "bad.json"
        bad.write_text("this is not json {{")
        # Write a good file
        good = PenPalette(name="Good", colors=("#123456",))
        save_user_palette(good)

        loaded = load_user_palettes()
        assert len(loaded) == 1
        assert loaded[0].name == "Good"

    def test_load_skips_file_with_invalid_hex(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plottter.color.palette.palette_dir", lambda: tmp_path
        )
        invalid = tmp_path / "invalid.json"
        invalid.write_text(
            json.dumps({"name": "Bad", "colors": ["not-a-hex"]})
        )
        loaded = load_user_palettes()
        assert loaded == []

    def test_load_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plottter.color.palette.palette_dir", lambda: tmp_path
        )
        loaded = load_user_palettes()
        assert loaded == []

    def test_slug_used_as_filename(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plottter.color.palette.palette_dir", lambda: tmp_path
        )
        p = PenPalette(name="My Set!", colors=("#aabbcc",))
        fp = save_user_palette(p)
        assert fp.name == "my-set.json"
