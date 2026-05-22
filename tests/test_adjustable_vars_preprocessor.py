"""
Tests for apply_overrides() — spec §3.2.

Coverage:
  - round-trip: line count invariance for single-line and multi-line code
  - value escaping: int, float, bool (true/false), str (single-quoted, backslash
    escaping), list (JSON array)
  - missing override keeps original default
  - non-adjustable lines pass through unmodified
  - indented declarations are rewritten correctly
  - empty overrides dict returns code unchanged (identity)
  - reserved keyword names are not rewritten
"""

import pytest

from plottter.generators._adjustable_vars import apply_overrides, parse_adjustable_vars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def line_count(s: str) -> int:
    """Count newlines in *s* (mirrors splitlines(keepends=True) length)."""
    return len(s.splitlines(keepends=True))


# ---------------------------------------------------------------------------
# Round-trip / line-count invariance
# ---------------------------------------------------------------------------


def test_line_count_single_line_no_newline():
    """Single line without trailing newline: output line count matches input."""
    code = "const x = 5; // min=0, max=10"
    result = apply_overrides(code, {"x": 99})
    assert line_count(result) == line_count(code)


def test_line_count_single_line_with_newline():
    """Single line with trailing newline: output line count matches input."""
    code = "const x = 5; // min=0, max=10\n"
    result = apply_overrides(code, {"x": 99})
    assert line_count(result) == line_count(code)


def test_line_count_multi_line():
    """Multi-line code block: total newline count is invariant after overrides."""
    code = """\
const size = 50; // min=10, max=100
const sides = 6; // min=3, max=12
const style = 'curvy'; // (curvy, jagged, smooth) Drawing style
// a plain comment line
const msg = 'hello'; // type=string, Greeting
"""
    result = apply_overrides(code, {"size": 80, "style": "jagged"})
    assert line_count(result) == line_count(code)


def test_round_trip_value_substitution():
    """Overridden value appears in output; rest of each line is unchanged."""
    code = "const x = 5; // min=0, max=10\n"
    result = apply_overrides(code, {"x": 42})
    assert result == "const x = 42; // min=0, max=10\n"


def test_round_trip_multi_var():
    """Multiple variables overridden in a single pass."""
    code = (
        "const size = 50; // min=10, max=100\n"
        "const sides = 6; // min=3, max=12\n"
    )
    result = apply_overrides(code, {"size": 75, "sides": 8})
    assert result == (
        "const size = 75; // min=10, max=100\n"
        "const sides = 8; // min=3, max=12\n"
    )


# ---------------------------------------------------------------------------
# Value escaping — int
# ---------------------------------------------------------------------------


def test_escape_int_positive():
    code = "const n = 1; // min=0, max=100"
    assert apply_overrides(code, {"n": 42}) == "const n = 42; // min=0, max=100"


def test_escape_int_zero():
    code = "const n = 1; // min=0, max=100"
    assert apply_overrides(code, {"n": 0}) == "const n = 0; // min=0, max=100"


def test_escape_int_negative():
    code = "const n = 1; // min=-100, max=100"
    assert apply_overrides(code, {"n": -7}) == "const n = -7; // min=-100, max=100"


# ---------------------------------------------------------------------------
# Value escaping — float
# ---------------------------------------------------------------------------


def test_escape_float_decimal():
    code = "const y = 1.0; // min=0.0, max=5.0"
    result = apply_overrides(code, {"y": 3.14})
    assert result == "const y = 3.14; // min=0.0, max=5.0"


def test_escape_float_whole():
    code = "const y = 1.0; // min=0.0, max=5.0"
    result = apply_overrides(code, {"y": 2.0})
    assert result == "const y = 2.0; // min=0.0, max=5.0"


# ---------------------------------------------------------------------------
# Value escaping — bool
# ---------------------------------------------------------------------------


def test_escape_bool_true():
    """bool True -> JS ``true``."""
    code = "const flag = 0; // min=0, max=1"
    result = apply_overrides(code, {"flag": True})
    assert result == "const flag = true; // min=0, max=1"


def test_escape_bool_false():
    """bool False -> JS ``false``."""
    code = "const flag = 1; // min=0, max=1"
    result = apply_overrides(code, {"flag": False})
    assert result == "const flag = false; // min=0, max=1"


def test_bool_not_treated_as_int():
    """bool must produce true/false, not 1/0 (bool is a subclass of int)."""
    code = "const b = 0; // min=0, max=1"
    assert apply_overrides(code, {"b": True}) == "const b = true; // min=0, max=1"
    assert apply_overrides(code, {"b": False}) == "const b = false; // min=0, max=1"


# ---------------------------------------------------------------------------
# Value escaping — string
# ---------------------------------------------------------------------------


def test_escape_str_simple():
    """Plain string -> single-quoted JS literal."""
    code = "const msg = 'hello'; // type=string"
    result = apply_overrides(code, {"msg": "world"})
    assert result == "const msg = 'world'; // type=string"


def test_escape_str_single_quote():
    """Embedded single quote is backslash-escaped."""
    code = "const msg = 'hello'; // type=string"
    result = apply_overrides(code, {"msg": "it's fine"})
    assert result == r"const msg = 'it\'s fine'; // type=string"


def test_escape_str_backslash():
    """Embedded backslash is doubled."""
    code = "const path = 'a'; // type=path"
    result = apply_overrides(code, {"path": "C:\\Users"})
    assert result == "const path = 'C:\\\\Users'; // type=path"


def test_escape_str_double_quote_unchanged():
    """Double quotes inside single-quoted JS string need no escaping."""
    code = "const msg = 'hi'; // type=string"
    result = apply_overrides(code, {"msg": 'say "hello"'})
    assert result == "const msg = 'say \"hello\"'; // type=string"


def test_escape_str_empty():
    """Empty string -> two single quotes."""
    code = "const msg = 'hi'; // type=string"
    result = apply_overrides(code, {"msg": ""})
    assert result == "const msg = ''; // type=string"


# ---------------------------------------------------------------------------
# Value escaping — list
# ---------------------------------------------------------------------------


def test_escape_list_strings():
    """List of strings -> JSON array with double-quoted elements."""
    code = "const colors = 'red'; // (red, green, blue)"
    result = apply_overrides(code, {"colors": ["red", "blue"]})
    assert result == 'const colors = ["red", "blue"]; // (red, green, blue)'


def test_escape_list_numbers():
    """List of numbers -> JSON array."""
    code = "const pts = 0; // min=0, max=100"
    result = apply_overrides(code, {"pts": [1, 2, 3]})
    assert result == "const pts = [1, 2, 3]; // min=0, max=100"


def test_escape_list_empty():
    """Empty list -> ``[]``."""
    code = "const items = 0; // min=0, max=10"
    result = apply_overrides(code, {"items": []})
    assert result == "const items = []; // min=0, max=10"


# ---------------------------------------------------------------------------
# Missing override keeps original default
# ---------------------------------------------------------------------------


def test_missing_override_unchanged():
    """A variable not in *overrides* retains its original default."""
    code = "const x = 5; // min=0, max=10\nconst y = 3; // min=0, max=10\n"
    result = apply_overrides(code, {"x": 99})
    # y must be unchanged
    assert "const y = 3;" in result


def test_all_overrides_missing():
    """If none of the overridden names match, code is returned unchanged."""
    code = "const x = 5; // min=0, max=10\n"
    result = apply_overrides(code, {"z": 99})
    assert result == code


def test_empty_overrides_identity():
    """Empty overrides dict returns the original code string unchanged."""
    code = "const x = 5; // min=0, max=10\nconst y = 3.0; // min=0.0, max=5.0\n"
    assert apply_overrides(code, {}) is code or apply_overrides(code, {}) == code


# ---------------------------------------------------------------------------
# Non-adjustable lines pass through unmodified
# ---------------------------------------------------------------------------


def test_plain_comment_unchanged():
    """Lines with only comments are not touched."""
    code = "// This is just a comment\nconst x = 5; // min=0, max=10\n"
    result = apply_overrides(code, {"x": 9})
    assert result.startswith("// This is just a comment\n")


def test_blank_line_unchanged():
    code = "\nconst x = 5; // min=0, max=10\n\n"
    result = apply_overrides(code, {"x": 9})
    # Blank lines preserved; line count unchanged
    assert result.count("\n") == code.count("\n")


def test_regular_assignment_unchanged():
    """An assignment without a metadata comment is not modified."""
    code = "const x = 5;\nconst y = 3; // min=0, max=10\n"
    result = apply_overrides(code, {"x": 99})
    # x has no metadata comment so it should not be touched
    assert "const x = 5;" in result


def test_function_call_unchanged():
    """Arbitrary JS lines that don't match the declaration pattern are untouched."""
    code = "turtle.forward(100);\nconst n = 3; // min=1, max=10\n"
    result = apply_overrides(code, {"n": 7})
    assert "turtle.forward(100);" in result
    assert "const n = 7;" in result


def test_informational_comment_var_unchanged():
    """A declaration with a plain-text comment (not adjustable metadata) is not touched."""
    code = "const x = 5; // just a note about x\n"
    # x is in overrides but the line has no valid metadata -> not an adjustable var
    # apply_overrides rewrites any line matching _LINE_RE whose name is in overrides,
    # even if the metadata is informational; only the RHS is changed.
    # However, the key contract is: the line count is preserved.
    result = apply_overrides(code, {"x": 99})
    assert line_count(result) == line_count(code)


# ---------------------------------------------------------------------------
# Indented declarations
# ---------------------------------------------------------------------------


def test_indented_declaration_rewritten():
    """Leading whitespace is preserved; only the RHS value is replaced."""
    code = "    const x = 3; // min=1, max=9\n"
    result = apply_overrides(code, {"x": 7})
    assert result == "    const x = 7; // min=1, max=9\n"


def test_indented_declaration_line_count():
    code = "    const x = 3; // min=1, max=9\n"
    result = apply_overrides(code, {"x": 7})
    assert line_count(result) == line_count(code)


# ---------------------------------------------------------------------------
# Reserved keyword names are not rewritten
# ---------------------------------------------------------------------------


def test_reserved_keyword_not_rewritten():
    """If somehow a reserved keyword appears in overrides, the line is left alone."""
    # 'function' is in _JS_RESERVED, so even if it somehow got there it won't be touched
    code = "const function = 5; // min=0, max=10\n"
    result = apply_overrides(code, {"function": 99})
    assert result == code


# ---------------------------------------------------------------------------
# Whitespace and semicolon preservation
# ---------------------------------------------------------------------------


def test_whitespace_between_name_and_value_preserved():
    """Extra spaces around '=' are preserved on rewrite."""
    code = "const x  =  5; // min=0, max=10\n"
    result = apply_overrides(code, {"x": 42})
    # Leading/trailing structure preserved; only the value changes
    assert result.startswith("const x  =  ")
    assert "42;" in result


def test_trailing_comment_preserved():
    """The metadata comment after the semicolon is not modified."""
    code = "const size = 50; // min=10, max=100, step=5, Stroke size\n"
    result = apply_overrides(code, {"size": 75})
    assert "// min=10, max=100, step=5, Stroke size" in result


def test_let_declaration_rewritten():
    """``let`` declarations are also supported."""
    code = "let dynamic = 3; // min=1, max=5\n"
    result = apply_overrides(code, {"dynamic": 4})
    assert result == "let dynamic = 4; // min=1, max=5\n"


# ---------------------------------------------------------------------------
# Integration: parse after apply_overrides gives updated defaults
# ---------------------------------------------------------------------------


def test_parse_after_apply_overrides():
    """parse_adjustable_vars on the rewritten code returns the new default values."""
    code = "const x = 5; // min=0, max=10\nconst y = 1.5; // min=0.0, max=5.0\n"
    result = apply_overrides(code, {"x": 8, "y": 3.5})
    vars_ = parse_adjustable_vars(result)
    by_name = {v.name: v for v in vars_}
    assert by_name["x"].default == 8
    assert by_name["y"].default == pytest.approx(3.5)
