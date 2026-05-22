"""
Tests for parse_adjustable_vars() covering every syntax form from spec §2.

One focused test per syntax variant plus edge-case tests:
  - numeric int (no step, no desc)
  - numeric int with step
  - numeric int with step and description
  - numeric float (default has decimal)
  - numeric float (min has decimal)
  - numeric float (max has decimal)
  - numeric float (step has decimal)
  - numeric with scientific notation
  - numeric with negative min
  - choice with plain identifiers
  - choice with single-quoted strings
  - choice with description
  - type=string
  - type=string with description
  - type=path
  - type=path with description
  - let keyword (not const)
  - '//' inside string literal (edge case)
  - reserved keyword deny-list
  - malformed (min without max) — silently skipped
  - malformed (only text comment) — silently skipped
  - malformed (choice with only 1 item) — silently skipped
  - multiple vars in one code block
  - duplicate names (first wins)
  - line number tracking
  - no comment on line — skipped
  - indented declaration
"""

import pytest

from plottter.generators._adjustable_vars import AdjustableVar, parse_adjustable_vars


# ---------------------------------------------------------------------------
# Numeric — int
# ---------------------------------------------------------------------------


def test_numeric_int_basic():
    code = "const x = 5; // min=0, max=10"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "x"
    assert v.kind == "int"
    assert v.default == 5
    assert v.min == 0.0
    assert v.max == 10.0
    assert v.step is None
    assert v.description == ""


def test_numeric_int_with_step():
    code = "const sides = 6; // min=3, max=12, step=1"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "sides"
    assert v.kind == "int"
    assert v.default == 6
    assert v.min == 3.0
    assert v.max == 12.0
    assert v.step == 1.0
    assert v.description == ""


def test_numeric_int_with_step_and_desc():
    code = "const x = 5; // min=0, max=10, step=2, X coordinate"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "x"
    assert v.kind == "int"
    assert v.default == 5
    assert v.min == 0.0
    assert v.max == 10.0
    assert v.step == 2.0
    assert v.description == "X coordinate"


def test_numeric_int_desc_no_step():
    code = "const seed = 12345; // min=0, max=99999, Random seed"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "seed"
    assert v.kind == "int"
    assert v.default == 12345
    assert v.min == 0.0
    assert v.max == 99999.0
    assert v.step is None
    assert v.description == "Random seed"


# ---------------------------------------------------------------------------
# Numeric — float
# ---------------------------------------------------------------------------


def test_numeric_float_default_decimal():
    """Float when the default value contains a decimal point."""
    code = "const y = 1.5; // min=0, max=5"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "y"
    assert v.kind == "float"
    assert v.default == 1.5


def test_numeric_float_min_decimal():
    """Float when min contains a decimal point."""
    code = "const t = 0; // min=0.0, max=1"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.kind == "float"
    assert v.min == 0.0
    assert v.max == 1.0


def test_numeric_float_max_decimal():
    """Float when max contains a decimal point."""
    code = "const r = 0; // min=0, max=1.0"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.kind == "float"


def test_numeric_float_step_decimal():
    """Float when step contains a decimal point."""
    code = "const y = 1; // min=0, max=5, step=0.5, Y position"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.kind == "float"
    assert v.step == 0.5
    assert v.description == "Y position"


def test_numeric_float_full_example():
    """Full float example from spec §2."""
    code = "const y = 1.5; // min=0.1, max=5.0, step=0.1, Y position"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "y"
    assert v.kind == "float"
    assert v.default == 1.5
    assert v.min == pytest.approx(0.1)
    assert v.max == pytest.approx(5.0)
    assert v.step == pytest.approx(0.1)
    assert v.description == "Y position"


def test_numeric_scientific_notation():
    """Scientific notation in min/max/step → float."""
    code = "const n = 0; // min=1e0, max=1e3"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.kind == "float"
    assert v.min == pytest.approx(1.0)
    assert v.max == pytest.approx(1000.0)


def test_numeric_negative_min():
    """Negative min value."""
    code = "const angle = 0; // min=-180, max=180"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.kind == "int"
    assert v.min == -180.0
    assert v.max == 180.0


# ---------------------------------------------------------------------------
# Choice
# ---------------------------------------------------------------------------


def test_choice_plain_identifiers():
    """Choice with bare identifier tokens."""
    code = "const style = 'curvy'; // (curvy, jagged, smooth) Drawing style"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "style"
    assert v.kind == "choice"
    assert v.default == "curvy"
    assert v.choices == ["curvy", "jagged", "smooth"]
    assert v.description == "Drawing style"


def test_choice_single_quoted_strings():
    """Choice with single-quoted strings."""
    code = "const color = 'red'; // ('red', 'green', 'blue') Pen color"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.kind == "choice"
    assert v.choices == ["red", "green", "blue"]
    assert v.default == "red"


def test_choice_no_description():
    """Choice with no trailing description."""
    code = "const mode = 'fill'; // (fill, stroke, both)"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.kind == "choice"
    assert v.choices == ["fill", "stroke", "both"]
    assert v.description == ""


def test_choice_default_not_in_list():
    """Default value need not be in choices — parser stores it as-is."""
    code = "const x = 'other'; // (a, b, c)"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.default == "other"
    assert v.choices == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# type=string
# ---------------------------------------------------------------------------


def test_string_type_basic():
    code = "const msg = 'hello'; // type=string"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "msg"
    assert v.kind == "string"
    assert v.default == "hello"
    assert v.description == ""


def test_string_type_with_description():
    code = "const message = 'Hello'; // type=string, Prompt text"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.kind == "string"
    assert v.default == "Hello"
    assert v.description == "Prompt text"


# ---------------------------------------------------------------------------
# type=path
# ---------------------------------------------------------------------------


def test_path_type_basic():
    code = "const svg = 'M0,0'; // type=path"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "svg"
    assert v.kind == "path"
    assert v.default == "M0,0"
    assert v.description == ""


def test_path_type_with_description():
    code = "const path = 'M0,0 L10,10'; // type=path, SVG path data"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.kind == "path"
    assert v.default == "M0,0 L10,10"
    assert v.description == "SVG path data"


# ---------------------------------------------------------------------------
# `let` keyword
# ---------------------------------------------------------------------------


def test_let_keyword():
    """let declarations are valid adjustable variables (spec §2)."""
    code = "let dynamic = 3; // min=1, max=5"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "dynamic"
    assert v.kind == "int"
    assert v.default == 3
    assert v.min == 1.0
    assert v.max == 5.0


# ---------------------------------------------------------------------------
# String-literal edge case
# ---------------------------------------------------------------------------


def test_comment_inside_string_literal():
    """'//' inside a string literal must NOT be treated as the comment start."""
    code = 'const x = "// not a comment"; // min=0, max=10'
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.name == "x"
    assert v.kind == "int"
    # Default value is the string content (the JS value is a string here,
    # but metadata says numeric — parser uses the raw default anyway)
    # The key assertion: parser didn't confuse the fake "//" for the real one.
    assert v.min == 0.0
    assert v.max == 10.0


def test_single_quoted_comment_string():
    """'//' inside single-quoted string literal is not a real comment."""
    code = "const x = '// not a comment'; // type=string, Label"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    v = vars_[0]
    assert v.kind == "string"
    assert v.default == "// not a comment"
    assert v.description == "Label"


# ---------------------------------------------------------------------------
# Reserved keyword deny-list (spec §10)
# ---------------------------------------------------------------------------


def test_reserved_keyword_function():
    """'function' is a JS reserved keyword — must be skipped."""
    code = "const function = 5; // min=0, max=10"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 0


def test_reserved_keyword_return():
    code = "const return = 1; // min=0, max=5"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 0


def test_reserved_keyword_class():
    code = "let class = 'foo'; // (foo, bar)"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 0


def test_reserved_keyword_let_as_name():
    """'let' is both a keyword and a valid declaration prefix — as a name, skip."""
    code = "const let = 5; // min=0, max=10"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 0


def test_reserved_keyword_const_as_name():
    code = "let const = 5; // min=0, max=10"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 0


# ---------------------------------------------------------------------------
# Malformed declarations — silently skipped
# ---------------------------------------------------------------------------


def test_malformed_min_without_max():
    """min= without max= — silently skipped."""
    code = "const x = 5; // min=0"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 0


def test_malformed_informational_comment():
    """Plain text comment — not an adjustable variable, skipped silently."""
    code = "const x = 5; // just some notes"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 0


def test_malformed_choice_single_item():
    """Choice with only one item — malformed, skipped."""
    code = "const x = 'a'; // (a) Only one option"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 0


def test_no_comment_on_line():
    """Declaration without any trailing comment — skipped."""
    code = "const x = 5;"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 0


# ---------------------------------------------------------------------------
# Multiple vars & ordering
# ---------------------------------------------------------------------------


def test_multiple_vars():
    """Multiple adjustable variables parsed in source order."""
    code = """\
const size = 50; // min=10, max=100, step=5, Stroke size
const sides = 6; // min=3, max=12, Number of polygon sides
const style = 'curvy'; // (curvy, jagged, smooth) Drawing style
const msg = 'hello'; // type=string, Greeting text
const path = 'M0,0'; // type=path, Vector path
"""
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 5
    assert [v.name for v in vars_] == ["size", "sides", "style", "msg", "path"]
    assert [v.kind for v in vars_] == ["int", "int", "choice", "string", "path"]


# ---------------------------------------------------------------------------
# Duplicate names
# ---------------------------------------------------------------------------


def test_duplicate_names_first_wins():
    """Duplicate variable names: first declaration wins."""
    code = """\
const x = 5; // min=0, max=10
const x = 99; // min=0, max=200
"""
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    assert vars_[0].default == 5
    assert vars_[0].max == 10.0


# ---------------------------------------------------------------------------
# Line number tracking
# ---------------------------------------------------------------------------


def test_line_numbers():
    """line field is 1-indexed source line number."""
    code = """\
// some header comment
const a = 1; // min=0, max=5
const b = 2; // min=0, max=5
"""
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 2
    assert vars_[0].line == 2
    assert vars_[1].line == 3


# ---------------------------------------------------------------------------
# Indented declaration
# ---------------------------------------------------------------------------


def test_indented_declaration():
    """Parser strips leading whitespace and handles indented declarations."""
    code = "    const x = 3; // min=1, max=9"
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 1
    assert vars_[0].name == "x"
    assert vars_[0].kind == "int"


# ---------------------------------------------------------------------------
# spec §2 summary examples
# ---------------------------------------------------------------------------


def test_spec_example_all_types():
    """The six example lines from spec §2 all parse correctly."""
    code = """\
const size = 50;             // min=10, max=100, step=5, Stroke size
const sides = 6;             // min=3, max=12, Number of polygon sides
const style = 'curvy';       // (curvy, jagged, smooth) Drawing style
const seed = 12345;          // min=0, max=99999, Random seed
const message = 'Hello';     // type=string, Prompt text
const path = 'M0,0 L10,10'; // type=path, SVG path data
"""
    vars_ = parse_adjustable_vars(code)
    assert len(vars_) == 6

    by_name = {v.name: v for v in vars_}

    assert by_name["size"].kind == "int"
    assert by_name["size"].step == 5.0
    assert by_name["size"].description == "Stroke size"

    assert by_name["sides"].kind == "int"
    assert by_name["sides"].description == "Number of polygon sides"

    assert by_name["style"].kind == "choice"
    assert by_name["style"].choices == ["curvy", "jagged", "smooth"]

    assert by_name["seed"].kind == "int"
    assert by_name["seed"].max == 99999.0

    assert by_name["message"].kind == "string"
    assert by_name["message"].default == "Hello"

    assert by_name["path"].kind == "path"
    assert by_name["path"].default == "M0,0 L10,10"
