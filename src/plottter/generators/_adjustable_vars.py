"""
Parse TurtleToy-style adjustable-variable declarations from JavaScript code.

Grammar (from spec §2):
  adjustable   ::= ("const" | "let") <name> "=" <default> ";" <space>* "//" <metadata>
  <metadata>   ::= <numeric-meta> | <choice-meta> | <string-meta> | <path-meta>
  <numeric-meta>  ::= "min=" <number> "," "max=" <number> ("," "step=" <number>)? ("," <desc>)?
  <choice-meta>   ::= "(" <choice> ("," <choice>)+ ")" <space>* <desc>?
  <string-meta>   ::= "type=string" ("," <desc>)?
  <path-meta>     ::= "type=path" ("," <desc>)?
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reserved JS keywords (spec §10) — these names cannot be adjustable vars.
# ---------------------------------------------------------------------------
_JS_RESERVED = frozenset(
    """
    break case catch class const continue debugger default delete
    do else enum export extends false finally for function if
    import in instanceof let new null return super switch this
    throw true try typeof var void while with yield async await
    """.split()
)

# Valid JS identifier pattern (compiled separately for extra validation)
_IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")

# Number pattern (signed, decimal, or scientific notation)
_NUMBER_PAT = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"

# ---------------------------------------------------------------------------
# Core line regex.
# Group 1: variable name
# Group 2: raw default value (everything before the semicolon, handles string literals)
# Group 3: metadata (everything after the real "//")
# ---------------------------------------------------------------------------
_LINE_RE = re.compile(
    r"""
    ^                                           # start of (trimmed) line
    (?:const|let)\s+                            # keyword
    ([A-Za-z_$][A-Za-z0-9_$]*)                 # name — group 1
    \s*=\s*
    (                                           # raw default — group 2
        (?:'[^'\\]*(?:\\.[^'\\]*)*'             #   single-quoted string
           |"[^"\\]*(?:\\.[^"\\]*)*"            #   double-quoted string
           |[^;'"]                              #   any non-semicolon, non-quote char
        )*
    )
    ;
    [^\S\n]*                                    # optional horizontal space
    //                                          # start of comment
    (.*)                                        # metadata — group 3
    $
    """,
    re.VERBOSE,
)

# Numeric metadata: min=N, max=N [, step=N] [, desc]
_NUMERIC_RE = re.compile(
    r"^\s*min\s*=\s*(?P<min>" + _NUMBER_PAT + r")"
    r"\s*,\s*max\s*=\s*(?P<max>" + _NUMBER_PAT + r")"
    r"(?:\s*,\s*step\s*=\s*(?P<step>" + _NUMBER_PAT + r"))?"
    r"(?:\s*,\s*(?P<desc>.+?))?"
    r"\s*$",
)

# Choice metadata: (choice1, choice2, ...) optional desc
_CHOICE_RE = re.compile(
    r"^\s*\((?P<choices>[^)]+)\)\s*(?P<desc>.*)\s*$",
)

# String-type metadata: type=string [, desc]
_STRING_RE = re.compile(
    r"^\s*type\s*=\s*string(?:\s*,\s*(?P<desc>.+?))?\s*$",
)

# Path-type metadata: type=path [, desc]
_PATH_RE = re.compile(
    r"^\s*type\s*=\s*path(?:\s*,\s*(?P<desc>.+?))?\s*$",
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AdjustableVar:
    name: str
    kind: Literal["int", "float", "choice", "string", "path"]
    default: Any
    min: float | None = None          # numeric only
    max: float | None = None          # numeric only
    step: float | None = None         # numeric only
    choices: list[str] | None = None  # choice only
    description: str = ""
    line: int = 0                     # 1-indexed source line number


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_float_literal(s: str) -> bool:
    """Return True if *s* looks like a float (has decimal point or exponent)."""
    return "." in s or "e" in s.lower()


def _parse_default(raw: str) -> Any:
    """Convert a raw JS default-value string to a Python value."""
    raw = raw.strip()
    # Single-quoted string
    if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
        return raw[1:-1]
    # Double-quoted string
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    # Numeric
    try:
        if _is_float_literal(raw):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_choice_item(s: str) -> str:
    """Strip optional surrounding quotes from a single choice item."""
    s = s.strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1]
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def _find_real_comment(line: str) -> int | None:
    """Return the index of the first '//' NOT inside a string literal.

    Returns None if no such '//' exists on the line.
    """
    in_single = False
    in_double = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and (in_single or in_double):
            i += 2  # skip escape sequence
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "/" and not in_single and not in_double:
            if i + 1 < len(line) and line[i + 1] == "/":
                return i
        i += 1
    return None


# ---------------------------------------------------------------------------
# Metadata parser
# ---------------------------------------------------------------------------


def _parse_metadata(
    name: str,
    default: Any,
    metadata: str,
    lineno: int,
) -> AdjustableVar | None:
    """Map metadata string to an AdjustableVar. Returns None if not adjustable."""

    # 1. Choice — starts with "("
    m = _CHOICE_RE.match(metadata)
    if m:
        raw_choices = m.group("choices")
        choices = [_parse_choice_item(c) for c in raw_choices.split(",")]
        choices = [c for c in choices if c]
        if len(choices) < 2:
            return None  # malformed — need at least 2 options
        desc = (m.group("desc") or "").strip()
        default_str = default if isinstance(default, str) else str(default)
        return AdjustableVar(
            name=name,
            kind="choice",
            default=default_str,
            choices=choices,
            description=desc,
            line=lineno,
        )

    # 2. type=string
    m = _STRING_RE.match(metadata)
    if m:
        desc = (m.group("desc") or "").strip()
        default_str = default if isinstance(default, str) else str(default)
        return AdjustableVar(
            name=name,
            kind="string",
            default=default_str,
            description=desc,
            line=lineno,
        )

    # 3. type=path
    m = _PATH_RE.match(metadata)
    if m:
        desc = (m.group("desc") or "").strip()
        default_str = default if isinstance(default, str) else str(default)
        return AdjustableVar(
            name=name,
            kind="path",
            default=default_str,
            description=desc,
            line=lineno,
        )

    # 4. Numeric — min= and max= required
    m = _NUMERIC_RE.match(metadata)
    if m:
        min_raw = m.group("min")
        max_raw = m.group("max")
        step_raw = m.group("step")  # may be None
        desc = (m.group("desc") or "").strip()

        try:
            min_val = float(min_raw)
            max_val = float(max_raw)
            step_val = float(step_raw) if step_raw is not None else None
        except (ValueError, TypeError):
            return None  # malformed number

        # Float if any of default/min/max/step contain '.' or 'e' (spec §2 rule 4)
        is_float = (
            _is_float_literal(min_raw)
            or _is_float_literal(max_raw)
            or (step_raw is not None and _is_float_literal(step_raw))
            or isinstance(default, float)
        )

        kind: Literal["int", "float"] = "float" if is_float else "int"

        if kind == "int":
            try:
                typed_default: Any = int(default) if isinstance(default, (int, float)) else int(float(str(default)))
            except (ValueError, TypeError):
                typed_default = default
        else:
            try:
                typed_default = float(default) if isinstance(default, (int, float)) else float(str(default))
            except (ValueError, TypeError):
                typed_default = default

        return AdjustableVar(
            name=name,
            kind=kind,
            default=typed_default,
            min=min_val,
            max=max_val,
            step=step_val,
            description=desc,
            line=lineno,
        )

    # Nothing matched — informational comment, not an adjustable variable
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_adjustable_vars(code: str) -> list[AdjustableVar]:
    """Scan *code* for adjustable-variable declarations. Returns them in source order.

    Malformed declarations (e.g. min without max, reserved keyword names) are
    silently skipped — does not raise. Duplicate names: first declaration wins.
    """
    results: list[AdjustableVar] = []
    seen: set[str] = set()

    for lineno, raw_line in enumerate(code.splitlines(), start=1):
        line = raw_line.strip()

        # Quick pre-checks to avoid running the regex on most lines
        if not (line.startswith("const ") or line.startswith("let ")):
            continue
        # Must have a real "//" comment (not inside a string literal)
        if _find_real_comment(raw_line) is None:
            continue

        m = _LINE_RE.match(line)
        if m is None:
            continue

        name = m.group(1)
        raw_default = m.group(2).strip()
        metadata = m.group(3).strip()

        # Deny-list check (spec §10)
        if name in _JS_RESERVED:
            continue

        # Duplicate check — first declaration wins (spec §2)
        if name in seen:
            log.debug(
                "parse_adjustable_vars: duplicate name %r on line %d, skipping",
                name,
                lineno,
            )
            continue

        default = _parse_default(raw_default)

        var = _parse_metadata(name, default, metadata, lineno)
        if var is None:
            continue  # informational comment — not adjustable

        seen.add(name)
        results.append(var)

    return results
