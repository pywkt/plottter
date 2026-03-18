"""User-defined preset persistence layer.

Stores custom presets per generator as JSON files in ~/.plottter/presets/.
Each file is named after the generator (e.g. ``flow_field.json``) and contains
a JSON array of ``{"name": "...", "params": {...}}`` objects.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from plottter.generators.base import Preset

logger = logging.getLogger(__name__)

_PRESETS_DIR = Path.home() / ".plottter" / "presets"


def _generator_filename(generator_name: str) -> str:
    """Return a safe filename stem for *generator_name*.

    Converts to lowercase and replaces any run of non-alphanumeric characters
    with a single underscore, e.g. ``"Flow Field"`` → ``"flow_field"``.
    """
    stem = re.sub(r"[^a-z0-9]+", "_", generator_name.lower()).strip("_")
    return stem or "generator"


def _presets_file(generator_name: str, presets_dir: Path | None = None) -> Path:
    base = presets_dir if presets_dir is not None else _PRESETS_DIR
    return base / f"{_generator_filename(generator_name)}.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_raw(path: Path) -> list[dict[str, Any]]:
    """Load and validate raw preset list from *path*.

    Returns an empty list (without raising) if the file is missing or corrupt.
    """
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            logger.warning("User presets file %s has unexpected format; ignoring.", path)
            return []
        validated: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict) and "name" in item and isinstance(item.get("params"), dict):
                validated.append({"name": str(item["name"]), "params": dict(item["params"])})
            else:
                logger.warning("Skipping malformed preset entry in %s: %r", path, item)
        return validated
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Could not read user presets from %s: %s", path, exc)
        return []


def _save_raw(path: Path, data: list[dict[str, Any]]) -> None:
    """Atomically write *data* to *path* using a temp-file rename."""
    _ensure_dir(path.parent)
    # Write to a sibling temp file then rename for atomicity.
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up the temp file if something went wrong.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_user_presets(
    generator_name: str,
    *,
    presets_dir: Path | None = None,
) -> list[Preset]:
    """Return all user presets for *generator_name*.

    Returns an empty list if no presets have been saved yet or if the file is
    corrupt (a warning is logged in that case).
    """
    path = _presets_file(generator_name, presets_dir)
    raw = _load_raw(path)
    return [Preset(name=item["name"], params=item["params"]) for item in raw]


def save_user_preset(
    generator_name: str,
    preset: Preset,
    *,
    presets_dir: Path | None = None,
) -> None:
    """Save *preset* for *generator_name*.

    If a preset with the same name already exists it is overwritten.  The
    directory ``~/.plottter/presets/`` (or *presets_dir*) is created if it
    does not yet exist.
    """
    path = _presets_file(generator_name, presets_dir)
    raw = _load_raw(path)
    # Overwrite if a preset with this name already exists.
    existing_index = next(
        (i for i, item in enumerate(raw) if item["name"] == preset.name), None
    )
    entry = {"name": preset.name, "params": preset.params}
    if existing_index is not None:
        raw[existing_index] = entry
    else:
        raw.append(entry)
    _save_raw(path, raw)


def delete_user_preset(
    generator_name: str,
    preset_name: str,
    *,
    presets_dir: Path | None = None,
) -> None:
    """Remove the preset named *preset_name* for *generator_name*.

    Does nothing if the preset or file does not exist.
    """
    path = _presets_file(generator_name, presets_dir)
    raw = _load_raw(path)
    filtered = [item for item in raw if item["name"] != preset_name]
    if len(filtered) == len(raw):
        return  # Nothing to do.
    _save_raw(path, filtered)


def rename_user_preset(
    generator_name: str,
    old_name: str,
    new_name: str,
    *,
    presets_dir: Path | None = None,
) -> None:
    """Rename a user preset from *old_name* to *new_name*.

    If *old_name* does not exist the call is a no-op.  If *new_name* already
    exists it will be overwritten.
    """
    path = _presets_file(generator_name, presets_dir)
    if old_name == new_name:
        return
    raw = _load_raw(path)
    # No-op if old_name doesn't exist.
    if not any(item["name"] == old_name for item in raw):
        return
    # Remove any existing entry with new_name (avoid duplicates).
    raw = [item for item in raw if item["name"] != new_name]
    for item in raw:
        if item["name"] == old_name:
            item["name"] = new_name
            break
    _save_raw(path, raw)
