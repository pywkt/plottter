"""ProjectController — bridges the Project model to GUI panels via Qt signals."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QUndoStack

from plottter.models import Canvas, Layer, Project

if TYPE_CHECKING:
    from plottter.models.path import Polyline


class ProjectController(QObject):
    """Wraps a Project model and exposes Qt signals for GUI panels to observe."""

    # Emitted with layer_id when a layer is added
    layer_added = pyqtSignal(str)
    # Emitted with layer_id when a layer is removed
    layer_removed = pyqtSignal(str)
    # Emitted with layer_id when a layer's properties (name, color, visibility, lock) change
    layer_changed = pyqtSignal(str)
    # Emitted when the layer order changes
    layers_reordered = pyqtSignal()
    # Emitted when the canvas size/margins change
    canvas_changed = pyqtSignal()
    # Emitted when a new project is loaded or created
    project_loaded = pyqtSignal()
    # Emitted with layer_id when paths in a layer change
    paths_changed = pyqtSignal(str)
    # Emitted with True/False when the modified flag changes
    modified_changed = pyqtSignal(bool)
    # Emitted with layer_id when the active (selected) layer in the layer panel changes
    active_layer_changed = pyqtSignal(str)
    # Emitted with layer_id when a layer's generator_info changes (e.g. after move sync)
    generator_info_changed = pyqtSignal(str)
    # Emitted when the project's saved masks change (save, delete, rename)
    masks_changed = pyqtSignal()

    def __init__(self, project: Project, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._project = project
        self._modified = False
        self._undo_stack = QUndoStack(self)
        self._undo_stack.cleanChanged.connect(self._on_clean_changed)
        self._active_layer_id: str | None = None

    def _on_clean_changed(self, clean: bool) -> None:
        self._set_modified(not clean)

    # ------------------------------------------------------------------
    # Project access
    # ------------------------------------------------------------------

    @property
    def current_project(self) -> Project:
        return self._project

    @property
    def modified(self) -> bool:
        return self._modified

    @property
    def undo_stack(self) -> QUndoStack:
        """The QUndoStack for wiring Undo/Redo actions in the main window."""
        return self._undo_stack

    def _set_modified(self, value: bool) -> None:
        if self._modified != value:
            self._modified = value
            self.modified_changed.emit(value)

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------

    def new_project(self, project: Project) -> None:
        """Replace the current project with a new one."""
        self._project = project
        self._undo_stack.clear()  # Resets stack; triggers cleanChanged(True) if was dirty
        self._set_modified(False)
        self.project_loaded.emit()

    def load_project(self, project: Project) -> None:
        """Load a project from file."""
        self._project = project
        self._undo_stack.clear()
        self._set_modified(False)
        self.project_loaded.emit()

    def mark_saved(self) -> None:
        """Mark the project as saved (marks undo stack clean position)."""
        self._undo_stack.setClean()

    # ------------------------------------------------------------------
    # Raw (no-undo) internal methods — called by QUndoCommand subclasses
    # ------------------------------------------------------------------

    def _raw_add_layer(self, layer: Layer) -> None:
        """Append a layer without pushing an undo command."""
        self._project.add_layer(layer)
        self._set_modified(True)
        self.layer_added.emit(layer.id)

    def _raw_insert_layer(self, layer: Layer, index: int) -> None:
        """Insert a layer at a specific index without pushing an undo command."""
        self._project.layers.insert(index, layer)
        self._set_modified(True)
        self.layer_added.emit(layer.id)

    def _raw_remove_layer(self, layer_id: str) -> None:
        """Remove a layer without pushing an undo command."""
        self._project.remove_layer(layer_id)
        self._set_modified(True)
        self.layer_removed.emit(layer_id)

    def _raw_reorder_layer(self, layer_id: str, new_index: int) -> None:
        """Reorder a layer without pushing an undo command."""
        self._project.reorder_layer(layer_id, new_index)
        self._set_modified(True)
        self.layers_reordered.emit()

    def _raw_set_layer_paths(self, layer_id: str, paths: list[Polyline]) -> None:
        """Replace layer paths without pushing an undo command."""
        layer = self._project.get_layer(layer_id)
        if layer is not None:
            layer.paths = paths
            self._set_modified(True)
            self.paths_changed.emit(layer_id)

    def _raw_set_layer_property(self, layer_id: str, prop_name: str, value: object) -> None:
        """Set a named layer property without pushing an undo command."""
        layer = self._project.get_layer(layer_id)
        if layer is not None:
            setattr(layer, prop_name, value)
            self._set_modified(True)
            self.layer_changed.emit(layer_id)

    def _raw_set_layer_generator_info(self, layer_id: str, info: dict | None) -> None:
        """Set generator_info without pushing an undo command; emits generator_info_changed."""
        layer = self._project.get_layer(layer_id)
        if layer is not None:
            layer.generator_info = info
            self._set_modified(True)
            self.generator_info_changed.emit(layer_id)

    def _raw_set_canvas(self, canvas: Canvas) -> None:
        """Replace the canvas without pushing an undo command."""
        self._project.canvas = canvas
        self._set_modified(True)
        self.canvas_changed.emit()

    def _raw_save_mask(self, name: str, mask_array: np.ndarray) -> None:
        """Save a mask without pushing an undo command."""
        self._project.save_mask(name, mask_array)
        self._set_modified(True)
        self.masks_changed.emit()

    def _raw_restore_mask_bytes(self, name: str, png_bytes: bytes) -> None:
        """Restore raw PNG bytes for a mask directly (used by undo to avoid re-encoding)."""
        self._project.masks[name] = png_bytes
        self._set_modified(True)
        self.masks_changed.emit()

    def _raw_delete_mask(self, name: str) -> None:
        """Delete a mask without pushing an undo command."""
        self._project.delete_mask(name)
        self._set_modified(True)
        self.masks_changed.emit()

    def _raw_rename_mask(self, old: str, new: str) -> None:
        """Rename a mask without pushing an undo command."""
        self._project.rename_mask(old, new)
        self._set_modified(True)
        self.masks_changed.emit()

    # ------------------------------------------------------------------
    # Layer management (undo-aware public API)
    # ------------------------------------------------------------------

    def add_layer(self, layer: Layer | None = None) -> Layer:
        """Add a new layer to the project (undoable)."""
        if layer is None:
            layer = Layer(name=f"Layer {len(self._project.layers) + 1}")
        from plottter.gui.commands import AddLayerCommand
        cmd = AddLayerCommand(self, layer)
        self._undo_stack.push(cmd)  # push() calls cmd.redo() = _raw_add_layer()
        return layer

    def remove_layer(self, layer_id: str) -> None:
        """Remove a layer (undoable — layer snapshot taken before removal)."""
        from plottter.gui.commands import RemoveLayerCommand
        cmd = RemoveLayerCommand(self, layer_id)
        self._undo_stack.push(cmd)

    def reorder_layer(self, layer_id: str, new_index: int) -> None:
        """Move a layer to a new index (undoable)."""
        old_index = next(
            (i for i, l in enumerate(self._project.layers) if l.id == layer_id),
            None,
        )
        if old_index is None:
            return
        from plottter.gui.commands import ReorderLayerCommand
        cmd = ReorderLayerCommand(self, layer_id, new_index, old_index)
        self._undo_stack.push(cmd)

    def get_layer(self, layer_id: str) -> Layer | None:
        """Return the layer with the given id."""
        return self._project.get_layer(layer_id)

    def duplicate_layer(self, layer_id: str) -> Layer:
        """Duplicate a layer and add it to the project (undoable)."""
        new_layer = self._project.duplicate_layer(layer_id)
        from plottter.gui.commands import AddLayerCommand
        cmd = AddLayerCommand(self, new_layer)
        self._undo_stack.push(cmd)
        return new_layer

    def merge_layers(self, layer_ids: list[str]) -> Layer:
        """Merge multiple layers into one, remove the originals (undoable)."""
        merged = self._project.merge_layers(layer_ids)
        from plottter.gui.commands import MergeLayersCommand
        cmd = MergeLayersCommand(self, layer_ids, merged)
        self._undo_stack.push(cmd)
        return merged

    def set_layer_name(self, layer_id: str, name: str) -> None:
        layer = self._project.get_layer(layer_id)
        if layer is not None:
            from plottter.gui.commands import SetLayerPropertyCommand
            cmd = SetLayerPropertyCommand(
                self, layer_id, "name", name, layer.name, "Rename Layer"
            )
            self._undo_stack.push(cmd)

    def set_layer_color(self, layer_id: str, color: str) -> None:
        layer = self._project.get_layer(layer_id)
        if layer is not None:
            from plottter.gui.commands import SetLayerPropertyCommand
            cmd = SetLayerPropertyCommand(
                self, layer_id, "color", color, layer.color, "Change Layer Color"
            )
            self._undo_stack.push(cmd)

    def set_layer_visible(self, layer_id: str, visible: bool) -> None:
        layer = self._project.get_layer(layer_id)
        if layer is not None:
            from plottter.gui.commands import SetLayerPropertyCommand
            cmd = SetLayerPropertyCommand(
                self, layer_id, "visible", visible, layer.visible, "Toggle Visibility"
            )
            self._undo_stack.push(cmd)

    def set_layer_locked(self, layer_id: str, locked: bool) -> None:
        layer = self._project.get_layer(layer_id)
        if layer is not None:
            from plottter.gui.commands import SetLayerPropertyCommand
            cmd = SetLayerPropertyCommand(
                self, layer_id, "locked", locked, layer.locked, "Toggle Lock"
            )
            self._undo_stack.push(cmd)

    def set_layer_opacity(self, layer_id: str, opacity: float) -> None:
        """Set a layer's opacity (0.0–1.0, undoable)."""
        layer = self._project.get_layer(layer_id)
        if layer is not None:
            opacity = max(0.0, min(1.0, opacity))
            from plottter.gui.commands import SetLayerPropertyCommand
            cmd = SetLayerPropertyCommand(
                self, layer_id, "opacity", opacity, layer.opacity, "Set Opacity"
            )
            self._undo_stack.push(cmd)

    def set_layer_paths(
        self,
        layer_id: str,
        paths: list[Polyline],
        description: str = "Set Paths",
    ) -> None:
        """Replace a layer's paths and emit paths_changed (undoable).

        The description appears in Edit > Undo / Redo menu text.
        Callers should pass descriptive strings like "Generate", "Optimize Paths",
        "Simplify Paths", etc.
        """
        layer = self._project.get_layer(layer_id)
        if layer is not None:
            # Shallow copy list; Point tuples are immutable so this is safe
            old_paths = [list(p) for p in layer.paths]
            from plottter.gui.commands import SetLayerPathsCommand
            cmd = SetLayerPathsCommand(self, layer_id, paths, old_paths, description)
            self._undo_stack.push(cmd)

    # ------------------------------------------------------------------
    # Mask management (undo-aware public API)
    # ------------------------------------------------------------------

    def save_mask(self, name: str, mask_array: np.ndarray) -> None:
        """Save a named mask (undoable)."""
        from plottter.gui.commands import SaveMaskCommand
        cmd = SaveMaskCommand(self, name, mask_array)
        self._undo_stack.push(cmd)

    def load_mask(self, name: str) -> np.ndarray:
        """Load a named mask as a float32 [0,1] array."""
        return self._project.load_mask(name)

    def delete_mask(self, name: str) -> None:
        """Delete a named mask (undoable)."""
        from plottter.gui.commands import DeleteMaskCommand
        cmd = DeleteMaskCommand(self, name)
        self._undo_stack.push(cmd)

    def rename_mask(self, old: str, new: str) -> None:
        """Rename a mask (undoable)."""
        from plottter.gui.commands import RenameMaskCommand
        cmd = RenameMaskCommand(self, old, new)
        self._undo_stack.push(cmd)

    def mask_names(self) -> list[str]:
        """Return a sorted list of saved mask names."""
        return sorted(self._project.masks.keys())

    # ------------------------------------------------------------------
    # Canvas management (undo-aware)
    # ------------------------------------------------------------------

    def set_canvas(self, canvas: Canvas) -> None:
        """Replace the canvas and emit canvas_changed (undoable)."""
        old_canvas = copy.copy(self._project.canvas)
        from plottter.gui.commands import SetCanvasCommand
        cmd = SetCanvasCommand(self, canvas, old_canvas)
        self._undo_stack.push(cmd)

    # ------------------------------------------------------------------
    # Registration marks
    # ------------------------------------------------------------------

    def set_registration_marks(self, enabled: bool, style: str | None = None) -> None:
        self._project.registration_marks = enabled
        if style is not None:
            self._project.reg_mark_style = style
        self._set_modified(True)
        self.canvas_changed.emit()

    # ------------------------------------------------------------------
    # Active layer tracking
    # ------------------------------------------------------------------

    @property
    def active_layer_id(self) -> str | None:
        """The id of the currently active (selected) layer in the layer panel."""
        return self._active_layer_id

    def set_active_layer(self, layer_id: str) -> None:
        """Set the active layer and emit active_layer_changed if it changed."""
        if self._active_layer_id != layer_id:
            self._active_layer_id = layer_id
            self.active_layer_changed.emit(layer_id)

    def add_paths_to_layer(
        self,
        layer_id: str,
        paths: list[Polyline],
        description: str = "Draw Shape",
    ) -> None:
        """Append polylines to a layer without replacing existing paths (undoable).

        Unlike ``set_layer_paths`` which replaces, this appends so that users
        can draw multiple shapes to build up a composition incrementally.
        Undo removes only the newly appended paths.
        """
        layer = self._project.get_layer(layer_id)
        if layer is not None and paths:
            from plottter.gui.commands import AppendPathsCommand
            cmd = AppendPathsCommand(self, layer_id, paths, description)
            self._undo_stack.push(cmd)

    def set_layer_generator_info(self, layer_id: str, info: dict | None) -> None:
        """Update a layer's generator_info (not undoable — UI metadata only).

        This stores the last-used generator name, mode, and params so that
        selecting the layer in the panel can restore the generator settings.
        Marks the project modified so the info is included in the next save.
        """
        layer = self._project.get_layer(layer_id)
        if layer is not None:
            layer.generator_info = info
            self._set_modified(True)
