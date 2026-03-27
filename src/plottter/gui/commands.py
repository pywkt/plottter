"""QUndoCommand subclasses for the Plottter undo/redo system.

Each command captures the state needed to undo/redo a single user action.
Commands call back into _raw_* methods on the ProjectController — these
methods mutate the model and emit signals without pushing a new command,
preventing infinite undo-push loops.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from PyQt6.QtGui import QUndoCommand

if TYPE_CHECKING:
    from plottter.gui.project_controller import ProjectController
    from plottter.models import Canvas, Layer
    from plottter.models.path import Polyline


class AddLayerCommand(QUndoCommand):
    """Add a layer to the project (optionally at a specific index)."""

    def __init__(
        self,
        controller: ProjectController,
        layer: Layer,
        insert_index: int | None = None,
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__(f"Add Layer '{layer.name}'", parent)
        self._controller = controller
        self._layer = layer
        self._insert_index = insert_index  # None = append

    def redo(self) -> None:
        if self._insert_index is not None:
            self._controller._raw_insert_layer(self._layer, self._insert_index)
        else:
            self._controller._raw_add_layer(self._layer)

    def undo(self) -> None:
        self._controller._raw_remove_layer(self._layer.id)


class RemoveLayerCommand(QUndoCommand):
    """Remove a layer from the project, restoring it on undo."""

    def __init__(
        self,
        controller: ProjectController,
        layer_id: str,
        parent: QUndoCommand | None = None,
    ) -> None:
        project = controller.current_project
        layer = project.get_layer(layer_id)
        name = layer.name if layer else layer_id
        super().__init__(f"Remove Layer '{name}'", parent)
        self._controller = controller
        self._layer_id = layer_id
        # Snapshot layer content and position before removal
        self._layer_snapshot = copy.deepcopy(layer) if layer else None
        self._original_index: int | None = next(
            (i for i, l in enumerate(project.layers) if l.id == layer_id),
            None,
        )

    def redo(self) -> None:
        self._controller._raw_remove_layer(self._layer_id)

    def undo(self) -> None:
        if self._layer_snapshot is not None and self._original_index is not None:
            self._controller._raw_insert_layer(self._layer_snapshot, self._original_index)


class ReorderLayerCommand(QUndoCommand):
    """Move a layer to a new position in the stack."""

    def __init__(
        self,
        controller: ProjectController,
        layer_id: str,
        new_index: int,
        old_index: int,
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__("Reorder Layers", parent)
        self._controller = controller
        self._layer_id = layer_id
        self._new_index = new_index
        self._old_index = old_index

    def redo(self) -> None:
        self._controller._raw_reorder_layer(self._layer_id, self._new_index)

    def undo(self) -> None:
        self._controller._raw_reorder_layer(self._layer_id, self._old_index)


class SetLayerPathsCommand(QUndoCommand):
    """Replace a layer's paths.

    Used for generate, clear, optimize, simplify, merge, clip, etc.
    The description parameter appears in the Edit > Undo menu text.
    """

    def __init__(
        self,
        controller: ProjectController,
        layer_id: str,
        new_paths: list[Polyline],
        old_paths: list[Polyline],
        description: str = "Set Paths",
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__(description, parent)
        self._controller = controller
        self._layer_id = layer_id
        self._new_paths = new_paths
        self._old_paths = old_paths

    def redo(self) -> None:
        self._controller._raw_set_layer_paths(self._layer_id, self._new_paths)

    def undo(self) -> None:
        self._controller._raw_set_layer_paths(self._layer_id, self._old_paths)


class SetLayerPropertyCommand(QUndoCommand):
    """Change a single named property of a layer.

    Supports: name, color, visible, locked, opacity.
    """

    def __init__(
        self,
        controller: ProjectController,
        layer_id: str,
        prop_name: str,
        new_value: object,
        old_value: object,
        description: str | None = None,
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__(description or f"Change Layer {prop_name.title()}", parent)
        self._controller = controller
        self._layer_id = layer_id
        self._prop_name = prop_name
        self._new_value = new_value
        self._old_value = old_value

    def redo(self) -> None:
        self._controller._raw_set_layer_property(
            self._layer_id, self._prop_name, self._new_value
        )

    def undo(self) -> None:
        self._controller._raw_set_layer_property(
            self._layer_id, self._prop_name, self._old_value
        )


class SetCanvasCommand(QUndoCommand):
    """Replace the project canvas (size, margins, paper preset)."""

    def __init__(
        self,
        controller: ProjectController,
        new_canvas: Canvas,
        old_canvas: Canvas,
        description: str = "Canvas Settings",
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__(description, parent)
        self._controller = controller
        self._new_canvas = new_canvas
        self._old_canvas = old_canvas

    def redo(self) -> None:
        self._controller._raw_set_canvas(self._new_canvas)

    def undo(self) -> None:
        self._controller._raw_set_canvas(self._old_canvas)


class MaskPaintCommand(QUndoCommand):
    """Record a before/after mask state for undo/redo of mask paint operations.

    The command holds numpy array snapshots of the mask taken before and after
    the operation and applies them back to the canvas widget on undo/redo.
    """

    def __init__(
        self,
        canvas,  # CanvasWidget — loosely typed to avoid circular import
        before: object,  # np.ndarray | None
        after: object,   # np.ndarray | None
        description: str = "Paint Mask",
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__(description, parent)
        self._canvas = canvas
        self._before = before
        self._after = after

    def redo(self) -> None:
        self._canvas.set_mask(self._after.copy() if self._after is not None else None)

    def undo(self) -> None:
        self._canvas.set_mask(self._before.copy() if self._before is not None else None)


class AppendPathsCommand(QUndoCommand):
    """Append polylines to a layer's existing paths.

    redo: appends the new paths; undo: truncates back to the original count.
    This is used by Shape Drawing to build up compositions incrementally.
    """

    def __init__(
        self,
        controller: ProjectController,
        layer_id: str,
        appended_paths: list[Polyline],
        description: str = "Draw Shape",
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__(description, parent)
        self._controller = controller
        self._layer_id = layer_id
        self._appended_paths = appended_paths
        # Record the existing path count so undo can truncate safely
        layer = controller.current_project.get_layer(layer_id)
        self._original_count: int = len(layer.paths) if layer is not None else 0

    def redo(self) -> None:
        layer = self._controller.current_project.get_layer(self._layer_id)
        if layer is not None:
            layer.paths = layer.paths + self._appended_paths
            self._controller._set_modified(True)
            self._controller.paths_changed.emit(self._layer_id)

    def undo(self) -> None:
        layer = self._controller.current_project.get_layer(self._layer_id)
        if layer is not None:
            layer.paths = layer.paths[: self._original_count]
            self._controller._set_modified(True)
            self._controller.paths_changed.emit(self._layer_id)


class MergeLayersCommand(QUndoCommand):
    """Merge multiple source layers into one new layer.

    redo: removes source layers, adds merged layer.
    undo: removes merged layer, re-inserts source layers at original positions.
    """

    def __init__(
        self,
        controller: ProjectController,
        source_layer_ids: list[str],
        merged_layer: Layer,
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__("Merge Layers", parent)
        self._controller = controller
        self._source_ids = list(source_layer_ids)
        self._merged_layer = merged_layer
        # Snapshot source layers with their original positions (before merge)
        project = controller.current_project
        self._source_snapshots: list[tuple[int, Layer]] = []
        for layer_id in source_layer_ids:
            idx = next(
                (i for i, l in enumerate(project.layers) if l.id == layer_id),
                None,
            )
            layer = project.get_layer(layer_id)
            if idx is not None and layer is not None:
                self._source_snapshots.append((idx, copy.deepcopy(layer)))

    def redo(self) -> None:
        for layer_id in self._source_ids:
            self._controller._raw_remove_layer(layer_id)
        self._controller._raw_add_layer(self._merged_layer)

    def undo(self) -> None:
        self._controller._raw_remove_layer(self._merged_layer.id)
        # Re-insert sources in ascending index order so positions line up
        for idx, layer in sorted(self._source_snapshots, key=lambda x: x[0]):
            self._controller._raw_insert_layer(layer, idx)


class SaveMaskCommand(QUndoCommand):
    """Save (or overwrite) a named mask.

    redo: encodes and stores the mask array.
    undo: restores the previous PNG bytes (or deletes the mask if it was new).
    """

    def __init__(
        self,
        controller: ProjectController,
        name: str,
        mask_array: object,  # np.ndarray — loosely typed to avoid import at module level
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__(f"Save Mask '{name}'", parent)
        self._controller = controller
        self._name = name
        # Deep-copy the array so that mutations by the caller after save_mask()
        # don't affect subsequent redo operations.
        self._mask_array = copy.deepcopy(mask_array)
        # Capture old state so undo can restore it
        self._old_png: bytes | None = controller.current_project.masks.get(name)

    def redo(self) -> None:
        self._controller._raw_save_mask(self._name, self._mask_array)

    def undo(self) -> None:
        if self._old_png is None:
            self._controller._raw_delete_mask(self._name)
        else:
            self._controller._raw_restore_mask_bytes(self._name, self._old_png)


class DeleteMaskCommand(QUndoCommand):
    """Delete a named mask.

    redo: removes the mask.
    undo: restores the mask from the captured PNG bytes.
    """

    def __init__(
        self,
        controller: ProjectController,
        name: str,
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__(f"Delete Mask '{name}'", parent)
        self._controller = controller
        self._name = name
        # Snapshot the PNG bytes before deletion
        self._saved_png: bytes | None = controller.current_project.masks.get(name)

    def redo(self) -> None:
        self._controller._raw_delete_mask(self._name)

    def undo(self) -> None:
        if self._saved_png is not None:
            self._controller._raw_restore_mask_bytes(self._name, self._saved_png)


class RenameMaskCommand(QUndoCommand):
    """Rename a mask from *old* to *new*.

    redo: renames old → new.
    undo: renames new → old.
    """

    def __init__(
        self,
        controller: ProjectController,
        old: str,
        new: str,
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__(f"Rename Mask '{old}' → '{new}'", parent)
        self._controller = controller
        self._old = old
        self._new = new

    def redo(self) -> None:
        self._controller._raw_rename_mask(self._old, self._new)

    def undo(self) -> None:
        self._controller._raw_rename_mask(self._new, self._old)


class MoveLayerCommand(QUndoCommand):
    """Translate a layer's paths and optionally sync generator offset params.

    When the generator has ``x_offset_mm`` / ``y_offset_mm`` parameters,
    those values in ``generator_info`` are updated alongside the path
    translation so that re-generating the layer preserves the new position.
    Both changes are bundled into one undoable operation so that Ctrl+Z
    restores both the paths *and* the offset parameters atomically.

    Pass ``new_generator_info=None`` to skip the generator_info update
    (for generators that don't have offset parameters, e.g. image-based ones).
    """

    def __init__(
        self,
        controller: ProjectController,
        layer_id: str,
        new_paths: list[Polyline],
        old_paths: list[Polyline],
        new_generator_info: dict | None = None,
        old_generator_info: dict | None = None,
        parent: QUndoCommand | None = None,
    ) -> None:
        super().__init__("Move Layer", parent)
        self._controller = controller
        self._layer_id = layer_id
        self._new_paths = new_paths
        self._old_paths = old_paths
        # None means "no generator_info update needed" (generator lacks offset params)
        self._new_generator_info = new_generator_info
        self._old_generator_info = old_generator_info
        self._update_gen_info = new_generator_info is not None

    def redo(self) -> None:
        self._controller._raw_set_layer_paths(self._layer_id, self._new_paths)
        if self._update_gen_info:
            self._controller._raw_set_layer_generator_info(
                self._layer_id, self._new_generator_info
            )

    def undo(self) -> None:
        self._controller._raw_set_layer_paths(self._layer_id, self._old_paths)
        if self._update_gen_info:
            self._controller._raw_set_layer_generator_info(
                self._layer_id, self._old_generator_info
            )
