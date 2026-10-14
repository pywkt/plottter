"""ParamForm — a self-contained form built from a generator's Parameter list.

The settings panel builds its parameter widgets inline (see
``_snapshot.set_generator``) because they are entangled with panel-only
features (FMM pick button, dynamic params, per-layer snapshots).  Dialogs
that need the same widgets without that baggage use this widget instead:
build it from ``generator.get_parameters()``, call :meth:`set_values` to
prefill, and :meth:`values` to read back a plain dict.

Supported parameter types mirror the panel: Float, Int, Choice, Bool,
String (single/multi-line), Expression, Font, Image, File.  ``visible_when``
conditions are honoured — rows show/hide live as their controlling Choice or
Bool widgets change.
"""

from __future__ import annotations

import functools
from typing import Any

from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from plottter.generators.base import (
    BoolParam,
    ChoiceParam,
    ExpressionParam,
    FileParam,
    FloatParam,
    FontParam,
    ImageParam,
    IntParam,
    Parameter,
    StringParam,
)
from plottter.gui.widgets.font_picker import FontPicker


class ParamForm(QWidget):
    """A QFormLayout of widgets generated from a list of :class:`Parameter`."""

    def __init__(
        self,
        parameters: list[Parameter],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._parameters: list[Parameter] = list(parameters)
        #: name -> the widget whose value is read back (QLineEdit for file rows)
        self._widgets: dict[str, QWidget] = {}
        #: name -> the label in the left column
        self._labels: dict[str, QLabel] = {}
        #: name -> the widget occupying the right column (row container for
        #: file/image params, otherwise the same object as ``_widgets[name]``)
        self._fields: dict[str, QWidget] = {}

        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        for param in self._parameters:
            self._add_row(param)

        self._update_visibility()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parameter_names(self) -> list[str]:
        return [p.name for p in self._parameters]

    def widget(self, name: str) -> QWidget | None:
        """Return the value-bearing widget for *name* (mainly for tests)."""
        return self._widgets.get(name)

    def values(self) -> dict[str, Any]:
        """Read every widget back into a ``{param_name: value}`` dict."""
        result: dict[str, Any] = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                result[name] = widget.value()
            elif isinstance(widget, QPlainTextEdit):
                result[name] = widget.toPlainText()
            elif isinstance(widget, QLineEdit):
                result[name] = widget.text()
            elif isinstance(widget, QComboBox):
                result[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                result[name] = widget.isChecked()
            elif isinstance(widget, FontPicker):
                result[name] = widget.font_path()
        return result

    def set_values(self, values: dict[str, Any]) -> None:
        """Push *values* into the widgets. Unknown names are ignored."""
        for name, value in values.items():
            widget = self._widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QPlainTextEdit):
                widget.setPlainText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, FontPicker):
                widget.set_font_path(str(value))
        self._update_visibility()

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------

    def _add_row(self, param: Parameter) -> None:
        label = QLabel(param.label)
        if param.description:
            label.setToolTip(param.description)

        field: QWidget
        widget: QWidget

        if isinstance(param, FloatParam):
            spin = QDoubleSpinBox()
            spin.setMinimum(param.min)
            spin.setMaximum(param.max)
            spin.setSingleStep(param.step)
            spin.setDecimals(4)
            spin.setValue(param.default)
            widget = field = spin
        elif isinstance(param, IntParam):
            ispin = QSpinBox()
            ispin.setMinimum(param.min)
            ispin.setMaximum(param.max)
            ispin.setSingleStep(param.step)
            ispin.setValue(param.default)
            widget = field = ispin
        elif isinstance(param, ChoiceParam):
            combo = QComboBox()
            combo.addItems(param.choices)
            if param.default in param.choices:
                combo.setCurrentIndex(param.choices.index(param.default))
            combo.currentTextChanged.connect(self._update_visibility)
            if param.choice_descriptions:
                descs = param.choice_descriptions

                def _update_tip(text: str, c: QComboBox = combo) -> None:
                    c.setToolTip(descs.get(text, param.description))

                combo.currentTextChanged.connect(_update_tip)
                _update_tip(combo.currentText())
            widget = field = combo
        elif isinstance(param, BoolParam):
            check = QCheckBox()
            check.setChecked(param.default)
            check.stateChanged.connect(self._update_visibility)
            widget = field = check
        elif isinstance(param, StringParam) and param.multiline:
            text = QPlainTextEdit(str(param.default))
            text.setMinimumHeight(160)
            text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
            widget = field = text
        elif isinstance(param, (StringParam, ExpressionParam)):
            widget = field = QLineEdit(str(param.default))
        elif isinstance(param, FontParam):
            picker = FontPicker()
            if param.default:
                picker.set_font_path(param.default)
            widget = field = picker
        elif isinstance(param, (ImageParam, FileParam)):
            if isinstance(param, FileParam):
                file_filter = param.filter
                title = "Select File"
            else:
                file_filter = (
                    "Images (*.jpg *.jpeg *.png *.webp *.gif *.bmp *.tiff);;All Files (*)"
                )
                title = "Select Image"
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            line_edit = QLineEdit(str(param.default) if param.default else "")
            browse = QPushButton("Browse…")
            browse.setFixedWidth(70)
            browse.clicked.connect(
                functools.partial(self._browse, line_edit, title, file_filter)
            )
            row.addWidget(line_edit)
            row.addWidget(browse)
            widget = line_edit
            field = container
        else:
            widget = field = QLineEdit(str(getattr(param, "default", "")))

        if param.description and not (
            isinstance(param, ChoiceParam) and param.choice_descriptions
        ):
            field.setToolTip(param.description)

        self._widgets[param.name] = widget
        self._labels[param.name] = label
        self._fields[param.name] = field
        self._layout.addRow(label, field)

    def _browse(self, line_edit: QLineEdit, title: str, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        if path:
            line_edit.setText(path)

    # ------------------------------------------------------------------
    # visible_when
    # ------------------------------------------------------------------

    def _update_visibility(self, *_args: Any) -> None:
        for param in self._parameters:
            if param.visible_when is None:
                continue
            visible = True
            for dep_name, allowed in param.visible_when.items():
                dep = self._widgets.get(dep_name)
                if dep is None:
                    continue
                if isinstance(dep, QComboBox):
                    current: Any = dep.currentText()
                elif isinstance(dep, QCheckBox):
                    current = dep.isChecked()
                else:
                    continue
                if current not in allowed:
                    visible = False
                    break
            self._labels[param.name].setVisible(visible)
            self._fields[param.name].setVisible(visible)
