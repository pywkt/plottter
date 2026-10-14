"""ColorSepParamsDialog — tweak a line-art generator's parameters for the
Color Separation "Generate Lines" step without round-tripping through
Image to Lines mode and a one-off user preset.

The dialog is built from the generator's ``get_parameters()`` via
:class:`ParamForm`, prefilled with the selected preset's values merged over
the generator defaults (or the current override when one is active).

Buttons:

* **Reset to Preset** — restore the prefilled preset values.
* **Save as Preset…** — persist the current values as a user preset for this
  generator (``~/.plottter/presets/``) and accept the dialog; the panel then
  selects that preset so no override is left dangling.
* **OK** — accept; the panel stores the values as a session override.
* **Cancel** — discard.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from plottter.generators.base import Generator, Preset
from plottter.gui.widgets.param_form import ParamForm


def default_params_for(generator: Generator) -> dict[str, Any]:
    """Return ``{name: default}`` for every parameter that declares a default."""
    out: dict[str, Any] = {}
    for p in generator.get_parameters():
        if hasattr(p, "default"):
            out[p.name] = p.default
    return out


class ColorSepParamsDialog(QDialog):
    """Modal editor for the line-generation parameters of one generator."""

    def __init__(
        self,
        generator: Generator,
        preset_name: str,
        base_params: dict[str, Any],
        initial_params: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._generator = generator
        self._preset_name = preset_name
        self._base_params = dict(base_params)
        #: Set when the user saved the values as a user preset; the panel
        #: selects this preset instead of keeping an override.
        self.saved_preset_name: str | None = None

        self.setWindowTitle(f"{generator.name} — Line Parameters")
        self.setMinimumWidth(460)
        self.setMinimumHeight(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        header = QLabel(f"Preset: <b>{preset_name}</b>")
        header.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(header)

        self._form = ParamForm(generator.get_parameters())
        self._form.set_values(self._base_params)
        if initial_params:
            self._form.set_values(initial_params)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._form)
        layout.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._reset_btn = QPushButton("Reset to Preset")
        self._save_btn = QPushButton("Save as Preset…")
        buttons.addButton(self._reset_btn, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.addButton(self._save_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._reset_btn.clicked.connect(self._on_reset)
        self._save_btn.clicked.connect(self._on_save_as_preset)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------

    def params(self) -> dict[str, Any]:
        """The current form values as a plain dict."""
        return self._form.values()

    def _on_reset(self) -> None:
        self._form.set_values(self._base_params)

    def _on_save_as_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Save as Preset",
            "Preset name:",
            text="" if self._preset_name == "Default" else f"{self._preset_name} (tweaked)",
        )
        name = name.strip()
        if not ok or not name:
            return
        try:
            from plottter.presets.user_presets import save_user_preset

            save_user_preset(self._generator.name, Preset(name=name, params=self.params()))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save Preset Failed", str(exc))
            return
        self.saved_preset_name = name
        self.accept()
