"""_PluginOpsMixin — plugin management and processing plugin execution."""

from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox


class _PluginOpsMixin:
    """Mixin providing plugin management operations for MainWindow."""

    def _on_manage_plugins(self) -> None:
        """Show the plugin management dialog."""
        from plottter.generators.plugin_loader import (
            create_user_plugin_dir,
            get_plugin_dirs,
            load_plugins,
        )
        from plottter.generators import GENERATORS

        # Ensure user plugin directory exists, then reload plugins
        user_dir = create_user_plugin_dir()
        new_names = load_plugins()
        plugin_dirs = get_plugin_dirs()

        dir_list = "\n".join(f"  • {d}" for d in plugin_dirs)
        gen_list = "\n".join(
            f"  • {name}" for name in sorted(GENERATORS.keys())
        ) or "  (none)"

        if new_names:
            newly = "\n".join(f"  + {n}" for n in new_names)
            msg = (
                f"Newly loaded plugins:\n{newly}\n\n"
                f"All registered generators:\n{gen_list}\n\n"
                f"Plugin directories searched:\n{dir_list}"
            )
        else:
            msg = (
                f"No new plugins found.\n\n"
                f"All registered generators:\n{gen_list}\n\n"
                f"Plugin directories:\n{dir_list}\n\n"
                f"Place .py files in the plugin directory to add custom generators.\n"
                f"User plugin directory: {user_dir}"
            )

        QMessageBox.information(self, "Plugin Manager", msg)
        # Refresh processing plugins menu in case new plugins were loaded
        self._rebuild_processing_plugins_menu()

    def _on_run_processing_plugin(self, plugin_cls: type) -> None:
        """Show a parameter dialog for *plugin_cls* and run it on the active layer."""
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QLabel,
            QLineEdit,
            QSpinBox,
            QVBoxLayout,
        )

        layer_id = self._controller.active_layer_id
        layer = self._controller.get_layer(layer_id) if layer_id else None
        if layer is None:
            QMessageBox.warning(
                self,
                plugin_cls.name,
                "No active layer selected.",
            )
            return
        if not layer.paths:
            QMessageBox.information(
                self,
                plugin_cls.name,
                "The active layer has no paths to process.",
            )
            return

        plugin = plugin_cls()
        parameters = plugin.get_parameters()

        # Build a simple parameter dialog if there are any parameters
        param_widgets: dict[str, object] = {}
        if parameters:
            from plottter.generators.base import (
                BoolParam,
                ChoiceParam,
                FloatParam,
                IntParam,
                StringParam,
            )

            dlg = QDialog(self)
            dlg.setWindowTitle(plugin_cls.name)
            dlg.setMinimumWidth(320)
            layout = QVBoxLayout(dlg)

            if plugin_cls.description:
                desc_label = QLabel(plugin_cls.description)
                desc_label.setWordWrap(True)
                layout.addWidget(desc_label)

            form = QFormLayout()
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            layout.addLayout(form)

            for param in parameters:
                if isinstance(param, FloatParam):
                    spin = QDoubleSpinBox()
                    spin.setRange(param.min, param.max)
                    spin.setSingleStep(param.step)
                    spin.setValue(param.default)
                    if param.description:
                        spin.setToolTip(param.description)
                    form.addRow(param.label + ":", spin)
                    param_widgets[param.name] = spin
                elif isinstance(param, IntParam):
                    spin = QSpinBox()
                    spin.setRange(param.min, param.max)
                    spin.setSingleStep(param.step)
                    spin.setValue(param.default)
                    if param.description:
                        spin.setToolTip(param.description)
                    form.addRow(param.label + ":", spin)
                    param_widgets[param.name] = spin
                elif isinstance(param, BoolParam):
                    cb = QCheckBox()
                    cb.setChecked(param.default)
                    if param.description:
                        cb.setToolTip(param.description)
                    form.addRow(param.label + ":", cb)
                    param_widgets[param.name] = cb
                elif isinstance(param, ChoiceParam):
                    combo = QComboBox()
                    combo.addItems(param.choices)
                    if param.default in param.choices:
                        combo.setCurrentText(param.default)
                    if param.description:
                        combo.setToolTip(param.description)
                    form.addRow(param.label + ":", combo)
                    param_widgets[param.name] = combo
                elif isinstance(param, StringParam):
                    edit = QLineEdit(param.default)
                    if param.description:
                        edit.setToolTip(param.description)
                    form.addRow(param.label + ":", edit)
                    param_widgets[param.name] = edit

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            layout.addWidget(buttons)

            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

        # Collect parameter values
        params: dict = {}
        from plottter.generators.base import (
            BoolParam,
            ChoiceParam,
            FloatParam,
            IntParam,
            StringParam,
        )
        for param in parameters:
            widget = param_widgets.get(param.name)
            if widget is None:
                params[param.name] = getattr(param, "default", None)
            elif isinstance(param, (FloatParam, IntParam)):
                params[param.name] = widget.value()  # type: ignore[attr-defined]
            elif isinstance(param, BoolParam):
                params[param.name] = widget.isChecked()  # type: ignore[attr-defined]
            elif isinstance(param, ChoiceParam):
                params[param.name] = widget.currentText()  # type: ignore[attr-defined]
            elif isinstance(param, StringParam):
                params[param.name] = widget.text()  # type: ignore[attr-defined]
            else:
                params[param.name] = getattr(param, "default", None)

        # Run the plugin
        try:
            new_paths = plugin.process(list(layer.paths), params)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                plugin_cls.name,
                f"Plugin error:\n{exc}",
            )
            return

        # Wrap in an undo command
        self._controller.set_layer_paths(layer.id, new_paths, plugin_cls.name)
        self.statusBar().showMessage(
            f"Processing plugin '{plugin_cls.name}' applied: "
            f"{len(layer.paths)} → {len(new_paths)} paths.",
            4000,
        )
