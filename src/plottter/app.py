"""Application entry point — creates QApplication and launches MainWindow."""

from __future__ import annotations

import sys


def main() -> None:
    from PyQt6.QtWidgets import QApplication
    from plottter.gui.dialogs.new_project import load_default_canvas
    from plottter.gui.main_window import MainWindow
    from plottter.gui.project_controller import ProjectController
    from plottter.models import Layer, Project

    app = QApplication(sys.argv)
    app.setApplicationName("Plottter")
    app.setApplicationVersion("0.1.0")

    # Default project: user's saved default canvas (falls back to A4) + one empty "Layer 1"
    canvas = load_default_canvas()
    project = Project(name="Untitled", canvas=canvas)
    project.add_layer(Layer(name="Layer 1", color="#000000"))

    controller = ProjectController(project)
    window = MainWindow(controller)
    window.show()

    sys.exit(app.exec())
